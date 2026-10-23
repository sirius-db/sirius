/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op/sirius_physical_exchange_source.hpp"

#include "creator/task_creator.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "exchange/exchange_service.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "op/sirius_physical_exchange_sink.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_device.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace sirius::op {

struct sirius_physical_exchange_source::state {
  std::mutex mutex;
  std::set<exchange::rank_t> const senders;
  bool const ordered;
  /// Unordered: the repository of the port from this rank's sink, once wired. Batches queued
  /// there can be spilled under memory pressure; a deque outside any repository cannot.
  cucascade::shared_data_repository* repo = nullptr;
  /// The query's repositories are cleared at its end, while a peer may still be sending: an
  /// arrival adds to `repo` only under a work lease, which the clearing waits for.
  exec::query_lifecycle_registry* lifecycle = nullptr;
  sirius::query_id_t query_id{};
  /// Unordered: batches that arrived before the port was wired, in arrival order.
  std::deque<std::shared_ptr<cucascade::data_batch>> queue;
  /// Ordered: batches by (sender, batch number), released together.
  std::map<std::pair<exchange::rank_t, std::uint64_t>, std::shared_ptr<cucascade::data_batch>> held;
  std::set<exchange::rank_t> ended;
  std::exception_ptr error;
  std::weak_ptr<pipeline::sirius_pipeline> pipeline;
  /// Where the empty batch of a stream that brought nothing is made; none: no empty batch.
  cucascade::memory::memory_space* gpu = nullptr;
  /// Whether a task has taken any batch from this source.
  bool delivered = false;

  state(std::set<exchange::rank_t> s, bool o) : senders(std::move(s)), ordered(o) {}

  /// Under mutex.
  [[nodiscard]] bool all_ended() const { return ended.size() == senders.size(); }

  /// Under mutex.
  [[nodiscard]] bool unordered_empty() const
  {
    return queue.empty() && (repo == nullptr || repo->total_size() == 0);
  }

  /// Under mutex. The stream ended having brought nothing: one empty batch is still owed, as a
  /// local producer would hand on, since operators such as a hash join wait for a batch per side.
  /// Not on a rank a gather does not target (no senders): what it holds is not part of the result,
  /// and an aggregate there would make a row of nothing.
  [[nodiscard]] bool owes_empty_batch() const
  {
    return gpu != nullptr && !senders.empty() && !delivered && !error && all_ended() &&
           unordered_empty() && held.empty();
  }

  /// Nominates the pipeline's head for a task, as STREAMING_SOURCE does on data.
  void schedule()
  {
    auto p = pipeline.lock();
    if (!p) { return; }
    auto* creator = p->get_task_creator();
    auto head     = p->get_source();
    if (creator && head) { creator->schedule(head.get()); }
  }

  /// Re-checks whether the pipeline finished, now nothing more will arrive.
  void recheck()
  {
    if (auto p = pipeline.lock()) { p->update_pipeline_status(false); }
  }
};

struct sirius_physical_exchange_source::receiver : exchange::inbound {
  std::shared_ptr<state> s;

  explicit receiver(std::shared_ptr<state> st) : s(std::move(st)) {}

  void on_batch(exchange::rank_t from,
                std::uint64_t batch,
                std::shared_ptr<cucascade::data_batch> data) override
  {
    {
      std::lock_guard lock{s->mutex};
      if (!s->senders.contains(from) || s->ended.contains(from)) {
        s->error = std::make_exception_ptr(
          sirius::internal_exception("exchange source: an unexpected batch from rank {}", from));
      } else if (s->ordered) {
        s->held.emplace(std::make_pair(from, batch), std::move(data));
        return;  // released once every sender has ended
      } else if (s->repo != nullptr) {
        auto const lease = s->lifecycle->try_acquire_work(s->query_id);
        if (!lease) { return; }  // the query is ending, and nothing will take this batch
        s->repo->add_data_batch(std::move(data));
      } else {
        s->queue.push_back(std::move(data));
      }
    }
    s->schedule();
  }

  void on_end(exchange::rank_t from) override
  {
    bool done = false;
    {
      std::lock_guard lock{s->mutex};
      if (!s->senders.contains(from)) { return; }
      s->ended.insert(from);
      done = s->all_ended();
    }
    if (done) {
      s->schedule();
      s->recheck();
    }
  }

  void on_error(std::exception_ptr error) override
  {
    {
      std::lock_guard lock{s->mutex};
      if (!s->error) { s->error = std::move(error); }
    }
    s->schedule();
  }
};

sirius_physical_exchange_source::sirius_physical_exchange_source(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<exchange::exchange_service> service,
  exchange::stream_id stream,
  std::set<exchange::rank_t> senders,
  bool ordered)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::EXCHANGE_SOURCE, std::move(types), estimated_cardinality),
    _service(std::move(service)),
    _stream(stream),
    _state(std::make_shared<state>(std::move(senders), ordered))
{
  if (!_service) { throw sirius::invalid_input_exception("exchange source: no exchange service"); }
  for (auto const r : _state->senders) {
    if (r >= _service->world_size()) {
      throw sirius::invalid_input_exception(
        "exchange source: sender rank {} of {}", r, _service->world_size());
    }
  }
  _service->attach(_stream, std::make_shared<receiver>(_state));
}

sirius_physical_exchange_source::~sirius_physical_exchange_source() { _service->detach(_stream); }

bool sirius_physical_exchange_source::ordered() const noexcept { return _state->ordered; }

void sirius_physical_exchange_source::set_query_lifecycle(exec::query_lifecycle_registry& lifecycle)
{
  std::lock_guard lock{_state->mutex};
  _state->lifecycle = &lifecycle;
}

void sirius_physical_exchange_source::set_memory_space(cucascade::memory::memory_space& gpu)
{
  std::lock_guard lock{_state->mutex};
  _state->gpu = &gpu;
}

void sirius_physical_exchange_source::set_pipeline(
  std::shared_ptr<pipeline::sirius_pipeline> pipeline)
{
  {
    std::lock_guard lock{_state->mutex};
    _state->pipeline = pipeline;
  }
  sirius_physical_operator::set_pipeline(std::move(pipeline));
}

sirius::OrderPreservationType sirius_physical_exchange_source::source_order() const
{
  return _state->ordered ? sirius::OrderPreservationType::FIXED_ORDER
                         : sirius::OrderPreservationType::NO_ORDER;
}

bool sirius_physical_exchange_source::is_sink() const
{
  auto const* parent = get_parent_op();
  return sirius_physical_operator::is_sink() ||
         (parent != nullptr && (parent->type == SiriusPhysicalOperatorType::MERGE_AGGREGATE ||
                                parent->type == SiriusPhysicalOperatorType::MERGE_TOP_N));
}

MemoryBarrierType sirius_physical_exchange_source::input_barrier_for(
  sirius_physical_operator const& /*producer*/) const
{
  return MemoryBarrierType::PIPELINE;
}

void sirius_physical_exchange_source::bind_repository()
{
  // Ports are wired after the plan is built, and a peer may send before then; the task creator
  // only asks once they are.
  if (_state->repo != nullptr || _state->ordered || _state->lifecycle == nullptr) { return; }
  auto pipeline = _state->pipeline.lock();
  if (!pipeline) { return; }
  for (auto const& p : get_ports()) {
    if (p->repo == nullptr) { continue; }
    _state->query_id = pipeline->get_query_id();
    _state->repo     = p->repo;
    for (auto& batch : _state->queue) {
      _state->repo->add_data_batch(std::move(batch));
    }
    _state->queue.clear();
    return;
  }
}

std::optional<task_creation_hint> sirius_physical_exchange_source::get_next_task_hint()
{
  bool finished = false;
  {
    std::lock_guard lock{_state->mutex};
    bind_repository();
    bool const ready = _state->error || !_state->unordered_empty() ||
                       (_state->all_ended() && !_state->held.empty()) || _state->owes_empty_batch();
    if (ready) { return task_creation_hint{TaskCreationHint::READY, this}; }
    finished = _state->all_ended();
  }
  // Walk on to this rank's producer, so its scans are scheduled while this source waits.
  for (auto const& p : get_ports()) {
    if (p->src_pipeline && !p->src_pipeline->is_pipeline_finished()) {
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                                &p->src_pipeline->get_operators()[0].get()};
    }
  }
  if (finished) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, nullptr};
}

bool sirius_physical_exchange_source::all_ports_empty()
{
  std::lock_guard lock{_state->mutex};
  bind_repository();
  return !_state->error && _state->all_ended() && _state->unordered_empty() &&
         _state->held.empty() && !_state->owes_empty_batch();
}

std::unique_ptr<operator_data> sirius_physical_exchange_source::get_next_task_input_data()
{
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  cucascade::memory::memory_space* empty_on = nullptr;
  {
    std::lock_guard lock{_state->mutex};
    if (_state->error) { std::rethrow_exception(_state->error); }
    bind_repository();
    if (!_state->queue.empty()) {
      batches.push_back(std::move(_state->queue.front()));
      _state->queue.pop_front();
    } else if (auto batch = _state->repo ? _state->repo->pop_next_data_batch() : nullptr) {
      batches.push_back(std::move(batch));
    } else if (_state->all_ended() && !_state->held.empty()) {
      for (auto& [key, batch] : _state->held) {
        batches.push_back(std::move(batch));
      }
      _state->held.clear();
    } else if (_state->owes_empty_batch()) {
      empty_on = _state->gpu;
    }
    _state->delivered = _state->delivered || !batches.empty() || empty_on != nullptr;
  }
  if (empty_on != nullptr) {
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{empty_on->get_device_id()}};
    // Shaped like what this rank's own sink on the stream was given, which can differ from the
    // declared types; the types only when that sink saw nothing.
    auto const* sink =
      children.empty() ? nullptr : dynamic_cast<sirius_physical_exchange_sink*>(children[0].get());
    auto empty = sink != nullptr ? sink->empty_like_input() : nullptr;
    if (!empty) {
      empty = has_physical_overrides() ? make_empty_table(get_physical_types())
                                       : make_empty_table(get_types());
    }
    batches.push_back(
      make_data_batch(std::move(empty), *empty_on, cudf::get_default_stream(), batch_telemetry()));
  }
  if (batches.empty()) { return nullptr; }
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::unique_ptr<operator_data> sirius_physical_exchange_source::execute(
  const operator_data& input, ::cuda::stream_ref /*stream*/)
{
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input).get_data_batches());
}

std::size_t sirius_physical_exchange_source::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  return stats.bytes;
}

}  // namespace sirius::op
