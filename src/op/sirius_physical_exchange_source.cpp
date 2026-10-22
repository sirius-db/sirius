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
#include "exchange/exchange_service.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>

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
  /// Unordered: batches in arrival order.
  std::deque<std::shared_ptr<cucascade::data_batch>> queue;
  /// Ordered: batches by (sender, batch number), released together.
  std::map<std::pair<exchange::rank_t, std::uint64_t>, std::shared_ptr<cucascade::data_batch>> held;
  std::set<exchange::rank_t> ended;
  std::exception_ptr error;
  std::weak_ptr<pipeline::sirius_pipeline> pipeline;

  state(std::set<exchange::rank_t> s, bool o) : senders(std::move(s)), ordered(o) {}

  /// Under mutex.
  [[nodiscard]] bool all_ended() const { return ended.size() == senders.size(); }

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

MemoryBarrierType sirius_physical_exchange_source::input_barrier_for(
  sirius_physical_operator const& /*producer*/) const
{
  return MemoryBarrierType::PIPELINE;
}

std::optional<task_creation_hint> sirius_physical_exchange_source::get_next_task_hint()
{
  bool finished = false;
  {
    std::lock_guard lock{_state->mutex};
    bool const ready =
      _state->error || !_state->queue.empty() || (_state->all_ended() && !_state->held.empty());
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
  return !_state->error && _state->all_ended() && _state->queue.empty() && _state->held.empty();
}

std::unique_ptr<operator_data> sirius_physical_exchange_source::get_next_task_input_data()
{
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  {
    std::lock_guard lock{_state->mutex};
    if (_state->error) { std::rethrow_exception(_state->error); }
    if (!_state->queue.empty()) {
      batches.push_back(std::move(_state->queue.front()));
      _state->queue.pop_front();
    } else if (_state->all_ended() && !_state->held.empty()) {
      for (auto& [key, batch] : _state->held) {
        batches.push_back(std::move(batch));
      }
      _state->held.clear();
    }
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
