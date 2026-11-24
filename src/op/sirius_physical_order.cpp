/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/sirius_physical_order.hpp"

#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/order/gpu_order_impl.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"  // IWYU pragma: keep
#include "sirius/exception.hpp"
#include "sirius_config.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius {
namespace op {

namespace {

//! Whether the two tables can be concatenated: same column count and the same cuDF type per
//! column. Batches of one port normally agree; a column whose carrier varies between batches
//! (see `sirius_physical_operator::set_alternate_physical_types`) does not, and such batches are
//! sorted as separate runs.
bool same_schema(cudf::table_view const& a, cudf::table_view const& b)
{
  if (a.num_columns() != b.num_columns()) { return false; }
  for (cudf::size_type i = 0; i < a.num_columns(); ++i) {
    if (a.column(i).type() != b.column(i).type()) { return false; }
  }
  return true;
}

//! Character bytes of every string column of @p table (empty when it has none).
std::vector<std::size_t> string_column_bytes(cudf::table_view const& table,
                                             ::cuda::stream_ref stream)
{
  std::vector<std::size_t> bytes;
  for (auto const& column : table) {
    if (column.type().id() != cudf::type_id::STRING) { continue; }
    bytes.push_back(column.is_empty()
                      ? 0
                      : static_cast<std::size_t>(
                          cudf::strings_column_view(column).chars_size(stream)));
  }
  return bytes;
}

struct process_counter_storage {
  std::atomic<uint64_t> tasks{0};
  std::atomic<uint64_t> input_batches{0};
  std::atomic<uint64_t> runs{0};
  std::atomic<uint64_t> concatenated_bytes{0};
};

process_counter_storage& process_storage()
{
  static process_counter_storage storage;
  return storage;
}

}  // namespace

std::optional<sort_run_limits> make_sort_run_limits(std::size_t gpu_bytes,
                                                    std::size_t num_gpus,
                                                    std::size_t executor_threads,
                                                    double fraction,
                                                    std::size_t concurrent_runs,
                                                    bool backpressure)
{
  auto const total = static_cast<std::size_t>(static_cast<double>(gpu_bytes) * fraction);
  auto const runs  = concurrent_runs != 0 ? concurrent_runs : std::max<std::size_t>(1, executor_threads);
  sort_run_limits limits;
  limits.max_run_bytes = total / runs;
  if (limits.max_run_bytes == 0) { return std::nullopt; }
  if (concurrent_runs != 0) { limits.max_concurrent_bytes = total; }
  if (backpressure) {
    limits.max_backlog_bytes =
      memory::saturating_mul(memory::saturating_mul(total, 2), std::max<std::size_t>(1, num_gpus));
  }
  return limits;
}

sirius_physical_order::sirius_physical_order(duckdb::vector<sirius::logical_type> types,
                                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                                             duckdb::vector<std::size_t> projections_p,
                                             std::size_t estimated_cardinality,
                                             bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

std::unique_ptr<operator_data> sirius_physical_order::execute(const operator_data& input_data,
                                                              ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_order::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Order by only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
    order_key_idx.push_back(idx);
    column_order.push_back(to_cudf_order(ord.type));
    null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }

  std::vector<int> proj_idx(projections.begin(), projections.end());
  const bool sort_key_encoding =
    _pipeline != nullptr ? _pipeline->get_operator_params().sort_key_encoding
                         : sirius::operator_params{}.sort_key_encoding;

  std::size_t sorted_batches     = 0;
  std::size_t runs               = 0;
  std::size_t concatenated_bytes = 0;
  auto const record              = [&] {
    _tasks.fetch_add(1, std::memory_order_relaxed);
    _input_batches.fetch_add(sorted_batches, std::memory_order_relaxed);
    _runs.fetch_add(runs, std::memory_order_relaxed);
    _concatenated_bytes.fetch_add(concatenated_bytes, std::memory_order_relaxed);
    auto& process = process_storage();
    process.tasks.fetch_add(1, std::memory_order_relaxed);
    process.input_batches.fetch_add(sorted_batches, std::memory_order_relaxed);
    process.runs.fetch_add(runs, std::memory_order_relaxed);
    process.concatenated_bytes.fetch_add(concatenated_bytes, std::memory_order_relaxed);
  };

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  auto const sort_one = [&](cucascade::read_only_data_batch const& batch,
                            cucascade::memory::memory_space& space) {
    auto sorted_batch = gpu_order_impl::local_order_by(batch,
                                                       order_key_idx,
                                                       column_order,
                                                       null_precedence,
                                                       proj_idx,
                                                       stream,
                                                       space,
                                                       batch_telemetry(),
                                                       sort_key_encoding);
    if (sorted_batch) { output_batches.push_back(std::move(sorted_batch)); }
  };

  if (!_coalesce) {
    for (auto const& batch : input_batches) {
      auto* space = batch.get_memory_space();
      if (!space) { continue; }
      sort_one(batch, *space);
      ++sorted_batches;
      ++runs;
    }
    record();
    return std::make_unique<pipelineable_operator_data>(output_batches);
  }

  // Coalescing: split the task's batches into sort units and sort each unit as one run. Group
  // planning already keeps a task within the limits using batch metadata; the split here uses the
  // real GPU tables, so the cuDF limits hold whatever the metadata said (a spilled batch may
  // carry no row count) and whoever built the input.
  std::vector<cucascade::read_only_data_batch> unit;
  cucascade::memory::memory_space* unit_space = nullptr;
  std::size_t unit_rows                       = 0;
  std::vector<std::size_t> unit_string_bytes;

  auto const flush = [&] {
    if (unit.empty()) { return; }
    if (unit.size() == 1) {
      sort_one(unit.front(), *unit_space);
    } else {
      // The concatenation lives only until its sorted copy exists.
      auto concatenated   = gpu_merge_impl::concat(unit, stream, *unit_space, batch_telemetry());
      auto const read_only = concatenated->to_read_only();
      concatenated_bytes  += read_only.get_data()->get_size_in_bytes();
      sort_one(read_only, *unit_space);
    }
    sorted_batches += unit.size();
    ++runs;
    unit.clear();
    unit_space = nullptr;
    unit_rows  = 0;
    unit_string_bytes.clear();
  };

  for (auto const& batch : input_batches) {
    auto* space = batch.get_memory_space();
    if (!space) { continue; }
    auto const view         = get_cudf_table_view(batch);
    auto const rows         = static_cast<std::size_t>(view.num_rows());
    auto const string_bytes = string_column_bytes(view, stream);
    bool fits               = !unit.empty() && space == unit_space &&
                same_schema(get_cudf_table_view(unit.front()), view) &&
                rows <= _limits.max_run_rows - std::min(unit_rows, _limits.max_run_rows);
    for (std::size_t i = 0; fits && i < string_bytes.size(); ++i) {
      fits = string_bytes[i] <= _limits.max_string_column_bytes -
                                  std::min(unit_string_bytes[i], _limits.max_string_column_bytes);
    }
    if (!fits) { flush(); }
    if (unit.empty()) {
      unit_space        = space;
      unit_string_bytes = string_bytes;
    } else {
      for (std::size_t i = 0; i < string_bytes.size(); ++i) {
        unit_string_bytes[i] += string_bytes[i];
      }
    }
    unit_rows += rows;
    unit.push_back(batch);
  }
  flush();
  record();

  SIRIUS_LOG_DEBUG("ORDER_BY: sorted {} input batches as {} runs", sorted_batches, runs);
  if (auto const& claimed = input.get_data_batches(); !claimed.empty() && claimed.front()) {
    finish_claim(claimed.front()->get_batch_id());
  }
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

void sirius_physical_order::enable_run_coalescing(sort_run_limits limits,
                                                  const std::vector<cudf::data_type>& physical_types)
{
  if (limits.max_run_bytes == 0) {
    throw std::invalid_argument("sirius_physical_order: max_run_bytes must be positive");
  }
  _limits            = limits;
  _coalesce          = true;
  _has_string_column = std::any_of(
    types.begin(), types.end(), [](sirius::logical_type const& type) { return type.is_varchar(); });
  _non_character_row_bytes = non_character_row_bytes(types, physical_types);
}

std::size_t sirius_physical_order::non_character_row_bytes(
  const duckdb::vector<sirius::logical_type>& types,
  const std::vector<cudf::data_type>& physical_types)
{
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (types[i].is_varchar()) {
      bytes += sizeof(int32_t);
      continue;
    }
    if (!types[i].is_fixed_width()) { continue; }
    auto width = types[i].fixed_width_byte_size();
    if (i < physical_types.size() && cudf::is_fixed_width(physical_types[i])) {
      width = std::min(width, cudf::size_of(physical_types[i]));
    }
    bytes += width;
  }
  return bytes;
}

port_residency_policy sirius_physical_order::input_residency_policy(std::string_view port_id) const
{
  auto policy = sirius_physical_operator::input_residency_policy(port_id);
  if (_coalesce && _limits.max_backlog_bytes != 0) { policy.downgrade_priority = -50; }
  return policy;
}

MemoryBarrierType sirius_physical_order::input_barrier_for(
  sirius_physical_operator const& producer) const
{
  if (_coalesce && producer.type == SiriusPhysicalOperatorType::PASSTHROUGH_SINK) {
    return MemoryBarrierType::PARTIAL;
  }
  return sirius_physical_operator::input_barrier_for(producer);
}

int sirius_physical_order::run_device(const std::vector<parked_batch>& parked,
                                      const std::vector<std::size_t>& members)
{
  std::unordered_map<int, std::size_t> bytes;
  for (auto const index : members) {
    if (parked[index].device >= 0) { bytes[parked[index].device] += parked[index].bytes; }
  }
  int device       = -1;
  std::size_t most = 0;
  for (auto const& [candidate, held] : bytes) {
    if (device < 0 || held > most || (held == most && candidate < device)) {
      device = candidate;
      most   = held;
    }
  }
  return device;
}

std::vector<std::size_t> sirius_physical_order::plan_next_run(
  const std::vector<parked_batch>& parked,
  const sort_run_limits& limits,
  bool has_string_column,
  bool producer_finished,
  const sorting_bytes_by_device& sorting)
{
  auto const unlimited = std::numeric_limits<std::size_t>::max();
  auto const max_bytes = limits.max_run_bytes == 0 ? unlimited : limits.max_run_bytes;
  auto const max_rows  = limits.max_run_rows == 0 ? unlimited : limits.max_run_rows;
  auto const max_string_bytes =
    has_string_column && limits.max_string_column_bytes != 0 ? limits.max_string_column_bytes
                                                             : unlimited;

  struct group {
    int device = -1;
    std::vector<std::size_t> members;
    std::size_t bytes        = 0;
    std::size_t rows         = 0;
    std::size_t string_bytes = 0;
    bool held_back           = false;  ///< complete, but its device may not sort another run yet
  };
  // A batch without a bound of its own may be all characters of one string column.
  auto const string_bytes_of = [](parked_batch const& batch) {
    return batch.string_bytes.value_or(batch.bytes);
  };
  auto const takes = [&](group const& g, parked_batch const& batch) {
    if (g.members.empty()) { return true; }
    return batch.bytes <= max_bytes - std::min(g.bytes, max_bytes) &&
           batch.rows.value_or(0) <= max_rows - std::min(g.rows, max_rows) &&
           string_bytes_of(batch) <= max_string_bytes - std::min(g.string_bytes, max_string_bytes);
  };
  auto const add = [&](group& g, std::size_t index, parked_batch const& batch) {
    g.members.push_back(index);
    g.bytes        = memory::saturating_add(g.bytes, batch.bytes);
    g.rows         = memory::saturating_add(g.rows, batch.rows.value_or(0));
    g.string_bytes = memory::saturating_add(g.string_bytes, string_bytes_of(batch));
  };
  auto const full = [&](group const& g) {
    return g.bytes >= max_bytes || g.rows >= max_rows || g.string_bytes >= max_string_bytes;
  };
  // A device with nothing being sorted takes any group, or a run larger than the limit would
  // never be sorted.
  auto const admitted = [&](int device, std::size_t bytes) {
    if (limits.max_concurrent_bytes == 0) { return true; }
    auto const it = sorting.find(device);
    if (it == sorting.end() || it->second == 0) { return true; }
    return bytes <= limits.max_concurrent_bytes - std::min(it->second, limits.max_concurrent_bytes);
  };

  // One open group per location, in order of first appearance.
  std::vector<group> groups;
  for (std::size_t i = 0; i < parked.size(); ++i) {
    auto const& batch = parked[i];
    auto it           = std::find_if(
      groups.begin(), groups.end(), [&](group const& g) { return g.device == batch.device; });
    if (it == groups.end()) {
      groups.push_back(group{.device = batch.device});
      it = std::prev(groups.end());
    }
    if (it->held_back) { continue; }
    if (takes(*it, batch)) {
      add(*it, i, batch);
      if (!full(*it)) { continue; }
    }
    if (admitted(it->device, it->bytes)) { return it->members; }
    it->held_back = true;
  }
  if (!producer_finished || parked.empty()) { return {}; }

  // Nothing more will arrive: pack what is left in repository order, whatever its location.
  group tail;
  for (std::size_t i = 0; i < parked.size(); ++i) {
    if (!takes(tail, parked[i])) { break; }
    add(tail, i, parked[i]);
  }
  if (!admitted(run_device(parked, tail.members), tail.bytes)) { return {}; }
  return tail.members;
}

std::vector<sirius_physical_order::parked_batch> sirius_physical_order::snapshot_parked(
  ::cucascade::shared_data_repository& repo)
{
  std::vector<parked_batch> parked;
  auto const ids = repo.get_batch_ids(0);
  parked.reserve(ids.size());
  for (auto const id : ids) {
    auto batch = repo.get_data_batch_by_id(id, 0);
    if (!batch) { continue; }
    parked_batch entry{.id = id};
    // A batch that is being moved between tiers is exclusively locked: do not wait for it here.
    // Its location is then "not on a GPU", which is where a batch under downgrade is going.
    auto const read_only = batch->try_to_read_only();
    if (read_only) {
      auto* space = read_only->get_memory_space();
      if (space != nullptr && space->get_tier() == cucascade::memory::Tier::GPU) {
        entry.device = space->get_device_id();
      }
    }
    auto cached = _stats_cache.find(id);
    if (cached == _stats_cache.end()) {
      cached_stats stats;
      if (read_only && read_only->get_data() != nullptr) {
        stats.rows  = representation_num_rows(*read_only->get_data());
        stats.bytes = read_only->get_data()->get_uncompressed_data_size_in_bytes();
      } else {
        auto const measured = get_batch_rows_and_bytes(*batch);
        stats.rows          = measured.rows;
        stats.bytes         = measured.bytes;
      }
      cached = _stats_cache.emplace(id, stats).first;
      ++_seen_batches;
      _seen_bytes = memory::saturating_add(_seen_bytes, stats.bytes);
    }
    entry.bytes = cached->second.bytes;
    entry.rows  = cached->second.rows;
    if (_has_string_column && entry.rows) {
      // What the batch holds beyond its fixed-width values and string offsets is at most the
      // characters of its string columns.
      auto const other   = memory::saturating_mul(*entry.rows, _non_character_row_bytes);
      entry.string_bytes = entry.bytes - std::min(entry.bytes, other);
    }
    parked.push_back(entry);
  }
  return parked;
}

std::optional<task_creation_hint> sirius_physical_order::get_next_task_hint()
{
  if (!_coalesce) { return sirius_physical_operator::get_next_task_hint(); }
  std::lock_guard<std::mutex> lg(lock);
  if (ports.size() != 1 || ports.begin()->second->repo == nullptr) {
    throw std::runtime_error("sirius_physical_order: coalescing needs exactly one input port");
  }
  auto* const port             = ports.begin()->second;
  bool const producer_finished = is_source_pipeline_finished();
  if (!plan_next_run(snapshot_parked(*port->repo),
                     _limits,
                     _has_string_column,
                     producer_finished,
                     _sorting_bytes)
         .empty()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  // Batches may remain after the producer finished only while their device is sorting as much as
  // it may; the task that finishes there schedules this operator again.
  if (producer_finished) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                            &(port->src_pipeline->get_operators()[0].get())};
}

bool sirius_physical_order::input_backlogged(std::string_view /*port_id*/)
{
  if (!_coalesce || _limits.max_backlog_bytes == 0) { return false; }
  std::lock_guard<std::mutex> lg(lock);
  if (ports.size() != 1 || ports.begin()->second->repo == nullptr) { return false; }
  auto* const port = ports.begin()->second;
  if (is_source_pipeline_finished()) { return false; }
  std::size_t waiting = 0;
  for (auto const& batch : snapshot_parked(*port->repo)) {
    waiting = memory::saturating_add(waiting, batch.bytes);
  }
  // The producer's unfinished tasks deliver about one average batch each. A task counts as
  // finished before its consumers are scheduled, so the request that follows the last of them
  // sees only the parked batches: at the limit one group is complete, and either it is claimed
  // or a run is being sorted, whose task schedules this operator when it finishes.
  if (port->src_pipeline != nullptr && _seen_batches != 0) {
    auto const created   = port->src_pipeline->get_tasks_created();
    auto const completed = port->src_pipeline->get_tasks_completed();
    auto const unfinished = created - std::min(created, completed);
    waiting = memory::saturating_add(
      waiting, memory::saturating_mul(unfinished, _seen_bytes / _seen_batches));
  }
  return waiting >= _limits.max_backlog_bytes;
}

void sirius_physical_order::finish_claim(uint64_t first_batch_id)
{
  {
    std::lock_guard<std::mutex> lg(lock);
    auto const it = _claims.find(first_batch_id);
    if (it == _claims.end()) { return; }
    auto& sorting = _sorting_bytes[it->second.device];
    sorting      -= std::min(sorting, it->second.bytes);
    _claims.erase(it);
  }
  // No producer task schedules this operator for a group that was held back, or for the
  // producer it stopped: ask for the next task here.
  if (_limits.max_concurrent_bytes == 0 && _limits.max_backlog_bytes == 0) { return; }
  if (auto pipeline = get_pipeline()) {
    if (auto* creator = pipeline->get_task_creator()) { creator->schedule(this); }
  }
}

std::unique_ptr<operator_data> sirius_physical_order::get_next_task_input_data()
{
  if (!_coalesce) { return sirius_physical_operator::get_next_task_input_data(); }
  std::lock_guard<std::mutex> lg(lock);
  if (ports.size() != 1 || ports.begin()->second->repo == nullptr) {
    throw std::runtime_error("sirius_physical_order: coalescing needs exactly one input port");
  }
  auto* const port = ports.begin()->second;
  // Read completion before the repository: once it is visible no further batch can arrive, so
  // the snapshot below is the complete remainder.
  bool const producer_finished = is_source_pipeline_finished();
  auto const parked            = snapshot_parked(*port->repo);
  auto const members =
    plan_next_run(parked, _limits, _has_string_column, producer_finished, _sorting_bytes);
  if (members.empty()) { return nullptr; }

  std::vector<std::shared_ptr<::cucascade::data_batch>> input_batches;
  input_batches.reserve(members.size());
  std::size_t claimed_bytes = 0;
  for (auto const index : members) {
    auto batch = port->repo->pop_data_batch_by_id(parked[index].id, 0);
    _stats_cache.erase(parked[index].id);
    if (batch) {
      claimed_bytes = memory::saturating_add(claimed_bytes, parked[index].bytes);
      input_batches.push_back(std::move(batch));
    }
  }
  if (input_batches.empty()) { return nullptr; }
  auto const device = run_device(parked, members);
  _claims[input_batches.front()->get_batch_id()] = claim{.device = device, .bytes = claimed_bytes};
  _sorting_bytes[device] = memory::saturating_add(_sorting_bytes[device], claimed_bytes);
  auto input = std::make_unique<pipelineable_operator_data>(std::move(input_batches));
  input->set_materialize_on_gpu(port->residency.materialize_inputs_on_gpu);
  return input;
}

order_by_counters sirius_physical_order::counters() const noexcept
{
  return {.tasks              = _tasks.load(std::memory_order_relaxed),
          .input_batches      = _input_batches.load(std::memory_order_relaxed),
          .runs               = _runs.load(std::memory_order_relaxed),
          .concatenated_bytes = _concatenated_bytes.load(std::memory_order_relaxed)};
}

order_by_counters sirius_physical_order::process_counters() noexcept
{
  auto const& process = process_storage();
  return {.tasks              = process.tasks.load(std::memory_order_relaxed),
          .input_batches      = process.input_batches.load(std::memory_order_relaxed),
          .runs               = process.runs.load(std::memory_order_relaxed),
          .concatenated_bytes = process.concatenated_bytes.load(std::memory_order_relaxed)};
}

void sirius_physical_order::reset_process_counters() noexcept
{
  auto& process = process_storage();
  process.tasks.store(0, std::memory_order_relaxed);
  process.input_batches.store(0, std::memory_order_relaxed);
  process.runs.store(0, std::memory_order_relaxed);
  process.concatenated_bytes.store(0, std::memory_order_relaxed);
}

void sirius_physical_order::on_finalize_operator()
{
  auto const stats = counters();
  SIRIUS_LOG_INFO(
    "[order_by][stats] op={} coalesce={} max_run_bytes={} max_concurrent_bytes={} "
    "max_backlog_bytes={} tasks={} batches_in={} runs_out={} concat_bytes={}",
    has_operator_id() ? std::to_string(get_operator_id()) : std::string("-"),
    _coalesce ? 1 : 0,
    _coalesce ? _limits.max_run_bytes : 0,
    _coalesce ? _limits.max_concurrent_bytes : 0,
    _coalesce ? _limits.max_backlog_bytes : 0,
    stats.tasks,
    stats.input_batches,
    stats.runs,
    stats.concatenated_bytes);
}

std::size_t sirius_physical_order::no_history_peak_memory_estimate(const input_stats& stats) const
{
  // One batch: the sort's scratch and the sorted copy. Several batches: their concatenation as
  // well.
  return memory::saturating_mul(stats.bytes, (_coalesce && stats.num_batches > 1) ? 3 : 2);
}

}  // namespace op
}  // namespace sirius
