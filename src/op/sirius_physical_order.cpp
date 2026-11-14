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
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

void sirius_physical_order::enable_run_coalescing(sort_run_limits limits)
{
  if (limits.max_run_bytes == 0) {
    throw std::invalid_argument("sirius_physical_order: max_run_bytes must be positive");
  }
  _limits            = limits;
  _coalesce          = true;
  _has_string_column = std::any_of(
    types.begin(), types.end(), [](sirius::logical_type const& type) { return type.is_varchar(); });
}

MemoryBarrierType sirius_physical_order::input_barrier_for(
  sirius_physical_operator const& producer) const
{
  if (_coalesce && producer.type == SiriusPhysicalOperatorType::PASSTHROUGH_SINK) {
    return MemoryBarrierType::PARTIAL;
  }
  return sirius_physical_operator::input_barrier_for(producer);
}

std::vector<std::size_t> sirius_physical_order::plan_next_run(
  const std::vector<parked_batch>& parked,
  const sort_run_limits& limits,
  bool has_string_column,
  bool producer_finished)
{
  // Decoded bytes bound every string column's character bytes, so one byte limit covers both.
  auto max_bytes = limits.max_run_bytes == 0 ? std::numeric_limits<std::size_t>::max()
                                             : limits.max_run_bytes;
  if (has_string_column && limits.max_string_column_bytes != 0) {
    max_bytes = std::min(max_bytes, limits.max_string_column_bytes);
  }
  auto const max_rows =
    limits.max_run_rows == 0 ? std::numeric_limits<std::size_t>::max() : limits.max_run_rows;

  struct group {
    int device = -1;
    std::vector<std::size_t> members;
    std::size_t bytes = 0;
    std::size_t rows  = 0;
  };
  auto const takes = [&](group const& g, parked_batch const& batch) {
    if (g.members.empty()) { return true; }
    return batch.bytes <= max_bytes - std::min(g.bytes, max_bytes) &&
           batch.rows.value_or(0) <= max_rows - std::min(g.rows, max_rows);
  };
  auto const add = [](group& g, std::size_t index, parked_batch const& batch) {
    g.members.push_back(index);
    g.bytes = memory::saturating_add(g.bytes, batch.bytes);
    g.rows  = memory::saturating_add(g.rows, batch.rows.value_or(0));
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
    if (!takes(*it, batch)) { return it->members; }
    add(*it, i, batch);
    if (it->bytes >= max_bytes || it->rows >= max_rows) { return it->members; }
  }
  if (!producer_finished || parked.empty()) { return {}; }

  // Nothing more will arrive: pack what is left in repository order, whatever its location.
  group tail;
  for (std::size_t i = 0; i < parked.size(); ++i) {
    if (!takes(tail, parked[i])) { break; }
    add(tail, i, parked[i]);
  }
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
    }
    entry.bytes = cached->second.bytes;
    entry.rows  = cached->second.rows;
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
  if (producer_finished) {
    if (port->repo->total_size() > 0) { return task_creation_hint{TaskCreationHint::READY, this}; }
    return std::nullopt;
  }
  if (!plan_next_run(snapshot_parked(*port->repo), _limits, _has_string_column, false).empty()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                            &(port->src_pipeline->get_operators()[0].get())};
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
  auto const members = plan_next_run(parked, _limits, _has_string_column, producer_finished);
  if (members.empty()) { return nullptr; }

  std::vector<std::shared_ptr<::cucascade::data_batch>> input_batches;
  input_batches.reserve(members.size());
  for (auto const index : members) {
    auto batch = port->repo->pop_data_batch_by_id(parked[index].id, 0);
    _stats_cache.erase(parked[index].id);
    if (batch) { input_batches.push_back(std::move(batch)); }
  }
  if (input_batches.empty()) { return nullptr; }
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
    "[order_by][stats] op={} coalesce={} max_run_bytes={} tasks={} batches_in={} runs_out={} "
    "concat_bytes={}",
    has_operator_id() ? std::to_string(get_operator_id()) : std::string("-"),
    _coalesce ? 1 : 0,
    _coalesce ? _limits.max_run_bytes : 0,
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
