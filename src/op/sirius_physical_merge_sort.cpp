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

#include "op/sirius_physical_merge_sort.hpp"

#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "exec/config.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/merge/merge_chunk_position.hpp"
#include "op/merge/windowed_merge_input.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/column_metadata.hpp>

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace sirius {
namespace op {

namespace {

/// GPU memory assumed for sizing when neither a provider nor a GPU-resident run names a device.
constexpr std::size_t kUnknownDeviceBytes = std::size_t{8} << 30;

bool is_nested_type(cudf::data_type type) noexcept
{
  return type.id() == cudf::type_id::LIST || type.id() == cudf::type_id::STRUCT;
}

std::vector<cudf::data_type> types_of(
  const std::vector<cucascade::memory::column_metadata>& columns)
{
  std::vector<cudf::data_type> result;
  result.reserve(columns.size());
  for (auto const& column : columns) {
    auto const id = static_cast<cudf::type_id>(column.type_id);
    result.push_back(cudf::is_fixed_point(cudf::data_type{id}) ? cudf::data_type{id, column.scale}
                                                               : cudf::data_type{id});
  }
  return result;
}

/// Column types of a batch representation, or nullopt when the representation does not expose
/// them (a compressed one).
std::optional<std::vector<cudf::data_type>> representation_column_types(
  const cucascade::idata_representation& data)
{
  if (auto const* gpu = dynamic_cast<const cucascade::gpu_table_representation*>(&data)) {
    auto const view = gpu->get_table_view();
    std::vector<cudf::data_type> result;
    result.reserve(static_cast<std::size_t>(view.num_columns()));
    for (auto const& column : view) {
      result.push_back(column.type());
    }
    return result;
  }
  if (auto const* host = dynamic_cast<const cucascade::host_data_representation*>(&data)) {
    auto const& table = host->get_host_table();
    if (!table) { return std::nullopt; }
    return types_of(table->columns);
  }
  if (auto const* disk = dynamic_cast<const cucascade::disk_data_representation*>(&data)) {
    return types_of(disk->get_disk_table().columns);
  }
  return std::nullopt;
}

/// Whether the windowed merge can read @p data by row range; otherwise it converts the whole
/// batch to the GPU tier.
bool has_row_range_reader(const cucascade::idata_representation& data)
{
  if (sirius::row_range_reader_registry::is_initialized()) {
    return sirius::row_range_reader_registry::get().has_reader(data);
  }
  return dynamic_cast<const cucascade::host_data_representation*>(&data) != nullptr ||
         dynamic_cast<const cucascade::disk_data_representation*>(&data) != nullptr;
}

/// Estimated bytes per row of each column: the fixed width, or an equal share of what the
/// fixed-width columns leave of the average row.
std::vector<std::size_t> estimate_column_widths(const std::vector<cudf::data_type>& types,
                                                std::size_t total_bytes,
                                                std::size_t total_rows)
{
  std::size_t fixed_bytes      = 0;
  std::size_t variable_columns = 0;
  for (auto const& type : types) {
    if (cudf::is_fixed_width(type)) {
      fixed_bytes += cudf::size_of(type);
    } else {
      ++variable_columns;
    }
  }
  auto const row_bytes =
    total_rows == 0 ? std::size_t{0} : (total_bytes + total_rows - 1) / total_rows;
  auto const variable_width =
    variable_columns == 0
      ? std::size_t{0}
      : std::max<std::size_t>(
          (row_bytes > fixed_bytes ? row_bytes - fixed_bytes : 0) / variable_columns, 1);
  std::vector<std::size_t> widths;
  widths.reserve(types.size());
  for (auto const& type : types) {
    widths.push_back(cudf::is_fixed_width(type) ? cudf::size_of(type) : variable_width);
  }
  return widths;
}

/// The columns one windowed task reads and how its output is assembled from them.
struct fetch_plan {
  std::vector<std::size_t> fetch_cols;  ///< input column indices, in fetch order
  std::vector<std::size_t> output;      ///< per output column, its index into fetch_cols
  merge::merge_key_spec keys;           ///< key_cols index into fetch_cols
};

/// A final pass fetches the output columns and then the keys they lack; an intermediate pass
/// fetches every column, because its output is read again as a run of the same partition.
fetch_plan make_fetch_plan(const merge::merge_key_spec& input_keys,
                           const std::vector<std::size_t>& output_columns,
                           std::size_t num_input_columns,
                           bool final_pass)
{
  fetch_plan plan;
  plan.keys              = input_keys;
  auto const fetch_index = [&plan](std::size_t column) {
    auto const it = std::find(plan.fetch_cols.begin(), plan.fetch_cols.end(), column);
    if (it != plan.fetch_cols.end()) {
      return static_cast<std::size_t>(it - plan.fetch_cols.begin());
    }
    plan.fetch_cols.push_back(column);
    return plan.fetch_cols.size() - 1;
  };
  if (final_pass) {
    for (auto column : output_columns) {
      plan.output.push_back(fetch_index(column));
    }
  } else {
    for (std::size_t column = 0; column < num_input_columns; ++column) {
      plan.output.push_back(fetch_index(column));
    }
  }
  for (auto& key : plan.keys.key_cols) {
    key = static_cast<cudf::size_type>(fetch_index(static_cast<std::size_t>(key)));
  }
  return plan;
}

std::size_t sum_widths(const std::vector<std::size_t>& widths,
                       const std::vector<std::size_t>& columns)
{
  std::size_t total = 0;
  for (auto column : columns) {
    total += widths[column];
  }
  return total;
}

}  // namespace

sirius_physical_merge_sort::sirius_physical_merge_sort(sirius_physical_order* order_by)
  : sirius_physical_merge_sort(
      order_by->types,                // copied by value
      copy_orders(order_by->orders),  // deep copy (contains unique_ptr<Expression>)
      order_by->projections,          // copied by value
      order_by->estimated_cardinality,
      order_by->is_index_sort)
{
  child_op = order_by;
}

sirius_physical_merge_sort::sirius_physical_merge_sort(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::vector<std::size_t> projections_p,
  std::size_t estimated_cardinality,
  bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_SORT, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

bool sirius_physical_merge_sort::has_late_materialization() const noexcept
{
  return !port_directive().empty() || !deferred_output().empty();
}

port_residency_policy sirius_physical_merge_sort::input_residency_policy(std::string_view) const
{
  if (_merge_settings.mode == sort_merge_mode::one_shot) { return {}; }
  return {.materialize_inputs_on_gpu = false, .downgrade_priority = 100};
}

std::size_t sirius_physical_merge_sort::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  if (stats.type == operator_data_type::WINDOWED_MERGE) { return stats.bytes; }
  return sirius_physical_operator::no_history_peak_memory_estimate(stats);
}

merge::merge_key_spec sirius_physical_merge_sort::input_key_spec() const
{
  merge::merge_key_spec spec;
  spec.key_cols.reserve(orders.size());
  spec.column_order.reserve(orders.size());
  spec.null_precedence.reserve(orders.size());
  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Merge sort only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
    spec.key_cols.push_back(idx);
    spec.column_order.push_back(to_cudf_order(ord.type));
    spec.null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }
  return spec;
}

std::vector<std::size_t> sirius_physical_merge_sort::output_columns(
  std::size_t num_input_columns) const
{
  if (!_final_projections.empty()) {
    return {_final_projections.begin(), _final_projections.end()};
  }
  std::vector<std::size_t> columns(num_input_columns);
  std::iota(columns.begin(), columns.end(), std::size_t{0});
  return columns;
}

bool sirius_physical_merge_sort::has_armable_partition(::cucascade::shared_data_repository& repo)
{
  auto const num_partitions = repo.num_partitions();
  for (std::size_t p = 0; p < num_partitions; ++p) {
    auto const it = _partitions.find(p);
    if (it == _partitions.end() || !it->second.armed) {
      if (it != _partitions.end() && it->second.in_flight) { continue; }
      if (repo.size(p) > 0) { return true; }
      continue;
    }
    if (!it->second.in_flight && !it->second.runs.empty()) { return true; }
  }
  return false;
}

std::optional<task_creation_hint> sirius_physical_merge_sort::get_next_task_hint()
{
  auto hint = sirius_physical_operator::get_next_task_hint();
  if (!hint.has_value() || hint->hint != TaskCreationHint::READY) { return hint; }
  std::lock_guard<std::mutex> lg(_drain_mutex);
  auto* repo = ports.begin()->second->repo;
  if (repo == nullptr || has_armable_partition(*repo)) { return hint; }
  // Every partition that still has runs has a task in flight; that task schedules this operator
  // again when it commits.
  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(_drain_mutex);

  auto* repo = ports.begin()->second->repo;
  if (repo == nullptr) { return nullptr; }
  auto const num_partitions = repo->num_partitions();
  for (std::size_t p = 0; p < num_partitions; ++p) {
    auto it = _partitions.find(p);
    if (it == _partitions.end()) {
      if (repo->size(p) == 0) { continue; }
      it = _partitions.emplace(p, partition_merge_state{}).first;
    }
    auto& state = it->second;
    if (state.in_flight) { continue; }
    if (!state.armed) {
      if (auto input = arm_partition(p, state, *repo)) { return input; }
    }
    if (state.runs.empty()) { continue; }
    return make_windowed_input(p, state, *repo);
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::make_one_shot_input(
  std::size_t partition, ::cucascade::shared_data_repository& repo)
{
  // Drain all batches of the partition into one merge task.
  std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
  while (true) {
    auto batch = repo.pop_next_data_batch(partition);
    if (!batch) { break; }
    all_batches.push_back(std::move(batch));
  }
  SIRIUS_LOG_DEBUG(
    "merge_sort: drained {} batches for partition {}", all_batches.size(), partition);
  if (all_batches.empty()) { return nullptr; }
  // The one_shot mode keeps the position it always had; the other modes encode every partition
  // the same way, so one-shot and windowed partitions of one query order consistently.
  auto const position = _merge_settings.mode == sort_merge_mode::one_shot
                          ? partition
                          : merge::make_merge_chunk_position(partition, 0);
  auto input          = std::make_unique<pipelineable_operator_data>(all_batches, position);
  input->set_materialize_on_gpu(true);
  return input;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::arm_partition(
  std::size_t partition, partition_merge_state& state, ::cucascade::shared_data_repository& repo)
{
  state.armed = true;
  state.route = merge::merge_route::one_shot;
  if (_merge_settings.mode == sort_merge_mode::one_shot) {
    return make_one_shot_input(partition, repo);
  }

  struct peeked_batch {
    std::uint64_t id;
    std::size_t rows;
    std::size_t bytes;
  };
  std::vector<peeked_batch> peeked;
  std::optional<std::vector<cudf::data_type>> schema;
  std::unordered_map<cucascade::memory::memory_space*, std::size_t> gpu_bytes;
  std::size_t total_bytes = 0;
  std::size_t total_rows  = 0;
  bool any_non_gpu        = false;
  bool rows_known         = true;

  // Peek: the batches stay in the repository, where they remain spillable.
  auto ids = repo.get_batch_ids(partition);
  std::sort(ids.begin(), ids.end());
  for (auto id : ids) {
    auto batch = repo.get_data_batch_by_id(id, partition);
    if (!batch) { continue; }
    auto const stats = sirius::get_batch_rows_and_bytes(*batch);
    if (!stats.rows.has_value()) { rows_known = false; }
    {
      auto const ro    = batch->to_read_only();
      auto const* data = ro.get_data();
      auto* space      = ro.get_memory_space();
      if (ro.get_current_tier() == cucascade::memory::Tier::GPU && space != nullptr) {
        gpu_bytes[space] += stats.bytes;
      } else {
        any_non_gpu = true;
      }
      if (!schema.has_value() && data != nullptr) { schema = representation_column_types(*data); }
    }
    peeked.push_back({id, stats.rows.value_or(0), stats.bytes});
    total_bytes += stats.bytes;
    total_rows += stats.rows.value_or(0);
  }

  merge::device_budget budget;
  if (_device_budget_provider) {
    budget = _device_budget_provider();
  } else {
    auto const most =
      std::max_element(gpu_bytes.begin(), gpu_bytes.end(), [](auto const& a, auto const& b) {
        return a.second < b.second;
      });
    if (most != gpu_bytes.end()) { _gpu_space_hint = most->first; }
    // The GPU executor's configured width is not reachable from an operator; its default is.
    budget.concurrent_tasks = static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads);
    if (_gpu_space_hint != nullptr) {
      budget.gpu_available_bytes = _gpu_space_hint->get_available_memory();
      budget.gpu_total_bytes     = _gpu_space_hint->get_max_memory();
    } else {
      SIRIUS_LOG_WARN(
        "merge_sort: partition {} has no GPU-resident run and no device budget provider; sizing "
        "its windows for {} bytes of GPU memory",
        partition,
        kUnknownDeviceBytes);
      budget.gpu_available_bytes = kUnknownDeviceBytes;
      budget.gpu_total_bytes     = kUnknownDeviceBytes;
    }
  }

  // The windowed route needs the schema, every batch's row count and plain column keys; a
  // partition that lacks one of them, or has no rows at all, is merged as it always was.
  auto const num_columns = schema.has_value() ? schema->size() : std::size_t{0};
  bool supported         = schema.has_value() && rows_known && total_rows > 0;
  std::vector<std::size_t> key_columns;
  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      supported = false;
      break;
    }
    key_columns.push_back(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
  }
  auto const out_columns = output_columns(num_columns);
  auto const in_range    = [num_columns](std::size_t column) { return column < num_columns; };
  supported = supported && std::all_of(key_columns.begin(), key_columns.end(), in_range) &&
              std::all_of(out_columns.begin(), out_columns.end(), in_range);

  auto const num_runs = static_cast<std::size_t>(
    std::count_if(peeked.begin(), peeked.end(), [](auto const& b) { return b.rows > 0; }));
  std::vector<std::size_t> widths;
  std::size_t row_width = total_rows == 0 ? 0 : (total_bytes + total_rows - 1) / total_rows;
  if (supported) {
    widths    = estimate_column_widths(*schema, total_bytes, total_rows);
    row_width = std::accumulate(widths.begin(), widths.end(), std::size_t{0});
  }
  auto const config = merge::derive_merge_config(_merge_settings, budget, num_runs, row_width);

  merge::route_inputs inputs{.partition_bytes = total_bytes,
                             .output_bytes    = total_bytes,
                             .any_non_gpu     = any_non_gpu,
                             .has_nested      = false,
                             .has_late_mat    = has_late_materialization()};
  if (supported) {
    // A fan-in pass rewrites every column, so with more runs than the cap a nested column
    // counts even when the output does not carry it.
    auto const multi_pass = num_runs > std::max<std::size_t>(config.fan_in, 2);
    for (std::size_t column = 0; column < num_columns; ++column) {
      if (!is_nested_type((*schema)[column])) { continue; }
      auto const fetched =
        std::find(out_columns.begin(), out_columns.end(), column) != out_columns.end() ||
        std::find(key_columns.begin(), key_columns.end(), column) != key_columns.end();
      if (fetched || multi_pass) { inputs.has_nested = true; }
    }
  }
  auto route = merge::choose_route(
    _merge_settings.mode, config.one_shot_fraction, budget.gpu_total_bytes, inputs);
  if (!supported) { route = merge::merge_route::one_shot; }
  SIRIUS_LOG_DEBUG(
    "merge_sort: partition {} armed: route={} runs={} rows={} bytes={} any_non_gpu={} "
    "has_nested={} has_late_mat={}",
    partition,
    route == merge::merge_route::windowed ? "windowed" : "one_shot",
    num_runs,
    total_rows,
    total_bytes,
    any_non_gpu,
    inputs.has_nested,
    inputs.has_late_mat);
  if (route == merge::merge_route::one_shot) { return make_one_shot_input(partition, repo); }

  state.route         = route;
  state.config        = config;
  state.column_types  = std::move(*schema);
  state.column_widths = std::move(widths);
  state.runs.reserve(num_runs);
  for (auto const& batch : peeked) {
    if (batch.rows == 0) {
      // Nothing to merge from an empty batch.
      repo.pop_data_batch_by_id(batch.id, partition);
      continue;
    }
    run_state run;
    run.cursor.run.batch_ids.push_back(batch.id);
    run.cursor.run.batch_rows.push_back(batch.rows);
    run.batch_bytes.push_back(batch.bytes);
    state.runs.push_back(std::move(run));
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::make_windowed_input(
  std::size_t partition, partition_merge_state& state, ::cucascade::shared_data_repository& repo)
{
  // An intermediate pass in progress keeps its runs until they are exhausted.
  std::vector<std::size_t> selected;
  for (std::size_t i = 0; i < state.runs.size(); ++i) {
    if (state.runs[i].in_pass) { selected.push_back(i); }
  }
  bool final_pass = selected.empty();
  if (selected.empty()) {
    std::vector<merge::run_summary> summaries;
    summaries.reserve(state.runs.size());
    for (auto const& run : state.runs) {
      summaries.push_back(
        {.bytes = std::accumulate(run.batch_bytes.begin(), run.batch_bytes.end(), std::size_t{0}),
         .rows  = run.cursor.remaining()});
    }
    auto plan  = merge::plan_pass(summaries, state.config.fan_in);
    selected   = std::move(plan.runs);
    final_pass = plan.final_pass;
    if (!final_pass) {
      for (auto i : selected) {
        state.runs[i].in_pass = true;
      }
      state.pending_run = {};
    }
  }

  std::vector<merge::merge_input_cursor> cursors;
  cursors.reserve(selected.size());
  std::unordered_map<int, std::size_t> device_bytes;
  std::size_t gpu_runs                   = 0;
  std::size_t largest_fallback_run_bytes = 0;
  for (auto i : selected) {
    auto const& run = state.runs[i];
    cursors.push_back(run.cursor);
    bool all_on_gpu = true;
    for (std::size_t b = 0; b < run.cursor.run.batch_ids.size(); ++b) {
      auto const id = run.cursor.run.batch_ids[b];
      auto batch    = repo.get_data_batch_by_id(id, partition);
      if (!batch) {
        throw internal_exception(
          "merge_sort: batch {} of partition {} left the repository before it was merged",
          id,
          partition);
      }
      // A batch that cannot be read-locked right now is being moved between tiers.
      auto const ro    = batch->try_to_read_only();
      auto const* data = ro.has_value() ? ro->get_data() : nullptr;
      auto* space      = ro.has_value() ? ro->get_memory_space() : nullptr;
      if (data != nullptr && space != nullptr &&
          ro->get_current_tier() == cucascade::memory::Tier::GPU) {
        device_bytes[space->get_device_id()] += run.batch_bytes[b];
        continue;
      }
      all_on_gpu = false;
      if (data != nullptr && !has_row_range_reader(*data)) {
        largest_fallback_run_bytes = std::max(largest_fallback_run_bytes, run.batch_bytes[b]);
      }
    }
    if (all_on_gpu) { ++gpu_runs; }
  }

  auto const num_columns = state.column_types.size();
  auto const plan =
    make_fetch_plan(input_key_spec(), output_columns(num_columns), num_columns, final_pass);
  std::vector<std::size_t> key_columns;
  key_columns.reserve(plan.keys.key_cols.size());
  for (auto key : plan.keys.key_cols) {
    key_columns.push_back(plan.fetch_cols[static_cast<std::size_t>(key)]);
  }

  auto input = std::make_unique<merge::windowed_merge_input>(
    partition,
    std::move(cursors),
    final_pass,
    state.next_chunk,
    merge::window_config{.window_rows         = state.config.window_rows,
                         .output_target_bytes = state.config.output_target_bytes},
    sum_widths(state.column_widths, plan.fetch_cols),
    sum_widths(state.column_widths, key_columns),
    gpu_runs,
    largest_fallback_run_bytes);
  if (!device_bytes.empty()) {
    input->set_preferred_device_id(
      std::max_element(device_bytes.begin(), device_bytes.end(), [](auto const& a, auto const& b) {
        return a.second < b.second;
      })->first);
  }
  state.task_runs = std::move(selected);
  state.in_flight = true;
  SIRIUS_LOG_DEBUG(
    "merge_sort: partition {} windowed task: runs={} of {} final_pass={} chunk={} gpu_runs={}",
    partition,
    state.task_runs.size(),
    state.runs.size(),
    final_pass,
    state.next_chunk,
    gpu_runs);
  return input;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute(const operator_data& input_data,
                                                                   ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_merge_sort::execute"};
  if (input_data.get_type() == operator_data_type::WINDOWED_MERGE) {
    return execute_windowed(dynamic_cast<const merge::windowed_merge_input&>(input_data), stream);
  }
  return execute_one_shot(dynamic_cast<const pipelineable_operator_data&>(input_data), stream);
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_one_shot(
  const pipelineable_operator_data& input, ::cuda::stream_ref stream)
{
  const auto& input_batches = input.get_read_only_batches();

  // Find memory space
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!space) { space = batch.get_memory_space(); }
  }

  if (input_batches.empty() || !space) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  // Helper lambda to apply final projection to a batch (removes sort-key-only columns)
  auto apply_final_projection =
    [this, stream, space](
      const cucascade::read_only_data_batch& ro) -> std::shared_ptr<cucascade::data_batch> {
    auto table_view = sirius::get_cudf_table_view(ro);
    std::vector<cudf::column_view> projected_cols;
    for (auto idx : _final_projections) {
      projected_cols.push_back(table_view.column(static_cast<cudf::size_type>(idx)));
    }
    auto projected_table = std::make_unique<cudf::table>(
      cudf::table_view(projected_cols), stream, space->get_default_allocator());
    return sirius::make_data_batch(std::move(projected_table), *space, stream, batch_telemetry());
  };

  // Single batch: no merge needed
  if (input_batches.size() == 1) {
    std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
    if (_final_projections.empty()) {
      // Forward the owned input batch (idle at park); no read lock carried.
      outputs.push_back(input.get_data_batches()[0]);
    } else {
      outputs.push_back(apply_final_projection(input_batches[0]));
    }
    return std::make_unique<pipelineable_operator_data>(outputs, input.get_position());
  }

  auto const keys = input_key_spec();

  auto merged_batch = gpu_merge_impl::merge_order_by(input_batches,
                                                     keys.key_cols,
                                                     keys.column_order,
                                                     keys.null_precedence,
                                                     stream,
                                                     *space,
                                                     batch_telemetry());

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  if (merged_batch) {
    if (_final_projections.empty()) {
      outputs.push_back(std::move(merged_batch));
    } else {
      auto ro = merged_batch->to_read_only();
      outputs.push_back(apply_final_projection(ro));
    }
  }
  return std::make_unique<pipelineable_operator_data>(outputs, input.get_position());
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_windowed(
  const merge::windowed_merge_input& input, ::cuda::stream_ref stream)
{
  auto const partition  = input.partition();
  auto const final_pass = input.final_pass();
  auto* repo            = ports.begin()->second->repo;
  if (repo == nullptr) {
    throw internal_exception("merge_sort: a windowed task needs the input repository");
  }

  std::vector<cudf::data_type> column_types;
  {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    column_types = _partitions.at(partition).column_types;
  }
  auto const num_columns = column_types.size();
  auto plan =
    make_fetch_plan(input_key_spec(), output_columns(num_columns), num_columns, final_pass);
  std::vector<cudf::data_type> fetch_types;
  fetch_types.reserve(plan.fetch_cols.size());
  for (auto column : plan.fetch_cols) {
    fetch_types.push_back(column_types[column]);
  }

  // The task works on its own copy of the cursors; the partition's are replaced only once the
  // merge has returned, so a failed attempt replays from where this one started.
  auto cursors = input.cursors();
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> runs;
  runs.reserve(cursors.size());
  for (auto const& cursor : cursors) {
    auto& batches = runs.emplace_back();
    batches.reserve(cursor.run.batch_ids.size());
    for (auto id : cursor.run.batch_ids) {
      auto batch = repo->get_data_batch_by_id(id, partition);
      if (!batch) {
        throw internal_exception(
          "merge_sort: batch {} of partition {} left the repository before it was merged",
          id,
          partition);
      }
      batches.push_back(std::move(batch));
    }
  }

  auto& space = input.gpu_memory_space();
  auto mr     = space.get_default_allocator();
  rmm::cuda_stream_view stream_view{stream.get()};
  merge::merge_round_stats stats;
  std::unique_ptr<cudf::table> merged;
  {
    // Scoped so the merger's read locks and batch references are gone before the commit pops.
    merge::windowed_merger merger(std::move(runs),
                                  cursors,
                                  plan.fetch_cols,
                                  std::move(fetch_types),
                                  plan.keys,
                                  input.config(),
                                  sirius::row_range_reader_registry::get(),
                                  space);
    merged = merger.run(stream_view, mr, &stats);
  }
  auto const rows_out      = merged ? static_cast<std::size_t>(merged->num_rows()) : std::size_t{0};
  auto const all_exhausted = std::all_of(
    cursors.begin(), cursors.end(), [](auto const& cursor) { return cursor.exhausted(); });
  if (rows_out == 0 && !all_exhausted) {
    throw internal_exception("merge_sort: a windowed task of partition {} made no progress",
                             partition);
  }

  std::shared_ptr<cucascade::data_batch> output_batch;
  if (rows_out > 0) {
    // Assemble the output columns from the fetched ones: key-only columns are dropped, and a
    // column the output repeats is copied for every use but its last.
    bool identity = plan.output.size() == plan.fetch_cols.size();
    for (std::size_t i = 0; identity && i < plan.output.size(); ++i) {
      identity = plan.output[i] == i;
    }
    if (!identity) {
      auto fetched = merged->release();
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.reserve(plan.output.size());
      for (std::size_t i = 0; i < plan.output.size(); ++i) {
        auto const source   = plan.output[i];
        auto const last_use = std::find(plan.output.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                                        plan.output.end(),
                                        source) == plan.output.end();
        columns.push_back(last_use
                            ? std::move(fetched[source])
                            : std::make_unique<cudf::column>(*fetched[source], stream_view, mr));
      }
      merged = std::make_unique<cudf::table>(std::move(columns));
    }
    output_batch = sirius::make_data_batch(std::move(merged), space, stream, batch_telemetry());
  }
  auto const output_bytes =
    output_batch && !final_pass ? sirius::get_batch_rows_and_bytes(*output_batch).bytes : 0;
  // An intermediate run is read again by a later pass of this partition.
  if (output_batch && !final_pass) { repo->add_data_batch(output_batch, partition); }

  // Commit: nothing after the merge may fail, or the cursors would be applied twice on retry.
  std::vector<std::shared_ptr<cucascade::data_batch>> consumed_batches;
  bool more_tasks = false;
  {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    auto& state = _partitions.at(partition);
    for (std::size_t i = 0; i < cursors.size(); ++i) {
      auto& run           = state.runs[state.task_runs[i]];
      auto const consumed = cursors[i].batch;
      // Popping a fully read batch frees it.
      for (std::size_t b = 0; b < consumed; ++b) {
        consumed_batches.push_back(
          repo->pop_data_batch_by_id(cursors[i].run.batch_ids[b], partition));
      }
      run.cursor   = std::move(cursors[i]);
      auto const n = static_cast<std::ptrdiff_t>(consumed);
      run.cursor.run.batch_ids.erase(run.cursor.run.batch_ids.begin(),
                                     run.cursor.run.batch_ids.begin() + n);
      run.cursor.run.batch_rows.erase(run.cursor.run.batch_rows.begin(),
                                      run.cursor.run.batch_rows.begin() + n);
      run.batch_bytes.erase(run.batch_bytes.begin(), run.batch_bytes.begin() + n);
      run.cursor.batch = 0;
    }
    if (final_pass) {
      if (output_batch) { ++state.next_chunk; }
    } else if (output_batch) {
      state.pending_run.cursor.run.batch_ids.push_back(output_batch->get_batch_id());
      state.pending_run.cursor.run.batch_rows.push_back(rows_out);
      state.pending_run.batch_bytes.push_back(output_bytes);
    }
    std::erase_if(state.runs, [](auto const& run) { return run.cursor.exhausted(); });
    auto const pass_in_progress = std::any_of(
      state.runs.begin(), state.runs.end(), [](auto const& run) { return run.in_pass; });
    if (!final_pass && !pass_in_progress) {
      // The intermediate run is complete: it joins the partition's runs.
      if (!state.pending_run.cursor.run.batch_ids.empty()) {
        state.runs.push_back(std::move(state.pending_run));
      }
      state.pending_run = {};
    }
    state.task_runs.clear();
    state.in_flight = false;
    more_tasks      = has_armable_partition(*repo);
  }
  consumed_batches.clear();
  SIRIUS_LOG_DEBUG(
    "merge_sort: partition {} windowed task done: final_pass={} rounds={} rows_out={} "
    "refill_rows={} fallback_conversions={}",
    partition,
    final_pass,
    stats.rounds,
    rows_out,
    stats.refill_rows,
    stats.fallback_conversions);

  // Nothing else asks this operator for its next task when one of its own tasks finishes.
  if (more_tasks) {
    if (auto pipeline = get_pipeline()) {
      if (auto* creator = pipeline->get_task_creator()) { creator->schedule(this); }
    }
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  if (final_pass && output_batch) {
    outputs.push_back(std::move(output_batch));
    return std::make_unique<pipelineable_operator_data>(
      std::move(outputs), merge::make_merge_chunk_position(partition, input.first_chunk()));
  }
  return std::make_unique<pipelineable_operator_data>(std::move(outputs));
}

}  // namespace op
}  // namespace sirius
