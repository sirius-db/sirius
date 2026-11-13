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

#include "compression/compressed_disk_representation.hpp"
#include "compression/compressed_payload_sources.hpp"
#include "compression/compressed_representation.hpp"
#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "data/host_column_carrier.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "exec/config.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/merge/merge_chunk_position.hpp"
#include "op/merge/merge_sort_stats_log.hpp"
#include "op/merge/windowed_merge_input.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <api/chunk_decode.hpp>
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

std::vector<cudf::data_type> types_of(
  const std::vector<cucascade::memory::column_metadata>& columns)
{
  std::vector<cudf::data_type> result;
  result.reserve(columns.size());
  for (auto const& column : columns) {
    result.push_back(sirius::host_column_carrier(column));
  }
  return result;
}

/// The counters of every merge sort operator of the process.
struct process_counter_state {
  std::mutex mutex;
  merge_sort_counters counters;
};

process_counter_state& process_counter_storage()
{
  static process_counter_state state;
  return state;
}

/// Applies @p update to the operator's counters and to the process-wide ones.
template <typename Update>
void count(merge_sort_counters& operator_counters, Update update)
{
  update(operator_counters);
  auto& process = process_counter_storage();
  std::lock_guard<std::mutex> lg(process.mutex);
  update(process.counters);
}

/// Whether the operator has nothing left to merge: no task in flight, no run of an armed
/// partition unread, and no batch waiting in a partition that is not armed yet. With the input
/// complete before the first task, this holds at the commit of the operator's last task only.
template <typename Partitions>
bool no_work_left(const Partitions& partitions, ::cucascade::shared_data_repository& repo)
{
  for (auto const& [partition, state] : partitions) {
    if (state.in_flight || !state.runs.empty()) { return false; }
  }
  auto const num_partitions = repo.num_partitions();
  for (std::size_t p = 0; p < num_partitions; ++p) {
    auto const it = partitions.find(p);
    if ((it == partitions.end() || !it->second.armed) && repo.size(p) > 0) { return false; }
  }
  return true;
}

/// The operator's one statistics line; @p counters are its final ones.
void log_stats(std::size_t operator_id, const merge_sort_counters& counters)
{
  SIRIUS_LOG_INFO("{}", merge::format_merge_sort_stats(operator_id, counters));
}

bool same_position(const merge::merge_input_cursor& a, const merge::merge_input_cursor& b)
{
  return a.batch == b.batch && a.row == b.row && a.run.batch_ids == b.run.batch_ids &&
         a.run.batch_rows == b.run.batch_rows;
}

/// Whether @p representation is its stored chunk, not a projection or a filtered view of it.
template <typename Representation>
bool is_whole_chunk(const Representation& representation)
{
  return !representation.selected_indices().has_value() &&
         representation.pushdown_scan() == nullptr && !representation.visibility_mask().has_mask();
}

/// How much work it is to take the schema of a spill-compressed batch from its header: 0 for a
/// host batch (the header is in memory), 1 for a device batch (the header is rebuilt from the
/// table), 2 for a disk batch (the header is read from the file). nullopt for a batch the
/// row-range readers refuse (a projection or a filtered view, whose columns are not the stored
/// ones, or a device batch without a table) and for any other representation.
std::optional<int> compressed_schema_cost(const cucascade::idata_representation& data)
{
  if (auto const* host = dynamic_cast<const sirius::compressed_host_representation*>(&data)) {
    return is_whole_chunk(*host) ? std::optional<int>{0} : std::nullopt;
  }
  if (auto const* device = dynamic_cast<const sirius::compressed_device_representation*>(&data)) {
    return is_whole_chunk(*device) && device->has_table() ? std::optional<int>{1} : std::nullopt;
  }
  if (dynamic_cast<const sirius::compressed_disk_representation*>(&data) != nullptr) { return 2; }
  return std::nullopt;
}

/// Column types of a spill-compressed batch, from its header; nullopt for a batch without a
/// compressed_schema_cost, or when the header cannot be parsed. Parses the header on every call.
std::optional<std::vector<cudf::data_type>> compressed_column_types(
  const cucascade::idata_representation& data)
{
  auto const* compressed = dynamic_cast<const sirius::simpatico_compressed_representation*>(&data);
  if (compressed == nullptr || !compressed_schema_cost(data).has_value()) { return std::nullopt; }
  try {
    auto const description = sirius::compression::describe(*compressed);
    std::vector<cudf::data_type> result;
    result.reserve(description->columns.size());
    for (auto const& column : description->columns) {
      result.push_back(column.dtype);
    }
    return result;
  } catch (const std::exception& e) {
    SIRIUS_LOG_WARN("merge_sort: cannot take the schema of a compressed batch from its header: {}",
                    e.what());
    return std::nullopt;
  }
}

/// Column types of an uncompressed batch representation; nullopt for any other representation
/// (see compressed_column_types for a spill-compressed one).
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

/// Per source column of a spill-compressed batch, whether its row-range reader can decode the
/// column's plan; empty when the header cannot be read. Asks the same cursor the reader builds,
/// which stages nothing when constructed. Parses the header: for a disk batch that reads the
/// file, for a device batch it runs on the default stream.
std::vector<bool> row_range_readable_columns(
  const sirius::simpatico_compressed_representation& compressed)
{
  try {
    auto const opened =
      sirius::compression::open_compressed_payload(compressed, cudf::get_default_stream());
    std::vector<bool> readable;
    readable.reserve(opened.description->columns.size());
    for (std::size_t column = 0; column < opened.description->columns.size(); ++column) {
      const simpatico::compressed_column_cursor cursor(opened.description,
                                                       column,
                                                       opened.source,
                                                       cudf::get_default_stream(),
                                                       rmm::mr::get_current_device_resource_ref());
      readable.push_back(cursor.supported());
    }
    return readable;
  } catch (const std::exception& e) {
    SIRIUS_LOG_WARN("merge_sort: cannot read the plans of a compressed batch from its header: {}",
                    e.what());
    return {};
  }
}

/// Whether the windowed merge has a row-range reader for @p data; otherwise it converts the
/// whole batch to the GPU tier. A compressed batch whose reader refuses it for its plan (known
/// only from its header, see sirius_physical_merge_sort::reads_by_row_range) still counts.
bool has_row_range_reader(const cucascade::idata_representation& data)
{
  if (dynamic_cast<const sirius::simpatico_compressed_representation*>(&data) != nullptr &&
      !compressed_schema_cost(data).has_value()) {
    return false;
  }
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

sirius_physical_merge_sort::device_budget_provider
sirius_physical_merge_sort::make_device_budget_provider(
  const ::cucascade::memory::memory_reservation_manager& memory_manager,
  std::size_t concurrent_tasks)
{
  return [&memory_manager, concurrent_tasks](const cucascade::memory::memory_space* space) {
    merge::device_budget budget;
    budget.concurrent_tasks = std::max<std::size_t>(concurrent_tasks, 1);
    if (space == nullptr) {
      // The task may land on any GPU: size its windows for the smallest one.
      for (auto const* candidate :
           memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
        if (space == nullptr || candidate->get_max_memory() < space->get_max_memory()) {
          space = candidate;
        }
      }
    }
    if (space != nullptr) {
      budget.gpu_available_bytes = space->get_available_memory();
      budget.gpu_total_bytes     = space->get_max_memory();
    }
    return budget;
  };
}

merge_sort_counters sirius_physical_merge_sort::counters() const
{
  std::lock_guard<std::mutex> lg(_drain_mutex);
  return _counters;
}

merge_sort_counters sirius_physical_merge_sort::process_counters()
{
  auto& process = process_counter_storage();
  std::lock_guard<std::mutex> lg(process.mutex);
  return process.counters;
}

void sirius_physical_merge_sort::reset_process_counters()
{
  auto& process = process_counter_storage();
  std::lock_guard<std::mutex> lg(process.mutex);
  process.counters = {};
}

std::optional<merge_sort_counters> sirius_physical_merge_sort::claim_stats_line()
{
  if (_stats_logged) { return std::nullopt; }
  _stats_logged = true;
  return _counters;
}

void sirius_physical_merge_sort::on_finalize_operator()
{
  std::optional<merge_sort_counters> final_counters;
  std::vector<std::unique_ptr<partition_merger>> mergers;
  {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    final_counters = claim_stats_line();
    // The runs stay: a task created after this still merges them, with a merger of its own.
    for (auto& [partition, state] : _partitions) {
      if (state.merger) { mergers.push_back(std::move(state.merger)); }
    }
  }
  // Outside the mutex: frees the window buffers the partitions kept on the GPU.
  mergers.clear();
  if (final_counters.has_value()) { log_stats(operator_id, *final_counters); }
}

void sirius_physical_merge_sort::on_input_batch_pushed(std::string_view,
                                                       ::cucascade::data_batch& batch)
{
  if (_merge_settings.mode == sort_merge_mode::one_shot) { return; }
  // A batch that cannot be read right now is sized when its partition is armed.
  auto const ro = batch.try_to_read_only();
  if (!ro.has_value() || ro->get_data() == nullptr) { return; }
  batch_size size{.rows  = sirius::representation_num_rows(*ro->get_data()),
                  .bytes = ro->get_data()->get_uncompressed_data_size_in_bytes()};
  std::lock_guard<std::mutex> lg(_pushed_mutex);
  _pushed_sizes.insert_or_assign(batch.get_batch_id(), size);
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
      if (it != _partitions.end() &&
          (it->second.in_flight || it->second.arm_deferred || it->second.header_parsing)) {
        continue;
      }
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
  auto* repo = ports.begin()->second->repo;
  if (repo == nullptr) { return nullptr; }
  while (true) {
    // The first batch found locked by a tier move while its partition was being armed.
    std::shared_ptr<cucascade::data_batch> busy;
    // The read lock of the compressed batch whose header names the schema of header_partition.
    std::optional<cucascade::read_only_data_batch> header_source;
    std::size_t header_partition = 0;
    std::optional<windowed_task_plan> task;
    {
      std::lock_guard<std::mutex> lg(_drain_mutex);
      bool task_in_flight       = false;
      auto const num_partitions = repo->num_partitions();
      for (std::size_t p = 0; p < num_partitions; ++p) {
        auto it = _partitions.find(p);
        if (it == _partitions.end()) {
          if (repo->size(p) == 0) { continue; }
          it = _partitions.emplace(p, partition_merge_state{}).first;
        }
        auto& state = it->second;
        if (state.in_flight) {
          task_in_flight = true;
          continue;
        }
        if (!state.armed) {
          // The thread that parses the partition's header arms it.
          if (state.header_parsing) { continue; }
          std::shared_ptr<cucascade::data_batch> locked;
          if (auto input = arm_partition(p, state, *repo, locked, header_source)) { return input; }
          if (header_source.has_value()) {
            state.header_parsing = true;
            header_partition     = p;
            break;
          }
          if (locked) {
            state.arm_deferred = true;
            if (!busy) { busy = std::move(locked); }
            continue;
          }
        }
        if (state.runs.empty()) { continue; }
        task = plan_windowed_task(p, state, *repo);
        break;
      }
      if (!task.has_value() && !header_source.has_value()) {
        // A task in flight arms the deferred partitions again when it commits.
        if (!busy || task_in_flight) { return nullptr; }
        for (auto& [p, state] : _partitions) {
          state.arm_deferred = false;
        }
      }
    }
    // Sizing the task reads its batches: without the mutex.
    if (task.has_value()) { return make_windowed_input(std::move(*task)); }
    if (header_source.has_value()) {
      // For a disk batch this reads the file's header, for a device batch it rebuilds the header
      // on the default stream: without the mutex, which commits and the task hint wait on. The
      // next round arms the partition from its batches as they are then, with this schema.
      auto schema = compressed_column_types(*header_source->get_data());
      header_source.reset();
      std::lock_guard<std::mutex> lg(_drain_mutex);
      auto& state          = _partitions.at(header_partition);
      state.header_schema  = std::move(schema);
      state.header_parsed  = true;
      state.header_parsing = false;
      continue;
    }
    // Nothing else would ask for this partition again: wait for the tier move to finish, without
    // the mutex, and arm again.
    (void)busy->to_read_only();
  }
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
  // The one_shot mode positions a partition's output by the partition index; the other modes
  // encode every partition the same way, so one-shot and windowed partitions of one query order
  // consistently.
  auto const position = _merge_settings.mode == sort_merge_mode::one_shot
                          ? partition
                          : merge::make_merge_chunk_position(partition, 0);
  auto input          = std::make_unique<pipelineable_operator_data>(all_batches, position);
  input->set_materialize_on_gpu(true);
  return input;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::arm_partition(
  std::size_t partition,
  partition_merge_state& state,
  ::cucascade::shared_data_repository& repo,
  std::shared_ptr<::cucascade::data_batch>& busy,
  std::optional<::cucascade::read_only_data_batch>& header_source)
{
  state.route         = merge::merge_route::one_shot;
  auto const one_shot = [&]() {
    state.armed = true;
    auto input  = make_one_shot_input(partition, repo);
    if (input) {
      // In flight until execute() commits it, so the statistics line waits for it.
      state.in_flight = true;
      count(_counters, [](merge_sort_counters& counters) { ++counters.one_shot_partitions; });
    }
    return input;
  };
  if (_merge_settings.mode == sort_merge_mode::one_shot) { return one_shot(); }

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

  // Peek: the batches stay in the repository, where they remain spillable. A batch that cannot
  // be read-locked right now is being moved between tiers: it counts as off the GPU, and its
  // size is the one recorded when it arrived.
  auto ids = repo.get_batch_ids(partition);
  std::sort(ids.begin(), ids.end());
  std::shared_ptr<cucascade::data_batch> first_locked;
  std::shared_ptr<cucascade::data_batch> unsized_locked;
  // The read lock of the compressed batch whose header is the cheapest to parse.
  std::optional<cucascade::read_only_data_batch> schema_source;
  int schema_source_cost = 0;
  for (auto id : ids) {
    auto batch = repo.get_data_batch_by_id(id, partition);
    if (!batch) { continue; }
    std::optional<batch_size> size;
    {
      std::lock_guard<std::mutex> lg(_pushed_mutex);
      if (auto const it = _pushed_sizes.find(id); it != _pushed_sizes.end()) { size = it->second; }
    }
    if (auto ro = batch->try_to_read_only(); ro.has_value()) {
      auto const* data = ro->get_data();
      auto* space      = ro->get_memory_space();
      if (!size.has_value()) {
        size = data == nullptr ? batch_size{}
                               : batch_size{.rows  = sirius::representation_num_rows(*data),
                                            .bytes = data->get_uncompressed_data_size_in_bytes()};
        std::lock_guard<std::mutex> lg(_pushed_mutex);
        _pushed_sizes.insert_or_assign(id, *size);
      }
      if (ro->get_current_tier() == cucascade::memory::Tier::GPU && space != nullptr) {
        gpu_bytes[space] += size->bytes;
      } else {
        any_non_gpu = true;
      }
      if (!schema.has_value() && data != nullptr) {
        schema = representation_column_types(*data);
        // A compressed batch names its columns only in its header. It is parsed by the caller,
        // for one batch, and only if no uncompressed batch has given the schema by then.
        if (auto const cost = schema.has_value() ? std::nullopt : compressed_schema_cost(*data);
            cost.has_value() && (!schema_source.has_value() || *cost < schema_source_cost)) {
          schema_source_cost = *cost;
          schema_source.emplace(std::move(*ro));
        }
      }
    } else {
      any_non_gpu = true;
      if (!size.has_value() && !unsized_locked) { unsized_locked = batch; }
      if (!first_locked) { first_locked = std::move(batch); }
    }
    if (!size.has_value()) { continue; }
    if (!size->rows.has_value()) { rows_known = false; }
    peeked.push_back({id, size->rows.value_or(0), size->bytes});
    total_bytes += size->bytes;
    total_rows += size->rows.value_or(0);
  }
  if (!schema.has_value() && schema_source.has_value()) {
    if (!state.header_parsed) {
      header_source = std::move(schema_source);
      return nullptr;
    }
    schema = state.header_schema;
  }
  schema_source.reset();
  // A locked batch that was never sized, or that may be the only one exposing the schema, has
  // to be read before the route can be chosen: the partition is armed at a later attempt.
  if (unsized_locked || (first_locked && !schema.has_value())) {
    busy = unsized_locked ? std::move(unsized_locked) : std::move(first_locked);
    return nullptr;
  }
  state.armed = true;
  {
    std::lock_guard<std::mutex> lg(_pushed_mutex);
    for (auto id : ids) {
      _pushed_sizes.erase(id);
    }
  }

  auto const most =
    std::max_element(gpu_bytes.begin(), gpu_bytes.end(), [](auto const& a, auto const& b) {
      return a.second < b.second;
    });
  merge::device_budget budget;
  if (_device_budget_provider) {
    budget = _device_budget_provider(most != gpu_bytes.end() ? most->first : nullptr);
  } else {
    // The GPU executor's configured width is not reachable from an operator; its default is.
    budget.concurrent_tasks = static_cast<std::size_t>(exec::default_gpu_pipeline_num_threads);
  }
  if (budget.gpu_total_bytes == 0) {
    // No provider, or one that knows of no GPU memory space.
    if (most != gpu_bytes.end()) { _gpu_space_hint = most->first; }
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
  // partition that lacks one of them, or has no rows at all, is merged one-shot.
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
      if (!cudf::is_nested((*schema)[column])) { continue; }
      auto const fetched =
        std::find(out_columns.begin(), out_columns.end(), column) != out_columns.end() ||
        std::find(key_columns.begin(), key_columns.end(), column) != key_columns.end();
      if (fetched || multi_pass) { inputs.has_nested = true; }
    }
  }
  auto route = merge::choose_route(
    _merge_settings.mode, config.one_shot_fraction, budget.gpu_total_bytes, inputs);
  if (!supported) { route = merge::merge_route::one_shot; }
  if (first_locked) {
    SIRIUS_LOG_DEBUG("merge_sort: partition {} armed while a tier move held one of its runs",
                     partition);
  }
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
  if (route == merge::merge_route::one_shot) { return one_shot(); }

  count(_counters, [](merge_sort_counters& counters) { ++counters.windowed_partitions; });
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
    run.id = state.next_run_id++;
    run.cursor.run.batch_ids.push_back(batch.id);
    run.cursor.run.batch_rows.push_back(batch.rows);
    run.batch_bytes.push_back(batch.bytes);
    state.runs.push_back(std::move(run));
  }
  return nullptr;
}

sirius_physical_merge_sort::windowed_task_plan sirius_physical_merge_sort::plan_windowed_task(
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
      state.pending_run    = {};
      state.pending_run.id = state.next_run_id++;
    }
  }

  windowed_task_plan task;
  task.partition   = partition;
  task.final_pass  = final_pass;
  task.first_chunk = state.next_chunk;
  task.config      = {.window_rows         = state.config.window_rows,
                      .output_target_bytes = state.config.output_target_bytes};
  // execute_windowed() continues the kept merger only for the same pass and window sizes.
  auto const* kept = state.merger.get();
  if (kept != nullptr && kept->space != nullptr) {
    task.kept_device = kept->space->get_device_id();
  }
  auto const continues = kept != nullptr && kept->final_pass == final_pass &&
                         kept->config.window_rows == task.config.window_rows &&
                         kept->config.output_target_bytes == task.config.output_target_bytes;
  task.cursors.reserve(selected.size());
  task.batches.reserve(selected.size());
  task.batch_bytes.reserve(selected.size());
  task.kept_buffer.reserve(selected.size());
  for (auto i : selected) {
    auto const& run = state.runs[i];
    auto& batches   = task.batches.emplace_back();
    batches.reserve(run.cursor.run.batch_ids.size());
    for (auto id : run.cursor.run.batch_ids) {
      auto batch = repo.get_data_batch_by_id(id, partition);
      if (!batch) {
        throw internal_exception(
          "merge_sort: batch {} of partition {} left the repository before it was merged",
          id,
          partition);
      }
      batches.push_back(std::move(batch));
    }
    task.cursors.push_back(run.cursor);
    task.batch_bytes.push_back(run.batch_bytes);
    bool held = false;
    if (continues) {
      auto const it = std::find(kept->run_ids.begin(), kept->run_ids.end(), run.id);
      auto const at = static_cast<std::size_t>(it - kept->run_ids.begin());
      held          = at < kept->holds_buffer.size() && kept->holds_buffer[at];
    }
    task.kept_buffer.push_back(held);
  }

  auto const num_columns = state.column_types.size();
  auto plan =
    make_fetch_plan(input_key_spec(), output_columns(num_columns), num_columns, final_pass);
  std::vector<std::size_t> key_columns;
  key_columns.reserve(plan.keys.key_cols.size());
  for (auto key : plan.keys.key_cols) {
    key_columns.push_back(plan.fetch_cols[static_cast<std::size_t>(key)]);
  }
  task.row_width      = sum_widths(state.column_widths, plan.fetch_cols);
  task.key_row_width  = sum_widths(state.column_widths, key_columns);
  task.string_columns = static_cast<std::size_t>(
    std::count_if(plan.fetch_cols.begin(), plan.fetch_cols.end(), [&state](auto column) {
      return state.column_types[column].id() == cudf::type_id::STRING;
    }));
  task.fetch_cols = std::move(plan.fetch_cols);

  state.task_runs.clear();
  for (auto i : selected) {
    state.task_runs.push_back(state.runs[i].id);
  }
  state.task_buffered.assign(selected.size(), false);
  state.in_flight = true;
  return task;
}

bool sirius_physical_merge_sort::reads_by_row_range(std::uint64_t batch_id,
                                                    const ::cucascade::idata_representation& data,
                                                    const std::vector<std::size_t>& columns)
{
  auto const* compressed = dynamic_cast<const simpatico_compressed_representation*>(&data);
  if (compressed == nullptr) { return true; }
  std::optional<std::vector<bool>> readable;
  {
    std::lock_guard<std::mutex> lg(_row_range_mutex);
    if (auto const it = _row_range_columns.find(batch_id); it != _row_range_columns.end()) {
      readable = it->second;
    }
  }
  if (!readable.has_value()) {
    readable = row_range_readable_columns(*compressed);
    std::lock_guard<std::mutex> lg(_row_range_mutex);
    _row_range_columns.insert_or_assign(batch_id, *readable);
  }
  return std::all_of(columns.begin(), columns.end(), [&readable](auto column) {
    return column < readable->size() && (*readable)[column];
  });
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::make_windowed_input(
  windowed_task_plan plan)
{
  auto const partition = plan.partition;
  auto const num_runs  = plan.cursors.size();
  std::unordered_map<int, std::size_t> device_bytes;
  std::size_t fallback_bytes   = 0;
  std::size_t compressed_runs  = 0;
  std::size_t kept_buffer_runs = 0;
  // Per run: the batch it is at is read through a window buffer, as far as can be told now.
  std::vector<bool> buffered(num_runs, false);
  try {
    for (std::size_t r = 0; r < num_runs; ++r) {
      // The largest batch of the run that may be converted whole to the GPU tier. The converted
      // batches of different runs are on the GPU at once, so the runs add up.
      std::size_t run_fallback_bytes = 0;
      bool any_compressed            = false;
      for (std::size_t b = 0; b < plan.batches[r].size(); ++b) {
        auto const& batch  = plan.batches[r][b];
        auto const bytes   = plan.batch_bytes[r][b];
        auto const current = b == plan.cursors[r].batch;
        // A batch that cannot be read-locked right now is being moved between tiers. What it
        // will be when the task reads it is not known: it may have no reader.
        auto const ro    = batch->try_to_read_only();
        auto const* data = ro.has_value() ? ro->get_data() : nullptr;
        if (data == nullptr) {
          run_fallback_bytes = std::max(run_fallback_bytes, bytes);
          if (current) { buffered[r] = true; }
          continue;
        }
        auto* space = ro->get_memory_space();
        if (space != nullptr && ro->get_current_tier() == cucascade::memory::Tier::GPU) {
          device_bytes[space->get_device_id()] += bytes;
          // Only an uncompressed table is sliced in place; a compressed device batch is read
          // into a window buffer like the batches of the other tiers.
          if (dynamic_cast<const cucascade::gpu_table_representation*>(data) != nullptr) {
            continue;
          }
        }
        if (current) { buffered[r] = true; }
        // No reader for the representation, or one that refuses this batch's plan: the merger
        // decodes the whole batch to the GPU.
        if (!has_row_range_reader(*data) ||
            !reads_by_row_range(batch->get_batch_id(), *data, plan.fetch_cols)) {
          run_fallback_bytes = std::max(run_fallback_bytes, bytes);
        }
        any_compressed = any_compressed ||
                         dynamic_cast<const simpatico_compressed_representation*>(data) != nullptr;
      }
      fallback_bytes = sirius::memory::saturating_add(fallback_bytes, run_fallback_bytes);
      if (any_compressed) { ++compressed_runs; }
      if (plan.kept_buffer[r] && buffered[r]) { ++kept_buffer_runs; }
    }
  } catch (...) {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    auto& state = _partitions.at(partition);
    state.task_runs.clear();
    state.task_buffered.clear();
    state.in_flight = false;
    throw;
  }

  // gpu_runs is 0: a run that is on the GPU now is charged its window buffers like any other.
  // Its batches stay in the repository, which is spilled first (input_residency_policy), so by
  // the time the task reads the run it may be on another tier.
  auto input = std::make_unique<merge::windowed_merge_input>(partition,
                                                             std::move(plan.cursors),
                                                             plan.final_pass,
                                                             plan.first_chunk,
                                                             plan.config,
                                                             plan.row_width,
                                                             plan.key_row_width,
                                                             0,
                                                             fallback_bytes,
                                                             compressed_runs,
                                                             plan.fetch_cols.size(),
                                                             plan.string_columns);
  // The kept merger already holds these runs' window buffers, allocated by an earlier task of
  // the partition: this task reuses them, so they are not reserved again.
  input->set_kept_buffer_runs(kept_buffer_runs);
  // The input shrinks outside the operator, when the executor retries its task after an OOM.
  input->set_shrink_observer([this] {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    count(_counters, [](merge_sort_counters& counters) { ++counters.shrinks; });
  });
  // On another device the kept merger, and every row it buffered, would be thrown away.
  if (plan.kept_device.has_value()) {
    input->set_preferred_device_id(*plan.kept_device);
  } else if (!device_bytes.empty()) {
    input->set_preferred_device_id(
      std::max_element(device_bytes.begin(), device_bytes.end(), [](auto const& a, auto const& b) {
        return a.second < b.second;
      })->first);
  }
  {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    _partitions.at(partition).task_buffered = std::move(buffered);
  }
  SIRIUS_LOG_DEBUG(
    "merge_sort: partition {} windowed task: runs={} final_pass={} chunk={} kept_buffer_runs={} "
    "fallback_bytes={}",
    partition,
    num_runs,
    plan.final_pass,
    plan.first_chunk,
    kept_buffer_runs,
    fallback_bytes);
  return input;
}

sirius_physical_merge_sort::commit_followup sirius_physical_merge_sort::finish_commit(
  ::cucascade::shared_data_repository& repo, bool always_reschedule)
{
  // A partition whose arming met a locked run gets another attempt.
  bool deferred = false;
  for (auto& [partition, state] : _partitions) {
    deferred           = deferred || state.arm_deferred;
    state.arm_deferred = false;
  }
  commit_followup followup;
  followup.reschedule = (always_reschedule || deferred) && has_armable_partition(repo);
  if (no_work_left(_partitions, repo)) { followup.final_counters = claim_stats_line(); }
  return followup;
}

void sirius_physical_merge_sort::after_commit(const commit_followup& followup)
{
  if (followup.final_counters.has_value()) { log_stats(operator_id, *followup.final_counters); }
  // Nothing else asks this operator for its next task when one of its own tasks finishes.
  if (followup.reschedule) {
    if (auto pipeline = get_pipeline()) {
      if (auto* creator = pipeline->get_task_creator()) { creator->schedule(this); }
    }
  }
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute(const operator_data& input_data,
                                                                   ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_merge_sort::execute"};
  if (input_data.get_type() == operator_data_type::WINDOWED_MERGE) {
    return execute_windowed(dynamic_cast<const merge::windowed_merge_input&>(input_data), stream);
  }
  auto const& one_shot = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto output          = execute_one_shot(one_shot, stream);

  // Commit: the partition's only task is done. An input that get_next_task_input_data() did not
  // make has no partition in flight and commits nothing.
  auto* repo = ports.empty() ? nullptr : ports.begin()->second->repo;
  commit_followup followup;
  if (auto const position = one_shot.get_position(); position.has_value() && repo != nullptr) {
    auto const partition = _merge_settings.mode == sort_merge_mode::one_shot
                             ? *position
                             : merge::split_merge_chunk_position(*position).first;
    std::lock_guard<std::mutex> lg(_drain_mutex);
    auto const it = _partitions.find(partition);
    if (it != _partitions.end() && it->second.in_flight &&
        it->second.route == merge::merge_route::one_shot) {
      it->second.in_flight = false;
      // This task was its partition's only one, and the task creator was offered the other
      // partitions when it made this task: only a partition whose arming was deferred until a
      // commit needs the operator scheduled again. That also keeps the one_shot mode's
      // scheduling as it is without the windowed route.
      followup = finish_commit(*repo, false);
    }
  }
  after_commit(followup);
  return output;
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

  // The task takes the partition's merger: only one task of a partition is in flight, so no
  // other thread uses it until the commit puts it back.
  std::vector<cudf::data_type> column_types;
  std::vector<std::uint64_t> task_run_ids;
  std::unique_ptr<partition_merger> kept;
  {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    auto& state  = _partitions.at(partition);
    column_types = state.column_types;
    task_run_ids = state.task_runs;
    kept         = std::move(state.merger);
  }
  auto const num_columns = column_types.size();
  auto plan =
    make_fetch_plan(input_key_spec(), output_columns(num_columns), num_columns, final_pass);
  std::vector<cudf::data_type> fetch_types;
  fetch_types.reserve(plan.fetch_cols.size());
  for (auto column : plan.fetch_cols) {
    fetch_types.push_back(column_types[column]);
  }

  auto& space = input.gpu_memory_space();
  auto mr     = space.get_default_allocator();
  rmm::cuda_stream_view stream_view{stream.get()};

  // The partition's merger continues only from the cursors it left, for the same runs, sizes and
  // device; any other task builds a new one from the cursors of its input.
  if (kept) {
    bool continues = kept->space == &space && kept->final_pass == final_pass &&
                     kept->config.window_rows == input.config().window_rows &&
                     kept->config.output_target_bytes == input.config().output_target_bytes;
    std::size_t matched = 0;
    for (std::size_t i = 0; continues && i < kept->cursors.size(); ++i) {
      auto const it = std::find(task_run_ids.begin(), task_run_ids.end(), kept->run_ids[i]);
      if (it == task_run_ids.end()) {
        // A run read to the end by an earlier task is no longer part of the partition.
        continues = kept->cursors[i].exhausted();
        continue;
      }
      ++matched;
      continues = same_position(
        kept->cursors[i], input.cursors()[static_cast<std::size_t>(it - task_run_ids.begin())]);
    }
    if (!continues || matched != task_run_ids.size()) { kept.reset(); }
  }
  if (!kept) {
    kept             = std::make_unique<partition_merger>();
    kept->run_ids    = std::move(task_run_ids);
    kept->cursors    = input.cursors();
    kept->space      = &space;
    kept->config     = input.config();
    kept->final_pass = final_pass;
    std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> runs;
    runs.reserve(kept->cursors.size());
    for (auto const& cursor : kept->cursors) {
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
    kept->merger.emplace(std::move(runs),
                         kept->cursors,
                         plan.fetch_cols,
                         std::move(fetch_types),
                         plan.keys,
                         input.config(),
                         sirius::row_range_reader_registry::get(),
                         space);
    std::lock_guard<std::mutex> lg(_drain_mutex);
    count(_counters, [](merge_sort_counters& counters) { ++counters.mergers_built; });
  }

  // The merger writes its cursors back only when run() returns. Until the commit below the task
  // owns the merger, so a failure anywhere before it destroys the merger (freeing the windows it
  // buffered) and leaves the partition's cursors where this task started: the retry replays.
  auto& cursors = kept->cursors;
  merge::merge_round_stats stats;
  auto merged              = kept->merger->run(stream_view, mr, &stats);
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

  // Batches read to the end leave the merger and its cursors here, and the repository at the
  // commit, which frees them.
  std::vector<std::uint64_t> consumed_ids;
  std::vector<std::size_t> consumed_counts(cursors.size(), 0);
  for (std::size_t i = 0; i < cursors.size(); ++i) {
    cursors[i].advance(0);
    consumed_counts[i] = cursors[i].batch;
    consumed_ids.insert(
      consumed_ids.end(),
      cursors[i].run.batch_ids.begin(),
      cursors[i].run.batch_ids.begin() + static_cast<std::ptrdiff_t>(cursors[i].batch));
  }
  kept->merger->rebase_consumed();

  // Commit: nothing from here on fails.
  std::vector<std::shared_ptr<cucascade::data_batch>> consumed_batches;
  consumed_batches.reserve(consumed_ids.size());
  commit_followup followup;
  {
    std::lock_guard<std::mutex> lg(_drain_mutex);
    auto& state = _partitions.at(partition);
    for (auto id : consumed_ids) {
      consumed_batches.push_back(repo->pop_data_batch_by_id(id, partition));
    }
    {
      std::lock_guard<std::mutex> row_range_lg(_row_range_mutex);
      for (auto id : consumed_ids) {
        _row_range_columns.erase(id);
      }
    }
    // A run still at the batch it was planned to read through a window buffer has that buffer
    // in the merger.
    kept->holds_buffer.assign(cursors.size(), false);
    for (std::size_t i = 0; i < cursors.size(); ++i) {
      auto const it = std::find(state.task_runs.begin(), state.task_runs.end(), kept->run_ids[i]);
      auto const at = static_cast<std::size_t>(it - state.task_runs.begin());
      kept->holds_buffer[i] = at < state.task_buffered.size() && state.task_buffered[at] &&
                              consumed_counts[i] == 0 && !cursors[i].exhausted();
    }
    // A task that shrank its windows after an OOM sets the sizes of the partition's later tasks:
    // they continue its merger, and do not run out of memory at the unshrunk sizes again.
    state.config.window_rows = std::min(state.config.window_rows, input.config().window_rows);
    state.config.output_target_bytes =
      std::min(state.config.output_target_bytes, input.config().output_target_bytes);
    for (std::size_t i = 0; i < cursors.size(); ++i) {
      auto const run =
        std::find_if(state.runs.begin(), state.runs.end(), [&](auto const& candidate) {
          return candidate.id == kept->run_ids[i];
        });
      // A run an earlier task read to the end has left the partition.
      if (run == state.runs.end()) { continue; }
      run->cursor = cursors[i];
      run->batch_bytes.erase(
        run->batch_bytes.begin(),
        run->batch_bytes.begin() + static_cast<std::ptrdiff_t>(consumed_counts[i]));
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
    // The merger stays with the partition while the next task reads the same runs: until the
    // partition is merged, or until the intermediate pass in progress ends. A finalized
    // operator keeps none: nothing would free its window buffers before the operator goes.
    if (!finalized.load() && (final_pass ? !state.runs.empty() : pass_in_progress)) {
      state.merger = std::move(kept);
    }
    count(_counters, [&](merge_sort_counters& counters) {
      ++counters.windowed_tasks;
      if (!final_pass) { ++counters.intermediate_tasks; }
      merge::accumulate(counters.rounds, stats);
    });
    state.task_runs.clear();
    state.task_buffered.clear();
    state.in_flight = false;
    // The partition's next task, if it has one, is made only when the operator is scheduled.
    followup = finish_commit(*repo, true);
  }
  // Outside the mutex: frees the windows of a merger the partition no longer needs.
  kept.reset();
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
  after_commit(followup);

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
