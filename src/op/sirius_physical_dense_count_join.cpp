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

#include "op/sirius_physical_dense_count_join.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/aggregate/dense_count_join_impl.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {
[[nodiscard]] cudf::size_type checked_cudf_size(std::size_t value, std::string_view what)
{
  auto const max = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  if (value > max) {
    throw sirius::invalid_input_exception(
      "dense_count_join: {} {} exceeds cudf::size_type max {}", what, value, max);
  }
  return static_cast<cudf::size_type>(value);
}

[[nodiscard]] cudf::column_view checked_column(cudf::table_view batch,
                                               std::size_t index,
                                               std::size_t batch_index,
                                               std::string_view role)
{
  auto const num_columns = batch.num_columns();
  if (num_columns < 0 || index >= static_cast<std::size_t>(num_columns)) {
    throw sirius::internal_exception(
      "dense_count_join: input batch {} has {} columns; {} column index {} is out of range",
      batch_index,
      num_columns,
      role,
      index);
  }
  return batch.column(static_cast<cudf::size_type>(index));
}

[[nodiscard]] int64_t checked_null_count(cudf::column_view const& column, std::size_t batch_index)
{
  auto const nulls = column.null_count();
  if (nulls < 0 || nulls > column.size()) {
    throw sirius::internal_exception(
      "dense_count_join: preserved batch {} has invalid null count {} for {} rows",
      batch_index,
      nulls,
      column.size());
  }
  return static_cast<int64_t>(nulls);
}

[[nodiscard]] int64_t checked_add_rows(int64_t total, int64_t rows, std::string_view side)
{
  if (rows < 0 || total > std::numeric_limits<int64_t>::max() - rows) {
    throw sirius::invalid_input_exception(
      "dense_count_join: {} row count exceeds BIGINT accounting capacity", side);
  }
  return total + rows;
}

[[nodiscard]] bool count_product_needs_validation(int64_t preserved_rows,
                                                  int64_t counted_rows,
                                                  bool count_star) noexcept
{
  auto const lhs = static_cast<uint64_t>(preserved_rows);
  auto const rhs =
    static_cast<uint64_t>(count_star ? std::max<int64_t>(counted_rows, 1) : counted_rows);
  auto const max = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  return rhs != 0 && lhs > max / rhs;
}

/// Per-batch eager-aggregation partial: distinct non-NULL keys with their per-key row count.
/// @p value_policy EXCLUDE counts only rows whose @p values entry is non-NULL (COUNT(col));
/// INCLUDE counts every row of the group (COUNT(*) / presence).
std::unique_ptr<cudf::table> sparse_partial_count(cudf::column_view const& keys,
                                                  cudf::column_view const& values,
                                                  cudf::null_policy value_policy,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  cudf::groupby::groupby gb(cudf::table_view({keys}), cudf::null_policy::EXCLUDE, cudf::sorted::NO);
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(value_policy));
  auto [group_keys, results] = gb.aggregate(requests, stream, mr);
  // cuDF groupby COUNT emits size_type (INT32); widen so the partial merge sums in INT64.
  auto count64 =
    cudf::cast(results[0].results[0]->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  auto key_cols = group_keys->release();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(key_cols[0]));
  columns.push_back(std::move(count64));
  return std::make_unique<cudf::table>(std::move(columns));
}

/// Merge two `(key, INT64 count)` partials and release both inputs after their use is enqueued.
std::unique_ptr<cudf::table> sparse_merge_pair(std::unique_ptr<cudf::table> lhs,
                                               std::unique_ptr<cudf::table> rhs,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::vector<cudf::table_view> views{lhs->view(), rhs->view()};
  auto combined = cudf::concatenate(views, stream, mr);
  lhs.reset();
  rhs.reset();

  auto merged = [&] {
    cudf::groupby::groupby gb(
      cudf::table_view({combined->view().column(0)}), cudf::null_policy::EXCLUDE, cudf::sorted::NO);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = combined->view().column(1);
    requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    auto [group_keys, results] = gb.aggregate(requests, stream, mr);
    auto key_cols              = group_keys->release();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(key_cols[0]));
    columns.push_back(std::move(results[0].results[0]));
    return std::make_unique<cudf::table>(std::move(columns));
  }();
  combined.reset();
  return merged;
}

/// Merge per-batch partials in balanced rounds so no all-input concatenation is resident.
std::unique_ptr<cudf::table> sparse_merge_partials(
  std::vector<std::unique_ptr<cudf::table>> partials,
  cudf::data_type key_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (partials.empty()) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      cudf::make_fixed_width_column(key_type, 0, cudf::mask_state::UNALLOCATED, stream, mr));
    columns.push_back(cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64}, 0, cudf::mask_state::UNALLOCATED, stream, mr));
    return std::make_unique<cudf::table>(std::move(columns));
  }
  while (partials.size() > 1) {
    std::vector<std::unique_ptr<cudf::table>> next;
    next.reserve(partials.size() / 2 + partials.size() % 2);
    for (std::size_t i = 0; i < partials.size(); i += 2) {
      if (i + 1 == partials.size()) {
        next.push_back(std::move(partials[i]));
      } else {
        auto lhs = std::move(partials[i]);
        auto rhs = std::move(partials[i + 1]);
        next.push_back(sparse_merge_pair(std::move(lhs), std::move(rhs), stream, mr));
      }
    }
    partials = std::move(next);
  }
  return std::move(partials.front());
}

cudf::column_view gather_map_view(rmm::device_uvector<cudf::size_type> const& indices)
{
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           checked_cudf_size(indices.size(), "sparse gather-map length"),
                           indices.data(),
                           nullptr,
                           0,
                           0,
                           {});
}

}  // namespace

sirius_physical_dense_count_join::sirius_physical_dense_count_join(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::size_t preserved_key_idx,
  std::size_t counted_key_idx,
  std::optional<std::size_t> counted_value_idx,
  uint64_t max_bins_bytes)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::DENSE_COUNT_JOIN, std::move(types), estimated_cardinality),
    _preserved_key_idx(preserved_key_idx),
    _counted_key_idx(counted_key_idx),
    _counted_value_idx(counted_value_idx),
    _max_bins_bytes(max_bins_bytes)
{
  if (this->types.size() != 2) {
    throw sirius::internal_exception(
      "dense_count_join: expected [key, count] output schema, got {} columns", this->types.size());
  }
}

std::string sirius_physical_dense_count_join::params_to_string() const
{
  return " (preserved_key=" + std::to_string(_preserved_key_idx) +
         ", counted_key=" + std::to_string(_counted_key_idx) +
         (_counted_value_idx ? ", count_col=" + std::to_string(*_counted_value_idx)
                             : std::string(", count_star")) +
         ", max_bins_bytes=" + std::to_string(_max_bins_bytes) + ")";
}

//===--------------------------------------------------------------------===//
// Pipeline construction
//===--------------------------------------------------------------------===//

void sirius_physical_dense_count_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Always a blocking boundary: the operator forms its own single-op pipeline; each child
  // subtree becomes the sink of its own pipeline feeding the "preserved"/"counted" ports
  // (wired by the pipeline converter's tree-parent lookup — see resolve_port_id).
  if (children.size() != 2) {
    throw sirius::internal_exception("dense_count_join: expected 2 children, got {}",
                                     children.size());
  }
  auto& sink_meta    = meta_pipeline.create_child_meta_pipeline(current, *this);
  auto& host_current = *sink_meta.get_base_pipeline();

  auto build_child_side = [&](sirius_physical_operator& child) {
    auto& child_meta = sink_meta.create_child_meta_pipeline(host_current, child);
    if (child.children.empty()) { return; }  // leaf source (e.g. GPU_SCAN) — single-op pipeline
    if (child.children.size() != 1) {
      // The planner gate restricts child roots to unary/leaf shapes; anything else is a bug.
      throw sirius::internal_exception(
        "dense_count_join: child subtree root must be unary or a leaf");
    }
    child_meta.build(*child.children[0]);
  };
  // Build the typically larger counted side first, mirroring join build-side-first order.
  build_child_side(*children[1]);
  build_child_side(*children[0]);
}

std::optional<task_creation_hint> sirius_physical_dense_count_join::get_next_task_hint()
{
  if (ports.empty()) { return std::nullopt; }

  // FULL-barrier semantics on every port: wait until each producing pipeline has finished.
  for (auto const& p : _ports_list) {
    if (p->src_pipeline && !p->src_pipeline->is_pipeline_finished()) {
      auto* producer = &(p->src_pipeline->get_operators()[0].get());
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
    }
  }
  // Both producers finished: one task drains everything. Unlike the base hint, READY does not
  // require every port to be non-empty — an empty counted side must still produce the
  // all-zero-count groups, and an empty preserved side must still drain the counted batches.
  if (!all_ports_empty()) { return task_creation_hint{TaskCreationHint::READY, this}; }
  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_dense_count_join::get_next_task_input_data()
{
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  std::size_t num_preserved = 0;

  auto* preserved_port = get_port(PRESERVED_PORT);
  if (preserved_port->repo != nullptr) {
    while (auto batch = preserved_port->repo->pop_next_data_batch()) {
      batches.push_back(std::move(batch));
      ++num_preserved;
    }
  }
  auto* counted_port = get_port(COUNTED_PORT);
  if (counted_port->repo != nullptr) {
    while (auto batch = counted_port->repo->pop_next_data_batch()) {
      batches.push_back(std::move(batch));
    }
  }
  if (batches.empty()) { return nullptr; }
  return std::make_unique<dense_count_join_input>(std::move(batches), num_preserved);
}

std::size_t sirius_physical_dense_count_join::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  using sirius::memory::saturating_add;
  using sirius::memory::saturating_mul;

  constexpr std::size_t allocation_floor = 1024 * 1024;
  auto const preserved_rows =
    children.size() == 2 ? children[0]->estimated_cardinality : estimated_cardinality;
  auto const counted_rows = children.size() == 2 ? children[1]->estimated_cardinality : 0;
  auto const groups =
    estimated_cardinality == 0 ? preserved_rows : std::min(preserved_rows, estimated_cardinality);
  auto const range = preserved_rows == 0 ? std::size_t{0} : std::max(groups, std::size_t{1});
  auto const wide  = preserved_rows >= std::numeric_limits<uint32_t>::max() ||
                    counted_rows >= std::numeric_limits<uint32_t>::max();
  auto const slot_bytes      = wide ? sizeof(uint64_t) : sizeof(uint32_t);
  auto const histogram_bytes = saturating_mul(saturating_mul(2, slot_bytes), range);
  auto const histogram_cap   = _max_bins_bytes > std::numeric_limits<std::size_t>::max()
                                 ? std::numeric_limits<std::size_t>::max()
                                 : static_cast<std::size_t>(_max_bins_bytes);

  auto const total_rows   = saturating_add(preserved_rows, counted_rows);
  bool const likely_dense = preserved_rows > 0 && histogram_bytes <= histogram_cap &&
                            range <= saturating_mul(8, preserved_rows) &&
                            range <= saturating_mul(2, total_rows) &&
                            histogram_bytes <= saturating_mul(4, stats.bytes);

  auto const cudf_row_limit = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  auto const bounded_groups = std::min(groups, cudf_row_limit);
  auto const output_rows    = std::min(saturating_add(bounded_groups, 1), cudf_row_limit);
  auto const key_width      = sirius::get_cudf_type(types[0]).id() == cudf::type_id::INT32
                                ? sizeof(int32_t)
                                : sizeof(int64_t);
  auto const selected_bytes = saturating_mul(sizeof(int64_t), bounded_groups);
  auto const output_bytes = saturating_mul(saturating_add(key_width, sizeof(int64_t)), output_rows);
  auto const mask_bytes   = static_cast<std::size_t>(
    cudf::bitmask_allocation_size_bytes(checked_cudf_size(output_rows, "output row bound")));

  auto dense_peak = saturating_add(allocation_floor, histogram_bytes);
  dense_peak      = saturating_add(dense_peak, selected_bytes);
  dense_peak      = saturating_add(dense_peak, output_bytes);
  dense_peak      = saturating_add(dense_peak, mask_bytes);
  dense_peak      = saturating_add(dense_peak, histogram_bytes);  // selection/CUB workspace

  auto const extrema_per_batch = saturating_mul(6, key_width);
  auto const minmax_peak =
    saturating_add(allocation_floor, saturating_mul(stats.num_batches, extrema_per_batch));
  auto const sparse_peak = saturating_add(allocation_floor, saturating_mul(16, stats.bytes));
  return std::max(likely_dense ? dense_peak : sparse_peak, minmax_peak);
}

//===--------------------------------------------------------------------===//
// Execution
//===--------------------------------------------------------------------===//

std::unique_ptr<operator_data> sirius_physical_dense_count_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_dense_count_join::execute"};
  auto const* input = dynamic_cast<const dense_count_join_input*>(&input_data);
  if (input == nullptr) {
    throw sirius::internal_exception("dense_count_join: unexpected input data type");
  }
  auto const ro_batches    = input->get_read_only_batches();
  auto const num_preserved = input->num_preserved_batches();
  if (num_preserved > ro_batches.size()) {
    throw sirius::internal_exception(
      "dense_count_join: input marks {} preserved batches but contains only {} total batches",
      num_preserved,
      ro_batches.size());
  }

  // INVARIANT (SCHED-RR contract): all input batches arrive on the task's reservation device
  // via gpu_pipeline_task::execute -> pipelineable_operator_data::prepare_for_processing ->
  // lock_or_prepare_batch. See docs/super-sirius/pipeline-execution.md
  // "Per-task-device contract under SCHED-RR".
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : ro_batches) {
    if (batch.get_memory_space() != nullptr) {
      space = batch.get_memory_space();
      break;
    }
  }
  if (space == nullptr) {
    throw sirius::internal_exception("dense_count_join: no memory space on input batches");
  }
  auto mr = space->get_default_allocator();

  auto const key_type   = sirius::get_cudf_type(types[0]);
  auto require_key_type = [&](cudf::column_view const& col, char const* side) {
    if (col.type().id() != key_type.id()) {
      throw sirius::internal_exception(
        "dense_count_join: {} key column carrier {} does not match declared key type {}",
        side,
        static_cast<int32_t>(col.type().id()),
        static_cast<int32_t>(key_type.id()));
    }
  };

  // Collect the key (and count-argument) column views per side.
  std::vector<cudf::column_view> preserved_keys;
  std::vector<cudf::column_view> counted_keys;
  std::vector<std::optional<cudf::column_view>> counted_values;
  int64_t preserved_rows          = 0;
  int64_t preserved_null_keys     = 0;
  int64_t counted_rows            = 0;
  std::size_t input_logical_bytes = 0;
  for (std::size_t i = 0; i < ro_batches.size(); ++i) {
    auto const* representation = ro_batches[i].get_data();
    if (representation == nullptr) {
      throw sirius::internal_exception("dense_count_join: input batch {} has no representation", i);
    }
    input_logical_bytes = sirius::memory::saturating_add(
      input_logical_bytes, representation->get_uncompressed_data_size_in_bytes());
    auto const batch_view = sirius::get_cudf_table_view(ro_batches[i]);
    if (i < num_preserved) {
      auto const col = checked_column(batch_view, _preserved_key_idx, i, "preserved key");
      require_key_type(col, "preserved");
      preserved_rows =
        checked_add_rows(preserved_rows, static_cast<int64_t>(col.size()), "preserved");
      preserved_null_keys =
        checked_add_rows(preserved_null_keys, checked_null_count(col, i), "preserved NULL-key");
      preserved_keys.push_back(col);
    } else {
      auto const col = checked_column(batch_view, _counted_key_idx, i, "counted key");
      require_key_type(col, "counted");
      counted_rows = checked_add_rows(counted_rows, static_cast<int64_t>(col.size()), "counted");
      counted_keys.push_back(col);
      if (_counted_value_idx) {
        counted_values.emplace_back(
          checked_column(batch_view, *_counted_value_idx, i, "COUNT argument"));
      } else {
        counted_values.emplace_back(std::nullopt);
      }
    }
  }

  bool const count_star = !_counted_value_idx.has_value();
  bool const check_product_overflow =
    count_product_needs_validation(preserved_rows, counted_rows, count_star);
  int64_t const non_null_keys = preserved_rows - preserved_null_keys;

  std::unique_ptr<cudf::table> output;
  if (non_null_keys == 0) {
    // No non-NULL preserved keys: the output is the NULL group alone (or empty).
    _last_strategy = strategy::DENSE;
    output = dense_count_empty_output(key_type, count_star, preserved_null_keys, stream, mr);
  } else {
    // Batch reductions and merging remain on-device until one final extrema readback.
    auto const min_max = dense_count_global_minmax(preserved_keys, stream, mr);
    if (!min_max) {
      throw sirius::internal_exception(
        "dense_count_join: minmax reported no valid keys but null accounting found {}",
        non_null_keys);
    }

    // Unsigned difference is exact for any int64 pair; a wrapped (zero) range means the domain
    // spans the full 64-bit space and can never take the dense path.
    uint64_t const range_u =
      static_cast<uint64_t>(min_max->second) - static_cast<uint64_t>(min_max->first) + 1;
    bool const wide = preserved_rows >= std::numeric_limits<uint32_t>::max() ||
                      counted_rows >= std::numeric_limits<uint32_t>::max();
    uint64_t const slot_bytes          = wide ? sizeof(uint64_t) : sizeof(uint32_t);
    uint64_t const combined_slot_bytes = 2 * slot_bytes;
    auto const size_max                = std::numeric_limits<std::size_t>::max();
    bool const layout_valid = range_u != 0 && range_u <= size_max / combined_slot_bytes &&
                              range_u <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    auto const histogram_bytes =
      layout_valid ? static_cast<std::size_t>(range_u * combined_slot_bytes) : size_max;
    auto const non_null_rows = static_cast<std::size_t>(non_null_keys);
    auto const total_rows = sirius::memory::saturating_add(static_cast<std::size_t>(preserved_rows),
                                                           static_cast<std::size_t>(counted_rows));
    bool const dense_ok   = layout_valid && range_u <= _max_bins_bytes / combined_slot_bytes &&
                          range_u <= sirius::memory::saturating_mul(8, non_null_rows) &&
                          range_u <= sirius::memory::saturating_mul(2, total_rows) &&
                          histogram_bytes <= sirius::memory::saturating_mul(4, input_logical_bytes);

    if (dense_ok) {
      _last_strategy   = strategy::DENSE;
      auto const range = static_cast<int64_t>(range_u);
      SIRIUS_LOG_INFO(
        "[dense_count_join] dense path: keys in [{}, {}] (range {}, {}-bit slots), preserved "
        "rows {} (null keys {}), counted rows {}",
        min_max->first,
        min_max->second,
        range,
        wide ? 64 : 32,
        preserved_rows,
        preserved_null_keys,
        counted_rows);
      dense_count_state state(min_max->first, range, wide, stream, mr);
      for (auto const& col : preserved_keys) {
        state.accumulate_preserved(col, stream);
      }
      for (std::size_t i = 0; i < counted_keys.size(); ++i) {
        state.accumulate_counted(
          counted_keys[i], counted_values[i] ? &*counted_values[i] : nullptr, stream);
      }
      output =
        state.emit(key_type, count_star, preserved_null_keys, stream, mr, check_product_overflow);
    } else {
      _last_strategy = strategy::SPARSE;
      SIRIUS_LOG_INFO(
        "[dense_count_join] sparse path: keys in [{}, {}], range {}, histogram bytes {}, "
        "input bytes {}, budget {}",
        min_max->first,
        min_max->second,
        range_u,
        histogram_bytes,
        input_logical_bytes,
        _max_bins_bytes);

      // Counted side: per-batch groupby-count partials, then a groupby-sum merge.
      std::vector<std::unique_ptr<cudf::table>> counted_partials;
      for (std::size_t i = 0; i < counted_keys.size(); ++i) {
        if (counted_keys[i].size() == 0) { continue; }
        auto const& values = counted_values[i] ? *counted_values[i] : counted_keys[i];
        auto const policy =
          counted_values[i] ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
        counted_partials.push_back(
          sparse_partial_count(counted_keys[i], values, policy, stream, mr));
      }
      auto counted_agg = sparse_merge_partials(std::move(counted_partials), key_type, stream, mr);

      // Preserved side: distinct keys with their multiplicity (duplicate preserved keys
      // multiply the per-key match count, matching join-then-group-by semantics).
      std::vector<std::unique_ptr<cudf::table>> preserved_partials;
      for (auto const& col : preserved_keys) {
        if (col.size() == 0) { continue; }
        preserved_partials.push_back(
          sparse_partial_count(col, col, cudf::null_policy::INCLUDE, stream, mr));
      }
      auto preserved_agg =
        sparse_merge_partials(std::move(preserved_partials), key_type, stream, mr);

      // preserved LEFT JOIN counted on the distinct keys; unmatched -> 0 matches.
      auto const preserved_key_view      = cudf::table_view({preserved_agg->view().column(0)});
      auto const counted_key_view        = cudf::table_view({counted_agg->view().column(0)});
      auto [left_indices, right_indices] = cudf::left_join(
        preserved_key_view, counted_key_view, cudf::null_equality::UNEQUAL, stream, mr);

      auto keys_out = cudf::gather(preserved_key_view,
                                   gather_map_view(*left_indices),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   mr);
      auto presence = cudf::gather(cudf::table_view({preserved_agg->view().column(1)}),
                                   gather_map_view(*left_indices),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   mr);
      auto matched  = cudf::gather(cudf::table_view({counted_agg->view().column(1)}),
                                  gather_map_view(*right_indices),
                                  cudf::out_of_bounds_policy::NULLIFY,
                                  stream,
                                  mr);
      left_indices.reset();
      right_indices.reset();
      preserved_agg.reset();
      counted_agg.reset();

      cudf::numeric_scalar<int64_t> zero(0, true, stream, mr);
      auto matched_filled = cudf::replace_nulls(matched->view().column(0), zero, stream, mr);
      matched.reset();
      if (count_star) {
        // COUNT(*): unmatched preserved rows survive the outer join as one row each.
        cudf::numeric_scalar<int64_t> one(1, true, stream, mr);
        matched_filled = cudf::binary_operation(matched_filled->view(),
                                                one,
                                                cudf::binary_operator::NULL_MAX,
                                                cudf::data_type{cudf::type_id::INT64},
                                                stream,
                                                mr);
      }
      if (check_product_overflow) {
        throw_if_count_product_overflows(
          presence->view().column(0), matched_filled->view(), stream, mr);
      }
      auto values = cudf::binary_operation(presence->view().column(0),
                                           matched_filled->view(),
                                           cudf::binary_operator::MUL,
                                           cudf::data_type{cudf::type_id::INT64},
                                           stream,
                                           mr);
      presence.reset();
      matched_filled.reset();

      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(std::move(keys_out->release()[0]));
      columns.push_back(std::move(values));
      output = std::make_unique<cudf::table>(std::move(columns));

      if (preserved_null_keys > 0) {
        if (output->num_rows() == std::numeric_limits<cudf::size_type>::max()) {
          throw sirius::invalid_input_exception(
            "dense_count_join: adding the NULL group would exceed cudf::size_type max {}",
            std::numeric_limits<cudf::size_type>::max());
        }
        auto null_group =
          dense_count_empty_output(key_type, count_star, preserved_null_keys, stream, mr);
        std::vector<cudf::table_view> parts{output->view(), null_group->view()};
        output = cudf::concatenate(parts, stream, mr);
      }
    }
  }

  SIRIUS_LOG_INFO("[dense_count_join] emitted {} group rows ({} strategy)",
                  output->num_rows(),
                  _last_strategy == strategy::DENSE ? "dense" : "sparse");

  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  results.push_back(sirius::make_data_batch(std::move(output), *space, stream, batch_telemetry()));
  return std::make_unique<pipelineable_operator_data>(std::move(results));
}

}  // namespace sirius::op
