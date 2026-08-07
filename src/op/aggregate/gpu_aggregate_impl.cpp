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

#include "op/aggregate/gpu_aggregate_impl.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/aggregate/aggregate_op_util.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <new>

#include <algorithm>

namespace sirius {
namespace op {

template <typename Base = cudf::aggregation>
std::unique_ptr<Base> get_local_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::MIN: return cudf::make_min_aggregation<Base>();
    case cudf::aggregation::Kind::MAX: return cudf::make_max_aggregation<Base>();
    case cudf::aggregation::Kind::COUNT_ALL:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::INCLUDE);
    case cudf::aggregation::Kind::COUNT_VALID:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::Kind::SUM: return cudf::make_sum_aggregation<Base>();
    default:
      throw std::runtime_error("Unsupported cudf aggregate kind in `get_local_aggregation()`: " +
                               std::to_string(static_cast<int>(kind)));
  }
}

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_ungrouped_aggregate(
  const cucascade::read_only_data_batch& input,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_ungrouped_aggregate()`");
  }
  std::vector<std::unique_ptr<cudf::column>> output_cols;
  auto input_table = get_cudf_table_view(input);
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& input_col       = input_table.column(aggregate_idx[i]);
    auto reduce_aggregation     = get_local_aggregation<cudf::reduce_aggregation>(aggregates[i]);
    cudf::data_type output_type = input_col.type();
    switch (aggregates[i]) {
      case cudf::aggregation::Kind::SUM: {
        switch (output_type.id()) {
          case cudf::type_id::INT8:
          case cudf::type_id::INT16:
          case cudf::type_id::INT32: {
            output_type = cudf::data_type(cudf::type_id::INT64);
            break;
          }
          case cudf::type_id::UINT8:
          case cudf::type_id::UINT16:
          case cudf::type_id::UINT32: {
            output_type = cudf::data_type(cudf::type_id::UINT64);
            break;
          }
          case cudf::type_id::DECIMAL64:
            if (input_col.type().id() == cudf::type_id::DECIMAL64) {
              output_type = cudf::data_type(cudf::type_id::DECIMAL128, output_type.scale());
            }
            break;
          case cudf::type_id::DECIMAL32:
            if (input_col.type().id() == cudf::type_id::DECIMAL32) {
              output_type = cudf::data_type(cudf::type_id::DECIMAL64, output_type.scale());
            }
            break;
          default: break;
        }
        break;
      }
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        output_type = cudf::data_type(cudf::type_id::INT64);
        break;
      }
      default: break;
    }
    auto output_scalar = cudf::reduce(
      input_col, *reduce_aggregation, output_type, stream, memory_space.get_default_allocator());
    output_cols.push_back(cudf::make_column_from_scalar(
      *output_scalar, 1, stream, memory_space.get_default_allocator()));
  }
  auto output_table = std::make_unique<cudf::table>(std::move(output_cols));

  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_grouped_aggregate(
  const cucascade::read_only_data_batch& input,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  const std::vector<std::vector<int>>& aggregate_struct_col_indices,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Sanity check
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_grouped_aggregate()`");
  }

  const bool has_struct_col_indices = !aggregate_struct_col_indices.empty();

  auto input_table = get_cudf_table_view(input);
  auto mr          = memory_space.get_default_allocator();

  // Bit-stable float sums: cuDF's hash groupby accumulates SUM via atomicAdd, whose combine
  // order varies run to run, and FP addition is not associative — the same rows can yield
  // per-group sums differing by ULPs across evaluations. Distributed plans compare such sums
  // for exact equality (TPC-H q15 probes `sum = max(sum)` across two evaluations of the same
  // CTE), so gather rows into a canonical (group keys, float values) order and declare the
  // keys presorted, which routes cuDF onto its sort-based, atomics-free aggregation. The
  // partial then depends only on the batch's row multiset.
  std::vector<cudf::size_type> float_sum_value_cols;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    int col_idx = aggregate_idx[i];
    // COLLECT_SET slots may carry the -1 struct-column sentinel; they are never SUM.
    if (col_idx < 0) { continue; }
    if (is_order_sensitive_sum(aggregates[i], input_table.column(col_idx).type()) &&
        std::find(float_sum_value_cols.begin(), float_sum_value_cols.end(), col_idx) ==
          float_sum_value_cols.end()) {
      float_sum_value_cols.push_back(col_idx);
    }
  }
  const bool use_canonical_sorted_groupby = !float_sum_value_cols.empty();
  std::unique_ptr<cudf::table> canonical_input;
  if (use_canonical_sorted_groupby) {
    std::vector<cudf::size_type> sort_cols(group_idx.begin(), group_idx.end());
    sort_cols.insert(sort_cols.end(), float_sum_value_cols.begin(), float_sum_value_cols.end());
    canonical_input = canonicalize_row_order(input_table, sort_cols, stream, mr);
    input_table     = canonical_input->view();
  }
  // ---------------------------------------------------------------------------------------
  // Single-label group keys for the sorted-groupby path.
  //
  // COLLECT_SET (our COUNT(DISTINCT ...) lowering) is not a hash aggregation in cudf, so
  // `cudf::groupby::aggregate` routes the whole request through the *sorted* groupby helper.
  // That helper's first step is `key_sort_order()`, which calls
  // `cudf::detail::stable_sorted_order(keys)` over the full input. For a multi-column key
  // table that is a `cub::DeviceMergeSort` driven by a lexicographic row comparator:
  // O(N log N) passes of random access over every key column.
  //
  // `cudf::detail::stable_sorted_order` has a fast path though: a *single*, non-nested,
  // fixed-width, null-free column is dispatched to `sorted_order_radix`
  // (`cub::DeviceRadixSort`), which is a handful of fully coalesced passes.
  //
  // So we collapse the whole key table into one dense INT32 label with `cudf::encode` and
  // group by that instead. `cudf::encode` returns the distinct key rows in sorted order plus
  // the per-row index into them, so the label ordering is exactly the lexicographic ordering
  // of the original key tuples -- group identity and group ordering are both preserved. The
  // original key columns are recovered after the aggregate with a gather at group cardinality.
  //
  // NULL semantics are preserved: `cudf::encode` builds the distinct set with
  // `null_equality::EQUAL` / `nan_equality::ALL_EQUAL` and searches it with
  // `null_order::AFTER`, which is precisely what the sorted groupby helper does for
  // `null_policy::INCLUDE`. A key tuple containing NULL therefore gets its own label and
  // becomes its own group, exactly as before.
  //
  // Only worth it when the sort actually dominates, so gate on a large input; and pointless
  // when the key table is already a single radix-sortable column.
  //
  // The gate on group cardinality matters most: `cudf::encode` is distinct + a sort of the
  // *distinct* rows + a binary search per input row. That is a large win when the groups are
  // few, but for a high-cardinality key (say COUNT(DISTINCT ...) grouped by an order key) the
  // distinct-key sort is as big as the sort we are trying to avoid, and we would come out
  // behind. An HLL estimate over the key rows is ~O(1) memory and one cheap pass, so use it to
  // decline the rewrite rather than guessing.
  constexpr cudf::size_type label_encode_min_rows = 1 << 20;
  constexpr double label_encode_max_group_ratio   = 0.01;

  auto const key_is_radix_sortable = [](cudf::column_view const& col) {
    return !col.has_nulls() && cudf::is_fixed_width(col.type());
  };

  bool const has_collect_set =
    std::any_of(aggregates.begin(), aggregates.end(), [](cudf::aggregation::Kind k) {
      return k == cudf::aggregation::Kind::COLLECT_SET;
    });
  bool const keys_already_radix =
    group_idx.size() == 1 && key_is_radix_sortable(input_table.column(group_idx[0]));
  bool const any_nested_key = std::any_of(group_idx.begin(), group_idx.end(), [&](int idx) {
    return cudf::is_nested(input_table.column(idx).type());
  });

  std::unique_ptr<cudf::table> label_key_values;  // distinct key rows, in sorted order
  std::unique_ptr<cudf::column> label_col;        // per-row index into `label_key_values`

  if (has_collect_set && !group_idx.empty() && !keys_already_radix && !any_nested_key &&
      input_table.num_rows() >= label_encode_min_rows) {
    try {
      std::vector<cudf::column_view> raw_key_cols;
      raw_key_cols.reserve(group_idx.size());
      for (int idx : group_idx) {
        raw_key_cols.push_back(input_table.column(idx));
      }
      auto const keys_view = cudf::table_view(raw_key_cols);
      cudf::approx_distinct_count adc(
        keys_view, 12, cudf::null_policy::INCLUDE, cudf::nan_policy::NAN_IS_VALID, stream);
      auto const ndv     = adc.estimate(stream);
      double const ratio = static_cast<double>(ndv) / input_table.num_rows();
      if (ratio >= label_encode_max_group_ratio) {
        SIRIUS_LOG_DEBUG(
          "local_grouped_agg: skipping group-key label encoding, group "
          "cardinality too high (ndv={}, rows={}, ratio={:.4f})",
          ndv,
          input_table.num_rows(),
          ratio);
      } else {
        auto encoded     = cudf::encode(keys_view, stream, mr);
        label_key_values = std::move(encoded.first);
        label_col        = std::move(encoded.second);
        SIRIUS_LOG_DEBUG(
          "local_grouped_agg: label-encoded {} group key column(s) into a single radix-sortable "
          "key for the COLLECT_SET sort path (rows={}, ndv={}, groups={})",
          group_idx.size(),
          input_table.num_rows(),
          ndv,
          label_key_values->num_rows());
      }
    } catch (const std::bad_alloc&) {
      // Out-of-memory (incl. rmm::out_of_memory) must reach the task retry /
      // downgrade machinery.
      throw;
    } catch (const cudf::cuda_error&) {
      // CUDA errors are sticky; falling back would compute on a broken context.
      throw;
    } catch (const std::exception& e) {
      // Non-fatal (e.g. an unsupported key shape): fall back to grouping on
      // the original key columns.
      label_key_values.reset();
      label_col.reset();
      SIRIUS_LOG_DEBUG(
        "local_grouped_agg: group-key label encoding failed ({}), "
        "falling back to multi-column keys",
        e.what());
    }
  }
  bool const use_label_keys = label_col != nullptr;

  // Dictionary-encode STRING group keys when:
  //  1. Average string length >= 4 bytes (short strings hash nearly as fast as
  //     int32, so the encode/decode overhead is not worthwhile), AND
  //  2. NDV / row_count < 10% (high-cardinality columns produce huge
  //     dictionaries that negate the hashing benefit).
  // The avg_len check is O(1) (single offset read) and gates the more
  // expensive HLL-based NDV estimate.
  constexpr double dict_encode_min_avg_len = 8.0;
  constexpr double dict_encode_max_ratio   = 0.10;
  std::vector<std::unique_ptr<cudf::column>> encoded_key_owners;
  std::vector<cudf::column_view> group_cols;
  group_cols.reserve(group_idx.size());
  for (int idx : group_idx) {
    if (use_label_keys) { break; }
    auto col = input_table.column(idx);
    // Dict-encoding is a hash-groupby optimization; the canonical path must hand the
    // presorted raw keys to the sort-based groupby unchanged.
    if (!use_canonical_sorted_groupby && col.type().id() == cudf::type_id::STRING &&
        col.size() > 0) {
      cudf::strings_column_view scv(col);
      auto avg_len = static_cast<double>(scv.chars_size(stream)) / col.size();
      if (avg_len >= dict_encode_min_avg_len) {
        cudf::approx_distinct_count adc(cudf::table_view({col}),
                                        12,
                                        cudf::null_policy::EXCLUDE,
                                        cudf::nan_policy::NAN_IS_VALID,
                                        stream);
        auto ndv     = adc.estimate(stream);
        double ratio = static_cast<double>(ndv) / col.size();
        if (ratio < dict_encode_max_ratio) {
          auto encoded =
            cudf::dictionary::encode(col, cudf::data_type{cudf::type_id::INT32}, stream, mr);
          group_cols.push_back(encoded->view());
          encoded_key_owners.push_back(std::move(encoded));
          SIRIUS_LOG_DEBUG(
            "local_grouped_agg: dict-encoding key col {} (avg_len={:.1f}, ndv={}, rows={})",
            idx,
            avg_len,
            ndv,
            col.size());
        } else {
          group_cols.push_back(col);
          SIRIUS_LOG_DEBUG(
            "local_grouped_agg: skipping dict-encode for key col {} "
            "(avg_len={:.1f}, ndv={}, rows={}, ratio={:.4f})",
            idx,
            avg_len,
            ndv,
            col.size(),
            ratio);
        }
      } else {
        group_cols.push_back(col);
        SIRIUS_LOG_DEBUG(
          "local_grouped_agg: skipping dict-encode for key col {} (avg_len={:.1f} < 4.0)",
          idx,
          avg_len);
      }
    } else {
      group_cols.push_back(col);
    }
  }
  if (use_label_keys) { group_cols.push_back(label_col->view()); }
  // Presorted keys force cuDF's deterministic sort-based aggregation (hash groupby would
  // reintroduce atomicAdd). The declared order must match canonicalize_row_order's.
  cudf::groupby::groupby grpby_obj(
    cudf::table_view(group_cols),
    cudf::null_policy::INCLUDE,
    use_canonical_sorted_groupby ? cudf::sorted::YES : cudf::sorted::NO,
    std::vector<cudf::order>(use_canonical_sorted_groupby ? group_cols.size() : 0,
                             cudf::order::ASCENDING),
    std::vector<cudf::null_order>(use_canonical_sorted_groupby ? group_cols.size() : 0,
                                  cudf::null_order::AFTER));

  // Make aggregation requests, group aggregations on the same column in the single request.
  // For multi-column COLLECT_SET, a synthetic negative key -(i+1) is used so that each such
  // aggregate gets its own request with a freshly synthesized struct column.
  std::unordered_map<int, std::vector<std::unique_ptr<cudf::groupby_aggregation>>> input_col_to_agg;
  std::unordered_map<int, std::vector<size_t>> input_col_to_output_idx;
  std::vector<int> input_col_order;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& aggregate_kind = aggregates[i];
    if (aggregate_kind == cudf::aggregation::Kind::SUM) {
      // The HUGEINT->BIGINT downcast guard: refuse a 64-bit integer sum that could wrap.
      throw_if_int64_sum_could_overflow(input_table.column(aggregate_idx[i]), stream, mr);
    }
    int aggregate_col_id;
    if (has_struct_col_indices && !aggregate_struct_col_indices[i].empty()) {
      // Multi-column COLLECT_SET: use a unique synthetic negative key for this slot.
      aggregate_col_id = -(static_cast<int>(i) + 1);
    } else {
      aggregate_col_id = aggregate_idx[i];
    }
    if (!input_col_to_agg.contains(aggregate_col_id)) {
      input_col_order.push_back(aggregate_col_id);
    }
    std::unique_ptr<cudf::groupby_aggregation> groupby_aggregation;
    if (aggregate_kind == cudf::aggregation::Kind::COLLECT_SET) {
      groupby_aggregation =
        cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
    } else {
      groupby_aggregation = get_local_aggregation<cudf::groupby_aggregation>(aggregate_kind);
    }
    input_col_to_agg[aggregate_col_id].push_back(std::move(groupby_aggregation));
    input_col_to_output_idx[aggregate_col_id].push_back(i);
  }

  // Temp struct columns for multi-col COLLECT_SET; must outlive the groupby call.
  std::vector<std::unique_ptr<cudf::column>> temp_struct_cols;

  std::vector<cudf::groupby::aggregation_request> requests;
  for (int aggregate_col_id : input_col_order) {
    cudf::groupby::aggregation_request request;
    if (aggregate_col_id < 0) {
      // Multi-col COLLECT_SET: synthesize a struct column from the component columns.
      // The synthetic key is -(slot_index + 1), so slot_index = -aggregate_col_id - 1.
      size_t slot_idx            = static_cast<size_t>(-aggregate_col_id - 1);
      const auto& struct_indices = aggregate_struct_col_indices[slot_idx];
      std::vector<std::unique_ptr<cudf::column>> struct_children;
      for (int col_idx : struct_indices) {
        struct_children.push_back(std::make_unique<cudf::column>(
          input_table.column(col_idx), stream, memory_space.get_default_allocator()));
      }
      auto struct_col = cudf::make_structs_column(input_table.num_rows(),
                                                  std::move(struct_children),
                                                  0,
                                                  rmm::device_buffer{},
                                                  stream,
                                                  memory_space.get_default_allocator());
      request.values  = struct_col->view();
      temp_struct_cols.push_back(std::move(struct_col));
    } else {
      request.values = input_table.column(aggregate_col_id);
    }
    request.aggregations = std::move(input_col_to_agg[aggregate_col_id]);
    requests.push_back(std::move(request));
  }

  // Call cudf groupby and populate output columns
  auto groupby_result = grpby_obj.aggregate(requests, stream, mr);
  auto output_cols    = groupby_result.first->release();

  // Expand the single label key back into the original group key columns. The groupby emits
  // one row per distinct label, so this gather runs at group cardinality, not input rows.
  if (use_label_keys) {
    auto key_table = cudf::gather(label_key_values->view(),
                                  output_cols[0]->view(),
                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                  stream,
                                  mr);
    output_cols    = key_table->release();
  }

  // Decode dictionary-encoded group key columns back to STRING
  for (size_t i = 0; i < group_idx.size(); i++) {
    if (output_cols[i]->type().id() == cudf::type_id::DICTIONARY32) {
      cudf::dictionary_column_view dict_view(output_cols[i]->view());
      output_cols[i] = cudf::dictionary::decode(dict_view, stream, mr);
    }
  }

  output_cols.resize(group_idx.size() + aggregate_idx.size());
  for (size_t i = 0; i < input_col_order.size(); ++i) {
    int aggregate_col_id     = input_col_order[i];
    auto& aggregation_result = groupby_result.second[i];

    // need to cast count aggregation result to int64 (not applicable for COLLECT_SET)
    if (requests[i].aggregations.size() == 1 &&
        requests[i].aggregations[0]->kind != cudf::aggregation::Kind::COLLECT_SET &&
        (requests[i].aggregations[0]->kind == cudf::aggregation::Kind::COUNT_VALID ||
         requests[i].aggregations[0]->kind == cudf::aggregation::Kind::COUNT_ALL)) {
      if (aggregation_result.results.size() != 1) {
        throw std::runtime_error("Expected 1 result for count aggregation, got " +
                                 std::to_string(aggregation_result.results.size()));
      }
      auto result_view = aggregation_result.results[0]->view();
      if (result_view.type().id() != cudf::type_id::INT64) {
        aggregation_result.results[0] = cudf::cast(result_view,
                                                   cudf::data_type(cudf::type_id::INT64),
                                                   stream,
                                                   memory_space.get_default_allocator());
      }
    }

    const auto& output_idx = input_col_to_output_idx[aggregate_col_id];
    for (size_t j = 0; j < output_idx.size(); ++j) {
      auto result_view = aggregation_result.results[j]->view();
      // Widen decimal result for SUM (expected by duckdb)
      if (requests[i].aggregations[j]->kind == cudf::aggregation::Kind::SUM) {
        if (requests[i].values.type().id() == cudf::type_id::DECIMAL64) {
          aggregation_result.results[j] =
            cudf::cast(result_view,
                       cudf::data_type(cudf::type_id::DECIMAL128, result_view.type().scale()),
                       stream,
                       memory_space.get_default_allocator());
        } else if (requests[i].values.type().id() == cudf::type_id::DECIMAL32) {
          aggregation_result.results[j] =
            cudf::cast(result_view,
                       cudf::data_type(cudf::type_id::DECIMAL64, result_view.type().scale()),
                       stream,
                       memory_space.get_default_allocator());
        }
      }
      size_t output_col_id       = group_idx.size() + output_idx[j];
      output_cols[output_col_id] = std::move(aggregation_result.results[j]);
    }
  }

  // Create the output data batch
  auto output_table = std::make_unique<cudf::table>(std::move(output_cols));
  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

}  // namespace op
}  // namespace sirius
