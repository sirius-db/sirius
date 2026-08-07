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

#include "op/merge/gpu_merge_impl.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/aggregate/aggregate_op_util.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/merge.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace sirius {
namespace op {

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::concat(
  const std::vector<cucascade::read_only_data_batch>& input,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error("`input` in `concat()` should at least contain two data batches");
  }

  // Pull input cudf tables and merge.
  std::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.reserve(input.size());
  for (const auto& batch : input) {
    input_cudf_table_views.push_back(get_cudf_table_view(batch));
  }
  auto output_cudf_table =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Create output data batch.
  return make_data_batch(std::move(output_cudf_table), memory_space, stream, telemetry_info);
}

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::merge_ungrouped_aggregate(
  const std::vector<cucascade::read_only_data_batch>& input,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<std::optional<cudf::size_type>>& merge_nth_index,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_ungrouped_aggregate()` should at least contain two data batches");
  }
  if (merge_nth_index.size() != aggregates.size()) {
    throw std::runtime_error(
      "`merge_nth_index` must have the same size as `aggregates` in `merge_ungrouped_aggregate()`");
  }

  // Pull input cudf tables and concatenate.
  std::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.reserve(input.size());
  for (const auto& batch : input) {
    input_cudf_table_views.push_back(get_cudf_table_view(batch));
  }
  if (input_cudf_table_views[0].num_columns() != static_cast<cudf::size_type>(aggregates.size())) {
    throw std::runtime_error(
      "mismatch between num columns and num aggregates in `merge_ungrouped_aggregate()`");
  }
  auto concatenated =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Aggregate on the concatenated table
  std::vector<std::unique_ptr<cudf::column>> output_cudf_cols;
  for (size_t c = 0; c < aggregates.size(); ++c) {
    std::unique_ptr<cudf::reduce_aggregation> reduce_aggregation = nullptr;
    cudf::column_view reduce_input = concatenated->get_column(c).view();
    std::unique_ptr<cudf::table> sorted_owner;
    cudf::data_type output_type = reduce_input.type();
    switch (aggregates[c]) {
      case cudf::aggregation::Kind::MIN: {
        reduce_aggregation = cudf::make_min_aggregation<cudf::reduce_aggregation>();
        break;
      }
      case cudf::aggregation::Kind::MAX: {
        reduce_aggregation = cudf::make_max_aggregation<cudf::reduce_aggregation>();
        break;
      }
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
          default: break;
        }
        // The HUGEINT->BIGINT downcast guard: refuse a 64-bit integer sum that could wrap.
        throw_if_int64_sum_could_overflow(
          reduce_input, stream, memory_space.get_default_allocator());
        if (is_order_sensitive_sum(aggregates[c], reduce_input.type())) {
          // FP addition is not associative and the concatenation order is exchange arrival
          // order: sort the partials so the scalar sum is bit-stable across runs.
          sorted_owner = cudf::sort(cudf::table_view({reduce_input}),
                                    {cudf::order::ASCENDING},
                                    {cudf::null_order::AFTER},
                                    stream,
                                    memory_space.get_default_allocator());
          reduce_input = sorted_owner->get_column(0).view();
        }
        reduce_aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        break;
      }
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        output_type        = cudf::data_type(cudf::type_id::INT64);
        reduce_aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        break;
      }
      case cudf::aggregation::Kind::NTH_ELEMENT: {
        if (!merge_nth_index[c].has_value()) {
          throw std::runtime_error(
            "NTH_ELEMENT aggregate requires a value in `merge_nth_index` in "
            "`merge_ungrouped_aggregate()`");
        }
        reduce_aggregation = cudf::make_nth_element_aggregation<cudf::reduce_aggregation>(
          *merge_nth_index[c], cudf::null_policy::INCLUDE);
        break;
      }
      default:
        throw std::runtime_error(
          "Unsupported cudf aggregate kind in `merge_ungrouped_aggregate()`: " +
          std::to_string(static_cast<int>(aggregates[c])));
    }
    auto output_scalar = cudf::reduce(
      reduce_input, *reduce_aggregation, output_type, stream, memory_space.get_default_allocator());
    output_cudf_cols.push_back(cudf::make_column_from_scalar(
      *output_scalar, 1, stream, memory_space.get_default_allocator()));
  }
  auto output_cudf_table = std::make_unique<cudf::table>(std::move(output_cudf_cols));

  // Create output data batch.
  return make_data_batch(std::move(output_cudf_table), memory_space, stream, telemetry_info);
}

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::merge_grouped_aggregate(
  const std::vector<cucascade::read_only_data_batch>& input,
  int num_group_cols,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_grouped_aggregate()` should at least contain two data batches");
  }

  // Pull input cudf tables and concatenate.
  std::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.reserve(input.size());
  for (const auto& batch : input) {
    input_cudf_table_views.push_back(get_cudf_table_view(batch));
  }
  if (input_cudf_table_views[0].num_columns() !=
      num_group_cols + static_cast<int>(aggregates.size())) {
    throw std::runtime_error(
      "`num columns = num_group_cols + num aggregates` not true in `merge_grouped_aggregate()`");
  }
  auto concatenated =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  auto mr = memory_space.get_default_allocator();

  // Bit-stable float sums during merge: partial batches arrive in nondeterministic exchange
  // order, and cuDF's hash groupby combines them via atomicAdd. Gather the concatenated
  // partials into a canonical (group keys, float partial values) order and reduce through the
  // sort-based groupby, so the merged sum depends only on the multiset of partial values —
  // not on sender arrival order or atomic scheduling. See local_grouped_aggregate for the
  // full rationale (TPC-H q15 compares these sums for exact equality).
  std::vector<cudf::size_type> canonical_sort_cols;
  canonical_sort_cols.reserve(num_group_cols + aggregates.size());
  for (int c = 0; c < num_group_cols; ++c) {
    canonical_sort_cols.push_back(c);
  }
  bool use_canonical_sorted_groupby = false;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    auto col_idx = num_group_cols + static_cast<cudf::size_type>(i);
    if (is_order_sensitive_sum(aggregates[i], concatenated->get_column(col_idx).type())) {
      canonical_sort_cols.push_back(col_idx);
      use_canonical_sorted_groupby = true;
    }
  }
  if (use_canonical_sorted_groupby) {
    concatenated = canonicalize_row_order(concatenated->view(), canonical_sort_cols, stream, mr);
  }

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
  group_cols.reserve(num_group_cols);
  for (int c = 0; c < num_group_cols; ++c) {
    auto col = concatenated->get_column(c).view();
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
            "merge_grouped_agg: dict-encoding key col {} (avg_len={:.1f}, ndv={}, rows={})",
            c,
            avg_len,
            ndv,
            col.size());
        } else {
          group_cols.push_back(col);
          SIRIUS_LOG_DEBUG(
            "merge_grouped_agg: skipping dict-encode for key col {} "
            "(avg_len={:.1f}, ndv={}, rows={}, ratio={:.4f})",
            c,
            avg_len,
            ndv,
            col.size(),
            ratio);
        }
      } else {
        group_cols.push_back(col);
        SIRIUS_LOG_DEBUG(
          "merge_grouped_agg: skipping dict-encode for key col {} (avg_len={:.1f} < 4.0)",
          c,
          avg_len);
      }
    } else {
      group_cols.push_back(col);
    }
  }
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
  std::vector<cudf::groupby::aggregation_request> requests;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    int aggregate_col_id = num_group_cols + static_cast<int>(i);
    cudf::groupby::aggregation_request request;
    request.values = concatenated->get_column(aggregate_col_id).view();
    switch (aggregates[i]) {
      case cudf::aggregation::Kind::MIN: {
        request.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case cudf::aggregation::Kind::MAX: {
        request.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case cudf::aggregation::Kind::SUM:
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        if (aggregates[i] == cudf::aggregation::Kind::SUM) {
          // The HUGEINT->BIGINT downcast guard: refuse a 64-bit integer sum that could wrap.
          // COUNT states are exempt: they sum row counts, which stay far below int64.
          throw_if_int64_sum_could_overflow(request.values, stream, mr);
        }
        request.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case cudf::aggregation::Kind::COLLECT_SET: {
        // Intermediate column is a LIST produced by local COLLECT_SET. MERGE_SETS unions the
        // per-partition lists and drops duplicates, producing a deduplicated LIST per group.
        request.aggregations.push_back(
          cudf::make_merge_sets_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        throw std::runtime_error(
          "Unsupported cudf aggregate kind in `merge_grouped_aggregate()`: " +
          std::to_string(static_cast<int>(aggregates[i])));
    }
    requests.push_back(std::move(request));
  }

  // Call cudf groupby and populate output columns
  auto groupby_result = grpby_obj.aggregate(requests, stream, mr);
  auto output_cols    = groupby_result.first->release();

  // Decode dictionary-encoded group key columns back to STRING
  for (size_t i = 0; i < static_cast<size_t>(num_group_cols); i++) {
    if (output_cols[i]->type().id() == cudf::type_id::DICTIONARY32) {
      cudf::dictionary_column_view dict_view(output_cols[i]->view());
      output_cols[i] = cudf::dictionary::decode(dict_view, stream, mr);
    }
  }

  for (auto& aggregation_result : groupby_result.second) {
    output_cols.push_back(std::move(aggregation_result.results[0]));
  }

  // Create the output data batch
  auto output_table = std::make_unique<cudf::table>(std::move(output_cols));
  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::merge_order_by(
  const std::vector<cucascade::read_only_data_batch>& input,
  const std::vector<int>& order_key_idx,
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_order_by()` should at least contain two data batches");
  }
  if (order_key_idx.size() != column_order.size() ||
      order_key_idx.size() != null_precedence.size()) {
    throw std::runtime_error(
      "mismatch between the sizes of `order_key_idx`, `column_order`, and "
      "`null_precedence` in `merge_order_by()`");
  }

  // Pull input cudf tables and merge.
  std::vector<cudf::table_view> input_tables;
  input_tables.reserve(input.size());
  for (const auto& batch : input) {
    input_tables.push_back(get_cudf_table_view(batch));
  }
  auto output_table = cudf::merge(input_tables,
                                  order_key_idx,
                                  column_order,
                                  null_precedence,
                                  stream,
                                  memory_space.get_default_allocator());

  // Create the output data batch
  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

}  // namespace op
}  // namespace sirius
