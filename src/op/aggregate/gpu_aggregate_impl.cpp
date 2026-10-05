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
#include "op/aggregate/group_key_labels.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/error.hpp>

#include <algorithm>
#include <new>
#include <numeric>

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
  ::cuda::stream_ref stream,
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

namespace {

/// Aggregate @p input_table grouped by the `group_idx` columns. Returns the table
/// `[keys..., aggregates...]`. See `gpu_aggregate_impl::local_grouped_aggregate()` for the
/// parameters.
std::unique_ptr<cudf::table> grouped_aggregate_table(
  cudf::table_view input_table,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  const std::vector<std::vector<int>>& aggregate_struct_col_indices,
  ::cuda::stream_ref stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_grouped_aggregate()`");
  }

  const bool has_struct_col_indices = !aggregate_struct_col_indices.empty();

  auto mr = memory_space.get_default_allocator();

  // COLLECT_SET uses cuDF's sorted groupby. Dense INT32 labels let that sort take its
  // single-column radix path while preserving the original keys' lexicographic order. A
  // single null-free fixed-width key is already radix-sortable; single nullable or
  // variable-width keys and non-nested multi-column keys may benefit from labels.
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
        auto key_labels  = detail::make_group_key_labels(keys_view, stream, mr);
        label_key_values = std::move(key_labels.sorted_unique_keys);
        label_col        = std::move(key_labels.labels);
        SIRIUS_LOG_DEBUG(
          "local_grouped_agg: label-encoded {} group key column(s) into a single radix-sortable "
          "key for the COLLECT_SET sort path (rows={}, ndv={}, groups={})",
          group_idx.size(),
          input_table.num_rows(),
          ndv,
          label_key_values->num_rows());
      }
    } catch (const std::bad_alloc&) {
      // Preserve the task's reservation retry and downgrade path.
      throw;
    } catch (const cudf::cuda_error&) {
      // Do not continue on a potentially invalid CUDA context.
      throw;
    } catch (const rmm::cuda_error&) {
      // RMM CUDA errors carry the same invalid-context risk as cuDF CUDA errors.
      throw;
    } catch (const std::exception& e) {
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
    if (col.type().id() == cudf::type_id::STRING && col.size() > 0) {
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
  cudf::groupby::groupby grpby_obj(cudf::table_view(group_cols), cudf::null_policy::INCLUDE);

  // Make aggregation requests, group aggregations on the same column in the single request.
  // For multi-column COLLECT_SET, a synthetic negative key -(i+1) is used so that each such
  // aggregate gets its own request with a freshly synthesized struct column.
  std::unordered_map<int, std::vector<std::unique_ptr<cudf::groupby_aggregation>>> input_col_to_agg;
  std::unordered_map<int, std::vector<size_t>> input_col_to_output_idx;
  std::vector<int> input_col_order;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& aggregate_kind = aggregates[i];
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
      auto struct_col =
        cudf::make_structs_column(input_table.num_rows(),
                                  std::move(struct_children),
                                  0,
                                  cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                  stream,
                                  memory_space.get_default_allocator());
      request.values = struct_col->view();
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

  return std::make_unique<cudf::table>(std::move(output_cols));
}

/// The aggregation that combines partial results of a local aggregation of kind @p kind.
std::unique_ptr<cudf::groupby_aggregation> make_reaggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::MIN:
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::MAX:
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::SUM:
    case cudf::aggregation::Kind::COUNT_ALL:
    case cudf::aggregation::Kind::COUNT_VALID:
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::COLLECT_SET:
      return cudf::make_merge_sets_aggregation<cudf::groupby_aggregation>();
    default:
      throw std::runtime_error("Unsupported cudf aggregate kind for grouping sets: " +
                               std::to_string(static_cast<int>(kind)));
  }
}

/// The empty grouping set over an empty input: one row with zero counts and NULL for every
/// other aggregate.
std::vector<std::unique_ptr<cudf::column>> make_empty_input_row(
  cudf::table_view aggregate_cols,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> row;
  row.reserve(aggregates.size());
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto const kind = aggregates[i];
    if (kind == cudf::aggregation::Kind::COUNT_ALL ||
        kind == cudf::aggregation::Kind::COUNT_VALID) {
      row.push_back(cast_if_needed(make_constant_column<int64_t>(0, 1, stream, mr),
                                   aggregate_cols.column(i).type(),
                                   stream,
                                   mr));
    } else if (kind == cudf::aggregation::Kind::COLLECT_SET) {
      // An empty set, so that COUNT(DISTINCT) of the empty input is 0.
      auto offsets = make_constant_column<cudf::size_type>(0, 2, stream, mr);
      auto child   = cudf::empty_like(cudf::lists_column_view(aggregate_cols.column(i)).child());
      row.push_back(
        cudf::make_lists_column(1, std::move(offsets), std::move(child), 0, rmm::device_buffer{}));
    } else {
      row.push_back(make_null_column(aggregate_cols.column(i), 1, stream, mr));
    }
  }
  return row;
}

/// Expand the partial aggregate @p partial, grouped by all @p num_keys keys, into one block of
/// rows per grouping set. See `gpu_aggregate_impl::local_grouping_sets_aggregate()`.
std::unique_ptr<cudf::table> expand_grouping_sets(
  cudf::table_view partial,
  std::size_t num_keys,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<std::set<std::size_t>>& grouping_sets,
  const std::vector<std::vector<std::size_t>>& grouping_functions,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<cudf::size_type> aggregate_col_ids(aggregates.size());
  std::iota(aggregate_col_ids.begin(), aggregate_col_ids.end(), static_cast<int>(num_keys));
  auto const partial_aggregates = partial.select(aggregate_col_ids);

  // Owners of the columns that the per-set views below reference.
  std::vector<std::vector<std::unique_ptr<cudf::column>>> owned(grouping_sets.size());
  std::vector<cudf::table_view> set_views;
  set_views.reserve(grouping_sets.size());

  for (std::size_t set_idx = 0; set_idx < grouping_sets.size(); ++set_idx) {
    auto const& set = grouping_sets[set_idx];
    auto& owner     = owned[set_idx];
    std::vector<cudf::column_view> keys(num_keys);
    std::vector<cudf::column_view> values;
    cudf::size_type num_rows = 0;

    if (set.size() == num_keys) {
      // The partial aggregate is already grouped by this set.
      for (std::size_t k = 0; k < num_keys; ++k) {
        keys[k] = partial.column(static_cast<cudf::size_type>(k));
      }
      for (auto const& col : partial_aggregates) {
        values.push_back(col);
      }
      num_rows = partial.num_rows();
    } else {
      std::vector<cudf::column_view> set_keys;
      std::unique_ptr<cudf::column> constant_key;
      if (set.empty()) {
        constant_key = make_constant_column<int8_t>(0, partial.num_rows(), stream, mr);
        set_keys.push_back(constant_key->view());
      }
      for (auto const k : set) {
        set_keys.push_back(partial.column(static_cast<cudf::size_type>(k)));
      }
      cudf::groupby::groupby grouper(cudf::table_view(set_keys), cudf::null_policy::INCLUDE);
      std::vector<cudf::groupby::aggregation_request> requests(aggregates.size());
      for (std::size_t i = 0; i < aggregates.size(); ++i) {
        requests[i].values = partial_aggregates.column(static_cast<cudf::size_type>(i));
        requests[i].aggregations.push_back(make_reaggregation(aggregates[i]));
      }
      auto [grouped_keys, results] = grouper.aggregate(requests, stream, mr);
      constant_key.reset();
      num_rows = grouped_keys->num_rows();

      if (set.empty() && num_rows == 0) {
        for (auto& col : make_empty_input_row(partial_aggregates, aggregates, stream, mr)) {
          values.push_back(col->view());
          owner.push_back(std::move(col));
        }
        num_rows = 1;
      } else {
        // Re-aggregating a COUNT with SUM widens it, so match the partial column types.
        for (std::size_t i = 0; i < results.size(); ++i) {
          auto col =
            cast_if_needed(std::move(results[i].results[0]),
                           partial_aggregates.column(static_cast<cudf::size_type>(i)).type(),
                           stream,
                           mr);
          values.push_back(col->view());
          owner.push_back(std::move(col));
        }
      }

      auto key_cols       = grouped_keys->release();
      std::size_t key_pos = set.empty() ? 1 : 0;
      for (std::size_t k = 0; k < num_keys; ++k) {
        if (set.contains(k)) {
          keys[k] = key_cols[key_pos]->view();
          owner.push_back(std::move(key_cols[key_pos++]));
        } else {
          auto nulls =
            make_null_column(partial.column(static_cast<cudf::size_type>(k)), num_rows, stream, mr);
          keys[k] = nulls->view();
          owner.push_back(std::move(nulls));
        }
      }
    }

    std::vector<cudf::column_view> cols(keys.begin(), keys.end());
    auto set_id =
      make_constant_column<int32_t>(static_cast<int32_t>(set_idx), num_rows, stream, mr);
    cols.push_back(set_id->view());
    owner.push_back(std::move(set_id));
    for (auto const& function : grouping_functions) {
      // GROUPING(a, b, ...) has one bit per argument, the first argument most significant, set
      // when the argument is not in the grouping set.
      int64_t value = 0;
      for (auto const k : function) {
        value = (value << 1) | (set.contains(k) ? 0 : 1);
      }
      auto col = make_constant_column<int64_t>(value, num_rows, stream, mr);
      cols.push_back(col->view());
      owner.push_back(std::move(col));
    }
    cols.insert(cols.end(), values.begin(), values.end());
    set_views.emplace_back(cols);
  }

  return cudf::concatenate(set_views, stream, mr);
}

}  // namespace

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_grouped_aggregate(
  const cucascade::read_only_data_batch& input,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  const std::vector<std::vector<int>>& aggregate_struct_col_indices,
  ::cuda::stream_ref stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  auto output_table = grouped_aggregate_table(get_cudf_table_view(input),
                                              group_idx,
                                              aggregates,
                                              aggregate_idx,
                                              aggregate_struct_col_indices,
                                              stream,
                                              memory_space);
  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_grouping_sets_aggregate(
  const cucascade::read_only_data_batch& input,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  const std::vector<std::vector<int>>& aggregate_struct_col_indices,
  const std::vector<std::set<std::size_t>>& grouping_sets,
  const std::vector<std::vector<std::size_t>>& grouping_functions,
  ::cuda::stream_ref stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  auto partial      = grouped_aggregate_table(get_cudf_table_view(input),
                                         group_idx,
                                         aggregates,
                                         aggregate_idx,
                                         aggregate_struct_col_indices,
                                         stream,
                                         memory_space);
  auto output_table = expand_grouping_sets(partial->view(),
                                           group_idx.size(),
                                           aggregates,
                                           grouping_sets,
                                           grouping_functions,
                                           stream,
                                           memory_space.get_default_allocator());
  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

}  // namespace op
}  // namespace sirius
