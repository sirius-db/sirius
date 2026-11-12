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

#include "op/window/ranking.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace sirius::op {
namespace {

cudf::data_type const result_type{cudf::type_id::INT64};

// Throws std::invalid_argument for an empty request or a column outside @p input.
void validate(cudf::table_view input,
              std::vector<cudf::size_type> const& partition_columns,
              window_order const& order,
              std::vector<ranking_function> const& functions)
{
  if (functions.empty()) { throw std::invalid_argument("Ranking window has no functions"); }
  if (order.columns.empty()) {
    throw std::invalid_argument("Ranking window needs at least one order key");
  }
  if (order.directions.size() != order.columns.size() ||
      order.nulls.size() != order.columns.size()) {
    throw std::invalid_argument("Ranking window needs a direction and NULL order for each key");
  }
  auto const in_range = [&](cudf::size_type column) {
    return column >= 0 && column < input.num_columns();
  };
  for (auto const column : partition_columns) {
    if (!in_range(column)) {
      throw std::invalid_argument("Ranking window partition column is outside the input");
    }
  }
  for (auto const column : order.columns) {
    if (!in_range(column)) {
      throw std::invalid_argument("Ranking window order key is outside the input");
    }
  }
}

// The order keys as one column: the key itself, or a struct view over the keys when there are
// several. The view does not own or copy the key columns. The rows are already sorted by every
// key, each with its own direction and NULL placement, so the scan only uses the struct to compare
// each row with the previous one for equality.
cudf::column_view order_values(cudf::table_view sorted, std::size_t partitions, std::size_t keys)
{
  if (keys == 1) { return sorted.column(static_cast<cudf::size_type>(partitions)); }
  std::vector<cudf::column_view> children;
  children.reserve(keys);
  for (std::size_t i = 0; i < keys; ++i) {
    children.push_back(sorted.column(static_cast<cudf::size_type>(partitions + i)));
  }
  return cudf::column_view(
    cudf::data_type{cudf::type_id::STRUCT}, sorted.num_rows(), nullptr, nullptr, 0, 0, children);
}

// The grouped scan that computes @p function over the sorted rows of each partition.
std::unique_ptr<cudf::groupby_scan_aggregation> make_scan(ranking_function function)
{
  switch (function) {
    case ranking_function::rank:
    case ranking_function::dense_rank:
      // The rows are already sorted and the groupby is told so. For presorted keys cuDF does not
      // sort the values again and only compares each row with the one before it, so the
      // direction and NULL order arguments are never read. INCLUDE ranks NULL as a value, so rows
      // with a NULL key are peers of each other and get a rank like any other row.
      return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
        function == ranking_function::rank ? cudf::rank_method::MIN : cudf::rank_method::DENSE,
        cudf::order::ASCENDING,
        cudf::null_policy::INCLUDE,
        cudf::null_order::AFTER);
    case ranking_function::row_number:
      // A running count of the rows in the partition.
      return cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(
        cudf::null_policy::INCLUDE);
  }
  throw std::invalid_argument("Unsupported ranking function");
}

}  // namespace

std::unique_ptr<cudf::table> evaluate_ranking_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  window_order const& order,
  std::vector<ranking_function> const& functions,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  validate(input, partition_columns, order, functions);
  if (input.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty;
    for (std::size_t i = 0; i < functions.size(); ++i) {
      empty.push_back(cudf::make_empty_column(result_type));
    }
    return std::make_unique<cudf::table>(std::move(empty));
  }

  // Without PARTITION BY the whole input is one partition, and a constant column is its key. A
  // constant column is already in sorted order, so it takes no part in the sort or the gather.
  std::unique_ptr<cudf::column> unpartitioned;
  if (partition_columns.empty()) {
    cudf::numeric_scalar<int8_t> const zero(0, true, stream, mr);
    unpartitioned = cudf::make_column_from_scalar(zero, input.num_rows(), stream, mr);
  }
  std::size_t const partition_count = partition_columns.size();

  // The partition columns followed by the order keys, sorted together.
  std::vector<cudf::column_view> key_views;
  for (auto const column : partition_columns) {
    key_views.push_back(input.column(column));
  }
  for (auto const column : order.columns) {
    key_views.push_back(input.column(column));
  }
  cudf::table_view const keys(key_views);
  // Equal partition values only need to be next to each other, so any column order, direction and
  // NULL placement will do for them.
  std::vector<cudf::order> directions(partition_count, cudf::order::ASCENDING);
  std::vector<cudf::null_order> nulls(partition_count, cudf::null_order::BEFORE);
  directions.insert(directions.end(), order.directions.begin(), order.directions.end());
  nulls.insert(nulls.end(), order.nulls.begin(), order.nulls.end());
  auto row_order = cudf::stable_sorted_order(keys, directions, nulls, stream, mr);

  // Only rank and dense_rank read the order keys. row_number needs the partition columns alone.
  bool const needs_order = std::ranges::any_of(
    functions, [](auto function) { return function != ranking_function::row_number; });
  std::size_t const gathered_count = needs_order ? key_views.size() : partition_count;
  std::unique_ptr<cudf::table> sorted;
  if (gathered_count > 0) {
    cudf::table_view const gathered_keys(std::vector<cudf::column_view>(
      key_views.begin(), key_views.begin() + static_cast<std::ptrdiff_t>(gathered_count)));
    sorted = cudf::gather(
      gathered_keys, row_order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  }

  // The partition columns come first in the sorted table, in the order of `partition_columns`.
  cudf::table_view partition_view;
  if (unpartitioned) {
    partition_view = cudf::table_view(std::vector<cudf::column_view>{unpartitioned->view()});
  } else {
    std::vector<cudf::size_type> sorted_partition_indices(partition_count);
    std::iota(sorted_partition_indices.begin(), sorted_partition_indices.end(), 0);
    partition_view = sorted->view().select(sorted_partition_indices);
  }
  // One entry for each group key. The constant key of a window without PARTITION BY counts as one.
  std::size_t const group_key_count = std::max<std::size_t>(partition_count, 1);
  std::vector<cudf::order> const partition_directions(group_key_count, cudf::order::ASCENDING);
  std::vector<cudf::null_order> const partition_nulls(group_key_count, cudf::null_order::BEFORE);
  cudf::groupby::groupby groups(partition_view,
                                cudf::null_policy::INCLUDE,
                                cudf::sorted::YES,
                                partition_directions,
                                partition_nulls);

  cudf::column_view values;
  if (needs_order) { values = order_values(sorted->view(), partition_count, order.columns.size()); }
  std::vector<cudf::groupby::scan_request> requests;
  for (auto const function : functions) {
    cudf::groupby::scan_request request;
    request.values = function == ranking_function::row_number ? partition_view.column(0) : values;
    request.aggregations.push_back(make_scan(function));
    requests.push_back(std::move(request));
  }
  auto [labels, scans] = groups.scan(requests, stream, mr);
  labels.reset();
  sorted.reset();
  unpartitioned.reset();

  std::vector<std::unique_ptr<cudf::column>> scanned_columns;
  for (auto& scan : scans) {
    scanned_columns.push_back(std::move(scan.results.front()));
  }
  scans.clear();

  // The scan results are in sorted order and need to be permuted back into the input row order.
  // Once that is done, apply any necessary result widening, one column at a time so that only the
  // narrow columns are live while the scatter runs.
  auto scanned   = std::make_unique<cudf::table>(std::move(scanned_columns));
  auto scattered = cudf::scatter(scanned->view(), row_order->view(), scanned->view(), stream, mr);
  scanned.reset();
  row_order.reset();
  auto columns = scattered->release();
  for (auto& column : columns) {
    if (column->type() != result_type) {
      column = cudf::cast(column->view(), result_type, stream, mr);
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op
