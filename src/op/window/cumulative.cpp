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

#include "op/window/cumulative.hpp"

#include "op/window/aggregate_types.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace sirius::op {
namespace {

// Whether a `min` or `max` over `type` is supported.
bool is_extremum_type(cudf::data_type type)
{
  using cudf::type_id;
  switch (type.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128:
    case type_id::TIMESTAMP_DAYS:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS: return true;
    default: return false;
  }
}

// The type of the result column of `function` over an argument of type `input`.
cudf::data_type result_type(aggregate_id function, cudf::data_type input)
{
  if (function == aggregate_id::avg) {
    throw std::invalid_argument("Unsupported cumulative window function");
  }
  if ((function == aggregate_id::min || function == aggregate_id::max) &&
      !is_extremum_type(input)) {
    throw std::invalid_argument("Unsupported cumulative window min or max argument type");
  }
  return window_result_type(function, input);
}

// Returns @p aggregates with `sum_no_overflow` replaced by `sum`. Throws std::invalid_argument for
// an empty request, an unsupported function or argument type, or a column outside @p input.
std::vector<cumulative_aggregate> normalize_aggregates(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  ranking_order const& order,
  std::vector<cumulative_aggregate> aggregates)
{
  if (aggregates.empty()) { throw std::invalid_argument("Cumulative window has no aggregates"); }
  if (order.columns.empty()) {
    throw std::invalid_argument("Cumulative window needs at least one order key");
  }
  if (order.directions.size() != order.columns.size() ||
      order.nulls.size() != order.columns.size()) {
    throw std::invalid_argument("Cumulative window needs a direction and NULL order for each key");
  }
  auto const in_range = [&](cudf::size_type column) {
    return column >= 0 && column < input.num_columns();
  };
  for (auto const column : partition_columns) {
    if (!in_range(column)) {
      throw std::invalid_argument("Cumulative window partition column is outside the input");
    }
  }
  for (auto const column : order.columns) {
    if (!in_range(column)) {
      throw std::invalid_argument("Cumulative window order key is outside the input");
    }
  }
  for (auto& aggregate : aggregates) {
    if (!in_range(aggregate.argument_column)) {
      throw std::invalid_argument("Cumulative window argument is outside the input");
    }
    if (aggregate.function == aggregate_id::sum_no_overflow) {
      aggregate.function = aggregate_id::sum;
    }
    (void)result_type(aggregate.function, input.column(aggregate.argument_column).type());
  }
  return aggregates;
}

// The grouped scan that computes @p function over the sorted rows of each partition.
std::unique_ptr<cudf::groupby_scan_aggregation> make_scan(aggregate_id function)
{
  switch (function) {
    case aggregate_id::sum: return cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
    case aggregate_id::count:
      return cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(
        cudf::null_policy::EXCLUDE);
    case aggregate_id::count_star:
      return cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(
        cudf::null_policy::INCLUDE);
    case aggregate_id::min: return cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
    case aggregate_id::max: return cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
    default: throw std::invalid_argument("Unsupported cumulative window function");
  }
}

}  // namespace

std::unique_ptr<cudf::table> evaluate_cumulative_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  ranking_order const& order,
  std::vector<cumulative_aggregate> const& requested,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const aggregates = normalize_aggregates(input, partition_columns, order, requested);
  if (input.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty;
    for (auto const& aggregate : aggregates) {
      empty.push_back(cudf::make_empty_column(
        result_type(aggregate.function, input.column(aggregate.argument_column).type())));
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

  // Sort the rows by the partition columns and then by the order keys. Equal partition values only
  // need to be next to each other, so any column order, direction and NULL placement will do for
  // them.
  std::vector<cudf::column_view> key_views;
  for (auto const column : partition_columns) {
    key_views.push_back(input.column(column));
  }
  for (auto const column : order.columns) {
    key_views.push_back(input.column(column));
  }
  std::vector<cudf::order> directions(partition_count, cudf::order::ASCENDING);
  std::vector<cudf::null_order> nulls(partition_count, cudf::null_order::BEFORE);
  directions.insert(directions.end(), order.directions.begin(), order.directions.end());
  nulls.insert(nulls.end(), order.nulls.begin(), order.nulls.end());
  auto row_order =
    cudf::stable_sorted_order(cudf::table_view(key_views), directions, nulls, stream, mr);

  // Gather the partition columns, then each distinct argument column, in sorted order.
  std::vector<cudf::column_view> gather_views(
    key_views.begin(), key_views.begin() + static_cast<std::ptrdiff_t>(partition_count));
  std::vector<cudf::size_type> argument_columns;
  std::vector<std::size_t> argument_position(aggregates.size());
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto const column    = aggregates[i].argument_column;
    auto const found     = std::ranges::find(argument_columns, column);
    argument_position[i] = static_cast<std::size_t>(found - argument_columns.begin());
    if (found == argument_columns.end()) {
      argument_columns.push_back(column);
      gather_views.push_back(input.column(column));
    }
  }
  auto sorted = cudf::gather(cudf::table_view(gather_views),
                             row_order->view(),
                             cudf::out_of_bounds_policy::DONT_CHECK,
                             stream,
                             mr);

  cudf::table_view partition_view;
  if (unpartitioned) {
    partition_view = cudf::table_view(std::vector<cudf::column_view>{unpartitioned->view()});
  } else {
    std::vector<cudf::size_type> partition_indices(partition_count);
    std::iota(partition_indices.begin(), partition_indices.end(), 0);
    partition_view = sorted->view().select(partition_indices);
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

  // A sum accumulates in a wider type than its argument, so it scans a widened copy. An 8 to 32 bit
  // integer is scanned as it is: cuDF sums it into INT64, and the result is cast to its result type
  // at the end.
  std::vector<std::unique_ptr<cudf::column>> widened(aggregates.size());
  std::vector<cudf::groupby::scan_request> requests;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto values =
      sorted->view().column(static_cast<cudf::size_type>(partition_count + argument_position[i]));
    if (aggregates[i].function == aggregate_id::sum && !is_narrow_integer(values.type())) {
      auto const target = window_sum_type(values.type());
      if (values.type() != target) {
        widened[i] = cudf::cast(values, target, stream, mr);
        values     = widened[i]->view();
      }
    }
    cudf::groupby::scan_request request;
    request.values = values;
    request.aggregations.push_back(make_scan(aggregates[i].function));
    requests.push_back(std::move(request));
  }
  auto [labels, scans] = groups.scan(requests, stream, mr);
  labels.reset();
  requests.clear();
  widened.clear();

  std::vector<std::unique_ptr<cudf::column>> scanned_columns;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto column = std::move(scans[i].results.front());
    // A grouped scan has no overflow detection, so a running sum can wrap in its 128 bit
    // accumulator. Checking every running sum against 10^38 also detects a wrap. A DECIMAL(38)
    // value is below 10^38 in magnitude, so the first running sum that reaches 10^38 in magnitude
    // is below 2 * 10^38 in magnitude. Below 2^127 it has not wrapped. From 2^127 up it wraps to a
    // value between -2^127 and -1.4 * 10^38, which is also beyond -10^38, and a negative sum wraps
    // the same way to a value beyond 10^38. A sum over a narrower argument stays far below 10^38,
    // because a table holds fewer than 2^31 rows.
    if (aggregates[i].function == aggregate_id::sum &&
        input.column(aggregates[i].argument_column).type().id() == cudf::type_id::DECIMAL128) {
      check_decimal_sum_fits(column->view(), stream, mr);
    }
    scanned_columns.push_back(std::move(column));
  }
  scans.clear();

  // The scan returns NULL for a row whose argument is NULL. The running value carries over such a
  // row, so fill each NULL with the closest value before it in its own partition. A NULL at the
  // start of a partition stays NULL. A result without NULLs needs no fill.
  std::vector<cudf::size_type> fill_indices;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    if (aggregates[i].function != aggregate_id::count &&
        aggregates[i].function != aggregate_id::count_star && scanned_columns[i]->has_nulls()) {
      fill_indices.push_back(static_cast<cudf::size_type>(i));
    }
  }
  if (!fill_indices.empty()) {
    std::vector<cudf::column_view> to_fill;
    for (auto const index : fill_indices) {
      to_fill.push_back(scanned_columns[static_cast<std::size_t>(index)]->view());
    }
    std::vector<cudf::replace_policy> const policies(fill_indices.size(),
                                                     cudf::replace_policy::PRECEDING);
    auto [fill_keys, filled] =
      groups.replace_nulls(cudf::table_view(to_fill), policies, stream, mr);
    fill_keys.reset();
    auto filled_columns = filled->release();
    for (std::size_t j = 0; j < fill_indices.size(); ++j) {
      scanned_columns[static_cast<std::size_t>(fill_indices[j])] = std::move(filled_columns[j]);
    }
  }
  sorted.reset();
  unpartitioned.reset();

  // The scan results are in sorted order and need to be permuted back into the input row order.
  auto scanned   = std::make_unique<cudf::table>(std::move(scanned_columns));
  auto scattered = cudf::scatter(scanned->view(), row_order->view(), scanned->view(), stream, mr);
  scanned.reset();
  row_order.reset();
  auto columns = scattered->release();
  // Apply the result types, such as INT64 for the counts, one column at a time.
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto const type =
      result_type(aggregates[i].function, input.column(aggregates[i].argument_column).type());
    if (columns[i]->type() != type) {
      columns[i] = cudf::cast(columns[i]->view(), type, stream, mr);
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op
