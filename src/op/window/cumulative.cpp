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

#include "op/aggregate/wide_integer_sum.hpp"
#include "op/window/whole_partition.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sirius::op {
namespace {

cudf::data_type const count_type{cudf::type_id::INT64};

// The type a `sum` over `input` accumulates in.
cudf::data_type sum_type(cudf::data_type input)
{
  using cudf::type_id;
  switch (input.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64: return wide_integer_sum_type();
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128: return cudf::data_type{type_id::DECIMAL128, input.scale()};
    case type_id::FLOAT32:
    case type_id::FLOAT64: return cudf::data_type{type_id::FLOAT64};
    default: throw std::invalid_argument("Unsupported cumulative window sum argument type");
  }
}

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
  switch (function) {
    case aggregate_id::sum: return sum_type(input);
    case aggregate_id::count:
    case aggregate_id::count_star: return count_type;
    case aggregate_id::min:
    case aggregate_id::max:
      if (!is_extremum_type(input)) {
        throw std::invalid_argument("Unsupported cumulative window min or max argument type");
      }
      return input;
    default: throw std::invalid_argument("Unsupported cumulative window function");
  }
}

// sum_no_overflow is a SUM. Throws std::invalid_argument for an empty request, an unsupported
// function or argument type, or a column outside @p input.
std::vector<cumulative_aggregate> validate(cudf::table_view input,
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

// A grouped scan has no overflow detection, so a running sum over DECIMAL128 values can wrap in its
// 128 bit accumulator. The same running sum over the values as FLOAT64 is close to the exact one
// (the relative error is far below 1e-6), so it tells whether the sum reaches 10^38 in magnitude.
// The limit is in the units of the FLOAT64 values.
void check_sum_below_limit(cudf::column_view approximate_sums,
                           int scale,
                           ::cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  double const limit = 1e38 * std::pow(10.0, scale) * (1.0 - 1e-6);
  auto bounds        = cudf::minmax(approximate_sums, stream, mr);
  for (auto const* bound : {bounds.first.get(), bounds.second.get()}) {
    if (!bound->is_valid(stream)) { continue; }
    auto const value = static_cast<cudf::numeric_scalar<double> const&>(*bound).value(stream);
    if (std::abs(value) >= limit) {
      throw std::overflow_error("Cumulative window sum exceeds DECIMAL(38)");
    }
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
  auto const aggregates = validate(input, partition_columns, order, requested);
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
  std::size_t const key_count = std::max<std::size_t>(partition_count, 1);
  std::vector<cudf::order> const partition_directions(key_count, cudf::order::ASCENDING);
  std::vector<cudf::null_order> const partition_nulls(key_count, cudf::null_order::BEFORE);
  std::vector<cudf::order> directions(partition_directions.begin(),
                                      partition_directions.begin() + partition_count);
  std::vector<cudf::null_order> nulls(partition_nulls.begin(),
                                      partition_nulls.begin() + partition_count);
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
    auto const found     = std::find(argument_columns.begin(), argument_columns.end(), column);
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
  cudf::groupby::groupby groups(partition_view,
                                cudf::null_policy::INCLUDE,
                                cudf::sorted::YES,
                                partition_directions,
                                partition_nulls);

  // A sum accumulates in a wider type than its argument, so it scans a widened copy.
  std::vector<std::unique_ptr<cudf::column>> widened(aggregates.size());
  std::vector<cudf::groupby::scan_request> requests;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto values =
      sorted->view().column(static_cast<cudf::size_type>(partition_count + argument_position[i]));
    if (aggregates[i].function == aggregate_id::sum) {
      auto const target = sum_type(values.type());
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
  // The FLOAT64 running sums that guard a DECIMAL128 sum against wrapping, after the requests
  // above.
  std::vector<std::unique_ptr<cudf::column>> approximate(aggregates.size());
  std::vector<std::size_t> guard_request(aggregates.size(), 0);
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto const type = input.column(aggregates[i].argument_column).type();
    if (aggregates[i].function != aggregate_id::sum || type.id() != cudf::type_id::DECIMAL128) {
      continue;
    }
    auto const values =
      sorted->view().column(static_cast<cudf::size_type>(partition_count + argument_position[i]));
    approximate[i] = cudf::cast(values, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
    cudf::groupby::scan_request request;
    request.values = approximate[i]->view();
    request.aggregations.push_back(make_scan(aggregate_id::sum));
    guard_request[i] = requests.size();
    requests.push_back(std::move(request));
  }
  auto [labels, scans] = groups.scan(requests, stream, mr);
  labels.reset();
  requests.clear();
  widened.clear();
  approximate.clear();
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    if (guard_request[i] != 0) {
      check_sum_below_limit(scans[guard_request[i]].results.front()->view(),
                            input.column(aggregates[i].argument_column).type().scale(),
                            stream,
                            mr);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> scanned_columns;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto column = std::move(scans[i].results.front());
    if (aggregates[i].function == aggregate_id::sum &&
        cudf::is_fixed_point(input.column(aggregates[i].argument_column).type())) {
      check_decimal_sum_fits(column->view(), stream, mr);
    }
    scanned_columns.push_back(std::move(column));
  }
  scans.clear();

  // The scan returns NULL for a row whose argument is NULL. The running value carries over such a
  // row, so fill each NULL with the closest value before it in its own partition. A NULL at the
  // start of a partition stays NULL.
  std::vector<cudf::size_type> fill_indices;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    if (aggregates[i].function != aggregate_id::count &&
        aggregates[i].function != aggregate_id::count_star) {
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
  // Apply the INT64 result type of the counts, one column at a time.
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    bool const is_count = aggregates[i].function == aggregate_id::count ||
                          aggregates[i].function == aggregate_id::count_star;
    if (is_count && columns[i]->type() != count_type) {
      columns[i] = cudf::cast(columns[i]->view(), count_type, stream, mr);
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op
