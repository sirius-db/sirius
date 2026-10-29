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
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <stdexcept>

namespace sirius::op {
namespace {

cudf::data_type const result_type{cudf::type_id::INT64};

void validate(cudf::table_view input,
              std::vector<cudf::size_type> const& partition_columns,
              ranking_order const& order,
              std::vector<ranking_function> const& functions)
{
  if (functions.empty()) { throw std::invalid_argument("Ranking window has no functions"); }
  if (partition_columns.empty()) {
    throw std::invalid_argument("Ranking window needs at least one partition column");
  }
  if (order.columns.empty()) {
    throw std::invalid_argument("Ranking window needs at least one order key");
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

// The order keys as one column: the key itself, or a struct of the keys when there are several.
cudf::column_view order_values(cudf::table_view sorted,
                               std::size_t partitions,
                               std::size_t keys,
                               std::unique_ptr<cudf::column>& owner,
                               ::cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  if (keys == 1) { return sorted.column(static_cast<cudf::size_type>(partitions)); }
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(keys);
  for (std::size_t i = 0; i < keys; ++i) {
    children.push_back(std::make_unique<cudf::column>(
      sorted.column(static_cast<cudf::size_type>(partitions + i)), stream, mr));
  }
  owner = cudf::make_structs_column(
    sorted.num_rows(), std::move(children), 0, rmm::device_buffer{}, stream, mr);
  return owner->view();
}

std::unique_ptr<cudf::groupby_scan_aggregation> make_scan(ranking_function function,
                                                          ranking_order const& order)
{
  switch (function) {
    case ranking_function::rank:
    case ranking_function::dense_rank:
      return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
        function == ranking_function::rank ? cudf::rank_method::MIN : cudf::rank_method::DENSE,
        order.direction,
        cudf::null_policy::INCLUDE,
        order.nulls);
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
  ranking_order const& order,
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

  // The partition columns followed by the order keys, sorted together.
  std::vector<cudf::column_view> key_views;
  for (auto const column : partition_columns) {
    key_views.push_back(input.column(column));
  }
  for (auto const column : order.columns) {
    key_views.push_back(input.column(column));
  }
  cudf::table_view const keys(key_views);
  std::vector<cudf::order> directions(partition_columns.size(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> nulls(partition_columns.size(), cudf::null_order::BEFORE);
  directions.insert(directions.end(), order.columns.size(), order.direction);
  nulls.insert(nulls.end(), order.columns.size(), order.nulls);
  auto row_order = cudf::stable_sorted_order(keys, directions, nulls, stream, mr);
  auto sorted =
    cudf::gather(keys, row_order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

  std::vector<cudf::column_view> partition_views;
  for (std::size_t i = 0; i < partition_columns.size(); ++i) {
    partition_views.push_back(sorted->view().column(static_cast<cudf::size_type>(i)));
  }
  cudf::groupby::groupby groups(
    cudf::table_view(partition_views),
    cudf::null_policy::INCLUDE,
    cudf::sorted::YES,
    std::vector<cudf::order>(partition_columns.size(), cudf::order::ASCENDING),
    std::vector<cudf::null_order>(partition_columns.size(), cudf::null_order::BEFORE));

  std::unique_ptr<cudf::column> order_owner;
  auto const values = order_values(
    sorted->view(), partition_columns.size(), order.columns.size(), order_owner, stream, mr);
  std::vector<cudf::groupby::scan_request> requests;
  for (auto const function : functions) {
    cudf::groupby::scan_request request;
    request.values = function == ranking_function::row_number ? partition_views.front() : values;
    request.aggregations.push_back(make_scan(function, order));
    requests.push_back(std::move(request));
  }
  auto [labels, scans] = groups.scan(requests, stream, mr);
  labels.reset();
  order_owner.reset();
  sorted.reset();

  std::vector<std::unique_ptr<cudf::column>> ranked;
  for (auto& scan : scans) {
    auto& column = scan.results.front();
    ranked.push_back(column->type() == result_type
                       ? std::move(column)
                       : cudf::cast(column->view(), result_type, stream, mr));
  }
  scans.clear();

  // The scan results are in sorted order. Put each one back at the position of its input row.
  cudf::table const ranked_table(std::move(ranked));
  return cudf::scatter(ranked_table.view(), row_order->view(), ranked_table.view(), stream, mr);
}

}  // namespace sirius::op
