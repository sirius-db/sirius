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

#include "op/aggregate/carried_columns.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>

#include <numeric>
#include <utility>

namespace sirius::op::detail {

std::unique_ptr<cudf::table> aggregate_carried_columns_only(cudf::table_view input,
                                                            cudf::size_type num_keys,
                                                            ::cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr)
{
  std::vector<cudf::size_type> keys(static_cast<std::size_t>(num_keys));
  std::iota(keys.begin(), keys.end(), cudf::size_type{0});
  return cudf::distinct(input,
                        keys,
                        cudf::duplicate_keep_option::KEEP_ANY,
                        cudf::null_equality::EQUAL,
                        cudf::nan_equality::ALL_EQUAL,
                        stream,
                        mr);
}

carried_row_selector append_carried_row_selector(
  std::vector<cudf::groupby::aggregation_request>& requests,
  cudf::table_view carried,
  carried_row_order const& order,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto row_positions = cudf::sequence(
    carried.num_rows(), cudf::numeric_scalar<cudf::size_type>(0, true, stream), stream, mr);
  carried_row_selector selector{};
  if (order.empty()) {
    selector.values = std::move(row_positions);
  } else {
    selector.permutation = cudf::sorted_order(
      carried.select(order.positions), order.column_order, order.null_precedence, stream, mr);
    // rank[permutation[i]] = i: every row's sorted position, a permutation of the row positions.
    cudf::table_view const positions{{row_positions->view()}};
    selector.values = std::move(
      cudf::scatter(positions, selector.permutation->view(), positions, stream, mr)->release()[0]);
  }
  cudf::groupby::aggregation_request request;
  request.values = selector.values->view();
  request.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  selector.request_idx = requests.size();
  requests.push_back(std::move(request));
  return selector;
}

std::vector<std::unique_ptr<cudf::column>> gather_carried_columns(
  cudf::table_view carried,
  carried_row_selector const& selector,
  cudf::column_view selected,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // The selector has no nulls, so every gathered row is in range.
  std::unique_ptr<cudf::column> rows;
  if (selector.permutation) {
    // MIN of a rank is a sorted position, not a row; the permutation maps it to its row.
    rows     = std::move(cudf::gather(cudf::table_view{{selector.permutation->view()}},
                                  selected,
                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                  stream,
                                  mr)
                       ->release()[0]);
    selected = rows->view();
  }
  return cudf::gather(carried, selected, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr)
    ->release();
}

}  // namespace sirius::op::detail
