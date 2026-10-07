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

#include "vss/join_result_shaping.hpp"

#include "vss/size_limits.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace sirius::vss {

namespace {

// The three result columns, in the order the intermediate schema declares them.
cudf::table_view triple_view(cudf::column_view const& left_rows,
                             cudf::column_view const& neighbors,
                             cudf::column_view const& distances)
{
  return cudf::table_view{{left_rows, neighbors, distances}};
}

shaped_join_result from_table(std::unique_ptr<cudf::table> t)
{
  auto cols = t->release();
  return shaped_join_result{std::move(cols[0]), std::move(cols[1]), std::move(cols[2])};
}

}  // namespace

std::unique_ptr<cudf::column> make_left_row_index(std::int64_t n_left,
                                                  std::int64_t k,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto const total = column_size(n_left * k, "vector join output");
  cudf::numeric_scalar<std::int32_t> const init(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const step(1, true, stream);
  auto positions = cudf::sequence(total, init, step, stream, mr);

  cudf::numeric_scalar<std::int32_t> const k_scalar(static_cast<std::int32_t>(k), true, stream);
  return cudf::binary_operation(positions->view(),
                                k_scalar,
                                cudf::binary_operator::DIV,
                                cudf::data_type{cudf::type_id::INT32},
                                stream,
                                mr);
}

shaped_join_result shape_per_row_top_k(std::unique_ptr<cudf::column> neighbors,
                                       std::unique_ptr<cudf::column> distances,
                                       std::int64_t n_left,
                                       std::int64_t k,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  return shaped_join_result{
    make_left_row_index(n_left, k, stream, mr), std::move(neighbors), std::move(distances)};
}

shaped_join_result shape_global_top_k(cudf::column_view const& neighbors,
                                      cudf::column_view const& distances,
                                      std::int64_t n_left,
                                      std::int64_t k,
                                      std::int64_t k_global,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto left_rows = make_left_row_index(n_left, k, stream, mr);

  // Rank every candidate pair by distance and keep the closest k_global. Ascending is
  // correct for both metrics: the fold works in distance space for cosine too, and the
  // similarity form is applied later in materialize.
  auto const order = cudf::sorted_order(
    cudf::table_view{{distances}}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, stream, mr);
  auto const available = static_cast<std::int64_t>(distances.size());
  auto const take      = static_cast<cudf::size_type>(std::min(k_global, available));
  auto const winners   = cudf::slice(order->view(), {0, take}).front();

  return from_table(cudf::gather(triple_view(left_rows->view(), neighbors, distances),
                                 winners,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr));
}

shaped_join_result shape_threshold(cudf::column_view const& neighbors,
                                   cudf::column_view const& distances,
                                   std::int64_t n_left,
                                   std::int64_t k,
                                   float max_distance,
                                   bool& truncated,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  auto left_rows = make_left_row_index(n_left, k, stream, mr);

  cudf::numeric_scalar<float> const threshold(max_distance, true, stream);
  auto const keep = cudf::binary_operation(distances,
                                           threshold,
                                           cudf::binary_operator::LESS_EQUAL,
                                           cudf::data_type{cudf::type_id::BOOL8},
                                           stream,
                                           mr);

  // Truncation test: the k-th (last) candidate of each left row sits at index i*k + k-1.
  // If any of those is itself inside the threshold, that row's neighbour list ran out
  // before the threshold did and pairs beyond k were never searched for.
  {
    cudf::numeric_scalar<std::int32_t> const first(static_cast<std::int32_t>(k - 1), true, stream);
    cudf::numeric_scalar<std::int32_t> const stride(static_cast<std::int32_t>(k), true, stream);
    auto const last_of_each =
      cudf::sequence(static_cast<cudf::size_type>(n_left), first, stride, stream, mr);
    auto const boundary = cudf::gather(cudf::table_view{{keep->view()}},
                                       last_of_each->view(),
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);
    auto const any      = cudf::reduce(boundary->get_column(0).view(),
                                  *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::BOOL8},
                                  stream,
                                  mr);
    // Synchronizes: the caller needs the verdict on the host to decide whether the answer
    // it is about to emit is complete.
    truncated =
      any->is_valid(stream) && static_cast<cudf::numeric_scalar<bool> const&>(*any).value(stream);
  }

  return from_table(cudf::apply_boolean_mask(
    triple_view(left_rows->view(), neighbors, distances), keep->view(), stream, mr));
}

}  // namespace sirius::vss
