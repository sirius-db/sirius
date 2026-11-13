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

#include "op/order/gpu_sampled_merge.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <numeric>
#include <stdexcept>

namespace sirius::op::sampled_merge {
namespace {

void validate_samples(cudf::table_view table, sort_comparison_config const& comparison)
{
  auto keys = comparison.directions.size();
  if (!keys || table.num_columns() != keys + 2) {
    throw std::invalid_argument("Sampled merge expects keys and two coordinate columns");
  }
  for (auto i : {keys, keys + 1}) {
    auto column = table.column(i);
    if (column.type().id() != cudf::type_id::UINT64 || column.nullable()) {
      throw std::invalid_argument("Sampled merge coordinates must be nonnullable UINT64");
    }
  }
}

// Invoked only when at least one complete block exists, so both the stride and
// last generated index fit size_type. No host vector of sample indices.
std::unique_ptr<cudf::column> block_indices(cudf::size_type rows,
                                            uint64_t stride,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  if (!stride) { throw std::invalid_argument("Sample stride must be positive"); }
  auto count = rows / stride;
  if (!count) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  cudf::numeric_scalar<cudf::size_type> start(stride - 1, true, stream, mr);
  cudf::numeric_scalar<cudf::size_type> step(stride, true, stream, mr);
  return cudf::sequence(static_cast<cudf::size_type>(count), start, step, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::table> gather_samples(cudf::table_view input,
                                            uint64_t ordinal,
                                            uint64_t stride,
                                            sort_comparison_config const& comparison,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto indices = block_indices(input.num_rows(), stride, stream, mr);
  auto keys    = cudf::gather(input.select(comparison.key_columns),
                           indices->view(),
                           cudf::out_of_bounds_policy::DONT_CHECK,
                           stream,
                           mr);
  auto columns = keys->release();
  cudf::numeric_scalar<uint64_t> input_id(ordinal, true, stream, mr);
  cudf::numeric_scalar<uint64_t> zero(0, true, stream, mr);
  cudf::numeric_scalar<uint64_t> start(stride - 1, true, stream, mr);
  cudf::numeric_scalar<uint64_t> step(stride, true, stream, mr);
  columns.push_back(cudf::sequence(indices->size(), input_id, zero, stream, mr));
  columns.push_back(cudf::sequence(indices->size(), start, step, stream, mr));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> order_sample_splitters(std::vector<cudf::table_view> const& samples,
                                                    uint64_t sample_step,
                                                    sort_comparison_config const& comparison,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (samples.empty() || !sample_step) {
    throw std::invalid_argument("Sampled merge needs a sample schema and positive sample step");
  }
  uint64_t rows = 0;
  for (auto table : samples) {
    validate_samples(table, comparison);
    rows += table.num_rows();
  }
  if (rows > static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::overflow_error("Sampled merge sample assembly exceeds cuDF row limit");
  }
  auto indices = block_indices(static_cast<cudf::size_type>(rows), sample_step, stream, mr);
  if (!indices->size()) { return cudf::empty_like(samples.front()); }
  auto assembled  = cudf::concatenate(samples, stream, mr);
  auto directions = comparison.directions;
  auto nulls      = comparison.null_precedence;
  directions.insert(directions.end(), 2, cudf::order::ASCENDING);
  nulls.insert(nulls.end(), 2, cudf::null_order::AFTER);
  auto order    = cudf::sorted_order(assembled->view(), directions, nulls, stream, mr);
  auto selected = cudf::gather(cudf::table_view({order->view()}),
                               indices->view(),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               mr);
  // Only the selected permutation entries survive the final gather. Frees use
  // this stream, so these owners can retire after their last queued read.
  order.reset();
  indices.reset();
  return cudf::gather(assembled->view(),
                      selected->view().column(0),
                      cudf::out_of_bounds_policy::DONT_CHECK,
                      stream,
                      mr);
}

std::unique_ptr<cudf::column> discover_cuts(cudf::table_view input,
                                            uint64_t ordinal,
                                            cudf::table_view splitters,
                                            sort_comparison_config const& comparison,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  validate_samples(splitters, comparison);
  auto keys = static_cast<cudf::size_type>(comparison.directions.size());
  std::vector<cudf::size_type> key_indices(keys);
  std::iota(key_indices.begin(), key_indices.end(), 0);
  auto input_keys    = input.select(comparison.key_columns);
  auto splitter_keys = splitters.select(key_indices);
  auto lower         = cudf::lower_bound(
    input_keys, splitter_keys, comparison.directions, comparison.null_precedence, stream, mr);
  auto upper = cudf::upper_bound(
    input_keys, splitter_keys, comparison.directions, comparison.null_precedence, stream, mr);
  auto result = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          splitters.num_rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto l      = lower->view().data<cudf::size_type>();
  auto h      = upper->view().data<cudf::size_type>();
  auto r      = splitters.column(keys).data<uint64_t>();
  auto j      = splitters.column(keys + 1).data<uint64_t>();
  thrust::transform(
    rmm::exec_policy(stream, mr),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator(splitters.num_rows()),
    result->mutable_view().data<cudf::size_type>(),
    [=] __device__(cudf::size_type t) {
      return ordinal < r[t] ? h[t] : ordinal > r[t] ? l[t] : static_cast<cudf::size_type>(j[t] + 1);
    });
  return result;
}

}  // namespace sirius::op::sampled_merge
