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

#include "op/order/gpu_order_impl.hpp"

#include "data/data_batch_utils.hpp"
#include "op/order/encoded_sort_order.hpp"

#include <cudf/sorting.hpp>

#include <cstddef>
namespace sirius {
namespace op {

namespace {
/// The encoded sort is used while its working memory is at most 1/N of the memory space.
constexpr std::size_t kEncodedSortMaxSpaceFraction = 4;
}  // namespace

std::shared_ptr<cucascade::data_batch> gpu_order_impl::local_order_by(
  const cucascade::read_only_data_batch& input,
  const std::vector<int>& order_key_idx,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence,
  const std::vector<int>& projections,
  ::cuda::stream_ref stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info,
  bool sort_key_encoding)
{
  if (order_key_idx.size() != column_order.size() ||
      order_key_idx.size() != null_precedence.size()) {
    throw std::runtime_error(
      "mismatch between the sizes of `order_key_idx`, `column_order`, and "
      "`null_precedence` in `local_order_by()`");
  }

  // Get sorted order
  auto input_table = get_cudf_table_view(input);
  std::vector<cudf::column_view> sort_cols;
  for (int idx : order_key_idx) {
    sort_cols.push_back(input_table.column(idx));
  }
  const cudf::table_view sort_keys(sort_cols);
  std::unique_ptr<cudf::column> sorted_order;
  if (sort_key_encoding) {
    // String keys and multi-column keys: radix sorts over an encoding of the keys. Returns
    // nullptr for the key shapes it leaves to cuDF.
    //
    // Memory: it allocates only from the memory space's allocator, at most
    // order::encoded_sort_working_bytes(rows) at a time (32 bytes per row, against about 8 for
    // cuDF's comparison sort), and frees its temporaries before returning. An input whose sort
    // would need more than a quarter of the memory space is left to cuDF. An allocation failure
    // is not caught here: like any operator's, it reaches the executor, which records the peak
    // and retries the task with a larger reservation.
    order::encoded_sort_options options;
    options.max_working_bytes = memory_space.get_max_memory() / kEncodedSortMaxSpaceFraction;
    sorted_order              = order::encoded_sorted_order(sort_keys,
                                               column_order,
                                               null_precedence,
                                               stream,
                                               memory_space.get_default_allocator(),
                                               options);
  }
  if (!sorted_order) {
    sorted_order = cudf::sorted_order(
      sort_keys, column_order, null_precedence, stream, memory_space.get_default_allocator());
  }

  // Do projection
  std::vector<cudf::column_view> project_input_cols;
  for (int idx : projections) {
    project_input_cols.push_back(input_table.column(idx));
  }
  auto output_table = cudf::gather(cudf::table_view(project_input_cols),
                                   sorted_order->view(),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   memory_space.get_default_allocator());

  // Create the output data batch
  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

}  // namespace op
}  // namespace sirius
