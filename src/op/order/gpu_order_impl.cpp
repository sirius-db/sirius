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
#include "log/logging.hpp"
#include "op/order/encoded_sort_order.hpp"

#include <cudf/sorting.hpp>

#include <new>
namespace sirius {
namespace op {

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
    try {
      sorted_order = order::encoded_sorted_order(
        sort_keys, column_order, null_precedence, stream, memory_space.get_default_allocator());
    } catch (const std::bad_alloc& e) {
      // The encoded path needs more working memory than cuDF's comparison sort (up to about 40
      // bytes per row against 8): under memory pressure take the smaller one.
      SIRIUS_LOG_DEBUG("local_order_by: encoded sort ran out of memory ({}), using cudf", e.what());
      sorted_order.reset();
    }
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
