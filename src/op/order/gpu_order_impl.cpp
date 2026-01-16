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

namespace sirius {
namespace op {

std::shared_ptr<cucascade::data_batch> gpu_order_impl::local_order_by(
  std::shared_ptr<cucascade::data_batch> input,
  const std::vector<int>& order_key_idx,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence,
  const std::vector<int>& projections,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  if (order_key_idx.size() != column_order.size() ||
      order_key_idx.size() != null_precedence.size()) {
    throw std::runtime_error(
      "mismatch between the sizes of `order_key_idx`, `column_order`, and "
      "`null_precedence` in `local_order_by()`");
  }

  // Get sorted order
  auto input_table = get_cudf_table_view(*input);
  std::vector<cudf::column_view> sort_cols;
  for (int idx : order_key_idx) {
    sort_cols.push_back(input_table.column(idx));
  }
  auto sorted_order =
    cudf::sorted_order(cudf::table_view(sort_cols), column_order, null_precedence);

  // Do projection
  std::vector<cudf::column_view> project_input_cols;
  for (int idx : projections) {
    project_input_cols.push_back(input_table.column(idx));
  }
  auto output_table = cudf::gather(cudf::table_view(project_input_cols), sorted_order->view());

  // Create the output data batch
  return make_data_batch(std::move(output_table), memory_space);
}

std::shared_ptr<cucascade::data_batch> gpu_order_impl::local_top_n(
  std::shared_ptr<cucascade::data_batch> input,
  const int limit,
  const int offset,
  const std::vector<int>& order_key_idx,
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence,
  const std::vector<int>& projections,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  if (order_key_idx.size() != column_order.size() ||
      order_key_idx.size() != null_precedence.size()) {
    throw std::runtime_error(
      "mismatch between the sizes of `order_key_idx`, `column_order`, and "
      "`null_precedence` in `local_top_n()`");
  }

  // Get sorted order
  auto input_table = get_cudf_table_view(*input);
  std::vector<cudf::column_view> sort_cols;
  for (int idx : order_key_idx) {
    sort_cols.push_back(input_table.column(idx));
  }
  auto sorted_order =
    cudf::sorted_order(cudf::table_view(sort_cols), column_order, null_precedence);

  // Get top `limit + offset` rows
  std::unique_ptr<cudf::table> output_table = nullptr;
  if (sorted_order->size() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    for (int idx : projections) {
      empty_cols.push_back(cudf::make_empty_column(input_table.column(idx).type()));
    }
    output_table = std::make_unique<cudf::table>(std::move(empty_cols));
  } else {
    auto sliced_sorted_order =
      (limit + offset >= static_cast<int>(sorted_order->size()))
        ? sorted_order->view()
        : cudf::slice(
            sorted_order->view(), {0, static_cast<cudf::size_type>(limit + offset)}, stream)[0];
    std::vector<cudf::column_view> project_input_cols;
    for (int idx : projections) {
      project_input_cols.push_back(input_table.column(idx));
    }
    output_table = cudf::gather(cudf::table_view(project_input_cols), sliced_sorted_order);
  }

  // Create the output data batch
  return make_data_batch(std::move(output_table), memory_space);
}

}  // namespace op
}  // namespace sirius
