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

#include <cudf/sorting.hpp>

namespace sirius {
namespace op {

std::shared_ptr<cucascade::data_batch> gpu_order_impl::local_order_by(
  const cucascade::read_only_data_batch& input,
  const std::vector<int>& order_key_idx,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence,
  const std::vector<int>& projections,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  const telemetry::batch_telemetry_info& telemetry_info)
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
  auto sorted_order = cudf::sorted_order(cudf::table_view(sort_cols),
                                         column_order,
                                         null_precedence,
                                         stream,
                                         memory_space.get_default_allocator());

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

  // Validate-and-retry guard (issue #1452): cudf::sorted_order intermittently returns a
  // permutation that does NOT sort the keys (observed on GB300/sm_103 + CUDA 13.3 whenever
  // several sorts run concurrently on different streams; both sorted_order and
  // stable_sorted_order are affected, ~30-70%% of multi-batch ORDER BY queries in the #1452
  // repro). Downstream MERGE_SORT feeds these batches to cudf::merge, whose sorted-input
  // contract violation silently duplicates and drops rows with the total count preserved.
  // The mis-sort is transient: re-running the sort has always produced a correct permutation
  // on the first retry in testing. Validate the gathered output and retry until sorted; give
  // up loudly rather than publish an unsorted batch. Remove once the upstream cudf/CUB defect
  // is fixed.
  constexpr int max_sort_attempts = 4;
  for (int attempt = 0;; ++attempt) {
    std::vector<cudf::column_view> out_keys;
    out_keys.reserve(order_key_idx.size());
    for (std::size_t i = 0; i < order_key_idx.size(); ++i) {
      // Sort keys keep their input indices in the projected output on this path (the ORDER_BY
      // operator projects all columns positionally).
      out_keys.push_back(output_table->view().column(order_key_idx[i]));
    }
    if (cudf::is_sorted(cudf::table_view(out_keys), column_order, null_precedence, stream)) {
      if (attempt > 0) {
        SIRIUS_LOG_WARN(
          "local_order_by: cudf::sorted_order returned an unsorted permutation; retry {} "
          "produced a correctly sorted batch ({} rows). See issue #1452.",
          attempt,
          output_table->num_rows());
      }
      break;
    }
    if (attempt + 1 >= max_sort_attempts) {
      throw std::runtime_error(
        "local_order_by: cudf::sorted_order returned an unsorted permutation and retries were "
        "exhausted (issue #1452)");
    }
    sorted_order = cudf::sorted_order(cudf::table_view(sort_cols),
                                      column_order,
                                      null_precedence,
                                      stream,
                                      memory_space.get_default_allocator());
    output_table = cudf::gather(cudf::table_view(project_input_cols),
                                sorted_order->view(),
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                stream,
                                memory_space.get_default_allocator());
  }

  // Create the output data batch
  return make_data_batch(std::move(output_table), memory_space, stream, telemetry_info);
}

}  // namespace op
}  // namespace sirius
