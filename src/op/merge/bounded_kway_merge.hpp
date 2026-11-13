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

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sirius::op::merge {

/**
 * @brief The sort keys of a k-way merge. All three vectors have one entry per key, in key order.
 */
struct merge_key_spec {
  std::vector<cudf::size_type> key_cols;  ///< indices into the window tables (fetched columns)
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
};

/**
 * @brief Result of one bounded_kway_merge call.
 */
struct bounded_merge_result {
  std::unique_ptr<cudf::table> output;    ///< exactly `target` rows, all window columns
  std::vector<cudf::size_type> consumed;  ///< per window; Σ consumed == output->num_rows()
};

/**
 * @brief Bounded k-way merge of sorted windows.
 *
 * Emits the first `target` rows of the stable merged order of k sorted windows (ties broken by
 * window index, then row), where target = min(max_rows, r_safe, Σ len) and
 * r_safe = min over windows with truncated[i] != 0 of (rank of window i's last row in the
 * combined order) + 1 (∞ if none is truncated). A window is "truncated" when its source run has
 * unread rows beyond it. All windows share one schema; nested key types are not supported.
 * Never materializes k·max_rows rows. One or two D2H syncs per call.
 *
 * @param windows The k sorted windows; each holds all fetched columns (keys and payload).
 * @param keys The key columns (indices into the window tables), their order and null precedence.
 * @param truncated One flag per window: non-zero when the window's source run has unread rows
 *        beyond it.
 * @param max_rows Upper bound on the rows emitted.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the output table.
 * @return The merged output and the number of rows consumed from the front of each window.
 */
bounded_merge_result bounded_kway_merge(std::span<const cudf::table_view> windows,
                                        const merge_key_spec& keys,
                                        std::span<const std::uint8_t> truncated,
                                        cudf::size_type max_rows,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

}  // namespace sirius::op::merge
