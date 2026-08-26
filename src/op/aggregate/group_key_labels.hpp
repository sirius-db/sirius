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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace sirius::op::detail {

/**
 * @brief Dense labels and the sorted key rows indexed by those labels
 */
struct group_key_labels {
  std::unique_ptr<cudf::table> sorted_unique_keys;
  std::unique_ptr<cudf::column> labels;
};

/**
 * @brief Builds dense labels for group key rows
 *
 * Distinct key rows are ordered lexicographically with nulls after valid values. Each returned
 * label is the row index of the corresponding input key in `group_key_labels::sorted_unique_keys`.
 * Null keys compare equal, and all NaN values compare equal.
 *
 * @throw cudf::logic_error if `keys` has no columns
 * @throw std::runtime_error if an input row does not match its distinct key
 *
 * @param keys Group key columns
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for returned allocations
 * @return The sorted unique keys and one non-nullable INT32 label per input row
 */
[[nodiscard]] group_key_labels make_group_key_labels(cudf::table_view const& keys,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace sirius::op::detail
