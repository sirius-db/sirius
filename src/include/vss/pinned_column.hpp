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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace cudf {
class column;
}  // namespace cudf
namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory
namespace sirius::scan_manager {
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace sirius::vss {

/**
 * @brief Concatenate a pinned column's GPU chunks into one contiguous column.
 *
 * Reads @p column_name from the GPU-resident @p pin (chunks are in pin /
 * index-build order) and concatenates them into a single offset-0, gap-free
 * column allocated on @p space. A single chunk is copied (so the result owns its
 * data) rather than viewed. Both the ANN index build and the ANN search use this
 * so the "same order the index was built in" invariant is enforced in one place.
 *
 * @throws internal_exception if @p column_name is absent from the pinned entry,
 *         or if any chunk resides on a different GPU than @p space (multi-GPU
 *         pinned tables are not supported yet).
 */
std::unique_ptr<cudf::column> concat_pinned_column(const scan_manager::pinned_entry& pin,
                                                   const std::string& column_name,
                                                   cucascade::memory::memory_space& space,
                                                   rmm::cuda_stream_view stream);

/// As above, but allocate the result through @p mr (e.g. a reservation's memory
/// resource) instead of @p space's default allocator. @p space is still used for
/// the multi-GPU placement check.
std::unique_ptr<cudf::column> concat_pinned_column(const scan_manager::pinned_entry& pin,
                                                   const std::string& column_name,
                                                   cucascade::memory::memory_space& space,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr);

/// Sum of the pinned column's chunk allocation sizes.
[[nodiscard]] std::size_t pinned_column_alloc_size(const scan_manager::pinned_entry& pin,
                                                   const std::string& column_name);

}  // namespace sirius::vss
