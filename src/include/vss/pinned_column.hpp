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
#include <vector>

namespace cudf {
class column;
class column_view;
}  // namespace cudf
namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory
namespace sirius::scan_manager {
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace sirius::vss {

/// Return the pinned column's GPU chunks as column_views in pin order, without
/// concatenating them. Every chunk must live on @p space's device. The views
/// borrow the pinned chunks, so @p pin must outlive their use.
///
/// @throws internal_exception if @p column_name is absent, or any chunk resides
///         on a different GPU than @p space.
[[nodiscard]] std::vector<cudf::column_view> pinned_column_chunk_views(
  const scan_manager::pinned_entry& pin,
  const std::string& column_name,
  cucascade::memory::memory_space& space);

}  // namespace sirius::vss
