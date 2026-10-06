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

#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

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
namespace cucascade {
class data_batch;
namespace memory {
class reservation;
}  // namespace memory
}  // namespace cucascade
namespace sirius::telemetry {
struct batch_telemetry_info;
}  // namespace sirius::telemetry

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

struct staged_pinned_column {
  std::vector<cudf::column_view> views;
  std::vector<std::shared_ptr<::cucascade::data_batch>> owners;
  std::vector<std::shared_ptr<::cucascade::memory::reservation>> reservations;
};

/// One chunk of @ref staged_pinned_column. @c owner and @c reservation are null for a GPU-tier
/// pin, where the view aliases the pinned chunk; for a HOST-tier pin they hold the staged copy
/// and free it when this struct is dropped.
struct staged_pinned_chunk {
  cudf::column_view view;
  std::shared_ptr<::cucascade::data_batch> owner;
  std::shared_ptr<::cucascade::memory::reservation> reservation;
};

/// Chunks the pinned column is stored in, on either tier.
/// Chunks the pinned column is stored in, on either tier.
[[nodiscard]] std::size_t pinned_column_chunk_count(const scan_manager::pinned_entry& pin,
                                                    const std::string& column_name);

/// Stage chunk @p chunk_index of a pinned column, whatever tier the pin lives on.
///
/// The per-chunk form of @ref stage_pinned_column: a caller that walks a column larger than
/// device memory stages one chunk, consumes it, and drops it before staging the next, so peak
/// device use is one chunk rather than the whole column.
[[nodiscard]] staged_pinned_chunk stage_pinned_column_chunk(
  const scan_manager::pinned_entry& pin,
  const std::string& column_name,
  std::size_t chunk_index,
  ::cucascade::memory::memory_space& gpu_space,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info);

/**
 * @brief Tier-agnostic replacement for @ref pinned_column_chunk_views.
 *
 * GPU-tier pins are aliased zero-copy as before; HOST-tier pins are staged to @p gpu_space.
 * Callers that must work against a corpus larger than device memory should stage a chunk at
 * a time instead of calling this, which stages every chunk at once.
 */
staged_pinned_column stage_pinned_column(const scan_manager::pinned_entry& pin,
                                         const std::string& column_name,
                                         ::cucascade::memory::memory_space& gpu_space,
                                         rmm::cuda_stream_view stream,
                                         const telemetry::batch_telemetry_info& telemetry_info);

/// The GPU memory space a pinned entry's chunks reside on (its first non-null
/// per-chunk space). The pinned_entry's top-level `memory_space` is not reliably
/// populated; the per-chunk spaces are. Throws if no chunk is GPU-resident.
[[nodiscard]] cucascade::memory::memory_space& pinned_entry_gpu_space(
  const scan_manager::pinned_entry& pin);

}  // namespace sirius::vss
