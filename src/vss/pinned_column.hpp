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

#include "vss/exact_rows.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

/// Who pays for the device copy of a HOST-tier chunk.
enum class staging_charge {
  /// A reservation of its own, held by the staged chunk: for copies that outlive a task or are
  /// made outside one.
  own_reservation,
  /// The reservation the calling task was admitted with: for a copy a task makes on its own
  /// stream and drops before it ends. A task that asked the space for a second reservation was
  /// refused whenever its siblings had reserved the rest, however small the chunk and however
  /// much of its own reservation was unused, and each retry doubled the task's reservation
  /// until nothing could grant it.
  task_reservation,
};

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
  const telemetry::batch_telemetry_info& telemetry_info,
  staging_charge charge = staging_charge::own_reservation);

/// A pinned FP32 vector column mapped for the device to read single rows in place: the tables
/// behind @ref exact_rows' pin form, and owning references to the chunks, so the rows stay readable
/// for as long as the map is held, whatever then happens to the pin.
struct pinned_rows_map {
  std::unique_ptr<rmm::device_buffer> chunk_rows, chunk_block, chunk_offset, blocks;
  /// The first row of each chunk, and the total after the last.
  std::vector<std::int64_t> chunk_first_row;
  std::int64_t n_chunks{0};
  int block_shift{0};
  std::vector<std::shared_ptr<void const>> owners;

  [[nodiscard]] std::int64_t n_rows() const noexcept { return chunk_first_row.back(); }
  /// Rows read through @p row_map (pin row of each row asked for), or by pin row when it is null.
  [[nodiscard]] exact_rows view(std::int64_t const* row_map = nullptr) const;
};

/// Maps the FP32 vector column @p column_name, @p dim wide, of either tier. nullopt when a chunk is
/// compressed, holds nulls, is not plain rows @p dim floats apart, or the host blocks are not a
/// power of two in size; such a column is staged instead.
[[nodiscard]] std::optional<pinned_rows_map> map_pinned_rows(
  const scan_manager::pinned_entry& pin,
  const std::string& column_name,
  std::int64_t dim,
  ::cucascade::memory::memory_space& gpu_space,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/// Rows start, start + stride, ... (count of them) of the column @p source reads, as a device LIST
/// column @p dim wide: a strided sample taken without staging the rest.
[[nodiscard]] std::unique_ptr<cudf::column> gather_pinned_rows(exact_rows const& source,
                                                               std::int64_t start,
                                                               std::int64_t stride,
                                                               std::int64_t count,
                                                               std::int64_t dim,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr);

/// Walks a pinned column a chunk at a time, copying the next HOST-tier chunk to the device while
/// the caller works on the current one.
///
/// The copy runs on a stream of its own; @ref take hands a chunk out bound to the caller's stream,
/// so its memory is freed behind the caller's work on it. A space that cannot grant a second
/// chunk's reservation gets the next chunk staged after the current one is released, as an
/// unprefetched walk would. GPU-tier chunks are aliased, so there is nothing to prefetch.
class pinned_chunk_walk {
 public:
  pinned_chunk_walk(const scan_manager::pinned_entry& pin,
                    std::string column_name,
                    ::cucascade::memory::memory_space& gpu_space,
                    rmm::cuda_stream_view stream,
                    const telemetry::batch_telemetry_info& telemetry_info);

  [[nodiscard]] std::size_t size() const noexcept { return _n_chunks; }

  /// Chunk @p chunk_index: the prefetched copy when there is one, else staged now on the caller's
  /// stream.
  [[nodiscard]] staged_pinned_chunk take(std::size_t chunk_index);

  /// Copies the chunk after @p chunk_index; call once the work on @p chunk_index is enqueued and
  /// before waiting for it. Returns when the copy is done.
  void prefetch_after(std::size_t chunk_index);

 private:
  const scan_manager::pinned_entry& _pin;
  std::string _column_name;
  ::cucascade::memory::memory_space& _gpu_space;
  rmm::cuda_stream_view _stream;
  const telemetry::batch_telemetry_info& _telemetry_info;
  std::size_t _n_chunks;
  rmm::cuda_stream _copy_stream;
  std::optional<staged_pinned_chunk> _ahead;
  std::size_t _ahead_index{0};
};

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
