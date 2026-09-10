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

#include "compressed_scan.hpp"
#include "compression/simpatico_compressed_representation.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <api/compressed_table_io.hpp>
#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simpatico {
class compressed_table;
}  // namespace simpatico

namespace sirius {

struct compressed_device_blob;  // defined in device_compressed_blob.hpp

/// A Simpatico-compressed chunk resident in pinned host memory.
///
/// The (small) structural header is a flat byte vector; the (large) payload —
/// every compressed leaf buffer, concatenated — lives in a cuCascade pinned
/// multi-block allocation drawn from the host tier's fixed_size_host_memory_resource
/// (the same pool the uncompressed pin path uses, so both share one tracked host
/// budget). Reconstruction re-parses the header (a compact binary node array —
/// cheap) and copies the payload straight back to the GPU.
///
/// Shared (via shared_ptr) among all representations that alias the same chunk
/// (e.g. after select_columns() or clone()); the pinned blocks are returned to
/// the pool when the last owner drops.
struct pinned_compressed_blob {
  std::vector<std::uint8_t> header;
  cucascade::memory::fixed_size_host_memory_resource::fixed_multiple_blocks_allocation payload;
  std::uint64_t payload_bytes = 0;
};

/// Per-column byte footprints recorded at pin time, one entry per column of the
/// chunk, so a projection can report exact sizes instead of scaling the totals.
struct per_column_byte_sizes {
  std::vector<std::size_t> compressed;
  std::vector<std::size_t> uncompressed;
};

// ── Block-aware copies over a cuCascade multi-block pinned allocation ─────────
//
// A single compressed buffer may straddle fixed-size block boundaries, so copies
// to/from the pinned payload are chunked at those boundaries (mirroring
// cuCascade's own host_data_representation::copy_between_blocks).

/// Copy @p size bytes from device @p src_device into the pinned payload at
/// logical byte offset @p dst_offset, enqueued on @p stream (device→host).
void copy_device_to_pinned_blocks(
  const void* src_device,
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& dst,
  std::uint64_t dst_offset,
  std::size_t size,
  rmm::cuda_stream_view stream);

/// Copy @p size bytes from the pinned payload at logical byte offset @p src_offset
/// into device @p dst_device, enqueued on @p stream (host→device).
void copy_pinned_blocks_to_device(
  const cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& src,
  std::uint64_t src_offset,
  void* dst_device,
  std::size_t size,
  rmm::cuda_stream_view stream);

/// Copy @p size bytes from the pinned payload at logical byte offset @p src_offset
/// into host memory at @p dst_host (host→host, synchronous). Used to read the small
/// per-chunk metadata buffers that size a chunk subset; see
/// simpatico::build_chunk_subset_header.
void copy_pinned_blocks_to_host(
  const cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& src,
  std::uint64_t src_offset,
  void* dst_host,
  std::size_t size);

/// Serve a COMPACTED payload out of the original pinned one.
///
/// The reader asks for byte ranges of the payload a synthesized subset header describes; @p gather
/// says where each of those bytes lives in the ORIGINAL payload. Ranges are ascending and
/// non-overlapping in destination order, so answering a request is a walk from the first range
/// that reaches into it.
///
/// Destination bytes no gather range covers are real and deliberate: a bitpack `packed` buffer
/// declares decode guard words past its last live word (simpatico::kBitpackDecodeGuardWords), and
/// the decode loads them unconditionally without their values reaching an output row. They are
/// zeroed rather than left undefined so a decode never reads uninitialized device memory.
///
/// Shared by every path that serves a chunk subset -- a pinned entry's converter and the .hpln
/// scan source -- because the zero-fill rule above is not obvious and a second copy of it would
/// diverge silently: a missing memset is uninitialized device memory, not a fault.
class gathered_payload_fetch {
 public:
  gathered_payload_fetch(
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation const& payload,
    std::vector<simpatico::gather_range> gather)
    : _payload(payload), _gather(std::move(gather))
  {
  }

  void operator()(std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) const;

 private:
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation const& _payload;
  std::vector<simpatico::gather_range> _gather;
};

/**
 * @brief HOST-tier idata_representation backed by a pinned Simpatico-compressed chunk.
 *
 * Holds a shared @ref pinned_compressed_blob plus schema metadata and an optional
 * column projection. The converter compressed_host_representation →
 * gpu_table_representation rebuilds the compressed_table from the blob
 * (read_compressed_table_from_memory), projects to the selected columns (if any),
 * then decompresses to a cudf::table. It is registered by
 * register_compression_converters().
 *
 * Multiple compressed_host_representation objects may share the same underlying
 * blob (e.g. after select_columns() or clone()).
 */
class compressed_host_representation : public simpatico_compressed_representation {
 public:
  /**
   * @brief Construct a compressed_host_representation owning a share of @p blob.
   *
   * @param memory_space        Host memory space this data is logically associated with.
   * @param blob                Pinned compressed chunk (header + payload).
   * @param column_names        All column names stored in the chunk, in column order.
   * @param compressed_bytes    Compressed footprint in bytes.
   * @param uncompressed_bytes  Original device footprint of the chunk
   *                            (cudf::table::alloc_size: data + null masks +
   *                            padding + string offsets/chars).
   * @param num_rows            Row count.
   * @param column_sizes        Optional per-column footprints; when present,
   *                            select_columns() sums the selected entries.
   */
  compressed_host_representation(
    cucascade::memory::memory_space& memory_space,
    std::shared_ptr<pinned_compressed_blob> blob,
    std::vector<std::string> column_names,
    std::size_t compressed_bytes,
    std::size_t uncompressed_bytes,
    std::int64_t num_rows,
    std::shared_ptr<const per_column_byte_sizes> column_sizes = nullptr);

  ~compressed_host_representation() override = default;

  // Non-copyable (shared ownership uses shared_ptr)
  compressed_host_representation(const compressed_host_representation&)            = delete;
  compressed_host_representation& operator=(const compressed_host_representation&) = delete;
  compressed_host_representation(compressed_host_representation&&)                 = delete;
  compressed_host_representation& operator=(compressed_host_representation&&)      = delete;

  // ── idata_representation interface ──────────────────────────────────────────

  /// Returns the compressed (payload) size.
  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _compressed_bytes; }

  /// Returns the logical (uncompressed) data size.
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override
  {
    return _uncompressed_bytes;
  }

  /// Clone shares the same backing blob (increments shared ownership).
  [[nodiscard]] std::unique_ptr<cucascade::idata_representation> clone(
    rmm::cuda_stream_view stream) override;

  // ── Projection ──────────────────────────────────────────────────────────────

  /**
   * @brief Return a projection that exposes only the requested column indices.
   *
   * The returned representation shares the same backing blob. The converter
   * will reconstruct all columns but decompress only the selected subset.
   *
   * @param indices  Indices into column_names() to expose (must be valid).
   */
  [[nodiscard]] std::unique_ptr<compressed_host_representation> select_columns(
    std::span<const std::size_t> indices) const;

  // ── Accessors ───────────────────────────────────────────────────────────────

  /// The structural header bytes (fed to read_compressed_table_from_memory).
  [[nodiscard]] std::span<const std::uint8_t> header() const noexcept { return _blob->header; }

  /// The pinned payload holding every compressed leaf buffer, concatenated.
  [[nodiscard]] const cucascade::memory::fixed_size_host_memory_resource::
    multiple_blocks_allocation&
    payload() const noexcept
  {
    return *_blob->payload;
  }

  [[nodiscard]] const std::vector<std::string>& column_names() const noexcept
  {
    return _column_names;
  }
  [[nodiscard]] std::int64_t num_rows() const noexcept { return _num_rows; }

  /// Column indices to project during decompression (nullopt = all columns).
  [[nodiscard]] const std::optional<std::vector<std::size_t>>& selected_indices() const noexcept
  {
    return _selected_indices;
  }

  /// Attach the scan whose filter this projection decodes under.
  ///
  /// Call only on a freshly projected representation the caller owns outright
  /// (as @ref select_columns returns): the scan's filter is a property of one
  /// query, never of the shared pinned chunk. One carrier for the whole
  /// request, so a clone copies a pointer and nothing can be forgotten.
  void set_pushdown_scan(std::shared_ptr<const decompression_pushdown_scan> scan)
  {
    _pushdown_scan = std::move(scan);
  }

  /// The attached scan, or null when the columns decompress unfiltered.
  [[nodiscard]] std::shared_ptr<const decompression_pushdown_scan> const& pushdown_scan()
    const noexcept
  {
    return _pushdown_scan;
  }

  /// Restrict this projection to a subset of the chunk's 1024-row simpatico decode chunks.
  ///
  /// @p chunks are batch-local decode-chunk ids, strictly ascending — the chunks a zone-map pass
  /// could not rule out. The converter then synthesizes a header describing only those chunks and
  /// fetches only their compressed bytes, so the pruned rows cost neither transfer nor decode; see
  /// simpatico::build_chunk_subset_header. A column the format cannot address that finely is
  /// served whole, which is always correct.
  ///
  /// @p rows is how many rows those chunks hold — what the served batch will contain. The
  /// representation's reported row count and byte footprints are scaled to it, so a reservation
  /// sized off this representation matches what the decode actually produces.
  ///
  /// Call only on a freshly projected representation the caller owns outright (as
  /// @ref select_columns returns): which chunks survive is a property of one query's filter,
  /// never of the shared pinned chunk.
  void set_surviving_chunks(std::vector<std::uint32_t> chunks, std::int64_t rows);

  /// The surviving decode chunks, or empty when every chunk is served.
  [[nodiscard]] std::span<const std::uint32_t> surviving_chunks() const noexcept
  {
    return _surviving_chunks ? std::span<const std::uint32_t>{*_surviving_chunks}
                             : std::span<const std::uint32_t>{};
  }

 private:
  /// Construct a projection sharing the same backing blob.
  compressed_host_representation(cucascade::memory::memory_space& memory_space,
                                 std::shared_ptr<pinned_compressed_blob> blob,
                                 std::vector<std::string> column_names,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows,
                                 std::optional<std::vector<std::size_t>> selected_indices,
                                 std::shared_ptr<const per_column_byte_sizes> column_sizes);

  std::shared_ptr<pinned_compressed_blob> _blob;
  std::vector<std::string> _column_names;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
  std::optional<std::vector<std::size_t>> _selected_indices;
  std::shared_ptr<const decompression_pushdown_scan> _pushdown_scan;
  std::shared_ptr<const per_column_byte_sizes> _column_sizes;
  /// Surviving 1024-row decode chunks; null when the whole chunk is served. Shared (never
  /// mutated after being set) so a clone copies a pointer.
  std::shared_ptr<const std::vector<std::uint32_t>> _surviving_chunks;
};

/// A Simpatico-compressed chunk resident in GPU (device) memory.
///
/// The device analog of @ref pinned_compressed_blob: the structural header is a
/// flat host byte vector, while the payload — every compressed leaf buffer,
/// concatenated — lives in a single contiguous rmm::device_buffer (device
/// allocations are contiguous, so no multi-block handling is needed). Pinning a
/// table to the GPU tier compressed keeps its device footprint small; the data
/// is decompressed on demand when a query materializes it.
///
/**
 * @brief GPU-tier idata_representation backed by a cached compressed_device_blob.
 *
 * Holds a shared compressed_device_blob: a single contiguous device payload buffer
 * plus a simpatico::compressed_table whose leaf channels_ are non-owning slices of
 * that payload (placed there at pin time via slab_memory_resource, no per-query copy).
 * The converter compressed_device_representation → gpu_table_representation calls
 * simpatico::decompress() directly on the cached table, decompressing only the selected
 * columns when a projection is set.
 */
class compressed_device_representation : public simpatico_compressed_representation {
 public:
  compressed_device_representation(
    cucascade::memory::memory_space& memory_space,
    std::shared_ptr<compressed_device_blob> blob,
    std::vector<std::string> column_names,
    std::size_t compressed_bytes,
    std::size_t uncompressed_bytes,
    std::int64_t num_rows,
    std::shared_ptr<const per_column_byte_sizes> column_sizes = nullptr);

  ~compressed_device_representation() override = default;

  compressed_device_representation(const compressed_device_representation&)            = delete;
  compressed_device_representation& operator=(const compressed_device_representation&) = delete;
  compressed_device_representation(compressed_device_representation&&)                 = delete;
  compressed_device_representation& operator=(compressed_device_representation&&)      = delete;

  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _compressed_bytes; }

  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override
  {
    return _uncompressed_bytes;
  }

  /// Clone shares the same cached table (increments shared ownership).
  [[nodiscard]] std::unique_ptr<cucascade::idata_representation> clone(
    rmm::cuda_stream_view stream) override;

  /// Projection sharing the same cached blob; decompress will skip non-selected columns.
  [[nodiscard]] std::unique_ptr<compressed_device_representation> select_columns(
    std::span<const std::size_t> indices) const;

  /// Whether @ref table is readable. A chunk may legitimately carry no blob —
  /// serving paths that need only the row count or a column projection never
  /// touch one — so anything that DOES read the table must ask first.
  [[nodiscard]] bool has_table() const noexcept;
  /// The cached compressed_table (defined in device_compressed_blob.hpp).
  [[nodiscard]] const simpatico::compressed_table& table() const noexcept;

  [[nodiscard]] const std::vector<std::string>& column_names() const noexcept
  {
    return _column_names;
  }
  [[nodiscard]] std::int64_t num_rows() const noexcept { return _num_rows; }

  [[nodiscard]] const std::optional<std::vector<std::size_t>>& selected_indices() const noexcept
  {
    return _selected_indices;
  }

  /// Attach the scan whose filter this projection decodes under.
  ///
  /// Call only on a freshly projected representation the caller owns outright
  /// (as @ref select_columns returns): the scan's filter is a property of one
  /// query, never of the shared pinned chunk. One carrier for the whole
  /// request, so a clone copies a pointer and nothing can be forgotten.
  void set_pushdown_scan(std::shared_ptr<const decompression_pushdown_scan> scan)
  {
    _pushdown_scan = std::move(scan);
  }

  /// The attached scan, or null when the columns decompress unfiltered.
  [[nodiscard]] std::shared_ptr<const decompression_pushdown_scan> const& pushdown_scan()
    const noexcept
  {
    return _pushdown_scan;
  }

 private:
  compressed_device_representation(cucascade::memory::memory_space& memory_space,
                                   std::shared_ptr<compressed_device_blob> blob,
                                   std::vector<std::string> column_names,
                                   std::size_t compressed_bytes,
                                   std::size_t uncompressed_bytes,
                                   std::int64_t num_rows,
                                   std::optional<std::vector<std::size_t>> selected_indices,
                                   std::shared_ptr<const per_column_byte_sizes> column_sizes);

  std::shared_ptr<compressed_device_blob> _blob;
  std::vector<std::string> _column_names;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
  std::optional<std::vector<std::size_t>> _selected_indices;
  std::shared_ptr<const decompression_pushdown_scan> _pushdown_scan;
  std::shared_ptr<const per_column_byte_sizes> _column_sizes;
};

}  // namespace sirius
