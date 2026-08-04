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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include "decode_pushdown.hpp"

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

/// Per-selected-column equality/IN pushdown for decompression.
///
/// Parallel to a representation's selected column list, so entry @c i describes
/// the @c i-th column the converter will decompress. Entry @c i holds the string
/// values that column is tested against; an empty entry decompresses normally.
///
/// A column with a non-empty entry comes back as a **BOOL8** column
/// (`value ∈ values`, nulls propagated) rather than its declared type: a
/// dictionary-compressed column answers the predicate from its key set and never
/// gathers the decoded chars (see @c simpatico::decode_predicate). Consumers
/// must therefore expect the type substitution — @c parquet_gpu_ingestible
/// rewrites its filter expression to a bare boolean reference when it sees one.
///
/// Held as plain strings rather than @c simpatico::decode_predicate so this
/// header stays free of the simpatico API (matching the forward-declared
/// @c compressed_table above); @c compression_converters.cpp converts at the
/// call boundary.
using decode_equality_pushdown = std::vector<std::vector<std::string>>;

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
class compressed_host_representation : public cucascade::idata_representation {
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
   */
  compressed_host_representation(cucascade::memory::memory_space& memory_space,
                                 std::shared_ptr<pinned_compressed_blob> blob,
                                 std::vector<std::string> column_names,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows);

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

  /// Attach an equality/IN pushdown, parallel to the selected column list.
  ///
  /// Call only on a freshly projected representation the caller owns outright
  /// (as @ref select_columns returns): the pushdown is a property of one scan's
  /// filter, never of the shared pinned chunk.
  void set_equality_pushdown(decode_equality_pushdown pushdown)
  {
    _equality_pushdown = std::move(pushdown);
  }

  /// The attached pushdown; empty when the columns decompress normally.
  [[nodiscard]] const decode_equality_pushdown& equality_pushdown() const noexcept
  {
    return _equality_pushdown;
  }

  /// Attach the scan's numeric-range pushdown (fused scan-filter pipeline),
  /// parallel to the selected column list; same freshly-projected-only
  /// ownership rule as the equality pushdown above.
  void set_range_pushdown(decode_range_pushdown pushdown, bool all_conjuncts_convertible)
  {
    _range_pushdown              = std::move(pushdown);
    _range_conjuncts_convertible = all_conjuncts_convertible;
  }

  /// The attached range pushdown; empty when no decode-time range applies.
  [[nodiscard]] const decode_range_pushdown& range_pushdown() const noexcept
  {
    return _range_pushdown;
  }

  /// True iff the attached ranges are the scan's WHOLE row-restricting filter —
  /// the iteration-1 gate for decode-side compaction (rows may only be dropped
  /// during decode when nothing else still filters the batch).
  [[nodiscard]] bool range_conjuncts_convertible() const noexcept
  {
    return _range_conjuncts_convertible;
  }

  /// Attach dynamic-membership probes (Phase A), parallel to the selected
  /// column list; same freshly-projected ownership rule. @p generation is the
  /// producing channel's monotonic filter_count() at snapshot time — the wave
  /// orchestrator echoes it so bail latches can be generation-keyed.
  void set_membership_pushdown(decode_membership_pushdown pushdown, std::uint64_t generation)
  {
    _membership_pushdown  = std::move(pushdown);
    _membership_generation = generation;
  }

  [[nodiscard]] const decode_membership_pushdown& membership_pushdown() const noexcept
  {
    return _membership_pushdown;
  }

  [[nodiscard]] std::uint64_t membership_generation() const noexcept
  {
    return _membership_generation;
  }

  /// Late-materialization capture request (SIRIUS_EXP_LATE_MAT, additive):
  /// when set on a freshly projected per-query representation (same ownership
  /// rule as the pushdown setters — NEVER the shared pinned chunk), a fused
  /// scan-filter decode that APPLIES the whole conjunction (row_filtered tag
  /// only, never a RULE-2 bail or partial coverage) moves its wave-1
  /// selection buffers onto the produced row_filtered_gpu_table_representation
  /// (captured_selection) instead of freeing them. Default false; the flag is
  /// a plain bool and adds no behavior anywhere until the converter reads it.
  void request_selection_capture() noexcept { _selection_capture_requested = true; }

  [[nodiscard]] bool selection_capture_requested() const noexcept
  {
    return _selection_capture_requested;
  }


 private:
  /// Construct a projection sharing the same backing blob.
  compressed_host_representation(cucascade::memory::memory_space& memory_space,
                                 std::shared_ptr<pinned_compressed_blob> blob,
                                 std::vector<std::string> column_names,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows,
                                 std::optional<std::vector<std::size_t>> selected_indices);

  std::shared_ptr<pinned_compressed_blob> _blob;
  std::vector<std::string> _column_names;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
  std::optional<std::vector<std::size_t>> _selected_indices;
  decode_equality_pushdown _equality_pushdown;
  decode_range_pushdown _range_pushdown;
  bool _range_conjuncts_convertible = false;
  decode_membership_pushdown _membership_pushdown;
  std::uint64_t _membership_generation = 0;
  /// Late-mat capture request (see request_selection_capture()).
  bool _selection_capture_requested = false;
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
class compressed_device_representation : public cucascade::idata_representation {
 public:
  compressed_device_representation(cucascade::memory::memory_space& memory_space,
                                   std::shared_ptr<compressed_device_blob> blob,
                                   std::vector<std::string> column_names,
                                   std::size_t compressed_bytes,
                                   std::size_t uncompressed_bytes,
                                   std::int64_t num_rows);

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

  /// Attach an equality/IN pushdown, parallel to the selected column list.
  ///
  /// Call only on a freshly projected representation the caller owns outright
  /// (as @ref select_columns returns): the pushdown is a property of one scan's
  /// filter, never of the shared pinned chunk.
  void set_equality_pushdown(decode_equality_pushdown pushdown)
  {
    _equality_pushdown = std::move(pushdown);
  }

  /// The attached pushdown; empty when the columns decompress normally.
  [[nodiscard]] const decode_equality_pushdown& equality_pushdown() const noexcept
  {
    return _equality_pushdown;
  }

  /// Attach the scan's numeric-range pushdown (fused scan-filter pipeline),
  /// parallel to the selected column list; same freshly-projected-only
  /// ownership rule as the equality pushdown above.
  void set_range_pushdown(decode_range_pushdown pushdown, bool all_conjuncts_convertible)
  {
    _range_pushdown              = std::move(pushdown);
    _range_conjuncts_convertible = all_conjuncts_convertible;
  }

  /// The attached range pushdown; empty when no decode-time range applies.
  [[nodiscard]] const decode_range_pushdown& range_pushdown() const noexcept
  {
    return _range_pushdown;
  }

  /// True iff the attached ranges are the scan's WHOLE row-restricting filter —
  /// the iteration-1 gate for decode-side compaction (rows may only be dropped
  /// during decode when nothing else still filters the batch).
  [[nodiscard]] bool range_conjuncts_convertible() const noexcept
  {
    return _range_conjuncts_convertible;
  }

  /// Attach dynamic-membership probes (Phase A), parallel to the selected
  /// column list; same freshly-projected ownership rule. @p generation is the
  /// producing channel's monotonic filter_count() at snapshot time — the wave
  /// orchestrator echoes it so bail latches can be generation-keyed.
  void set_membership_pushdown(decode_membership_pushdown pushdown, std::uint64_t generation)
  {
    _membership_pushdown  = std::move(pushdown);
    _membership_generation = generation;
  }

  [[nodiscard]] const decode_membership_pushdown& membership_pushdown() const noexcept
  {
    return _membership_pushdown;
  }

  [[nodiscard]] std::uint64_t membership_generation() const noexcept
  {
    return _membership_generation;
  }

  /// Late-materialization capture request (SIRIUS_EXP_LATE_MAT, additive):
  /// when set on a freshly projected per-query representation (same ownership
  /// rule as the pushdown setters — NEVER the shared pinned chunk), a fused
  /// scan-filter decode that APPLIES the whole conjunction (row_filtered tag
  /// only, never a RULE-2 bail or partial coverage) moves its wave-1
  /// selection buffers onto the produced row_filtered_gpu_table_representation
  /// (captured_selection) instead of freeing them. Default false; the flag is
  /// a plain bool and adds no behavior anywhere until the converter reads it.
  void request_selection_capture() noexcept { _selection_capture_requested = true; }

  [[nodiscard]] bool selection_capture_requested() const noexcept
  {
    return _selection_capture_requested;
  }


  /// Reservation-time fused scan-filter probe (host metadata only — never
  /// touches the device).
  struct fused_scan_reservation_probe {
    /// The fused pipeline is expected to compact this projection at decode:
    /// env gate on, at least one attached ACTIVE range, the ranges cover the
    /// whole filter, and every selected column's plan probes
    /// survivor-compactable (bitpack tier_a, delta tier_a_delta, or
    /// dict-K5) — the reservation-time mirror of the converter's RULE 1.
    bool planned = false;
    /// True when the batch's survivor count is capped by the RULE-2
    /// selectivity ceiling — i.e. NO selected column classifies tier_dict_k5
    /// (dict-K5 batches skip the RULE-2 bail: the masked key gather wins at
    /// every selectivity). When false, size survivor-dependent reservations
    /// for up to full-width survivors.
    bool rule2_bounded = false;
  };

  /// See @ref fused_scan_reservation_probe. RULE 2 itself (the post-CNT bail)
  /// is data-dependent and invisible here; callers scaling reservations by
  /// the RULE-2 ceiling accept the rare bail-path over-allocation.
  [[nodiscard]] fused_scan_reservation_probe probe_fused_scan_reservation() const;

 private:
  compressed_device_representation(cucascade::memory::memory_space& memory_space,
                                   std::shared_ptr<compressed_device_blob> blob,
                                   std::vector<std::string> column_names,
                                   std::size_t compressed_bytes,
                                   std::size_t uncompressed_bytes,
                                   std::int64_t num_rows,
                                   std::optional<std::vector<std::size_t>> selected_indices);

  std::shared_ptr<compressed_device_blob> _blob;
  std::vector<std::string> _column_names;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
  std::optional<std::vector<std::size_t>> _selected_indices;
  decode_equality_pushdown _equality_pushdown;
  decode_range_pushdown _range_pushdown;
  bool _range_conjuncts_convertible = false;
  decode_membership_pushdown _membership_pushdown;
  std::uint64_t _membership_generation = 0;
  /// Late-mat capture request (see request_selection_capture()).
  bool _selection_capture_requested = false;
};

}  // namespace sirius
