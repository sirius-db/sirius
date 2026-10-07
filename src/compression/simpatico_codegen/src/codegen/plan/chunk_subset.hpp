// SPDX-License-Identifier: Apache-2.0
//
// Turning "these 1024-row chunks survive" into "these bytes to move".
//
// Serving a subset of a compressed column's chunks is only sound if the result is still a VALID
// column: the decode must see per-chunk metadata whose entries correspond one-to-one with the bulk
// bytes it was given. That works because `bp_offsets` -- where each chunk's bits start inside
// `packed` -- is not stored; it is scanned at decode time from the `chunk_count`/`chunk_bits` that
// were loaded (src/bridge/offsets_cumsum.cu). Hand the decode a compacted metadata array plus the
// matching packed bytes and an ordinary full decode yields exactly the surviving rows.
//
// Everything here is pure arithmetic over host-side metadata, so it is testable without a GPU and
// without I/O. Which operators may be subsetted at all is a separate question, answered by
// simpatico::supports_chunk_subset (codegen/plan/operator_registry.hpp).

#pragma once

#include "codegen/plan/operator_registry.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace simpatico {

/// A half-open byte range within one buffer.
struct byte_range {
  std::uint64_t offset{0};
  std::uint64_t size{0};

  [[nodiscard]] std::uint64_t end() const noexcept { return offset + size; }
  [[nodiscard]] bool operator==(byte_range const&) const = default;
};

/// What to move, and how big the result is.
struct buffer_subset {
  /// Ranges into the ORIGINAL buffer, ascending and coalesced, to be concatenated in order.
  std::vector<byte_range> ranges;
  /// Bytes the compacted buffer occupies. Not always the sum of `ranges` -- a bitpack `packed`
  /// buffer also carries decode guard words past its last live word.
  std::uint64_t compacted_size{0};

  [[nodiscard]] std::uint64_t moved_bytes() const noexcept
  {
    std::uint64_t n = 0;
    for (auto const& r : ranges) {
      n += r.size;
    }
    return n;
  }
};

/// Append @p r to @p out, merging into the previous range when they touch or overlap, and
/// bridging a gap of up to @p max_gap_bytes.
///
/// Bridging deliberately moves bytes that were pruned. Over a network a separate request costs far
/// more than a few unused KB, and even locally a copy has per-call overhead, so the caller sets
/// the threshold from what its transport charges per request. 0 merges only adjacent ranges.
void append_coalesced(std::vector<byte_range>& out, byte_range r, std::uint64_t max_gap_bytes = 0);

/// Bytes to move for a fixed-stride per-chunk metadata buffer (bitpack's chunk_min / chunk_count /
/// chunk_bits, for's references, delta's delta_first): entry c lives at c * @p elem_size.
///
/// @p surviving_chunks must be ascending and within [0, @p n_chunks).
[[nodiscard]] buffer_subset plan_metadata_subset(std::size_t elem_size,
                                                 std::size_t n_chunks,
                                                 std::span<std::uint32_t const> surviving_chunks,
                                                 std::uint64_t max_gap_bytes = 0);

/// Bytes to move for a `bulk_fixed_stride` buffer: chunk c starts at (rows before c) * @p
/// elem_size and runs for its own row count, so nothing operator-specific is needed. The final
/// chunk of a column is short.
[[nodiscard]] buffer_subset plan_fixed_stride_subset(
  std::size_t elem_size,
  std::uint64_t total_rows,
  std::uint64_t rows_per_chunk,
  std::span<std::uint32_t const> surviving_chunks,
  std::uint64_t max_gap_bytes = 0);

/// The per-chunk metadata a `bulk_variable` buffer needs to size its chunks. Only bitpack has one
/// today; the fields are named for it because there is nothing to generalise over yet.
struct chunk_sizing_metadata {
  std::span<std::int32_t const> chunk_count;
  std::span<std::uint8_t const> chunk_bits;
};

/// Bytes to move for @p buffer of a leaf of type @p kind, dispatching on the buffer's registered
/// layout (simpatico::buffer_layout).
///
/// This is the entry point a caller should use: it keeps operator-specific arithmetic out of the
/// caller, so nothing above this line names an operator. `per_chunk_metadata` and
/// `bulk_fixed_stride` are handled generically; `bulk_variable` is the only case that consults
/// @p sizing, and the only case an operator has to be taught about.
///
/// Returns nullopt when the buffer cannot be subsetted -- an unclassified operator, a
/// `whole_column` buffer, or a `bulk_variable` buffer whose sizing metadata was not supplied. The
/// caller must then fetch that buffer whole, which is always correct.
[[nodiscard]] std::optional<buffer_subset> plan_buffer_subset(
  OpId kind,
  std::string_view buffer,
  std::size_t elem_size,
  std::uint64_t total_rows,
  std::uint64_t rows_per_chunk,
  std::span<std::uint32_t const> surviving_chunks,
  chunk_sizing_metadata const& sizing = {},
  std::uint64_t max_gap_bytes         = 0);

/// Bytes to move for a bitpack `packed` buffer.
///
/// Chunk c occupies `n_words[c] = (chunk_count[c] * chunk_bits[c] + 31) / 32` words starting at the
/// exclusive prefix sum -- the same derivation simpatico_compute_bp_offsets_scan performs on
/// device, so the compacted buffer's own scan reproduces these offsets relative to its own start.
///
/// The compacted size carries the decode's guard words: simpatico_bitunpack_one loads
/// packed[w .. w+2] unconditionally, so the last live word must be readable two words past its end.
[[nodiscard]] buffer_subset plan_bitpack_packed_subset(
  std::span<std::int32_t const> chunk_count,
  std::span<std::uint8_t const> chunk_bits,
  std::span<std::uint32_t const> surviving_chunks,
  std::uint64_t max_gap_bytes = 0);

/// Words the decode reads past the last live word of a `packed` buffer; see
/// plan_bitpack_packed_subset and tests/test_bitpack_layout_contract.cpp.
inline constexpr std::uint64_t kBitpackDecodeGuardWords = 3;

}  // namespace simpatico
