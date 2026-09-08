// SPDX-License-Identifier: Apache-2.0

#include "codegen/plan/chunk_subset.hpp"

#include <algorithm>

namespace simpatico {

void append_coalesced(std::vector<byte_range>& out, byte_range r, std::uint64_t max_gap_bytes)
{
  if (r.size == 0) { return; }
  if (!out.empty()) {
    auto& last = out.back();
    // Ranges are produced in ascending order, so only the previous one can touch this.
    if (r.offset <= last.end() + max_gap_bytes) {
      last.size = std::max(last.end(), r.end()) - last.offset;
      return;
    }
  }
  out.push_back(r);
}

buffer_subset plan_metadata_subset(std::size_t elem_size,
                                   std::size_t n_chunks,
                                   std::span<std::uint32_t const> surviving_chunks,
                                   std::uint64_t max_gap_bytes)
{
  buffer_subset out;
  if (elem_size == 0) { return out; }
  for (auto const chunk : surviving_chunks) {
    if (chunk >= n_chunks) { continue; }  // defensive: a stale survivor cannot address bytes
    append_coalesced(
      out.ranges, {static_cast<std::uint64_t>(chunk) * elem_size, elem_size}, max_gap_bytes);
    out.compacted_size += elem_size;
  }
  return out;
}

buffer_subset plan_fixed_stride_subset(std::size_t elem_size,
                                       std::uint64_t total_rows,
                                       std::uint64_t rows_per_chunk,
                                       std::span<std::uint32_t const> surviving_chunks,
                                       std::uint64_t max_gap_bytes)
{
  buffer_subset out;
  if (elem_size == 0 || rows_per_chunk == 0) { return out; }
  auto const n_chunks = (total_rows + rows_per_chunk - 1) / rows_per_chunk;
  for (auto const chunk : surviving_chunks) {
    if (chunk >= n_chunks) { continue; }
    auto const first = static_cast<std::uint64_t>(chunk) * rows_per_chunk;
    // The last chunk of a column is short.
    auto const rows = std::min(rows_per_chunk, total_rows - first);
    append_coalesced(out.ranges, {first * elem_size, rows * elem_size}, max_gap_bytes);
    out.compacted_size += rows * elem_size;
  }
  return out;
}

std::optional<buffer_subset> plan_buffer_subset(OpId kind,
                                                std::string_view buffer,
                                                std::size_t elem_size,
                                                std::uint64_t total_rows,
                                                std::uint64_t rows_per_chunk,
                                                std::span<std::uint32_t const> surviving_chunks,
                                                chunk_sizing_metadata const& sizing,
                                                std::uint64_t max_gap_bytes)
{
  auto const layout = buffer_layout(kind, buffer);
  if (!layout) { return std::nullopt; }  // unclassified operator or unknown buffer: fetch whole
  auto const n_chunks =
    rows_per_chunk == 0 ? 0 : (total_rows + rows_per_chunk - 1) / rows_per_chunk;
  switch (*layout) {
    case ChannelLayout::per_chunk_metadata:
      return plan_metadata_subset(elem_size, n_chunks, surviving_chunks, max_gap_bytes);
    case ChannelLayout::bulk_fixed_stride:
      return plan_fixed_stride_subset(
        elem_size, total_rows, rows_per_chunk, surviving_chunks, max_gap_bytes);
    case ChannelLayout::bulk_variable:
      // The one case needing operator-specific arithmetic. Without its sizing metadata there is
      // no way to know where a chunk's bytes are, so refuse rather than guess.
      if (kind == OpId::Bitpack && !sizing.chunk_count.empty()) {
        return plan_bitpack_packed_subset(
          sizing.chunk_count, sizing.chunk_bits, surviving_chunks, max_gap_bytes);
      }
      return std::nullopt;
    case ChannelLayout::whole_column: return std::nullopt;
  }
  return std::nullopt;
}

buffer_subset plan_bitpack_packed_subset(std::span<std::int32_t const> chunk_count,
                                         std::span<std::uint8_t const> chunk_bits,
                                         std::span<std::uint32_t const> surviving_chunks,
                                         std::uint64_t max_gap_bytes)
{
  buffer_subset out;
  auto const n_chunks = std::min(chunk_count.size(), chunk_bits.size());
  if (n_chunks == 0) { return out; }

  auto const words_of = [&](std::size_t c) -> std::uint64_t {
    auto const bits =
      static_cast<std::uint64_t>(chunk_count[c]) * static_cast<std::uint64_t>(chunk_bits[c]);
    return (bits + 31) / 32;
  };

  // One forward pass: carry the prefix while walking the ascending survivor list, so this is
  // O(n_chunks) rather than a prefix sum per survivor.
  std::uint64_t prefix_words = 0;
  std::size_t next           = 0;
  std::uint64_t live_words   = 0;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    auto const w = words_of(c);
    if (next < surviving_chunks.size() && surviving_chunks[next] == c) {
      // A zero-width chunk (bits == 0, every value equal to chunk_min) contributes no bytes but
      // is still a surviving chunk: its metadata entry carries the whole answer.
      append_coalesced(out.ranges, {prefix_words * 4, w * 4}, max_gap_bytes);
      live_words += w;
      ++next;
    }
    prefix_words += w;
  }
  out.compacted_size = (live_words + kBitpackDecodeGuardWords) * 4;
  return out;
}

}  // namespace simpatico
