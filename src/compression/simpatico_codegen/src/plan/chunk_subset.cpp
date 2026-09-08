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
