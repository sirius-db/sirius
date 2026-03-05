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

#include "op/scan/cached_ranges.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace sirius::op::scan {

cache_ranges::cache_ranges(std::vector<range> ranges,
                           std::vector<std::byte*> buffers,
                           std::size_t chunk_sz,
                           int device_id,
                           int numa_id)
  : ranges_(std::move(ranges)),
    buffers_(std::move(buffers)),
    chunk_size_(chunk_sz),
    device_id_(device_id),
    numa_id_(numa_id)
{
  assert(chunk_size_ > 0 && "chunk_size must be > 0");

  // Coalesce adjacent (head-to-tail) ranges.
  // Two ranges are neighbours when range[i].end == range[i+1].offset.
  // Merge them into a single range so that get_ranges() can satisfy large
  // contiguous reads with fewer look-ups and fewer copy operations.
  if (!ranges_.empty()) {
    std::vector<range> coalesced;
    coalesced.reserve(ranges_.size());
    coalesced.push_back(ranges_[0]);
    for (std::size_t i = 1; i < ranges_.size(); ++i) {
      auto& last = coalesced.back();
      auto const end =
        static_cast<std::size_t>(last.offset()) + static_cast<std::size_t>(last.size());
      if (end == static_cast<std::size_t>(ranges_[i].offset())) {
        // Extend the last coalesced range to swallow the neighbour.
        last = range{last.offset(), last.size() + ranges_[i].size()};
      } else {
        coalesced.push_back(ranges_[i]);
      }
    }
    ranges_ = std::move(coalesced);
  }

  // Pre-compute the byte offset inside the flat packed buffer at which
  // each range starts.  This lets get_ranges() translate a logical
  // address into a (chunk_index, intra-chunk offset) pair quickly.
  pack_offsets_.reserve(ranges_.size());
  std::size_t running = 0;
  for (auto const& r : ranges_) {
    pack_offsets_.push_back(running);
    running += static_cast<std::size_t>(r.size());
  }
  total_packed_bytes_ = running;
}

std::vector<std::span<const std::byte>> cache_ranges::get_ranges(std::size_t query_offset,
                                                                 std::size_t query_size) const
{
  if (query_size == 0) return {};

  std::vector<std::span<const std::byte>> result;

  std::size_t remaining  = query_size;
  std::size_t cur_offset = query_offset;

  while (remaining > 0) {
    // Find the range that contains cur_offset.
    auto idx = find_range(cur_offset);
    if (idx == ranges_.size()) {
      throw std::out_of_range("cache_ranges::get_ranges: offset not covered by any cached range");
    }

    auto const& r         = ranges_[idx];
    std::size_t range_end = static_cast<std::size_t>(r.offset() + r.size());

    if (cur_offset + remaining > range_end) {
      // The query spans beyond a single cached range – we only
      // support queries that are fully covered by the cache.
      throw std::out_of_range("cache_ranges::get_ranges: query extends beyond a cached range");
    }

    // How many bytes into range r is cur_offset?
    std::size_t range_local = cur_offset - static_cast<std::size_t>(r.offset());

    // Flat packed byte position for this byte.
    std::size_t pack_pos = pack_offsets_[idx] + range_local;

    // Emit spans chunk by chunk until we have satisfied `remaining`.
    while (remaining > 0) {
      std::size_t chunk_idx   = pack_pos / chunk_size_;
      std::size_t chunk_local = pack_pos % chunk_size_;

      // How many bytes are available in this chunk from chunk_local?
      // (The last chunk may hold fewer than chunk_size_ bytes.)
      std::size_t chunk_avail = chunk_bytes_available(chunk_idx, chunk_local);

      std::size_t take = std::min(remaining, chunk_avail);

      result.emplace_back(buffers_[chunk_idx] + chunk_local, take);

      pack_pos += take;
      remaining -= take;
    }
  }

  return result;
}

std::size_t cache_ranges::find_range(std::size_t offset) const noexcept
{
  // Upper-bound on range.offset gives us the first range whose offset
  // is strictly greater than `offset`; one step back is our candidate.
  auto it =
    std::upper_bound(ranges_.begin(), ranges_.end(), offset, [](std::size_t off, range const& r) {
      return off < static_cast<std::size_t>(r.offset());
    });

  if (it == ranges_.begin()) return ranges_.size();  // before all ranges

  --it;
  std::size_t idx = static_cast<std::size_t>(it - ranges_.begin());

  // Verify that offset actually falls inside this range.
  if (offset >= static_cast<std::size_t>(ranges_[idx].offset() + ranges_[idx].size()))
    return ranges_.size();  // in a gap between ranges

  return idx;
}

std::size_t cache_ranges::chunk_bytes_available(std::size_t chunk_idx,
                                                std::size_t intra_chunk_offset) const noexcept
{
  std::size_t chunk_start = chunk_idx * chunk_size_;
  std::size_t chunk_end   = std::min(chunk_start + chunk_size_, total_packed_bytes_);
  std::size_t chunk_used  = chunk_end - chunk_start;
  return chunk_used - intra_chunk_offset;
}

std::size_t cache_ranges::max_offset() const noexcept
{
  if (ranges_.empty()) return 0;
  return static_cast<std::size_t>(ranges_.back().offset() + ranges_.back().size());
}

}  // namespace sirius::op::scan
