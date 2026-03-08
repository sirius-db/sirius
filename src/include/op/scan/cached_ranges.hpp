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

#include <cudf/io/text/byte_range_info.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace sirius::op::scan {

// ---------------------------------------------------------------------------
// cache_ranges
//
// Stores data for a set of non-overlapping, sorted byte ranges.
// The payload bytes are packed consecutively into fixed-size chunks:
//
//   chunk_size = 10, ranges = [{offset=2, size=8}, {offset=50, size=51}]
//
//   chunk 0 : [r0 bytes 0..7 | r1 bytes 0..1]   (8 + 2 = 10)
//   chunk 1 : [r1 bytes 2..11]                   (10)
//   chunk 2 : [r1 bytes 12..21]                  (10)
//   chunk 3 : [r1 bytes 22..31]                  (10)
//   chunk 4 : [r1 bytes 32..41]                  (10)
//   chunk 5 : [r1 bytes 42..50]                  (9, last chunk may be partial)
//
// get_ranges(query_offset, query_size) returns a vector of std::span<const byte>
// that together cover exactly the requested bytes, each span pointing into the
// appropriate chunk buffer.
// ---------------------------------------------------------------------------

class cache_ranges {
 public:
  using range = cudf::io::text::byte_range_info;

  // ranges     – sorted, non-overlapping descriptors of the cached regions.
  // buffers    – packed chunk buffers (ownership transferred).
  // chunk_sz   – fixed size of every chunk (last chunk may be smaller).
  // device_id  – CUDA device id of the target device (hint for batch copies). -1 = unset.
  // numa_id    – NUMA node id of the pinned host memory (hint for batch copies). -1 = unset.
  cache_ranges(std::vector<range> ranges,
               std::vector<std::byte*> buffers,
               std::size_t chunk_sz,
               int device_id = -1,
               int numa_id   = -1);

  // Returns a list of spans that together cover [query_offset, query_offset+query_size).
  // Each span points directly into one of the underlying chunk buffers.
  // Throws std::out_of_range if the query does not fall within a cached range.
  [[nodiscard]] std::vector<std::span<const std::byte>> get_ranges(std::size_t query_offset,
                                                                   std::size_t query_size) const;

  [[nodiscard]] std::size_t max_offset() const noexcept;

  // Returns the CUDA device id hint (-1 if not set).
  [[nodiscard]] int device_id() const noexcept { return device_id_; }

  // Returns the NUMA node id hint for the host buffers (-1 if not set).
  [[nodiscard]] int numa_id() const noexcept { return numa_id_; }

 private:
  // Binary-search: return the index of the range that contains `offset`,
  // or ranges_.size() if none.
  [[nodiscard]] std::size_t find_range(std::size_t offset) const noexcept;

  // Returns the number of usable bytes in chunk `chunk_idx` starting from
  // `intra_chunk_offset`.  The last chunk may hold fewer than chunk_size_ bytes.
  [[nodiscard]] std::size_t chunk_bytes_available(std::size_t chunk_idx,
                                                  std::size_t intra_chunk_offset) const noexcept;

  std::vector<range> ranges_;
  std::vector<std::size_t> pack_offsets_;  // packed start of each range
  std::size_t total_packed_bytes_{0};
  std::vector<std::byte*> buffers_;
  std::size_t chunk_size_;
  int device_id_{-1};
  int numa_id_{-1};
};

}  // namespace sirius::op::scan
