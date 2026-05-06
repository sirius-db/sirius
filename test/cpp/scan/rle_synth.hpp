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

// Synthetic RLE-segment builders shared by unit tests and codec benches.
// Layout matches duckdb/src/storage/compression/rle.cpp:
//   [0..8)                            uint64 rle_count_offset
//   [8 .. 8 + entry_count*sizeof(T))  values
//   [...optional zero padding...]
//   [rle_count_offset .. end)         counts (uint16_t)
// The builder below packs values back-to-back with no padding (a valid
// sub-shape of the on-disk format).

#include <cuda/scan/gpu_decode_rle.cuh>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sirius::test::decode::rle {

/// Build a self-contained RLE segment block. Throws if sizes mismatch.
template <typename T>
inline std::vector<uint8_t> make_rle_block(std::vector<T> const& values,
                                           std::vector<uint16_t> const& counts)
{
  if (values.size() != counts.size()) {
    throw std::runtime_error(
      "make_rle_block: values and counts size mismatch (caller error)");
  }
  size_t entry_count        = values.size();
  size_t values_bytes       = entry_count * sizeof(T);
  size_t counts_bytes       = entry_count * sizeof(uint16_t);
  uint64_t rle_count_offset = ::sirius::cuda::scan::RLE_HEADER_SIZE + values_bytes;

  std::vector<uint8_t> block(rle_count_offset + counts_bytes, 0);
  std::memcpy(block.data(), &rle_count_offset, sizeof(rle_count_offset));
  if (values_bytes > 0) {
    std::memcpy(block.data() + ::sirius::cuda::scan::RLE_HEADER_SIZE,
                values.data(),
                values_bytes);
  }
  if (counts_bytes > 0) {
    std::memcpy(block.data() + rle_count_offset, counts.data(), counts_bytes);
  }
  return block;
}

/// Build a block of `n_runs` runs, each `run_len` rows, with monotonically
/// increasing values (0, 1, 2, ...).
template <typename T>
inline std::vector<uint8_t> make_uniform_runs(uint32_t n_runs, uint16_t run_len)
{
  std::vector<T> values(n_runs);
  for (uint32_t i = 0; i < n_runs; ++i) values[i] = static_cast<T>(i);
  std::vector<uint16_t> counts(n_runs, run_len);
  return make_rle_block<T>(values, counts);
}

}  // namespace sirius::test::decode::rle
