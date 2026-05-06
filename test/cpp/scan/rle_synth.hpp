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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sirius::test::decode::rle {

/// Build a self-contained RLE segment block. Throws if sizes mismatch.
/// rle_count_offset is rounded up to 8 bytes to match DuckDB's
/// `AlignValue<8>` in `FlushSegment` — keeps `counts[]` naturally aligned
/// for uint16 device reads.
template <typename T>
inline std::vector<uint8_t> make_rle_block(std::vector<T> const& values,
                                           std::vector<uint16_t> const& counts)
{
  if (values.size() != counts.size()) {
    throw std::runtime_error(
      "make_rle_block: values and counts size mismatch (caller error)");
  }
  size_t entry_count          = values.size();
  size_t values_bytes         = entry_count * sizeof(T);
  size_t counts_bytes         = entry_count * sizeof(uint16_t);
  uint64_t minimal_offset     = ::sirius::cuda::scan::RLE_HEADER_SIZE + values_bytes;
  uint64_t rle_count_offset   = (minimal_offset + 7u) & ~uint64_t{7u};

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

/// Build a block whose run lengths follow a Pareto-like distribution —
/// many medium runs + a few very long ones, like a sorted low-cardinality
/// column. `x_min` is the floor run length (mean ≈ α/(α-1) · x_min ≈ 3·x_min
/// at α=1.5). Tune so total_rows / mean stays well below the build cap.
template <typename T>
inline std::vector<uint8_t> make_pareto_runs(uint32_t total_rows,
                                             uint32_t seed,
                                             double x_min     = 50.0,
                                             uint16_t cap_max = 2048,
                                             uint32_t* out_entries = nullptr)
{
  std::vector<T> values;
  std::vector<uint16_t> counts;
  uint32_t lcg     = seed ? seed : 0x9E3779B9u;
  uint32_t emitted = 0;
  T next_value     = T(0);
  while (emitted < total_rows) {
    lcg            = lcg * 1664525u + 1013904223u;
    double const u = static_cast<double>(lcg) / static_cast<double>(0xFFFFFFFFu);
    double const x = x_min / std::pow(1.0 - 0.999 * u, 1.0 / 1.5);
    uint16_t run_len =
      static_cast<uint16_t>(std::min<double>(cap_max, std::max(1.0, x)));
    if (emitted + run_len > total_rows) run_len = static_cast<uint16_t>(total_rows - emitted);
    values.push_back(next_value++);
    counts.push_back(run_len);
    emitted += run_len;
  }
  if (out_entries) *out_entries = static_cast<uint32_t>(values.size());
  return make_rle_block<T>(values, counts);
}

}  // namespace sirius::test::decode::rle
