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

//===----------------------------------------------------------------------===//
// Synthetic BITPACKING-segment builders. Shared by the unit tests and the
// codec microbenches. Format mirrors the on-disk DuckDB BITPACKING layout
// the kernel parses:
//   [0..8)               metadata_end (uint64)
//   [8..)                per-mode header (frame, [width], [delta_offset])
//   [data_off..)         packed stream (FOR / DELTA_FOR only)
//   [metadata_end-4*N..  per-group metadata trailer
//    metadata_end)        (one uint32 per group, last entry first)
//===----------------------------------------------------------------------===//

#include <cuda/scan/gpu_decode_bitpacking.cuh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sirius::test::decode::bitpacking {

template <typename T>
inline std::vector<uint32_t> pack_values(std::vector<T> const& values, uint32_t width)
{
  if (width == 0 || values.empty()) return std::vector<uint32_t>(1, 0u);
  size_t total_bits  = static_cast<size_t>(values.size()) * width;
  size_t total_words = (total_bits + 31u) / 32u + 1u;  // +1 guard word
  std::vector<uint32_t> packed(total_words, 0u);
  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t v = static_cast<uint64_t>(values[i]);
    if (width < 64) v &= ((uint64_t{1} << width) - 1);
    size_t bit_pos  = i * width;
    size_t word_idx = bit_pos / 32;
    size_t bit_off  = bit_pos % 32;
    packed[word_idx] |= static_cast<uint32_t>(v << bit_off);
    if (bit_off + width > 32) {
      packed[word_idx + 1] |= static_cast<uint32_t>(v >> (32 - bit_off));
    }
    if (sizeof(T) > 4 && bit_off > 0 && bit_off + width > 64) {
      packed[word_idx + 2] |= static_cast<uint32_t>(v >> (64 - bit_off));
    }
  }
  return packed;
}

template <typename T>
inline std::vector<uint8_t> make_constant_block(T value)
{
  std::vector<uint8_t> block(64, 0);
  uint64_t metadata_end = 32;
  std::memcpy(block.data(), &metadata_end, sizeof(metadata_end));
  std::memcpy(block.data() + 8, &value, sizeof(T));
  uint32_t encoded =
    (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::CONSTANT) << 24) | 8u;
  std::memcpy(block.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
inline std::vector<uint8_t> make_constant_delta_block(T frame, T delta)
{
  std::vector<uint8_t> block(64, 0);
  uint64_t metadata_end = 32;
  std::memcpy(block.data(), &metadata_end, sizeof(metadata_end));
  std::memcpy(block.data() + 8, &frame, sizeof(T));
  std::memcpy(block.data() + 8 + sizeof(T), &delta, sizeof(T));
  uint32_t encoded =
    (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::CONSTANT_DELTA) << 24) | 8u;
  std::memcpy(block.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
inline std::vector<uint8_t> make_for_block(T frame, uint32_t width, std::vector<T> const& values)
{
  auto packed           = pack_values<T>(values, width);
  size_t packed_bytes   = packed.size() * sizeof(uint32_t);
  size_t header_end     = 8 + 2 * sizeof(T) + packed_bytes;
  size_t metadata_end_v = header_end + 4;
  size_t block_size     = std::max<size_t>(metadata_end_v, 64);
  std::vector<uint8_t> block(block_size, 0);
  std::memcpy(block.data(), &metadata_end_v, sizeof(uint64_t));
  T width_t = static_cast<T>(width);
  std::memcpy(block.data() + 8, &frame, sizeof(T));
  std::memcpy(block.data() + 8 + sizeof(T), &width_t, sizeof(T));
  std::memcpy(block.data() + 8 + 2 * sizeof(T), packed.data(), packed_bytes);
  uint32_t encoded = (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::FOR) << 24) | 8u;
  std::memcpy(block.data() + metadata_end_v - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
inline std::vector<uint8_t> make_delta_for_block(T frame,
                                                 T delta_offset,
                                                 uint32_t width,
                                                 std::vector<T> const& packed_values)
{
  auto packed           = pack_values<T>(packed_values, width);
  size_t packed_bytes   = packed.size() * sizeof(uint32_t);
  size_t header_end     = 8 + 3 * sizeof(T) + packed_bytes;
  size_t metadata_end_v = header_end + 4;
  size_t block_size     = std::max<size_t>(metadata_end_v, 64);
  std::vector<uint8_t> block(block_size, 0);
  std::memcpy(block.data(), &metadata_end_v, sizeof(uint64_t));
  T width_t = static_cast<T>(width);
  std::memcpy(block.data() + 8, &frame, sizeof(T));
  std::memcpy(block.data() + 8 + sizeof(T), &width_t, sizeof(T));
  std::memcpy(block.data() + 8 + 2 * sizeof(T), &delta_offset, sizeof(T));
  std::memcpy(block.data() + 8 + 3 * sizeof(T), packed.data(), packed_bytes);
  uint32_t encoded =
    (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::DELTA_FOR) << 24) | 8u;
  std::memcpy(block.data() + metadata_end_v - 4, &encoded, sizeof(encoded));
  return block;
}

}  // namespace sirius::test::decode::bitpacking
