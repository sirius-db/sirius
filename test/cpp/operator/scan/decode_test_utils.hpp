/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// Synthetic-data helpers for kernel-level decode unit tests.
//
// The kernels in src/cuda/scan consume bytes laid out exactly the way DuckDB
// writes them on disk.  These helpers construct that layout in host memory,
// stage it on device, and read decoded outputs back so tests can assert
// against expected values without standing up a full DuckDB connection.
//
// Coverage as A1 ships: bitpacking (CONSTANT / CONSTANT_DELTA / FOR /
// DELTA_FOR mode) + validity-mask fill / popcount.  Other codecs are added
// alongside their kernel imports in PR A2 and PR A3.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sirius::test::decode {

//===----------------------------------------------------------------------===//
// Bit packing
//===----------------------------------------------------------------------===//

/// Pack `values.size()` values, each in `width` bits, LSB-first within each
/// 32-bit word, with one extra guard word at the end (matches what the kernel
/// expects via `unpack_value`).
template <typename T>
inline std::vector<uint32_t> pack_values(std::vector<T> const& values, uint32_t width)
{
  if (width == 0 || values.empty()) return std::vector<uint32_t>(1, 0);
  size_t total_bits  = static_cast<size_t>(values.size()) * width;
  size_t total_words = (total_bits + 31) / 32 + 1;  // +1 guard
  std::vector<uint32_t> packed(total_words, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t v = static_cast<uint64_t>(values[i]);
    if (width < 64) v &= ((1ULL << width) - 1);
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

//===----------------------------------------------------------------------===//
// DuckDB-bitpacking-format block construction (single segment, single
// metadata group) at block_offset 0.  Returns a host byte buffer that the
// kernel can decode when staged on device.
//
// Layout (matches the kernel's reads in gpu_decode_bitpacking.cu):
//   bytes [0..8)  metadata_end (uint64)  — set so the encoded entry sits
//                                          at metadata_end - 4
//   bytes [8..)   segment data (mode-dependent)
//   bytes [M-4..M) encoded entry: (mode << 24) | data_off
//===----------------------------------------------------------------------===//

template <typename T>
inline std::vector<uint8_t> make_constant_block(T value)
{
  std::vector<uint8_t> block(64, 0);
  uint64_t metadata_end = 32;
  std::memcpy(block.data(), &metadata_end, sizeof(metadata_end));
  std::memcpy(block.data() + 8, &value, sizeof(T));  // v0
  // v1 unused for CONSTANT but the kernel still reads it; leave 0.
  uint32_t encoded =
    (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::CONSTANT) << 24) | 8;
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
    (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::CONSTANT_DELTA) << 24) | 8;
  std::memcpy(block.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
inline std::vector<uint8_t> make_for_block(T frame, uint32_t width, std::vector<T> const& values)
{
  // values are the *delta-from-frame* values that get bitpacked.
  auto packed = pack_values<T>(values, width);
  // Block layout:
  //   [0..8)               metadata_end
  //   [8..8+T)             frame
  //   [8+T..8+2T)          width (encoded as T)
  //   [8+2T..8+2T+packed)  packed bytes
  //   [M-4..M)             encoded entry
  size_t packed_bytes = packed.size() * sizeof(uint32_t);
  size_t M_min        = 8 + 2 * sizeof(T) + packed_bytes + 4;  // +4 for encoded entry slot
  size_t block_size   = std::max<size_t>(M_min, 64);
  std::vector<uint8_t> block(block_size, 0);
  uint64_t metadata_end = M_min - 4 + 4;  // include space for the encoded entry
  std::memcpy(block.data(), &metadata_end, sizeof(metadata_end));
  T width_t = static_cast<T>(width);
  std::memcpy(block.data() + 8, &frame, sizeof(T));
  std::memcpy(block.data() + 8 + sizeof(T), &width_t, sizeof(T));
  std::memcpy(block.data() + 8 + 2 * sizeof(T), packed.data(), packed_bytes);
  uint32_t encoded = (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::FOR) << 24) | 8;
  std::memcpy(block.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
inline std::vector<uint8_t> make_delta_for_block(T frame,
                                                 T delta_offset,
                                                 uint32_t width,
                                                 std::vector<T> const& packed_values)
{
  auto packed         = pack_values<T>(packed_values, width);
  size_t packed_bytes = packed.size() * sizeof(uint32_t);
  // Layout adds a third T (delta_offset) before the packed bytes.
  size_t M_min      = 8 + 3 * sizeof(T) + packed_bytes + 4;
  size_t block_size = std::max<size_t>(M_min, 64);
  std::vector<uint8_t> block(block_size, 0);
  uint64_t metadata_end = M_min;
  std::memcpy(block.data(), &metadata_end, sizeof(metadata_end));
  T width_t = static_cast<T>(width);
  std::memcpy(block.data() + 8, &frame, sizeof(T));
  std::memcpy(block.data() + 8 + sizeof(T), &width_t, sizeof(T));
  std::memcpy(block.data() + 8 + 2 * sizeof(T), &delta_offset, sizeof(T));
  std::memcpy(block.data() + 8 + 3 * sizeof(T), packed.data(), packed_bytes);
  uint32_t encoded =
    (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::DELTA_FOR) << 24) | 8;
  std::memcpy(block.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return block;
}

//===----------------------------------------------------------------------===//
// DuckDB-RLE-format block construction.  Layout (within the segment slot):
//
//   bytes [0..8)                     uint64_t rle_count_offset
//   bytes [8 .. rle_count_offset)    Values: T[entry_count]
//   bytes [rle_count_offset .. )     Counts: uint16_t[entry_count]
//
// Entry count is implicit — gpu_decode_rle's CPU header parser walks the
// counts array adding to a running total until it equals row_count, so
// the test harness must size `counts` so its sum equals the row_count
// passed at decode time.
//===----------------------------------------------------------------------===//

template <typename T>
inline std::vector<uint8_t> make_rle_block(std::vector<T> const& values,
                                           std::vector<uint16_t> const& counts)
{
  // Caller bug if these don't match — surface it instead of producing
  // a malformed block that decodes silently wrong.
  if (values.size() != counts.size()) {
    throw std::runtime_error("make_rle_block: values and counts size mismatch (caller error)");
  }
  size_t entry_count        = values.size();
  size_t values_bytes       = entry_count * sizeof(T);
  size_t counts_bytes       = entry_count * sizeof(uint16_t);
  uint64_t rle_count_offset = 8 + values_bytes;
  std::vector<uint8_t> block(8 + values_bytes + counts_bytes, 0);
  std::memcpy(block.data(), &rle_count_offset, sizeof(rle_count_offset));
  if (values_bytes > 0) std::memcpy(block.data() + 8, values.data(), values_bytes);
  if (counts_bytes > 0) std::memcpy(block.data() + 8 + values_bytes, counts.data(), counts_bytes);
  return block;
}

//===----------------------------------------------------------------------===//
// Device buffer helpers (don't drag in cucascade — kernel tests only need
// raw cudaMalloc/cudaMemcpy semantics).
//===----------------------------------------------------------------------===//

struct device_alloc {
  void* ptr   = nullptr;
  size_t size = 0;

  device_alloc() = default;
  explicit device_alloc(size_t bytes) : size(bytes)
  {
    if (bytes > 0) ::cudaMalloc(&ptr, bytes);
  }
  device_alloc(device_alloc const&)            = delete;
  device_alloc& operator=(device_alloc const&) = delete;
  device_alloc(device_alloc&& o) noexcept : ptr(o.ptr), size(o.size)
  {
    o.ptr  = nullptr;
    o.size = 0;
  }
  device_alloc& operator=(device_alloc&& o) noexcept
  {
    if (this != &o) {
      if (ptr) ::cudaFree(ptr);
      ptr   = o.ptr;
      size  = o.size;
      o.ptr = nullptr;
    }
    return *this;
  }
  ~device_alloc()
  {
    if (ptr) ::cudaFree(ptr);
  }
};

inline device_alloc upload(std::vector<uint8_t> const& host_bytes, cudaStream_t stream)
{
  device_alloc d(host_bytes.size());
  ::cudaMemcpyAsync(d.ptr, host_bytes.data(), host_bytes.size(), ::cudaMemcpyHostToDevice, stream);
  return d;
}

template <typename T>
inline std::vector<T> download(void const* d_ptr, size_t count, cudaStream_t stream)
{
  std::vector<T> out(count);
  ::cudaMemcpyAsync(out.data(), d_ptr, count * sizeof(T), ::cudaMemcpyDeviceToHost, stream);
  ::cudaStreamSynchronize(stream);
  return out;
}

}  // namespace sirius::test::decode
