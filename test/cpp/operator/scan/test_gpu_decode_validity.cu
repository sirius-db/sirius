/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

//===----------------------------------------------------------------------===//
// Direct-launch tests for the validity-mask kernels.
//
// kernel_fill_valid initialises a mask to all-1s; kernel_count_valid_bits
// popcounts it on-GPU.  These are the two primitives every codec's null
// decode relies on, so testing them in isolation catches regressions in
// either the bit fill, the per-thread popcount, the tree reduction, or
// the tail-bit masking applied to the last word of the mask.
//
// The test launches kernels with `<<<>>>` syntax, so this file must be
// compiled by nvcc (.cu).  sirius_unittest's CUDA_SEPARABLE_COMPILATION +
// CUDA_RESOLVE_DEVICE_SYMBOLS properties ensure cross-TU device symbols
// link via the cmake_device_link step.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_validity.cuh"

#include <cuda_runtime.h>

#include <catch.hpp>

#include <cstdint>
#include <vector>

using sirius::cuda::scan::kernel_count_valid_bits;
using sirius::cuda::scan::kernel_fill_valid;

namespace {

/// Minimal device buffer wrapper — the project's decode_test_utils.hpp has
/// one but it's a host header that's compiled by clang in the .cpp tests;
/// duplicating the tiny version here keeps this .cu file self-contained
/// rather than dragging the whole .hpp through nvcc.
struct device_buf {
  void* ptr  = nullptr;
  size_t len = 0;
  explicit device_buf(size_t bytes) : len(bytes)
  {
    if (bytes > 0) ::cudaMalloc(&ptr, bytes);
  }
  device_buf(device_buf const&)            = delete;
  device_buf& operator=(device_buf const&) = delete;
  ~device_buf()
  {
    if (ptr) ::cudaFree(ptr);
  }
};

uint32_t run_count_valid_bits(uint64_t const* h_mask, uint32_t num_words, uint32_t total_rows)
{
  device_buf d_mask(num_words * sizeof(uint64_t));
  device_buf d_count(sizeof(uint32_t));

  ::cudaMemcpy(d_mask.ptr, h_mask, num_words * sizeof(uint64_t), cudaMemcpyHostToDevice);
  ::cudaMemset(d_count.ptr, 0, sizeof(uint32_t));

  kernel_count_valid_bits<<<1, 256>>>(static_cast<const uint64_t*>(d_mask.ptr),
                                      num_words,
                                      total_rows,
                                      static_cast<uint32_t*>(d_count.ptr));
  ::cudaDeviceSynchronize();

  uint32_t h_count = 0;
  ::cudaMemcpy(&h_count, d_count.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  return h_count;
}

uint32_t naive_popcount(uint64_t const* mask, uint32_t num_words, uint32_t total_rows)
{
  uint32_t total = 0;
  for (uint32_t i = 0; i < num_words; ++i) {
    uint64_t w = mask[i];
    if (i == num_words - 1) {
      uint32_t tail = total_rows & 63;
      if (tail > 0) w &= (1ULL << tail) - 1;
    }
    total += __builtin_popcountll(w);
  }
  return total;
}

}  // anonymous namespace

TEST_CASE("validity: kernel_fill_valid sets every word to all-1s", "[scan][validity][gpu_decode]")
{
  SECTION("aligned: 256 rows = 4 words")
  {
    constexpr uint32_t num_words = 4;
    device_buf d_mask(num_words * sizeof(uint64_t));

    uint32_t blocks = (num_words + 255) / 256;
    kernel_fill_valid<<<blocks, 256>>>(static_cast<uint64_t*>(d_mask.ptr), num_words);
    REQUIRE(::cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint64_t> h_mask(num_words);
    ::cudaMemcpy(h_mask.data(), d_mask.ptr, num_words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (auto w : h_mask)
      REQUIRE(w == ~0ULL);
  }

  SECTION("multi-CTA: 1000 words exceeds one CTA's 256-thread span")
  {
    constexpr uint32_t num_words = 1000;
    device_buf d_mask(num_words * sizeof(uint64_t));

    uint32_t blocks = (num_words + 255) / 256;
    kernel_fill_valid<<<blocks, 256>>>(static_cast<uint64_t*>(d_mask.ptr), num_words);
    REQUIRE(::cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint64_t> h_mask(num_words);
    ::cudaMemcpy(h_mask.data(), d_mask.ptr, num_words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < num_words; ++i)
      REQUIRE(h_mask[i] == ~0ULL);
  }
}

TEST_CASE("validity: kernel_count_valid_bits popcount", "[scan][validity][gpu_decode]")
{
  SECTION("all-valid mask, word-aligned row count")
  {
    constexpr uint32_t total_rows = 128;
    constexpr uint32_t num_words  = (total_rows + 63) / 64;
    std::vector<uint64_t> mask(num_words, ~0ULL);
    REQUIRE(run_count_valid_bits(mask.data(), num_words, total_rows) == total_rows);
  }

  SECTION("all-zero mask returns 0")
  {
    constexpr uint32_t total_rows = 200;
    constexpr uint32_t num_words  = (total_rows + 63) / 64;
    std::vector<uint64_t> mask(num_words, 0ULL);
    REQUIRE(run_count_valid_bits(mask.data(), num_words, total_rows) == 0);
  }

  SECTION("partial last word: tail bits beyond row_count must not be counted")
  {
    // 100 rows = 2 words.  Word 1 has only 36 live bits (bits 64..99); the
    // remaining 28 high bits are padding the kernel must mask off.
    constexpr uint32_t total_rows = 100;
    constexpr uint32_t num_words  = 2;
    std::vector<uint64_t> mask{
      ~0ULL,  // word 0: all 64 valid
      ~0ULL,  // word 1: all 64 set, but only bits 0..35 are live
    };
    auto count = run_count_valid_bits(mask.data(), num_words, total_rows);
    REQUIRE(count == total_rows);  // 64 + 36 = 100, NOT 128
  }

  SECTION("mixed pattern matches naive host popcount")
  {
    constexpr uint32_t total_rows = 333;
    constexpr uint32_t num_words  = (total_rows + 63) / 64;  // 6
    std::vector<uint64_t> mask{
      0xAAAAAAAAAAAAAAAAULL,
      0x5555555555555555ULL,
      0xFFFFFFFF00000000ULL,
      0x0000FFFFFFFF0000ULL,
      0xDEADBEEFCAFEBABEULL,
      0x0000000FFFFFFFFFULL,
    };
    auto expected = naive_popcount(mask.data(), num_words, total_rows);
    auto actual   = run_count_valid_bits(mask.data(), num_words, total_rows);
    REQUIRE(actual == expected);
  }

  SECTION("row_count is exact multiple of 64 — no tail masking")
  {
    constexpr uint32_t total_rows = 256;  // exactly 4 words
    constexpr uint32_t num_words  = 4;
    std::vector<uint64_t> mask{
      0xFFFFFFFFFFFFFFFFULL, 0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL, 0xAAAAAAAAAAAAAAAAULL};
    auto expected = naive_popcount(mask.data(), num_words, total_rows);
    auto actual   = run_count_valid_bits(mask.data(), num_words, total_rows);
    REQUIRE(actual == expected);
  }
}
