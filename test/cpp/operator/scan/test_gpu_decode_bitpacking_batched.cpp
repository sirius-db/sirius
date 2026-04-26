/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "operator/scan/decode_test_utils.hpp"

#include "cuda/scan/gpu_decode.cuh"

#include <catch.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <cstdint>
#include <vector>

using namespace sirius::cuda::scan;
using sirius::test::decode::download;
using sirius::test::decode::make_constant_block;
using sirius::test::decode::make_constant_delta_block;
using sirius::test::decode::make_delta_for_block;
using sirius::test::decode::make_for_block;
using sirius::test::decode::upload;

namespace {

/// Wrap one block's bytes into a single-segment, single-group descriptor and
/// run the batched bitpacking kernel; return host-decoded values.
template <typename T>
std::vector<T> decode_one_segment(std::vector<uint8_t> const& block_bytes,
                                  uint32_t row_count,
                                  bool is_signed)
{
  auto stream  = cudf::get_default_stream();
  auto cstream = stream.value();

  auto d_block = upload(block_bytes, cstream);

  sirius::test::decode::device_alloc d_output(static_cast<size_t>(row_count) * sizeof(T));

  batched_bp_seg_desc desc{};
  desc.d_block           = static_cast<const uint8_t*>(d_block.ptr);
  desc.block_offset      = 0;
  desc.group_idx         = 0;
  desc.group_row_count   = row_count;
  desc.global_row_offset = 0;

  std::vector<batched_bp_seg_desc> descs{desc};
  gpu_decode_bitpacking_batched(descs.data(),
                                static_cast<uint32_t>(descs.size()),
                                d_output.ptr,
                                static_cast<uint32_t>(sizeof(T)),
                                is_signed,
                                stream);

  return download<T>(d_output.ptr, row_count, cstream);
}

}  // anonymous namespace

TEST_CASE("bitpacking batched: CONSTANT mode broadcasts the constant",
          "[scan][bitpacking][gpu_decode]")
{
  SECTION("int32, value 42, 100 rows")
  {
    auto block = make_constant_block<int32_t>(42);
    auto out   = decode_one_segment<int32_t>(block, 100, /*is_signed=*/true);
    REQUIRE(out.size() == 100);
    for (auto v : out) REQUIRE(v == 42);
  }

  SECTION("int64, value 5'000'000'000, 256 rows")
  {
    int64_t v = 5'000'000'000LL;
    auto block = make_constant_block<int64_t>(v);
    auto out   = decode_one_segment<int64_t>(block, 256, /*is_signed=*/true);
    REQUIRE(out.size() == 256);
    for (auto x : out) REQUIRE(x == v);
  }

  SECTION("uint8, value 200, 33 rows (non-warp-aligned)")
  {
    auto block = make_constant_block<uint8_t>(200);
    auto out   = decode_one_segment<uint8_t>(block, 33, /*is_signed=*/false);
    REQUIRE(out.size() == 33);
    for (auto v : out) REQUIRE(v == 200);
  }
}

TEST_CASE("bitpacking batched: CONSTANT_DELTA mode produces an arithmetic progression",
          "[scan][bitpacking][gpu_decode]")
{
  SECTION("int32, frame=10 delta=3, 50 rows")
  {
    auto block = make_constant_delta_block<int32_t>(10, 3);
    auto out   = decode_one_segment<int32_t>(block, 50, /*is_signed=*/true);
    REQUIRE(out.size() == 50);
    for (uint32_t i = 0; i < 50; ++i) REQUIRE(out[i] == 10 + 3 * static_cast<int32_t>(i));
  }
}

TEST_CASE("bitpacking batched: FOR mode unpacks frame + packed values",
          "[scan][bitpacking][gpu_decode]")
{
  SECTION("int32, width=5, frame=100, 32 rows (one warp)")
  {
    std::vector<int32_t> packed_values(32);
    for (int32_t i = 0; i < 32; ++i) packed_values[i] = i;  // 0..31, fits in 5 bits
    auto block = make_for_block<int32_t>(/*frame=*/100, /*width=*/5, packed_values);
    auto out   = decode_one_segment<int32_t>(block, 32, /*is_signed=*/true);
    REQUIRE(out.size() == 32);
    for (uint32_t i = 0; i < 32; ++i) REQUIRE(out[i] == 100 + static_cast<int32_t>(i));
  }

  SECTION("int32, width=12, 100 rows (multi-warp, non-aligned tail)")
  {
    std::vector<int32_t> packed_values(100);
    for (int32_t i = 0; i < 100; ++i) packed_values[i] = i * 7 % 4096;  // fits in 12 bits
    auto block = make_for_block<int32_t>(/*frame=*/-50, /*width=*/12, packed_values);
    auto out   = decode_one_segment<int32_t>(block, 100, /*is_signed=*/true);
    REQUIRE(out.size() == 100);
    for (uint32_t i = 0; i < 100; ++i) REQUIRE(out[i] == -50 + (i * 7 % 4096));
  }

  SECTION("int64, width=33, 64 rows (3-word unpack path)")
  {
    // width > 32 with non-zero bit_off forces unpack_value's third-word read.
    std::vector<int64_t> packed_values(64);
    for (int64_t i = 0; i < 64; ++i) packed_values[i] = (i * 0x1234567LL) & ((1LL << 33) - 1);
    auto block = make_for_block<int64_t>(/*frame=*/1'000'000LL, /*width=*/33, packed_values);
    auto out   = decode_one_segment<int64_t>(block, 64, /*is_signed=*/true);
    REQUIRE(out.size() == 64);
    for (uint32_t i = 0; i < 64; ++i) {
      int64_t expected = 1'000'000LL + ((static_cast<int64_t>(i) * 0x1234567LL) & ((1LL << 33) - 1));
      REQUIRE(out[i] == expected);
    }
  }
}

TEST_CASE("bitpacking batched: DELTA_FOR mode prefix-sums packed deltas",
          "[scan][bitpacking][gpu_decode]")
{
  SECTION("int32, width=4, frame=0, delta_offset=10, 16 rows")
  {
    // After the kernel: value[i] = (sum of packed[0..i]) + delta_offset.
    std::vector<int32_t> packed_values{1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1};
    auto block = make_delta_for_block<int32_t>(/*frame=*/0,
                                               /*delta_offset=*/10,
                                               /*width=*/4,
                                               packed_values);
    auto out = decode_one_segment<int32_t>(block, 16, /*is_signed=*/true);
    REQUIRE(out.size() == 16);
    int32_t running = 10;
    for (uint32_t i = 0; i < 16; ++i) {
      running += packed_values[i];
      REQUIRE(out[i] == running);
    }
  }
}

TEST_CASE("bitpacking batched: multi-segment dispatch packs distinct outputs",
          "[scan][bitpacking][gpu_decode]")
{
  // Two CONSTANT segments with different values; verify each writes to its own
  // output region via global_row_offset.
  auto stream   = cudf::get_default_stream();
  auto cstream  = stream.value();
  auto block_a  = make_constant_block<int32_t>(7);
  auto block_b  = make_constant_block<int32_t>(99);
  auto d_a      = upload(block_a, cstream);
  auto d_b      = upload(block_b, cstream);

  constexpr uint32_t rows_a = 50;
  constexpr uint32_t rows_b = 70;
  sirius::test::decode::device_alloc d_output((rows_a + rows_b) * sizeof(int32_t));

  std::vector<batched_bp_seg_desc> descs{
    {static_cast<const uint8_t*>(d_a.ptr), 0, 0, rows_a, 0},
    {static_cast<const uint8_t*>(d_b.ptr), 0, 0, rows_b, rows_a},
  };

  gpu_decode_bitpacking_batched(descs.data(),
                                static_cast<uint32_t>(descs.size()),
                                d_output.ptr,
                                /*type_size=*/4,
                                /*is_signed=*/true,
                                stream);

  auto out = download<int32_t>(d_output.ptr, rows_a + rows_b, cstream);
  for (uint32_t i = 0; i < rows_a; ++i) REQUIRE(out[i] == 7);
  for (uint32_t i = 0; i < rows_b; ++i) REQUIRE(out[rows_a + i] == 99);
}
