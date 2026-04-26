/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "cuda/scan/gpu_decode.cuh"
#include "operator/scan/decode_test_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>

#include <cstdint>
#include <numeric>
#include <vector>

using sirius::cuda::scan::gpu_decode_rle;
using sirius::test::decode::download;
using sirius::test::decode::make_rle_block;
using sirius::test::decode::upload;

namespace {

/// Run gpu_decode_rle with a host-source block and return decoded values.
///
/// gpu_decode_rle parses its 8-byte RLE header on the HOST (so segment_data
/// must be a host pointer), then does the GPU work either by H2D-copying
/// the segment itself or by reusing a caller-provided device scratch buffer.
/// The simplest test path is "no scratch, no skip" — the function does its
/// own H2D and frees on completion.
template <typename T>
std::vector<T> decode_one_segment(std::vector<uint8_t> const& block_bytes, uint32_t row_count)
{
  auto stream  = cudf::get_default_stream();
  auto cstream = stream.value();
  sirius::test::decode::device_alloc d_output(static_cast<size_t>(row_count) * sizeof(T));

  gpu_decode_rle(/*segment_data=*/block_bytes.data(),
                 /*segment_size=*/block_bytes.size(),
                 /*block_offset=*/0,
                 row_count,
                 /*type_size=*/sizeof(T),
                 d_output.ptr,
                 stream);

  return download<T>(d_output.ptr, row_count, cstream);
}

/// Build the expected expanded vector from values + counts.  Mirrors the
/// host-side prefix-sum walk that gpu_decode_rle does on the CPU before
/// dispatching to its expand kernel.
template <typename T>
std::vector<T> expand_runs(std::vector<T> const& values, std::vector<uint16_t> const& counts)
{
  std::vector<T> out;
  for (size_t i = 0; i < values.size(); ++i) {
    for (uint16_t c = 0; c < counts[i]; ++c)
      out.push_back(values[i]);
  }
  return out;
}

}  // anonymous namespace

TEST_CASE("RLE decode: single-entry run broadcasts one value", "[scan][rle][gpu_decode]")
{
  SECTION("int32, 100 rows, value 42")
  {
    std::vector<int32_t> values{42};
    std::vector<uint16_t> counts{100};
    auto block = make_rle_block<int32_t>(values, counts);
    auto out   = decode_one_segment<int32_t>(block, 100);
    REQUIRE(out.size() == 100);
    for (auto v : out)
      REQUIRE(v == 42);
  }

  SECTION("int64, 50 rows, value 5'000'000'000")
  {
    int64_t v = 5'000'000'000LL;
    std::vector<int64_t> values{v};
    std::vector<uint16_t> counts{50};
    auto block = make_rle_block<int64_t>(values, counts);
    auto out   = decode_one_segment<int64_t>(block, 50);
    REQUIRE(out.size() == 50);
    for (auto x : out)
      REQUIRE(x == v);
  }

  SECTION("uint8, 33 rows (non-warp-aligned), value 200")
  {
    std::vector<uint8_t> values{200};
    std::vector<uint16_t> counts{33};
    auto block = make_rle_block<uint8_t>(values, counts);
    auto out   = decode_one_segment<uint8_t>(block, 33);
    REQUIRE(out.size() == 33);
    for (auto v : out)
      REQUIRE(v == 200);
  }
}

TEST_CASE("RLE decode: multi-entry runs are expanded in order", "[scan][rle][gpu_decode]")
{
  SECTION("int32, four runs of varying length")
  {
    std::vector<int32_t> values{10, 20, 30, 40};
    std::vector<uint16_t> counts{5, 3, 8, 4};  // total = 20 rows
    auto block    = make_rle_block<int32_t>(values, counts);
    auto out      = decode_one_segment<int32_t>(block, 20);
    auto expected = expand_runs<int32_t>(values, counts);
    REQUIRE(out.size() == expected.size());
    REQUIRE(out == expected);
  }

  SECTION("int16, alternating values, exercises smaller type stride")
  {
    std::vector<int16_t> values{-1000, 1000, -1000, 1000, -1000};
    std::vector<uint16_t> counts{7, 7, 7, 7, 7};  // total = 35
    auto block    = make_rle_block<int16_t>(values, counts);
    auto out      = decode_one_segment<int16_t>(block, 35);
    auto expected = expand_runs<int16_t>(values, counts);
    REQUIRE(out == expected);
  }
}

TEST_CASE("RLE decode: cross-CTA boundary (rows > 256)", "[scan][rle][gpu_decode]")
{
  // Default kernel threads = 256, so 600 rows spans 3 CTAs.  Each output
  // row binary-searches the cumsum; this case verifies the binary search
  // is correct across CTA boundaries and within shared memory.
  std::vector<int32_t> values;
  std::vector<uint16_t> counts;
  for (int32_t i = 0; i < 12; ++i) {
    values.push_back(i * 11);
    counts.push_back(50);  // 12 runs × 50 = 600 rows
  }
  auto block    = make_rle_block<int32_t>(values, counts);
  auto out      = decode_one_segment<int32_t>(block, 600);
  auto expected = expand_runs<int32_t>(values, counts);
  REQUIRE(out.size() == 600);
  REQUIRE(out == expected);
}

TEST_CASE("RLE decode: large entry count exercises the gmem path", "[scan][rle][gpu_decode]")
{
  // gpu_decode_rle uses the smem variant when entry_count <= 4096
  // (RLE_SMEM_MAX_ENTRIES) and the gmem variant above that.  This test
  // crosses the threshold to validate the gmem path.
  constexpr uint32_t entries = 5000;
  std::vector<int32_t> values(entries);
  std::iota(values.begin(), values.end(), 1);  // 1, 2, 3, ...
  std::vector<uint16_t> counts(entries, 1);    // 1 row each → 5000 rows total
  auto block = make_rle_block<int32_t>(values, counts);
  auto out   = decode_one_segment<int32_t>(block, entries);
  REQUIRE(out.size() == entries);
  for (uint32_t i = 0; i < entries; ++i)
    REQUIRE(out[i] == static_cast<int32_t>(i + 1));
}
