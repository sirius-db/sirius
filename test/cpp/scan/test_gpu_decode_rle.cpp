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

// RLE decode tests. Happy-path cases assert decoded output for synthetic
// segments matching DuckDB's on-disk layout. Defensive cases pin the
// zero-fill contract for malformed metadata.

#include "scan/decode_test_utils.hpp"
#include "scan/rle_synth.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_rle.cuh>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

using duckdb::CompressionType;
using sirius::cuda::scan::decode_rle_data;
using sirius::cuda::scan::gpu_codec_run;
using sirius::cuda::scan::gpu_column_decode_input;
using sirius::cuda::scan::gpu_decode_table;
using sirius::cuda::scan::gpu_segment_desc;
using sirius::cuda::scan::RLE_HEADER_SIZE;
using sirius::test::decode::download;
using sirius::test::decode::rle::make_rle_block;

namespace {

auto const I16 = cudf::data_type{cudf::type_id::INT16};
auto const I32 = cudf::data_type{cudf::type_id::INT32};
auto const I64 = cudf::data_type{cudf::type_id::INT64};
auto const U8  = cudf::data_type{cudf::type_id::UINT8};

// Decode a single-segment column from one block of bytes.
template <typename T>
std::vector<T> decode_one(std::vector<uint8_t> const& bytes,
                          cudf::data_type type,
                          uint32_t row_count)
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_column_decode_input col;
  col.out_type   = type;
  col.total_rows = row_count;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_RLE,
                      {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                        static_cast<uint32_t>(d_seg.size()),
                                        0,
                                        row_count}}});

  auto t = gpu_decode_table({col}, stream.view(), mr);
  REQUIRE(t->num_rows() == static_cast<cudf::size_type>(row_count));
  return download<T>(t->get_column(0).view().data<T>(), row_count, stream.value());
}

// Reference expansion against which kernel output is asserted.
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

}  // namespace

TEST_CASE("gpu_decode_table RLE - single-entry run broadcasts one value", "[scan][decode][rle]")
{
  SECTION("int32 100 rows, value 42")
  {
    auto bytes = make_rle_block<int32_t>({42}, {100});
    auto out   = decode_one<int32_t>(bytes, I32, 100);
    REQUIRE(out.size() == 100);
    for (auto v : out)
      REQUIRE(v == 42);
  }

  SECTION("int64 50 rows, large value")
  {
    int64_t v  = 5'000'000'000LL;
    auto bytes = make_rle_block<int64_t>({v}, {50});
    auto out   = decode_one<int64_t>(bytes, I64, 50);
    for (auto x : out)
      REQUIRE(x == v);
  }

  SECTION("uint8 33 rows (non-warp-aligned)")
  {
    auto bytes = make_rle_block<uint8_t>({200}, {33});
    auto out   = decode_one<uint8_t>(bytes, U8, 33);
    for (auto v : out)
      REQUIRE(v == 200);
  }
}

TEST_CASE("gpu_decode_table RLE - multi-entry runs are expanded in order", "[scan][decode][rle]")
{
  SECTION("int32 four runs of varying length")
  {
    std::vector<int32_t> values{10, 20, 30, 40};
    std::vector<uint16_t> counts{5, 3, 8, 4};  // total = 20
    auto bytes    = make_rle_block<int32_t>(values, counts);
    auto out      = decode_one<int32_t>(bytes, I32, 20);
    auto expected = expand_runs<int32_t>(values, counts);
    REQUIRE(out == expected);
  }

  SECTION("int16 alternating values")
  {
    std::vector<int16_t> values{-1000, 1000, -1000, 1000, -1000};
    std::vector<uint16_t> counts{7, 7, 7, 7, 7};  // total = 35
    auto bytes    = make_rle_block<int16_t>(values, counts);
    auto out      = decode_one<int16_t>(bytes, I16, 35);
    auto expected = expand_runs<int16_t>(values, counts);
    REQUIRE(out == expected);
  }
}

TEST_CASE("gpu_decode_table RLE - cross-CTA boundary (rows > RLE_ROWS_PER_CHUNK)",
          "[scan][decode][rle]")
{
  // 3072 rows → spans 2 CTAs; verifies binary search on the 2nd CTA's
  // local_row_start = 2048 still resolves to the right entry.
  std::vector<int32_t> values;
  std::vector<uint16_t> counts;
  for (int32_t i = 0; i < 12; ++i) {
    values.push_back(i * 11);
    counts.push_back(256);
  }
  auto bytes    = make_rle_block<int32_t>(values, counts);
  auto out      = decode_one<int32_t>(bytes, I32, 12 * 256);
  auto expected = expand_runs<int32_t>(values, counts);
  REQUIRE(out == expected);
}

TEST_CASE("gpu_decode_table RLE - warp-broadcast straddles run boundary", "[scan][decode][rle]")
{
  // Avg run 33 ≥ warpSize triggers long_runs_heuristic; runs don't align
  // with warp boundaries, so warps straddle entries and exercise the
  // per-lane fallback inside the warp-broadcast branch.
  constexpr uint32_t N_RUNS  = 64;
  constexpr uint16_t RUN_LEN = 33;
  std::vector<int32_t> values(N_RUNS);
  std::iota(values.begin(), values.end(), 0);
  std::vector<uint16_t> counts(N_RUNS, RUN_LEN);
  uint32_t const total_rows = N_RUNS * RUN_LEN;  // 2112
  auto bytes                = make_rle_block<int32_t>(values, counts);
  auto out                  = decode_one<int32_t>(bytes, I32, total_rows);
  auto expected             = expand_runs<int32_t>(values, counts);
  REQUIRE(out == expected);
}

TEST_CASE("gpu_decode_table RLE - near-cap entry count", "[scan][decode][rle]")
{
  // Near the build kernel's max-entries cap; each run is 1 row.
  constexpr uint32_t N = 4000;
  std::vector<int32_t> values(N);
  std::iota(values.begin(), values.end(), 1);
  std::vector<uint16_t> counts(N, 1);
  auto bytes = make_rle_block<int32_t>(values, counts);
  auto out   = decode_one<int32_t>(bytes, I32, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == static_cast<int32_t>(i + 1));
}

TEST_CASE("gpu_decode_table RLE - multi-segment column", "[scan][decode][rle]")
{
  // Two segments with different run shapes land in their own row ranges.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  auto bytes_a = make_rle_block<int32_t>({7}, {50});
  auto bytes_b = make_rle_block<int32_t>({100, 200}, {30, 20});  // 50 rows total
  rmm::device_buffer d_a(bytes_a.data(), bytes_a.size(), stream.view());
  rmm::device_buffer d_b(bytes_b.data(), bytes_b.size(), stream.view());

  gpu_column_decode_input col;
  col.out_type   = I32;
  col.total_rows = 100;
  col.data.push_back(
    {CompressionType::COMPRESSION_RLE,
     {gpu_segment_desc{
        static_cast<uint8_t const*>(d_a.data()), static_cast<uint32_t>(d_a.size()), 0, 50},
      gpu_segment_desc{
        static_cast<uint8_t const*>(d_b.data()), static_cast<uint32_t>(d_b.size()), 50, 50}}});

  auto t   = gpu_decode_table({col}, stream.view(), mr);
  auto out = download<int32_t>(t->get_column(0).view().data<int32_t>(), 100, stream.value());
  for (uint32_t i = 0; i < 50; ++i)
    REQUIRE(out[i] == 7);
  for (uint32_t i = 0; i < 30; ++i)
    REQUIRE(out[50 + i] == 100);
  for (uint32_t i = 0; i < 20; ++i)
    REQUIRE(out[80 + i] == 200);
}

TEST_CASE("gpu_decode_table RLE - segment with on-disk padding", "[scan][decode][rle]")
{
  // DuckDB pads values→counts to alignment, so rle_count_offset can exceed
  // 8 + entry_count*sizeof(T). The parser must trust the header value, not
  // compute counts position from values size.
  uint64_t header_size  = RLE_HEADER_SIZE;
  uint64_t values_bytes = 2 * sizeof(int32_t);
  uint64_t pad_bytes    = 4;
  uint64_t off          = header_size + values_bytes + pad_bytes;
  uint64_t counts_bytes = 2 * sizeof(uint16_t);

  std::vector<uint8_t> bytes(off + counts_bytes, 0);
  std::memcpy(bytes.data(), &off, sizeof(off));
  int32_t values[2] = {77, 88};
  std::memcpy(bytes.data() + header_size, values, values_bytes);
  uint16_t counts[2] = {10, 20};
  std::memcpy(bytes.data() + off, counts, counts_bytes);

  auto out = decode_one<int32_t>(bytes, I32, 30);
  for (uint32_t i = 0; i < 10; ++i)
    REQUIRE(out[i] == 77);
  for (uint32_t i = 0; i < 20; ++i)
    REQUIRE(out[10 + i] == 88);
}

TEST_CASE("gpu_decode_table RLE - trailing slack past counts decodes correctly",
          "[scan][decode][rle]")
{
  // DuckDB's block allocator gives the last segment in a 256 KB block all
  // remaining unused space — bytes past the real RLE data are zero-fill slack.
  // The kernel must bound iteration on the values-region size; otherwise it
  // walks the slack, hits a zero count, and zero-fills the column output.
  std::vector<int32_t> values{1, 2, 3};
  std::vector<uint16_t> counts{10, 20, 30};
  auto bytes = sirius::test::decode::rle::make_rle_block<int32_t>(values, counts);
  bytes.resize(bytes.size() + 4096, 0);

  auto out      = decode_one<int32_t>(bytes, I32, 60);
  auto expected = expand_runs<int32_t>(values, counts);
  REQUIRE(out == expected);
}

TEST_CASE("gpu_decode_table RLE - narrow-type alignment padding decodes correctly",
          "[scan][decode][rle]")
{
  // Narrow types (sizeof(T)<8) get 0..7 bytes of alignment padding between
  // values and counts; worst case uint8 with 3 entries → 5 bytes. 0xFF
  // poison after the segment proves the values-region bound + match-vs-
  // malformed reordering absorb any over-read into the tile.
  std::vector<uint8_t> values{11, 22, 33};
  std::vector<uint16_t> counts{5, 10, 15};  // sum = 30
  auto seg                 = sirius::test::decode::rle::make_rle_block<uint8_t>(values, counts);
  uint32_t const seg_bytes = static_cast<uint32_t>(seg.size());
  std::vector<uint8_t> bytes(seg_bytes + 64, 0xFF);
  std::memcpy(bytes.data(), seg.data(), seg_bytes);

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 30;
  gpu_column_decode_input col;
  col.out_type   = U8;
  col.total_rows = total_rows;
  col.has_nulls  = false;
  col.data.push_back(
    {CompressionType::COMPRESSION_RLE,
     {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()), seg_bytes, 0, total_rows}}});

  auto t   = gpu_decode_table({col}, stream.view(), mr);
  auto out = download<uint8_t>(t->get_column(0).view().data<uint8_t>(), total_rows, stream.value());
  auto expected = expand_runs<uint8_t>(values, counts);
  REQUIRE(out == expected);
}

// Defensive guards: pre-fill output with a 0xCC canary, call decode_rle_data
// directly (skipping the dispatcher's allocate-fresh path), then assert
// every byte is zero.

namespace {
inline std::vector<int32_t> decode_invalid_with_canary(rmm::cuda_stream& stream,
                                                       rmm::mr::cuda_async_memory_resource& mr,
                                                       gpu_codec_run const& run,
                                                       uint32_t total_rows)
{
  std::vector<uint8_t> canary(size_t{total_rows} * sizeof(int32_t), 0xCC);
  rmm::device_buffer d_out(canary.data(), canary.size(), stream.view());
  decode_rle_data(
    run, static_cast<uint8_t*>(d_out.data()), I32, sizeof(int32_t), stream.view(), mr);
  return download<int32_t>(d_out.data(), total_rows, stream.value());
}
}  // namespace

TEST_CASE("gpu_decode_table RLE - many-entry segment scans correctly", "[scan][decode][rle]")
{
  // > BUILD_TILE_ENTRIES (4096) but well below RLE_BUILD_MAX_ENTRIES.
  // Exercises the multi-tile build path AND the gmem-cumsum expand path.
  constexpr uint32_t N = 12000;
  std::vector<int32_t> values(N);
  std::iota(values.begin(), values.end(), 1);
  std::vector<uint16_t> counts(N, 1);
  auto bytes = make_rle_block<int32_t>(values, counts);
  auto out   = decode_one<int32_t>(bytes, I32, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == static_cast<int32_t>(i + 1));
}

TEST_CASE("gpu_decode_table RLE - over-cap entry count zero-fills",
          "[scan][decode][rle][defensive]")
{
  // Exceeds RLE_BUILD_MAX_ENTRIES (90112) — a synthetic case beyond what
  // DuckDB ever writes (max ~87K for T=int8, ~26K for T=int64).
  constexpr uint32_t N = 100000;
  std::vector<int32_t> values(N);
  std::iota(values.begin(), values.end(), 1);
  std::vector<uint16_t> counts(N, 1);
  auto bytes = make_rle_block<int32_t>(values, counts);

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());
  gpu_codec_run run{
    CompressionType::COMPRESSION_RLE,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_invalid_with_canary(stream, mr, run, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - rle_count_offset past segment zero-fills",
          "[scan][decode][rle][defensive]")
{
  // rle_count_offset > segment_size → host parser refuses, kernel zero-fills.
  std::vector<uint8_t> bytes(64, 0);
  uint64_t bogus = 1ULL << 30;
  std::memcpy(bytes.data(), &bogus, sizeof(bogus));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 32;
  gpu_codec_run run{CompressionType::COMPRESSION_RLE,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - rle_count_offset below header zero-fills",
          "[scan][decode][rle][defensive]")
{
  // rle_count_offset < RLE_HEADER_SIZE (8) overlaps the header itself.
  std::vector<uint8_t> bytes(64, 0);
  uint64_t bogus = 4;
  std::memcpy(bytes.data(), &bogus, sizeof(bogus));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 16;
  gpu_codec_run run{CompressionType::COMPRESSION_RLE,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - count walk underflows row_count zero-fills",
          "[scan][decode][rle][defensive]")
{
  // counts sum < row_count → walk runs out, segment refused.
  auto bytes = make_rle_block<int32_t>({1, 2}, {5, 5});  // sum = 10

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 100;  // claim more than counts provide
  gpu_codec_run run{CompressionType::COMPRESSION_RLE,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - count walk overflows row_count zero-fills",
          "[scan][decode][rle][defensive]")
{
  // counts sum overshoots row_count; last count straddles the boundary.
  auto bytes = make_rle_block<int32_t>({1, 2}, {50, 60});  // sum = 110

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 100;
  gpu_codec_run run{CompressionType::COMPRESSION_RLE,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - zero count with sum underflow zero-fills",
          "[scan][decode][rle][defensive]")
{
  // Interior zero count + sum never reaches n_segment_rows → malformed.
  // (Zeros after the sum-match are slack — handled by match-over-malformed,
  // not this check.)
  auto bytes = make_rle_block<int32_t>({1, 99, 2}, {10, 0, 5});  // sum = 15

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 20;
  gpu_codec_run run{CompressionType::COMPRESSION_RLE,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - segment too small for header zero-fills",
          "[scan][decode][rle][defensive]")
{
  // Segment shorter than RLE_HEADER_SIZE — host parser can't even read the
  // offset. Kernel still zero-fills the descriptor's row range.
  std::vector<uint8_t> bytes(4, 0);

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 8;
  gpu_codec_run run{CompressionType::COMPRESSION_RLE,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - unsupported type_size throws", "[scan][decode][rle][defensive]")
{
  // 16-byte types are out of scope for this dispatcher.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  auto bytes = make_rle_block<int32_t>({1}, {8});
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  auto const DEC128 = cudf::data_type{cudf::type_id::DECIMAL128, /*scale=*/-2};
  gpu_column_decode_input col;
  col.out_type   = DEC128;
  col.total_rows = 8;
  col.data.push_back(
    {CompressionType::COMPRESSION_RLE,
     {gpu_segment_desc{
       static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, 8}}});
  REQUIRE_THROWS_WITH(gpu_decode_table({col}, stream.view(), mr),
                      Catch::Contains("viability invariant violated"));
}

// Bench-scale correctness checks. The microbenches in bench_decode_codecs.cpp
// discard their output, so a silent miscompile would still report multi-
// hundred-GiB/s throughput. Each codec PR should ship a verify TEST_CASE
// per bench shape using `verify_decoded_column` from decode_test_utils.hpp.

using ::sirius::test::decode::verify_decoded_column;

namespace {

template <typename T>
auto build_uniform_runs_column(uint32_t n_runs,
                               uint16_t run_len,
                               uint32_t n_segs,
                               cudf::data_type type,
                               rmm::cuda_stream& stream,
                               std::vector<rmm::device_buffer>& bufs)
{
  using ::sirius::test::decode::rle::make_uniform_runs;
  uint32_t const seg_rows = n_runs * run_len;
  std::vector<gpu_segment_desc> segs;
  bufs.reserve(n_segs);
  segs.reserve(n_segs);
  auto seg_bytes = make_uniform_runs<T>(n_runs, run_len);
  for (uint32_t i = 0; i < n_segs; ++i) {
    bufs.emplace_back(seg_bytes.data(), seg_bytes.size(), stream.view());
    segs.push_back({static_cast<uint8_t const*>(bufs.back().data()),
                    static_cast<uint32_t>(bufs.back().size()),
                    i * seg_rows,
                    seg_rows});
  }
  gpu_column_decode_input col;
  col.out_type   = type;
  col.total_rows = n_segs * seg_rows;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_RLE, segs});
  return col;
}

}  // namespace

TEST_CASE("gpu_decode_table RLE bench-scale - long_runs verify", "[scan][decode][rle][verify]")
{
  // Mirrors `bench RLE int64 long_runs`: 1092 segments × 16 entries × 7680.
  constexpr uint32_t N_RUNS = 16, N_SEGS = 1092;
  constexpr uint16_t RUN_LEN  = 7680;
  constexpr uint32_t SEG_ROWS = N_RUNS * RUN_LEN;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  std::vector<rmm::device_buffer> bufs;
  auto col = build_uniform_runs_column<int64_t>(N_RUNS, RUN_LEN, N_SEGS, I64, stream, bufs);

  verify_decoded_column<int64_t>(stream.view(), mr, col, [](uint32_t r) -> int64_t {
    return static_cast<int64_t>((r % SEG_ROWS) / RUN_LEN);
  });
}

TEST_CASE("gpu_decode_table RLE bench-scale - medium_runs verify", "[scan][decode][rle][verify]")
{
  constexpr uint32_t N_RUNS = 1024, N_SEGS = 1092;
  constexpr uint16_t RUN_LEN  = 120;
  constexpr uint32_t SEG_ROWS = N_RUNS * RUN_LEN;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  std::vector<rmm::device_buffer> bufs;
  auto col = build_uniform_runs_column<int64_t>(N_RUNS, RUN_LEN, N_SEGS, I64, stream, bufs);

  verify_decoded_column<int64_t>(stream.view(), mr, col, [](uint32_t r) -> int64_t {
    return static_cast<int64_t>((r % SEG_ROWS) / RUN_LEN);
  });
}

TEST_CASE("gpu_decode_table RLE bench-scale - short_runs verify (gmem path)",
          "[scan][decode][rle][verify]")
{
  // Bumped to 5000 entries so it exceeds RLE_SMEM_MAX_ENTRIES (4096) and
  // exercises the gmem-cumsum expand fallback at scale.
  constexpr uint32_t N_RUNS = 5000, N_SEGS = 200;
  constexpr uint16_t RUN_LEN  = 24;
  constexpr uint32_t SEG_ROWS = N_RUNS * RUN_LEN;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  std::vector<rmm::device_buffer> bufs;
  auto col = build_uniform_runs_column<int32_t>(N_RUNS, RUN_LEN, N_SEGS, I32, stream, bufs);

  verify_decoded_column<int32_t>(stream.view(), mr, col, [](uint32_t r) -> int32_t {
    return static_cast<int32_t>((r % SEG_ROWS) / RUN_LEN);
  });
}

TEST_CASE("gpu_decode_table RLE bench-scale - pareto_runs verify", "[scan][decode][rle][verify]")
{
  using ::sirius::test::decode::rle::make_pareto_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = 122880;
  constexpr uint32_t N_SEGS   = 32;

  std::vector<rmm::device_buffer> bufs;
  std::vector<gpu_segment_desc> segs;
  std::vector<int64_t> expected;
  expected.reserve(static_cast<size_t>(SEG_ROWS) * N_SEGS);
  bufs.reserve(N_SEGS);
  segs.reserve(N_SEGS);

  for (uint32_t s = 0; s < N_SEGS; ++s) {
    auto seg_bytes = make_pareto_runs<int64_t>(SEG_ROWS, /*seed=*/s + 1, 400.0);
    bufs.emplace_back(seg_bytes.data(), seg_bytes.size(), stream.view());
    segs.push_back({static_cast<uint8_t const*>(bufs.back().data()),
                    static_cast<uint32_t>(bufs.back().size()),
                    s * SEG_ROWS,
                    SEG_ROWS});

    // Reproduce the bench's LCG path for the expected expansion.
    uint32_t lcg       = (s + 1) ? (s + 1) : 0x9E3779B9u;
    uint32_t emitted   = 0;
    int64_t next_value = 0;
    while (emitted < SEG_ROWS) {
      lcg              = lcg * 1664525u + 1013904223u;
      double const u   = static_cast<double>(lcg) / static_cast<double>(0xFFFFFFFFu);
      double const x   = 400.0 / std::pow(1.0 - 0.999 * u, 1.0 / 1.5);
      uint16_t run_len = static_cast<uint16_t>(std::min<double>(2048.0, std::max(1.0, x)));
      if (emitted + run_len > SEG_ROWS) run_len = static_cast<uint16_t>(SEG_ROWS - emitted);
      for (uint16_t k = 0; k < run_len; ++k)
        expected.push_back(next_value);
      ++next_value;
      emitted += run_len;
    }
  }

  gpu_column_decode_input col;
  col.out_type   = I64;
  col.total_rows = N_SEGS * SEG_ROWS;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_RLE, segs});

  verify_decoded_column<int64_t>(
    stream.view(), mr, col, [&](uint32_t r) -> int64_t { return expected[r]; });
}
