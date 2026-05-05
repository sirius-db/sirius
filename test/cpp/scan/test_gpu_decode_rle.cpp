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

//===----------------------------------------------------------------------===//
// RLE decode tests. Each happy-path case stages a synthetic segment that
// matches DuckDB's on-disk layout (rle_count_offset header → values → counts)
// and asserts the decoded output. Defensive guards (corrupt offset, count
// underflow / overflow, zero count) are pinned by their own cases so the
// kernel's deterministic-zero-fill contract doesn't silently regress if
// upstream filtering ever changes.
//===----------------------------------------------------------------------===//

#include "scan/decode_test_utils.hpp"
#include "scan/rle_synth.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_rle.cuh>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

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

auto const I8  = cudf::data_type{cudf::type_id::INT8};
auto const I16 = cudf::data_type{cudf::type_id::INT16};
auto const I32 = cudf::data_type{cudf::type_id::INT32};
auto const I64 = cudf::data_type{cudf::type_id::INT64};
auto const U8  = cudf::data_type{cudf::type_id::UINT8};

/// Wrap one block's bytes into a single-segment column and decode it.
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

/// Build the expected expanded vector from values + counts. Mirrors the
/// host-side prefix-sum walk that decode_rle_data does internally.
template <typename T>
std::vector<T> expand_runs(std::vector<T> const& values,
                           std::vector<uint16_t> const& counts)
{
  std::vector<T> out;
  for (size_t i = 0; i < values.size(); ++i) {
    for (uint16_t c = 0; c < counts[i]; ++c) out.push_back(values[i]);
  }
  return out;
}

}  // namespace

TEST_CASE("gpu_decode_table RLE - single-entry run broadcasts one value",
          "[scan][decode][rle]")
{
  SECTION("int32 100 rows, value 42")
  {
    auto bytes = make_rle_block<int32_t>({42}, {100});
    auto out   = decode_one<int32_t>(bytes, I32, 100);
    REQUIRE(out.size() == 100);
    for (auto v : out) REQUIRE(v == 42);
  }

  SECTION("int64 50 rows, large value")
  {
    int64_t v  = 5'000'000'000LL;
    auto bytes = make_rle_block<int64_t>({v}, {50});
    auto out   = decode_one<int64_t>(bytes, I64, 50);
    for (auto x : out) REQUIRE(x == v);
  }

  SECTION("uint8 33 rows (non-warp-aligned)")
  {
    auto bytes = make_rle_block<uint8_t>({200}, {33});
    auto out   = decode_one<uint8_t>(bytes, U8, 33);
    for (auto v : out) REQUIRE(v == 200);
  }
}

TEST_CASE("gpu_decode_table RLE - multi-entry runs are expanded in order",
          "[scan][decode][rle]")
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
  // 12 runs × 256 rows each = 3072 rows → 2 CTAs (RLE_ROWS_PER_CHUNK=2048).
  // Verifies the binary search is correct across CTA boundaries — the
  // second CTA's local_row_start = 2048 must still resolve to the right
  // entry.
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

TEST_CASE("gpu_decode_table RLE - large entry_count exercises gmem path",
          "[scan][decode][rle]")
{
  // 5000 entries > RLE_SMEM_MAX_ENTRIES (4096) → kernel falls through to
  // gmem-resident cumsum. Each run is 1 row so output is a sequence.
  constexpr uint32_t N = 5000;
  std::vector<int32_t> values(N);
  std::iota(values.begin(), values.end(), 1);
  std::vector<uint16_t> counts(N, 1);
  auto bytes = make_rle_block<int32_t>(values, counts);
  auto out   = decode_one<int32_t>(bytes, I32, N);
  for (uint32_t i = 0; i < N; ++i) REQUIRE(out[i] == static_cast<int32_t>(i + 1));
}

TEST_CASE("gpu_decode_table RLE - multi-segment column", "[scan][decode][rle]")
{
  // Two segments encoded with different run shapes; verify each lands in
  // its own row range via segment.row_offset.
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
  for (uint32_t i = 0; i < 50; ++i) REQUIRE(out[i] == 7);
  for (uint32_t i = 0; i < 30; ++i) REQUIRE(out[50 + i] == 100);
  for (uint32_t i = 0; i < 20; ++i) REQUIRE(out[80 + i] == 200);
}

TEST_CASE("gpu_decode_table RLE - segment with on-disk padding", "[scan][decode][rle]")
{
  // DuckDB's encoder aligns the values→counts boundary; rle_count_offset
  // can be greater than 8 + entry_count*sizeof(T). Synthesise that shape
  // (4 bytes of zero padding between values and counts) to confirm the
  // host parser reads counts at the offset it finds in the header rather
  // than computing it from an assumed layout.
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
  for (uint32_t i = 0; i < 10; ++i) REQUIRE(out[i] == 77);
  for (uint32_t i = 0; i < 20; ++i) REQUIRE(out[10 + i] == 88);
}

//===----------------------------------------------------------------------===//
// Defensive guards. Each test pre-fills the output with a 0xCC canary and
// invokes `decode_rle_data` directly (bypassing the dispatcher's allocate-
// fresh path) so we can prove the kernel deterministically zero-fills on
// malformed metadata, rather than relying on REQUIRE_NOTHROW alone.
//===----------------------------------------------------------------------===//

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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - count walk underflows row_count zero-fills",
          "[scan][decode][rle][defensive]")
{
  // counts sum to less than row_count → walk runs out before reaching
  // row_count. Host parser refuses, kernel zero-fills.
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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - count walk overflows row_count zero-fills",
          "[scan][decode][rle][defensive]")
{
  // counts sum to more than row_count and the last count straddles the
  // boundary (final total != row_count). Refused.
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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - zero count inside walk zero-fills",
          "[scan][decode][rle][defensive]")
{
  // DuckDB's encoder never emits zero counts; if we see one, treat the
  // segment as malformed instead of infinite-looping or under-filling.
  auto bytes = make_rle_block<int32_t>({1, 99, 2}, {10, 0, 10});

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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table RLE - unsupported type_size throws",
          "[scan][decode][rle][defensive]")
{
  // DECIMAL128 is 16-byte storage; RLE on 128-bit values is refused by the
  // viability walker upstream, kernel throws as a defensive backstop.
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
