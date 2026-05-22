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
// BITPACKING decode tests. Each case stages synthetic segment bytes that
// match DuckDB's on-disk layout (metadata trailer + per-mode header +
// optional packed stream) and asserts the decoded output. Defensive guards
// (INVALID mode, corrupt metadata_end, bogus width) are pinned by their own
// cases so they don't silently become dead code if upstream filtering ever
// changes — the kernel must treat malformed input as a no-op.
//===----------------------------------------------------------------------===//

#include "scan/bitpacking_synth.hpp"
#include "scan/decode_test_utils.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_bitpacking.cuh>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using duckdb::CompressionType;
using sirius::cuda::scan::BitpackingMode;
using sirius::cuda::scan::BP_META_GROUP_SIZE;
using sirius::cuda::scan::decode_bitpacking_data;
using sirius::cuda::scan::gpu_codec_run;
using sirius::cuda::scan::gpu_column_decode_input;
using sirius::cuda::scan::gpu_decode_table;
using sirius::cuda::scan::gpu_segment_desc;
using sirius::test::decode::download;
using sirius::test::decode::one_codec_column;
using sirius::test::decode::segment;
using sirius::test::decode::bitpacking::make_constant_block;
using sirius::test::decode::bitpacking::make_constant_delta_block;
using sirius::test::decode::bitpacking::make_delta_for_block;
using sirius::test::decode::bitpacking::make_for_block;
using sirius::test::decode::bitpacking::pack_values;

namespace {

auto const I32 = cudf::data_type{cudf::type_id::INT32};
auto const I64 = cudf::data_type{cudf::type_id::INT64};
auto const U8  = cudf::data_type{cudf::type_id::UINT8};
auto const U16 = cudf::data_type{cudf::type_id::UINT16};

// Synthetic-segment builders moved to scan/bitpacking_synth.hpp so the
// codec microbenches can use them too.

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
  col.data.push_back({CompressionType::COMPRESSION_BITPACKING,
                      {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                        static_cast<uint32_t>(d_seg.size()),
                                        0,
                                        row_count}}});

  auto t = gpu_decode_table({col}, stream.view(), mr);
  REQUIRE(t->num_rows() == static_cast<cudf::size_type>(row_count));
  return download<T>(t->get_column(0).view().data<T>(), row_count, stream.value());
}

}  // namespace

TEST_CASE("gpu_decode_table BITPACKING - CONSTANT broadcasts value", "[scan][decode][bitpacking]")
{
  SECTION("int32 value 42, 100 rows")
  {
    auto bytes = make_constant_block<int32_t>(42);
    auto out   = decode_one<int32_t>(bytes, I32, 100);
    REQUIRE(out.size() == 100);
    for (auto v : out)
      REQUIRE(v == 42);
  }

  SECTION("int64 large value, 257 rows (vectorised + scalar tail)")
  {
    int64_t v  = 5'000'000'000LL;
    auto bytes = make_constant_block<int64_t>(v);
    auto out   = decode_one<int64_t>(bytes, I64, 257);
    REQUIRE(out.size() == 257);
    for (auto x : out)
      REQUIRE(x == v);
  }

  SECTION("uint8 0x5A, 200 rows (TPV=16 lanes per int4 broadcast)")
  {
    // Narrowest type — sizeof(T)=1 → TPV=16, the broadest int4 broadcast layout.
    // Exercises the static_assert(sizeof(vec_t) % sizeof(T) == 0) at sizeof(T)=1
    // and the 12 full int4 stores + 8 scalar tail elements.
    auto bytes = make_constant_block<uint8_t>(0x5A);
    auto out   = decode_one<uint8_t>(bytes, U8, 200);
    for (auto v : out)
      REQUIRE(v == 0x5A);
  }
}

TEST_CASE("gpu_decode_table BITPACKING - CONSTANT_DELTA arithmetic progression",
          "[scan][decode][bitpacking]")
{
  SECTION("int32 frame=10 delta=3, 50 rows")
  {
    auto bytes = make_constant_delta_block<int32_t>(10, 3);
    auto out   = decode_one<int32_t>(bytes, I32, 50);
    REQUIRE(out.size() == 50);
    for (uint32_t i = 0; i < 50; ++i)
      REQUIRE(out[i] == 10 + 3 * static_cast<int32_t>(i));
  }

  SECTION("int64 frame=-1000 delta=7, 1024 rows (full-warp coverage)")
  {
    auto bytes = make_constant_delta_block<int64_t>(-1000, 7);
    auto out   = decode_one<int64_t>(bytes, I64, 1024);
    REQUIRE(out.size() == 1024);
    for (uint32_t i = 0; i < 1024; ++i)
      REQUIRE(out[i] == -1000 + 7 * static_cast<int64_t>(i));
  }
}

TEST_CASE("gpu_decode_table BITPACKING - FOR unpacks frame + packed deltas",
          "[scan][decode][bitpacking]")
{
  SECTION("int32 width=5, frame=100, 32 rows (one warp)")
  {
    std::vector<int32_t> deltas(32);
    for (int32_t i = 0; i < 32; ++i)
      deltas[i] = i;
    auto bytes = make_for_block<int32_t>(100, 5, deltas);
    auto out   = decode_one<int32_t>(bytes, I32, 32);
    for (uint32_t i = 0; i < 32; ++i)
      REQUIRE(out[i] == 100 + static_cast<int32_t>(i));
  }

  SECTION("int32 width=12, frame=-50, 100 rows (multi-warp, non-aligned tail)")
  {
    std::vector<int32_t> deltas(100);
    for (int32_t i = 0; i < 100; ++i)
      deltas[i] = (i * 7) % 4096;
    auto bytes = make_for_block<int32_t>(-50, 12, deltas);
    auto out   = decode_one<int32_t>(bytes, I32, 100);
    for (uint32_t i = 0; i < 100; ++i)
      REQUIRE(out[i] == -50 + ((i * 7) % 4096));
  }

  SECTION("int64 width=33, 64 rows (3-word unpack path)")
  {
    // width > 32 with non-zero bit_off forces unpack_value's third-word read.
    std::vector<int64_t> deltas(64);
    for (int64_t i = 0; i < 64; ++i)
      deltas[i] = (i * 0x1234567LL) & ((1LL << 33) - 1);
    auto bytes = make_for_block<int64_t>(1'000'000LL, 33, deltas);
    auto out   = decode_one<int64_t>(bytes, I64, 64);
    for (uint32_t i = 0; i < 64; ++i) {
      int64_t expected =
        1'000'000LL + ((static_cast<int64_t>(i) * 0x1234567LL) & ((1LL << 33) - 1));
      REQUIRE(out[i] == expected);
    }
  }

  SECTION("uint16 width=10, 200 rows")
  {
    std::vector<uint16_t> deltas(200);
    for (uint16_t i = 0; i < 200; ++i)
      deltas[i] = (i * 3) % 1024;
    auto bytes = make_for_block<uint16_t>(500, 10, deltas);
    auto out   = decode_one<uint16_t>(bytes, U16, 200);
    for (uint32_t i = 0; i < 200; ++i)
      REQUIRE(out[i] == 500u + (i * 3u) % 1024u);
  }

  SECTION("int32 width=1 (smallest non-degenerate width)")
  {
    // Single-bit packing — 64 alternating 0/1 values into two 32-bit words.
    // Pins the unpack_value mask compute at the smallest width.
    std::vector<int32_t> deltas(64);
    for (int32_t i = 0; i < 64; ++i)
      deltas[i] = i & 1;
    auto bytes = make_for_block<int32_t>(50, 1, deltas);
    auto out   = decode_one<int32_t>(bytes, I32, 64);
    for (uint32_t i = 0; i < 64; ++i)
      REQUIRE(out[i] == 50 + (static_cast<int32_t>(i) & 1));
  }

  SECTION("int32 width=32 (max-width, no inter-word straddle)")
  {
    // Each value occupies a full word — bit_off is always 0 — so the straddle
    // branch in unpack_value never fires. Pins the upper boundary of the width
    // validator (kernel rejects width > sizeof(T)*8 at the bound, not at it).
    std::vector<int32_t> deltas(33);
    for (int32_t i = 0; i < 33; ++i)
      deltas[i] = 0x10000000 + i;
    auto bytes = make_for_block<int32_t>(0, 32, deltas);
    auto out   = decode_one<int32_t>(bytes, I32, 33);
    for (uint32_t i = 0; i < 33; ++i)
      REQUIRE(out[i] == 0x10000000 + static_cast<int32_t>(i));
  }

  SECTION("int64 width=64 (max-width, `width >= 64` mask branch)")
  {
    // unpack_value selects `(width >= 64) ? ~uint64_t{0} : make_mask(width)`.
    // width=64 is the explicit "no mask" path that lets the full 64-bit value
    // through unchanged.
    std::vector<int64_t> deltas(40);
    for (int64_t i = 0; i < 40; ++i)
      deltas[i] = (1LL << 50) + i;
    auto bytes = make_for_block<int64_t>(0, 64, deltas);
    auto out   = decode_one<int64_t>(bytes, I64, 40);
    for (uint32_t i = 0; i < 40; ++i)
      REQUIRE(out[i] == (1LL << 50) + static_cast<int64_t>(i));
  }

  SECTION("int64 width=63 (true 3-word straddle in unpack_value)")
  {
    // unpack_value's third-word read fires when bit_off > 0 AND
    // bit_off + width > 64. The existing "width=33" case can't satisfy this
    // (max bit_off=31 → 31+33=64, not strictly >64). width=63 gives bit_off
    // values that climb past 0 → 3-word path is actually exercised.
    std::vector<int64_t> deltas(100);
    for (int64_t i = 0; i < 100; ++i)
      deltas[i] = (i * 0x123456789LL) & (static_cast<int64_t>((1ULL << 63) - 1ULL));
    auto bytes = make_for_block<int64_t>(0, 63, deltas);
    auto out   = decode_one<int64_t>(bytes, I64, 100);
    for (uint32_t i = 0; i < 100; ++i) {
      int64_t expected =
        (static_cast<int64_t>(i) * 0x123456789LL) & (static_cast<int64_t>((1ULL << 63) - 1ULL));
      REQUIRE(out[i] == expected);
    }
  }
}

TEST_CASE("gpu_decode_table BITPACKING - DELTA_FOR prefix-sums packed deltas",
          "[scan][decode][bitpacking]")
{
  SECTION("int32 width=4, frame=0, delta_offset=10, 16 rows")
  {
    std::vector<int32_t> packed{1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1};
    auto bytes      = make_delta_for_block<int32_t>(0, 10, 4, packed);
    auto out        = decode_one<int32_t>(bytes, I32, 16);
    int32_t running = 10;
    for (uint32_t i = 0; i < 16; ++i) {
      running += packed[i];
      REQUIRE(out[i] == running);
    }
  }

  SECTION("int64 width=8, frame=1, delta_offset=1000, 1000 rows (BlockScan stress)")
  {
    std::vector<int64_t> packed(1000);
    for (size_t i = 0; i < packed.size(); ++i)
      packed[i] = (i * 17 + 3) & 0xFF;
    auto bytes      = make_delta_for_block<int64_t>(1, 1000, 8, packed);
    auto out        = decode_one<int64_t>(bytes, I64, 1000);
    int64_t running = 1000;
    for (uint32_t i = 0; i < 1000; ++i) {
      running += 1 + packed[i];
      REQUIRE(out[i] == running);
    }
  }
}

TEST_CASE("gpu_decode_table BITPACKING - multi-segment column", "[scan][decode][bitpacking]")
{
  // Two segments encoded with different modes — verify each lands in its
  // own row range via segment.row_offset.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  auto bytes_a = make_constant_block<int32_t>(7);
  auto bytes_b = make_for_block<int32_t>(/*frame=*/100, /*width=*/4, std::vector<int32_t>(50, 3));
  rmm::device_buffer d_a(bytes_a.data(), bytes_a.size(), stream.view());
  rmm::device_buffer d_b(bytes_b.data(), bytes_b.size(), stream.view());

  gpu_column_decode_input col;
  col.out_type   = I32;
  col.total_rows = 100;
  col.data.push_back(
    {CompressionType::COMPRESSION_BITPACKING,
     {gpu_segment_desc{
        static_cast<uint8_t const*>(d_a.data()), static_cast<uint32_t>(d_a.size()), 0, 50},
      gpu_segment_desc{
        static_cast<uint8_t const*>(d_b.data()), static_cast<uint32_t>(d_b.size()), 50, 50}}});

  auto t   = gpu_decode_table({col}, stream.view(), mr);
  auto out = download<int32_t>(t->get_column(0).view().data<int32_t>(), 100, stream.value());
  for (uint32_t i = 0; i < 50; ++i)
    REQUIRE(out[i] == 7);
  for (uint32_t i = 0; i < 50; ++i)
    REQUIRE(out[50 + i] == 103);
}

TEST_CASE("gpu_decode_table BITPACKING - CONSTANT/CONSTANT_DELTA at unaligned dst row offset",
          "[scan][decode][bitpacking]")
{
  // CONSTANT and CONSTANT_DELTA paths do 16B vector stores to
  // out = d_output + global_row_offset. Alignment holds within a single
  // segment (group offsets are multiples of BP_META_GROUP_SIZE = 2048) but
  // breaks across stacked segments whose cumulative row count is not a
  // multiple of sizeof(vec_t)/sizeof(T). The kernel must detect this and
  // fall back to scalar stores; otherwise the vector store faults with
  // cudaErrorMisalignedAddress.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  // Segment A: 50 int32 rows = 200 bytes, not a multiple of 16. Segment B's
  // destination then starts at byte offset 200 — unaligned for vec_t stores.
  auto bytes_a = make_for_block<int32_t>(/*frame=*/0, /*width=*/4, std::vector<int32_t>(50, 1));
  rmm::device_buffer d_a(bytes_a.data(), bytes_a.size(), stream.view());

  SECTION("CONSTANT at unaligned row offset")
  {
    auto bytes_b = make_constant_block<int32_t>(77);
    rmm::device_buffer d_b(bytes_b.data(), bytes_b.size(), stream.view());
    gpu_column_decode_input col;
    col.out_type   = I32;
    col.total_rows = 100;
    col.data.push_back(
      {CompressionType::COMPRESSION_BITPACKING,
       {gpu_segment_desc{
          static_cast<uint8_t const*>(d_a.data()), static_cast<uint32_t>(d_a.size()), 0, 50},
        gpu_segment_desc{
          static_cast<uint8_t const*>(d_b.data()), static_cast<uint32_t>(d_b.size()), 50, 50}}});
    auto t   = gpu_decode_table({col}, stream.view(), mr);
    auto out = download<int32_t>(t->get_column(0).view().data<int32_t>(), 100, stream.value());
    for (uint32_t i = 0; i < 50; ++i)
      REQUIRE(out[i] == 1);
    for (uint32_t i = 0; i < 50; ++i)
      REQUIRE(out[50 + i] == 77);
  }

  SECTION("CONSTANT_DELTA at unaligned row offset")
  {
    auto bytes_b = make_constant_delta_block<int32_t>(/*frame=*/1000, /*delta=*/3);
    rmm::device_buffer d_b(bytes_b.data(), bytes_b.size(), stream.view());
    gpu_column_decode_input col;
    col.out_type   = I32;
    col.total_rows = 100;
    col.data.push_back(
      {CompressionType::COMPRESSION_BITPACKING,
       {gpu_segment_desc{
          static_cast<uint8_t const*>(d_a.data()), static_cast<uint32_t>(d_a.size()), 0, 50},
        gpu_segment_desc{
          static_cast<uint8_t const*>(d_b.data()), static_cast<uint32_t>(d_b.size()), 50, 50}}});
    auto t   = gpu_decode_table({col}, stream.view(), mr);
    auto out = download<int32_t>(t->get_column(0).view().data<int32_t>(), 100, stream.value());
    for (uint32_t i = 0; i < 50; ++i)
      REQUIRE(out[i] == 1);
    for (uint32_t i = 0; i < 50; ++i)
      REQUIRE(out[50 + i] == 1000 + static_cast<int32_t>(i) * 3);
  }
}

TEST_CASE("gpu_decode_table BITPACKING - multi-group segment", "[scan][decode][bitpacking]")
{
  // One segment that spans more than BP_META_GROUP_SIZE rows. The kernel
  // dispatches one CTA per group; each group's metadata sits in the
  // segment trailer at metadata_end - (group_idx+1)*4. Build with two
  // CONSTANT groups so we can assert per-group output without a packed
  // stream.
  using T                       = int32_t;
  constexpr uint32_t group_rows = BP_META_GROUP_SIZE;
  constexpr uint32_t total_rows = group_rows + 500;
  std::vector<uint8_t> bytes;
  // group 0: data_off = 8, holds T(7). group 1: data_off = 8 + sizeof(T),
  // holds T(99). Trailer (last entry first): group 1 entry at metadata_end-4,
  // group 0 entry at metadata_end-8.
  size_t header_bytes   = 8 + 2 * sizeof(T);
  size_t trailer_bytes  = 2 * sizeof(uint32_t);
  uint64_t metadata_end = header_bytes + trailer_bytes;
  bytes.assign(metadata_end, 0);
  std::memcpy(bytes.data(), &metadata_end, sizeof(uint64_t));
  T v0 = 7, v1 = 99;
  std::memcpy(bytes.data() + 8, &v0, sizeof(T));
  std::memcpy(bytes.data() + 8 + sizeof(T), &v1, sizeof(T));
  uint32_t entry0 = (static_cast<uint32_t>(BitpackingMode::CONSTANT) << 24) | 8u;
  uint32_t entry1 = (static_cast<uint32_t>(BitpackingMode::CONSTANT) << 24) | (8u + sizeof(T));
  // Trailer: kernel reads entry K at metadata_end - (K+1)*4.
  // Group 0 → metadata_end - 4 (entry0); group 1 → metadata_end - 8 (entry1).
  std::memcpy(bytes.data() + metadata_end - 4, &entry0, sizeof(uint32_t));
  std::memcpy(bytes.data() + metadata_end - 8, &entry1, sizeof(uint32_t));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_column_decode_input col;
  col.out_type   = I32;
  col.total_rows = total_rows;
  col.data.push_back({CompressionType::COMPRESSION_BITPACKING,
                      {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                        static_cast<uint32_t>(d_seg.size()),
                                        0,
                                        total_rows}}});

  auto t   = gpu_decode_table({col}, stream.view(), mr);
  auto out = download<int32_t>(t->get_column(0).view().data<int32_t>(), total_rows, stream.value());
  for (uint32_t i = 0; i < group_rows; ++i)
    REQUIRE(out[i] == 7);
  for (uint32_t i = 0; i < total_rows - group_rows; ++i)
    REQUIRE(out[group_rows + i] == 99);
}

TEST_CASE("gpu_decode_table BITPACKING - multi-group segment, group 1 is FOR",
          "[scan][decode][bitpacking]")
{
  // Group 0: CONSTANT, fills BP_META_GROUP_SIZE rows. Group 1: FOR, decodes
  // a packed stream — exercises the multi-group dispatcher routing each
  // group to its own decode path (prior multi-group test was CONSTANT/CONSTANT,
  // never lit up the FOR cp.async + unpack path through a non-zero
  // global_row_offset).
  using T                       = int32_t;
  constexpr uint32_t group_rows = BP_META_GROUP_SIZE;
  constexpr uint32_t g1_rows    = 500;
  constexpr uint32_t total_rows = group_rows + g1_rows;
  constexpr T g0_value          = 7;
  constexpr T g1_frame          = 100;
  constexpr uint32_t g1_width   = 8;

  std::vector<T> g1_packed(g1_rows);
  for (uint32_t i = 0; i < g1_rows; ++i)
    g1_packed[i] = static_cast<T>((i * 3 + 1) & 0xFF);

  auto packed_words           = pack_values<T>(g1_packed, g1_width);
  size_t const pkd_bytes      = packed_words.size() * sizeof(uint32_t);
  size_t const g0_data_off    = 8;
  size_t const g1_data_off    = g0_data_off + sizeof(T);
  size_t const header_end     = g1_data_off + 2 * sizeof(T) + pkd_bytes;
  uint64_t const metadata_end = header_end + 2 * sizeof(uint32_t);

  std::vector<uint8_t> bytes(metadata_end, 0);
  std::memcpy(bytes.data(), &metadata_end, sizeof(uint64_t));
  std::memcpy(bytes.data() + g0_data_off, &g0_value, sizeof(T));
  T const width_t = static_cast<T>(g1_width);
  std::memcpy(bytes.data() + g1_data_off, &g1_frame, sizeof(T));
  std::memcpy(bytes.data() + g1_data_off + sizeof(T), &width_t, sizeof(T));
  std::memcpy(bytes.data() + g1_data_off + 2 * sizeof(T), packed_words.data(), pkd_bytes);

  uint32_t const entry0 =
    (static_cast<uint32_t>(BitpackingMode::CONSTANT) << 24) | static_cast<uint32_t>(g0_data_off);
  uint32_t const entry1 =
    (static_cast<uint32_t>(BitpackingMode::FOR) << 24) | static_cast<uint32_t>(g1_data_off);
  std::memcpy(bytes.data() + metadata_end - 4, &entry0, sizeof(uint32_t));
  std::memcpy(bytes.data() + metadata_end - 8, &entry1, sizeof(uint32_t));

  auto out = decode_one<int32_t>(bytes, I32, total_rows);
  for (uint32_t i = 0; i < group_rows; ++i)
    REQUIRE(out[i] == g0_value);
  for (uint32_t i = 0; i < g1_rows; ++i)
    REQUIRE(out[group_rows + i] == g1_frame + g1_packed[i]);
}

TEST_CASE(
  "gpu_decode_table BITPACKING - multi-group segment, group 1 is DELTA_FOR with "
  "non-trivial delta_offset",
  "[scan][decode][bitpacking]")
{
  // Group 0: CONSTANT. Group 1: DELTA_FOR with a non-zero delta_offset that
  // gets prefix-summed into every output row. Catches both the multi-group
  // routing AND the WarpScan + delta_offset bias path on a non-zero
  // global_row_offset.
  using T                       = int32_t;
  constexpr uint32_t group_rows = BP_META_GROUP_SIZE;
  constexpr uint32_t g1_rows    = 500;
  constexpr uint32_t total_rows = group_rows + g1_rows;
  constexpr T g0_value          = 0;
  constexpr T g1_frame          = 2;
  constexpr T g1_delta_offset   = 1000;
  constexpr uint32_t g1_width   = 8;

  std::vector<T> g1_packed(g1_rows);
  for (uint32_t i = 0; i < g1_rows; ++i)
    g1_packed[i] = static_cast<T>((i * 5 + 7) & 0xFF);

  auto packed_words           = pack_values<T>(g1_packed, g1_width);
  size_t const pkd_bytes      = packed_words.size() * sizeof(uint32_t);
  size_t const g0_data_off    = 8;
  size_t const g1_data_off    = g0_data_off + sizeof(T);
  size_t const header_end     = g1_data_off + 3 * sizeof(T) + pkd_bytes;
  uint64_t const metadata_end = header_end + 2 * sizeof(uint32_t);

  std::vector<uint8_t> bytes(metadata_end, 0);
  std::memcpy(bytes.data(), &metadata_end, sizeof(uint64_t));
  std::memcpy(bytes.data() + g0_data_off, &g0_value, sizeof(T));
  T const width_t = static_cast<T>(g1_width);
  std::memcpy(bytes.data() + g1_data_off, &g1_frame, sizeof(T));
  std::memcpy(bytes.data() + g1_data_off + sizeof(T), &width_t, sizeof(T));
  std::memcpy(bytes.data() + g1_data_off + 2 * sizeof(T), &g1_delta_offset, sizeof(T));
  std::memcpy(bytes.data() + g1_data_off + 3 * sizeof(T), packed_words.data(), pkd_bytes);

  uint32_t const entry0 =
    (static_cast<uint32_t>(BitpackingMode::CONSTANT) << 24) | static_cast<uint32_t>(g0_data_off);
  uint32_t const entry1 =
    (static_cast<uint32_t>(BitpackingMode::DELTA_FOR) << 24) | static_cast<uint32_t>(g1_data_off);
  std::memcpy(bytes.data() + metadata_end - 4, &entry0, sizeof(uint32_t));
  std::memcpy(bytes.data() + metadata_end - 8, &entry1, sizeof(uint32_t));

  auto out = decode_one<int32_t>(bytes, I32, total_rows);
  for (uint32_t i = 0; i < group_rows; ++i)
    REQUIRE(out[i] == g0_value);

  // Reference DELTA_FOR: raw[i] = frame + packed[i]; out[i] = delta_offset + sum_{k≤i} raw[k].
  T running = g1_delta_offset;
  for (uint32_t i = 0; i < g1_rows; ++i) {
    running = static_cast<T>(running + g1_frame + g1_packed[i]);
    REQUIRE(out[group_rows + i] == running);
  }
}

TEST_CASE("gpu_decode_table BITPACKING - row_count exactly BP_META_GROUP_SIZE (no partial tail)",
          "[scan][decode][bitpacking]")
{
  // Pins the off-by-one in the kernel's `if (idx >= rc) break` — exactly 2048
  // rows means every iter of the unpack loop is in-bounds, no tail guard fires.
  // FOR is chosen because it exercises the cp.async + unpack loop, where an
  // off-by-one in `rc` would silently mismask the last warp's stores.
  constexpr uint32_t rc = BP_META_GROUP_SIZE;
  std::vector<int32_t> deltas(rc);
  for (uint32_t i = 0; i < rc; ++i)
    deltas[i] = static_cast<int32_t>(i & 0xFF);
  auto bytes = make_for_block<int32_t>(100, 8, deltas);
  auto out   = decode_one<int32_t>(bytes, I32, rc);
  for (uint32_t i = 0; i < rc; ++i)
    REQUIRE(out[i] == 100 + static_cast<int32_t>(i & 0xFF));
}

TEST_CASE("gpu_decode_table BITPACKING - 3-group segment with mixed modes",
          "[scan][decode][bitpacking]")
{
  // Three groups, three different modes. Pins:
  //   (1) per-CTA mode routing across all three FOR-family paths in one launch,
  //   (2) metadata trailer indexing at K=2 (kernel reads metadata_end - (K+1)*4
  //       = metadata_end - 12) — existing multi-group test only goes to K=1.
  // Layout: group 0 CONSTANT, group 1 FOR, group 2 DELTA_FOR (partial tail).
  using T                       = int32_t;
  constexpr uint32_t group_rows = BP_META_GROUP_SIZE;
  constexpr uint32_t g2_rows    = 500;
  constexpr uint32_t total_rows = 2 * group_rows + g2_rows;

  constexpr T g0_value        = 7;
  constexpr T g1_frame        = 100;
  constexpr uint32_t g1_width = 8;
  constexpr T g2_frame        = 2;
  constexpr T g2_delta_offset = 1000;
  constexpr uint32_t g2_width = 8;

  std::vector<T> g1_packed(group_rows);
  for (uint32_t i = 0; i < group_rows; ++i)
    g1_packed[i] = static_cast<T>((i * 3 + 1) & 0xFF);
  std::vector<T> g2_packed(g2_rows);
  for (uint32_t i = 0; i < g2_rows; ++i)
    g2_packed[i] = static_cast<T>((i * 5 + 7) & 0xFF);

  auto g1_packed_words      = pack_values<T>(g1_packed, g1_width);
  auto g2_packed_words      = pack_values<T>(g2_packed, g2_width);
  size_t const g1_pkd_bytes = g1_packed_words.size() * sizeof(uint32_t);
  size_t const g2_pkd_bytes = g2_packed_words.size() * sizeof(uint32_t);

  size_t const g0_data_off = 8;                                           // CONSTANT: 1 T
  size_t const g1_data_off = g0_data_off + sizeof(T);                     // FOR: 2 T + packed
  size_t const g2_data_off = g1_data_off + 2 * sizeof(T) + g1_pkd_bytes;  // DELTA_FOR: 3 T + packed
  size_t const header_end  = g2_data_off + 3 * sizeof(T) + g2_pkd_bytes;
  uint64_t const metadata_end = header_end + 3 * sizeof(uint32_t);

  std::vector<uint8_t> bytes(metadata_end, 0);
  std::memcpy(bytes.data(), &metadata_end, sizeof(uint64_t));
  std::memcpy(bytes.data() + g0_data_off, &g0_value, sizeof(T));

  T const g1_width_t = static_cast<T>(g1_width);
  std::memcpy(bytes.data() + g1_data_off, &g1_frame, sizeof(T));
  std::memcpy(bytes.data() + g1_data_off + sizeof(T), &g1_width_t, sizeof(T));
  std::memcpy(bytes.data() + g1_data_off + 2 * sizeof(T), g1_packed_words.data(), g1_pkd_bytes);

  T const g2_width_t = static_cast<T>(g2_width);
  std::memcpy(bytes.data() + g2_data_off, &g2_frame, sizeof(T));
  std::memcpy(bytes.data() + g2_data_off + sizeof(T), &g2_width_t, sizeof(T));
  std::memcpy(bytes.data() + g2_data_off + 2 * sizeof(T), &g2_delta_offset, sizeof(T));
  std::memcpy(bytes.data() + g2_data_off + 3 * sizeof(T), g2_packed_words.data(), g2_pkd_bytes);

  uint32_t const entry0 =
    (static_cast<uint32_t>(BitpackingMode::CONSTANT) << 24) | static_cast<uint32_t>(g0_data_off);
  uint32_t const entry1 =
    (static_cast<uint32_t>(BitpackingMode::FOR) << 24) | static_cast<uint32_t>(g1_data_off);
  uint32_t const entry2 =
    (static_cast<uint32_t>(BitpackingMode::DELTA_FOR) << 24) | static_cast<uint32_t>(g2_data_off);
  std::memcpy(bytes.data() + metadata_end - 4, &entry0, sizeof(uint32_t));   // K=0
  std::memcpy(bytes.data() + metadata_end - 8, &entry1, sizeof(uint32_t));   // K=1
  std::memcpy(bytes.data() + metadata_end - 12, &entry2, sizeof(uint32_t));  // K=2

  auto out = decode_one<int32_t>(bytes, I32, total_rows);
  for (uint32_t i = 0; i < group_rows; ++i)
    REQUIRE(out[i] == g0_value);
  for (uint32_t i = 0; i < group_rows; ++i)
    REQUIRE(out[group_rows + i] == g1_frame + g1_packed[i]);
  T running = g2_delta_offset;
  for (uint32_t i = 0; i < g2_rows; ++i) {
    running = static_cast<T>(running + g2_frame + g2_packed[i]);
    REQUIRE(out[2 * group_rows + i] == running);
  }
}

TEST_CASE("gpu_decode_table BITPACKING - DELTA_FOR uint16 small-type prefix-sum",
          "[scan][decode][bitpacking]")
{
  // Lock down the prefix-sum path on a narrower type than int32/int64. uint16
  // wraps at 65536, so values stay below that; mild deltas prove the unpack +
  // WarpScan + delta_offset bias chain handles 16-bit accumulators without
  // truncation in the cub::WarpScan<T> instantiation.
  using T                  = uint16_t;
  constexpr uint32_t rc    = 1024;
  constexpr T frame        = 1;
  constexpr T delta_offset = 100;
  constexpr uint32_t width = 8;

  std::vector<T> packed_vals(rc);
  for (uint32_t i = 0; i < rc; ++i)
    packed_vals[i] = static_cast<T>((i * 7 + 13) & 0x1F);  // 0..31; max sum ≈ 100 + 1024*32 < 65536

  auto bytes = make_delta_for_block<T>(frame, delta_offset, width, packed_vals);
  auto out   = decode_one<T>(bytes, U16, rc);

  T running = delta_offset;
  for (uint32_t i = 0; i < rc; ++i) {
    running = static_cast<T>(running + frame + packed_vals[i]);
    REQUIRE(out[i] == running);
  }
}

//===----------------------------------------------------------------------===//
// Defensive guards. Each test pre-fills the output with a 0xCC canary and
// invokes `decode_bitpacking_data` directly (bypassing the dispatcher's
// allocate-fresh path) so we can prove the kernel deterministically zero-
// fills on malformed metadata, rather than relying on REQUIRE_NOTHROW alone.
//===----------------------------------------------------------------------===//

namespace {
inline std::vector<int32_t> decode_invalid_with_canary(rmm::cuda_stream& stream,
                                                       rmm::mr::cuda_async_memory_resource& mr,
                                                       gpu_codec_run const& run,
                                                       uint32_t total_rows)
{
  std::vector<uint8_t> canary(size_t{total_rows} * sizeof(int32_t), 0xCC);
  rmm::device_buffer d_out(canary.data(), canary.size(), stream.view());
  decode_bitpacking_data(
    run, static_cast<uint8_t*>(d_out.data()), I32, sizeof(int32_t), stream.view(), mr);
  return download<int32_t>(d_out.data(), total_rows, stream.value());
}
}  // namespace

TEST_CASE("gpu_decode_table BITPACKING - INVALID mode zero-fills",
          "[scan][decode][bitpacking][defensive]")
{
  std::vector<uint8_t> bytes(64, 0);
  uint64_t metadata_end = 32;
  std::memcpy(bytes.data(), &metadata_end, sizeof(metadata_end));
  uint32_t encoded = (static_cast<uint32_t>(BitpackingMode::INVALID) << 24) | 8u;
  std::memcpy(bytes.data() + metadata_end - 4, &encoded, sizeof(encoded));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 32;
  gpu_codec_run run{CompressionType::COMPRESSION_BITPACKING,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table BITPACKING - corrupt metadata_end zero-fills",
          "[scan][decode][bitpacking][defensive]")
{
  std::vector<uint8_t> bytes(64, 0);
  // metadata_end past the segment buffer — kernel demotes to INVALID and
  // zero-fills rather than read past d_seg.
  uint64_t bogus_end = 1ULL << 30;
  std::memcpy(bytes.data(), &bogus_end, sizeof(bogus_end));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 16;
  gpu_codec_run run{CompressionType::COMPRESSION_BITPACKING,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table BITPACKING - width > sizeof(T)*8 zero-fills",
          "[scan][decode][bitpacking][defensive]")
{
  // FOR block with width patched to 100 (wider than int32) — kernel must
  // demote to INVALID and zero-fill rather than read past the packed stream.
  auto bytes          = make_for_block<int32_t>(0, 5, std::vector<int32_t>(8, 1));
  int32_t bogus_width = 100;
  std::memcpy(bytes.data() + 8 + sizeof(int32_t), &bogus_width, sizeof(int32_t));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 8;
  gpu_codec_run run{CompressionType::COMPRESSION_BITPACKING,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table BITPACKING - packed stream past metadata_end zero-fills",
          "[scan][decode][bitpacking][defensive]")
{
  // FOR block with declared row_count that would push packed_words past
  // metadata_end. Kernel catches the packed-stream bound check, demotes to
  // INVALID, and zero-fills.
  auto bytes = make_for_block<int32_t>(/*frame=*/0, /*width=*/4, std::vector<int32_t>(8, 1));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 2048;
  gpu_codec_run run{CompressionType::COMPRESSION_BITPACKING,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_invalid_with_canary(stream, mr, run, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0);
}

TEST_CASE("gpu_decode_table BITPACKING - unsupported type_size throws",
          "[scan][decode][bitpacking][defensive]")
{
  // BOOL is 1 byte (handled), but DECIMAL128 with INT128 backing is 16 —
  // bitpacking is not defined for 128-bit values upstream, and the kernel
  // should refuse rather than pretend.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  auto bytes = make_constant_block<int32_t>(0);
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  auto const DEC128 = cudf::data_type{cudf::type_id::DECIMAL128, /*scale=*/-2};
  gpu_column_decode_input col;
  col.out_type   = DEC128;
  col.total_rows = 8;
  col.data.push_back(
    {CompressionType::COMPRESSION_BITPACKING,
     {gpu_segment_desc{
       static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, 8}}});
  REQUIRE_THROWS_WITH(gpu_decode_table({col}, stream.view(), mr),
                      Catch::Contains("viability invariant violated"));
}
