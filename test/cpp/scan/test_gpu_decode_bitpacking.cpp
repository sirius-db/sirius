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

namespace {

auto const I32 = cudf::data_type{cudf::type_id::INT32};
auto const I64 = cudf::data_type{cudf::type_id::INT64};
auto const U16 = cudf::data_type{cudf::type_id::UINT16};

//===----------------------------------------------------------------------===//
// Synthetic-segment builders.
//
// Layout (matches the kernel's parse in gpu_decode_bitpacking.cu):
//   [0..8)               metadata_end (uint64)
//   [8..)                per-mode header
//   [data_off..)         packed stream (FOR / DELTA_FOR only)
//   [metadata_end-4*N..  per-group metadata trailer
//    metadata_end)        (one uint32 per group, last entry first)
//
// Each helper builds a single-group segment at data_off=8 with the metadata
// entry at metadata_end-4. Multi-group segments are constructed manually in
// the multi-group test case.
//===----------------------------------------------------------------------===//

template <typename T>
std::vector<uint32_t> pack_values(std::vector<T> const& values, uint32_t width)
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
std::vector<uint8_t> make_constant_block(T value)
{
  std::vector<uint8_t> block(64, 0);
  uint64_t metadata_end = 32;
  std::memcpy(block.data(), &metadata_end, sizeof(metadata_end));
  std::memcpy(block.data() + 8, &value, sizeof(T));
  uint32_t encoded = (static_cast<uint32_t>(BitpackingMode::CONSTANT) << 24) | 8u;
  std::memcpy(block.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
std::vector<uint8_t> make_constant_delta_block(T frame, T delta)
{
  std::vector<uint8_t> block(64, 0);
  uint64_t metadata_end = 32;
  std::memcpy(block.data(), &metadata_end, sizeof(metadata_end));
  std::memcpy(block.data() + 8, &frame, sizeof(T));
  std::memcpy(block.data() + 8 + sizeof(T), &delta, sizeof(T));
  uint32_t encoded = (static_cast<uint32_t>(BitpackingMode::CONSTANT_DELTA) << 24) | 8u;
  std::memcpy(block.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
std::vector<uint8_t> make_for_block(T frame, uint32_t width, std::vector<T> const& values)
{
  auto packed         = pack_values<T>(values, width);
  size_t packed_bytes = packed.size() * sizeof(uint32_t);
  // [metadata_end][frame:T][width:T][packed_bytes][...trailer:uint32]
  size_t header_end     = 8 + 2 * sizeof(T) + packed_bytes;
  size_t metadata_end_v = header_end + 4;
  size_t block_size     = std::max<size_t>(metadata_end_v, 64);
  std::vector<uint8_t> block(block_size, 0);
  std::memcpy(block.data(), &metadata_end_v, sizeof(uint64_t));
  T width_t = static_cast<T>(width);
  std::memcpy(block.data() + 8, &frame, sizeof(T));
  std::memcpy(block.data() + 8 + sizeof(T), &width_t, sizeof(T));
  std::memcpy(block.data() + 8 + 2 * sizeof(T), packed.data(), packed_bytes);
  uint32_t encoded = (static_cast<uint32_t>(BitpackingMode::FOR) << 24) | 8u;
  std::memcpy(block.data() + metadata_end_v - 4, &encoded, sizeof(encoded));
  return block;
}

template <typename T>
std::vector<uint8_t> make_delta_for_block(T frame,
                                          T delta_offset,
                                          uint32_t width,
                                          std::vector<T> const& packed_values)
{
  auto packed         = pack_values<T>(packed_values, width);
  size_t packed_bytes = packed.size() * sizeof(uint32_t);
  // [metadata_end][frame:T][width:T][delta_offset:T][packed_bytes][trailer:uint32]
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
  uint32_t encoded = (static_cast<uint32_t>(BitpackingMode::DELTA_FOR) << 24) | 8u;
  std::memcpy(block.data() + metadata_end_v - 4, &encoded, sizeof(encoded));
  return block;
}

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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
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
  for (uint32_t i = 0; i < total_rows; ++i) REQUIRE(out[i] == 0);
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
