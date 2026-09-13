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

// String-codec round trips and malformed-segment checks.

#include "scan/decode_test_utils.hpp"
#include "scan/strings_synth.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_strings.cuh>
#include <cuda/scan/strings/dict_fsst.cuh>
#include <cuda/scan/strings/dictionary.cuh>
#include <cuda/scan/strings/fsst.cuh>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using duckdb::CompressionType;
using sirius::cuda::scan::gpu_codec_run;
using sirius::cuda::scan::gpu_decode_strings_column;
using sirius::cuda::scan::gpu_segment_desc;
using sirius::cuda::scan::gpu_string_codec_run;
using sirius::cuda::scan::gpu_string_column_decode_input;
using sirius::cuda::scan::gpu_string_segment_desc;
using sirius::test::decode::download;
using sirius::test::decode::strings::make_dict_fsst_segment;
using sirius::test::decode::strings::make_dict_segment;
using sirius::test::decode::strings::make_fsst_segment;
using sirius::test::decode::strings::make_uncompressed_segment;

namespace {

// Prevent decoder kernels from reading beyond the small backing allocation.
struct reject_decode_allocation_resource {
  size_t allocations = 0;
  rmm::mr::callback_memory_resource resource{
    [](size_t, rmm::cuda_stream_view, void* arg) -> void* {
      ++*static_cast<size_t*>(arg);
      throw std::logic_error("invalid metadata reached decode scratch allocation");
    },
    [](void*, size_t, rmm::cuda_stream_view, void*) {},
    &allocations};
};

/// Stage segment bytes on device; build a single-segment string column input
/// and run it through `gpu_decode_strings_column`. Returns the decoded
/// strings as a vector<string> for direct comparison.
std::vector<std::string> decode_one_dict(std::vector<uint8_t> const& bytes,
                                         uint32_t row_count,
                                         uint32_t max_string_length)
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()),
                              static_cast<uint32_t>(d_seg.size()),
                              0,
                              row_count,
                              0,
                              max_string_length};
  gpu_string_column_decode_input col;
  col.total_rows = row_count;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_DICTIONARY, {seg}});

  auto column = gpu_decode_strings_column(col, stream.view(), mr);
  cudf::strings_column_view scv(column->view());

  std::vector<int32_t> offsets =
    download<int32_t>(scv.offsets().data<int32_t>(), row_count + 1, stream.value());
  std::vector<uint8_t> chars =
    download<uint8_t>(scv.chars_begin(stream), scv.chars_size(stream), stream.value());

  std::vector<std::string> out(row_count);
  for (uint32_t i = 0; i < row_count; ++i) {
    int32_t s = offsets[i];
    int32_t e = offsets[i + 1];
    out[i].assign(reinterpret_cast<char const*>(chars.data() + s), static_cast<size_t>(e - s));
  }
  return out;
}

/// Decode one synth segment to host strings. FSST / DICT_FSST round-trips.
std::vector<std::string> decode_one_segment(std::vector<uint8_t> const& bytes,
                                            duckdb::CompressionType codec,
                                            uint32_t row_count,
                                            uint32_t max_string_length,
                                            uint32_t lead_pad = 0)
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  // Optionally stage the segment behind `lead_pad` padding bytes so its base lands at a
  // non-naturally-aligned device address, exercising the codec's alignment-agnostic reads.
  std::vector<uint8_t> staged;
  uint8_t const* host_src = bytes.data();
  size_t host_size        = bytes.size();
  if (lead_pad) {
    staged.assign(lead_pad, uint8_t{0});
    staged.insert(staged.end(), bytes.begin(), bytes.end());
    host_src  = staged.data();
    host_size = staged.size();
  }
  rmm::device_buffer d_seg(host_src, host_size, stream.view());

  gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()) + lead_pad,
                              static_cast<uint32_t>(bytes.size()),
                              0,
                              row_count,
                              0,
                              max_string_length};
  gpu_string_column_decode_input col;
  col.total_rows = row_count;
  col.has_nulls  = false;
  col.data.push_back({codec, {seg}});

  auto column = gpu_decode_strings_column(col, stream.view(), mr);
  cudf::strings_column_view scv(column->view());

  std::vector<int32_t> offsets =
    download<int32_t>(scv.offsets().data<int32_t>(), row_count + 1, stream.value());
  std::vector<std::string> out(row_count);
  if (scv.chars_size(stream) > 0) {
    std::vector<uint8_t> chars =
      download<uint8_t>(scv.chars_begin(stream), scv.chars_size(stream), stream.value());
    for (uint32_t i = 0; i < row_count; ++i) {
      int32_t s = offsets[i];
      int32_t e = offsets[i + 1];
      out[i].assign(reinterpret_cast<char const*>(chars.data() + s), static_cast<size_t>(e - s));
    }
  }
  return out;
}

std::vector<std::string> decode_invalid_with_canary(std::vector<uint8_t> const& bytes,
                                                    duckdb::CompressionType codec,
                                                    uint32_t row_count)
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()),
                              static_cast<uint32_t>(d_seg.size()),
                              0,
                              row_count,
                              0,
                              16u};  // upper-bound stub; chars buffer sized from this
  gpu_string_column_decode_input col;
  col.total_rows = row_count;
  col.has_nulls  = false;
  col.data.push_back({codec, {seg}});

  auto column = gpu_decode_strings_column(col, stream.view(), mr);
  cudf::strings_column_view scv(column->view());
  std::vector<int32_t> offsets =
    download<int32_t>(scv.offsets().data<int32_t>(), row_count + 1, stream.value());

  std::vector<std::string> out(row_count);
  if (scv.chars_size(stream) > 0) {
    std::vector<uint8_t> chars =
      download<uint8_t>(scv.chars_begin(stream), scv.chars_size(stream), stream.value());
    for (uint32_t i = 0; i < row_count; ++i) {
      int32_t s = offsets[i];
      int32_t e = offsets[i + 1];
      out[i].assign(reinterpret_cast<char const*>(chars.data() + s), static_cast<size_t>(e - s));
    }
  }
  return out;
}

}  // namespace

TEST_CASE("gpu_decode_strings DICT_FSST padded lengths round-trip in dictionary modes",
          "[scan][decode][strings][dict_fsst]")
{
  auto mode = GENERATE(uint8_t{0}, uint8_t{1});
  std::vector<std::string> dict{"",
                                "",
                                std::string(1000, 'a') + "red",
                                std::string(1000, 'b') + "green",
                                std::string(1000, 'c') + "blue"};
  std::vector<uint32_t> selections;
  for (uint32_t i = 0; i < 97; ++i) {
    selections.push_back(i % dict.size());
  }
  auto bytes           = make_dict_fsst_segment(dict, selections, mode);
  uint32_t const width = bytes[9];
  auto align8          = [](uint32_t n) { return (n + 7u) & ~7u; };
  REQUIRE(align8((dict.size() * width + 7u) / 8u) != align8(32u * width / 8u));
  auto output =
    decode_one_segment(bytes, CompressionType::COMPRESSION_DICT_FSST, selections.size(), 1005);
  REQUIRE(output.size() == selections.size());
  for (size_t i = 0; i < output.size(); ++i) {
    REQUIRE(output[i] == dict[selections[i]]);
  }
}

TEST_CASE("gpu_decode_strings DICT_FSST rejects invalid host metadata before preparation",
          "[scan][decode][strings][dict_fsst][defensive]")
{
  auto malformed = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
  CAPTURE(malformed);
  // Large logical regions exercise arithmetic checks without allocating their payloads.
  std::vector<uint8_t> backing(8192, 0);
  uint32_t dict_size    = 0;
  uint32_t dict_count   = 2;
  uint32_t symtab_size  = 0;
  backing[8]            = 0;
  backing[9]            = 1;
  backing[10]           = 1;
  uint32_t logical_size = 64;
  uint32_t row_count    = 1;
  uint32_t row_start    = 0;
  switch (malformed) {
    case 0: backing[9] = 33; break;
    case 1:
      backing[8]  = 1;
      symtab_size = 4096;
      break;
    case 2:
      backing[8]  = 1;
      dict_size   = 128;
      symtab_size = 17;
      break;
    case 3:
      backing[8]  = 2;
      dict_size   = 128;
      symtab_size = 17;
      break;
    case 4: dict_count = 0; break;
    case 5: row_start = 512; break;
    case 6:
      backing[8]  = 2;
      row_start   = 2;
      symtab_size = 17;
      break;
    case 7: backing[10] = 33; break;
    case 8:
      row_start = UINT32_MAX;
      row_count = 2;
      break;
    case 9:
      dict_count  = UINT32_MAX;
      backing[9]  = 0;
      backing[10] = 0;
      break;
    case 10:
      dict_count  = (1u << 30) + 2u;
      backing[9]  = 0;
      backing[10] = 0;
      break;
    case 11:
      dict_count   = (1u << 26) + 1u;
      backing[9]   = 32;
      backing[10]  = 0;
      logical_size = UINT32_MAX;
      break;
    case 12:
      backing[9]   = 0;
      backing[10]  = 32;
      row_start    = 1u << 26;
      logical_size = UINT32_MAX;
      break;
    case 13:
      backing[9]  = 0;
      backing[10] = 0;
      row_start   = std::numeric_limits<int32_t>::max();
      break;
    case 14: backing[8] = 1; break;
    case 15:
      backing[8]   = 1;
      backing[9]   = 0;
      backing[10]  = 0;
      logical_size = 16;
      break;
    case 16:
      backing[8]  = 1;
      symtab_size = 16;
      break;
    case 17:
      backing[8]  = 2;
      symtab_size = 16;
      break;
  }
  std::memcpy(backing.data(), &dict_size, 4);
  std::memcpy(backing.data() + 4, &dict_count, 4);
  std::memcpy(backing.data() + 12, &symtab_size, 4);
  rmm::cuda_stream stream;
  reject_decode_allocation_resource mr;
  rmm::device_buffer device(backing.data(), backing.size(), stream.view());
  gpu_string_segment_desc segment{
    static_cast<uint8_t const*>(device.data()), logical_size, 0, row_count, row_start, 16};
  gpu_string_codec_run run{CompressionType::COMPRESSION_DICT_FSST, {segment}};
  CHECK_THROWS_AS(sirius::cuda::scan::prepare_dict_fsst(run, stream.view(), mr.resource),
                  std::runtime_error);
  if (malformed >= 14) {
    CHECK_THROWS_WITH(sirius::cuda::scan::prepare_dict_fsst(run, stream.view(), mr.resource),
                      Catch::Contains("is shorter than the FSST symbol table header"));
  }
  CHECK(mr.allocations == 0);
}

TEST_CASE("gpu_decode_strings FSST rejects a symbol table shorter than its fixed header",
          "[scan][decode][strings][fsst][defensive]")
{
  auto const table_bytes = GENERATE(uint32_t{1}, uint32_t{16});
  CAPTURE(table_bytes);
  uint32_t const header[] = {0, 16 + table_bytes, 0, 16};
  std::vector<uint8_t> bytes(16 + table_bytes, 0);
  std::memcpy(bytes.data(), header, sizeof(header));
  rmm::cuda_stream stream;
  rmm::device_buffer device(bytes.data(), bytes.size(), stream.view());
  gpu_string_codec_run run{CompressionType::COMPRESSION_FSST,
                           {{static_cast<uint8_t const*>(device.data()),
                             static_cast<uint32_t>(bytes.size()),
                             0,
                             1,
                             0,
                             0}}};
  CHECK_THROWS_AS(sirius::cuda::scan::prepare_fsst(run, stream.view()), std::runtime_error);
  CHECK_THROWS_WITH(sirius::cuda::scan::prepare_fsst(run, stream.view()),
                    Catch::Contains("symbol table header"));
}

TEST_CASE("gpu_decode_strings FSST admits exactly the fixed symbol table header",
          "[scan][decode][strings][fsst][defensive]")
{
  uint32_t const header[] = {0, 33, 0, 16};
  std::vector<uint8_t> bytes(33, 0);
  std::memcpy(bytes.data(), header, sizeof(header));
  rmm::cuda_stream stream;
  rmm::device_buffer device(bytes.data(), bytes.size(), stream.view());
  gpu_string_codec_run run{CompressionType::COMPRESSION_FSST,
                           {{static_cast<uint8_t const*>(device.data()), 33, 0, 1, 0, 0}}};
  REQUIRE_NOTHROW(sirius::cuda::scan::prepare_fsst(run, stream.view()));
  auto prepared = sirius::cuda::scan::prepare_fsst(run, stream.view());
  REQUIRE(prepared.total_fsst_row_count == 1);
  REQUIRE(prepared.length_descs.size() == 1);
}

TEST_CASE("gpu_decode_strings DICT_FSST rejects cumulative dictionary offset overflow",
          "[scan][decode][strings][dict_fsst][defensive]")
{
  uint32_t const dict_count    = 1u << 30;
  uint32_t const segment_count = 4;
  REQUIRE(uint64_t{segment_count} * (uint64_t{dict_count} + 1) > UINT32_MAX);
  std::vector<uint8_t> backing(16, 0);
  std::memcpy(backing.data() + 4, &dict_count, sizeof(dict_count));
  rmm::cuda_stream stream;
  rmm::device_buffer device(backing.data(), backing.size(), stream.view());
  reject_decode_allocation_resource mr;
  gpu_string_codec_run run{CompressionType::COMPRESSION_DICT_FSST, {}};
  for (uint32_t row = 0; row < segment_count; ++row) {
    run.segments.push_back({static_cast<uint8_t const*>(device.data()), 16, row, 1, 0, 0});
  }
  CHECK_THROWS_AS(sirius::cuda::scan::prepare_dict_fsst(run, stream.view(), mr.resource),
                  std::runtime_error);
  CHECK(mr.allocations == 0);
}

TEST_CASE("gpu_decode_strings FSST rejects a slice after the segment head",
          "[scan][decode][strings][fsst][defensive]")
{
  auto bytes = make_fsst_segment({"first", "second string"});
  rmm::cuda_stream stream;
  rmm::device_buffer device(bytes.data(), bytes.size(), stream.view());
  gpu_string_segment_desc segment{
    static_cast<uint8_t const*>(device.data()), static_cast<uint32_t>(bytes.size()), 0, 2, 0, 13};
  gpu_string_codec_run run{CompressionType::COMPRESSION_FSST, {segment}};
  REQUIRE_NOTHROW(sirius::cuda::scan::prepare_fsst(run, stream.view()));
  run.segments[0].seg_row_start = 1;
  run.segments[0].row_count     = 1;
  REQUIRE_THROWS_AS(sirius::cuda::scan::prepare_fsst(run, stream.view()), std::runtime_error);
}

TEST_CASE("gpu_decode_strings DICTIONARY rejects a bit index beyond INT32_MAX",
          "[scan][decode][strings][dictionary][defensive]")
{
  uint32_t const rows         = (1u << 26) + 1u;
  uint32_t const index_offset = 20u + rows * 4u;
  uint32_t const header[]     = {0, index_offset + 4u, index_offset, 1, 32};
  rmm::cuda_stream stream;
  rmm::device_buffer device(header, sizeof(header), stream.view());
  gpu_string_codec_run run{
    CompressionType::COMPRESSION_DICTIONARY,
    {{static_cast<uint8_t const*>(device.data()), UINT32_MAX, 0, rows, 0, 0}}};
  REQUIRE_THROWS_AS(sirius::cuda::scan::prepare_dict(run, stream.view()), std::runtime_error);
}

TEST_CASE("gpu_decode_strings FSST rejects a bit index beyond INT32_MAX",
          "[scan][decode][strings][fsst][defensive]")
{
  uint32_t const rows          = (1u << 26) + 1u;
  uint32_t const symtab_offset = 16u + rows * 4u;
  uint32_t const header[]      = {0, symtab_offset + 17u, 32, symtab_offset};
  rmm::cuda_stream stream;
  rmm::device_buffer device(header, sizeof(header), stream.view());
  gpu_string_codec_run run{
    CompressionType::COMPRESSION_FSST,
    {{static_cast<uint8_t const*>(device.data()), UINT32_MAX, 0, rows, 0, 0}}};
  REQUIRE_THROWS_AS(sirius::cuda::scan::prepare_fsst(run, stream.view()), std::runtime_error);
}

// --- UNCOMPRESSED happy path ---

TEST_CASE("gpu_decode_strings UNCOMPRESSED - all empty strings",
          "[scan][decode][strings][uncompressed]")
{
  std::vector<std::string> rows(8, "");
  auto bytes = make_uncompressed_segment(rows);
  auto out   = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_UNCOMPRESSED,
                                static_cast<uint32_t>(rows.size()),
                                /*max_len=*/4u);
  REQUIRE(out.size() == rows.size());
  for (auto const& s : out)
    REQUIRE(s.empty());
}

TEST_CASE("gpu_decode_strings UNCOMPRESSED - varied lengths including empty",
          "[scan][decode][strings][uncompressed]")
{
  std::vector<std::string> rows = {"", "x", "", "abcd", "longer-string-here", "", "z"};
  auto bytes                    = make_uncompressed_segment(rows);
  auto out                      = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_UNCOMPRESSED,
                                static_cast<uint32_t>(rows.size()),
                                /*max_len=*/32u);
  REQUIRE(out.size() == rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    REQUIRE(out[i] == rows[i]);
  }
}

TEST_CASE("gpu_decode_strings UNCOMPRESSED - many rows across multiple CTAs",
          "[scan][decode][strings][uncompressed]")
{
  // 4096 rows exercises multi-CTA dispatch + cumulative offset growth.
  std::vector<std::string> rows;
  rows.reserve(4096);
  for (uint32_t i = 0; i < 4096; ++i) {
    rows.push_back("row_" + std::to_string(i));
  }
  auto bytes = make_uncompressed_segment(rows);
  auto out   = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_UNCOMPRESSED,
                                static_cast<uint32_t>(rows.size()),
                                /*max_len=*/16u);
  REQUIRE(out.size() == rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    REQUIRE(out[i] == rows[i]);
  }
}

TEST_CASE("gpu_decode_strings UNCOMPRESSED - tolerates unaligned segment base",
          "[scan][decode][strings][uncompressed]")
{
  // RMM device buffers are 256B-aligned; staging the segment behind 1-3 pad bytes forces a
  // non-int32-aligned base, exercising the alignment-agnostic offset reads (load_unaligned).
  std::vector<std::string> rows = {"", "x", "abcd", "longer-string-here", "", "zz", "q"};
  auto bytes                    = make_uncompressed_segment(rows);
  for (uint32_t pad : {1u, 2u, 3u}) {
    auto out = decode_one_segment(bytes,
                                  CompressionType::COMPRESSION_UNCOMPRESSED,
                                  static_cast<uint32_t>(rows.size()),
                                  /*max_len=*/32u,
                                  /*lead_pad=*/pad);
    REQUIRE(out.size() == rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      INFO("pad=" << pad << " row=" << i);
      REQUIRE(out[i] == rows[i]);
    }
  }
}

// --- UNCOMPRESSED defensive path ---

TEST_CASE("gpu_decode_strings UNCOMPRESSED - segment_size below header zero-fills",
          "[scan][decode][strings][uncompressed][defensive]")
{
  // Build a valid segment then truncate below the [dict_size:4][dict_end:4] + offsets
  // region. The kernel's bounds check (`limit >= 8u + end_row * 4u`) should
  // detect this and zero-fill all lengths.
  std::vector<std::string> rows = {"abc", "defg", "hi"};
  auto bytes                    = make_uncompressed_segment(rows);
  bytes.resize(8);  // header only; offsets array missing
  auto out = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_UNCOMPRESSED,
                                static_cast<uint32_t>(rows.size()),
                                /*max_len=*/8u);
  REQUIRE(out.size() == rows.size());
  for (auto const& s : out)
    REQUIRE(s.empty());
}

// --- DICTIONARY happy path ---

TEST_CASE("gpu_decode_strings DICTIONARY - basic round-trip", "[scan][decode][strings][dictionary]")
{
  std::vector<std::string> dict = {"", "alpha", "beta", "gamma"};
  std::vector<uint32_t> sel     = {1, 2, 3, 1, 2, 3, 1};
  auto bytes                    = make_dict_segment(dict, sel);
  auto out = decode_one_dict(bytes, static_cast<uint32_t>(sel.size()), /*max_len=*/8u);
  REQUIRE(out.size() == sel.size());
  REQUIRE(out[0] == "alpha");
  REQUIRE(out[1] == "beta");
  REQUIRE(out[2] == "gamma");
  REQUIRE(out[3] == "alpha");
  REQUIRE(out[4] == "beta");
  REQUIRE(out[5] == "gamma");
  REQUIRE(out[6] == "alpha");
}

TEST_CASE("gpu_decode_strings DICTIONARY - NULL via index 0", "[scan][decode][strings][dictionary]")
{
  std::vector<std::string> dict = {"", "x"};  // entry 0 = NULL
  std::vector<uint32_t> sel     = {0, 1, 0, 1, 1};
  auto bytes                    = make_dict_segment(dict, sel);
  auto out = decode_one_dict(bytes, static_cast<uint32_t>(sel.size()), /*max_len=*/4u);
  REQUIRE(out[0].empty());
  REQUIRE(out[1] == "x");
  REQUIRE(out[2].empty());
  REQUIRE(out[3] == "x");
  REQUIRE(out[4] == "x");
}

TEST_CASE("gpu_decode_strings DICTIONARY - empty dict, all NULL",
          "[scan][decode][strings][dictionary]")
{
  std::vector<std::string> dict = {""};
  std::vector<uint32_t> sel     = {0, 0, 0};
  auto bytes                    = make_dict_segment(dict, sel);
  auto out = decode_one_dict(bytes, static_cast<uint32_t>(sel.size()), /*max_len=*/4u);
  for (auto& s : out)
    REQUIRE(s.empty());
}

TEST_CASE("gpu_decode_strings DICTIONARY - corrupt index_buffer_offset throws",
          "[scan][decode][strings][dictionary][defensive]")
{
  std::vector<uint8_t> bytes(64, 0);
  uint32_t hdr[5] = {/*dict_size=*/4u,
                     /*dict_end=*/64u,
                     /*idx_buf_off=*/1u << 30,
                     /*idx_buf_count=*/2u,
                     /*width=*/1u};
  std::memcpy(bytes.data(), hdr, sizeof(hdr));
  REQUIRE_THROWS_AS(decode_invalid_with_canary(bytes, CompressionType::COMPRESSION_DICTIONARY, 8),
                    std::runtime_error);
}

TEST_CASE("gpu_decode_strings DICTIONARY - bitpacking_width > 32 throws",
          "[scan][decode][strings][dictionary][defensive]")
{
  std::vector<uint8_t> bytes(128, 0);
  uint32_t hdr[5] = {0u, 64u, 28u, 1u, /*width=*/100u};
  std::memcpy(bytes.data(), hdr, sizeof(hdr));
  REQUIRE_THROWS_AS(decode_invalid_with_canary(bytes, CompressionType::COMPRESSION_DICTIONARY, 4),
                    std::runtime_error);
}

TEST_CASE("gpu_decode_strings FSST - corrupt dict_end > segment_size throws",
          "[scan][decode][strings][fsst][defensive]")
{
  std::vector<uint8_t> bytes(64, 0);
  uint32_t hdr[4] = {/*dict_size=*/100u,
                     /*dict_end=*/1u << 30,
                     /*bitpacking_width=*/8u,
                     /*fsst_symbol_table_offset=*/16u};
  std::memcpy(bytes.data(), hdr, sizeof(hdr));
  REQUIRE_THROWS_AS(decode_invalid_with_canary(bytes, CompressionType::COMPRESSION_FSST, 8),
                    std::runtime_error);
}

TEST_CASE("gpu_decode_strings FSST - bitpacking_width > 32 throws",
          "[scan][decode][strings][fsst][defensive]")
{
  std::vector<uint8_t> bytes(64, 0);
  uint32_t hdr[4] = {/*dict_size=*/8u, /*dict_end=*/40u, /*bitpacking_width=*/64u, /*sym_off=*/16u};
  std::memcpy(bytes.data(), hdr, sizeof(hdr));
  REQUIRE_THROWS_AS(decode_invalid_with_canary(bytes, CompressionType::COMPRESSION_FSST, 4),
                    std::runtime_error);
}

TEST_CASE("gpu_decode_strings DICT_FSST - mode > 2 throws",
          "[scan][decode][strings][dict_fsst][defensive]")
{
  std::vector<uint8_t> bytes(128, 0);
  uint32_t dict_size = 0, dict_count = 1;
  uint8_t mode = 9, slens_w = 0, didx_w = 0, _pad = 0;
  uint32_t symtab_size = 0;
  std::memcpy(bytes.data(), &dict_size, 4);
  std::memcpy(bytes.data() + 4, &dict_count, 4);
  bytes[8]  = mode;
  bytes[9]  = slens_w;
  bytes[10] = didx_w;
  bytes[11] = _pad;
  std::memcpy(bytes.data() + 12, &symtab_size, 4);
  REQUIRE_THROWS_AS(decode_invalid_with_canary(bytes, CompressionType::COMPRESSION_DICT_FSST, 4),
                    std::runtime_error);
}

TEST_CASE("gpu_decode_strings DICT_FSST - bytes_size below header throws",
          "[scan][decode][strings][dict_fsst][defensive]")
{
  std::vector<uint8_t> bytes(8, 0xAB);
  REQUIRE_THROWS_AS(decode_invalid_with_canary(bytes, CompressionType::COMPRESSION_DICT_FSST, 3),
                    std::runtime_error);
}

TEST_CASE("gpu_decode_strings - unsupported codec throws", "[scan][decode][strings][defensive]")
{
  std::vector<uint8_t> bytes(16, 0);
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());
  gpu_string_segment_desc seg{
    static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, 4, 0, 8u};
  gpu_string_column_decode_input col;
  col.total_rows = 4;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_CONSTANT, {seg}});
  REQUIRE_THROWS_WITH(gpu_decode_strings_column(col, stream.view(), mr),
                      Catch::Contains("viability invariant violated"));
}

// --- FSST happy path (synthetic segments via libduckdb FSST encoder) ---

TEST_CASE("gpu_decode_strings FSST - basic round-trip", "[scan][decode][strings][fsst]")
{
  std::vector<std::string> rows = {
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy cat",
    "the rain in spain falls mainly on the plain",
    "the quick brown fox jumps over the lazy dog",
    "lorem ipsum dolor sit amet consectetur adipiscing",
  };
  auto bytes = make_fsst_segment(rows);
  auto out   = decode_one_segment(
    bytes, CompressionType::COMPRESSION_FSST, static_cast<uint32_t>(rows.size()), /*max_len=*/64u);
  REQUIRE(out.size() == rows.size());
  for (size_t i = 0; i < rows.size(); ++i)
    REQUIRE(out[i] == rows[i]);
}

TEST_CASE("gpu_decode_strings FSST - many distinct strings", "[scan][decode][strings][fsst]")
{
  uint32_t const ROWS = 10000;  // bench-scale verify
  std::vector<std::string> rows(ROWS);
  for (uint32_t i = 0; i < ROWS; ++i) {
    rows[i] = "row_" + std::to_string(i) + "_data_" + std::to_string(i * 31u);
  }
  auto bytes = make_fsst_segment(rows);
  auto out   = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_FSST,
                                ROWS,
                                /*max_len=*/64u);
  REQUIRE(out.size() == ROWS);
  for (uint32_t i = 0; i < ROWS; ++i) {
    if (out[i] != rows[i]) {
      FAIL("FSST round-trip mismatch at row=" << i << " expected='" << rows[i] << "' got='"
                                              << out[i] << "'");
    }
  }
}

// --- DICT_FSST happy path: modes 0 (raw dict), 1 (FSST dict), 2 (no dict) ---

TEST_CASE("gpu_decode_strings DICT_FSST mode 0 (DICTIONARY) - basic round-trip",
          "[scan][decode][strings][dict_fsst]")
{
  std::vector<std::string> dict = {"", "alpha", "beta", "gamma"};
  std::vector<uint32_t> sel     = {1, 2, 3, 0, 1, 2};
  auto bytes                    = make_dict_fsst_segment(dict, sel, /*mode=*/0);
  auto out                      = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_DICT_FSST,
                                static_cast<uint32_t>(sel.size()),
                                /*max_len=*/8u);
  REQUIRE(out.size() == sel.size());
  REQUIRE(out[0] == "alpha");
  REQUIRE(out[1] == "beta");
  REQUIRE(out[2] == "gamma");
  REQUIRE(out[3].empty());  // sel=0 → NULL/empty
  REQUIRE(out[4] == "alpha");
  REQUIRE(out[5] == "beta");
}

TEST_CASE("gpu_decode_strings DICT_FSST mode 1 (DICT_FSST) - basic round-trip",
          "[scan][decode][strings][dict_fsst]")
{
  std::vector<std::string> dict = {
    "", "the quick brown fox jumps", "the lazy dog sleeps", "lorem ipsum dolor sit amet"};
  std::vector<uint32_t> sel = {1, 2, 3, 1, 0, 2};
  auto bytes                = make_dict_fsst_segment(dict, sel, /*mode=*/1);
  auto out                  = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_DICT_FSST,
                                static_cast<uint32_t>(sel.size()),
                                /*max_len=*/64u);
  REQUIRE(out.size() == sel.size());
  REQUIRE(out[0] == dict[1]);
  REQUIRE(out[1] == dict[2]);
  REQUIRE(out[2] == dict[3]);
  REQUIRE(out[3] == dict[1]);
  REQUIRE(out[4].empty());  // sel=0 → NULL/empty
  REQUIRE(out[5] == dict[2]);
}

TEST_CASE("gpu_decode_strings DICT_FSST mode 2 (FSST_ONLY) - basic round-trip",
          "[scan][decode][strings][dict_fsst]")
{
  std::vector<std::string> rows = {
    "alpha apple",
    "beta banana",
    "gamma grape",
    "delta dragonfruit",
  };
  std::vector<std::string> entries(rows.size() + 1);
  entries[0] = "";
  for (size_t i = 0; i < rows.size(); ++i)
    entries[i + 1] = rows[i];
  auto bytes = make_dict_fsst_segment(entries, /*selections=*/{}, /*mode=*/2);
  auto out   = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_DICT_FSST,
                                static_cast<uint32_t>(rows.size()),
                                /*max_len=*/32u);
  REQUIRE(out.size() == rows.size());
  for (size_t i = 0; i < rows.size(); ++i)
    REQUIRE(out[i] == rows[i]);
}

TEST_CASE("gpu_decode_strings DICT_FSST mode 1 - bench-scale verify",
          "[scan][decode][strings][dict_fsst][verify]")
{
  uint32_t const DICT = 256;
  uint32_t const ROWS = 50000;
  std::vector<std::string> dict(DICT);
  dict[0] = "";
  for (uint32_t k = 1; k < DICT; ++k) {
    dict[k] = "entry_" + std::to_string(k) + "_payload_" + std::to_string(k * 17u);
  }
  std::vector<uint32_t> sel(ROWS);
  for (uint32_t i = 0; i < ROWS; ++i)
    sel[i] = (i % (DICT - 1u)) + 1u;
  auto bytes = make_dict_fsst_segment(dict, sel, /*mode=*/1);
  auto out   = decode_one_segment(bytes,
                                CompressionType::COMPRESSION_DICT_FSST,
                                ROWS,
                                /*max_len=*/64u);
  REQUIRE(out.size() == ROWS);
  for (uint32_t i = 0; i < ROWS; ++i) {
    if (out[i] != dict[sel[i]]) {
      FAIL("DICT_FSST mode-1 mismatch at row=" << i << " sel=" << sel[i] << " expected='"
                                               << dict[sel[i]] << "' got='" << out[i] << "'");
    }
  }
}
