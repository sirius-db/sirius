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

#include "scan/decode_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

using duckdb::CompressionType;
using sirius::cuda::scan::gpu_column_decode_input;
using sirius::cuda::scan::gpu_segment_desc;
using sirius::test::decode::decode_env;
using sirius::test::decode::download;
using sirius::test::decode::one_codec_column;
using sirius::test::decode::pack_validity;
using sirius::test::decode::require_constant_broadcast;
using sirius::test::decode::require_uncompressed_roundtrip;
using sirius::test::decode::segment;
using sirius::test::decode::upload;

namespace {
auto const I32 = cudf::data_type{cudf::type_id::INT32};
auto const I64 = cudf::data_type{cudf::type_id::INT64};
auto const F32 = cudf::data_type{cudf::type_id::FLOAT32};
auto const F64 = cudf::data_type{cudf::type_id::FLOAT64};
auto const STR = cudf::data_type{cudf::type_id::STRING};
auto const DEC128 =
  cudf::data_type{cudf::type_id::DECIMAL128, /*scale=*/-2};  // 16-byte physical width
}  // namespace

TEST_CASE_METHOD(decode_env, "gpu_decode_table - empty input yields empty table", "[scan][decode]")
{
  auto t = decode({});
  REQUIRE(t->num_columns() == 0);
  REQUIRE(t->num_rows() == 0);
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - UNCOMPRESSED single segment int32",
                 "[scan][decode]")
{
  std::vector<int32_t> values(1024);
  std::iota(values.begin(), values.end(), -512);
  require_uncompressed_roundtrip(*this, I32, values);
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - UNCOMPRESSED float", "[scan][decode]")
{
  require_uncompressed_roundtrip<float>(*this, F32, {1.5f, -2.25f, 3.75f, 0.0f, -1.0f});
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - UNCOMPRESSED double", "[scan][decode]")
{
  require_uncompressed_roundtrip<double>(*this, F64, {1.5, -2.25, 3.75, 0.0, -1.0});
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - UNCOMPRESSED multi-segment int64",
                 "[scan][decode]")
{
  // Two segments tiling the column [0..800) and [800..2000). The dispatcher's
  // multi-segment path (batched-memcpy kernel) handles this differently from
  // the single-segment fast path, so it gets its own test.
  std::vector<int64_t> a(800), b(1200);
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 800);
  auto da  = upload(a, stream.view());
  auto db  = upload(b, stream.view());
  auto col = one_codec_column(I64,
                              a.size() + b.size(),
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              {segment(da, 0, a.size()), segment(db, a.size(), b.size())});

  auto t = decode({col});
  auto out =
    download<int64_t>(t->get_column(0).view().data<int64_t>(), col.total_rows, stream.value());
  std::vector<int64_t> expected(a);
  expected.insert(expected.end(), b.begin(), b.end());
  REQUIRE(out == expected);
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - UNCOMPRESSED with nulls", "[scan][decode]")
{
  std::vector<int32_t> values = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<bool> valid     = {true, true, false, true, false, false, true, true};
  auto valid_bytes            = pack_validity(valid);
  auto d_data                 = upload(values, stream.view());
  rmm::device_buffer d_validity(valid_bytes.data(), valid_bytes.size(), stream.view());

  auto col = one_codec_column(I32,
                              values.size(),
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              {segment(d_data, 0, values.size())},
                              /*has_nulls=*/true);
  col.validity.push_back(
    {CompressionType::COMPRESSION_UNCOMPRESSED, {segment(d_validity, 0, values.size())}});

  auto t = decode({col});
  auto out =
    download<int32_t>(t->get_column(0).view().data<int32_t>(), values.size(), stream.value());
  REQUIRE(t->get_column(0).null_count() == 3);
  for (size_t i = 0; i < values.size(); ++i)
    if (valid[i]) REQUIRE(out[i] == values[i]);
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - UNCOMPRESSED multi-segment validity",
                 "[scan][decode]")
{
  // Validity arrives as multiple byte-aligned runs; each segment's row_offset
  // is a multiple of 8, and runs cover disjoint row ranges.
  constexpr uint32_t N = 32;
  std::vector<int32_t> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> all_valid(N, true);
  for (uint32_t i = 0; i < N; ++i)
    all_valid[i] = (i % 5) != 0;  // every 5th row is null
  auto packed_a = pack_validity({all_valid.begin(), all_valid.begin() + 16});
  auto packed_b = pack_validity({all_valid.begin() + 16, all_valid.end()});

  auto d_data = upload(values, stream.view());
  rmm::device_buffer d_va(packed_a.data(), packed_a.size(), stream.view());
  rmm::device_buffer d_vb(packed_b.data(), packed_b.size(), stream.view());

  auto col = one_codec_column(I32,
                              N,
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              {segment(d_data, 0, N)},
                              /*has_nulls=*/true);
  col.validity.push_back(
    {CompressionType::COMPRESSION_UNCOMPRESSED, {segment(d_va, 0, 16), segment(d_vb, 16, 16)}});

  auto t            = decode({col});
  size_t null_count = 0;
  for (uint32_t i = 0; i < N; ++i)
    if (!all_valid[i]) ++null_count;
  REQUIRE(t->get_column(0).null_count() == static_cast<cudf::size_type>(null_count));
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - CONSTANT int32 single segment", "[scan][decode]")
{
  require_constant_broadcast<int32_t>(*this, I32, /*value=*/42, /*rows=*/2048);
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - CONSTANT double", "[scan][decode]")
{
  require_constant_broadcast<double>(*this, F64, /*value=*/3.14159, /*rows=*/100);
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - CONSTANT multi-segment different values",
                 "[scan][decode]")
{
  std::vector<int64_t> va = {7}, vb = {-3};
  auto da = upload(va, stream.view()), db = upload(vb, stream.view());
  auto col = one_codec_column(
    I64, 1500, CompressionType::COMPRESSION_CONSTANT, {segment(da, 0, 800), segment(db, 800, 700)});
  auto t   = decode({col});
  auto out = download<int64_t>(t->get_column(0).view().data<int64_t>(), 1500, stream.value());
  for (size_t i = 0; i < 800; ++i)
    REQUIRE(out[i] == 7);
  for (size_t i = 800; i < 1500; ++i)
    REQUIRE(out[i] == -3);
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - CONSTANT 16-byte decimal128 (scalar fallback path)",
                 "[scan][decode]")
{
  // Exercises the kernel's `else` branch (sizeof(T) > 8) — the int4 vectorized
  // path is gated on sizeof(T) <= 8. Compared as raw bytes so the test does
  // not depend on cudf's decimal128 type-check on .data<T>().
  std::vector<uint8_t> pattern(16);
  for (uint32_t i = 0; i < 16; ++i)
    pattern[i] = static_cast<uint8_t>(0xA0u + i);
  rmm::device_buffer d_val(pattern.data(), pattern.size(), stream.view());

  constexpr uint32_t ROWS = 257;
  auto col                = one_codec_column(
    DEC128, ROWS, CompressionType::COMPRESSION_CONSTANT, {segment(d_val, 0, ROWS)});
  auto t = decode({col});

  auto bytes = download<uint8_t>(
    static_cast<uint8_t const*>(t->get_column(0).view().head()), ROWS * 16, stream.value());
  for (uint32_t r = 0; r < ROWS; ++r)
    for (uint32_t i = 0; i < 16; ++i)
      REQUIRE(bytes[r * 16 + i] == pattern[i]);
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - mixed codecs across columns", "[scan][decode]")
{
  std::vector<int32_t> a(500), b(500);
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 500);
  std::vector<int64_t> k = {99};
  std::vector<double> dv(1000, 1.5);
  auto da = upload(a, stream.view()), db = upload(b, stream.view());
  auto dk = upload(k, stream.view()), dd = upload(dv, stream.view());

  std::vector<gpu_column_decode_input> cols = {
    one_codec_column(I32,
                     1000,
                     CompressionType::COMPRESSION_UNCOMPRESSED,
                     {segment(da, 0, 500), segment(db, 500, 500)}),
    one_codec_column(I64, 1000, CompressionType::COMPRESSION_CONSTANT, {segment(dk, 0, 1000)}),
    one_codec_column(F64, 1000, CompressionType::COMPRESSION_UNCOMPRESSED, {segment(dd, 0, 1000)})};

  auto t = decode(cols);
  REQUIRE(t->num_columns() == 3);
  auto o0 = download<int32_t>(t->get_column(0).view().data<int32_t>(), 1000, stream.value());
  auto o1 = download<int64_t>(t->get_column(1).view().data<int64_t>(), 1000, stream.value());
  auto o2 = download<double>(t->get_column(2).view().data<double>(), 1000, stream.value());
  for (size_t i = 0; i < 1000; ++i) {
    REQUIRE(o0[i] == static_cast<int32_t>(i));
    REQUIRE(o1[i] == 99);
    REQUIRE(o2[i] == 1.5);
  }
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - throws on unsupported codec", "[scan][decode]")
{
  std::vector<int32_t> dummy = {0};
  auto d                     = upload(dummy, stream.view());
  // BITPACKING is a real DuckDB codec the dispatcher doesn't yet implement —
  // picking it gives us a concrete "unimplemented codec" path without
  // inventing a fake enum.
  auto col = one_codec_column(I32, 1, CompressionType::COMPRESSION_BITPACKING, {segment(d, 0, 1)});

  REQUIRE_THROWS_WITH(decode({col}), Catch::Contains("viability invariant violated"));
}

TEST_CASE_METHOD(decode_env, "gpu_decode_table - throws on non-fixed-width type", "[scan][decode]")
{
  // STRING is variable-width, so `cudf::size_of` would itself throw if we
  // ever reached it. The dispatcher must refuse the type *before* asking cudf
  // for its size; this test pins that ordering.
  std::vector<uint8_t> bytes = {0};
  auto d                     = upload(bytes, stream.view());
  auto col =
    one_codec_column(STR, 1, CompressionType::COMPRESSION_UNCOMPRESSED, {segment(d, 0, 1)});

  REQUIRE_THROWS_WITH(decode({col}), Catch::Contains("non-fixed-width type"));
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - throws on unaligned validity row_offset",
                 "[scan][decode]")
{
  std::vector<int32_t> values(16, 0);
  std::vector<uint8_t> v_bytes = {0xFFu, 0xFFu};
  auto d_data                  = upload(values, stream.view());
  rmm::device_buffer d_validity(v_bytes.data(), v_bytes.size(), stream.view());

  auto col = one_codec_column(I32,
                              values.size(),
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              {segment(d_data, 0, values.size())},
                              /*has_nulls=*/true);
  // row_offset = 4 is byte-misaligned (4 % 8 != 0); the dispatcher must reject
  // it before issuing a memcpy at a fractional byte offset.
  col.validity.push_back({CompressionType::COMPRESSION_UNCOMPRESSED,
                          {segment(d_validity, /*row_offset=*/4, /*row_count=*/12)}});

  REQUIRE_THROWS_WITH(decode({col}), Catch::Contains("not byte-aligned"));
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - has_nulls with zero rows produces empty mask",
                 "[scan][decode]")
{
  gpu_column_decode_input col;
  col.out_type   = I32;
  col.total_rows = 0;
  col.has_nulls  = true;
  // No data runs, no validity runs — the dispatcher must still produce a valid
  // 0-row column rather than launching kernels on an empty mask.

  auto t = decode({col});
  REQUIRE(t->num_rows() == 0);
  REQUIRE(t->get_column(0).size() == 0);
  REQUIRE(t->get_column(0).null_count() == 0);
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - throws on UNCOMPRESSED segment too small for row_count",
                 "[scan][decode]")
{
  // Segment claims 16 rows but the source buffer only holds 4 int32s
  // (16 bytes vs the 64 needed). Without the bytes_size check the dispatcher
  // would memcpy past end-of-buffer.
  std::vector<int32_t> small_buffer = {1, 2, 3, 4};
  auto d                            = upload(small_buffer, stream.view());
  auto col =
    one_codec_column(I32, 16, CompressionType::COMPRESSION_UNCOMPRESSED, {segment(d, 0, 16)});

  REQUIRE_THROWS_WITH(decode({col}), Catch::Contains("UNCOMPRESSED segment bytes_size"));
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - throws on validity segment too small for row_count",
                 "[scan][decode]")
{
  // Claims 64 rows of validity but only one byte (8 bits) is staged.
  std::vector<int32_t> values(64, 0);
  std::vector<uint8_t> too_small_validity = {0xFFu};
  auto d_data                             = upload(values, stream.view());
  rmm::device_buffer d_validity(
    too_small_validity.data(), too_small_validity.size(), stream.view());

  auto col = one_codec_column(I32,
                              values.size(),
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              {segment(d_data, 0, values.size())},
                              /*has_nulls=*/true);
  col.validity.push_back({CompressionType::COMPRESSION_UNCOMPRESSED,
                          {segment(d_validity, /*row_offset=*/0, /*row_count=*/64)}});

  REQUIRE_THROWS_WITH(decode({col}), Catch::Contains("validity segment bytes_size"));
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - throws on segment row_offset+row_count beyond total_rows",
                 "[scan][decode]")
{
  // Claims a segment covering [50..150) but the column only has 100 rows.
  // Without the bounds check the codec would write 50 rows past end-of-buffer.
  std::vector<int32_t> data(100, 7);
  auto d   = upload(data, stream.view());
  auto col = one_codec_column(
    I32, 100, CompressionType::COMPRESSION_UNCOMPRESSED, {segment(d, /*row_offset=*/50, 100)});

  REQUIRE_THROWS_WITH(decode({col}), Catch::Contains("> total_rows"));
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - CONSTANT broadcast tolerates misaligned destination",
                 "[scan][decode]")
{
  // Two CONSTANT runs on an int16 column: the first segment ends at row 4,
  // putting the second segment's destination at base + 8 bytes — 8-byte
  // aligned but NOT 16-byte aligned. The kernel must fall back to scalar
  // stores rather than emit a misaligned int4 store (UB in C++ and an actual
  // perf/fault risk on some GPUs).
  cudf::data_type const I16{cudf::type_id::INT16};
  std::vector<int16_t> va = {111}, vb = {-222};
  auto da  = upload(va, stream.view());
  auto db  = upload(vb, stream.view());
  auto col = one_codec_column(I16,
                              12,
                              CompressionType::COMPRESSION_CONSTANT,
                              {segment(da, /*row_offset=*/0, /*row_count=*/4),
                               segment(db, /*row_offset=*/4, /*row_count=*/8)});
  auto t   = decode({col});
  auto out = download<int16_t>(t->get_column(0).view().data<int16_t>(), 12, stream.value());
  for (size_t i = 0; i < 4; ++i)
    REQUIRE(out[i] == 111);
  for (size_t i = 4; i < 12; ++i)
    REQUIRE(out[i] == -222);
}

TEST_CASE_METHOD(decode_env,
                 "gpu_decode_table - throws on row count beyond cudf::size_type max",
                 "[scan][decode]")
{
  // The throw fires from the row-count value alone, before any device
  // allocation — no need for the dummy segment to actually back that many
  // rows. Pins the guard so a future edit doesn't silently drop it.
  std::vector<int32_t> dummy = {0};
  auto d                     = upload(dummy, stream.view());

  gpu_column_decode_input col;
  col.out_type   = I32;
  col.total_rows = static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1u;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_UNCOMPRESSED, {segment(d, 0, 1)}});

  REQUIRE_THROWS_WITH(decode({col}), Catch::Contains("cudf::size_type max"));
}
