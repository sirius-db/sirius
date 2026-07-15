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

#include "catch.hpp"
#include "op/partition/crc32_partition_hash.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace sirius::op;
using decimal_key = crc32_partition_hash::decimal_key;

namespace {

// ---- CPU reference: the exact StarRocks primitive (zlib's crc32, poly 0xEDB88320) ----
// StarRocks' `zlib_crc_hash` is literally `crc32(seed, data, len)`, so use zlib directly as the
// oracle rather than reimplementing it.

uint32_t cpu_crc32(uint32_t seed, uint8_t const* p, std::size_t len)
{
  return static_cast<uint32_t>(crc32(seed, p, static_cast<uInt>(len)));
}

// StarRocks folds 4 zero bytes for a null row.
uint32_t cpu_fold_null(uint32_t seed)
{
  uint32_t const zero = 0;
  return cpu_crc32(seed, reinterpret_cast<uint8_t const*>(&zero), 4);
}

template <typename T>
uint32_t cpu_fold_fixed(uint32_t seed, T const& value, bool valid)
{
  return valid ? cpu_crc32(seed, reinterpret_cast<uint8_t const*>(&value), sizeof(T))
               : cpu_fold_null(seed);
}

uint32_t cpu_fold_string(uint32_t seed, std::string const& value, bool valid)
{
  if (!valid) { return cpu_fold_null(seed); }
  if (value.empty()) { return seed; }  // empty string is a no-op
  return cpu_crc32(seed, reinterpret_cast<uint8_t const*>(value.data()), value.size());
}

// DecimalV2 / DecimalV3(27,9) split fold: int64 integer part then int32 fractional part of the
// int128 value (C++ truncated division), matching StarRocks' DecimalV2Value.
uint32_t cpu_fold_decimal_split(uint32_t seed, __int128 v)
{
  int64_t const iv = static_cast<int64_t>(v / static_cast<__int128>(1000000000));
  int32_t const fv = static_cast<int32_t>(v % static_cast<__int128>(1000000000));
  uint32_t h       = cpu_crc32(seed, reinterpret_cast<uint8_t const*>(&iv), 8);
  return cpu_crc32(h, reinterpret_cast<uint8_t const*>(&fv), 4);
}

// ---- cudf column builders (host data -> device columns) ----

rmm::cuda_stream_view test_stream() { return cudf::get_default_stream(); }
rmm::device_async_resource_ref test_mr() { return cudf::get_current_device_resource_ref(); }

// Build a null mask from a per-row validity vector (empty vector => no mask / all valid).
rmm::device_buffer make_mask(std::vector<bool> const& valid, cudf::size_type& null_count)
{
  auto const n         = static_cast<cudf::size_type>(valid.size());
  auto const num_words = cudf::num_bitmask_words(n);
  std::vector<cudf::bitmask_type> words(static_cast<std::size_t>(num_words), 0);
  null_count = 0;
  for (cudf::size_type i = 0; i < n; ++i) {
    if (valid[i]) {
      words[i / 32] |= (cudf::bitmask_type{1} << (i % 32));
    } else {
      ++null_count;
    }
  }
  // cudf requires the mask buffer to be padded to its allocation size (multiple of 64 bytes).
  auto const buf_size = cudf::bitmask_allocation_size_bytes(n);
  rmm::device_buffer buf(buf_size, test_stream(), test_mr());
  cudaMemset(buf.data(), 0, buf_size);
  cudaMemcpy(
    buf.data(), words.data(), words.size() * sizeof(cudf::bitmask_type), cudaMemcpyHostToDevice);
  return buf;
}

template <typename T>
std::unique_ptr<cudf::column> make_fixed(cudf::data_type dt,
                                         std::vector<T> const& values,
                                         std::vector<bool> const& valid = {})
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col =
    cudf::make_fixed_width_column(dt, n, cudf::mask_state::UNALLOCATED, test_stream(), test_mr());
  cudaMemcpy(col->mutable_view().template data<T>(),
             values.data(),
             values.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  if (!valid.empty()) {
    cudf::size_type null_count = 0;
    auto mask                  = make_mask(valid, null_count);
    col->set_null_mask(std::move(mask), null_count);
  }
  return col;
}

std::unique_ptr<cudf::column> make_strings(std::vector<std::string> const& values,
                                           std::vector<bool> const& valid = {})
{
  auto const n = static_cast<cudf::size_type>(values.size());
  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(n) + 1, 0);
  for (cudf::size_type i = 0; i < n; ++i) {
    offsets[i + 1] = offsets[i] + static_cast<cudf::size_type>(values[i].size());
  }
  auto const total = offsets[n];
  std::vector<char> chars(static_cast<std::size_t>(total));
  for (cudf::size_type i = 0; i < n; ++i) {
    std::memcpy(chars.data() + offsets[i], values[i].data(), values[i].size());
  }

  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               n + 1,
                                               cudf::mask_state::UNALLOCATED,
                                               test_stream(),
                                               test_mr());
  cudaMemcpy(offsets_col->mutable_view().data<cudf::size_type>(),
             offsets.data(),
             offsets.size() * sizeof(cudf::size_type),
             cudaMemcpyHostToDevice);

  rmm::device_buffer chars_buf(static_cast<std::size_t>(total), test_stream(), test_mr());
  if (total > 0) {
    cudaMemcpy(chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice);
  }

  rmm::device_buffer null_mask{0, test_stream(), test_mr()};
  cudf::size_type null_count = 0;
  if (!valid.empty()) { null_mask = make_mask(valid, null_count); }

  return cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), null_count, std::move(null_mask));
}

std::vector<uint32_t> run_gpu(cudf::table_view const& keys,
                              std::vector<decimal_key> const& dks = {})
{
  auto hashes = crc32_partition_hash::compute(keys, dks, test_stream(), test_mr());
  test_stream().synchronize();
  std::vector<uint32_t> host(hashes.size());
  cudaMemcpy(host.data(), hashes.data(), hashes.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  return host;
}

}  // namespace

TEST_CASE("CRC32 reference matches published standard check values", "[crc32][partition]")
{
  // Anchor the CPU oracle to published standard CRC-32 values, independent of the GPU kernel and of
  // any StarRocks-semantics assumption. These pin the base algorithm (poly 0xEDB88320, seed 0).
  char const* s = "123456789";
  REQUIRE(cpu_crc32(0, reinterpret_cast<uint8_t const*>(s), 9) == 0xCBF43926u);
  REQUIRE(cpu_crc32(0, reinterpret_cast<uint8_t const*>("abc"), 3) == 0x352441C2u);
  uint32_t const four_zeros = 0;
  REQUIRE(cpu_crc32(0, reinterpret_cast<uint8_t const*>(&four_zeros), 4) == 0x2144DF1Cu);
}

TEST_CASE("CRC32 GPU output matches published constants for single values", "[crc32][partition]")
{
  // Independent of the CPU oracle: assert the GPU against externally-known standard CRC-32 values.
  // "abc" -> CRC32("abc"); a null row and an INT32 zero both feed four zero bytes -> CRC32(0x00*4).
  auto sc = make_strings({"abc"});
  REQUIRE(run_gpu(cudf::table_view({sc->view()}))[0] == 0x352441C2u);

  auto zc = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, {int32_t{0}});
  REQUIRE(run_gpu(cudf::table_view({zc->view()}))[0] == 0x2144DF1Cu);

  auto nc = make_fixed<int64_t>(cudf::data_type{cudf::type_id::INT64}, {int64_t{7}}, {false});
  REQUIRE(run_gpu(cudf::table_view({nc->view()}))[0] == 0x2144DF1Cu);
}

TEST_CASE("CRC32 GPU matches independent zlib golden vectors", "[crc32][partition]")
{
  // Golden values precomputed with Python's zlib (a third, independent CRC-32 implementation) for
  // the split fold, multi-column chaining, and null-after-value — catching a shared mistake that a
  // self-referential CPU oracle would miss.

  // DecimalV2 split fold of int128 values scaled by 10^9: 1234.0, -1.5, 0.
  std::vector<__int128_t> dv = {
    __int128_t{1234} * 1000000000, __int128_t{-1500000000}, __int128_t{0}};
  auto dcol = make_fixed<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, -9}, dv);
  auto dg   = run_gpu(cudf::table_view({dcol->view()}), {{0, 27, true}});
  REQUIRE(dg[0] == 0x7B8B1E9Eu);
  REQUIRE(dg[1] == 0x37105F5Du);
  REQUIRE(dg[2] == 0x7BD5C66Fu);

  // int32(0x01020304) then int64(0x1122334455667788).
  auto m0 = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, {int32_t{0x01020304}});
  auto m1 =
    make_fixed<int64_t>(cudf::data_type{cudf::type_id::INT64}, {int64_t{0x1122334455667788}});
  REQUIRE(run_gpu(cudf::table_view({m0->view(), m1->view()}))[0] == 0x8AA97128u);

  // int32(42) then a null column (4 zero bytes).
  auto n0 = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, {int32_t{42}});
  auto n1 = make_fixed<int64_t>(cudf::data_type{cudf::type_id::INT64}, {int64_t{0}}, {false});
  REQUIRE(run_gpu(cudf::table_view({n0->view(), n1->view()}))[0] == 0x0D94A1F7u);

  // string("abc") then a null column.
  auto s0 = make_strings({"abc"});
  auto s1 = make_fixed<int64_t>(cudf::data_type{cudf::type_id::INT64}, {int64_t{0}}, {false});
  REQUIRE(run_gpu(cudf::table_view({s0->view(), s1->view()}))[0] == 0x2B033BE3u);
}

TEST_CASE("CRC32 single INT32 column matches hand-computed bytes", "[crc32][partition]")
{
  int32_t const v = 0x01020304;
  auto col        = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, {v});
  cudf::table_view keys({col->view()});

  // Little-endian: the four bytes fed are 04 03 02 01.
  uint8_t const le[4] = {0x04, 0x03, 0x02, 0x01};
  auto const expected = cpu_crc32(0, le, 4);

  auto const gpu = run_gpu(keys);
  REQUIRE(gpu.size() == 1);
  REQUIRE(gpu[0] == expected);
  REQUIRE(gpu[0] == cpu_fold_fixed<int32_t>(0, v, true));
}

TEST_CASE("CRC32 null rows fold four zero bytes", "[crc32][partition]")
{
  std::vector<int64_t> vals = {5, 0, 7, 0};
  std::vector<bool> valid   = {true, false, true, false};
  auto col = make_fixed<int64_t>(cudf::data_type{cudf::type_id::INT64}, vals, valid);
  cudf::table_view keys({col->view()});

  auto const gpu = run_gpu(keys);
  REQUIRE(gpu.size() == vals.size());
  for (std::size_t i = 0; i < vals.size(); ++i) {
    REQUIRE(gpu[i] == cpu_fold_fixed<int64_t>(0, vals[i], valid[i]));
  }
  // A null bigint folds only 4 zero bytes, not 8.
  REQUIRE(gpu[1] == cpu_fold_null(0));
}

TEST_CASE("CRC32 empty and null strings", "[crc32][partition]")
{
  std::vector<std::string> vals = {"", "abc", "", "hello"};
  std::vector<bool> valid       = {true, true, false, true};
  auto col                      = make_strings(vals, valid);
  cudf::table_view keys({col->view()});

  auto const gpu = run_gpu(keys);
  REQUIRE(gpu.size() == vals.size());
  REQUIRE(gpu[0] == 0u);  // valid empty string: hash stays at the seed (0)
  REQUIRE(gpu[1] == cpu_fold_string(0, "abc", true));
  REQUIRE(gpu[2] == cpu_fold_null(0));  // null string: 4 zero bytes
  REQUIRE(gpu[3] == cpu_fold_string(0, "hello", true));
}

TEST_CASE("CRC32 DecimalV3 (general) hashes raw unscaled bytes", "[crc32][partition]")
{
  // DecimalV3 with a spec whose precision != 27 (or scale != -9) uses the raw-unscaled-bytes fold.
  std::vector<int64_t> d64 = {12345, -678, 0, 900000};
  auto c64 = make_fixed<int64_t>(cudf::data_type{cudf::type_id::DECIMAL64, -2}, d64);
  auto g64 = run_gpu(cudf::table_view({c64->view()}), {{0, 18, false}});
  for (std::size_t i = 0; i < d64.size(); ++i) {
    REQUIRE(g64[i] == cpu_fold_fixed<int64_t>(0, d64[i], true));
  }

  std::vector<__int128_t> d128 = {
    __int128_t{1}, __int128_t{-1}, (__int128_t{1} << 100), __int128_t{0}};
  auto c128 = make_fixed<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, -2}, d128);
  auto g128 = run_gpu(cudf::table_view({c128->view()}), {{0, 38, false}});
  for (std::size_t i = 0; i < d128.size(); ++i) {
    REQUIRE(g128[i] == cpu_fold_fixed<__int128_t>(0, d128[i], true));
  }
}

TEST_CASE("CRC32 DecimalV2 and DecimalV3(27,9) use the split fold", "[crc32][partition]")
{
  // int128 values scaled by 10^9 (e.g. 1234.0 -> 1234e9, -1.5 -> -1.5e9).
  std::vector<__int128_t> v = {
    __int128_t{1234} * 1000000000, __int128_t{-1500000000}, __int128_t{0}};
  auto col = make_fixed<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, -9}, v);

  // DecimalV2: always the split fold.
  auto v2 = run_gpu(cudf::table_view({col->view()}), {{0, 27, true}});
  // DecimalV3 precision 27, scale 9: StarRocks' compatibility split fold — must match V2 exactly.
  auto v3 = run_gpu(cudf::table_view({col->view()}), {{0, 27, false}});
  for (std::size_t i = 0; i < v.size(); ++i) {
    REQUIRE(v2[i] == cpu_fold_decimal_split(0, v[i]));
    REQUIRE(v3[i] == v2[i]);
  }
}

TEST_CASE("CRC32 multi-column mixed types with nulls match CPU oracle", "[crc32][partition]")
{
  constexpr int n = 64;
  std::vector<int32_t> a(n);
  std::vector<bool> a_valid(n);
  std::vector<std::string> b(n);
  std::vector<bool> b_valid(n);
  std::vector<int64_t> c(n);  // DECIMAL64 storage
  std::vector<bool> c_valid(n);

  for (int i = 0; i < n; ++i) {
    a[i]       = (i * 2654435761u) & 0x7fffffff;
    a_valid[i] = (i % 5) != 0;
    b[i]       = (i % 3 == 0) ? std::string() : ("key_" + std::to_string(i * 7));
    b_valid[i] = (i % 7) != 0;
    c[i]       = static_cast<int64_t>(i) * 1000 - 12345;
    c_valid[i] = (i % 4) != 0;
  }

  auto ca = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, a, a_valid);
  auto cb = make_strings(b, b_valid);
  auto cc = make_fixed<int64_t>(cudf::data_type{cudf::type_id::DECIMAL64, -3}, c, c_valid);
  cudf::table_view keys({ca->view(), cb->view(), cc->view()});

  auto const gpu = run_gpu(keys, {{2, 18, false}});  // col 2 is DecimalV3 (raw)
  REQUIRE(static_cast<int>(gpu.size()) == n);
  for (int i = 0; i < n; ++i) {
    uint32_t h = 0;  // seed
    h          = cpu_fold_fixed<int32_t>(h, a[i], a_valid[i]);
    h          = cpu_fold_string(h, b[i], b_valid[i]);
    h          = cpu_fold_fixed<int64_t>(h, c[i], c_valid[i]);
    REQUIRE(gpu[i] == h);
  }
}

TEST_CASE("CRC32 key column order changes the hash", "[crc32][partition]")
{
  std::vector<int32_t> x = {1, 2, 3, 4};
  std::vector<int64_t> y = {10, 20, 30, 40};
  auto cx                = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, x);
  auto cy                = make_fixed<int64_t>(cudf::data_type{cudf::type_id::INT64}, y);

  auto xy = run_gpu(cudf::table_view({cx->view(), cy->view()}));
  auto yx = run_gpu(cudf::table_view({cy->view(), cx->view()}));

  // Both must match the CPU oracle in their respective fold order...
  for (std::size_t i = 0; i < x.size(); ++i) {
    uint32_t h_xy = cpu_fold_fixed<int64_t>(cpu_fold_fixed<int32_t>(0, x[i], true), y[i], true);
    uint32_t h_yx = cpu_fold_fixed<int32_t>(cpu_fold_fixed<int64_t>(0, y[i], true), x[i], true);
    REQUIRE(xy[i] == h_xy);
    REQUIRE(yx[i] == h_yx);
  }
  // ...and the two orders must differ for at least one row.
  bool any_diff = false;
  for (std::size_t i = 0; i < x.size(); ++i) {
    any_diff = any_diff || (xy[i] != yx[i]);
  }
  REQUIRE(any_diff);
}

TEST_CASE("CRC32 covers every supported fixed-width dispatch branch", "[crc32][partition]")
{
  // One column per remaining switch branch, so a wrong-width dispatch (e.g. reading an INT16 as
  // INT32) diverges from the oracle's per-type byte count and is caught.
  constexpr int n = 8;
  std::vector<uint8_t> vbool(n), vu8(n);
  std::vector<int8_t> v8(n);
  std::vector<int16_t> v16(n);
  std::vector<uint16_t> vu16(n);
  std::vector<uint32_t> vu32(n);
  std::vector<uint64_t> vu64(n);
  std::vector<float> vf(n);
  std::vector<double> vd(n);
  std::vector<int32_t> vdec32(n);  // DECIMAL32 storage
  for (int i = 0; i < n; ++i) {
    vbool[i]  = static_cast<uint8_t>(i & 1);
    vu8[i]    = static_cast<uint8_t>(i * 37 + 1);
    v8[i]     = static_cast<int8_t>(i * 11 - 40);
    v16[i]    = static_cast<int16_t>(i * 2027 - 8000);
    vu16[i]   = static_cast<uint16_t>(i * 4099 + 3);
    vu32[i]   = static_cast<uint32_t>(i * 2654435761u);
    vu64[i]   = static_cast<uint64_t>(i) * 1099511628211ull + 7;
    vf[i]     = static_cast<float>(i) * 1.5f - 3.25f;
    vd[i]     = static_cast<double>(i) * 2.5 - 100.0;
    vdec32[i] = i * 137 - 400;
  }

  auto cbool = make_fixed<uint8_t>(cudf::data_type{cudf::type_id::BOOL8}, vbool);
  auto cu8   = make_fixed<uint8_t>(cudf::data_type{cudf::type_id::UINT8}, vu8);
  auto c8    = make_fixed<int8_t>(cudf::data_type{cudf::type_id::INT8}, v8);
  auto c16   = make_fixed<int16_t>(cudf::data_type{cudf::type_id::INT16}, v16);
  auto cu16  = make_fixed<uint16_t>(cudf::data_type{cudf::type_id::UINT16}, vu16);
  auto cu32  = make_fixed<uint32_t>(cudf::data_type{cudf::type_id::UINT32}, vu32);
  auto cu64  = make_fixed<uint64_t>(cudf::data_type{cudf::type_id::UINT64}, vu64);
  auto cf    = make_fixed<float>(cudf::data_type{cudf::type_id::FLOAT32}, vf);
  auto cd    = make_fixed<double>(cudf::data_type{cudf::type_id::FLOAT64}, vd);
  auto cdec  = make_fixed<int32_t>(cudf::data_type{cudf::type_id::DECIMAL32, -2}, vdec32);

  cudf::table_view keys({cbool->view(),
                         cu8->view(),
                         c8->view(),
                         c16->view(),
                         cu16->view(),
                         cu32->view(),
                         cu64->view(),
                         cf->view(),
                         cd->view(),
                         cdec->view()});
  auto const gpu = run_gpu(keys, {{9, 18, false}});  // col 9 is DecimalV3 (raw)
  REQUIRE(static_cast<int>(gpu.size()) == n);
  for (int i = 0; i < n; ++i) {
    uint32_t h = 0;
    h          = cpu_fold_fixed<uint8_t>(h, vbool[i], true);
    h          = cpu_fold_fixed<uint8_t>(h, vu8[i], true);
    h          = cpu_fold_fixed<int8_t>(h, v8[i], true);
    h          = cpu_fold_fixed<int16_t>(h, v16[i], true);
    h          = cpu_fold_fixed<uint16_t>(h, vu16[i], true);
    h          = cpu_fold_fixed<uint32_t>(h, vu32[i], true);
    h          = cpu_fold_fixed<uint64_t>(h, vu64[i], true);
    h          = cpu_fold_fixed<float>(h, vf[i], true);
    h          = cpu_fold_fixed<double>(h, vd[i], true);
    h          = cpu_fold_fixed<int32_t>(h, vdec32[i], true);
    REQUIRE(gpu[i] == h);
  }
}

TEST_CASE("CRC32 returns empty for zero-row inputs", "[crc32][partition]")
{
  // A zero-row STRING column may be the childless make_empty_column representation; validation must
  // not touch its (absent) offsets child. Result is an empty hash vector.
  auto empty_str = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto empty_i32 = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  cudf::table_view keys({empty_str->view(), empty_i32->view()});
  auto hashes = crc32_partition_hash::compute(keys, {}, test_stream(), test_mr());
  test_stream().synchronize();
  REQUIRE(hashes.size() == 0);
}

TEST_CASE("CRC32 handles sliced fixed-width columns (offset != 0)", "[crc32][partition]")
{
  // Exercises the offset-adjusted data base (head + offset*width) and mask offset for a sliced
  // fixed column. Fixed columns may be sliced (unlike strings, which require offset 0).
  std::vector<int32_t> v  = {10, 20, 30, 40, 50, 60};
  std::vector<bool> valid = {true, false, true, true, false, true};
  auto col                = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, v, valid);
  auto sliced             = cudf::slice(col->view(), {2, 5})[0];  // logical rows 2..4, offset 2
  cudf::table_view keys({sliced});

  auto const gpu = run_gpu(keys);
  REQUIRE(gpu.size() == 3);
  for (int i = 0; i < 3; ++i) {
    REQUIRE(gpu[i] == cpu_fold_fixed<int32_t>(0, v[2 + i], valid[2 + i]));
  }
}

TEST_CASE("CRC32 rejects mismatched or missing decimal specs", "[crc32][partition]")
{
  auto d64 = make_fixed<int64_t>(cudf::data_type{cudf::type_id::DECIMAL64, -2}, {int64_t{1}});
  auto d128s9 =
    make_fixed<__int128_t>(cudf::data_type{cudf::type_id::DECIMAL128, -9}, {__int128_t{1}});
  auto i32 = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, {int32_t{1}});
  auto sr  = test_stream();
  auto mr  = test_mr();

  // A decimal key with no spec is rejected (cudf can't disambiguate V2/V3).
  REQUIRE_THROWS_AS(crc32_partition_hash::compute(cudf::table_view({d64->view()}), {}, sr, mr),
                    std::invalid_argument);
  // A spec on a non-decimal column is rejected.
  REQUIRE_THROWS_AS(
    crc32_partition_hash::compute(cudf::table_view({i32->view()}), {{0, 18, false}}, sr, mr),
    std::invalid_argument);
  // DecimalV2 must be DECIMAL128 with scale -9.
  REQUIRE_THROWS_AS(
    crc32_partition_hash::compute(cudf::table_view({d64->view()}), {{0, 27, true}}, sr, mr),
    std::invalid_argument);
  // Out-of-range and duplicate spec column indices are rejected.
  REQUIRE_THROWS_AS(
    crc32_partition_hash::compute(cudf::table_view({d128s9->view()}), {{5, 27, true}}, sr, mr),
    std::invalid_argument);
  REQUIRE_THROWS_AS(crc32_partition_hash::compute(
                      cudf::table_view({d128s9->view()}), {{0, 27, true}, {0, 27, false}}, sr, mr),
                    std::invalid_argument);
}

TEST_CASE("CRC32 rejects unsupported and sliced columns before launching", "[crc32][partition]")
{
  auto sr = test_stream();
  auto mr = test_mr();

  // Empty key list.
  cudf::table_view const empty_keys{std::vector<cudf::column_view>{}};
  REQUIRE_THROWS_AS(crc32_partition_hash::compute(empty_keys, {}, sr, mr), std::invalid_argument);

  // Unsupported type (date/timestamp is deferred).
  auto ts = make_fixed<int32_t>(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, {int32_t{1}, 2, 3});
  REQUIRE_THROWS_AS(crc32_partition_hash::compute(cudf::table_view({ts->view()}), {}, sr, mr),
                    std::invalid_argument);

  // Sliced string column (offset != 0).
  auto s      = make_strings({"a", "bb", "ccc", "dddd"});
  auto sliced = cudf::slice(s->view(), {1, 3})[0];  // offset 1
  REQUIRE_THROWS_AS(crc32_partition_hash::compute(cudf::table_view({sliced}), {}, sr, mr),
                    std::invalid_argument);

  // A supported column followed by an unsupported one must also throw (and not launch on col 0).
  auto ok = make_fixed<int32_t>(cudf::data_type{cudf::type_id::INT32}, {int32_t{1}, 2, 3});
  REQUIRE_THROWS_AS(
    crc32_partition_hash::compute(cudf::table_view({ok->view(), ts->view()}), {}, sr, mr),
    std::invalid_argument);
}
