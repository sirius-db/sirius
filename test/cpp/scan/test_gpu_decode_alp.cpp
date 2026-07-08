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

// ALP / ALPRD decode tests: happy-path round-trips against host references,
// plus canary-buffer defensive tests that pre-fill output with 0xCC and
// assert the kernel zero-filled the row range on each malformed input.

#include "scan/alp_synth.hpp"
#include "scan/decode_test_utils.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_alp.cuh>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/storage/compression/alp/alp_constants.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using duckdb::CompressionType;
using sirius::cuda::scan::decode_alp_data;
using sirius::cuda::scan::decode_alprd_data;
using sirius::cuda::scan::gpu_codec_run;
using sirius::cuda::scan::gpu_column_decode_input;
using sirius::cuda::scan::gpu_decode_table;
using sirius::cuda::scan::gpu_segment_desc;
using sirius::test::decode::download;
using sirius::test::decode::one_codec_column;
using sirius::test::decode::segment;
using sirius::test::decode::upload;
using sirius::test::decode::verify_decoded_column;
using sirius::test::decode::alp::alp_compressed_vector;
using sirius::test::decode::alp::alp_uncompressed_vector;
using sirius::test::decode::alp::alp_vector;
using sirius::test::decode::alp::alprd_compressed_vector;
using sirius::test::decode::alp::alprd_uncompressed_vector;
using sirius::test::decode::alp::alprd_vector;
using sirius::test::decode::alp::make_alp_block;
using sirius::test::decode::alp::make_alprd_block;
using sirius::test::decode::alp::synth_alp_segment;
using sirius::test::decode::alp::synth_alprd_segment;

namespace {

auto const F32 = cudf::data_type{cudf::type_id::FLOAT32};
auto const F64 = cudf::data_type{cudf::type_id::FLOAT64};

template <typename T>
std::vector<T> decode_one_alp(std::vector<uint8_t> const& bytes,
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
  col.data.push_back({CompressionType::COMPRESSION_ALP,
                      {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                        static_cast<uint32_t>(d_seg.size()),
                                        0,
                                        row_count}}});
  auto t = gpu_decode_table({col}, stream.view(), mr);
  REQUIRE(t->num_rows() == static_cast<cudf::size_type>(row_count));
  return download<T>(t->get_column(0).view().data<T>(), row_count, stream.value());
}

template <typename T>
std::vector<T> decode_one_alprd(std::vector<uint8_t> const& bytes,
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
  col.data.push_back({CompressionType::COMPRESSION_ALPRD,
                      {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                        static_cast<uint32_t>(d_seg.size()),
                                        0,
                                        row_count}}});
  auto t = gpu_decode_table({col}, stream.view(), mr);
  REQUIRE(t->num_rows() == static_cast<cudf::size_type>(row_count));
  return download<T>(t->get_column(0).view().data<T>(), row_count, stream.value());
}

/// Reference ALP decode on host: enc → real-T plain via FACT × FRAC.
/// Uses DuckDB's source-of-truth tables so this stays in sync with the
/// kernel's `d_alp_fact` / `alp_frac_ptr<T>()` mirrors.
inline double alp_decode_f64_ref(uint64_t for_value,
                                 uint64_t mantissa,
                                 uint8_t factor,
                                 uint8_t exponent)
{
  int64_t enc = static_cast<int64_t>(mantissa + for_value);
  return static_cast<double>(enc) * static_cast<double>(duckdb::AlpConstants::FACT_ARR[factor]) *
         duckdb::AlpTypedConstants<double>::FRAC_ARR[exponent];
}

inline float alp_decode_f32_ref(uint64_t for_value,
                                uint64_t mantissa,
                                uint8_t factor,
                                uint8_t exponent)
{
  int64_t enc = static_cast<int64_t>(mantissa + for_value);
  return static_cast<float>(enc) * static_cast<float>(duckdb::AlpConstants::FACT_ARR[factor]) *
         duckdb::AlpTypedConstants<float>::FRAC_ARR[exponent];
}

}  // namespace

//===----------------------------------------------------------------------===//
// ALP happy-path tests.
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_decode_table ALP - single compressed vector, double, factor/exponent ≠ 0",
          "[scan][decode][alp]")
{
  // 100 values: i / 100.0 for i in [0,100). Encoded as integers i with
  // factor=0, exponent=2, FOR=0, bit_width=7.
  uint32_t const N = 100;
  std::vector<uint64_t> mantissas(N);
  for (uint32_t i = 0; i < N; ++i)
    mantissas[i] = i;

  alp_compressed_vector<double> cv;
  cv.exponent     = 2;
  cv.factor       = 0;
  cv.frame_of_ref = 0;
  cv.bit_width    = 7;
  cv.row_count    = N;
  cv.mantissas    = mantissas;

  std::vector<alp_vector<double>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes = make_alp_block<double>(vecs);
  auto out   = decode_one_alp<double>(bytes, F64, N);

  for (uint32_t i = 0; i < N; ++i) {
    double expected = alp_decode_f64_ref(0, mantissas[i], 0, 2);
    REQUIRE(out[i] == Approx(expected).epsilon(1e-12));
  }
}

TEST_CASE("gpu_decode_table ALP - single compressed vector with exceptions, float",
          "[scan][decode][alp]")
{
  uint32_t const N = 64;
  std::vector<uint64_t> mantissas(N);
  for (uint32_t i = 0; i < N; ++i)
    mantissas[i] = i * 7u;

  alp_compressed_vector<float> cv;
  cv.exponent     = 1;
  cv.factor       = 0;
  cv.frame_of_ref = 100;
  cv.bit_width    = 10;
  cv.row_count    = N;
  cv.mantissas    = mantissas;
  // Two exceptions — overwrite positions 5 and 17 with values that don't
  // round-trip the (factor, exponent) encoding.
  cv.exceptions.push_back(3.1415926F);
  cv.exception_positions.push_back(5);
  cv.exceptions.push_back(2.7182818F);
  cv.exception_positions.push_back(17);

  std::vector<alp_vector<float>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes = make_alp_block<float>(vecs);
  auto out   = decode_one_alp<float>(bytes, F32, N);

  for (uint32_t i = 0; i < N; ++i) {
    if (i == 5) {
      // Exception bytes are stored verbatim and bit-cast back, no arithmetic
      // ⇒ require bit-exact equality, not approximate.
      REQUIRE(out[i] == 3.1415926F);
    } else if (i == 17) {
      REQUIRE(out[i] == 2.7182818F);
    } else {
      float expected = alp_decode_f32_ref(100, mantissas[i], 0, 1);
      REQUIRE(out[i] == Approx(expected).epsilon(1e-6F));
    }
  }
}

TEST_CASE("gpu_decode_table ALP - uncompressed-sentinel vector double", "[scan][decode][alp]")
{
  // A vector with the 0xFF exponent sentinel: raw doubles, no encoding.
  uint32_t const N = 50;
  std::vector<double> raw(N);
  for (uint32_t i = 0; i < N; ++i)
    raw[i] = static_cast<double>(i) * 3.14159 + 0.5;

  alp_uncompressed_vector<double> uv;
  uv.row_count = N;
  uv.values    = raw;

  std::vector<alp_vector<double>> vecs;
  vecs.emplace_back(std::move(uv));
  auto bytes = make_alp_block<double>(vecs);
  auto out   = decode_one_alp<double>(bytes, F64, N);

  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == raw[i]);
}

TEST_CASE("gpu_decode_table ALP - multi-vector segment crosses 1024-row boundary",
          "[scan][decode][alp]")
{
  // Two vectors: one full (1024 rows) + one short (300 rows). Verifies the
  // last-vector tail-row case + multi-vector backwards trailer parsing.
  uint32_t const N0 = 1024;
  uint32_t const N1 = 300;
  std::vector<uint64_t> m0(N0), m1(N1);
  for (uint32_t i = 0; i < N0; ++i)
    m0[i] = i;
  for (uint32_t i = 0; i < N1; ++i)
    m1[i] = i + 5000u;

  alp_compressed_vector<double> cv0;
  cv0.exponent     = 0;
  cv0.factor       = 0;
  cv0.frame_of_ref = 0;
  cv0.bit_width    = 11;
  cv0.row_count    = N0;
  cv0.mantissas    = m0;

  alp_compressed_vector<double> cv1;
  cv1.exponent     = 0;
  cv1.factor       = 0;
  cv1.frame_of_ref = 0;
  cv1.bit_width    = 13;
  cv1.row_count    = N1;
  cv1.mantissas    = m1;

  std::vector<alp_vector<double>> vecs;
  vecs.emplace_back(std::move(cv0));
  vecs.emplace_back(std::move(cv1));
  auto bytes = make_alp_block<double>(vecs);
  auto out   = decode_one_alp<double>(bytes, F64, N0 + N1);

  for (uint32_t i = 0; i < N0; ++i)
    REQUIRE(out[i] == static_cast<double>(m0[i]));
  for (uint32_t i = 0; i < N1; ++i)
    REQUIRE(out[N0 + i] == static_cast<double>(m1[i]));
}

//===----------------------------------------------------------------------===//
// ALPRD happy-path tests.
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_decode_table ALPRD - canonical layout, double", "[scan][decode][alprd]")
{
  // Double bit pattern split into 16 left bits + 48 right bits.
  // Two left-part dictionary entries: 0xABCD and 0x1234.
  uint8_t const right_bw     = 48;
  uint8_t const left_bw      = 1;  // dict_size = 2 → log2 = 1
  std::vector<uint16_t> dict = {0xABCD, 0x1234};

  uint32_t const N = 80;
  std::vector<uint16_t> left_idx(N);
  std::vector<uint64_t> right_parts(N);
  for (uint32_t i = 0; i < N; ++i) {
    left_idx[i]    = (i & 1u);  // alternate dict entries
    right_parts[i] = (uint64_t{i} * 0x1234567ULL) & ((uint64_t{1} << right_bw) - 1u);
  }

  alprd_compressed_vector<double> cv;
  cv.row_count    = N;
  cv.left_indices = left_idx;
  cv.right_parts  = right_parts;

  std::vector<alprd_vector<double>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes = make_alprd_block<double>(right_bw, left_bw, dict, vecs);
  auto out   = decode_one_alprd<double>(bytes, F64, N);

  for (uint32_t i = 0; i < N; ++i) {
    uint64_t expected_bits =
      (static_cast<uint64_t>(dict[left_idx[i]]) << right_bw) | right_parts[i];
    double expected;
    std::memcpy(&expected, &expected_bits, sizeof(double));
    REQUIRE(out[i] == expected);
  }
}

TEST_CASE("gpu_decode_table ALPRD - exception patches left-part bits, double",
          "[scan][decode][alprd]")
{
  uint8_t const right_bw     = 48;
  uint8_t const left_bw      = 2;
  std::vector<uint16_t> dict = {0x1111, 0x2222, 0x3333, 0x4444};

  uint32_t const N = 64;
  std::vector<uint16_t> left_idx(N, 0);
  std::vector<uint64_t> right_parts(N);
  for (uint32_t i = 0; i < N; ++i) {
    right_parts[i] = i * 0x9ULL;
  }

  alprd_compressed_vector<double> cv;
  cv.row_count    = N;
  cv.left_indices = left_idx;
  cv.right_parts  = right_parts;
  cv.exception_left_values.push_back(0xDEAD);  // not in dict
  cv.exception_positions.push_back(7);

  std::vector<alprd_vector<double>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes = make_alprd_block<double>(right_bw, left_bw, dict, vecs);
  auto out   = decode_one_alprd<double>(bytes, F64, N);

  for (uint32_t i = 0; i < N; ++i) {
    uint64_t left_part     = (i == 7) ? 0xDEADULL : static_cast<uint64_t>(dict[0]);
    uint64_t expected_bits = (left_part << right_bw) | right_parts[i];
    double expected;
    std::memcpy(&expected, &expected_bits, sizeof(double));
    REQUIRE(out[i] == expected);
  }
}

TEST_CASE("gpu_decode_table ALPRD - uncompressed-sentinel vector float", "[scan][decode][alprd]")
{
  uint32_t const N = 30;
  std::vector<float> raw(N);
  for (uint32_t i = 0; i < N; ++i)
    raw[i] = std::sqrt(static_cast<float>(i + 1));

  alprd_uncompressed_vector<float> uv;
  uv.row_count = N;
  uv.values    = raw;

  std::vector<alprd_vector<float>> vecs;
  vecs.emplace_back(std::move(uv));
  // right_bw / left_bw / dict in the header are still present even on the
  // uncompressed-sentinel path — the kernel skips them when it reads the
  // 0xFFFF marker. Use any valid values.
  auto bytes = make_alprd_block<float>(/*right_bw=*/24, /*left_bw=*/0, /*dict=*/{0u}, vecs);
  auto out   = decode_one_alprd<float>(bytes, F32, N);

  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == raw[i]);
}

//===----------------------------------------------------------------------===//
// Defensive zero-fill guards — direct kernel call with 0xCC canary buffer.
//===----------------------------------------------------------------------===//

namespace {

template <typename T>
std::vector<T> decode_alp_invalid_with_canary(rmm::cuda_stream& stream,
                                              rmm::mr::cuda_async_memory_resource& mr,
                                              gpu_codec_run const& run,
                                              cudf::data_type type,
                                              uint32_t total_rows)
{
  std::vector<uint8_t> canary(size_t{total_rows} * sizeof(T), 0xCC);
  rmm::device_buffer d_out(canary.data(), canary.size(), stream.view());
  decode_alp_data(run, static_cast<uint8_t*>(d_out.data()), type, sizeof(T), stream.view(), mr);
  return download<T>(d_out.data(), total_rows, stream.value());
}

template <typename T>
std::vector<T> decode_alprd_invalid_with_canary(rmm::cuda_stream& stream,
                                                rmm::mr::cuda_async_memory_resource& mr,
                                                gpu_codec_run const& run,
                                                cudf::data_type type,
                                                uint32_t total_rows)
{
  std::vector<uint8_t> canary(size_t{total_rows} * sizeof(T), 0xCC);
  rmm::device_buffer d_out(canary.data(), canary.size(), stream.view());
  decode_alprd_data(run, static_cast<uint8_t*>(d_out.data()), type, sizeof(T), stream.view(), mr);
  return download<T>(d_out.data(), total_rows, stream.value());
}

}  // namespace

TEST_CASE("gpu_decode_table ALP - metadata_end past segment zero-fills",
          "[scan][decode][alp][defensive]")
{
  // Garbage segment with metadata_end = 1 GiB.
  std::vector<uint8_t> bytes(64u, 0u);
  uint32_t bogus = 1u << 30;
  std::memcpy(bytes.data(), &bogus, sizeof(bogus));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 32;
  gpu_codec_run run{CompressionType::COMPRESSION_ALP,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_alp_invalid_with_canary<double>(stream, mr, run, F64, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0.0);
}

TEST_CASE("gpu_decode_table ALP - vector pointer below segment header zero-fills",
          "[scan][decode][alp][defensive]")
{
  // Build a valid-looking trailer where the vector pointer points back at
  // metadata_end (out of range — pointer must be < metadata_end).
  std::vector<uint8_t> bytes(32u, 0u);
  uint32_t metadata_end = static_cast<uint32_t>(bytes.size());
  std::memcpy(bytes.data(), &metadata_end, sizeof(metadata_end));
  uint32_t bogus_ptr = metadata_end;  // invalid: pointer ≥ metadata_end
  std::memcpy(bytes.data() + metadata_end - 4, &bogus_ptr, sizeof(bogus_ptr));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const total_rows = 16;
  gpu_codec_run run{CompressionType::COMPRESSION_ALP,
                    {gpu_segment_desc{static_cast<uint8_t const*>(d_seg.data()),
                                      static_cast<uint32_t>(d_seg.size()),
                                      0,
                                      total_rows}}};
  auto out = decode_alp_invalid_with_canary<float>(stream, mr, run, F32, total_rows);
  for (uint32_t i = 0; i < total_rows; ++i)
    REQUIRE(out[i] == 0.0F);
}

TEST_CASE("gpu_decode_table ALP - bit_width > sizeof(T)*8 zero-fills",
          "[scan][decode][alp][defensive]")
{
  // Build a legitimate-looking ALP block but corrupt bit_width to 65 — the
  // kernel must refuse and zero-fill rather than reading past the buffer.
  uint32_t const N = 32;
  std::vector<uint64_t> m(N, 0u);
  alp_compressed_vector<double> cv;
  cv.exponent     = 0;
  cv.factor       = 0;
  cv.frame_of_ref = 0;
  cv.bit_width    = 11;  // valid
  cv.row_count    = N;
  cv.mantissas    = m;
  std::vector<alp_vector<double>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes = make_alp_block<double>(vecs);

  // Find the per-vector bit_width byte (offset 12 within the vector header,
  // and the vector header starts at byte 4 in this single-vector block).
  bytes[4 + 12] = 65;  // > 64 → invalid for double

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_codec_run run{
    CompressionType::COMPRESSION_ALP,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_alp_invalid_with_canary<double>(stream, mr, run, F64, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0.0);
}

TEST_CASE("gpu_decode_table ALP - exception payload past metadata_end zero-fills",
          "[scan][decode][alp][defensive]")
{
  // Construct a vector that claims 1000 exceptions in a 32-row block —
  // payload trivially overshoots metadata_end. Kernel must zero-fill.
  uint32_t const N = 32;
  std::vector<uint64_t> m(N, 7u);
  alp_compressed_vector<double> cv;
  cv.exponent     = 0;
  cv.factor       = 0;
  cv.frame_of_ref = 0;
  cv.bit_width    = 4;
  cv.row_count    = N;
  cv.mantissas    = m;
  std::vector<alp_vector<double>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes = make_alp_block<double>(vecs);

  // exceptions_count is at offset 4+2 (vector starts at 4, exc_count at +2).
  uint16_t huge = 1000;
  std::memcpy(bytes.data() + 4 + 2, &huge, sizeof(huge));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_codec_run run{
    CompressionType::COMPRESSION_ALP,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_alp_invalid_with_canary<double>(stream, mr, run, F64, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0.0);
}

TEST_CASE("gpu_decode_table ALPRD - dict_size > MAX_DICTIONARY_SIZE zero-fills",
          "[scan][decode][alprd][defensive]")
{
  // Hand-craft a minimum-shape ALPRD segment, then poison dict_size byte to
  // 16 (> MAX_DICTIONARY_SIZE=8). Kernel must refuse and zero-fill.
  std::vector<uint16_t> dict = {0x1111, 0x2222};  // legitimate
  alprd_uncompressed_vector<double> uv;
  uv.row_count = 8;
  uv.values    = std::vector<double>(8, 1.5);
  std::vector<alprd_vector<double>> vecs;
  vecs.emplace_back(std::move(uv));
  auto bytes = make_alprd_block<double>(/*right_bw=*/48, /*left_bw=*/1, dict, vecs);
  bytes[6]   = 16;  // poison dict_size

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const N = 8;
  gpu_codec_run run{
    CompressionType::COMPRESSION_ALPRD,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_alprd_invalid_with_canary<double>(stream, mr, run, F64, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0.0);
}

TEST_CASE("gpu_decode_table ALP - factor > 18 zero-fills", "[scan][decode][alp][defensive]")
{
  // Build a valid ALP block then poison the factor byte (offset 1 within
  // the vector header) to 19 — past `d_alp_fact[19]`.
  uint32_t const N = 32;
  std::vector<uint64_t> m(N, 0u);
  alp_compressed_vector<double> cv;
  cv.exponent     = 0;
  cv.factor       = 0;
  cv.frame_of_ref = 0;
  cv.bit_width    = 4;
  cv.row_count    = N;
  cv.mantissas    = m;
  std::vector<alp_vector<double>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes   = make_alp_block<double>(vecs);
  bytes[4 + 1] = 19;  // poison factor

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_codec_run run{
    CompressionType::COMPRESSION_ALP,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_alp_invalid_with_canary<double>(stream, mr, run, F64, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0.0);
}

TEST_CASE("gpu_decode_table ALP - exponent > MAX_EXPONENT zero-fills",
          "[scan][decode][alp][defensive]")
{
  // f32 MAX_EXPONENT = 10. Set exponent = 11 — would overshoot d_alp_frac_f32[11].
  uint32_t const N = 32;
  std::vector<uint64_t> m(N, 0u);
  alp_compressed_vector<float> cv;
  cv.exponent     = 0;
  cv.factor       = 0;
  cv.frame_of_ref = 0;
  cv.bit_width    = 4;
  cv.row_count    = N;
  cv.mantissas    = m;
  std::vector<alp_vector<float>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes   = make_alp_block<float>(vecs);
  bytes[4 + 0] = 11;  // poison exponent for float (max is 10)

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_codec_run run{
    CompressionType::COMPRESSION_ALP,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_alp_invalid_with_canary<float>(stream, mr, run, F32, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0.0F);
}

TEST_CASE("gpu_decode_table ALP - vector pointer way past metadata_end zero-fills",
          "[scan][decode][alp][defensive]")
{
  // Build a valid ALP block then poison the trailer pointer to land far
  // past metadata_end. Tests the strict `data_off >= metadata_end` bound
  // (existing `pointer == metadata_end` test only hits the boundary).
  uint32_t const N = 16;
  std::vector<uint64_t> m(N, 0u);
  alp_compressed_vector<double> cv;
  cv.exponent     = 0;
  cv.factor       = 0;
  cv.frame_of_ref = 0;
  cv.bit_width    = 4;
  cv.row_count    = N;
  cv.mantissas    = m;
  std::vector<alp_vector<double>> vecs;
  vecs.emplace_back(std::move(cv));
  auto bytes = make_alp_block<double>(vecs);

  uint32_t metadata_end;
  std::memcpy(&metadata_end, bytes.data(), sizeof(metadata_end));
  uint32_t bogus = metadata_end + 1024u;
  std::memcpy(bytes.data() + metadata_end - 4, &bogus, sizeof(bogus));

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  gpu_codec_run run{
    CompressionType::COMPRESSION_ALP,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_alp_invalid_with_canary<double>(stream, mr, run, F64, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0.0);
}

TEST_CASE("gpu_decode_table ALPRD - right_bw > sizeof(T)*8 zero-fills",
          "[scan][decode][alprd][defensive]")
{
  // Poison right_bw to 65 for a double segment — exceeds storage width.
  std::vector<uint16_t> dict = {0u};
  alprd_uncompressed_vector<double> uv;
  uv.row_count = 4;
  uv.values    = std::vector<double>(4, 2.5);
  std::vector<alprd_vector<double>> vecs;
  vecs.emplace_back(std::move(uv));
  auto bytes = make_alprd_block<double>(/*right_bw=*/64, /*left_bw=*/0, dict, vecs);
  bytes[4]   = 65;  // poison right_bw

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());

  uint32_t const N = 4;
  gpu_codec_run run{
    CompressionType::COMPRESSION_ALPRD,
    {gpu_segment_desc{
      static_cast<uint8_t const*>(d_seg.data()), static_cast<uint32_t>(d_seg.size()), 0, N}}};
  auto out = decode_alprd_invalid_with_canary<double>(stream, mr, run, F64, N);
  for (uint32_t i = 0; i < N; ++i)
    REQUIRE(out[i] == 0.0);
}

//===----------------------------------------------------------------------===//
// Bench-scale verify TEST_CASEs — one per microbench shape in
// bench_decode_codecs.cpp, asserting every output row against a host model.
// Bench discards its output, so silent-miscompile coverage at scale lives here.
//===----------------------------------------------------------------------===//

TEST_CASE("gpu_decode_table ALP bench-scale - f64 bw=11 verify", "[scan][decode][alp][verify]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS   = 32u * 1024u;
  constexpr uint32_t ROWS     = N_VECS * 1024u;
  constexpr uint8_t BIT_WIDTH = 11u;
  uint64_t const mask         = (uint64_t{1} << BIT_WIDTH) - 1u;

  auto seg_bytes = synth_alp_segment<double>(N_VECS, BIT_WIDTH);
  auto d_seg     = upload(seg_bytes, stream.view());
  auto col =
    one_codec_column(F64, ROWS, CompressionType::COMPRESSION_ALP, {segment(d_seg, 0, ROWS)});

  verify_decoded_column<double>(stream.view(), mr, col, [](uint32_t row) {
    uint32_t const v = row >> 10;
    uint32_t const i = row & 1023u;
    return static_cast<double>(static_cast<int64_t>((uint64_t{i} ^ uint64_t{v}) & mask));
  });
}

TEST_CASE("gpu_decode_table ALP bench-scale - f32 bw=20 verify", "[scan][decode][alp][verify]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS   = 32u * 1024u;
  constexpr uint32_t ROWS     = N_VECS * 1024u;
  constexpr uint8_t BIT_WIDTH = 20u;
  uint64_t const mask         = (uint64_t{1} << BIT_WIDTH) - 1u;

  auto seg_bytes = synth_alp_segment<float>(N_VECS, BIT_WIDTH);
  auto d_seg     = upload(seg_bytes, stream.view());
  auto col =
    one_codec_column(F32, ROWS, CompressionType::COMPRESSION_ALP, {segment(d_seg, 0, ROWS)});

  verify_decoded_column<float>(stream.view(), mr, col, [](uint32_t row) {
    uint32_t const v = row >> 10;
    uint32_t const i = row & 1023u;
    return static_cast<float>(static_cast<int64_t>((uint64_t{i} ^ uint64_t{v}) & mask));
  });
}

TEST_CASE("gpu_decode_table ALPRD bench-scale - f64 r48/l3 verify", "[scan][decode][alprd][verify]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS  = 32u * 1024u;
  constexpr uint32_t ROWS    = N_VECS * 1024u;
  constexpr uint8_t RIGHT_BW = 48u;
  constexpr uint8_t LEFT_BW  = 3u;
  std::vector<uint16_t> dict = {0xAAAA, 0x5555, 0x1234, 0xABCD, 0xDEAD, 0xBEEF, 0xCAFE, 0xF00D};

  auto seg_bytes = synth_alprd_segment<double>(N_VECS, RIGHT_BW, LEFT_BW, dict);
  auto d_seg     = upload(seg_bytes, stream.view());
  auto col =
    one_codec_column(F64, ROWS, CompressionType::COMPRESSION_ALPRD, {segment(d_seg, 0, ROWS)});

  uint64_t const r_mask = (uint64_t{1} << RIGHT_BW) - 1u;
  uint16_t const l_mask = static_cast<uint16_t>((uint16_t{1} << LEFT_BW) - 1u);
  verify_decoded_column<double>(stream.view(), mr, col, [&dict](uint32_t row) {
    uint32_t const v          = row >> 10;
    uint32_t const i          = row & 1023u;
    uint32_t const left_idx   = (i ^ v) & l_mask;
    uint64_t const right_part = (uint64_t{i} * 0x9E3779B97F4A7C15ULL) & r_mask;
    uint64_t const bits       = (uint64_t{dict[left_idx]} << RIGHT_BW) | right_part;
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
  });
}

// Power-of-2 / byte-aligned widths. The bw=11/20/r48-l3 cases above stress the
// 64-bit-straddle path; these mirror the byte-aligned shapes that ALP-encoded
// TPC-H integer-derived columns hit far more often than odd widths.

TEST_CASE("gpu_decode_table ALP bench-scale - f64 bw=8 verify", "[scan][decode][alp][verify]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS   = 32u * 1024u;
  constexpr uint32_t ROWS     = N_VECS * 1024u;
  constexpr uint8_t BIT_WIDTH = 8u;
  uint64_t const mask         = (uint64_t{1} << BIT_WIDTH) - 1u;

  auto seg_bytes = synth_alp_segment<double>(N_VECS, BIT_WIDTH);
  auto d_seg     = upload(seg_bytes, stream.view());
  auto col =
    one_codec_column(F64, ROWS, CompressionType::COMPRESSION_ALP, {segment(d_seg, 0, ROWS)});

  verify_decoded_column<double>(stream.view(), mr, col, [](uint32_t row) {
    uint32_t const v = row >> 10;
    uint32_t const i = row & 1023u;
    return static_cast<double>(static_cast<int64_t>((uint64_t{i} ^ uint64_t{v}) & mask));
  });
}

TEST_CASE("gpu_decode_table ALP bench-scale - f64 bw=16 verify", "[scan][decode][alp][verify]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS   = 32u * 1024u;
  constexpr uint32_t ROWS     = N_VECS * 1024u;
  constexpr uint8_t BIT_WIDTH = 16u;
  uint64_t const mask         = (uint64_t{1} << BIT_WIDTH) - 1u;

  auto seg_bytes = synth_alp_segment<double>(N_VECS, BIT_WIDTH);
  auto d_seg     = upload(seg_bytes, stream.view());
  auto col =
    one_codec_column(F64, ROWS, CompressionType::COMPRESSION_ALP, {segment(d_seg, 0, ROWS)});

  verify_decoded_column<double>(stream.view(), mr, col, [](uint32_t row) {
    uint32_t const v = row >> 10;
    uint32_t const i = row & 1023u;
    return static_cast<double>(static_cast<int64_t>((uint64_t{i} ^ uint64_t{v}) & mask));
  });
}

TEST_CASE("gpu_decode_table ALP bench-scale - f32 bw=32 verify", "[scan][decode][alp][verify]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS   = 32u * 1024u;
  constexpr uint32_t ROWS     = N_VECS * 1024u;
  constexpr uint8_t BIT_WIDTH = 32u;

  auto seg_bytes = synth_alp_segment<float>(N_VECS, BIT_WIDTH);
  auto d_seg     = upload(seg_bytes, stream.view());
  auto col =
    one_codec_column(F32, ROWS, CompressionType::COMPRESSION_ALP, {segment(d_seg, 0, ROWS)});

  // bw=32 covers the full f32 mantissa width; no mask needed.
  verify_decoded_column<float>(stream.view(), mr, col, [](uint32_t row) {
    uint32_t const v = row >> 10;
    uint32_t const i = row & 1023u;
    return static_cast<float>(static_cast<int64_t>(uint64_t{i} ^ uint64_t{v}));
  });
}

TEST_CASE("gpu_decode_table ALPRD bench-scale - f64 r56/l2 verify", "[scan][decode][alprd][verify]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS  = 32u * 1024u;
  constexpr uint32_t ROWS    = N_VECS * 1024u;
  constexpr uint8_t RIGHT_BW = 56u;
  constexpr uint8_t LEFT_BW  = 2u;
  std::vector<uint16_t> dict = {0xAAAA, 0x5555, 0x1234, 0xABCD};

  auto seg_bytes = synth_alprd_segment<double>(N_VECS, RIGHT_BW, LEFT_BW, dict);
  auto d_seg     = upload(seg_bytes, stream.view());
  auto col =
    one_codec_column(F64, ROWS, CompressionType::COMPRESSION_ALPRD, {segment(d_seg, 0, ROWS)});

  uint64_t const r_mask = (uint64_t{1} << RIGHT_BW) - 1u;
  uint16_t const l_mask = static_cast<uint16_t>((uint16_t{1} << LEFT_BW) - 1u);
  verify_decoded_column<double>(stream.view(), mr, col, [&dict](uint32_t row) {
    uint32_t const v          = row >> 10;
    uint32_t const i          = row & 1023u;
    uint32_t const left_idx   = (i ^ v) & l_mask;
    uint64_t const right_part = (uint64_t{i} * 0x9E3779B97F4A7C15ULL) & r_mask;
    uint64_t const bits       = (uint64_t{dict[left_idx]} << RIGHT_BW) | right_part;
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
  });
}
