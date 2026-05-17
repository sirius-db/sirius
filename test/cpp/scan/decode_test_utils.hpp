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

#pragma once

//===----------------------------------------------------------------------===//
// Helpers for kernel-level decode unit tests. Synthetic byte buffers staged
// on device via rmm::device_buffer.
//===----------------------------------------------------------------------===//

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_native_decode.cuh>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <vector>

namespace sirius::test::decode {

template <typename T>
inline rmm::device_buffer upload(std::vector<T> const& host, rmm::cuda_stream_view stream)
{
  return rmm::device_buffer(host.data(), host.size() * sizeof(T), stream);
}

template <typename T>
inline std::vector<T> download(void const* d_ptr, size_t count, cudaStream_t stream)
{
  std::vector<T> out(count);
  ::cudaMemcpyAsync(out.data(), d_ptr, count * sizeof(T), ::cudaMemcpyDeviceToHost, stream);
  ::cudaStreamSynchronize(stream);
  return out;
}

inline std::vector<uint8_t> pack_validity(std::vector<bool> const& bits)
{
  std::vector<uint8_t> out((bits.size() + 7) / 8, 0xFFu);
  for (size_t i = 0; i < bits.size(); ++i) {
    if (!bits[i]) out[i / 8] &= ~static_cast<uint8_t>(1u << (i % 8));
  }
  return out;
}

inline ::sirius::cuda::scan::gpu_segment_desc segment(rmm::device_buffer const& buf,
                                                      uint32_t row_offset,
                                                      uint32_t row_count)
{
  return ::sirius::cuda::scan::gpu_segment_desc{static_cast<uint8_t const*>(buf.data()),
                                                static_cast<uint32_t>(buf.size()),
                                                row_offset,
                                                row_count};
}

/// Builds a `gpu_column_decode_input` with one codec_run on `data`.
inline ::sirius::cuda::scan::gpu_column_decode_input one_codec_column(
  cudf::data_type type,
  uint32_t total_rows,
  duckdb::CompressionType codec,
  std::vector<::sirius::cuda::scan::gpu_segment_desc> segs,
  bool has_nulls = false)
{
  ::sirius::cuda::scan::gpu_column_decode_input col;
  col.out_type   = type;
  col.total_rows = total_rows;
  col.has_nulls  = has_nulls;
  col.data.push_back({codec, std::move(segs)});
  return col;
}

/// Test fixture used via `TEST_CASE_METHOD(decode_env, ...)`. Carries the
/// stream + memory resource that every decode test needs and exposes a
/// shorthand `decode(...)` that wires both into `gpu_decode_table`.
struct decode_env {
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  std::unique_ptr<cudf::table> decode(
    std::vector<::sirius::cuda::scan::gpu_column_decode_input> const& cols)
  {
    return ::sirius::cuda::scan::gpu_decode_table(cols, stream.view(), mr);
  }
};

/// Round-trip a single UNCOMPRESSED column: upload `values` as one segment,
/// decode, REQUIRE the output bytes match. Used by the per-type happy-path
/// tests; multi-segment / with-nulls cases stay inline because their shape
/// differs.
template <typename T>
inline void require_uncompressed_roundtrip(decode_env& env,
                                           cudf::data_type type,
                                           std::vector<T> const& values)
{
  auto d   = upload(values, env.stream.view());
  auto col = one_codec_column(type,
                              static_cast<uint32_t>(values.size()),
                              duckdb::CompressionType::COMPRESSION_UNCOMPRESSED,
                              {segment(d, 0, static_cast<uint32_t>(values.size()))});
  auto t   = env.decode({col});
  REQUIRE(t->num_rows() == static_cast<cudf::size_type>(values.size()));
  REQUIRE(t->get_column(0).null_count() == 0);
  REQUIRE(download<T>(t->get_column(0).view().template data<T>(),
                      values.size(),
                      env.stream.value()) == values);
}

/// Broadcast `value` across `rows` rows via a CONSTANT segment of one element,
/// decode, REQUIRE every output row equals `value`.
template <typename T>
inline void require_constant_broadcast(decode_env& env,
                                       cudf::data_type type,
                                       T value,
                                       uint32_t rows)
{
  std::vector<T> v = {value};
  auto d           = upload(v, env.stream.view());
  auto col         = one_codec_column(
    type, rows, duckdb::CompressionType::COMPRESSION_CONSTANT, {segment(d, 0, rows)});
  auto t   = env.decode({col});
  auto out = download<T>(t->get_column(0).view().template data<T>(), rows, env.stream.value());
  for (auto x : out)
    REQUIRE(x == value);
}

/// Decode `col` and assert each output row matches `expected_row(i)`.
/// Use this from any codec's bench-scale verify TEST_CASE — the bench
/// microbench tests in bench_decode_codecs.cpp discard their output, so
/// every codec PR should ship a matching verify test using this helper to
/// catch silent miscompiles at scale.
///
/// `expected_row` may be any callable taking `uint32_t row -> T`.
template <typename T, typename ExpectedRow>
inline void verify_decoded_column(rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr,
                                  ::sirius::cuda::scan::gpu_column_decode_input const& col,
                                  ExpectedRow expected_row)
{
  auto t = ::sirius::cuda::scan::gpu_decode_table({col}, stream, mr);
  auto out =
    download<T>(t->get_column(0).view().template data<T>(), col.total_rows, stream.value());
  for (uint32_t i = 0; i < col.total_rows; ++i) {
    T const want = expected_row(i);
    if (out[i] != want) {
      // Stream T directly so floating-point T renders without a lossy
      // long-long cast (`static_cast<long long>(3.14)` would print 3).
      FAIL("row=" << i << " expected=" << want << " got=" << out[i]);
    }
  }
}

}  // namespace sirius::test::decode
