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
// Codec-dispatch microbench. Tagged `[!benchmark]` so default unittest runs
// skip it. Numbers paste into test/data/decode_baselines/<arch>.json.
//===----------------------------------------------------------------------===//

#include "scan/decode_test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

using duckdb::CompressionType;
using sirius::cuda::scan::gpu_column_decode_input;
using sirius::cuda::scan::gpu_decode_table;
using sirius::cuda::scan::gpu_segment_desc;
using sirius::test::decode::one_codec_column;
using sirius::test::decode::segment;
using sirius::test::decode::upload;

namespace {

double bench_seconds(rmm::cuda_stream& stream,
                     std::vector<gpu_column_decode_input> const& cols,
                     rmm::mr::cuda_async_memory_resource& mr,
                     int iters  = 10,
                     int warmup = 3)
{
  for (int i = 0; i < warmup; ++i)
    (void)gpu_decode_table(cols, stream.view(), mr);
  cudaStreamSynchronize(stream.value());

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s, stream.value());
  for (int i = 0; i < iters; ++i)
    (void)gpu_decode_table(cols, stream.view(), mr);
  cudaEventRecord(e, stream.value());
  cudaEventSynchronize(e);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  return (ms / 1000.0) / iters;
}

constexpr double GIB = double(1ULL << 30);

}  // namespace

TEST_CASE("bench UNCOMPRESSED int64 single 32MiB", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr size_t ROWS = 4 << 20;
  std::vector<int64_t> values(ROWS, 0);
  auto d   = upload(values, stream.view());
  auto col = one_codec_column(cudf::data_type{cudf::type_id::INT64},
                              ROWS,
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              {segment(d, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(ROWS * sizeof(int64_t));
  std::printf(
    "[bench] UNCOMPRESSED int64 single 32MiB: %.6fs  write=%.1f GiB/s  rd+wr=%.1f GiB/s\n",
    sec,
    bytes_w / sec / GIB,
    2.0 * bytes_w / sec / GIB);
}

TEST_CASE("bench UNCOMPRESSED int64 multi 16x2MiB", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = 256u << 10;  // 256K rows = 2 MiB per segment
  constexpr uint32_t N_SEGS   = 16;
  std::vector<int64_t> values(SEG_ROWS, 0);
  std::vector<rmm::device_buffer> bufs;
  std::vector<gpu_segment_desc> segs;
  bufs.reserve(N_SEGS);
  segs.reserve(N_SEGS);
  for (uint32_t i = 0; i < N_SEGS; ++i) {
    bufs.push_back(upload(values, stream.view()));
    segs.push_back(segment(bufs.back(), i * SEG_ROWS, SEG_ROWS));
  }
  auto col = one_codec_column(cudf::data_type{cudf::type_id::INT64},
                              N_SEGS * SEG_ROWS,
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int64_t));
  std::printf(
    "[bench] UNCOMPRESSED int64 16x2MiB:      %.6fs  write=%.1f GiB/s  rd+wr=%.1f GiB/s\n",
    sec,
    bytes_w / sec / GIB,
    2.0 * bytes_w / sec / GIB);
}

TEST_CASE("bench UNCOMPRESSED int64 multi 1024x32KiB", "[!benchmark][scan][decode]")
{
  // Many small segments — the realistic DuckDB shape (one column = many
  // per-row-group segments).
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = 4096;  // 4K rows = 32 KiB per int64 segment
  constexpr uint32_t N_SEGS   = 1024;
  std::vector<int64_t> values(SEG_ROWS, 0);
  std::vector<rmm::device_buffer> bufs;
  std::vector<gpu_segment_desc> segs;
  bufs.reserve(N_SEGS);
  segs.reserve(N_SEGS);
  for (uint32_t i = 0; i < N_SEGS; ++i) {
    bufs.push_back(upload(values, stream.view()));
    segs.push_back(segment(bufs.back(), i * SEG_ROWS, SEG_ROWS));
  }
  auto col = one_codec_column(cudf::data_type{cudf::type_id::INT64},
                              N_SEGS * SEG_ROWS,
                              CompressionType::COMPRESSION_UNCOMPRESSED,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int64_t));
  std::printf(
    "[bench] UNCOMPRESSED int64 1024x32KiB:   %.6fs  write=%.1f GiB/s  rd+wr=%.1f GiB/s\n",
    sec,
    bytes_w / sec / GIB,
    2.0 * bytes_w / sec / GIB);
}

TEST_CASE("bench raw cudaMemcpyAsync 1024x32KiB (reference, no dispatcher)",
          "[!benchmark][scan][decode]")
{
  // Reference: same workload as the 1024x32KiB dispatcher bench, issued as
  // per-segment cudaMemcpyAsync without the batched kernel.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr size_t SEG_BYTES = 32u << 10;
  constexpr uint32_t N_SEGS  = 1024;
  std::vector<int64_t> values(SEG_BYTES / sizeof(int64_t), 0);
  std::vector<rmm::device_buffer> srcs;
  srcs.reserve(N_SEGS);
  for (uint32_t i = 0; i < N_SEGS; ++i)
    srcs.push_back(upload(values, stream.view()));
  rmm::device_buffer dst(N_SEGS * SEG_BYTES, stream.view());
  cudaStreamSynchronize(stream.value());

  // Warmup + timed loop.
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int w = 0; w < 3; ++w) {
    for (uint32_t i = 0; i < N_SEGS; ++i)
      cudaMemcpyAsync(static_cast<uint8_t*>(dst.data()) + size_t{i} * SEG_BYTES,
                      srcs[i].data(),
                      SEG_BYTES,
                      cudaMemcpyDeviceToDevice,
                      stream.value());
  }
  cudaStreamSynchronize(stream.value());
  cudaEventRecord(s, stream.value());
  for (int it = 0; it < 10; ++it) {
    for (uint32_t i = 0; i < N_SEGS; ++i)
      cudaMemcpyAsync(static_cast<uint8_t*>(dst.data()) + size_t{i} * SEG_BYTES,
                      srcs[i].data(),
                      SEG_BYTES,
                      cudaMemcpyDeviceToDevice,
                      stream.value());
  }
  cudaEventRecord(e, stream.value());
  cudaEventSynchronize(e);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  double sec     = (ms / 1000.0) / 10;
  double bytes_w = double(size_t{N_SEGS} * SEG_BYTES);
  std::printf(
    "[bench] raw cudaMemcpyAsync 1024x32KiB:  %.6fs  write=%.1f GiB/s  rd+wr=%.1f GiB/s\n",
    sec,
    bytes_w / sec / GIB,
    2.0 * bytes_w / sec / GIB);
}

TEST_CASE("bench CONSTANT int64 32MiB", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr size_t ROWS  = 4 << 20;
  std::vector<int64_t> v = {12345};
  auto d                 = upload(v, stream.view());
  auto col               = one_codec_column(cudf::data_type{cudf::type_id::INT64},
                              ROWS,
                              CompressionType::COMPRESSION_CONSTANT,
                                            {segment(d, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(ROWS * sizeof(int64_t));
  std::printf(
    "[bench] CONSTANT     int64 32MiB:        %.6fs  write=%.1f GiB/s\n", sec, bytes_w / sec / GIB);
}
