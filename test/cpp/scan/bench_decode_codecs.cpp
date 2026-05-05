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
#include "scan/rle_synth.hpp"

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

//===----------------------------------------------------------------------===//
// RLE benches.
//
// Each bench encodes one segment per row group (DuckDB row-group max 122880
// rows) and tiles many segments to reach the workload size. The kernel
// expands one chunk of 2048 rows per CTA, so a 122880-row segment splits
// into 60 CTAs — well above the launch-overhead floor.
//
// Three shapes:
//   long_runs   — few entries per segment; near-broadcast pattern, exercise
//                 the shmem cumsum cache + value broadcast.
//   medium_runs — typical sorted-low-cardinality column.
//   short_runs  — one row per entry, exercises the gmem-cumsum path when
//                 entry_count exceeds the shmem cap.
//===----------------------------------------------------------------------===//

namespace {

constexpr uint32_t RLE_BENCH_SEG_ROWS = 122880;  // DuckDB row-group max

}  // namespace

TEST_CASE("bench RLE int64 long_runs (16 entries/seg) 122M rows",
          "[!benchmark][scan][decode]")
{
  using ::sirius::test::decode::rle::make_uniform_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  // 16 entries × 7680 run_len = 122880 rows per segment. Run length sits
  // well above warp width so the value-broadcast path dominates.
  constexpr uint32_t N_RUNS  = 16;
  constexpr uint16_t RUN_LEN = static_cast<uint16_t>(RLE_BENCH_SEG_ROWS / N_RUNS);
  constexpr uint32_t SEG_ROWS = N_RUNS * RUN_LEN;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  auto seg_bytes              = make_uniform_runs<int64_t>(N_RUNS, RUN_LEN);

  std::vector<rmm::device_buffer> bufs;
  std::vector<gpu_segment_desc> segs;
  bufs.reserve(N_SEGS);
  segs.reserve(N_SEGS);
  for (uint32_t i = 0; i < N_SEGS; ++i) {
    bufs.emplace_back(seg_bytes.data(), seg_bytes.size(), stream.view());
    segs.push_back(segment(bufs.back(), i * SEG_ROWS, SEG_ROWS));
  }
  auto col = one_codec_column(cudf::data_type{cudf::type_id::INT64},
                              N_SEGS * SEG_ROWS,
                              CompressionType::COMPRESSION_RLE,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int64_t));
  std::printf("[bench] RLE          int64 long_runs    %u rows: %.6fs  write=%.1f GiB/s\n",
              N_SEGS * SEG_ROWS,
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench RLE int64 medium_runs (1024 entries/seg) 122M rows",
          "[!benchmark][scan][decode]")
{
  using ::sirius::test::decode::rle::make_uniform_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  // 1024 entries × 120 run_len = 122880 rows per segment. cumsum (4 KiB)
  // fits easily in shmem; binary search is log2(1024) = 10 levels.
  constexpr uint32_t N_RUNS  = 1024;
  constexpr uint16_t RUN_LEN = static_cast<uint16_t>(RLE_BENCH_SEG_ROWS / N_RUNS);
  constexpr uint32_t SEG_ROWS = N_RUNS * RUN_LEN;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  auto seg_bytes              = make_uniform_runs<int64_t>(N_RUNS, RUN_LEN);

  std::vector<rmm::device_buffer> bufs;
  std::vector<gpu_segment_desc> segs;
  bufs.reserve(N_SEGS);
  segs.reserve(N_SEGS);
  for (uint32_t i = 0; i < N_SEGS; ++i) {
    bufs.emplace_back(seg_bytes.data(), seg_bytes.size(), stream.view());
    segs.push_back(segment(bufs.back(), i * SEG_ROWS, SEG_ROWS));
  }
  auto col = one_codec_column(cudf::data_type{cudf::type_id::INT64},
                              N_SEGS * SEG_ROWS,
                              CompressionType::COMPRESSION_RLE,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int64_t));
  std::printf("[bench] RLE          int64 medium_runs  %u rows: %.6fs  write=%.1f GiB/s\n",
              N_SEGS * SEG_ROWS,
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench RLE int32 short_runs (8192 entries/seg gmem) 65M rows",
          "[!benchmark][scan][decode]")
{
  using ::sirius::test::decode::rle::make_uniform_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  // 8192 entries > RLE_SMEM_MAX_ENTRIES (4096) → gmem-cumsum binary-search
  // path. 15 run_len keeps run lengths small but >0; segment row count =
  // 8192 × 15 = 122880 (matches DuckDB row-group max). Segment count cut to
  // 1/2 of the int64 benches because the gmem path is intentionally slower
  // and we want bench wall time to stay reasonable.
  constexpr uint32_t N_RUNS  = 8192;
  constexpr uint16_t RUN_LEN = 15;  // 8192*15 = 122880
  constexpr uint32_t SEG_ROWS = N_RUNS * RUN_LEN;
  constexpr uint32_t N_SEGS   = (64u << 20) / SEG_ROWS;
  auto seg_bytes              = make_uniform_runs<int32_t>(N_RUNS, RUN_LEN);

  std::vector<rmm::device_buffer> bufs;
  std::vector<gpu_segment_desc> segs;
  bufs.reserve(N_SEGS);
  segs.reserve(N_SEGS);
  for (uint32_t i = 0; i < N_SEGS; ++i) {
    bufs.emplace_back(seg_bytes.data(), seg_bytes.size(), stream.view());
    segs.push_back(segment(bufs.back(), i * SEG_ROWS, SEG_ROWS));
  }
  auto col = one_codec_column(cudf::data_type{cudf::type_id::INT32},
                              N_SEGS * SEG_ROWS,
                              CompressionType::COMPRESSION_RLE,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int32_t));
  std::printf("[bench] RLE          int32 short_runs   %u rows: %.6fs  write=%.1f GiB/s\n",
              N_SEGS * SEG_ROWS,
              sec,
              bytes_w / sec / GIB);
}
