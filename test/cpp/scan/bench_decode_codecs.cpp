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

#include "scan/alp_synth.hpp"
#include "scan/bitpacking_synth.hpp"
#include "scan/decode_test_utils.hpp"
#include "scan/rle_synth.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_bitpacking.cuh>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <variant>
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

namespace {

constexpr uint32_t RLE_BENCH_SEG_ROWS = 122880;  // DuckDB row-group max

}  // namespace

TEST_CASE("bench RLE int64 long_runs (16 entries/seg) 128Mi rows", "[!benchmark][scan][decode]")
{
  using ::sirius::test::decode::rle::make_uniform_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  // 16 runs of 7680 rows each — long-run / value-broadcast pattern.
  constexpr uint32_t N_RUNS   = 16;
  constexpr uint16_t RUN_LEN  = static_cast<uint16_t>(RLE_BENCH_SEG_ROWS / N_RUNS);
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

TEST_CASE("bench RLE int64 medium_runs (1024 entries/seg) 128Mi rows", "[!benchmark][scan][decode]")
{
  using ::sirius::test::decode::rle::make_uniform_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  // 1024 runs of 120 rows — cumsum (4 KiB) fits in shmem.
  constexpr uint32_t N_RUNS   = 1024;
  constexpr uint16_t RUN_LEN  = static_cast<uint16_t>(RLE_BENCH_SEG_ROWS / N_RUNS);
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

TEST_CASE("bench RLE int64 pareto_runs (skewed distribution) 128Mi rows",
          "[!benchmark][scan][decode]")
{
  // Realistic shape: Pareto-distributed run lengths. Many short runs +
  // a few long ones, like sorted low-cardinality columns in TPC-H.
  using ::sirius::test::decode::rle::make_pareto_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  // 122880 rows per segment with Pareto x_min=400 → mean run ~1200, ~100
  // entries per segment. Comfortably within the build cap and matches the
  // shape of TPC-H sorted-low-cardinality columns (l_returnflag-class).
  constexpr uint32_t SEG_ROWS = 122880;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;

  std::vector<rmm::device_buffer> bufs;
  std::vector<gpu_segment_desc> segs;
  bufs.reserve(N_SEGS);
  segs.reserve(N_SEGS);
  for (uint32_t i = 0; i < N_SEGS; ++i) {
    auto seg_bytes = make_pareto_runs<int64_t>(SEG_ROWS, /*seed=*/i + 1, /*x_min=*/400.0);
    bufs.emplace_back(seg_bytes.data(), seg_bytes.size(), stream.view());
    segs.push_back(segment(bufs.back(), i * SEG_ROWS, SEG_ROWS));
  }
  auto col = one_codec_column(cudf::data_type{cudf::type_id::INT64},
                              N_SEGS * SEG_ROWS,
                              CompressionType::COMPRESSION_RLE,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int64_t));
  std::printf("[bench] RLE          int64 pareto_runs   %u rows: %.6fs  write=%.1f GiB/s\n",
              N_SEGS * SEG_ROWS,
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench RLE int32 short_runs (4096 entries/seg) 65M rows", "[!benchmark][scan][decode]")
{
  using ::sirius::test::decode::rle::make_uniform_runs;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  // At the build kernel's max-entry cap; each run is 30 rows.
  constexpr uint32_t N_RUNS   = 4096;
  constexpr uint16_t RUN_LEN  = 30;  // 4096*30 = 122880
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

// BITPACKING benches.
//
// Each segment is BP_META_GROUP_SIZE rows so it dispatches as one CTA (the
// production shape — DuckDB writes ~one group per segment for bitpacked
// columns). The 128M-row workloads slice into ~64K segments; one batched
// kernel launch per column.
//===----------------------------------------------------------------------===//

using ::sirius::test::decode::bitpacking::make_constant_block;
using ::sirius::test::decode::bitpacking::make_delta_for_block;
using ::sirius::test::decode::bitpacking::make_for_block;

TEST_CASE("bench BITPACKING int64 FOR width=8 128M rows", "[!benchmark][scan][decode]")
{
  // Production shape: one segment per metadata group. Width=8 — 8x
  // compression vs UNCOMPRESSED, exercises the unpack hot path (bit_off
  // varies, no third-word reads).
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = ::sirius::cuda::scan::BP_META_GROUP_SIZE;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  std::vector<int64_t> deltas(SEG_ROWS);
  for (uint32_t i = 0; i < SEG_ROWS; ++i)
    deltas[i] = i & 0xFF;
  auto seg_bytes = make_for_block<int64_t>(/*frame=*/1000, /*width=*/8, deltas);

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
                              CompressionType::COMPRESSION_BITPACKING,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int64_t));
  std::printf("[bench] BITPACKING   int64 FOR w=8 128M rows: %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench BITPACKING int32 FOR width=12 128M rows", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = ::sirius::cuda::scan::BP_META_GROUP_SIZE;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  std::vector<int32_t> deltas(SEG_ROWS);
  for (uint32_t i = 0; i < SEG_ROWS; ++i)
    deltas[i] = (i * 7) & 0xFFF;
  auto seg_bytes = make_for_block<int32_t>(0, 12, deltas);

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
                              CompressionType::COMPRESSION_BITPACKING,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int32_t));
  std::printf("[bench] BITPACKING   int32 FOR w=12 128M rows: %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench BITPACKING int64 CONSTANT 128M rows", "[!benchmark][scan][decode]")
{
  // Vectorised int4 store path — should track UNCOMPRESSED CONSTANT (which
  // hits the GDDR fill ceiling). Confirms the L1-bypass + 16-byte store
  // path matches the type_dispatcher CONSTANT broadcast.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = ::sirius::cuda::scan::BP_META_GROUP_SIZE;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  auto seg_bytes              = make_constant_block<int64_t>(12345);

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
                              CompressionType::COMPRESSION_BITPACKING,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int64_t));
  std::printf("[bench] BITPACKING   int64 CONSTANT 128M rows: %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

//===----------------------------------------------------------------------===//
// Phase-2 perf coverage: natural-width FOR (the per-natural-width template
// optimisation target), narrow non-natural width, and DELTA_FOR (the
// BlockScan path). Uses the shared bitpacking_synth.hpp builders.
//===----------------------------------------------------------------------===//

TEST_CASE("bench BITPACKING int32 FOR width=16 128M rows", "[!benchmark][scan][decode]")
{
  // Natural width 16 — every value aligns on a 16-bit boundary; an unpack
  // that's branch-free on aligned reads should beat the runtime-shift path.
  // Use this as the per-natural-width-template before/after target.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = ::sirius::cuda::scan::BP_META_GROUP_SIZE;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  std::vector<int32_t> deltas(SEG_ROWS);
  for (uint32_t i = 0; i < SEG_ROWS; ++i)
    deltas[i] = (i * 13) & 0xFFFF;
  auto seg_bytes = make_for_block<int32_t>(/*frame=*/0, /*width=*/16, deltas);

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
                              CompressionType::COMPRESSION_BITPACKING,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int32_t));
  std::printf("[bench] BITPACKING   int32 FOR w=16 128M rows: %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench BITPACKING int32 FOR width=5 128M rows", "[!benchmark][scan][decode]")
{
  // Very narrow width — exercises crossing 32-bit boundaries on most reads.
  // Validates that any cp.async / vector-load optimisation doesn't regress
  // the cross-boundary unpack path.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = ::sirius::cuda::scan::BP_META_GROUP_SIZE;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  std::vector<int32_t> deltas(SEG_ROWS);
  for (uint32_t i = 0; i < SEG_ROWS; ++i)
    deltas[i] = i & 0x1F;
  auto seg_bytes = make_for_block<int32_t>(/*frame=*/0, /*width=*/5, deltas);

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
                              CompressionType::COMPRESSION_BITPACKING,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int32_t));
  std::printf("[bench] BITPACKING   int32 FOR w=5 128M rows:  %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench BITPACKING int32 DELTA_FOR width=8 128M rows", "[!benchmark][scan][decode]")
{
  // DELTA_FOR exercises the WarpScan path (per-warp inclusive scan + warp-
  // aggregate exchange) introduced for this codec. All deltas are tiny so
  // width=8 is sufficient.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = ::sirius::cuda::scan::BP_META_GROUP_SIZE;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  std::vector<int32_t> deltas(SEG_ROWS);
  for (uint32_t i = 0; i < SEG_ROWS; ++i)
    deltas[i] = (i & 0xFF);
  auto seg_bytes =
    make_delta_for_block<int32_t>(/*frame=*/100, /*delta_offset=*/0, /*width=*/8, deltas);

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
                              CompressionType::COMPRESSION_BITPACKING,
                              segs);

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{N_SEGS} * SEG_ROWS * sizeof(int32_t));
  std::printf("[bench] BITPACKING   int32 DELTA_FOR w=8 128M rows: %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

//===----------------------------------------------------------------------===//
// ALP / ALPRD benches. Tagged `[!benchmark]`; numbers paste into
// test/data/decode_baselines/<arch>.json under the alp / alprd keys.
//
// `bw_pinned` cases (most useful for self-relative phase-2 perf comparisons)
// fix bit_width per case so runs are reproducible across machines.
//===----------------------------------------------------------------------===//

using sirius::test::decode::alp::synth_alp_segment;
using sirius::test::decode::alp::synth_alprd_segment;

TEST_CASE("bench ALP double bw=11 32K vectors (32M rows)", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS = 32u * 1024u;
  constexpr uint32_t ROWS   = N_VECS * 1024u;

  auto seg_bytes = synth_alp_segment<double>(N_VECS, /*bit_width=*/11);
  auto d_seg     = sirius::test::decode::upload(seg_bytes, stream.view());
  auto col =
    sirius::test::decode::one_codec_column(cudf::data_type{cudf::type_id::FLOAT64},
                                           ROWS,
                                           CompressionType::COMPRESSION_ALP,
                                           {sirius::test::decode::segment(d_seg, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{ROWS} * sizeof(double));
  std::printf("[bench] ALP          f64 bw=11 32M rows:      %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench ALP float bw=20 32K vectors (32M rows)", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS = 32u * 1024u;
  constexpr uint32_t ROWS   = N_VECS * 1024u;

  auto seg_bytes = synth_alp_segment<float>(N_VECS, /*bit_width=*/20);
  auto d_seg     = sirius::test::decode::upload(seg_bytes, stream.view());
  auto col =
    sirius::test::decode::one_codec_column(cudf::data_type{cudf::type_id::FLOAT32},
                                           ROWS,
                                           CompressionType::COMPRESSION_ALP,
                                           {sirius::test::decode::segment(d_seg, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{ROWS} * sizeof(float));
  std::printf("[bench] ALP          f32 bw=20 32M rows:      %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench ALPRD double right_bw=48 left_bw=3 32K vectors (32M rows)",
          "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS = 32u * 1024u;
  constexpr uint32_t ROWS   = N_VECS * 1024u;

  std::vector<uint16_t> dict = {0xAAAA, 0x5555, 0x1234, 0xABCD, 0xDEAD, 0xBEEF, 0xCAFE, 0xF00D};
  auto seg_bytes = synth_alprd_segment<double>(N_VECS, /*right_bw=*/48, /*left_bw=*/3, dict);
  auto d_seg     = sirius::test::decode::upload(seg_bytes, stream.view());
  auto col =
    sirius::test::decode::one_codec_column(cudf::data_type{cudf::type_id::FLOAT64},
                                           ROWS,
                                           CompressionType::COMPRESSION_ALPRD,
                                           {sirius::test::decode::segment(d_seg, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{ROWS} * sizeof(double));
  std::printf("[bench] ALPRD        f64 r48/l3 32M rows:      %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

// Power-of-2 / byte-aligned widths. ALP-encoded TPC-H integer-derived columns
// (DECIMAL, dates, narrow ints) cluster on bw=8/16/32; the bw=11/20 cases
// above exercise the worst-case 64-bit-straddle bit-extract path.

TEST_CASE("bench ALP double bw=8 32K vectors (32M rows)", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS = 32u * 1024u;
  constexpr uint32_t ROWS   = N_VECS * 1024u;

  auto seg_bytes = synth_alp_segment<double>(N_VECS, /*bit_width=*/8);
  auto d_seg     = sirius::test::decode::upload(seg_bytes, stream.view());
  auto col =
    sirius::test::decode::one_codec_column(cudf::data_type{cudf::type_id::FLOAT64},
                                           ROWS,
                                           CompressionType::COMPRESSION_ALP,
                                           {sirius::test::decode::segment(d_seg, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{ROWS} * sizeof(double));
  std::printf("[bench] ALP          f64 bw=8  32M rows:      %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench ALP double bw=16 32K vectors (32M rows)", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS = 32u * 1024u;
  constexpr uint32_t ROWS   = N_VECS * 1024u;

  auto seg_bytes = synth_alp_segment<double>(N_VECS, /*bit_width=*/16);
  auto d_seg     = sirius::test::decode::upload(seg_bytes, stream.view());
  auto col =
    sirius::test::decode::one_codec_column(cudf::data_type{cudf::type_id::FLOAT64},
                                           ROWS,
                                           CompressionType::COMPRESSION_ALP,
                                           {sirius::test::decode::segment(d_seg, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{ROWS} * sizeof(double));
  std::printf("[bench] ALP          f64 bw=16 32M rows:      %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench ALP float bw=32 32K vectors (32M rows)", "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS = 32u * 1024u;
  constexpr uint32_t ROWS   = N_VECS * 1024u;

  auto seg_bytes = synth_alp_segment<float>(N_VECS, /*bit_width=*/32);
  auto d_seg     = sirius::test::decode::upload(seg_bytes, stream.view());
  auto col =
    sirius::test::decode::one_codec_column(cudf::data_type{cudf::type_id::FLOAT32},
                                           ROWS,
                                           CompressionType::COMPRESSION_ALP,
                                           {sirius::test::decode::segment(d_seg, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{ROWS} * sizeof(float));
  std::printf("[bench] ALP          f32 bw=32 32M rows:      %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}

TEST_CASE("bench ALPRD double right_bw=56 left_bw=2 32K vectors (32M rows)",
          "[!benchmark][scan][decode]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t N_VECS = 32u * 1024u;
  constexpr uint32_t ROWS   = N_VECS * 1024u;

  std::vector<uint16_t> dict = {0xAAAA, 0x5555, 0x1234, 0xABCD};
  auto seg_bytes = synth_alprd_segment<double>(N_VECS, /*right_bw=*/56, /*left_bw=*/2, dict);
  auto d_seg     = sirius::test::decode::upload(seg_bytes, stream.view());
  auto col =
    sirius::test::decode::one_codec_column(cudf::data_type{cudf::type_id::FLOAT64},
                                           ROWS,
                                           CompressionType::COMPRESSION_ALPRD,
                                           {sirius::test::decode::segment(d_seg, 0, ROWS)});

  double sec     = bench_seconds(stream, {col}, mr);
  double bytes_w = double(size_t{ROWS} * sizeof(double));
  std::printf("[bench] ALPRD        f64 r56/l2 32M rows:      %.6fs  write=%.1f GiB/s\n",
              sec,
              bytes_w / sec / GIB);
}
