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
#include "scan/strings_synth.hpp"

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_bitpacking.cuh>
#include <cuda/scan/gpu_decode_strings.cuh>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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

//===----------------------------------------------------------------------===//
// String-codec benches. Segments are synthesized via strings_synth.hpp
// (FSST / DICT_FSST go through `duckdb_fsst_create`). FSST / DICT_FSST
// benches accept a fixture override via `SIRIUS_BENCH_<CODEC>_FIXTURE` +
// `_ROWS` env vars to run against a real DuckDB-extracted segment.
//===----------------------------------------------------------------------===//

namespace {

double bench_strings_seconds(rmm::cuda_stream& stream,
                             sirius::cuda::scan::gpu_string_column_decode_input const& col,
                             rmm::mr::cuda_async_memory_resource& mr,
                             int iters  = 10,
                             int warmup = 3)
{
  for (int i = 0; i < warmup; ++i)
    (void)sirius::cuda::scan::gpu_decode_strings_column(col, stream.view(), mr);
  cudaStreamSynchronize(stream.value());

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s, stream.value());
  for (int i = 0; i < iters; ++i)
    (void)sirius::cuda::scan::gpu_decode_strings_column(col, stream.view(), mr);
  cudaEventRecord(e, stream.value());
  cudaEventSynchronize(e);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  return (ms / 1000.0) / iters;
}

/// Load raw segment bytes from a path in env var `var_name`. Returns empty
/// vector if the var is unset or the file isn't readable.
std::vector<uint8_t> load_fixture_bytes(char const* var_name)
{
  char const* path = std::getenv(var_name);
  if (path == nullptr || *path == '\0') return {};
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto sz = f.tellg();
  if (sz <= 0) return {};
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(sz));
  f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(sz));
  return bytes;
}

}  // namespace

TEST_CASE("bench DICTIONARY 1M rows / 1024 dict / 16-byte avg",
          "[!benchmark][scan][decode][strings]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  constexpr uint32_t DICT_COUNT = 1024;
  constexpr uint32_t ROWS       = 1u << 20;
  std::vector<std::string> dict(DICT_COUNT);
  dict[0] = "";  // index 0 = NULL sentinel
  for (uint32_t k = 1; k < DICT_COUNT; ++k) {
    dict[k] = std::string(16, static_cast<char>('a' + (k % 26)));
  }
  std::vector<uint32_t> sel(ROWS);
  for (uint32_t i = 0; i < ROWS; ++i)
    sel[i] = 1u + (i % (DICT_COUNT - 1u));
  auto bytes = sirius::test::decode::strings::make_dict_segment(dict, sel);

  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());
  sirius::cuda::scan::gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()),
                                                  static_cast<uint32_t>(d_seg.size()),
                                                  0,
                                                  ROWS,
                                                  0,
                                                  /*max_string_length=*/16u};
  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = ROWS;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_DICTIONARY, {seg}});

  double sec           = bench_strings_seconds(stream, col, mr);
  double bytes_decoded = double(ROWS) * 16.0;
  std::printf(
    "[bench] DICTIONARY    1M/1024d/16B:       %.6fs  decode=%.1f Mr/s  write=%.1f GiB/s\n",
    sec,
    double(ROWS) / sec / 1e6,
    bytes_decoded / sec / GIB);
}

TEST_CASE("bench DICTIONARY low-cardinality / 2 entries × 2000B",
          "[!benchmark][scan][decode][strings]")
{
  // DuckDB caps DICTIONARY at <4096B total dict (DICTIONARY_ENCODE_THRESHOLD,
  // dict_fsst/compression.cpp:34). Low-cardinality long-string columns
  // (log-template variants, status descriptions) stay DICTIONARY at
  // per-entry sizes up to ~2KB. Stresses the gather pattern at lengths the
  // 16B fixture can't reach.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  constexpr uint32_t DICT_COUNT = 3;  // [0]=NULL + 2 real entries × 2000B = 4000B < 4096
  constexpr uint32_t ENTRY_LEN  = 2000;
  constexpr uint32_t ROWS       = 128u * 1024u;  // 128K × 2000B = 256MB output
  std::vector<std::string> dict(DICT_COUNT);
  dict[0] = "";
  for (uint32_t k = 1; k < DICT_COUNT; ++k) {
    dict[k] = sirius::test::decode::strings::make_tpch_like_comment(
      static_cast<uint64_t>(k) * 0x9E3779B97F4A7C15ull, ENTRY_LEN, ENTRY_LEN + 1u);
  }
  std::vector<uint32_t> sel(ROWS);
  for (uint32_t i = 0; i < ROWS; ++i)
    sel[i] = 1u + (i % (DICT_COUNT - 1u));
  auto bytes = sirius::test::decode::strings::make_dict_segment(dict, sel);

  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());
  sirius::cuda::scan::gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()),
                                                  static_cast<uint32_t>(d_seg.size()),
                                                  0,
                                                  ROWS,
                                                  0,
                                                  ENTRY_LEN + 1u};
  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = ROWS;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_DICTIONARY, {seg}});

  double sec           = bench_strings_seconds(stream, col, mr);
  double bytes_decoded = double(ROWS) * double(ENTRY_LEN);
  std::printf(
    "[bench] DICT-2KB      128K/2d/2000B:        %.6fs  decode=%.1f Mr/s  write=%.1f GiB/s\n",
    sec,
    double(ROWS) / sec / 1e6,
    bytes_decoded / sec / GIB);
}

TEST_CASE("bench FSST 1M rows / TPC-H-like comments", "[!benchmark][scan][decode][strings]")
{
  // TPC-H-style l_comment workload: 5-150 byte comments built from dbgen's
  // grammar (nouns/verbs/adjectives/adverbs/prepositions). High redundancy
  // of common words ("the", "carefully", "pending"). FSST symbol-table
  // coverage and length distribution match what production varchar columns
  // actually look like — much more demanding of the gather kernel than
  // structured "row_N_payload_X" strings.
  // Override with SIRIUS_BENCH_FSST_FIXTURE + SIRIUS_BENCH_FSST_ROWS.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  uint32_t rows = 1u << 20;
  std::vector<uint8_t> bytes;
  size_t total_input_bytes = 0;
  auto override_bytes      = load_fixture_bytes("SIRIUS_BENCH_FSST_FIXTURE");
  if (!override_bytes.empty()) {
    char const* rows_str = std::getenv("SIRIUS_BENCH_FSST_ROWS");
    if (rows_str != nullptr) rows = static_cast<uint32_t>(std::atol(rows_str));
    bytes = std::move(override_bytes);
  } else {
    std::vector<std::string> synth(rows);
    for (uint32_t i = 0; i < rows; ++i) {
      synth[i] = sirius::test::decode::strings::make_tpch_like_comment(
        static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull, 5u, 150u);
      total_input_bytes += synth[i].size();
    }
    bytes = sirius::test::decode::strings::make_fsst_segment(synth);
  }

  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());
  sirius::cuda::scan::gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()),
                                                  static_cast<uint32_t>(d_seg.size()),
                                                  0,
                                                  rows,
                                                  0,
                                                  /*max_string_length=*/160u};
  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = rows;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_FSST, {seg}});

  double sec     = bench_strings_seconds(stream, col, mr);
  double avg_len = total_input_bytes ? double(total_input_bytes) / rows : 0.0;
  std::printf(
    "[bench] FSST          TPC-H-like %.0fB avg: %.6fs  decode=%.1f Mr/s  write=%.1f GiB/s\n",
    avg_len,
    sec,
    double(rows) / sec / 1e6,
    double(rows) * avg_len / sec / GIB);
}

TEST_CASE("bench FSST 1M rows / long comments", "[!benchmark][scan][decode][strings]")
{
  // Long-row variant: 200-800 byte comments built from the same TPC-H
  // grammar, mean ~500B. Stress for the per-row warp-cooperative path
  // where a single warp may spend many chunks of work on one row.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  uint32_t rows = 1u << 20;
  std::vector<uint8_t> bytes;
  size_t total_input_bytes = 0;
  auto override_bytes      = load_fixture_bytes("SIRIUS_BENCH_FSST_LONG_FIXTURE");
  if (!override_bytes.empty()) {
    char const* rows_str = std::getenv("SIRIUS_BENCH_FSST_LONG_ROWS");
    if (rows_str != nullptr) rows = static_cast<uint32_t>(std::atol(rows_str));
    bytes = std::move(override_bytes);
  } else {
    std::vector<std::string> synth(rows);
    for (uint32_t i = 0; i < rows; ++i) {
      synth[i] = sirius::test::decode::strings::make_tpch_like_comment(
        static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull, 200u, 800u);
      total_input_bytes += synth[i].size();
    }
    bytes = sirius::test::decode::strings::make_fsst_segment(synth);
  }

  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());
  sirius::cuda::scan::gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()),
                                                  static_cast<uint32_t>(d_seg.size()),
                                                  0,
                                                  rows,
                                                  0,
                                                  /*max_string_length=*/800u};
  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = rows;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_FSST, {seg}});

  double sec     = bench_strings_seconds(stream, col, mr);
  double avg_len = total_input_bytes ? double(total_input_bytes) / rows : 0.0;
  std::printf(
    "[bench] FSST-long     %.0fB avg:            %.6fs  decode=%.1f Mr/s  write=%.1f GiB/s\n",
    avg_len,
    sec,
    double(rows) / sec / 1e6,
    double(rows) * avg_len / sec / GIB);
}

namespace {

/// Measured D2D bandwidth — sets the practical bandwidth ceiling against
/// which decoder write throughput is reported. 256 MiB transfer is large
/// enough to saturate the controller while staying inside RMM pool budgets.
double measure_peak_d2d_gbs(rmm::cuda_stream& stream,
                            rmm::mr::cuda_async_memory_resource& mr,
                            int iters  = 10,
                            int warmup = 3)
{
  constexpr size_t COPY_BYTES = size_t{256} << 20;
  rmm::device_buffer src(COPY_BYTES, stream, mr);
  rmm::device_buffer dst(COPY_BYTES, stream, mr);
  for (int i = 0; i < warmup; ++i)
    cudaMemcpyAsync(dst.data(), src.data(), COPY_BYTES, cudaMemcpyDeviceToDevice, stream.value());
  cudaStreamSynchronize(stream.value());
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s, stream.value());
  for (int i = 0; i < iters; ++i)
    cudaMemcpyAsync(dst.data(), src.data(), COPY_BYTES, cudaMemcpyDeviceToDevice, stream.value());
  cudaEventRecord(e, stream.value());
  cudaEventSynchronize(e);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  // D2D copy moves 2× bytes across the bus (1 read + 1 write).
  double sec     = (ms / 1000.0) / iters;
  double bytes_x = 2.0 * double(COPY_BYTES);
  return bytes_x / sec / 1e9;  // GB/s (decimal, matching spec sheets)
}

}  // namespace

TEST_CASE("bench FSST realistic multi-segment / TPC-H-like comments",
          "[!benchmark][scan][decode][strings]")
{
  // Realistic DuckDB layout: FSST segments cap at Storage::DEFAULT_BLOCK_SIZE
  // (~256 KiB). For TPC-H l_comment (~78 B avg), that's ~5–7 K rows/segment.
  // 1 M rows therefore split into ~150–200 segments, which is the regime the
  // Phase A+B kernel's grid = num_segments is sized for.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  double peak_d2d_gbs = measure_peak_d2d_gbs(stream, mr);
  std::printf("[bench] peak D2D bandwidth (measured): %.1f GB/s\n", peak_d2d_gbs);

  uint32_t const rows = 1u << 20;
  std::vector<std::string> synth(rows);
  size_t total_input_bytes = 0;
  for (uint32_t i = 0; i < rows; ++i) {
    synth[i] = sirius::test::decode::strings::make_tpch_like_comment(
      static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull, 5u, 150u);
    total_input_bytes += synth[i].size();
  }
  auto seg_blobs = sirius::test::decode::strings::make_fsst_segments_chunked(synth);

  size_t total_seg_bytes = 0;
  uint32_t max_seg_rows  = 0;
  uint32_t max_seg_bytes = 0;
  for (auto const& [b, rc] : seg_blobs) {
    total_seg_bytes += b.size();
    max_seg_rows  = std::max(max_seg_rows, rc);
    max_seg_bytes = std::max(max_seg_bytes, static_cast<uint32_t>(b.size()));
  }
  double avg_seg_rows  = double(rows) / seg_blobs.size();
  double avg_seg_bytes = double(total_seg_bytes) / seg_blobs.size();

  // Pad each segment up to 8 B so its on-device base is 8 B aligned — kernel
  // reads `packed = reinterpret_cast<uint32_t const*>(base + 16)` and Phase A
  // reads 64-bit windows out of it.
  constexpr size_t SEG_ALIGN = 8;
  std::vector<size_t> seg_offsets(seg_blobs.size());
  size_t alloc_bytes = 0;
  for (size_t k = 0; k < seg_blobs.size(); ++k) {
    seg_offsets[k] = alloc_bytes;
    alloc_bytes += (seg_blobs[k].first.size() + SEG_ALIGN - 1) & ~(SEG_ALIGN - 1);
  }
  rmm::device_buffer d_all(alloc_bytes, stream.view());
  std::vector<sirius::cuda::scan::gpu_string_segment_desc> segs;
  segs.reserve(seg_blobs.size());
  uint32_t row_cursor = 0;
  auto const* d_base  = static_cast<uint8_t const*>(d_all.data());
  for (size_t k = 0; k < seg_blobs.size(); ++k) {
    auto const& [b, rc] = seg_blobs[k];
    cudaMemcpyAsync(const_cast<uint8_t*>(d_base) + seg_offsets[k],
                    b.data(),
                    b.size(),
                    cudaMemcpyHostToDevice,
                    stream.value());
    segs.push_back({d_base + seg_offsets[k],
                    static_cast<uint32_t>(b.size()),
                    row_cursor,
                    rc,
                    /*seg_row_start=*/0,
                    /*max_string_length=*/160u});
    row_cursor += rc;
  }
  cudaStreamSynchronize(stream.value());

  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = rows;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_FSST, std::move(segs)});

  double sec      = bench_strings_seconds(stream, col, mr);
  double avg_len  = double(total_input_bytes) / rows;
  double out_gbs  = double(rows) * avg_len / sec / 1e9;  // decimal GB/s
  double pct_peak = 100.0 * out_gbs / peak_d2d_gbs;

  std::printf(
    "[bench] FSST realistic  %.0fB avg, %zu segs (avg %.0f rows / %.0f B), max %u rows / %u B:\n",
    avg_len,
    seg_blobs.size(),
    avg_seg_rows,
    avg_seg_bytes,
    max_seg_rows,
    max_seg_bytes);
  std::printf(
    "[bench]   %.6fs  decode=%.1f Mr/s  output write=%.1f GB/s  (%.1f%% of measured peak D2D)\n",
    sec,
    double(rows) / sec / 1e6,
    out_gbs,
    pct_peak);
}

TEST_CASE("bench FSST realistic multi-segment / long comments",
          "[!benchmark][scan][decode][strings]")
{
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  double peak_d2d_gbs = measure_peak_d2d_gbs(stream, mr);
  std::printf("[bench] peak D2D bandwidth (measured): %.1f GB/s\n", peak_d2d_gbs);

  uint32_t const rows = 1u << 20;
  std::vector<std::string> synth(rows);
  size_t total_input_bytes = 0;
  for (uint32_t i = 0; i < rows; ++i) {
    synth[i] = sirius::test::decode::strings::make_tpch_like_comment(
      static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull, 200u, 800u);
    total_input_bytes += synth[i].size();
  }
  auto seg_blobs = sirius::test::decode::strings::make_fsst_segments_chunked(synth);

  size_t total_seg_bytes = 0;
  uint32_t max_seg_rows  = 0;
  uint32_t max_seg_bytes = 0;
  for (auto const& [b, rc] : seg_blobs) {
    total_seg_bytes += b.size();
    max_seg_rows  = std::max(max_seg_rows, rc);
    max_seg_bytes = std::max(max_seg_bytes, static_cast<uint32_t>(b.size()));
  }
  double avg_seg_rows  = double(rows) / seg_blobs.size();
  double avg_seg_bytes = double(total_seg_bytes) / seg_blobs.size();

  constexpr size_t SEG_ALIGN = 8;
  std::vector<size_t> seg_offsets(seg_blobs.size());
  size_t alloc_bytes = 0;
  for (size_t k = 0; k < seg_blobs.size(); ++k) {
    seg_offsets[k] = alloc_bytes;
    alloc_bytes += (seg_blobs[k].first.size() + SEG_ALIGN - 1) & ~(SEG_ALIGN - 1);
  }
  rmm::device_buffer d_all(alloc_bytes, stream.view());
  std::vector<sirius::cuda::scan::gpu_string_segment_desc> segs;
  segs.reserve(seg_blobs.size());
  uint32_t row_cursor = 0;
  auto const* d_base  = static_cast<uint8_t const*>(d_all.data());
  for (size_t k = 0; k < seg_blobs.size(); ++k) {
    auto const& [b, rc] = seg_blobs[k];
    cudaMemcpyAsync(const_cast<uint8_t*>(d_base) + seg_offsets[k],
                    b.data(),
                    b.size(),
                    cudaMemcpyHostToDevice,
                    stream.value());
    segs.push_back({d_base + seg_offsets[k],
                    static_cast<uint32_t>(b.size()),
                    row_cursor,
                    rc,
                    0,
                    /*max_string_length=*/800u});
    row_cursor += rc;
  }
  cudaStreamSynchronize(stream.value());

  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = rows;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_FSST, std::move(segs)});

  double sec      = bench_strings_seconds(stream, col, mr);
  double avg_len  = double(total_input_bytes) / rows;
  double out_gbs  = double(rows) * avg_len / sec / 1e9;
  double pct_peak = 100.0 * out_gbs / peak_d2d_gbs;

  std::printf(
    "[bench] FSST realistic-long %.0fB avg, %zu segs (avg %.0f rows / %.0f B), max %u rows / %u "
    "B:\n",
    avg_len,
    seg_blobs.size(),
    avg_seg_rows,
    avg_seg_bytes,
    max_seg_rows,
    max_seg_bytes);
  std::printf(
    "[bench]   %.6fs  decode=%.1f Mr/s  output write=%.1f GB/s  (%.1f%% of measured peak D2D)\n",
    sec,
    double(rows) / sec / 1e6,
    out_gbs,
    pct_peak);
}

TEST_CASE("bench DICT_FSST mode 1 1M rows / TPC-H-like dict", "[!benchmark][scan][decode][strings]")
{
  // DICT_FSST mode 1 with a TPC-H-like dict: ~50K unique l_comments collapse
  // to a 50K-entry dict in a real lineitem segment, but DuckDB caps a
  // segment's dict at the bitpacking-width-fits-in-segment limit, so per
  // segment ~256-2048 entries. We use 1024 entries here. Real comment shape
  // (5-150 byte words from dbgen grammar) gives realistic FSST behavior.
  // Override via SIRIUS_BENCH_DICT_FSST_FIXTURE / _ROWS.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  uint32_t const DICT = 1024;
  uint32_t rows       = 1u << 20;
  std::vector<uint8_t> bytes;
  size_t total_dict_bytes = 0;
  auto override_bytes     = load_fixture_bytes("SIRIUS_BENCH_DICT_FSST_FIXTURE");
  if (!override_bytes.empty()) {
    char const* rows_str = std::getenv("SIRIUS_BENCH_DICT_FSST_ROWS");
    if (rows_str != nullptr) rows = static_cast<uint32_t>(std::atol(rows_str));
    bytes = std::move(override_bytes);
  } else {
    std::vector<std::string> dict(DICT);
    dict[0] = "";  // reserved NULL
    for (uint32_t k = 1; k < DICT; ++k) {
      dict[k] = sirius::test::decode::strings::make_tpch_like_comment(
        static_cast<uint64_t>(k) * 0xBF58476D1CE4E5B9ull, 5u, 150u);
      total_dict_bytes += dict[k].size();
    }
    std::vector<uint32_t> sel(rows);
    for (uint32_t i = 0; i < rows; ++i)
      sel[i] = (i % (DICT - 1u)) + 1u;
    bytes = sirius::test::decode::strings::make_dict_fsst_segment(dict, sel, /*mode=*/1);
  }

  rmm::device_buffer d_seg(bytes.data(), bytes.size(), stream.view());
  sirius::cuda::scan::gpu_string_segment_desc seg{static_cast<uint8_t const*>(d_seg.data()),
                                                  static_cast<uint32_t>(d_seg.size()),
                                                  0,
                                                  rows,
                                                  0,
                                                  /*max_string_length=*/160u};
  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = rows;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_DICT_FSST, {seg}});

  double sec          = bench_strings_seconds(stream, col, mr);
  double avg_dict_len = total_dict_bytes ? double(total_dict_bytes) / (DICT - 1u) : 0.0;
  std::printf(
    "[bench] DICT_FSST mode-1 TPC-H-like %.0fB avg dict: %.6fs  decode=%.1f Mr/s  write=%.1f "
    "GiB/s\n",
    avg_dict_len,
    sec,
    double(rows) / sec / 1e6,
    double(rows) * avg_dict_len / sec / GIB);
}

TEST_CASE("bench DICT_FSST realistic multi-segment mode 1", "[!benchmark][scan][decode][strings]")
{
  // Realistic DuckDB layout: DICT_FSST segments cap at Storage::DEFAULT_BLOCK_SIZE
  // (~256 KiB). For TPC-H-like comments and mode-1 dicts this gives roughly
  // 1-4 K rows per segment depending on dict reuse; 1 M rows therefore split
  // into many segments and exercise the per-segment host work in
  // prepare_dict_fsst (2 sync D2H per segment) as well as kernel scaling.
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;

  double peak_d2d_gbs = measure_peak_d2d_gbs(stream, mr);
  std::printf("[bench] peak D2D bandwidth (measured): %.1f GB/s\n", peak_d2d_gbs);

  uint32_t const rows = 1u << 20;
  std::vector<std::string> synth(rows);
  size_t total_input_bytes = 0;
  for (uint32_t i = 0; i < rows; ++i) {
    synth[i] = sirius::test::decode::strings::make_tpch_like_comment(
      static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull, 5u, 150u);
    total_input_bytes += synth[i].size();
  }
  auto seg_blobs =
    sirius::test::decode::strings::make_dict_fsst_segments_chunked(synth, /*mode=*/1);

  size_t total_seg_bytes = 0;
  uint32_t max_seg_rows  = 0;
  uint32_t max_seg_bytes = 0;
  for (auto const& [b, rc] : seg_blobs) {
    total_seg_bytes += b.size();
    max_seg_rows  = std::max(max_seg_rows, rc);
    max_seg_bytes = std::max(max_seg_bytes, static_cast<uint32_t>(b.size()));
  }
  double avg_seg_rows  = double(rows) / seg_blobs.size();
  double avg_seg_bytes = double(total_seg_bytes) / seg_blobs.size();

  // Pad each segment up to 8 B so its on-device base stays 8-aligned (header
  // read at d_bytes[0..16) crosses no alignment boundary).
  constexpr size_t SEG_ALIGN = 8;
  std::vector<size_t> seg_offsets(seg_blobs.size());
  size_t alloc_bytes = 0;
  for (size_t k = 0; k < seg_blobs.size(); ++k) {
    seg_offsets[k] = alloc_bytes;
    alloc_bytes += (seg_blobs[k].first.size() + SEG_ALIGN - 1) & ~(SEG_ALIGN - 1);
  }
  rmm::device_buffer d_all(alloc_bytes, stream.view());
  std::vector<sirius::cuda::scan::gpu_string_segment_desc> segs;
  segs.reserve(seg_blobs.size());
  uint32_t row_cursor = 0;
  auto const* d_base  = static_cast<uint8_t const*>(d_all.data());
  for (size_t k = 0; k < seg_blobs.size(); ++k) {
    auto const& [b, rc] = seg_blobs[k];
    cudaMemcpyAsync(const_cast<uint8_t*>(d_base) + seg_offsets[k],
                    b.data(),
                    b.size(),
                    cudaMemcpyHostToDevice,
                    stream.value());
    segs.push_back({d_base + seg_offsets[k],
                    static_cast<uint32_t>(b.size()),
                    row_cursor,
                    rc,
                    /*seg_row_start=*/0,
                    /*max_string_length=*/160u});
    row_cursor += rc;
  }
  cudaStreamSynchronize(stream.value());

  sirius::cuda::scan::gpu_string_column_decode_input col;
  col.total_rows = rows;
  col.has_nulls  = false;
  col.data.push_back({CompressionType::COMPRESSION_DICT_FSST, std::move(segs)});

  double sec      = bench_strings_seconds(stream, col, mr);
  double avg_len  = double(total_input_bytes) / rows;
  double out_gbs  = double(rows) * avg_len / sec / 1e9;
  double pct_peak = 100.0 * out_gbs / peak_d2d_gbs;

  std::printf(
    "[bench] DICT_FSST realistic mode-1 %.0fB avg, %zu segs (avg %.0f rows / %.0f B), "
    "max %u rows / %u B:\n",
    avg_len,
    seg_blobs.size(),
    avg_seg_rows,
    avg_seg_bytes,
    max_seg_rows,
    max_seg_bytes);
  std::printf(
    "[bench]   %.6fs  decode=%.1f Mr/s  output write=%.1f GB/s  (%.1f%% of measured peak D2D)\n",
    sec,
    double(rows) / sec / 1e6,
    out_gbs,
    pct_peak);
}
