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

#include "scan/bitpacking_synth.hpp"
#include "scan/decode_test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_bitpacking.cuh>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
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
// BITPACKING benches.
//
// Each segment is BP_META_GROUP_SIZE rows so it dispatches as one CTA (the
// production shape — DuckDB writes ~one group per segment for bitpacked
// columns). The 4M-row workloads slice into ~2K segments; one batched
// kernel launch per column.
//===----------------------------------------------------------------------===//

namespace {

/// Helpers building synthetic bitpacked segments at on-disk layout.
template <typename T>
std::vector<uint32_t> bp_pack(std::vector<T> const& values, uint32_t width)
{
  if (width == 0 || values.empty()) return std::vector<uint32_t>(1, 0u);
  size_t total_bits  = static_cast<size_t>(values.size()) * width;
  size_t total_words = (total_bits + 31u) / 32u + 1u;
  std::vector<uint32_t> packed(total_words, 0u);
  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t v = static_cast<uint64_t>(values[i]);
    if (width < 64) v &= ((uint64_t{1} << width) - 1);
    size_t bit_pos = i * width, word_idx = bit_pos / 32, bit_off = bit_pos % 32;
    packed[word_idx] |= static_cast<uint32_t>(v << bit_off);
    if (bit_off + width > 32) packed[word_idx + 1] |= static_cast<uint32_t>(v >> (32 - bit_off));
    if (sizeof(T) > 4 && bit_off > 0 && bit_off + width > 64)
      packed[word_idx + 2] |= static_cast<uint32_t>(v >> (64 - bit_off));
  }
  return packed;
}

template <typename T>
std::vector<uint8_t> bp_for_segment(T frame, uint32_t width, std::vector<T> const& values)
{
  auto packed = bp_pack<T>(values, width);
  size_t pb   = packed.size() * sizeof(uint32_t);
  // [metadata_end:u64][frame:T][width:T][packed][trailer:u32]
  uint64_t metadata_end = 8 + 2 * sizeof(T) + pb + 4;
  std::vector<uint8_t> bytes(metadata_end, 0);
  std::memcpy(bytes.data(), &metadata_end, sizeof(uint64_t));
  T width_t = static_cast<T>(width);
  std::memcpy(bytes.data() + 8, &frame, sizeof(T));
  std::memcpy(bytes.data() + 8 + sizeof(T), &width_t, sizeof(T));
  std::memcpy(bytes.data() + 8 + 2 * sizeof(T), packed.data(), pb);
  uint32_t encoded = (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::FOR) << 24) | 8u;
  std::memcpy(bytes.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return bytes;
}

template <typename T>
std::vector<uint8_t> bp_constant_segment(T value)
{
  std::vector<uint8_t> bytes(64, 0);
  uint64_t metadata_end = 32;
  std::memcpy(bytes.data(), &metadata_end, sizeof(uint64_t));
  std::memcpy(bytes.data() + 8, &value, sizeof(T));
  uint32_t encoded =
    (static_cast<uint32_t>(::sirius::cuda::scan::BitpackingMode::CONSTANT) << 24) | 8u;
  std::memcpy(bytes.data() + metadata_end - 4, &encoded, sizeof(encoded));
  return bytes;
}

}  // namespace

TEST_CASE("bench BITPACKING int64 FOR width=8 128M rows", "[!benchmark][scan][decode]")
{
  // Production shape: one segment per metadata group, ~2K segments per 4M
  // rows. Width=8 — 8x compression vs UNCOMPRESSED, exercises the unpack
  // hot path (bit_off varies, no third-word reads).
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  constexpr uint32_t SEG_ROWS = ::sirius::cuda::scan::BP_META_GROUP_SIZE;
  constexpr uint32_t N_SEGS   = (128u << 20) / SEG_ROWS;
  std::vector<int64_t> deltas(SEG_ROWS);
  for (uint32_t i = 0; i < SEG_ROWS; ++i)
    deltas[i] = i & 0xFF;
  auto seg_bytes = bp_for_segment<int64_t>(/*frame=*/1000, /*width=*/8, deltas);

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
  auto seg_bytes = bp_for_segment<int32_t>(0, 12, deltas);

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
  auto seg_bytes              = bp_constant_segment<int64_t>(12345);

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
  using ::sirius::test::decode::bitpacking::make_for_block;
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
  using ::sirius::test::decode::bitpacking::make_for_block;
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
  // DELTA_FOR exercises cub::BlockScan + blocked->striped shmem exchange.
  // Target for the WarpScan optimisation (cudf delta_binary.cuh:260-261
  // pattern). All deltas are tiny so width=8 is sufficient.
  using ::sirius::test::decode::bitpacking::make_delta_for_block;
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
