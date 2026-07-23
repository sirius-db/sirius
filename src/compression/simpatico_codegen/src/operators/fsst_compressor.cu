// SPDX-License-Identifier: Apache-2.0
// FSST-GPU (CompactionV5T) leaf codec — an opaque, self-describing byte-buffer
// compressor, wired in like snappy/ans. Typically applied to str_split's `chars`
// channel. The heavy lifting lives in the external fsst_gpu library
// (gtsst::compressors::CompactionV5TCompressor); this file only marshals cudf
// buffers in and out and pads the input to the codec's block granularity.
//
// CompactionV5T overflows a u32 above 2^32 uncompressed bytes, so a single
// buffer can hold at most ~4 GB. This operator transparently splits larger
// inputs into <4 GB segments, compresses each into its own self-describing
// payload, and packs them behind a small chunk directory. Single-chunk inputs
// (the common case, when Sirius hands over a segment < the chunk size) carry
// only a 40-byte frame header.

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <compressors/compactionv5t/compaction-compressor.cuh>
#include <compressors/compactionv5t/compaction-defines.cuh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {

namespace {

using gtsst::compressors::CompactionV5TCompressor;

// Uncompressed bytes per chunk: 3 GiB floored to the codec block size — safely
// under the 2^32 (4 GiB) format limit, with headroom for the compressed size.
constexpr size_t kChunkTargetBytes = size_t{3} << 30;

// Each compressed chunk starts at a 256-byte boundary within the payload, matching
// the alignment the decoder gets when a payload sits at the start of an rmm buffer.
constexpr size_t kChunkAlign = 256;

inline void check_cuda(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    // Clear the sticky per-thread error so an unrelated later launch on this
    // worker thread doesn't report it as its own failure.
    cudaGetLastError();
    throw std::runtime_error(std::string("fsst: ") + what + ": " + cudaGetErrorString(e));
  }
}

inline size_t round_up(size_t n, size_t m) { return ((n + m - 1) / m) * m; }

// Worst-case compressed size of one padded chunk. configure_compression's own
// compression_buffer_size (input + 76800) undersizes the headers, whose
// dominant term — the per-tile offset table, THREAD_COUNT * 4 bytes per block
// (~0.16% of input) — outgrows that fixed margin above ~48 MB when every block
// stores raw (FSST-incompressible input), and the library then writes past the
// buffer. Mirror its header layout and assume all-raw data.
inline size_t worst_case_dst_bytes(size_t padded)
{
  namespace cv5t        = gtsst::compressors::compactionv5t;
  size_t const n_blocks = padded / cv5t::BLOCK_SIZE;
  size_t const n_tables = n_blocks == 0 ? 1 : (n_blocks - 1) / cv5t::SUPER_BLOCK_SIZE + 1;
  return padded + sizeof(gtsst::compressors::CompactionV5TFileHeader) +
         n_tables * sizeof(gtsst::compressors::GBaseHeader) +
         n_blocks * (sizeof(gtsst::compressors::CompactionV5TBlockHeader) +
                     cv5t::THREAD_COUNT * sizeof(std::uint32_t));
}

// On-device payload frame:
//   u64 n_chunks
//   u64 chunk_uncomp          (uncompressed bytes per non-final chunk)
//   u64 total_uncomp          (original byte count)
//   n_chunks * { u64 comp_offset; u64 comp_size }   // offset from payload start
//   [pad] <compressed chunk 0> [pad] <compressed chunk 1> ...
struct FrameHeader {
  std::uint64_t n_chunks;
  std::uint64_t chunk_uncomp;
  std::uint64_t total_uncomp;
};
inline size_t dir_bytes(size_t n_chunks)
{
  return sizeof(FrameHeader) + n_chunks * 2 * sizeof(std::uint64_t);
}

}  // namespace

std::unique_ptr<compressed_representation> fsst_compressor::compress(
  cudf::column_view col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const dt      = col.type();
  auto const n_rows  = col.size();
  size_t const bytes = static_cast<size_t>(n_rows) * cudf::size_of(dt);

  if (bytes == 0) {
    return std::make_unique<fsst_compressed_representation>(
      dt, n_rows, std::make_unique<rmm::device_buffer>(0, stream, mr), 0, 0);
  }

  CompactionV5TCompressor comp;
  auto const probe           = comp.configure_compression(0);
  size_t const block         = probe.block_size;
  std::uint8_t const pad_sym = probe.padding_symbol;

  size_t const chunk_uncomp = (kChunkTargetBytes / block) * block;
  size_t const n_chunks     = (bytes + chunk_uncomp - 1) / chunk_uncomp;

  // Reused across chunks: a padded input staging buffer and the encoder scratch,
  // both sized for the largest chunk actually processed.
  size_t const max_chunk = std::min(chunk_uncomp, round_up(bytes, block));
  rmm::device_buffer src(max_chunk, stream, mr);
  rmm::device_buffer tmp(comp.configure_compression(max_chunk).temp_buffer_size, stream, mr);
  stream.synchronize();

  std::vector<std::unique_ptr<rmm::device_buffer>> chunk_dst(n_chunks);
  std::vector<size_t> comp_size(n_chunks);

  for (size_t c = 0; c < n_chunks; ++c) {
    size_t const off    = c * chunk_uncomp;
    size_t const len    = std::min(chunk_uncomp, bytes - off);
    size_t const padded = round_up(len, block);

    auto cfg                    = comp.configure_compression(padded);
    cfg.true_input_size         = len;
    cfg.device_buffers          = true;
    cfg.compression_buffer_size = worst_case_dst_bytes(padded);

    if (padded > len) {
      check_cuda(
        cudaMemsetAsync(
          static_cast<std::uint8_t*>(src.data()) + len, pad_sym, padded - len, stream.value()),
        "memset pad");
    }
    check_cuda(
      cudaMemcpyAsync(
        src.data(), col.head<std::uint8_t>() + off, len, cudaMemcpyDeviceToDevice, stream.value()),
      "copy input");
    stream.synchronize();

    auto dst = std::make_unique<rmm::device_buffer>(cfg.compression_buffer_size, stream, mr);
    stream.synchronize();

    size_t out_size = 0;
    gtsst::CompressionStatistics stats{};
    auto const st = comp.compress(static_cast<const std::uint8_t*>(src.data()),
                                  static_cast<std::uint8_t*>(dst->data()),
                                  static_cast<std::uint8_t*>(tmp.data()),
                                  cfg,
                                  &out_size,
                                  stats);
    if (st != gtsst::gtsstSuccess) {
      throw std::runtime_error("fsst: compress failed with status " + std::to_string(st));
    }
    if (out_size > cfg.compression_buffer_size) {
      throw std::runtime_error("fsst: compress out_size " + std::to_string(out_size) +
                               " overflows dst buffer " +
                               std::to_string(cfg.compression_buffer_size));
    }
    chunk_dst[c] = std::move(dst);
    comp_size[c] = out_size;
  }

  // Lay out the frame: header + directory, then each compressed chunk aligned.
  std::vector<std::uint64_t> comp_off(n_chunks);
  size_t cursor = dir_bytes(n_chunks);
  for (size_t c = 0; c < n_chunks; ++c) {
    cursor      = round_up(cursor, kChunkAlign);
    comp_off[c] = cursor;
    cursor += comp_size[c];
  }
  size_t const payload_bytes = cursor;

  std::vector<std::uint8_t> host_dir(dir_bytes(n_chunks));
  FrameHeader fh{n_chunks, chunk_uncomp, bytes};
  std::memcpy(host_dir.data(), &fh, sizeof(fh));
  auto* ent = reinterpret_cast<std::uint64_t*>(host_dir.data() + sizeof(fh));
  for (size_t c = 0; c < n_chunks; ++c) {
    ent[2 * c]     = comp_off[c];
    ent[2 * c + 1] = comp_size[c];
  }

  auto payload = std::make_unique<rmm::device_buffer>(payload_bytes, stream, mr);
  check_cuda(
    cudaMemcpyAsync(
      payload->data(), host_dir.data(), host_dir.size(), cudaMemcpyHostToDevice, stream.value()),
    "write frame dir");
  for (size_t c = 0; c < n_chunks; ++c) {
    check_cuda(cudaMemcpyAsync(static_cast<std::uint8_t*>(payload->data()) + comp_off[c],
                               chunk_dst[c]->data(),
                               comp_size[c],
                               cudaMemcpyDeviceToDevice,
                               stream.value()),
               "pack chunk");
  }
  stream.synchronize();

  return std::make_unique<fsst_compressed_representation>(
    dt, n_rows, std::move(payload), payload_bytes, bytes);
}

std::unique_ptr<cudf::column> fsst_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  auto const dt      = original_type;
  auto const n_rows  = num_rows;
  size_t const bytes = uncompressed_size;

  if (bytes == 0) {
    return std::make_unique<cudf::column>(
      dt, n_rows, rmm::device_buffer{0, stream, mr}, rmm::device_buffer{}, 0);
  }

  auto const* base = static_cast<const std::uint8_t*>(payload_data());

  FrameHeader fh{};
  check_cuda(cudaMemcpy(&fh, base, sizeof(fh), cudaMemcpyDeviceToHost), "read frame header");
  size_t const n_chunks     = fh.n_chunks;
  size_t const chunk_uncomp = fh.chunk_uncomp;

  std::vector<std::uint64_t> dir(2 * n_chunks);
  check_cuda(
    cudaMemcpy(
      dir.data(), base + sizeof(fh), dir.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
    "read frame dir");

  CompactionV5TCompressor comp;
  // Decode exactly once — skip the library's 20-iter kernel-timing loop, which only exists to
  // populate last_decode_kernel_ms for the standalone benchmark.
  comp.measure_decode_kernel = false;
  size_t const block         = comp.configure_compression(0).block_size;

  // Whole-block decode writes; size the output to the padded total across chunks.
  size_t padded_total = 0;
  for (size_t c = 0; c < n_chunks; ++c) {
    size_t const len = std::min(chunk_uncomp, bytes - c * chunk_uncomp);
    padded_total += round_up(len, block);
  }
  rmm::device_buffer out(padded_total, stream, mr);
  stream.synchronize();

  for (size_t c = 0; c < n_chunks; ++c) {
    size_t const off = c * chunk_uncomp;
    size_t const len = std::min(chunk_uncomp, bytes - off);

    gtsst::DecompressionConfiguration dcfg{};
    dcfg.input_buffer_size         = dir[2 * c + 1];
    dcfg.decompression_buffer_size = round_up(len, block);
    dcfg.device_buffers            = true;

    size_t out_size = 0;
    auto const st   = comp.decompress(
      base + dir[2 * c], static_cast<std::uint8_t*>(out.data()) + off, dcfg, &out_size);
    if (st != gtsst::gtsstSuccess) {
      throw std::runtime_error("fsst: decompress failed with status " + std::to_string(st));
    }
  }

  return std::make_unique<cudf::column>(dt, n_rows, std::move(out), rmm::device_buffer{}, 0);
}

}  // namespace simpatico
