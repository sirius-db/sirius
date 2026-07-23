// SPDX-License-Identifier: Apache-2.0
// FSST-GPU (CompactionV5T) leaf codec — an opaque, self-describing byte-buffer
// compressor, wired in like snappy/ans. Typically applied to str_split's `chars`
// channel. The heavy lifting lives in the external fsst_gpu library
// (gtsst::compressors::CompactionV5TCompressor); this file only marshals cudf
// buffers in and out and pads the input to the codec's block granularity.

#include "codegen/plan/representation.hpp"

#include <compressors/compactionv5t/compaction-compressor.cuh>

#include <cudf/column/column.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace simpatico {

namespace {

using gtsst::compressors::CompactionV5TCompressor;

inline void check_cuda(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("fsst: ") + what + ": " + cudaGetErrorString(e));
  }
}

// CompactionV5T requires the input to be a whole number of blocks; it stores the
// real pre-padding length in its header, so decompress() returns the exact size.
inline size_t round_up(size_t n, size_t block) { return ((n + block - 1) / block) * block; }

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
  auto const probe            = comp.configure_compression(0);
  size_t const block          = probe.block_size;
  std::uint8_t const pad_sym  = probe.padding_symbol;
  size_t const padded         = round_up(bytes, block);

  auto cfg              = comp.configure_compression(padded);
  cfg.true_input_size   = bytes;
  cfg.device_buffers    = true;

  // Padded, codec-aligned input copy (rmm buffers are 256B-aligned).
  rmm::device_buffer src(padded, stream, mr);
  check_cuda(cudaMemsetAsync(src.data(), pad_sym, padded, stream.value()), "memset pad");
  check_cuda(cudaMemcpyAsync(src.data(), col.head<std::uint8_t>(), bytes,
                             cudaMemcpyDeviceToDevice, stream.value()),
             "copy input");

  auto dst = std::make_unique<rmm::device_buffer>(cfg.compression_buffer_size, stream, mr);
  rmm::device_buffer tmp(cfg.temp_buffer_size, stream, mr);
  // fsst_gpu launches on its own stream, so make the padded input visible first.
  stream.synchronize();

  size_t out_size = 0;
  gtsst::CompressionStatistics stats{};
  auto const st = comp.compress(static_cast<const std::uint8_t*>(src.data()),
                                static_cast<std::uint8_t*>(dst->data()),
                                static_cast<std::uint8_t*>(tmp.data()), cfg, &out_size, stats);
  check_cuda(cudaDeviceSynchronize(), "compress sync");
  if (st != gtsst::gtsstSuccess) {
    throw std::runtime_error("fsst: compress failed with status " + std::to_string(st));
  }

  return std::make_unique<fsst_compressed_representation>(dt, n_rows, std::move(dst), out_size,
                                                          bytes);
}

std::unique_ptr<cudf::column> fsst_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  auto const dt     = original_type;
  auto const n_rows = num_rows;
  size_t const bytes = uncompressed_size;

  if (bytes == 0) {
    return std::make_unique<cudf::column>(dt, n_rows, rmm::device_buffer{0, stream, mr},
                                          rmm::device_buffer{}, 0);
  }

  CompactionV5TCompressor comp;
  // Decode exactly once — skip the library's 20-iter kernel-timing loop, which only exists to
  // populate last_decode_kernel_ms for the standalone benchmark.
  comp.measure_decode_kernel = false;
  size_t const block  = comp.configure_compression(0).block_size;
  size_t const padded = round_up(bytes, block);

  // Decoder writes whole blocks; over-allocate to the padded length, expose n_rows.
  rmm::device_buffer out(padded, stream, mr);
  stream.synchronize();

  gtsst::DecompressionConfiguration dcfg{};
  dcfg.input_buffer_size         = payload_size();
  dcfg.decompression_buffer_size = padded;
  dcfg.device_buffers            = true;

  size_t out_size = 0;
  auto const st   = comp.decompress(static_cast<const std::uint8_t*>(payload_data()),
                                  static_cast<std::uint8_t*>(out.data()), dcfg, &out_size);
  if (st != gtsst::gtsstSuccess) {
    throw std::runtime_error("fsst: decompress failed with status " + std::to_string(st));
  }

  return std::make_unique<cudf::column>(dt, n_rows, std::move(out), rmm::device_buffer{}, 0);
}

}  // namespace simpatico
