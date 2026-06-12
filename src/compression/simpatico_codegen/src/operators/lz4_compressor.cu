// SPDX-License-Identifier: Apache-2.0
// nvcomp LZ4 compressor (high-level LZ4Manager API).
// Compress/decompress bodies are shared via nvcomp_simple_compressor.hpp.
// We always use NVCOMP_TYPE_CHAR so data is treated as an opaque byte stream.

#include "nvcomp_simple_compressor.hpp"

#include <nvcomp/lz4.hpp>

namespace simpatico {

namespace {

constexpr size_t kLz4ChunkSize = 64 * 1024;

struct lz4_manager_cache {
  std::unique_ptr<nvcomp::LZ4Manager> mgr;
  cudaStream_t stream = nullptr;
};

static thread_local lz4_manager_cache tls_lz4_mgr;

nvcomp::LZ4Manager* get_lz4_manager(cudaStream_t s)
{
  if (tls_lz4_mgr.mgr && tls_lz4_mgr.stream == s) return tls_lz4_mgr.mgr.get();
  // Always NVCOMP_TYPE_CHAR: we compress columns as opaque byte streams.
  nvcompBatchedLZ4CompressOpts_t copts = nvcompBatchedLZ4CompressDefaultOpts;
  copts.data_type                      = NVCOMP_TYPE_CHAR;
  tls_lz4_mgr.mgr                      = std::make_unique<nvcomp::LZ4Manager>(
    kLz4ChunkSize, copts, nvcompBatchedLZ4DecompressDefaultOpts, s);
  tls_lz4_mgr.stream = s;
  return tls_lz4_mgr.mgr.get();
}

}  // namespace

std::unique_ptr<cudf::column> lz4_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return detail::nvcomp_decompress_impl(
    get_lz4_manager, compressed_data.get(), compressed_size, original_type, num_rows, stream, mr);
}

std::unique_ptr<compressed_representation> lz4_compressor::compress(
  cudf::column_view col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const dt = col.type();
  auto const n  = col.size();
  if (n == 0) {
    return std::make_unique<lz4_compressed_representation>(
      dt, 0, std::make_unique<rmm::device_buffer>(0, stream, mr), 0, 0);
  }
  auto [buf, actual]  = detail::nvcomp_compress_impl(get_lz4_manager, col, stream, mr);
  size_t const uncomp = static_cast<size_t>(n) * cudf::size_of(dt);
  return std::make_unique<lz4_compressed_representation>(dt, n, std::move(buf), actual, uncomp);
}

std::unique_ptr<cudf::column> lz4_compressor::decompress(compressed_representation const& rep,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto const* r = dynamic_cast<lz4_compressed_representation const*>(&rep);
  if (!r) return nullptr;
  return r->decompress(stream, mr);
}

}  // namespace simpatico
