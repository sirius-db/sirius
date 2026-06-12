// SPDX-License-Identifier: Apache-2.0
// nvcomp Snappy compressor (high-level SnappyManager API).
// Compress/decompress bodies are shared via nvcomp_simple_compressor.hpp.

#include "nvcomp_simple_compressor.hpp"

#include <nvcomp/snappy.hpp>

namespace simpatico {

namespace {

constexpr size_t kSnappyChunkSize = 64 * 1024;

struct snappy_manager_cache {
  std::unique_ptr<nvcomp::SnappyManager> mgr;
  cudaStream_t stream = nullptr;
};

static thread_local snappy_manager_cache tls_snappy_mgr;

nvcomp::SnappyManager* get_snappy_manager(cudaStream_t s)
{
  if (tls_snappy_mgr.mgr && tls_snappy_mgr.stream == s) return tls_snappy_mgr.mgr.get();
  tls_snappy_mgr.mgr =
    std::make_unique<nvcomp::SnappyManager>(kSnappyChunkSize,
                                            nvcompBatchedSnappyCompressDefaultOpts,
                                            nvcompBatchedSnappyDecompressDefaultOpts,
                                            s);
  tls_snappy_mgr.stream = s;
  return tls_snappy_mgr.mgr.get();
}

}  // namespace

std::unique_ptr<cudf::column> snappy_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return detail::nvcomp_decompress_impl(get_snappy_manager,
                                        compressed_data.get(),
                                        compressed_size,
                                        original_type,
                                        num_rows,
                                        stream,
                                        mr);
}

std::unique_ptr<compressed_representation> snappy_compressor::compress(
  cudf::column_view col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const dt = col.type();
  auto const n  = col.size();
  if (n == 0) {
    return std::make_unique<snappy_compressed_representation>(
      dt, 0, std::make_unique<rmm::device_buffer>(0, stream, mr), 0, 0);
  }
  auto [buf, actual]  = detail::nvcomp_compress_impl(get_snappy_manager, col, stream, mr);
  size_t const uncomp = static_cast<size_t>(n) * cudf::size_of(dt);
  return std::make_unique<snappy_compressed_representation>(dt, n, std::move(buf), actual, uncomp);
}

std::unique_ptr<cudf::column> snappy_compressor::decompress(compressed_representation const& rep,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  auto const* r = dynamic_cast<snappy_compressed_representation const*>(&rep);
  if (!r) return nullptr;
  return r->decompress(stream, mr);
}

}  // namespace simpatico
