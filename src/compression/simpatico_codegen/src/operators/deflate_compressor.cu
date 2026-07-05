// SPDX-License-Identifier: Apache-2.0
// nvcomp GDeflate compressor, exposed as "deflate" in the DSL.
// Uses the high-level GdeflateManager API.
// Compress/decompress bodies are shared via nvcomp_simple_compressor.hpp.
//
// DSL: `input -> deflate`
// Uses nvcomp default opts (algorithm=1: high-throughput, low ratio).

#include "nvcomp_simple_compressor.hpp"

#include <nvcomp/gdeflate.hpp>

namespace simpatico {

namespace {

constexpr size_t kDeflateChunkSize = 64 * 1024;

struct deflate_manager_cache {
  std::unique_ptr<nvcomp::GdeflateManager> mgr;
  cudaStream_t stream = nullptr;
};

static thread_local deflate_manager_cache tls_deflate_mgr;

nvcomp::GdeflateManager* get_deflate_manager(cudaStream_t s)
{
  if (tls_deflate_mgr.mgr && tls_deflate_mgr.stream == s) return tls_deflate_mgr.mgr.get();
  tls_deflate_mgr.mgr =
    std::make_unique<nvcomp::GdeflateManager>(kDeflateChunkSize,
                                              nvcompBatchedGdeflateCompressDefaultOpts,
                                              nvcompBatchedGdeflateDecompressDefaultOpts,
                                              s);
  tls_deflate_mgr.stream = s;
  return tls_deflate_mgr.mgr.get();
}

}  // namespace

std::unique_ptr<cudf::column> deflate_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return detail::nvcomp_decompress_impl(get_deflate_manager,
                                        compressed_data.get(),
                                        compressed_size,
                                        original_type,
                                        num_rows,
                                        stream,
                                        mr);
}

std::unique_ptr<compressed_representation> deflate_compressor::compress(
  cudf::column_view col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const dt = col.type();
  auto const n  = col.size();
  if (n == 0) {
    return std::make_unique<deflate_compressed_representation>(
      dt, 0, std::make_unique<rmm::device_buffer>(0, stream, mr), 0, 0);
  }
  auto [buf, actual]  = detail::nvcomp_compress_impl(get_deflate_manager, col, stream, mr);
  size_t const uncomp = static_cast<size_t>(n) * cudf::size_of(dt);
  return std::make_unique<deflate_compressed_representation>(dt, n, std::move(buf), actual, uncomp);
}

std::unique_ptr<cudf::column> deflate_compressor::decompress(compressed_representation const& rep,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  auto const* r = dynamic_cast<deflate_compressed_representation const*>(&rep);
  if (!r) return nullptr;
  return r->decompress(stream, mr);
}

namespace detail {
void release_deflate_manager_scratch()
{
  if (tls_deflate_mgr.mgr) tls_deflate_mgr.mgr->deallocate_gpu_mem();
}
}  // namespace detail

}  // namespace simpatico
