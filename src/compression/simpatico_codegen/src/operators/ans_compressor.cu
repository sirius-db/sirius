// SPDX-License-Identifier: Apache-2.0
// nvcomp ANS (entropy coder) compressor.
//
// Uses the nvcomp high-level Manager API (`nvcomp::ANSManager`), which handles
// batching internally. The Manager is cached thread-locally keyed by
// `(chunk_size, stream)` to avoid the per-call constructor cost. nvcomp requires
// the Manager not outlive its `user_stream`, so the cache invalidates on stream
// change.

#include "codegen/plan/representation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <nvcomp/ans.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace simpatico {

namespace {
// Internal chunk size used by the Manager. 64 KB matches the LZ4/Snappy
// sweet spot — big enough to amortize per-chunk overhead, small enough
// for parallelism. ANS supports up to 16 MB chunks (`nvcompANSCompressionMaxAllowedChunkSize`).
constexpr size_t kAnsManagerChunkSize = 64 * 1024;

// Thread-local cache of the ANSManager. Rebuilt on stream change.
struct ans_manager_cache {
  std::unique_ptr<nvcomp::ANSManager> mgr;
  cudaStream_t stream = nullptr;
  size_t chunk_size   = 0;
};

static thread_local ans_manager_cache tls_ans_mgr;

nvcomp::ANSManager* get_ans_manager(cudaStream_t s)
{
  if (tls_ans_mgr.mgr && tls_ans_mgr.stream == s &&
      tls_ans_mgr.chunk_size == kAnsManagerChunkSize) {
    return tls_ans_mgr.mgr.get();
  }
  tls_ans_mgr.mgr        = std::make_unique<nvcomp::ANSManager>(kAnsManagerChunkSize,
                                                         nvcompBatchedANSCompressDefaultOpts,
                                                         nvcompBatchedANSDecompressDefaultOpts,
                                                         s);
  tls_ans_mgr.stream     = s;
  tls_ans_mgr.chunk_size = kAnsManagerChunkSize;
  return tls_ans_mgr.mgr.get();
}
}  // namespace

std::vector<compressible_output> ans_compressed_representation::named_channels() const
{
  if (!serialized_output) {
    // Use a private non-blocking stream: the producing stream is always synced
    // before named_channels() is called, so compressed_data is already coherent.
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    rmm::cuda_stream_view stream(s);
    auto mr  = rmm::mr::get_current_device_resource_ref();
    auto out = cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::UINT8),
                                             static_cast<cudf::size_type>(compressed_size),
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
    if (compressed_size > 0 && compressed_data) {
      cudaMemcpyAsync(out->mutable_view().head<uint8_t>(),
                      compressed_data->data(),
                      compressed_size,
                      cudaMemcpyDeviceToDevice,
                      stream.value());
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    serialized_output = std::move(out);
  }
  return {{"output", serialized_output->view()}};
}

std::unique_ptr<cudf::column> ans_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (num_rows == 0 || compressed_data == nullptr || compressed_size == 0) {
    return cudf::make_fixed_width_column(
      original_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  auto* mgr = get_ans_manager(stream.value());

  auto const* comp_ptr = static_cast<uint8_t const*>(compressed_data->data());
  auto dconfig         = mgr->configure_decompression(comp_ptr);

  auto out_col = cudf::make_fixed_width_column(
    original_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_ptr = static_cast<uint8_t*>(out_col->mutable_view().head<void>());

  mgr->decompress(out_ptr, comp_ptr, dconfig);
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) return nullptr;
  return out_col;
}

std::unique_ptr<compressed_representation> ans_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const dt = column_to_compress.type();
  auto const n  = column_to_compress.size();

  if (n == 0) {
    return std::make_unique<ans_compressed_representation>(
      dt,
      0,
      std::make_unique<rmm::device_buffer>(0, stream, mr),
      /*comp_size=*/0,
      /*uncomp_size=*/0);
  }
  if (!cudf::is_fixed_width(dt)) {
    throw std::runtime_error("ans_compressor: only fixed-width columns supported");
  }

  size_t const uncompressed_size = static_cast<size_t>(n) * cudf::size_of(dt);

  auto* mgr = get_ans_manager(stream.value());

  auto cconfig = mgr->configure_compression(uncompressed_size);
  auto comp_buf =
    std::make_unique<rmm::device_buffer>(cconfig.max_compressed_buffer_size, stream, mr);

  // Device-side single-element size buffer to receive the actual
  // compressed size. NVCOMP_NATIVE bitstream embeds size in header;
  // we still want it on host for ratio reporting.
  rmm::device_buffer comp_size_dev(sizeof(size_t), stream, mr);

  mgr->compress(static_cast<uint8_t const*>(column_to_compress.head<void>()),
                static_cast<uint8_t*>(comp_buf->data()),
                cconfig,
                static_cast<size_t*>(comp_size_dev.data()));

  size_t actual_size = 0;
  if (cudaMemcpyAsync(&actual_size,
                      comp_size_dev.data(),
                      sizeof(size_t),
                      cudaMemcpyDeviceToHost,
                      stream.value()) != cudaSuccess) {
    throw std::runtime_error("ans_compressor: D2H size copy failed");
  }
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    throw std::runtime_error("ans_compressor: stream sync failed");
  }

  return std::make_unique<ans_compressed_representation>(
    dt, n, std::move(comp_buf), actual_size, uncompressed_size);
}

std::unique_ptr<cudf::column> ans_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* ans_repr = dynamic_cast<ans_compressed_representation const*>(&data_to_decompress);
  if (ans_repr == nullptr) return nullptr;
  return ans_repr->decompress(stream, mr);
}

}  // namespace simpatico
