// SPDX-License-Identifier: Apache-2.0
// nvcomp Bitcomp standalone compressor.
//
// Mirror of ans_compressor.cu using `nvcomp::BitcompManager`. Bitcomp
// targets numeric data with high zero-density / low-magnitude residues —
// a natural fit downstream of delta or RLE chains.
//
// DSL surface:
//   `input -> bitcomp`           algorithm=0 (default, best ratio).
//   `input -> bitcomp_default`   alias for bitcomp.
//   `input -> bitcomp_sparse`    algorithm=1 — faster on zero-rich data.
//
// The compress-time algorithm is stashed on the rep so decompress
// reuses the same Manager cache slot.

#include "codegen/plan/representation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <nvcomp/bitcomp.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace simpatico {

namespace {
constexpr size_t kBitcompManagerChunkSize = 64 * 1024;

struct bitcomp_manager_cache {
  std::unique_ptr<nvcomp::BitcompManager> mgr;
  cudaStream_t stream = nullptr;
  size_t chunk_size   = 0;
  int algorithm       = -1;
};

static thread_local bitcomp_manager_cache tls_bitcomp_mgr;

nvcomp::BitcompManager* get_bitcomp_manager(cudaStream_t s, int algorithm)
{
  if (tls_bitcomp_mgr.mgr && tls_bitcomp_mgr.stream == s &&
      tls_bitcomp_mgr.chunk_size == kBitcompManagerChunkSize &&
      tls_bitcomp_mgr.algorithm == algorithm) {
    return tls_bitcomp_mgr.mgr.get();
  }
  nvcompBatchedBitcompCompressOpts_t copts = nvcompBatchedBitcompCompressDefaultOpts;
  copts.algorithm                          = algorithm;
  tls_bitcomp_mgr.mgr                      = std::make_unique<nvcomp::BitcompManager>(
    kBitcompManagerChunkSize, copts, nvcompBatchedBitcompDecompressDefaultOpts, s);
  tls_bitcomp_mgr.stream     = s;
  tls_bitcomp_mgr.chunk_size = kBitcompManagerChunkSize;
  tls_bitcomp_mgr.algorithm  = algorithm;
  return tls_bitcomp_mgr.mgr.get();
}
}  // namespace

// Parses a bitcomp suffix. `suffix` is the part after the underscore
// in `bitcomp_<suffix>`. Recognises "default" (algorithm=0, == bare
// bitcomp) and "sparse" (algorithm=1). Returns false on anything
// else so the caller can surface a clear error.
bool parse_bitcomp_suffix(std::string_view suffix, int* algorithm)
{
  if (suffix == "default") {
    *algorithm = 0;
    return true;
  }
  if (suffix == "sparse") {
    *algorithm = 1;
    return true;
  }
  return false;
}

std::vector<compressible_output> bitcomp_compressed_representation::named_channels() const
{
  if (!serialized_output) {
    // Use a private non-blocking stream: the producing stream is always synced
    // before named_channels() is called, so compressed_data is already coherent.
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    rmm::cuda_stream_view sv(s);
    auto mr  = rmm::mr::get_current_device_resource_ref();
    auto out = cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::UINT8),
                                             static_cast<cudf::size_type>(compressed_size),
                                             cudf::mask_state::UNALLOCATED,
                                             sv,
                                             mr);
    if (compressed_size > 0 && compressed_data) {
      cudaMemcpyAsync(out->mutable_view().head<uint8_t>(),
                      compressed_data->data(),
                      compressed_size,
                      cudaMemcpyDeviceToDevice,
                      s);
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    serialized_output = std::move(out);
  }
  return {{"output", serialized_output->view()}};
}

std::unique_ptr<cudf::column> bitcomp_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (num_rows == 0 || compressed_data == nullptr || compressed_size == 0) {
    return cudf::make_fixed_width_column(
      original_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  auto* mgr = get_bitcomp_manager(stream.value(), compress_algorithm);

  auto const* comp_ptr = static_cast<uint8_t const*>(compressed_data->data());
  auto dconfig         = mgr->configure_decompression(comp_ptr);

  auto out_col = cudf::make_fixed_width_column(
    original_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_ptr = static_cast<uint8_t*>(out_col->mutable_view().head<void>());

  mgr->decompress(out_ptr, comp_ptr, dconfig);
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) return nullptr;
  return out_col;
}

std::unique_ptr<compressed_representation> bitcomp_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const dt = column_to_compress.type();
  auto const n  = column_to_compress.size();

  if (n == 0) {
    return std::make_unique<bitcomp_compressed_representation>(
      dt,
      0,
      std::make_unique<rmm::device_buffer>(0, stream, mr),
      /*comp_size=*/0,
      /*uncomp_size=*/0,
      algorithm_);
  }
  if (!cudf::is_fixed_width(dt)) {
    throw std::runtime_error("bitcomp_compressor: only fixed-width columns supported");
  }

  size_t const uncompressed_size = static_cast<size_t>(n) * cudf::size_of(dt);

  auto* mgr = get_bitcomp_manager(stream.value(), algorithm_);

  auto cconfig = mgr->configure_compression(uncompressed_size);
  auto comp_buf =
    std::make_unique<rmm::device_buffer>(cconfig.max_compressed_buffer_size, stream, mr);

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
    throw std::runtime_error("bitcomp_compressor: D2H size copy failed");
  }
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    throw std::runtime_error("bitcomp_compressor: stream sync failed");
  }

  return std::make_unique<bitcomp_compressed_representation>(
    dt, n, std::move(comp_buf), actual_size, uncompressed_size, algorithm_);
}

std::unique_ptr<cudf::column> bitcomp_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* bc_repr = dynamic_cast<bitcomp_compressed_representation const*>(&data_to_decompress);
  if (bc_repr == nullptr) return nullptr;
  return bc_repr->decompress(stream, mr);
}

}  // namespace simpatico
