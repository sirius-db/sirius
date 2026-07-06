// SPDX-License-Identifier: Apache-2.0
// nvcomp ANS (entropy coder) compressor.
//
// Uses the nvcomp high-level Manager API (`nvcomp::ANSManager`), which handles
// batching internally. The Manager is cached thread-locally keyed by
// `(chunk_size, stream)` to avoid the per-call constructor cost. nvcomp requires
// the Manager not outlive its `user_stream`, so the cache invalidates on stream
// change.

#include "codegen/plan/representation.hpp"
#include "nvcomp_string_support.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

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

std::vector<compressible_output> ans_compressed_representation::named_channels(
  rmm::cuda_stream_view stream) const
{
  if (!serialized_output) {
    // The resulting column's device_buffer remembers ``stream`` for its own
    // eventual deallocation, so the caller-supplied stream must stay valid
    // for the column's lifetime (a private stream destroyed at the end of
    // this scope would leave a dangling handle). The producing stream is
    // always synced before named_channels() is called, so compressed_data
    // is already coherent.
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
    cudaStreamSynchronize(stream.value());
    serialized_output = std::move(out);
  }
  return {{"output", serialized_output->view()}};
}

std::unique_ptr<cudf::column> ans_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  // STRING: split the payload into the offsets and chars codec streams,
  // decompress each, and rebuild the strings column.
  if (original_type.id() == cudf::type_id::STRING) {
    auto decompress_bytes =
      [&](void const* comp, std::size_t /*comp_size*/, void* out, std::size_t /*out_bytes*/) {
        auto* mgr    = get_ans_manager(stream.value());
        auto dconfig = mgr->configure_decompression(static_cast<uint8_t const*>(comp));
        mgr->decompress(static_cast<uint8_t*>(out), static_cast<uint8_t const*>(comp), dconfig);
      };
    auto col = detail::rebuild_string_column(compressed_data ? compressed_data->data() : nullptr,
                                             compressed_size,
                                             offsets_compressed_size,
                                             offsets_uncompressed_size,
                                             uncompressed_size,
                                             offsets_type,
                                             num_rows,
                                             decompress_bytes,
                                             stream,
                                             mr);
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) return nullptr;
    return col;
  }

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
  // STRING: compress the offsets and chars streams independently with ANS and
  // store them concatenated in the rep's payload.
  if (dt.id() == cudf::type_id::STRING) {
    auto compress_bytes =
      [&](void const* ptr,
          std::size_t bytes) -> std::pair<std::unique_ptr<rmm::device_buffer>, std::size_t> {
      auto* mgr    = get_ans_manager(stream.value());
      auto cconfig = mgr->configure_compression(bytes);
      auto buf =
        std::make_unique<rmm::device_buffer>(cconfig.max_compressed_buffer_size, stream, mr);
      rmm::device_buffer size_dev(sizeof(size_t), stream, mr);
      mgr->compress(static_cast<uint8_t const*>(ptr),
                    static_cast<uint8_t*>(buf->data()),
                    cconfig,
                    static_cast<size_t*>(size_dev.data()));
      size_t actual = 0;
      cudaMemcpyAsync(
        &actual, size_dev.data(), sizeof(size_t), cudaMemcpyDeviceToHost, stream.value());
      cudaStreamSynchronize(stream.value());
      return {std::move(buf), actual};
    };

    auto cs  = detail::compress_string_column(column_to_compress, compress_bytes, stream, mr);
    auto rep = std::make_unique<ans_compressed_representation>(
      dt, cs.num_rows, std::move(cs.payload), cs.total_compressed_size, cs.chars_uncompressed_size);
    rep->offsets_compressed_size   = cs.offsets_compressed_size;
    rep->offsets_uncompressed_size = cs.offsets_uncompressed_size;
    rep->offsets_type              = cs.offsets_type;
    return rep;
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

namespace detail {
void release_ans_manager_scratch()
{
  if (tls_ans_mgr.mgr) tls_ans_mgr.mgr->deallocate_gpu_mem();
}
}  // namespace detail

}  // namespace simpatico
