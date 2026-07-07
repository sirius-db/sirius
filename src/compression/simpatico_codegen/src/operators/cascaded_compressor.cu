// SPDX-License-Identifier: Apache-2.0
// nvcomp Cascaded compressor.
//
// Uses the nvcomp high-level Manager API (`nvcomp::CascadedManager`), which
// handles batching internally. The Manager is cached thread-locally keyed by
// `(stream, chunk_size, num_deltas, num_RLEs, use_bp, nvcompType_t)` to avoid
// the per-call constructor cost.
//
// DSL surface:
//   `input -> nvcomp_cascaded`               nvcomp defaults
//                                            (num_deltas=1, num_RLEs=2, use_bp=1).
//   `input -> nvcomp_cascaded_<N>D<M>R<K>B`  explicit opts: N deltas, M RLEs,
//                                            K bitpack (0 or 1).
//
// `nvcompType_t` is selected per cuDF dtype so the integer representation
// matches what nvcomp expects (e.g. delta over int32 needs NVCOMP_TYPE_INT).

#include "codegen/plan/representation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <nvcomp/cascaded.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace simpatico {

namespace {

constexpr size_t kCascadedManagerChunkSize = 64 * 1024;

nvcompType_t cascaded_type_for(cudf::data_type t)
{
  switch (t.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8: return NVCOMP_TYPE_CHAR;
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16: return NVCOMP_TYPE_SHORT;
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32: return NVCOMP_TYPE_INT;
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT64: return NVCOMP_TYPE_LONGLONG;
    case cudf::type_id::FLOAT32: return NVCOMP_TYPE_INT;
    // Chrono types at their integer storage width.
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS: return NVCOMP_TYPE_INT;
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS: return NVCOMP_TYPE_LONGLONG;
    default: return NVCOMP_TYPE_INT;
  }
}

struct cascaded_manager_cache {
  std::unique_ptr<nvcomp::CascadedManager> mgr;
  cudaStream_t stream = nullptr;
  size_t chunk_size   = 0;
  int num_RLEs        = -1;
  int num_deltas      = -1;
  int use_bp          = -1;
  nvcompType_t type   = NVCOMP_TYPE_INT;
};

static thread_local cascaded_manager_cache tls_cascaded_mgr;

nvcomp::CascadedManager* get_cascaded_manager(cudaStream_t s,
                                              nvcompBatchedCascadedCompressOpts_t const& copts)
{
  if (tls_cascaded_mgr.mgr && tls_cascaded_mgr.stream == s &&
      tls_cascaded_mgr.chunk_size == kCascadedManagerChunkSize &&
      tls_cascaded_mgr.num_RLEs == copts.num_RLEs &&
      tls_cascaded_mgr.num_deltas == copts.num_deltas && tls_cascaded_mgr.use_bp == copts.use_bp &&
      tls_cascaded_mgr.type == copts.type) {
    return tls_cascaded_mgr.mgr.get();
  }
  tls_cascaded_mgr.mgr = std::make_unique<nvcomp::CascadedManager>(
    kCascadedManagerChunkSize, copts, nvcompBatchedCascadedDecompressDefaultOpts, s);
  tls_cascaded_mgr.stream     = s;
  tls_cascaded_mgr.chunk_size = kCascadedManagerChunkSize;
  tls_cascaded_mgr.num_RLEs   = copts.num_RLEs;
  tls_cascaded_mgr.num_deltas = copts.num_deltas;
  tls_cascaded_mgr.use_bp     = copts.use_bp;
  tls_cascaded_mgr.type       = copts.type;
  return tls_cascaded_mgr.mgr.get();
}

}  // namespace

// Parse a suffix of the form "<N>D<M>R<K>B" into (deltas, rles, bp).
// Returns true on success. Single decimal digit per field. Empty suffix
// means "use nvcomp defaults" → returns false (caller falls back).
bool parse_nvcomp_cascaded_suffix(std::string_view suffix, int* deltas, int* rles, int* bp)
{
  if (suffix.empty()) return false;
  size_t i        = 0;
  auto read_field = [&](char letter, int* out) -> bool {
    if (i >= suffix.size() || suffix[i] < '0' || suffix[i] > '9') return false;
    *out = suffix[i] - '0';
    ++i;
    if (i >= suffix.size() || suffix[i] != letter) return false;
    ++i;
    return true;
  };
  if (!read_field('D', deltas)) return false;
  if (!read_field('R', rles)) return false;
  if (!read_field('B', bp)) return false;
  if (i != suffix.size()) return false;
  if (*bp != 0 && *bp != 1) return false;
  return true;
}

std::vector<compressible_output> cascaded_compressed_representation::named_channels(
  rmm::cuda_stream_view stream) const
{
  if (!serialized_output) {
    // The resulting column's device_buffer remembers ``stream`` for its own
    // eventual deallocation, so the caller-supplied stream must stay valid
    // for the column's lifetime (a private stream destroyed at the end of
    // this scope would leave a dangling handle).
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

std::unique_ptr<cudf::column> cascaded_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (num_rows == 0 || compressed_data == nullptr || compressed_size == 0) {
    return cudf::make_fixed_width_column(
      original_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  nvcompBatchedCascadedCompressOpts_t copts = nvcompBatchedCascadedCompressDefaultOpts;
  copts.type                                = cascaded_type_for(original_type);
  copts.num_deltas                          = compress_num_deltas;
  copts.num_RLEs                            = compress_num_RLEs;
  copts.use_bp                              = compress_use_bp;

  auto* mgr = get_cascaded_manager(stream.value(), copts);

  auto const* comp_ptr = static_cast<uint8_t const*>(compressed_data->data());
  auto dconfig         = mgr->configure_decompression(comp_ptr);

  auto out_col = cudf::make_fixed_width_column(
    original_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_ptr = static_cast<uint8_t*>(out_col->mutable_view().head<void>());

  mgr->decompress(out_ptr, comp_ptr, dconfig);
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) return nullptr;
  return out_col;
}

std::unique_ptr<compressed_representation> cascaded_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const dt = column_to_compress.type();
  auto const n  = column_to_compress.size();

  if (n == 0) {
    return std::make_unique<cascaded_compressed_representation>(
      dt,
      0,
      std::make_unique<rmm::device_buffer>(0, stream, mr),
      /*comp_size=*/0,
      /*uncomp_size=*/0,
      num_deltas_,
      num_RLEs_,
      use_bp_);
  }
  if (!cudf::is_fixed_width(dt)) {
    throw std::runtime_error("nvcomp_cascaded: only fixed-width columns supported");
  }

  size_t const uncompressed_size = static_cast<size_t>(n) * cudf::size_of(dt);

  nvcompBatchedCascadedCompressOpts_t copts = nvcompBatchedCascadedCompressDefaultOpts;
  copts.type                                = cascaded_type_for(dt);
  copts.num_deltas                          = num_deltas_;
  copts.num_RLEs                            = num_RLEs_;
  copts.use_bp                              = use_bp_;

  auto* mgr = get_cascaded_manager(stream.value(), copts);

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
    throw std::runtime_error("nvcomp_cascaded: D2H size copy failed");
  }
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    throw std::runtime_error("nvcomp_cascaded: stream sync failed");
  }

  return std::make_unique<cascaded_compressed_representation>(
    dt, n, std::move(comp_buf), actual_size, uncompressed_size, num_deltas_, num_RLEs_, use_bp_);
}

std::unique_ptr<cudf::column> cascaded_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* repr = dynamic_cast<cascaded_compressed_representation const*>(&data_to_decompress);
  if (repr == nullptr) return nullptr;
  return repr->decompress(stream, mr);
}

namespace detail {
void release_cascaded_manager_scratch()
{
  if (tls_cascaded_mgr.mgr) tls_cascaded_mgr.mgr->deallocate_gpu_mem();
}
}  // namespace detail

}  // namespace simpatico
