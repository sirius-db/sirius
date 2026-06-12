// SPDX-License-Identifier: Apache-2.0
// Shared compress / decompress implementation for simple nvcomp Manager codecs
// (Snappy, LZ4, GDeflate).
//
// Each codec's .cu file:
//   1. Provides a thread-local Manager cache and a `get_*_manager(stream)` fn.
//   2. Calls `nvcomp_compress_impl` / `nvcomp_decompress_impl` passing that fn.
//
// The `nvcomp_simple_rep_base` template in representation.hpp supplies
// `named_channels()` inline (identical for all codecs — lazy D2D copy to a
// UINT8 column) so there is nothing to share there.

#pragma once

#include "codegen/plan/representation.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace simpatico {
namespace detail {

// Compress a fixed-width column through `*mgr`.  Returns the worst-case
// allocated device_buffer and the actual compressed byte count.
// Throws std::runtime_error on CUDA failures.
template <typename MgrT>
std::pair<std::unique_ptr<rmm::device_buffer>, size_t> nvcomp_compress_impl(
  MgrT* (*get_mgr)(cudaStream_t),
  cudf::column_view const& col,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (!cudf::is_fixed_width(col.type())) {
    throw std::runtime_error("nvcomp simple compressor: only fixed-width columns supported");
  }
  size_t const uncompressed_size =
    static_cast<size_t>(col.size()) * static_cast<size_t>(cudf::size_of(col.type()));

  auto* mgr    = get_mgr(stream.value());
  auto cconfig = mgr->configure_compression(uncompressed_size);
  auto comp_buf =
    std::make_unique<rmm::device_buffer>(cconfig.max_compressed_buffer_size, stream, mr);

  rmm::device_buffer comp_size_dev(sizeof(size_t), stream, mr);

  mgr->compress(static_cast<uint8_t const*>(col.head<void>()),
                static_cast<uint8_t*>(comp_buf->data()),
                cconfig,
                static_cast<size_t*>(comp_size_dev.data()));

  size_t actual_size = 0;
  if (cudaMemcpyAsync(&actual_size,
                      comp_size_dev.data(),
                      sizeof(size_t),
                      cudaMemcpyDeviceToHost,
                      stream.value()) != cudaSuccess) {
    throw std::runtime_error("nvcomp simple compressor: D2H size copy failed");
  }
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    throw std::runtime_error("nvcomp simple compressor: stream sync failed");
  }
  return {std::move(comp_buf), actual_size};
}

// Decompress a payload produced by `nvcomp_compress_impl` back to a column.
template <typename MgrT>
std::unique_ptr<cudf::column> nvcomp_decompress_impl(MgrT* (*get_mgr)(cudaStream_t),
                                                     rmm::device_buffer const* compressed_data,
                                                     size_t compressed_size,
                                                     cudf::data_type orig_type,
                                                     cudf::size_type n_rows,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  if (n_rows == 0 || compressed_data == nullptr || compressed_size == 0) {
    return cudf::make_fixed_width_column(
      orig_type, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  auto* mgr            = get_mgr(stream.value());
  auto const* comp_ptr = static_cast<uint8_t const*>(compressed_data->data());
  auto dconfig         = mgr->configure_decompression(comp_ptr);

  auto out_col =
    cudf::make_fixed_width_column(orig_type, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  mgr->decompress(static_cast<uint8_t*>(out_col->mutable_view().head<void>()), comp_ptr, dconfig);

  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) return nullptr;
  return out_col;
}

}  // namespace detail
}  // namespace simpatico
