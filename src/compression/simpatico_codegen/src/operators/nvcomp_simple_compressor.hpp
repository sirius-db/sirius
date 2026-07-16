// SPDX-License-Identifier: Apache-2.0
// Shared compress / decompress helpers for the byte-stream nvcomp codecs
// (LZ4, Snappy, GDeflate, Cascaded, ANS, bitcomp).
//
// All codecs drive nvcomp through the low-level *batched* API via a
// `batched_codec_ops` bundle (see nvcomp_batched_codec.hpp). That API leaves
// every device allocation to us (RMM), so nvcomp never allocates its own memory
// and an out-of-memory condition throws cleanly instead of faulting a kernel and
// corrupting the CUDA context. Each codec's .cu file builds its ops once and
// calls the fixed-width helpers here (and, for STRING-capable codecs, the
// primitives in nvcomp_string_support.hpp).

#pragma once

#include "codegen/plan/representation.hpp"
#include "nvcomp_batched_codec.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace simpatico {
namespace detail {

// Compress a fixed-width column through `ops`. Returns the compressed frame and
// its actual byte count.
inline std::pair<std::unique_ptr<rmm::device_buffer>, std::size_t> nvcomp_compress_impl(
  batched_codec_ops const& ops,
  cudf::column_view const& col,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (!cudf::is_fixed_width(col.type())) {
    throw std::runtime_error("nvcomp simple compressor: only fixed-width columns supported");
  }
  std::size_t const uncompressed_size =
    static_cast<std::size_t>(col.size()) * static_cast<std::size_t>(cudf::size_of(col.type()));

  return batched_compress_bytes(ops, col.head<void>(), uncompressed_size, stream, mr);
}

// Decompress a frame produced by nvcomp_compress_impl back to a fixed-width column.
inline std::unique_ptr<cudf::column> nvcomp_decompress_impl(batched_codec_ops const& ops,
                                                            const void* compressed_data,
                                                            std::size_t compressed_size,
                                                            cudf::data_type orig_type,
                                                            cudf::size_type n_rows,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  if (n_rows == 0 || compressed_data == nullptr || compressed_size == 0) {
    return cudf::make_fixed_width_column(
      orig_type, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  auto out_col =
    cudf::make_fixed_width_column(orig_type, n_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  std::size_t const out_bytes =
    static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(cudf::size_of(orig_type));
  batched_decompress_bytes(ops,
                           compressed_data,
                           compressed_size,
                           out_col->mutable_view().head<void>(),
                           out_bytes,
                           stream,
                           mr);
  return out_col;
}

}  // namespace detail
}  // namespace simpatico
