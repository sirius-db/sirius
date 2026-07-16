// SPDX-License-Identifier: Apache-2.0
// nvcomp Bitcomp standalone compressor.
//
// Mirror of ans_compressor.cu using the nvcomp low-level batched Bitcomp API via
// a `batched_codec_ops` bundle (all device memory owned by RMM). Bitcomp targets
// numeric data with high zero-density / low-magnitude residues — a natural fit
// downstream of delta or RLE chains.
//
// DSL surface:
//   `input -> bitcomp`           algorithm=0 (default, best ratio).
//   `input -> bitcomp_default`   alias for bitcomp.
//   `input -> bitcomp_sparse`    algorithm=1 — faster on zero-rich data.
//
// The compress-time algorithm is stashed on the rep so decompress rebuilds the
// matching ops.

#include "codegen/plan/representation.hpp"
#include "nvcomp_batched_codec.hpp"
#include "nvcomp_simple_compressor.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <nvcomp/bitcomp.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace simpatico {

namespace {
constexpr size_t kBitcompManagerChunkSize = 64 * 1024;

// Build the batched ops for a bitcomp algorithm variant. Cheap to build, so we
// construct per call rather than caching.
detail::batched_codec_ops make_bitcomp_ops(int algorithm)
{
  nvcompBatchedBitcompCompressOpts_t copts   = nvcompBatchedBitcompCompressDefaultOpts;
  copts.algorithm                            = algorithm;
  nvcompBatchedBitcompDecompressOpts_t dopts = nvcompBatchedBitcompDecompressDefaultOpts;

  detail::batched_codec_ops o;
  o.chunk_size             = kBitcompManagerChunkSize;
  o.compress_get_temp_size = [copts](size_t nc, size_t mc, size_t* tb, size_t mt) {
    return nvcompBatchedBitcompCompressGetTempSizeAsync(nc, mc, copts, tb, mt);
  };
  o.compress_get_max_output = [copts](size_t mc, size_t* mo) {
    return nvcompBatchedBitcompCompressGetMaxOutputChunkSize(mc, copts, mo);
  };
  o.compress_async = [copts](void const* const* up,
                             size_t const* ub,
                             size_t mc,
                             size_t nc,
                             void* t,
                             size_t tb,
                             void* const* cp,
                             size_t* cb,
                             nvcompStatus_t* st,
                             cudaStream_t s) {
    return nvcompBatchedBitcompCompressAsync(up, ub, mc, nc, t, tb, cp, cb, copts, st, s);
  };
  o.decompress_get_temp_size = [dopts](size_t nc, size_t mc, size_t* tb, size_t mt) {
    return nvcompBatchedBitcompDecompressGetTempSizeAsync(nc, mc, dopts, tb, mt);
  };
  o.decompress_async = [dopts](void const* const* cp,
                               size_t const* cb,
                               size_t const* ubuf,
                               size_t* actual,
                               size_t nc,
                               void* t,
                               size_t tb,
                               void* const* up,
                               nvcompStatus_t* st,
                               cudaStream_t s) {
    return nvcompBatchedBitcompDecompressAsync(cp, cb, ubuf, actual, nc, t, tb, up, dopts, st, s);
  };
  return o;
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

std::unique_ptr<cudf::column> bitcomp_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return detail::nvcomp_decompress_impl(make_bitcomp_ops(compress_algorithm),
                                        payload_data(),
                                        payload_size(),
                                        original_type,
                                        num_rows,
                                        stream,
                                        mr);
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
    throw std::runtime_error(
      "bitcomp_compressor: only fixed-width columns supported; use str_split");
  }

  size_t const uncompressed_size = static_cast<size_t>(n) * cudf::size_of(dt);

  auto [comp_buf, actual_size] =
    detail::nvcomp_compress_impl(make_bitcomp_ops(algorithm_), column_to_compress, stream, mr);

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

namespace detail {
}  // namespace detail

}  // namespace simpatico
