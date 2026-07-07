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
#include "nvcomp_string_support.hpp"

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

std::vector<compressible_output> bitcomp_compressed_representation::named_channels(
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

std::unique_ptr<cudf::column> bitcomp_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  // STRING: split the payload into the offsets and chars codec streams,
  // decompress each, and rebuild the strings column.
  if (original_type.id() == cudf::type_id::STRING) {
    auto ops = make_bitcomp_ops(compress_algorithm);
    auto decompress_bytes =
      [&](void const* comp, std::size_t comp_size, void* out, std::size_t out_bytes) {
        detail::nvcomp_decompress_bytes(ops, comp, comp_size, out, out_bytes, stream, mr);
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

  return detail::nvcomp_decompress_impl(make_bitcomp_ops(compress_algorithm),
                                        compressed_data.get(),
                                        compressed_size,
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
  // STRING: compress the offsets and chars streams independently with bitcomp
  // and store them concatenated in the rep's payload.
  if (dt.id() == cudf::type_id::STRING) {
    auto ops            = make_bitcomp_ops(algorithm_);
    auto compress_bytes = [&](void const* ptr, std::size_t bytes) {
      return detail::nvcomp_compress_bytes(ops, ptr, bytes, stream, mr);
    };

    auto cs  = detail::compress_string_column(column_to_compress, compress_bytes, stream, mr);
    auto rep = std::make_unique<bitcomp_compressed_representation>(dt,
                                                                   cs.num_rows,
                                                                   std::move(cs.payload),
                                                                   cs.total_compressed_size,
                                                                   cs.chars_uncompressed_size,
                                                                   algorithm_);
    rep->offsets_compressed_size   = cs.offsets_compressed_size;
    rep->offsets_uncompressed_size = cs.offsets_uncompressed_size;
    rep->offsets_type              = cs.offsets_type;
    return rep;
  }

  if (!cudf::is_fixed_width(dt)) {
    throw std::runtime_error("bitcomp_compressor: only fixed-width columns supported");
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
void release_bitcomp_manager_scratch() {}  // no cached manager under the batched API
}  // namespace detail

}  // namespace simpatico
