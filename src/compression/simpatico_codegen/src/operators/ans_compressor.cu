// SPDX-License-Identifier: Apache-2.0
// nvcomp ANS (entropy coder) compressor.
//
// Uses the nvcomp low-level batched API via a `batched_codec_ops` bundle (see
// nvcomp_batched_codec.{hpp,cu}); all device memory is owned by RMM so nvcomp
// never allocates its own. Handles fixed-width columns and, via
// nvcomp_string_support.hpp, STRING columns (offsets + chars sub-streams).

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

#include <nvcomp/ans.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace simpatico {

namespace {
// Internal chunk size used by the Manager. 64 KB matches the LZ4/Snappy
// sweet spot — big enough to amortize per-chunk overhead, small enough
// for parallelism. ANS supports up to 16 MB chunks (`nvcompANSCompressionMaxAllowedChunkSize`).
constexpr size_t kAnsManagerChunkSize = 64 * 1024;

detail::batched_codec_ops const& ans_ops()
{
  static detail::batched_codec_ops const ops = [] {
    nvcompBatchedANSCompressOpts_t copts   = nvcompBatchedANSCompressDefaultOpts;
    nvcompBatchedANSDecompressOpts_t dopts = nvcompBatchedANSDecompressDefaultOpts;

    detail::batched_codec_ops o;
    o.chunk_size             = kAnsManagerChunkSize;
    o.compress_get_temp_size = [copts](size_t nc, size_t mc, size_t* tb, size_t mt) {
      return nvcompBatchedANSCompressGetTempSizeAsync(nc, mc, copts, tb, mt);
    };
    o.compress_get_max_output = [copts](size_t mc, size_t* mo) {
      return nvcompBatchedANSCompressGetMaxOutputChunkSize(mc, copts, mo);
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
      return nvcompBatchedANSCompressAsync(up, ub, mc, nc, t, tb, cp, cb, copts, st, s);
    };
    o.decompress_get_temp_size = [dopts](size_t nc, size_t mc, size_t* tb, size_t mt) {
      return nvcompBatchedANSDecompressGetTempSizeAsync(nc, mc, dopts, tb, mt);
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
      return nvcompBatchedANSDecompressAsync(cp, cb, ubuf, actual, nc, t, tb, up, dopts, st, s);
    };
    return o;
  }();
  return ops;
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
      [&](void const* comp, std::size_t comp_size, void* out, std::size_t out_bytes) {
        detail::nvcomp_decompress_bytes(ans_ops(), comp, comp_size, out, out_bytes, stream, mr);
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

  return detail::nvcomp_decompress_impl(
    ans_ops(), compressed_data.get(), compressed_size, original_type, num_rows, stream, mr);
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
    auto compress_bytes = [&](void const* ptr, std::size_t bytes) {
      return detail::nvcomp_compress_bytes(ans_ops(), ptr, bytes, stream, mr);
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

  auto [comp_buf, actual_size] =
    detail::nvcomp_compress_impl(ans_ops(), column_to_compress, stream, mr);

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
void release_ans_manager_scratch() {}  // no cached manager under the batched API
}  // namespace detail

}  // namespace simpatico
