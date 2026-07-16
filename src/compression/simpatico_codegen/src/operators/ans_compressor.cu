// SPDX-License-Identifier: Apache-2.0
// nvcomp ANS (entropy coder) compressor.
//
// Uses the nvcomp low-level batched API via a `batched_codec_ops` bundle (see
// nvcomp_batched_codec.{hpp,cu}); all device memory is owned by RMM so nvcomp
// never allocates its own. Fixed-width columns only; STRING columns are handled
// upstream via str_split.

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

std::unique_ptr<cudf::column> ans_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return detail::nvcomp_decompress_impl(
    ans_ops(), payload_data(), payload_size(), original_type, num_rows, stream, mr);
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
    throw std::runtime_error("ans_compressor: only fixed-width columns supported; use str_split");
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
}  // namespace detail

}  // namespace simpatico
