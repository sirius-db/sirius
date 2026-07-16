// SPDX-License-Identifier: Apache-2.0
// nvcomp LZ4 compressor (low-level batched API).
// Compress/decompress bodies are shared via nvcomp_simple_compressor.hpp; the
// self-describing frame format and all device allocations are handled by
// nvcomp_batched_codec.{hpp,cu}. We always use NVCOMP_TYPE_CHAR so data is
// treated as an opaque byte stream.

#include "nvcomp_batched_codec.hpp"
#include "nvcomp_simple_compressor.hpp"

#include <nvcomp/lz4.h>

namespace simpatico {

namespace {

constexpr size_t kLz4ChunkSize = 64 * 1024;

// Build the codec ops once. Pure functions of their args (opts baked in), so a
// single shared instance is safe across threads and calls.
detail::batched_codec_ops const& lz4_ops()
{
  static detail::batched_codec_ops const ops = [] {
    nvcompBatchedLZ4CompressOpts_t copts   = nvcompBatchedLZ4CompressDefaultOpts;
    copts.data_type                        = NVCOMP_TYPE_CHAR;
    nvcompBatchedLZ4DecompressOpts_t dopts = nvcompBatchedLZ4DecompressDefaultOpts;

    detail::batched_codec_ops o;
    o.chunk_size             = kLz4ChunkSize;
    o.compress_get_temp_size = [copts](size_t nc, size_t mc, size_t* tb, size_t mt) {
      return nvcompBatchedLZ4CompressGetTempSizeAsync(nc, mc, copts, tb, mt);
    };
    o.compress_get_max_output = [copts](size_t mc, size_t* mo) {
      return nvcompBatchedLZ4CompressGetMaxOutputChunkSize(mc, copts, mo);
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
      return nvcompBatchedLZ4CompressAsync(up, ub, mc, nc, t, tb, cp, cb, copts, st, s);
    };
    o.decompress_get_temp_size = [dopts](size_t nc, size_t mc, size_t* tb, size_t mt) {
      return nvcompBatchedLZ4DecompressGetTempSizeAsync(nc, mc, dopts, tb, mt);
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
      return nvcompBatchedLZ4DecompressAsync(cp, cb, ubuf, actual, nc, t, tb, up, dopts, st, s);
    };
    return o;
  }();
  return ops;
}

}  // namespace

std::unique_ptr<cudf::column> lz4_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return detail::nvcomp_decompress_impl(
    lz4_ops(), payload_data(), payload_size(), original_type, num_rows, stream, mr);
}

std::unique_ptr<compressed_representation> lz4_compressor::compress(
  cudf::column_view col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const dt = col.type();
  auto const n  = col.size();
  if (n == 0) {
    return std::make_unique<lz4_compressed_representation>(
      dt, 0, std::make_unique<rmm::device_buffer>(0, stream, mr), 0, 0);
  }
  auto [buf, actual]  = detail::nvcomp_compress_impl(lz4_ops(), col, stream, mr);
  size_t const uncomp = static_cast<size_t>(n) * cudf::size_of(dt);
  return std::make_unique<lz4_compressed_representation>(dt, n, std::move(buf), actual, uncomp);
}

std::unique_ptr<cudf::column> lz4_compressor::decompress(compressed_representation const& rep,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto const* r = dynamic_cast<lz4_compressed_representation const*>(&rep);
  if (!r) return nullptr;
  return r->decompress(stream, mr);
}

namespace detail {
}  // namespace detail

}  // namespace simpatico
