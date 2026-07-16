// SPDX-License-Identifier: Apache-2.0
// nvcomp Snappy compressor (low-level batched API).
// Compress/decompress bodies are shared via nvcomp_simple_compressor.hpp; the
// frame format and all device allocations live in nvcomp_batched_codec.{hpp,cu}.

#include "nvcomp_batched_codec.hpp"
#include "nvcomp_simple_compressor.hpp"

#include <nvcomp/snappy.h>

namespace simpatico {

namespace {

constexpr size_t kSnappyChunkSize = 64 * 1024;

detail::batched_codec_ops const& snappy_ops()
{
  static detail::batched_codec_ops const ops = [] {
    nvcompBatchedSnappyCompressOpts_t copts   = nvcompBatchedSnappyCompressDefaultOpts;
    nvcompBatchedSnappyDecompressOpts_t dopts = nvcompBatchedSnappyDecompressDefaultOpts;

    detail::batched_codec_ops o;
    o.chunk_size             = kSnappyChunkSize;
    o.compress_get_temp_size = [copts](size_t nc, size_t mc, size_t* tb, size_t mt) {
      return nvcompBatchedSnappyCompressGetTempSizeAsync(nc, mc, copts, tb, mt);
    };
    o.compress_get_max_output = [copts](size_t mc, size_t* mo) {
      return nvcompBatchedSnappyCompressGetMaxOutputChunkSize(mc, copts, mo);
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
      return nvcompBatchedSnappyCompressAsync(up, ub, mc, nc, t, tb, cp, cb, copts, st, s);
    };
    o.decompress_get_temp_size = [dopts](size_t nc, size_t mc, size_t* tb, size_t mt) {
      return nvcompBatchedSnappyDecompressGetTempSizeAsync(nc, mc, dopts, tb, mt);
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
      return nvcompBatchedSnappyDecompressAsync(cp, cb, ubuf, actual, nc, t, tb, up, dopts, st, s);
    };
    return o;
  }();
  return ops;
}

}  // namespace

std::unique_ptr<cudf::column> snappy_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return detail::nvcomp_decompress_impl(
    snappy_ops(), payload_data(), payload_size(), original_type, num_rows, stream, mr);
}

std::unique_ptr<compressed_representation> snappy_compressor::compress(
  cudf::column_view col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const dt = col.type();
  auto const n  = col.size();
  if (n == 0) {
    return std::make_unique<snappy_compressed_representation>(
      dt, 0, std::make_unique<rmm::device_buffer>(0, stream, mr), 0, 0);
  }
  auto [buf, actual]  = detail::nvcomp_compress_impl(snappy_ops(), col, stream, mr);
  size_t const uncomp = static_cast<size_t>(n) * cudf::size_of(dt);
  return std::make_unique<snappy_compressed_representation>(dt, n, std::move(buf), actual, uncomp);
}

std::unique_ptr<cudf::column> snappy_compressor::decompress(compressed_representation const& rep,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  auto const* r = dynamic_cast<snappy_compressed_representation const*>(&rep);
  if (!r) return nullptr;
  return r->decompress(stream, mr);
}

namespace detail {
}  // namespace detail

}  // namespace simpatico
