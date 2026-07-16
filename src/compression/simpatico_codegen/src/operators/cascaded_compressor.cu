// SPDX-License-Identifier: Apache-2.0
// nvcomp Cascaded compressor.
//
// Uses the nvcomp low-level batched Cascaded API via a `batched_codec_ops`
// bundle (see nvcomp_batched_codec.{hpp,cu}); all device memory is owned by RMM.
// Ops are built per call because Cascaded's opts depend on the column dtype and
// the plan's (deltas, RLEs, bp) knobs.
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

#include <nvcomp/cascaded.h>

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

// Build the batched ops for a specific Cascaded configuration. Cascaded's opts
// depend on the column dtype and the plan's (deltas, RLEs, bp) knobs, so ops are
// built per call rather than cached in a static.
detail::batched_codec_ops make_cascaded_ops(nvcompBatchedCascadedCompressOpts_t copts)
{
  nvcompBatchedCascadedDecompressOpts_t dopts = nvcompBatchedCascadedDecompressDefaultOpts;

  detail::batched_codec_ops o;
  o.chunk_size             = kCascadedManagerChunkSize;
  o.compress_get_temp_size = [copts](size_t nc, size_t mc, size_t* tb, size_t mt) {
    return nvcompBatchedCascadedCompressGetTempSizeAsync(nc, mc, copts, tb, mt);
  };
  o.compress_get_max_output = [copts](size_t mc, size_t* mo) {
    return nvcompBatchedCascadedCompressGetMaxOutputChunkSize(mc, copts, mo);
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
    return nvcompBatchedCascadedCompressAsync(up, ub, mc, nc, t, tb, cp, cb, copts, st, s);
  };
  o.decompress_get_temp_size = [dopts](size_t nc, size_t mc, size_t* tb, size_t mt) {
    return nvcompBatchedCascadedDecompressGetTempSizeAsync(nc, mc, dopts, tb, mt);
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
    return nvcompBatchedCascadedDecompressAsync(cp, cb, ubuf, actual, nc, t, tb, up, dopts, st, s);
  };
  return o;
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

std::unique_ptr<cudf::column> cascaded_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (num_rows == 0 || payload_data() == nullptr || payload_size() == 0) {
    return cudf::make_fixed_width_column(
      original_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  nvcompBatchedCascadedCompressOpts_t copts = nvcompBatchedCascadedCompressDefaultOpts;
  copts.type                                = cascaded_type_for(original_type);
  copts.num_deltas                          = compress_num_deltas;
  copts.num_RLEs                            = compress_num_RLEs;
  copts.use_bp                              = compress_use_bp;

  return detail::nvcomp_decompress_impl(
    make_cascaded_ops(copts), payload_data(), payload_size(), original_type, num_rows, stream, mr);
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

  auto [comp_buf, actual_size] =
    detail::nvcomp_compress_impl(make_cascaded_ops(copts), column_to_compress, stream, mr);

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
}  // namespace detail

}  // namespace simpatico
