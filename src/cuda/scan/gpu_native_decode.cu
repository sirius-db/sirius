/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===----------------------------------------------------------------------===//
// GPU decode pipeline — entry point and codec dispatch.
//
// gpu_decode_table decodes one row-group slice. For each column it runs the
// data codec(s) into a values buffer and the validity codec(s) into a null
// mask, builds a cudf::column, then finishes the table with one batched
// null-count.
//
// A column's data is a sequence of runs, each carrying one codec.
// dispatch_data_run routes a run by codec:
//
//   UNCOMPRESSED   device-to-device copy                   (this file)
//   CONSTANT       broadcast one value to every row        (this file)
//   RLE            prefix-sum build + value scatter        (gpu_decode_rle.cu)
//   BITPACKING     bit-unpack per 2048-row group           (gpu_decode_bitpacking.cu)
//   ALP / ALPRD    bit-unpack + float reconstruction       (gpu_decode_alp.cu)
//   string codecs  two-phase length / scan / gather        (gpu_decode_strings.cu)
//
// The fixed-width codecs share one-CTA-per-block work partitioning
// (detail/decode_common.cuh) and the device primitives in detail/
// (bit_unpack, shared_staging, vectorized_store, byte_copy). Each on-disk
// layout is documented at the top of its codec file. Malformed metadata
// decodes to a zero-filled output instead of faulting.
//===----------------------------------------------------------------------===//

#include "cuda/scan/detail/vectorized_store.cuh"
#include "cuda/scan/gpu_decode_alp.cuh"
#include "cuda/scan/gpu_decode_bitpacking.cuh"
#include "cuda/scan/gpu_decode_rle.cuh"
#include "cuda/scan/gpu_native_decode.cuh"

#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::cuda::scan {

namespace {

/// Block dim used by every kernel in this file.
constexpr uint32_t BLOCK_DIM = 256;

/// Maximum bytes one block of `kernel_batched_memcpy` copies.
constexpr uint32_t COPY_CHUNK_BYTES = 64u << 10;

/// `cudf::size_type` is signed int32. Catch a `total_rows` overflow before
/// it silently wraps in the static_cast at column construction.
constexpr uint32_t MAX_ROWS_PER_COLUMN =
  static_cast<uint32_t>(std::numeric_limits<cudf::size_type>::max());

/// Throws if any segment's row range falls outside `[0, total_rows)`. Output
/// buffers are sized to `total_rows`, so an overshoot would write out of bounds.
void validate_segment_bounds(std::vector<gpu_codec_run> const& runs,
                             uint32_t total_rows,
                             char const* what)
{
  for (auto const& run : runs) {
    for (auto const& seg : run.segments) {
      size_t end = size_t{seg.row_offset} + size_t{seg.row_count};
      if (end > total_rows) {
        throw std::runtime_error(std::string("gpu_decode_table: ") + what +
                                 " segment row_offset (" + std::to_string(seg.row_offset) +
                                 ") + row_count (" + std::to_string(seg.row_count) +
                                 ") > total_rows (" + std::to_string(total_rows) + ")");
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// CONSTANT broadcast — write one device-resident value across the column.
//===----------------------------------------------------------------------===//

/// Writes `*src` to every position in `dest[0, rows)` via `detail::vec_fill`.
template <typename T>
__global__ void kernel_broadcast_constant(T* __restrict__ dest, T const* __restrict__ src, int rows)
{
  T const val      = *src;
  int const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int const stride = gridDim.x * blockDim.x;
  detail::vec_fill<T>(dest, rows, tid, stride, [val](int) { return val; });
}

/// `cudf::type_dispatcher` functor. Projects each cudf type to its on-device
/// storage type (`device_storage_type_t<T>`, e.g. decimal128 -> __int128_t) so
/// the broadcast stride matches the column's physical width.
struct broadcast_dispatch {
  template <typename T, std::enable_if_t<cudf::is_fixed_width<T>(), int> = 0>
  void operator()(uint8_t* d_dest,
                  uint8_t const* d_val_src,
                  uint32_t row_count,
                  rmm::cuda_stream_view stream) const
  {
    using StorageT = cudf::device_storage_type_t<T>;
    constexpr uint32_t TPV =
      (sizeof(StorageT) <= 8 && 16u % sizeof(StorageT) == 0) ? 16u / sizeof(StorageT) : 1u;
    uint32_t blocks = std::max(1u, (row_count / TPV + BLOCK_DIM - 1) / BLOCK_DIM);
    kernel_broadcast_constant<StorageT><<<blocks, BLOCK_DIM, 0, stream.value()>>>(
      reinterpret_cast<StorageT*>(d_dest), reinterpret_cast<StorageT const*>(d_val_src), row_count);
  }
  // Strings, lists, structs etc. should never reach here — the public entry
  // refuses non-fixed-width types up front. Defensive throw in case it does.
  template <typename T, std::enable_if_t<!cudf::is_fixed_width<T>(), int> = 0>
  void operator()(uint8_t*, uint8_t const*, uint32_t, rmm::cuda_stream_view) const
  {
    throw std::runtime_error(
      "gpu_decode_table: viability invariant violated — CONSTANT on non-fixed-width type");
  }
};

void launch_broadcast_constant(uint8_t* d_dest,
                               uint8_t const* d_val_src,
                               cudf::data_type type,
                               uint32_t row_count,
                               rmm::cuda_stream_view stream)
{
  cudf::type_dispatcher(type, broadcast_dispatch{}, d_dest, d_val_src, row_count, stream);
}

/// @todo [KEVIN]: consider batching to align execution and design here.
void decode_constant_data(gpu_codec_run const& run,
                          uint8_t* d_output,
                          cudf::data_type type,
                          uint32_t type_size,
                          rmm::cuda_stream_view stream)
{
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    if (seg.bytes_size < type_size) {
      throw std::runtime_error("gpu_decode_table: CONSTANT segment bytes_size (" +
                               std::to_string(seg.bytes_size) + ") < type_size (" +
                               std::to_string(type_size) + ")");
    }
    launch_broadcast_constant(
      d_output + size_t{seg.row_offset} * type_size, seg.d_bytes, type, seg.row_count, stream);
  }
}

//===----------------------------------------------------------------------===//
// UNCOMPRESSED batched copy.
//===----------------------------------------------------------------------===//

/// One unit of D→D copy work for `kernel_batched_memcpy`. A segment that
/// exceeds `COPY_CHUNK_BYTES` is split into multiple of these.
struct copy_chunk_desc {
  uint8_t const* src;
  uint8_t* dst;
  uint32_t bytes;
};

/// Each block copies one chunk, picking the widest aligned access the chunk's
/// pointers and length jointly support: 16-byte (int4), 4-byte, or per-byte.
__global__ void kernel_batched_memcpy(copy_chunk_desc const* __restrict__ chunks, uint32_t n_chunks)
{
  uint32_t cid = blockIdx.x;
  if (cid >= n_chunks) return;
  auto const c    = chunks[cid];
  uint32_t tid    = threadIdx.x;
  uint32_t stride = blockDim.x;

  // Common low bits across src, dst, and length determine the maximum
  // alignment we can use.
  uintptr_t lsb = reinterpret_cast<uintptr_t>(c.src) | reinterpret_cast<uintptr_t>(c.dst) | c.bytes;

  if ((lsb & 15u) == 0) {
    uint32_t v_count = c.bytes / 16;
    int4* dst4       = reinterpret_cast<int4*>(c.dst);
    int4 const* src4 = reinterpret_cast<int4 const*>(c.src);
    for (uint32_t i = tid; i < v_count; i += stride)
      dst4[i] = src4[i];
    return;
  }
  if ((lsb & 3u) == 0) {
    uint32_t w_count      = c.bytes / 4;
    uint32_t* dst32       = reinterpret_cast<uint32_t*>(c.dst);
    uint32_t const* src32 = reinterpret_cast<uint32_t const*>(c.src);
    for (uint32_t i = tid; i < w_count; i += stride)
      dst32[i] = src32[i];
    return;
  }
  for (uint32_t i = tid; i < c.bytes; i += stride)
    c.dst[i] = c.src[i];
}

/// Copies UNCOMPRESSED segments to the output. A single live segment uses a
/// direct `cudaMemcpyAsync`; multiple segments are split into `COPY_CHUNK_BYTES`
/// chunks and copied by one `kernel_batched_memcpy` launch.
void decode_uncompressed_data(gpu_codec_run const& run,
                              uint8_t* d_output,
                              uint32_t type_size,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  // Each live segment must own at least row_count * type_size bytes — the
  // memcpy below would otherwise read past the end of the source buffer.
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    size_t needed = size_t{seg.row_count} * type_size;
    if (size_t{seg.bytes_size} < needed) {
      throw std::runtime_error("gpu_decode_table: UNCOMPRESSED segment bytes_size (" +
                               std::to_string(seg.bytes_size) + ") < required " +
                               std::to_string(needed));
    }
  }

  // First-pass: see whether we have zero / one / many live (non-empty)
  // segments. Stops at the second hit so we don't walk the full vector when
  // we already know it's a multi-segment run.
  gpu_segment_desc const* live_seg = nullptr;
  uint32_t live_count              = 0;
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    live_seg = &seg;
    ++live_count;
    if (live_count > 1) break;
  }
  if (live_count == 0) return;

  if (live_count == 1) {
    RMM_CUDA_TRY(cudaMemcpyAsync(d_output + size_t{live_seg->row_offset} * type_size,
                                 live_seg->d_bytes,
                                 size_t{live_seg->row_count} * type_size,
                                 cudaMemcpyDeviceToDevice,
                                 stream.value()));
    return;
  }

  // Multi-segment path: split each segment into <= COPY_CHUNK_BYTES chunks
  // so per-block work stays uniform regardless of segment-size variance.
  std::vector<copy_chunk_desc> hchunks;
  hchunks.reserve(run.segments.size() * 4);
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    size_t bytes = size_t{seg.row_count} * type_size;
    auto* dst    = d_output + size_t{seg.row_offset} * type_size;
    size_t off   = 0;
    while (off < bytes) {
      uint32_t this_chunk = static_cast<uint32_t>(std::min<size_t>(COPY_CHUNK_BYTES, bytes - off));
      hchunks.push_back({seg.d_bytes + off, dst + off, this_chunk});
      off += this_chunk;
    }
  }

  size_t n = hchunks.size();
  rmm::device_uvector<copy_chunk_desc> d_chunks(n, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_chunks.data(),
                               hchunks.data(),
                               n * sizeof(copy_chunk_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  kernel_batched_memcpy<<<static_cast<uint32_t>(n), BLOCK_DIM, 0, stream.value()>>>(
    d_chunks.data(), static_cast<uint32_t>(n));
}

/// Unused `cub::DeviceMemcpy::Batched` alternative to `decode_uncompressed_data`,
/// kept as a portable reference path.
[[maybe_unused]] void decode_uncompressed_data_cub(gpu_codec_run const& run,
                                                   uint8_t* d_output,
                                                   uint32_t type_size,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  std::vector<void const*> h_src;
  std::vector<void*> h_dst;
  std::vector<size_t> h_sizes;
  h_src.reserve(run.segments.size());
  h_dst.reserve(run.segments.size());
  h_sizes.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    auto const bytes = size_t{seg.row_count} * type_size;
    if (size_t{seg.bytes_size} < bytes) {
      throw std::runtime_error("gpu_decode_table: UNCOMPRESSED segment bytes_size (" +
                               std::to_string(seg.bytes_size) + ") < required " +
                               std::to_string(bytes));
    }
    h_src.push_back(static_cast<void const*>(seg.d_bytes));
    h_dst.push_back(static_cast<void*>(d_output + size_t{seg.row_offset} * type_size));
    h_sizes.push_back(bytes);
  }

  auto const n = h_src.size();
  //===----------0 live segments----------===//
  if (n == 0) return;

  //===----------1 live segment----------===//
  if (n == 1) {
    RMM_CUDA_TRY(
      cudaMemcpyAsync(h_dst[0], h_src[0], h_sizes[0], cudaMemcpyDeviceToDevice, stream.value()));
    return;
  }

  //===---------->1 live segment----------===//
  rmm::device_uvector<void const*> d_src(n, stream, mr);
  rmm::device_uvector<void*> d_dst(n, stream, mr);
  rmm::device_uvector<size_t> d_sizes(n, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(
    d_src.data(), h_src.data(), n * sizeof(void const*), cudaMemcpyHostToDevice, stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(
    d_dst.data(), h_dst.data(), n * sizeof(void*), cudaMemcpyHostToDevice, stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(
    d_sizes.data(), h_sizes.data(), n * sizeof(size_t), cudaMemcpyHostToDevice, stream.value()));

  size_t tmp_bytes = 0;
  RMM_CUDA_TRY(cub::DeviceMemcpy::Batched(nullptr,
                                          tmp_bytes,
                                          d_src.data(),
                                          d_dst.data(),
                                          d_sizes.data(),
                                          static_cast<int64_t>(n),
                                          stream.value()));
  rmm::device_buffer d_tmp(tmp_bytes, stream, mr);
  RMM_CUDA_TRY(cub::DeviceMemcpy::Batched(d_tmp.data(),
                                          tmp_bytes,
                                          d_src.data(),
                                          d_dst.data(),
                                          d_sizes.data(),
                                          static_cast<int64_t>(n),
                                          stream.value()));
}

/// Routes a data run to its codec impl. Adding a codec means adding one case
/// here and a corresponding `decode_<codec>_data` (or batched equivalent).
void dispatch_data_run(gpu_codec_run const& run,
                       uint8_t* d_output,
                       cudf::data_type type,
                       uint32_t type_size,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  switch (run.codec) {
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
      decode_uncompressed_data(run, d_output, type_size, stream, mr);
      return;
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
      decode_constant_data(run, d_output, type, type_size, stream);
      return;
    case duckdb::CompressionType::COMPRESSION_RLE:
      decode_rle_data(run, d_output, type, type_size, stream, mr);
      return;
    case duckdb::CompressionType::COMPRESSION_BITPACKING:
      decode_bitpacking_data(run, d_output, type, type_size, stream, mr);
      return;
    case duckdb::CompressionType::COMPRESSION_ALP:
      decode_alp_data(run, d_output, type, type_size, stream, mr);
      return;
    case duckdb::CompressionType::COMPRESSION_ALPRD:
      decode_alprd_data(run, d_output, type, type_size, stream, mr);
      return;
    default:
      throw std::runtime_error("gpu_decode_table: viability invariant violated — data codec " +
                               std::to_string(static_cast<int>(run.codec)) + " not implemented");
  }
}

//===----------------------------------------------------------------------===//
// Validity decode.
//
// Only UNCOMPRESSED is supported here. DuckDB does encode validity with
// other codecs (e.g. ROARING bitmaps), but those are decoded host-side
// upstream and arrive at this dispatcher as plain validity bytes.
//===----------------------------------------------------------------------===//

void dispatch_validity_run(gpu_codec_run const& run,
                           uint8_t* d_mask,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  if (run.codec != duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
    throw std::runtime_error("gpu_decode_table: viability invariant violated — validity codec " +
                             std::to_string(static_cast<int>(run.codec)) + " not implemented");
  }

  std::vector<void const*> h_src;
  std::vector<void*> h_dst;
  std::vector<size_t> h_sizes;
  h_src.reserve(run.segments.size());
  h_dst.reserve(run.segments.size());
  h_sizes.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    if (seg.row_offset % 8 != 0) {
      throw std::runtime_error("gpu_decode_table: validity row_offset (" +
                               std::to_string(seg.row_offset) + ") not byte-aligned");
    }
    auto const bytes = ::cuda::ceil_div(seg.row_count, 8);
    if (seg.bytes_size < bytes) {
      throw std::runtime_error("gpu_decode_table: validity segment bytes_size (" +
                               std::to_string(seg.bytes_size) + ") < required " +
                               std::to_string(bytes));
    }
    h_src.push_back(static_cast<void const*>(seg.d_bytes));
    h_dst.push_back(static_cast<void*>(d_mask + seg.row_offset / 8));
    h_sizes.push_back(bytes);
  }

  auto const n_live_segments = h_src.size();
  //===----------0 live segments----------===//
  if (n_live_segments == 0) return;

  //===----------1 live segment----------===//
  if (n_live_segments == 1) {
    RMM_CUDA_TRY(
      cudaMemcpyAsync(h_dst[0], h_src[0], h_sizes[0], cudaMemcpyDeviceToDevice, stream.value()));
    return;
  }

  //===---------->1 live segments----------===//
  rmm::device_uvector<void const*> d_src(n_live_segments, stream, mr);
  rmm::device_uvector<void*> d_dst(n_live_segments, stream, mr);
  rmm::device_uvector<size_t> d_sizes(n_live_segments, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_src.data(),
                               h_src.data(),
                               n_live_segments * sizeof(void const*),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(d_dst.data(),
                               h_dst.data(),
                               n_live_segments * sizeof(void*),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  RMM_CUDA_TRY(cudaMemcpyAsync(d_sizes.data(),
                               h_sizes.data(),
                               n_live_segments * sizeof(size_t),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  size_t tmp_bytes = 0;
  RMM_CUDA_TRY(cub::DeviceMemcpy::Batched(nullptr,
                                          tmp_bytes,
                                          d_src.data(),
                                          d_dst.data(),
                                          d_sizes.data(),
                                          static_cast<int64_t>(n_live_segments),
                                          stream.value()));
  rmm::device_buffer d_tmp(tmp_bytes, stream, mr);
  RMM_CUDA_TRY(cub::DeviceMemcpy::Batched(d_tmp.data(),
                                          tmp_bytes,
                                          d_src.data(),
                                          d_dst.data(),
                                          d_sizes.data(),
                                          static_cast<int64_t>(n_live_segments),
                                          stream.value()));
}

//===----------------------------------------------------------------------===//
// Per-column decode helpers.
//===----------------------------------------------------------------------===//

/// Allocates the column's data buffer, validates type metadata, and runs
/// every codec_run in `col.data` into the buffer.
rmm::device_buffer decode_column_data(gpu_column_decode_input const& col,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  // Refuse non-fixed-width types up front — `cudf::size_of` itself throws on
  // strings/lists/structs, so the diagnostic must fire before we call it.
  //
  // This is NOT a catch-all for DuckDB → cudf type narrowing. A HUGEINT that
  // upstream maps to INT64 still satisfies `is_fixed_width`; the resulting
  // 8-byte stride would mismatch the 16-byte segment layout. The viability
  // walker upstream is responsible for refusing those columns; this throw
  // only catches the obvious cases (variable-width types).
  if (!cudf::is_fixed_width(col.out_type)) {
    throw std::runtime_error(
      "gpu_decode_table: viability invariant violated — non-fixed-width type id " +
      std::to_string(static_cast<int>(col.out_type.id())));
  }
  auto type_size = static_cast<uint32_t>(cudf::size_of(col.out_type));
  validate_segment_bounds(col.data, col.total_rows, "data");

  rmm::device_buffer data_buf(size_t{col.total_rows} * type_size, stream, mr);
  auto* d_output = static_cast<uint8_t*>(data_buf.data());
  for (auto const& run : col.data) {
    dispatch_data_run(run, d_output, col.out_type, type_size, stream, mr);
  }
  return data_buf;
}

/// Builds the column's null mask: starts from an all-valid bitmask, then
/// overlays each validity codec_run onto it. Null counting is deferred to a
/// single batched `cudf::batch_null_count` call in `gpu_decode_table`.
///
/// Returns an empty buffer when the column has no rows, doesn't carry nulls,
/// or carries no validity runs (the empty buffer signals "no nulls" downstream
/// and avoids an unnecessary mask allocation + popcount).
rmm::device_buffer decode_column_validity(gpu_column_decode_input const& col,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  if (!col.has_nulls || col.total_rows == 0 || col.validity.empty()) {
    return rmm::device_buffer{};
  }
  validate_segment_bounds(col.validity, col.total_rows, "validity");

  // create_null_mask gives us a buffer pre-filled with all-1s, sized to
  // cudf's bitmask layout (Arrow-compatible 64-byte padding). Validity
  // segments overwrite specific byte ranges below; rows no segment covers
  // remain implicitly valid.
  rmm::device_buffer null_mask = cudf::create_null_mask(
    static_cast<cudf::size_type>(col.total_rows), cudf::mask_state::ALL_VALID, stream, mr);
  for (auto const& run : col.validity) {
    dispatch_validity_run(run, static_cast<uint8_t*>(null_mask.data()), stream, mr);
  }
  return null_mask;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API.
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> gpu_decode_table(std::vector<gpu_column_decode_input> const& cols,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  size_t num_cols = cols.size();
  if (num_cols == 0) {
    return std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  }

  // A `gpu_decode_table` call decodes one row-group slice — every column
  // covers the same row range. `cudf::batch_null_count` (used below to
  // compute null counts in one batched call) takes a single uniform
  // [start, stop), so this is the natural place to enforce the invariant.
  uint32_t common_rows = cols[0].total_rows;
  for (size_t ci = 1; ci < num_cols; ++ci) {
    if (cols[ci].total_rows != common_rows) {
      throw std::runtime_error("gpu_decode_table: all columns must share the same total_rows");
    }
  }
  // `cudf::size_type` is signed int32. With uniformity established above,
  // one check covers every column.
  if (common_rows > MAX_ROWS_PER_COLUMN) {
    throw std::runtime_error("gpu_decode_table: total_rows (" + std::to_string(common_rows) +
                             ") > cudf::size_type max");
  }

  std::vector<rmm::device_buffer> data_bufs;
  std::vector<rmm::device_buffer> null_masks;
  data_bufs.reserve(num_cols);
  null_masks.reserve(num_cols);
  for (size_t ci = 0; ci < num_cols; ++ci) {
    data_bufs.emplace_back(decode_column_data(cols[ci], stream, mr));
    null_masks.emplace_back(decode_column_validity(cols[ci], stream, mr));
  }

  // One batched popcount across every null mask in the table. Columns that
  // didn't allocate a mask (no nulls or zero rows) pass nullptr; cudf's
  // batch_null_count returns 0 for those without launching kernel work for
  // them. The call also performs the single D2H sync we need before
  // building the cudf::table.
  std::vector<cudf::bitmask_type const*> mask_ptrs(num_cols, nullptr);
  for (size_t ci = 0; ci < num_cols; ++ci) {
    mask_ptrs[ci] = static_cast<cudf::bitmask_type const*>(null_masks[ci].data());
  }
  std::vector<cudf::size_type> null_counts =
    cudf::batch_null_count(mask_ptrs, 0, static_cast<cudf::size_type>(common_rows), stream);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(num_cols);
  for (size_t ci = 0; ci < num_cols; ++ci) {
    columns.push_back(std::make_unique<cudf::column>(cols[ci].out_type,
                                                     static_cast<cudf::size_type>(common_rows),
                                                     std::move(data_bufs[ci]),
                                                     std::move(null_masks[ci]),
                                                     null_counts[ci]));
  }

  // Catch any sticky kernel-launch error from the work above. Synchronous
  // CUDA API calls were each wrapped in RMM_CUDA_TRY, so this only sees
  // launch-side failures (bad grid size, out-of-resources, etc.).
  RMM_CUDA_TRY(cudaPeekAtLastError());
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::cuda::scan
