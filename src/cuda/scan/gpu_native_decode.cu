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
// GPU-native decode dispatcher.
//
// Top-down structure of this file:
//   1. Validity-mask kernels (fill all-valid; popcount valid bits).
//   2. CONSTANT broadcast (cudf::type_dispatcher → typed kernel).
//   3. UNCOMPRESSED data copy (single-segment DMA fast path; multi-segment
//      batched-memcpy kernel).
//   4. Per-run dispatcher switches (data, validity).
//   5. Per-column helpers (decode_column_data, decode_column_validity).
//   6. Public `gpu_decode_table` entry — drives the per-column loop, then
//      issues a single stream sync + bulk D2H readback for null counts.
//
// Kernel-launch errors are caught at the end via `cudaPeekAtLastError`;
// every synchronous CUDA API call is individually wrapped in `RMM_CUDA_TRY`.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_native_decode.cuh"

#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

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

/// Block dim used by every kernel in this file. The popcount kernel's
/// shared-mem reduction width is hard-coded to this value; a `static_assert`
/// inside the kernel guards the coupling.
constexpr uint32_t BLOCK_DIM = 256;

/// Block-count cap for the popcount kernel. Past this point we'd just be
/// adding scheduling cost — the kernel grid-strides over the remainder.
/// Sized to saturate the SM count of the smallest target arch.
constexpr uint32_t MAX_POPCOUNT_BLOCKS = 512;

/// Maximum bytes one block of `kernel_batched_memcpy` will copy. Splitting
/// large segments into chunks of this size keeps per-block work uniform so
/// the grid doesn't end up with stragglers. Tuned empirically; if you change
/// it, re-run the [!benchmark][scan][decode] cases.
constexpr uint32_t COPY_CHUNK_BYTES = 64u << 10;

/// `cudf::size_type` is signed int32. Catch a `total_rows` overflow before
/// it silently wraps in the static_cast at column construction.
constexpr uint32_t MAX_ROWS_PER_COLUMN =
  static_cast<uint32_t>(std::numeric_limits<cudf::size_type>::max());

//===----------------------------------------------------------------------===//
// Validity mask kernels.
//===----------------------------------------------------------------------===//

/// Initialises the bitmask to all-valid. Validity segments later overlay
/// their bytes onto specific row ranges; rows that no segment covers stay
/// implicit-valid.
__global__ void kernel_fill_valid(uint64_t* mask, uint32_t num_words)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_words) mask[idx] = ~0ULL;
}

/// Counts set bits in `mask[0..num_words)` and atomic-adds the total into
/// `*d_valid_count`. The caller must zero `*d_valid_count` before launch.
///
/// The last word may contain padding bits past `total_rows`; those are
/// masked out so they don't inflate the count.
__global__ void kernel_count_valid_bits(uint64_t const* __restrict__ mask,
                                        uint32_t num_words,
                                        uint32_t total_rows,
                                        uint32_t* __restrict__ d_valid_count)
{
  static_assert(BLOCK_DIM == 256, "shared-mem reduction width is hard-coded to 256");
  __shared__ uint32_t s_counts[BLOCK_DIM];
  uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t valid  = 0;
  for (uint32_t i = tid; i < num_words; i += stride) {
    uint64_t word = mask[i];
    if (i == num_words - 1) {
      uint32_t tail = total_rows & 63;
      if (tail > 0) word &= (1ULL << tail) - 1;
    }
    valid += __popcll(word);
  }
  s_counts[threadIdx.x] = valid;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) s_counts[threadIdx.x] += s_counts[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(d_valid_count, s_counts[0]);
}

//===----------------------------------------------------------------------===//
// CONSTANT broadcast.
//
// We do NOT use `cudf::make_column_from_scalar` here. That path does a D2H
// `device_scalar::value(stream)` read internally to feed `thrust::fill_n`,
// which would add one synchronisation per CONSTANT segment. Our value is
// already on device; the kernel below dereferences the device pointer
// directly so the broadcast stays stream-async.
//===----------------------------------------------------------------------===//

/// Reads a single value from `*src` (already on device) and writes it to
/// every position in `dest[0..rows)`. When `sizeof(T)` divides 16 we pack
/// the value into an `int4` and store 16 bytes per thread; otherwise we fall
/// back to scalar stores (covers e.g. the int128 backing decimal128).
template <typename T>
__global__ void kernel_broadcast_constant(T* __restrict__ dest,
                                          T const* __restrict__ src,
                                          uint32_t rows)
{
  T val           = *src;
  uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = gridDim.x * blockDim.x;

  if constexpr (sizeof(T) <= 8 && 16u % sizeof(T) == 0) {
    constexpr uint32_t TPV = 16u / sizeof(T);  // Ts per int4
    int4 vpacked;
    T* lanes = reinterpret_cast<T*>(&vpacked);
#pragma unroll
    for (uint32_t i = 0; i < TPV; ++i)
      lanes[i] = val;

    uint32_t v_rows = rows / TPV;
    int4* d4        = reinterpret_cast<int4*>(dest);
    for (uint32_t i = tid; i < v_rows; i += stride)
      d4[i] = vpacked;

    // Tail rows that didn't fit a full int4.
    uint32_t tail_start = v_rows * TPV;
    for (uint32_t i = tail_start + tid; i < rows; i += stride)
      dest[i] = val;
  } else {
    for (uint32_t i = tid; i < rows; i += stride)
      dest[i] = val;
  }
}

/// Functor for `cudf::type_dispatcher`. We project each cudf type to its
/// on-device storage type via `device_storage_type_t<T>` — for plain numeric
/// types this is just `T`, but for fixed-point types the dispatched class
/// (e.g. `numeric::decimal128`) is wider than its actual column storage
/// (the underlying `Rep`, e.g. `__int128_t`). Without this projection the
/// kernel's stride would mismatch the column's physical width.
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

/// Each block copies one chunk. Picks the widest aligned access path the
/// chunk's pointers + length jointly support: 16-byte (int4), 4-byte
/// (uint32_t), or per-byte. DuckDB segment offsets aren't always 16-byte
/// aligned, so the narrower fallbacks really do trigger.
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

/// Single-segment runs go through a direct `cudaMemcpyAsync` (the GPU's DMA
/// fast path). Multi-segment runs are issued as one batched-kernel launch:
/// the per-call CUDA API setup cost otherwise dominates once the run has
/// more than a few dozen segments — the per-arch baseline JSON in
/// `test/data/decode_baselines/` carries the measurements that justify this.
void decode_uncompressed_data(gpu_codec_run const& run,
                              uint8_t* d_output,
                              uint32_t type_size,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
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

void dispatch_validity_run(gpu_codec_run const& run, uint8_t* d_mask, rmm::cuda_stream_view stream)
{
  if (run.codec != duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
    throw std::runtime_error("gpu_decode_table: viability invariant violated — validity codec " +
                             std::to_string(static_cast<int>(run.codec)) + " not implemented");
  }
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    // Validity bits pack 8 rows per byte. A non-byte-aligned `row_offset`
    // would mean a partial-byte memcpy destination, which is meaningless —
    // treat it as a malformed segment.
    if (seg.row_offset % 8 != 0) {
      throw std::runtime_error("gpu_decode_table: validity row_offset (" +
                               std::to_string(seg.row_offset) + ") not byte-aligned");
    }
    size_t bytes = (size_t{seg.row_count} + 7) / 8;
    RMM_CUDA_TRY(cudaMemcpyAsync(
      d_mask + seg.row_offset / 8, seg.d_bytes, bytes, cudaMemcpyDeviceToDevice, stream.value()));
  }
}

//===----------------------------------------------------------------------===//
// Per-column decode helpers.
//===----------------------------------------------------------------------===//

/// Allocates the column's data buffer, validates type metadata, and runs
/// every codec_run in `col.data` into the buffer.
rmm::device_buffer decode_column_data(gpu_column_decode_input const& col,
                                      size_t column_index,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (col.total_rows > MAX_ROWS_PER_COLUMN) {
    throw std::runtime_error("gpu_decode_table: column " + std::to_string(column_index) + " has " +
                             std::to_string(col.total_rows) + " rows > cudf::size_type max");
  }
  // Refuse non-fixed-width types here — `cudf::size_of` itself throws on
  // strings/lists/structs, so the diagnostic must fire before we call it.
  // Also catches DuckDB → cudf type narrowing (e.g. HUGEINT → INT64) where
  // the cudf physical width would not match the segment bytes.
  if (!cudf::is_fixed_width(col.out_type)) {
    throw std::runtime_error(
      "gpu_decode_table: viability invariant violated — non-fixed-width type id " +
      std::to_string(static_cast<int>(col.out_type.id())));
  }
  auto type_size = static_cast<uint32_t>(cudf::size_of(col.out_type));

  rmm::device_buffer data_buf(size_t{col.total_rows} * type_size, stream, mr);
  auto* d_output = static_cast<uint8_t*>(data_buf.data());
  for (auto const& run : col.data) {
    dispatch_data_run(run, d_output, col.out_type, type_size, stream, mr);
  }
  return data_buf;
}

/// Allocates the null mask (sized for `total_rows`), pre-fills it all-valid,
/// then overlays each validity codec_run onto it. The popcount kernel writes
/// the live count into `*d_valid_count_slot`; the public entry reads all
/// slots back together after a single stream sync.
///
/// Returns an empty buffer when the column has no nulls or no rows; in that
/// case the slot is left untouched and the public entry skips reading it.
rmm::device_buffer decode_column_validity(gpu_column_decode_input const& col,
                                          uint32_t* d_valid_count_slot,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  if (!col.has_nulls || col.total_rows == 0) return rmm::device_buffer{};

  size_t mask_bytes =
    cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(col.total_rows));
  rmm::device_buffer null_mask(mask_bytes, stream, mr);
  uint32_t num_words = static_cast<uint32_t>((col.total_rows + 63) / 64);
  auto* d_mask_words = static_cast<uint64_t*>(null_mask.data());

  uint32_t fill_blocks = (num_words + BLOCK_DIM - 1) / BLOCK_DIM;
  kernel_fill_valid<<<fill_blocks, BLOCK_DIM, 0, stream.value()>>>(d_mask_words, num_words);
  for (auto const& run : col.validity) {
    dispatch_validity_run(run, static_cast<uint8_t*>(null_mask.data()), stream);
  }

  RMM_CUDA_TRY(cudaMemsetAsync(d_valid_count_slot, 0, sizeof(uint32_t), stream.value()));
  uint32_t cnt_blocks =
    std::min(MAX_POPCOUNT_BLOCKS, std::max(1u, (num_words + BLOCK_DIM - 1) / BLOCK_DIM));
  kernel_count_valid_bits<<<cnt_blocks, BLOCK_DIM, 0, stream.value()>>>(
    d_mask_words, num_words, col.total_rows, d_valid_count_slot);
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

  // Deferred-readback bookkeeping. Each null-bearing column gets one device
  // uint32 that the popcount kernel atomic-adds into. After all per-column
  // work is queued we sync once and read every slot back in a single D2H —
  // the alternative is one sync per column, which would serialise the work.
  //
  //   null_idx     : indices (into `cols`) of columns with has_nulls=true.
  //   col_to_slot  : reverse map from column index to slot index, -1 if the
  //                  column has no nulls.
  std::vector<size_t> null_idx;
  for (size_t i = 0; i < num_cols; ++i)
    if (cols[i].has_nulls) null_idx.push_back(i);
  rmm::device_uvector<uint32_t> d_valid_counts_buf(null_idx.size(), stream, mr);
  uint32_t* d_valid_counts = null_idx.empty() ? nullptr : d_valid_counts_buf.data();
  std::vector<int> col_to_slot(num_cols, -1);
  for (size_t i = 0; i < null_idx.size(); ++i)
    col_to_slot[null_idx[i]] = static_cast<int>(i);

  std::vector<rmm::device_buffer> data_bufs;
  std::vector<rmm::device_buffer> null_masks;
  data_bufs.reserve(num_cols);
  null_masks.reserve(num_cols);

  for (size_t ci = 0; ci < num_cols; ++ci) {
    auto const& col = cols[ci];
    data_bufs.emplace_back(decode_column_data(col, ci, stream, mr));

    uint32_t* slot = (col_to_slot[ci] >= 0) ? d_valid_counts + col_to_slot[ci] : nullptr;
    null_masks.emplace_back(decode_column_validity(col, slot, stream, mr));
  }

  // Bulk D2H of all valid-count slots, then one stream sync for the whole
  // batch — both the kernel work and this memcpy land before we proceed.
  std::vector<uint32_t> h_valid_counts(null_idx.size(), 0);
  if (!null_idx.empty()) {
    RMM_CUDA_TRY(cudaMemcpyAsync(h_valid_counts.data(),
                                 d_valid_counts,
                                 null_idx.size() * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(num_cols);
  for (size_t ci = 0; ci < num_cols; ++ci) {
    auto const& col            = cols[ci];
    cudf::size_type null_count = 0;
    // `decode_column_validity` skips writing the slot when total_rows == 0,
    // so don't read it in that case.
    if (col_to_slot[ci] >= 0 && col.total_rows > 0) {
      null_count = static_cast<cudf::size_type>(col.total_rows - h_valid_counts[col_to_slot[ci]]);
    }
    columns.push_back(std::make_unique<cudf::column>(col.out_type,
                                                     static_cast<cudf::size_type>(col.total_rows),
                                                     std::move(data_bufs[ci]),
                                                     std::move(null_masks[ci]),
                                                     null_count));
  }

  // Catch any sticky kernel-launch error from the work above. Synchronous
  // CUDA API calls were each wrapped in RMM_CUDA_TRY, so this only sees
  // launch-side failures (bad grid size, out-of-resources, etc.).
  RMM_CUDA_TRY(cudaPeekAtLastError());
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::cuda::scan
