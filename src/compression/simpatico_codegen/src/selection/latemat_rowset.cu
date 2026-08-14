// SPDX-License-Identifier: Apache-2.0
//
// latemat_rowset.cu — chunk-bucketed row-set construction for late
// materialization (SIRIUS_EXP_LATE_MAT). See codegen/selection/row_set.hpp.
//
// Nothing on the fused scan-filter path calls into this file; it is dead
// code unless the late materializer runs.
//
// All entry points are stream-ordered on the given stream except where a
// host sync is documented (bucket_sorted_local_ids — T sizes the exact CSR
// arrays; called at prepare_selection time only, never on a consumer path).

#include "codegen/selection/row_set.hpp"
#include "codegen/selection/selection.hpp"

#include <rmm/device_buffer.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::codegen {

namespace {

constexpr int kBlock         = 256;
constexpr int kWordsPerChunk = 32;  // 1024 rows / 32 bits

inline void throw_on_cuda(cudaError_t err, char const* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("latemat_rowset: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

inline int grid_for(std::int64_t items, int per_block)
{
  std::int64_t g = (items + per_block - 1) / per_block;
  if (g < 1) g = 1;
  if (g > 65535) g = 65535;  // grid-stride covers the rest
  return static_cast<int>(g);
}

// Element count with a 16-byte tail guard for buffers handed to cub device
// algorithms. CCCL 3's DeviceScan issues PREDICATED full-vector tail loads
// whose addresses extend past num_items on partial tiles; compute-sanitizer
// flags them as invalid 0-byte reads when the allocation is exactly
// num_items-sized (memcheck finding on 4- and 12-byte tiny-geometry inputs:
// 64 errors, all at bucket_sorted_local_ids' ExclusiveSum input). The guard
// keeps every such tail address inside the allocation; counts/semantics are
// unchanged (the guard elements are never read as data). Applied to EVERY
// cub-touched allocation in this file — the class, not the instance.
template <typename T>
inline std::size_t guarded_bytes(std::int64_t count)
{
  return (static_cast<std::size_t>(count) + (16 / sizeof(T)) + 1) * sizeof(T);
}

// ── Degenerate-size prefix sums without cub ─────────────────────────────────
// Round 2 of the memcheck findings: with the global tails guarded, ONE real
// invalid access remained — a 16-byte __shared__ read OOB (addr 0x400, thread
// 128) inside cub DeviceScanKernel's OWN shared staging on tiny num_items
// (T=1..3). That one cannot be fixed by padding caller buffers (cub sizes its
// own shared), outputs are bit-correct in plain runs (partial-tile staging
// over-read), but it is UB and aborts under compute-sanitizer. Bypass: for
// small n the prefix sum runs as a trivially-auditable single-thread kernel
// (prepare-time only, <=2048 sequential adds ≈ µs — these sites feed
// host-driven allocation anyway); large n keeps cub (full-tile dominated;
// never flagged, and the production CNT wave has run byte-identical at
// 29k-chunk scale all campaign).
constexpr std::int64_t kSmallScanMax = 2048;

template <typename T>
__global__ void small_prefix_sum_kernel(T const* __restrict__ in,
                                        std::int64_t n,
                                        bool inclusive,
                                        T* __restrict__ out)
{
  T acc{};
  for (std::int64_t i = 0; i < n; ++i) {
    T const v = in[i];
    if (inclusive) {
      acc += v;
      out[i] = acc;
    } else {
      out[i] = acc;
      acc += v;
    }
  }
}

template <typename T>
void device_prefix_sum(T const* in,
                       T* out,
                       std::int64_t n,
                       bool inclusive,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr,
                       char const* what)
{
  if (n <= 0) { return; }
  if (n <= kSmallScanMax) {
    small_prefix_sum_kernel<T><<<1, 1, 0, stream.value()>>>(in, n, inclusive, out);
    throw_on_cuda(cudaGetLastError(), what);
    return;
  }
  std::size_t temp_bytes = 0;
  if (inclusive) {
    throw_on_cuda(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, in, out, n, stream.value()),
                  what);
    rmm::device_buffer temp(temp_bytes, stream, mr);
    throw_on_cuda(
      cub::DeviceScan::InclusiveSum(temp.data(), temp_bytes, in, out, n, stream.value()), what);
  } else {
    throw_on_cuda(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, in, out, n, stream.value()),
                  what);
    rmm::device_buffer temp(temp_bytes, stream, mr);
    throw_on_cuda(
      cub::DeviceScan::ExclusiveSum(temp.data(), temp_bytes, in, out, n, stream.value()), what);
  }
}

// id -> chunk id stream for the RLE pass (CSR keys).
__global__ void chunk_of_id_kernel(std::uint32_t const* __restrict__ ids,
                                   std::int64_t count,
                                   std::uint32_t* __restrict__ out)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    out[i] = ids[i] >> 10;
  }
}

// in-chunk position, narrowed to the u16 canonical form (10 significant bits).
__global__ void in_chunk_offsets_kernel(std::uint32_t const* __restrict__ ids,
                                        std::int64_t count,
                                        std::uint16_t* __restrict__ out)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    out[i] = static_cast<std::uint16_t>(ids[i] & 1023u);
  }
}

// CSR -> ascending batch-local int32 rows. Block b expands touched chunk b.
__global__ void expand_local_indices_kernel(std::uint32_t const* __restrict__ chunk_ids,
                                            std::uint32_t const* __restrict__ offs,
                                            std::uint16_t const* __restrict__ in_chunk,
                                            std::int32_t* __restrict__ out)
{
  std::uint32_t const base      = offs[blockIdx.x];
  std::uint32_t const cnt       = offs[blockIdx.x + 1] - base;
  std::int32_t const chunk_base = static_cast<std::int32_t>(chunk_ids[blockIdx.x]) * 1024;
  for (std::uint32_t k = threadIdx.x; k < cnt; k += blockDim.x) {
    out[base + k] = chunk_base + static_cast<std::int32_t>(in_chunk[base + k]);
  }
}

// CSR -> fused-format mask words for the touched chunks only (the caller
// zeroes the whole strip first, preserving the tail-zero invariant). Block b
// stages its chunk's 32 words in shared memory — a chunk belongs to exactly
// one block, so no global atomics.
__global__ void scatter_mask_kernel(std::uint32_t const* __restrict__ chunk_ids,
                                    std::uint32_t const* __restrict__ offs,
                                    std::uint16_t const* __restrict__ in_chunk,
                                    std::uint32_t* __restrict__ mask_words)
{
  __shared__ std::uint32_t words[kWordsPerChunk];
  if (threadIdx.x < kWordsPerChunk) words[threadIdx.x] = 0u;
  __syncthreads();
  std::uint32_t const base = offs[blockIdx.x];
  std::uint32_t const cnt  = offs[blockIdx.x + 1] - base;
  for (std::uint32_t k = threadIdx.x; k < cnt; k += blockDim.x) {
    std::uint32_t const pos = in_chunk[base + k];
    atomicOr(&words[pos >> 5], 1u << (pos & 31u));
  }
  __syncthreads();
  if (threadIdx.x < kWordsPerChunk) {
    mask_words[static_cast<std::int64_t>(chunk_ids[blockIdx.x]) * kWordsPerChunk + threadIdx.x] =
      words[threadIdx.x];
  }
}

// Scatter per-touched-chunk survivor counts into the dense all-chunks count
// array (pre-zeroed by the caller).
__global__ void scatter_chunk_counts_kernel(std::uint32_t const* __restrict__ chunk_ids,
                                            std::uint32_t const* __restrict__ offs,
                                            std::int64_t num_touched,
                                            std::uint32_t* __restrict__ dense_counts)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t b = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       b < num_touched;
       b += stride) {
    dense_counts[chunk_ids[b]] = offs[b + 1] - offs[b];
  }
}

__global__ void set_u32_kernel(std::uint32_t* __restrict__ dst, std::uint32_t value)
{
  *dst = value;
}

__global__ void iota_i32_kernel(std::int32_t* __restrict__ out, std::int64_t count)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    out[i] = static_cast<std::int32_t>(i);
  }
}

// New-run flags over the sorted id stream (1 at position 0 and wherever the
// value changes).
__global__ void run_flags_kernel(std::uint64_t const* __restrict__ sorted,
                                 std::int64_t count,
                                 std::int32_t* __restrict__ flags)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    flags[i] = (i == 0 || sorted[i] != sorted[i - 1]) ? 1 : 0;
  }
}

// Scatter pass after the rank scan: unique ids (idempotent duplicate writes)
// and order-restoration ranks back to original positions. Also parks the
// unique COUNT as a device scalar (thread of the last element), so no host
// sync is needed inside sort_unique_global_ids.
__global__ void scatter_unique_and_rank_kernel(std::uint64_t const* __restrict__ sorted,
                                               std::int32_t const* __restrict__ perm,
                                               std::int32_t const* __restrict__ rank_incl,
                                               std::int64_t count,
                                               std::uint64_t* __restrict__ unique_out,
                                               std::int32_t* __restrict__ restore_rank,
                                               std::int32_t* __restrict__ count_out)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    std::int32_t const r  = rank_incl[i] - 1;  // inclusive scan of flags -> 0-based rank
    unique_out[r]         = sorted[i];         // duplicates write the same value
    restore_rank[perm[i]] = r;
    if (i == count - 1) { *count_out = rank_incl[i]; }
  }
}

// One thread per batch boundary: binary search the first index with
// sorted[idx] >= bound. The element count is read from `count_dev` when
// non-null (device-bounded search — lets the upstream sort skip its count
// sync), else `max_count` is exact.
__global__ void batch_starts_kernel(std::uint64_t const* __restrict__ sorted,
                                    std::int64_t max_count,
                                    std::int32_t const* __restrict__ count_dev,
                                    std::uint64_t const* __restrict__ bounds,
                                    std::int32_t num_bounds,
                                    std::int64_t* __restrict__ starts)
{
  std::int32_t const b = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= num_bounds) { return; }
  std::int64_t const count =
    count_dev != nullptr ? static_cast<std::int64_t>(*count_dev) : max_count;
  std::uint64_t const bound = bounds[b];
  std::int64_t lo = 0, hi = count;
  while (lo < hi) {
    std::int64_t const mid = lo + ((hi - lo) >> 1);
    if (sorted[mid] < bound) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  starts[b] = lo;
}

__global__ void global_to_local_kernel(std::uint64_t const* __restrict__ ids,
                                       std::int64_t count,
                                       std::uint64_t base_rows,
                                       std::uint32_t* __restrict__ out)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    out[i] = static_cast<std::uint32_t>(ids[i] - base_rows);
  }
}

// Multi-source fixed-width gather (SIRIUS_EXP_LATE_MAT_V2 multi-batch raw
// path): element i comes from the batch found by binary search over row_start
// (B+1 exclusive starts). Templated on the element width so the copy is one
// load/store.
template <typename Elem>
__global__ void multi_source_gather_kernel(void const* const* __restrict__ bases,
                                           std::int64_t const* __restrict__ row_start,
                                           std::int32_t num_batches,
                                           std::uint64_t const* __restrict__ ids,
                                           std::int64_t count,
                                           Elem* __restrict__ out)
{
  std::int64_t const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    std::int64_t const id = static_cast<std::int64_t>(ids[i]);
    // upper_bound(row_start, id) - 1: B is small, the array stays in cache.
    std::int32_t lo = 0, hi = num_batches;
    while (lo < hi) {
      std::int32_t const mid = lo + ((hi - lo) >> 1);
      if (row_start[mid + 1] <= id) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out[i] = static_cast<Elem const*>(bases[lo])[id - row_start[lo]];
  }
}

}  // namespace

void multi_source_gather_fixed(void const* const* bases_dev,
                               std::int64_t const* row_start_dev,
                               std::int32_t num_batches,
                               std::size_t elem_size,
                               std::uint64_t const* ids,
                               std::int64_t count,
                               void* out,
                               rmm::cuda_stream_view stream)
{
  if (count == 0) { return; }
  if (bases_dev == nullptr || row_start_dev == nullptr || ids == nullptr || out == nullptr ||
      num_batches <= 0) {
    throw std::runtime_error("latemat_rowset: multi_source_gather_fixed: invalid inputs");
  }
  auto const grid = grid_for(count, kBlock);
  switch (elem_size) {
    case 1:
      multi_source_gather_kernel<std::uint8_t><<<grid, kBlock, 0, stream.value()>>>(
        bases_dev, row_start_dev, num_batches, ids, count, static_cast<std::uint8_t*>(out));
      break;
    case 2:
      multi_source_gather_kernel<std::uint16_t><<<grid, kBlock, 0, stream.value()>>>(
        bases_dev, row_start_dev, num_batches, ids, count, static_cast<std::uint16_t*>(out));
      break;
    case 4:
      multi_source_gather_kernel<std::uint32_t><<<grid, kBlock, 0, stream.value()>>>(
        bases_dev, row_start_dev, num_batches, ids, count, static_cast<std::uint32_t*>(out));
      break;
    case 8:
      multi_source_gather_kernel<std::uint64_t><<<grid, kBlock, 0, stream.value()>>>(
        bases_dev, row_start_dev, num_batches, ids, count, static_cast<std::uint64_t*>(out));
      break;
    case 16:
      multi_source_gather_kernel<uint4><<<grid, kBlock, 0, stream.value()>>>(
        bases_dev, row_start_dev, num_batches, ids, count, static_cast<uint4*>(out));
      break;
    default:
      throw std::runtime_error("latemat_rowset: multi_source_gather_fixed: unsupported width " +
                               std::to_string(elem_size));
  }
  throw_on_cuda(cudaGetLastError(), "multi-source gather");
}

owned_chunk_row_set bucket_sorted_local_ids(std::uint32_t const* sorted_local_ids,
                                            std::int64_t count,
                                            std::int64_t num_rows,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  owned_chunk_row_set set;
  set.num_survivors = count;
  set.num_rows      = num_rows;
  if (count == 0) { return set; }
  if (count > num_rows) {
    throw std::runtime_error(
      "latemat_rowset: bucket_sorted_local_ids: count > num_rows "
      "(duplicate or out-of-range ids?)");
  }

  // Run-length encode the chunk-id stream. Worst-case sized outputs, then
  // exact-size copies. (CCCL 3 removed cub's TransformInputIterator, so the
  // chunk-id stream is materialized — 4 B/survivor of prepare-time scratch.)
  rmm::device_buffer chunk_stream(guarded_bytes<std::uint32_t>(count), stream, mr);
  chunk_of_id_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    sorted_local_ids, count, static_cast<std::uint32_t*>(chunk_stream.data()));
  throw_on_cuda(cudaGetLastError(), "chunk-id stream");
  auto const* chunk_it = static_cast<std::uint32_t const*>(chunk_stream.data());
  rmm::device_buffer unique_chunks(guarded_bytes<std::uint32_t>(count), stream, mr);
  rmm::device_buffer run_lengths(guarded_bytes<std::uint32_t>(count), stream, mr);
  rmm::device_buffer num_runs_dev(sizeof(std::int64_t), stream, mr);

  std::size_t temp_bytes = 0;
  throw_on_cuda(
    cub::DeviceRunLengthEncode::Encode(nullptr,
                                       temp_bytes,
                                       chunk_it,
                                       static_cast<std::uint32_t*>(unique_chunks.data()),
                                       static_cast<std::uint32_t*>(run_lengths.data()),
                                       static_cast<std::int64_t*>(num_runs_dev.data()),
                                       count,
                                       stream.value()),
    "rle size query");
  rmm::device_buffer temp(temp_bytes, stream, mr);
  throw_on_cuda(
    cub::DeviceRunLengthEncode::Encode(temp.data(),
                                       temp_bytes,
                                       chunk_it,
                                       static_cast<std::uint32_t*>(unique_chunks.data()),
                                       static_cast<std::uint32_t*>(run_lengths.data()),
                                       static_cast<std::int64_t*>(num_runs_dev.data()),
                                       count,
                                       stream.value()),
    "rle encode");

  // The documented host sync: T sizes the exact CSR arrays.
  std::int64_t num_touched = 0;
  throw_on_cuda(cudaMemcpyAsync(&num_touched,
                                num_runs_dev.data(),
                                sizeof(std::int64_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()),
                "num_runs D2H");
  throw_on_cuda(cudaStreamSynchronize(stream.value()), "num_runs sync");
  set.num_touched = num_touched;

  // Exact-size chunk id list.
  set.chunk_ids =
    rmm::device_buffer(static_cast<std::size_t>(num_touched) * sizeof(std::uint32_t), stream, mr);
  throw_on_cuda(cudaMemcpyAsync(set.chunk_ids.data(),
                                unique_chunks.data(),
                                static_cast<std::size_t>(num_touched) * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToDevice,
                                stream.value()),
                "chunk ids copy");

  // Exclusive scan of run lengths -> chunk_out_offsets[0..T], last = count.
  set.chunk_out_offsets =
    rmm::device_buffer(guarded_bytes<std::uint32_t>(num_touched + 1), stream, mr);
  auto* offs = static_cast<std::uint32_t*>(set.chunk_out_offsets.data());
  device_prefix_sum<std::uint32_t>(static_cast<std::uint32_t*>(run_lengths.data()),
                                   offs,
                                   num_touched,
                                   /*inclusive=*/false,
                                   stream,
                                   mr,
                                   "offsets scan");
  set_u32_kernel<<<1, 1, 0, stream.value()>>>(offs + num_touched,
                                              static_cast<std::uint32_t>(count));
  throw_on_cuda(cudaGetLastError(), "offsets tail");

  // u16 in-chunk positions.
  set.in_chunk_offsets =
    rmm::device_buffer(static_cast<std::size_t>(count) * sizeof(std::uint16_t), stream, mr);
  in_chunk_offsets_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    sorted_local_ids, count, static_cast<std::uint16_t*>(set.in_chunk_offsets.data()));
  throw_on_cuda(cudaGetLastError(), "in-chunk offsets");
  return set;
}

sorted_unique_ids sort_unique_global_ids(std::uint64_t const* ids,
                                         std::int64_t count,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  sorted_unique_ids out;
  out.original_count = count;
  if (count <= 0) { return out; }

  // Radix sort (u64 keys, int32 original positions as values). Guarded tails
  // like every cub-touched buffer in this file.
  rmm::device_buffer keys_a(guarded_bytes<std::uint64_t>(count), stream, mr);
  rmm::device_buffer keys_b(guarded_bytes<std::uint64_t>(count), stream, mr);
  rmm::device_buffer vals_a(guarded_bytes<std::int32_t>(count), stream, mr);
  rmm::device_buffer vals_b(guarded_bytes<std::int32_t>(count), stream, mr);
  throw_on_cuda(cudaMemcpyAsync(keys_a.data(),
                                ids,
                                static_cast<std::size_t>(count) * sizeof(std::uint64_t),
                                cudaMemcpyDeviceToDevice,
                                stream.value()),
                "ids copy");
  iota_i32_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    static_cast<std::int32_t*>(vals_a.data()), count);
  throw_on_cuda(cudaGetLastError(), "iota");

  cub::DoubleBuffer<std::uint64_t> d_keys(static_cast<std::uint64_t*>(keys_a.data()),
                                          static_cast<std::uint64_t*>(keys_b.data()));
  cub::DoubleBuffer<std::int32_t> d_vals(static_cast<std::int32_t*>(vals_a.data()),
                                         static_cast<std::int32_t*>(vals_b.data()));
  std::size_t temp_bytes = 0;
  throw_on_cuda(cub::DeviceRadixSort::SortPairs(
                  nullptr, temp_bytes, d_keys, d_vals, count, 0, 64, stream.value()),
                "radix sort size query");
  rmm::device_buffer temp(temp_bytes, stream, mr);
  throw_on_cuda(cub::DeviceRadixSort::SortPairs(
                  temp.data(), temp_bytes, d_keys, d_vals, count, 0, 64, stream.value()),
                "radix sort");
  std::uint64_t const* sorted = d_keys.Current();
  std::int32_t const* perm    = d_vals.Current();

  // New-run flags -> inclusive scan -> 0-based ranks; scatter unique ids and
  // order-restoration ranks in one pass.
  rmm::device_buffer flags(guarded_bytes<std::int32_t>(count), stream, mr);
  run_flags_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    sorted, count, static_cast<std::int32_t*>(flags.data()));
  throw_on_cuda(cudaGetLastError(), "run flags");
  rmm::device_buffer rank_incl(guarded_bytes<std::int32_t>(count), stream, mr);
  device_prefix_sum<std::int32_t>(static_cast<std::int32_t*>(flags.data()),
                                  static_cast<std::int32_t*>(rank_incl.data()),
                                  count,
                                  /*inclusive=*/true,
                                  stream,
                                  mr,
                                  "rank scan");

  // NO host sync (sync-surgery rev): the unique-id buffer is allocated
  // worst-case (count entries — the unique scatter is rank-idempotent, so the
  // valid prefix is exactly unique_count entries), the scatter grid covers
  // all count elements regardless, and the unique count is parked in
  // count_dev for the caller's single boundary sync
  // (split_sorted_ids_by_batch reads it back for prepare_selection).
  out.ids = rmm::device_buffer(static_cast<std::size_t>(count) * sizeof(std::uint64_t), stream, mr);
  out.restore_rank =
    rmm::device_buffer(static_cast<std::size_t>(count) * sizeof(std::int32_t), stream, mr);
  out.count_dev = rmm::device_buffer(sizeof(std::int32_t), stream, mr);
  scatter_unique_and_rank_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    sorted,
    perm,
    static_cast<std::int32_t const*>(rank_incl.data()),
    count,
    static_cast<std::uint64_t*>(out.ids.data()),
    static_cast<std::int32_t*>(out.restore_rank.data()),
    static_cast<std::int32_t*>(out.count_dev.data()));
  throw_on_cuda(cudaGetLastError(), "unique/rank scatter");
  return out;
}

std::vector<std::int64_t> split_sorted_ids_by_batch(
  std::uint64_t const* sorted_ids,
  std::int64_t max_count,
  std::int32_t const* count_dev,
  std::vector<std::int64_t> const& batch_row_start,
  std::int64_t* count_out,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_bounds = static_cast<std::int32_t>(batch_row_start.size());
  std::vector<std::int64_t> starts(static_cast<std::size_t>(num_bounds), 0);
  if (count_out != nullptr) { *count_out = 0; }
  if (num_bounds == 0 || max_count == 0) {
    if (count_out != nullptr && count_dev == nullptr) { *count_out = max_count; }
    return starts;
  }

  std::vector<std::uint64_t> host_bounds(static_cast<std::size_t>(num_bounds));
  for (std::size_t k = 0; k < host_bounds.size(); ++k) {
    host_bounds[k] = static_cast<std::uint64_t>(batch_row_start[k]);
  }
  rmm::device_buffer bounds(
    host_bounds.data(), host_bounds.size() * sizeof(std::uint64_t), stream, mr);
  rmm::device_buffer starts_dev(host_bounds.size() * sizeof(std::int64_t), stream, mr);
  batch_starts_kernel<<<grid_for(num_bounds, kBlock), kBlock, 0, stream.value()>>>(
    sorted_ids,
    max_count,
    count_dev,
    static_cast<std::uint64_t const*>(bounds.data()),
    num_bounds,
    static_cast<std::int64_t*>(starts_dev.data()));
  throw_on_cuda(cudaGetLastError(), "batch starts");
  // THE single boundary host sync of the canonical prepare: starts (host-
  // driven slicing/allocation) + the folded-in element count.
  std::int32_t actual_count32 = 0;
  throw_on_cuda(cudaMemcpyAsync(starts.data(),
                                starts_dev.data(),
                                host_bounds.size() * sizeof(std::int64_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()),
                "batch starts D2H");
  if (count_dev != nullptr) {
    throw_on_cuda(
      cudaMemcpyAsync(
        &actual_count32, count_dev, sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream.value()),
      "count D2H");
  }
  throw_on_cuda(cudaStreamSynchronize(stream.value()), "batch starts sync");
  if (count_out != nullptr) {
    *count_out = count_dev != nullptr ? static_cast<std::int64_t>(actual_count32) : max_count;
  }
  return starts;
}

void global_slice_to_local(std::uint64_t const* ids,
                           std::int64_t count,
                           std::int64_t batch_row_start,
                           std::uint32_t* out_local,
                           rmm::cuda_stream_view stream)
{
  if (count == 0) { return; }
  global_to_local_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    ids, count, static_cast<std::uint64_t>(batch_row_start), out_local);
  throw_on_cuda(cudaGetLastError(), "global to local");
}

void row_set_to_local_indices(chunk_row_set const& set,
                              std::int32_t* out,
                              rmm::cuda_stream_view stream)
{
  if (set.num_survivors == 0) { return; }
  if (!set.valid() || out == nullptr) {
    throw std::runtime_error("latemat_rowset: row_set_to_local_indices: invalid inputs");
  }
  expand_local_indices_kernel<<<static_cast<unsigned>(set.num_touched),
                                kBlock,
                                0,
                                stream.value()>>>(
    set.chunk_ids, set.chunk_out_offsets, set.in_chunk_offsets, out);
  throw_on_cuda(cudaGetLastError(), "expand local indices");
}

void row_set_to_mask(chunk_row_set const& set,
                     std::uint32_t* mask_words,
                     std::uint32_t* all_chunk_offsets,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  if (mask_words == nullptr || all_chunk_offsets == nullptr || !set.valid() || set.num_rows <= 0) {
    throw std::runtime_error("latemat_rowset: row_set_to_mask: invalid inputs");
  }
  std::int64_t const nc        = selection_mask::ChunksFor(set.num_rows);
  std::int64_t const num_words = selection_mask::WordsFor(set.num_rows);

  // Full-strip zero first (tail-zero invariant + untouched chunks).
  throw_on_cuda(
    cudaMemsetAsync(
      mask_words, 0, static_cast<std::size_t>(num_words) * sizeof(std::uint32_t), stream.value()),
    "mask zero");
  if (set.num_survivors > 0) {
    scatter_mask_kernel<<<static_cast<unsigned>(set.num_touched), kBlock, 0, stream.value()>>>(
      set.chunk_ids, set.chunk_out_offsets, set.in_chunk_offsets, mask_words);
    throw_on_cuda(cudaGetLastError(), "mask scatter");
  }

  // Dense per-chunk counts -> exclusive scan -> all_chunk_offsets[0..nc],
  // last = num_survivors (written directly — known without a device reduce).
  rmm::device_buffer dense_counts(guarded_bytes<std::uint32_t>(nc), stream, mr);
  throw_on_cuda(cudaMemsetAsync(dense_counts.data(), 0, dense_counts.size(), stream.value()),
                "counts zero");  // full buffer incl. guard tail
  if (set.num_survivors > 0) {
    scatter_chunk_counts_kernel<<<grid_for(set.num_touched, kBlock), kBlock, 0, stream.value()>>>(
      set.chunk_ids,
      set.chunk_out_offsets,
      set.num_touched,
      static_cast<std::uint32_t*>(dense_counts.data()));
    throw_on_cuda(cudaGetLastError(), "counts scatter");
  }
  device_prefix_sum<std::uint32_t>(static_cast<std::uint32_t*>(dense_counts.data()),
                                   all_chunk_offsets,
                                   nc,
                                   /*inclusive=*/false,
                                   stream,
                                   mr,
                                   "chunk offsets scan");
  set_u32_kernel<<<1, 1, 0, stream.value()>>>(all_chunk_offsets + nc,
                                              static_cast<std::uint32_t>(set.num_survivors));
  throw_on_cuda(cudaGetLastError(), "chunk offsets tail");
}

}  // namespace sirius::codegen
