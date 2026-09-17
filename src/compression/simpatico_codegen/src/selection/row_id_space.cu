// SPDX-License-Identifier: Apache-2.0
//
// row_id_space.cu — global post-join ids to per-batch sorted local ids
// (codegen/selection/row_id_space.hpp).
//
// The sync discipline is the point of this file, so it is worth stating once.
// The conversion is host-driven at exactly one place: batches are sliced and
// their buffers sized on the host, so the slice boundaries must come back.
// Everything else here is asynchronous, and the unique count rides home in that
// same copy rather than costing a second sync — which is why
// sort_unique_global_ids sizes for the worst case and leaves its count on
// device. A sync per step would be the natural way to write this and would put
// three of them on a path that needs one.

#include "codegen/selection/row_id_space.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::codegen {

namespace {

constexpr int kBlock = 256;

inline void throw_on_cuda(cudaError_t err, char const* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("row_id_space: ") + what + ": " + cudaGetErrorString(err));
  }
}

inline int grid_for(std::int64_t items, int per_block)
{
  std::int64_t g = (items + per_block - 1) / per_block;
  if (g < 1) g = 1;
  if (g > 4096) g = 4096;  // grid-stride covers the rest
  return static_cast<int>(g);
}

__global__ void iota_kernel(std::int32_t* __restrict__ out, std::int64_t count)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    out[i] = static_cast<std::int32_t>(i);
  }
}

// 1 where a sorted id differs from its predecessor — i.e. the first occurrence
// of each distinct value, which is the one the deduplicated array keeps.
__global__ void first_occurrence_kernel(std::uint64_t const* __restrict__ sorted_ids,
                                        std::int64_t count,
                                        std::uint32_t* __restrict__ flag)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    flag[i] = (i == 0 || sorted_ids[i] != sorted_ids[i - 1]) ? 1u : 0u;
  }
}

// Compact the distinct ids and record where every ORIGINAL element went.
//
// `rank` is the inclusive scan of the first-occurrence flags, so every element
// of a run of equal ids shares rank-1 — the slot that run collapses into. The
// scatter of the id is therefore idempotent (each member of the run writes the
// same value to the same slot) and needs no predicate, while the restore rank
// is written once per original element, at its pre-sort position.
__global__ void compact_and_restore_kernel(std::uint64_t const* __restrict__ sorted_ids,
                                           std::int32_t const* __restrict__ sorted_pos,
                                           std::uint32_t const* __restrict__ rank,
                                           std::int64_t count,
                                           std::uint64_t* __restrict__ unique_ids,
                                           std::int32_t* __restrict__ restore_rank,
                                           std::int32_t* __restrict__ count_dev)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    std::int32_t const slot     = static_cast<std::int32_t>(rank[i]) - 1;
    unique_ids[slot]            = sorted_ids[i];
    restore_rank[sorted_pos[i]] = slot;
    if (i == count - 1) { *count_dev = static_cast<std::int32_t>(rank[i]); }
  }
}

// One thread per batch boundary: the first index whose id is >= that batch's
// first row. Binary search rather than a pass over the ids — B is a handful
// and the id list is the long thing.
__global__ void lower_bound_kernel(std::uint64_t const* __restrict__ sorted_ids,
                                   std::int32_t const* __restrict__ count_dev,
                                   std::int64_t max_count,
                                   std::int64_t const* __restrict__ batch_row_start,
                                   std::int64_t num_bounds,
                                   std::int64_t* __restrict__ out_starts)
{
  auto const n = count_dev != nullptr ? static_cast<std::int64_t>(*count_dev) : max_count;
  for (std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < num_bounds;
       k += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    auto const target = static_cast<std::uint64_t>(batch_row_start[k]);
    std::int64_t lo = 0, hi = n;
    while (lo < hi) {
      std::int64_t const mid = lo + (hi - lo) / 2;
      if (sorted_ids[mid] < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out_starts[k] = lo;
  }
}

__global__ void to_local_kernel(std::uint64_t const* __restrict__ ids,
                                std::int64_t count,
                                std::int64_t batch_row_start,
                                std::int32_t* __restrict__ out_local)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    // Signed, and deliberately not clamped for small mismatches: an id from a nearby batch
    // lands outside [0, batch rows) in the int64 difference and the bucketer rejects it there.
    // But a difference outside int32 range would silently wrap back into range on the narrowing
    // cast below (e.g. 2^32 + 500 -> 500), which a pinned entry past 2^32 rows can reach. Range-
    // check the int64 difference first and emit a guaranteed-rejected sentinel instead of
    // truncating.
    auto const diff = static_cast<std::int64_t>(ids[i]) - batch_row_start;
    out_local[i]    = (diff >= std::numeric_limits<std::int32_t>::min() &&
                    diff <= std::numeric_limits<std::int32_t>::max())
                        ? static_cast<std::int32_t>(diff)
                        : std::int32_t{-1};
  }
}

}  // namespace

sorted_unique_ids sort_unique_global_ids(std::uint64_t const* ids,
                                         std::int64_t count,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  sorted_unique_ids out;
  out.original_count = count;
  out.count_dev      = rmm::device_buffer(sizeof(std::int32_t), stream, mr);
  throw_on_cuda(cudaMemsetAsync(out.count_dev.data(), 0, sizeof(std::int32_t), stream.value()),
                "count clear");
  if (count == 0) { return out; }
  if (count < 0 || ids == nullptr) {
    throw std::runtime_error("row_id_space: sort from an unbound id list");
  }
  if (count > static_cast<std::int64_t>(INT32_MAX)) {
    // The ranks are int32 because one materialized output is one cudf column.
    throw std::runtime_error("row_id_space: more ids than an int32 rank can address");
  }

  auto const n_u64 = static_cast<std::size_t>(count) * sizeof(std::uint64_t);
  auto const n_i32 = static_cast<std::size_t>(count) * sizeof(std::int32_t);
  out.ids          = rmm::device_buffer(n_u64, stream, mr);
  out.restore_rank = rmm::device_buffer(n_i32, stream, mr);

  rmm::device_buffer sorted_ids(n_u64, stream, mr);
  rmm::device_buffer pos(n_i32, stream, mr);
  rmm::device_buffer sorted_pos(n_i32, stream, mr);
  rmm::device_buffer flag(static_cast<std::size_t>(count) * sizeof(std::uint32_t), stream, mr);
  rmm::device_buffer rank(static_cast<std::size_t>(count) * sizeof(std::uint32_t), stream, mr);

  iota_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    static_cast<std::int32_t*>(pos.data()), count);
  throw_on_cuda(cudaPeekAtLastError(), "iota launch");

  std::size_t tmp_bytes = 0;
  throw_on_cuda(cub::DeviceRadixSort::SortPairs(nullptr,
                                                tmp_bytes,
                                                ids,
                                                static_cast<std::uint64_t*>(sorted_ids.data()),
                                                static_cast<std::int32_t const*>(pos.data()),
                                                static_cast<std::int32_t*>(sorted_pos.data()),
                                                static_cast<int>(count),
                                                0,
                                                64,
                                                stream.value()),
                "sort probe");
  rmm::device_buffer tmp(tmp_bytes, stream, mr);
  throw_on_cuda(cub::DeviceRadixSort::SortPairs(tmp.data(),
                                                tmp_bytes,
                                                ids,
                                                static_cast<std::uint64_t*>(sorted_ids.data()),
                                                static_cast<std::int32_t const*>(pos.data()),
                                                static_cast<std::int32_t*>(sorted_pos.data()),
                                                static_cast<int>(count),
                                                0,
                                                64,
                                                stream.value()),
                "sort");

  first_occurrence_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    static_cast<std::uint64_t const*>(sorted_ids.data()),
    count,
    static_cast<std::uint32_t*>(flag.data()));
  throw_on_cuda(cudaPeekAtLastError(), "first_occurrence launch");

  std::size_t scan_bytes = 0;
  throw_on_cuda(cub::DeviceScan::InclusiveSum(nullptr,
                                              scan_bytes,
                                              static_cast<std::uint32_t const*>(flag.data()),
                                              static_cast<std::uint32_t*>(rank.data()),
                                              static_cast<int>(count),
                                              stream.value()),
                "unique scan probe");
  rmm::device_buffer scan_tmp(scan_bytes, stream, mr);
  throw_on_cuda(cub::DeviceScan::InclusiveSum(scan_tmp.data(),
                                              scan_bytes,
                                              static_cast<std::uint32_t const*>(flag.data()),
                                              static_cast<std::uint32_t*>(rank.data()),
                                              static_cast<int>(count),
                                              stream.value()),
                "unique scan");

  compact_and_restore_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    static_cast<std::uint64_t const*>(sorted_ids.data()),
    static_cast<std::int32_t const*>(sorted_pos.data()),
    static_cast<std::uint32_t const*>(rank.data()),
    count,
    static_cast<std::uint64_t*>(out.ids.data()),
    static_cast<std::int32_t*>(out.restore_rank.data()),
    static_cast<std::int32_t*>(out.count_dev.data()));
  throw_on_cuda(cudaPeekAtLastError(), "compact_and_restore launch");

  // No sync: the transients are freed stream-ordered on the same stream, and
  // the count goes home in the caller's own boundary sync.
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
  if (batch_row_start.size() < 2) {
    throw std::runtime_error("row_id_space: split needs at least one batch");
  }
  auto const num_bounds = static_cast<std::int64_t>(batch_row_start.size());
  std::vector<std::int64_t> starts(static_cast<std::size_t>(num_bounds), 0);

  if (max_count == 0 || sorted_ids == nullptr) {
    if (count_out != nullptr) { *count_out = 0; }
    return starts;  // every batch's slice is empty, and all bounds are 0
  }

  auto const bounds_bytes = static_cast<std::size_t>(num_bounds) * sizeof(std::int64_t);
  rmm::device_buffer starts_dev(bounds_bytes, stream, mr);
  rmm::device_buffer batch_dev(bounds_bytes, stream, mr);
  throw_on_cuda(cudaMemcpyAsync(batch_dev.data(),
                                batch_row_start.data(),
                                bounds_bytes,
                                cudaMemcpyHostToDevice,
                                stream.value()),
                "batch starts H2D");

  lower_bound_kernel<<<grid_for(num_bounds, kBlock), kBlock, 0, stream.value()>>>(
    sorted_ids,
    count_dev,
    max_count,
    static_cast<std::int64_t const*>(batch_dev.data()),
    num_bounds,
    static_cast<std::int64_t*>(starts_dev.data()));
  throw_on_cuda(cudaPeekAtLastError(), "lower_bound launch");

  // THE sync. Both the boundaries and the unique count come back in it.
  std::int32_t unique_count = 0;
  throw_on_cuda(
    cudaMemcpyAsync(
      starts.data(), starts_dev.data(), bounds_bytes, cudaMemcpyDeviceToHost, stream.value()),
    "batch bounds D2H");
  if (count_dev != nullptr) {
    throw_on_cuda(
      cudaMemcpyAsync(
        &unique_count, count_dev, sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream.value()),
      "unique count D2H");
  }
  throw_on_cuda(cudaStreamSynchronize(stream.value()), "batch bounds sync");

  if (count_out != nullptr) {
    *count_out = count_dev != nullptr ? static_cast<std::int64_t>(unique_count) : max_count;
  }
  return starts;
}

void global_slice_to_local(std::uint64_t const* ids,
                           std::int64_t count,
                           std::int64_t batch_row_start,
                           std::int32_t* out_local,
                           rmm::cuda_stream_view stream)
{
  if (count == 0) { return; }
  if (count < 0 || ids == nullptr || out_local == nullptr) {
    throw std::runtime_error("row_id_space: local conversion over unbound buffers");
  }
  to_local_kernel<<<grid_for(count, kBlock), kBlock, 0, stream.value()>>>(
    ids, count, batch_row_start, out_local);
  throw_on_cuda(cudaPeekAtLastError(), "to_local launch");
}

}  // namespace sirius::codegen
