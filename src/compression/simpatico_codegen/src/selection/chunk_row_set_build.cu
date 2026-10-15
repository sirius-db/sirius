// SPDX-License-Identifier: Apache-2.0
//
// chunk_row_set_build.cu — bucket a post-join selection into the chunk CSR
// (codegen/selection/chunk_row_set.hpp).
//
// The selection wave next door builds its enumerations FROM a mask, so its work
// is naturally per-chunk: it counts every chunk, scans every chunk, and the
// index list falls out of that scan for free. A selection that arrives after
// the scan has no mask, and per-chunk work would be the wrong shape for it —
// a join leaving 444k rows over 63k of a batch's chunks must not pay for the
// millions of chunks it does not touch. So everything here is O(S) in the
// survivors, and the batch's chunk count appears only in a bounds check.
//
// Ascending ids make that possible: a chunk boundary is a local test between
// neighbours, so the touched chunks are the boundaries, their count is one
// scan, and the CSR is a scatter to the scan's ranks. Three passes over S, no
// pass over C, no atomics, no per-chunk arena to allocate and zero.
//
// The one host sync is unavoidable and deliberate: num_touched IS the grid the
// launcher will use, and a grid is a host-side launch parameter. Same shape as
// run_selection_cnt's survivor_count sync, and for the same reason.

#include "codegen/selection/chunk_row_set.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::codegen {

namespace {

constexpr int kBlock = 256;

inline void throw_on_cuda(cudaError_t err, char const* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("chunk_row_set_build: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

inline int grid_for(std::int64_t items, int per_block)
{
  std::int64_t g = (items + per_block - 1) / per_block;
  if (g < 1) g = 1;
  if (g > 4096) g = 4096;  // grid-stride covers the rest
  return static_cast<int>(g);
}

// Pass 1: in-chunk positions and the input's own validity — both from the same
// load of row_ids, since each is a function of a row id and its predecessor.
// The chunk-boundary flags are not materialised: the scan reads them through
// boundary_at, and the scatter recovers them from rank steps.
//
// `bad` is set (never cleared) by any thread that sees an id out of range or
// out of order. Checking here rather than trusting the caller is the same
// argument the uint16 positions make: a post-join caller is exactly the one
// whose ordering we cannot verify by construction.
__device__ __forceinline__ std::uint16_t scan_one(
  std::int32_t id, std::int32_t prev, bool has_prev, std::int64_t num_rows, std::uint32_t* bad)
{
  if (id < 0 || id >= num_rows) {
    // Every derived value below would be out of bounds. Use the neutral one
    // anyway: the scan runs before the host learns of this.
    *bad = 1u;
    return 0u;
  }
  if (has_prev && prev >= id) { *bad = 1u; }  // a repeat would decode the row twice
  return static_cast<std::uint16_t>(id & (::codegen::kChunkSize - 1));
}

// Four ids per thread when row_ids and in_chunk_rows are vector-aligned (16 B / 8 B): one 16-byte
// load, one 8-byte store. Otherwise, and for the sub-quad tail, one id per iteration.
__global__ void row_ids_scan_kernel(std::int32_t const* __restrict__ row_ids,
                                    std::int64_t num_ids,
                                    std::int64_t num_rows,
                                    std::uint16_t* __restrict__ in_chunk_rows,
                                    std::uint32_t* __restrict__ bad)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  auto const tid    = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool const vec    = ((reinterpret_cast<std::uintptr_t>(row_ids) & 15u) == 0) &&
                   ((reinterpret_cast<std::uintptr_t>(in_chunk_rows) & 7u) == 0);
  std::int64_t const scalar_from = vec ? (num_ids & ~std::int64_t{3}) : 0;
  if (vec) {
    for (std::int64_t i = tid * 4; i < scalar_from; i += stride * 4) {
      int4 const v            = *reinterpret_cast<int4 const*>(row_ids + i);
      std::int32_t const prev = i != 0 ? row_ids[i - 1] : 0;
      ushort4 out;
      out.x                                          = scan_one(v.x, prev, i != 0, num_rows, bad);
      out.y                                          = scan_one(v.y, v.x, true, num_rows, bad);
      out.z                                          = scan_one(v.z, v.y, true, num_rows, bad);
      out.w                                          = scan_one(v.w, v.z, true, num_rows, bad);
      *reinterpret_cast<ushort4*>(in_chunk_rows + i) = out;
    }
  }
  for (std::int64_t i = scalar_from + tid; i < num_ids; i += stride) {
    in_chunk_rows[i] = scan_one(row_ids[i], i != 0 ? row_ids[i - 1] : 0, i != 0, num_rows, bad);
  }
}

// 1 where row_ids[i] opens a new chunk (always at i == 0), else 0. Fed to the boundary scan as a
// transform iterator, so the flags never exist in memory. Out-of-range ids give a meaningless but
// harmless value: pass 1 has flagged them and the build throws before the result is used.
struct boundary_at {
  std::int32_t const* row_ids;
  __host__ __device__ std::uint32_t operator()(std::int64_t i) const
  {
    constexpr int kShift = 10;
    static_assert((1 << kShift) == ::codegen::kChunkSize, "chunk size must be 2^10");
    if (i == 0) return 1u;
    return (row_ids[i - 1] >> kShift) != (row_ids[i] >> kShift) ? 1u : 0u;
  }
};

// Pass 3: one entry per boundary. `rank` is the inclusive scan of the flags, so
// a boundary at i is where rank steps up (or i == 0): block rank[i]-1, whose slice starts at i. The
// last id closes the CSR: block_offsets[T] = S.
__device__ __forceinline__ void scatter_one(std::int64_t i,
                                            std::uint32_t r,
                                            std::uint32_t prev_r,
                                            std::int32_t const* row_ids,
                                            std::uint32_t* chunk_ids,
                                            std::uint32_t* block_offsets)
{
  if (i == 0 || r != prev_r) {
    std::uint32_t const b = r - 1u;
    chunk_ids[b]          = static_cast<std::uint32_t>(row_ids[i] >> 10);
    block_offsets[b]      = static_cast<std::uint32_t>(i);
  }
}

// Four ranks per thread when rank is 16-byte aligned; scalar otherwise and for the tail.
__global__ void scatter_blocks_kernel(std::int32_t const* __restrict__ row_ids,
                                      std::int64_t num_ids,
                                      std::uint32_t const* __restrict__ rank,
                                      std::uint32_t* __restrict__ chunk_ids,
                                      std::uint32_t* __restrict__ block_offsets)
{
  static_assert((1 << 10) == ::codegen::kChunkSize, "chunk size must be 2^10");
  auto const stride              = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  auto const tid                 = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool const vec                 = (reinterpret_cast<std::uintptr_t>(rank) & 15u) == 0;
  std::int64_t const scalar_from = vec ? (num_ids & ~std::int64_t{3}) : 0;
  if (vec) {
    for (std::int64_t i = tid * 4; i < scalar_from; i += stride * 4) {
      uint4 const r            = *reinterpret_cast<uint4 const*>(rank + i);
      std::uint32_t const prev = i != 0 ? rank[i - 1] : 0u;
      scatter_one(i, r.x, prev, row_ids, chunk_ids, block_offsets);
      scatter_one(i + 1, r.y, r.x, row_ids, chunk_ids, block_offsets);
      scatter_one(i + 2, r.z, r.y, row_ids, chunk_ids, block_offsets);
      scatter_one(i + 3, r.w, r.z, row_ids, chunk_ids, block_offsets);
    }
  }
  for (std::int64_t i = scalar_from + tid; i < num_ids; i += stride) {
    scatter_one(i, rank[i], i != 0 ? rank[i - 1] : 0u, row_ids, chunk_ids, block_offsets);
  }
  if (tid == 0) { block_offsets[rank[num_ids - 1]] = static_cast<std::uint32_t>(num_ids); }
}

}  // namespace

chunk_row_set_owner build_chunk_row_set(std::int32_t const* row_ids,
                                        std::int64_t num_ids,
                                        std::int64_t num_rows,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  if (num_rows <= 0) {
    throw std::runtime_error("chunk_row_set_build: build over a batch of no rows");
  }
  chunk_row_set_owner out;
  out.num_rows = num_rows;
  if (num_ids == 0) { return out; }  // an empty selection needs no arrays
  if (num_ids < 0 || row_ids == nullptr) {
    throw std::runtime_error("chunk_row_set_build: build from an unbound id list");
  }
  if (num_ids > static_cast<std::int64_t>(INT32_MAX)) {
    // block_offsets is uint32, and CUB's scan counts in int — the tighter of
    // the two is the limit, so the guard is the one the code actually relies on.
    throw std::runtime_error("chunk_row_set_build: more ids than the scan can address");
  }

  out.num_survivors = num_ids;
  out.in_chunk_rows =
    rmm::device_buffer(static_cast<std::size_t>(num_ids) * sizeof(std::uint16_t), stream, mr);

  // Inclusive scan of the boundary flags. Transient, and the scatter still
  // reads it AFTER the D2H sync below, so what makes freeing it on return safe
  // is the stream-ordered deallocation — not that sync.
  rmm::device_buffer rank_buf(
    static_cast<std::size_t>(num_ids) * sizeof(std::uint32_t), stream, mr);
  rmm::device_buffer bad_buf(sizeof(std::uint32_t), stream, mr);
  auto* rank = static_cast<std::uint32_t*>(rank_buf.data());
  auto* bad  = static_cast<std::uint32_t*>(bad_buf.data());
  throw_on_cuda(cudaMemsetAsync(bad, 0, sizeof(std::uint32_t), stream.get()), "bad flag clear");

  row_ids_scan_kernel<<<grid_for((num_ids + 3) / 4, kBlock), kBlock, 0, stream.get()>>>(
    row_ids, num_ids, num_rows, static_cast<std::uint16_t*>(out.in_chunk_rows.data()), bad);
  throw_on_cuda(cudaPeekAtLastError(), "row_ids_scan launch");

  auto const boundary = thrust::make_transform_iterator(
    thrust::make_counting_iterator<std::int64_t>(0), boundary_at{row_ids});
  std::size_t tmp_bytes = 0;
  throw_on_cuda(cub::DeviceScan::InclusiveSum(
                  nullptr, tmp_bytes, boundary, rank, static_cast<int>(num_ids), stream.get()),
                "boundary scan probe");
  rmm::device_buffer tmp(tmp_bytes, stream, mr);
  throw_on_cuda(cub::DeviceScan::InclusiveSum(
                  tmp.data(), tmp_bytes, boundary, rank, static_cast<int>(num_ids), stream.get()),
                "boundary scan");

  // The one host sync: T is the grid, and a grid is a host-side value. The
  // validity flag rides along on the same sync rather than costing a second.
  std::uint32_t touched = 0;
  std::uint32_t invalid = 0;
  throw_on_cuda(
    cudaMemcpyAsync(
      &touched, rank + (num_ids - 1), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream.get()),
    "num_touched D2H");
  throw_on_cuda(
    cudaMemcpyAsync(&invalid, bad, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream.get()),
    "validity D2H");
  throw_on_cuda(cudaStreamSynchronize(stream.get()), "num_touched sync");

  if (invalid != 0u) {
    throw std::runtime_error(
      "chunk_row_set_build: row ids must be strictly increasing and within the batch");
  }

  out.num_touched = static_cast<std::int64_t>(touched);
  out.chunk_ids =
    rmm::device_buffer(static_cast<std::size_t>(touched) * sizeof(std::uint32_t), stream, mr);
  out.block_offsets =
    rmm::device_buffer((static_cast<std::size_t>(touched) + 1) * sizeof(std::uint32_t), stream, mr);

  scatter_blocks_kernel<<<grid_for((num_ids + 3) / 4, kBlock), kBlock, 0, stream.get()>>>(
    row_ids,
    num_ids,
    rank,
    static_cast<std::uint32_t*>(out.chunk_ids.data()),
    static_cast<std::uint32_t*>(out.block_offsets.data()));
  throw_on_cuda(cudaPeekAtLastError(), "scatter_blocks launch");

  // The transient buffers are freed as this returns, on the same stream the
  // scatter was enqueued on, so the stream-ordered deallocation already happens
  // after the scatter has read them. No second host sync for that.
  if (!out.view().valid()) {
    throw std::runtime_error("chunk_row_set_build: built a row set that fails its own contract");
  }
  return out;
}

}  // namespace sirius::codegen
