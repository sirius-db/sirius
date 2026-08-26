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

// Pass 1: in-chunk positions, chunk-boundary flags, and the input's own
// validity — all from the same load of row_ids, since each is a function of a
// row id and its predecessor.
//
// `bad` is set (never cleared) by any thread that sees an id out of range or
// out of order. Checking here rather than trusting the caller is the same
// argument the uint16 positions make: a post-join caller is exactly the one
// whose ordering we cannot verify by construction.
__global__ void row_ids_scan_kernel(std::int32_t const* __restrict__ row_ids,
                                    std::int64_t num_ids,
                                    std::int64_t num_rows,
                                    std::uint16_t* __restrict__ in_chunk_rows,
                                    std::uint32_t* __restrict__ boundary,
                                    std::uint32_t* __restrict__ bad)
{
  auto const chunk_rows = static_cast<std::int64_t>(::codegen::kChunkSize);
  auto const stride     = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_ids;
       i += stride) {
    std::int64_t const id = row_ids[i];
    if (id < 0 || id >= num_rows) {
      // Every derived value below would be out of bounds. Write the neutral one
      // anyway: the scan runs before the host learns of this, and it must not
      // read a flag that was never stored.
      *bad             = 1u;
      in_chunk_rows[i] = 0u;
      boundary[i]      = 0u;
      continue;
    }
    std::int64_t const chunk = id / chunk_rows;
    in_chunk_rows[i]         = static_cast<std::uint16_t>(id - chunk * chunk_rows);
    if (i == 0) {
      boundary[i] = 1u;
    } else {
      std::int64_t const prev = row_ids[i - 1];
      if (prev >= id) { *bad = 1u; }  // a repeat would decode the row twice
      boundary[i] = (prev / chunk_rows != chunk) ? 1u : 0u;
    }
  }
}

// Pass 3: one entry per boundary. `rank` is the inclusive scan of the flags, so
// a boundary at i is block rank[i]-1, and its slice starts at i. The last id
// closes the CSR: block_offsets[T] = S.
__global__ void scatter_blocks_kernel(std::int32_t const* __restrict__ row_ids,
                                      std::int64_t num_ids,
                                      std::uint32_t const* __restrict__ boundary,
                                      std::uint32_t const* __restrict__ rank,
                                      std::uint32_t* __restrict__ chunk_ids,
                                      std::uint32_t* __restrict__ block_offsets)
{
  auto const chunk_rows = static_cast<std::int64_t>(::codegen::kChunkSize);
  auto const stride     = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_ids;
       i += stride) {
    if (boundary[i] != 0u) {
      std::uint32_t const b = rank[i] - 1u;
      chunk_ids[b]          = static_cast<std::uint32_t>(row_ids[i] / chunk_rows);
      block_offsets[b]      = static_cast<std::uint32_t>(i);
    }
    if (i == num_ids - 1) { block_offsets[rank[i]] = static_cast<std::uint32_t>(num_ids); }
  }
}

}  // namespace

chunk_row_set_owner build_chunk_row_set(std::int32_t const* row_ids,
                                        std::int64_t num_ids,
                                        std::int64_t num_rows,
                                        rmm::cuda_stream_view stream,
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

  // Boundary flags and their inclusive scan. Both are transient, and the
  // scatter still reads them AFTER the D2H sync below, so what makes freeing
  // them on return safe is the stream-ordered deallocation — not that sync.
  rmm::device_buffer boundary_buf(
    static_cast<std::size_t>(num_ids) * sizeof(std::uint32_t), stream, mr);
  rmm::device_buffer rank_buf(
    static_cast<std::size_t>(num_ids) * sizeof(std::uint32_t), stream, mr);
  rmm::device_buffer bad_buf(sizeof(std::uint32_t), stream, mr);
  auto* boundary = static_cast<std::uint32_t*>(boundary_buf.data());
  auto* rank     = static_cast<std::uint32_t*>(rank_buf.data());
  auto* bad      = static_cast<std::uint32_t*>(bad_buf.data());
  throw_on_cuda(cudaMemsetAsync(bad, 0, sizeof(std::uint32_t), stream.value()), "bad flag clear");

  row_ids_scan_kernel<<<grid_for(num_ids, kBlock), kBlock, 0, stream.value()>>>(
    row_ids,
    num_ids,
    num_rows,
    static_cast<std::uint16_t*>(out.in_chunk_rows.data()),
    boundary,
    bad);
  throw_on_cuda(cudaPeekAtLastError(), "row_ids_scan launch");

  std::size_t tmp_bytes = 0;
  throw_on_cuda(cub::DeviceScan::InclusiveSum(
                  nullptr, tmp_bytes, boundary, rank, static_cast<int>(num_ids), stream.value()),
                "boundary scan probe");
  rmm::device_buffer tmp(tmp_bytes, stream, mr);
  throw_on_cuda(cub::DeviceScan::InclusiveSum(
                  tmp.data(), tmp_bytes, boundary, rank, static_cast<int>(num_ids), stream.value()),
                "boundary scan");

  // The one host sync: T is the grid, and a grid is a host-side value. The
  // validity flag rides along on the same sync rather than costing a second.
  std::uint32_t touched = 0;
  std::uint32_t invalid = 0;
  throw_on_cuda(cudaMemcpyAsync(&touched,
                                rank + (num_ids - 1),
                                sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()),
                "num_touched D2H");
  throw_on_cuda(
    cudaMemcpyAsync(&invalid, bad, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream.value()),
    "validity D2H");
  throw_on_cuda(cudaStreamSynchronize(stream.value()), "num_touched sync");

  if (invalid != 0u) {
    throw std::runtime_error(
      "chunk_row_set_build: row ids must be strictly increasing and within the batch");
  }

  out.num_touched = static_cast<std::int64_t>(touched);
  out.chunk_ids =
    rmm::device_buffer(static_cast<std::size_t>(touched) * sizeof(std::uint32_t), stream, mr);
  out.block_offsets =
    rmm::device_buffer((static_cast<std::size_t>(touched) + 1) * sizeof(std::uint32_t), stream, mr);

  scatter_blocks_kernel<<<grid_for(num_ids, kBlock), kBlock, 0, stream.value()>>>(
    row_ids,
    num_ids,
    boundary,
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
