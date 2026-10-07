// SPDX-License-Identifier: Apache-2.0
//
// chunk_row_set_convert.cu — deriving the mask enumeration from a chunk CSR
// (codegen/selection/chunk_row_set.hpp).
//
// A selection arriving after the scan can only be BUILT as a CSR, since the
// mask form is a by-product of a mask such a selection does not have. Deriving
// the mask here is what lets the CSR be the only construction path, and it is
// how a plan with no random-access decode is served at all.
//
// One block per touched chunk, matching the grid the sparse decode itself
// uses: a block already owns a contiguous slice of the CSR, so it sets that
// chunk's mask bits with no cross-block coordination.

#include "codegen/selection/chunk_row_set.hpp"
#include "codegen/selection/selection.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::codegen {

namespace {

constexpr int kBlock = 256;

inline void throw_on_cuda(cudaError_t err, char const* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("chunk_row_set_convert: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

// Set this chunk's mask words, and record how many survivors it has.
//
// One warp per touched chunk: a chunk typically holds only tens to hundreds of survivors, far too
// little for a 256-thread block. The warp assembles the chunk's 32 mask words in a warp-private
// shared buffer (shared atomics contend only within the warp) and writes them out as one coalesced
// 128-byte store, instead of one global atomicOr per survivor. The warp owns the chunk's words, so
// it writes all 32, zeros included. The count is a single plain store, since exactly one warp owns
// each touched chunk.
__global__ void to_mask_kernel(std::uint32_t const* __restrict__ chunk_ids,
                               std::uint32_t const* __restrict__ block_offsets,
                               std::uint16_t const* __restrict__ in_chunk_rows,
                               std::int64_t num_touched,
                               std::uint32_t* __restrict__ mask_words,
                               std::uint32_t* __restrict__ counts)
{
  constexpr int kWordsPerChunk = ::codegen::kChunkSize / 32;
  static_assert(kWordsPerChunk == 32, "one lane per mask word");
  extern __shared__ std::uint32_t smem_words[];
  int const lane             = threadIdx.x & 31;
  std::uint32_t* const words = smem_words + (threadIdx.x >> 5) * kWordsPerChunk;
  auto const warps_per_grid  = (static_cast<std::int64_t>(gridDim.x) * blockDim.x) >> 5;
  for (auto b = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
       b < num_touched;
       b += warps_per_grid) {
    auto const chunk = static_cast<std::int64_t>(chunk_ids[b]);
    auto const lo    = block_offsets[b];
    auto const hi    = block_offsets[b + 1];
    words[lane]      = 0u;
    __syncwarp();
    for (std::uint32_t k = lo + lane; k < hi; k += 32) {
      std::uint32_t const pos = in_chunk_rows[k];
      atomicOr(&words[pos / 32], 1u << (pos % 32));
    }
    __syncwarp();
    mask_words[chunk * kWordsPerChunk + lane] = words[lane];
    if (lane == 0) { counts[chunk] = hi - lo; }
    __syncwarp();
  }
}

__global__ void offsets_tail_kernel(std::uint32_t const* __restrict__ counts,
                                    std::int64_t num_chunks,
                                    std::uint32_t* __restrict__ offsets)
{
  offsets[num_chunks] = offsets[num_chunks - 1] + counts[num_chunks - 1];
}

}  // namespace

void row_set_to_mask(chunk_row_set const& rows,
                     std::uint32_t* mask_words,
                     std::uint32_t* all_chunk_offsets,
                     ::cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
{
  if (mask_words == nullptr || all_chunk_offsets == nullptr || rows.num_rows <= 0) {
    throw std::runtime_error("chunk_row_set_convert: mask expansion over unbound buffers");
  }
  auto const num_chunks = selection_mask::ChunksFor(rows.num_rows);
  auto const num_words  = selection_mask::WordsFor(rows.num_rows);

  // Untouched chunks and the tail have to be written, not left: a mask
  // consumer reads every word of every strip, so anything not cleared here is
  // read as a survivor that does not exist.
  throw_on_cuda(
    cudaMemsetAsync(
      mask_words, 0, static_cast<std::size_t>(num_words) * sizeof(std::uint32_t), stream.get()),
    "mask clear");

  rmm::device_buffer counts(
    static_cast<std::size_t>(num_chunks) * sizeof(std::uint32_t), stream, mr);
  throw_on_cuda(
    cudaMemsetAsync(
      counts.data(), 0, static_cast<std::size_t>(num_chunks) * sizeof(std::uint32_t), stream.get()),
    "counts clear");

  if (rows.num_survivors > 0) {
    if (!rows.valid()) {
      throw std::runtime_error("chunk_row_set_convert: mask expansion from an invalid row set");
    }
    constexpr int kWarpsPerBlock = kBlock / 32;
    auto const grid              = static_cast<unsigned>(
      std::min<std::int64_t>((rows.num_touched + kWarpsPerBlock - 1) / kWarpsPerBlock, 65535));
    to_mask_kernel<<<grid,
                     kBlock,
                     kWarpsPerBlock*(::codegen::kChunkSize / 32) * sizeof(std::uint32_t),
                     stream.get()>>>(rows.chunk_ids,
                                     rows.block_offsets,
                                     rows.in_chunk_rows,
                                     rows.num_touched,
                                     mask_words,
                                     static_cast<std::uint32_t*>(counts.data()));
    throw_on_cuda(cudaPeekAtLastError(), "to_mask launch");
  }

  std::size_t tmp_bytes = 0;
  throw_on_cuda(cub::DeviceScan::ExclusiveSum(nullptr,
                                              tmp_bytes,
                                              static_cast<std::uint32_t const*>(counts.data()),
                                              all_chunk_offsets,
                                              static_cast<int>(num_chunks),
                                              stream.get()),
                "chunk offsets scan probe");
  rmm::device_buffer tmp(tmp_bytes, stream, mr);
  throw_on_cuda(cub::DeviceScan::ExclusiveSum(tmp.data(),
                                              tmp_bytes,
                                              static_cast<std::uint32_t const*>(counts.data()),
                                              all_chunk_offsets,
                                              static_cast<int>(num_chunks),
                                              stream.get()),
                "chunk offsets scan");
  offsets_tail_kernel<<<1, 1, 0, stream.get()>>>(
    static_cast<std::uint32_t const*>(counts.data()), num_chunks, all_chunk_offsets);
  throw_on_cuda(cudaPeekAtLastError(), "chunk offsets tail launch");
}

}  // namespace sirius::codegen
