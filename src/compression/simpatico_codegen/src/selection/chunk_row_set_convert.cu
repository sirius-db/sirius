// SPDX-License-Identifier: Apache-2.0
//
// chunk_row_set_convert.cu — deriving the mask and index-list enumerations
// from a chunk CSR (codegen/selection/chunk_row_set.hpp).
//
// One selection, three shapes. The decode's consumers each take whichever they
// are handed, so nothing forces a choice — but a selection arriving after the
// scan can only be BUILT as a CSR, since the other two are by-products of a
// mask that a post-join selection does not have. Deriving them here is what
// lets that be the only construction path, rather than keeping a second one
// alive for the selections that happen to come with a mask.
//
// Both directions launch one block per touched chunk, matching the grid the
// sparse decode itself uses: a block already owns a contiguous slice of the
// CSR, so it can write that slice's indices or set that chunk's mask bits with
// no cross-block coordination.

#include "codegen/selection/chunk_row_set.hpp"
#include "codegen/selection/selection.hpp"

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
    throw std::runtime_error(std::string("chunk_row_set_convert: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

// Block b owns chunk chunk_ids[b] and the CSR slice
// [block_offsets[b], block_offsets[b+1]) — the same division of labour the
// sparse decode makes, so the output slot is the CSR position itself.
__global__ void to_indices_kernel(std::uint32_t const* __restrict__ chunk_ids,
                                  std::uint32_t const* __restrict__ block_offsets,
                                  std::uint16_t const* __restrict__ in_chunk_rows,
                                  std::int32_t* __restrict__ out)
{
  auto const b     = static_cast<std::int64_t>(blockIdx.x);
  auto const chunk = static_cast<std::int64_t>(chunk_ids[b]) * ::codegen::kChunkSize;
  for (std::uint32_t k = block_offsets[b] + threadIdx.x; k < block_offsets[b + 1];
       k += blockDim.x) {
    out[k] = static_cast<std::int32_t>(chunk + in_chunk_rows[k]);
  }
}

// Set this chunk's mask bits, and record how many survivors it has.
//
// The bit writes are atomic because a 32-bit word covers 32 rows and a chunk's
// survivors are spread over the block's threads; the count is a single plain
// store, since exactly one block owns each touched chunk.
__global__ void to_mask_kernel(std::uint32_t const* __restrict__ chunk_ids,
                               std::uint32_t const* __restrict__ block_offsets,
                               std::uint16_t const* __restrict__ in_chunk_rows,
                               std::uint32_t* __restrict__ mask_words,
                               std::uint32_t* __restrict__ counts)
{
  auto const b         = static_cast<std::int64_t>(blockIdx.x);
  auto const chunk     = static_cast<std::int64_t>(chunk_ids[b]);
  auto const word_base = chunk * (::codegen::kChunkSize / 32);
  auto const lo        = block_offsets[b];
  auto const hi        = block_offsets[b + 1];
  for (std::uint32_t k = lo + threadIdx.x; k < hi; k += blockDim.x) {
    std::uint32_t const pos = in_chunk_rows[k];
    atomicOr(&mask_words[word_base + pos / 32], 1u << (pos % 32));
  }
  if (threadIdx.x == 0) { counts[chunk] = hi - lo; }
}

__global__ void offsets_tail_kernel(std::uint32_t const* __restrict__ counts,
                                    std::int64_t num_chunks,
                                    std::uint32_t* __restrict__ offsets)
{
  offsets[num_chunks] = offsets[num_chunks - 1] + counts[num_chunks - 1];
}

}  // namespace

void row_set_to_local_indices(chunk_row_set const& rows,
                              std::int32_t* out,
                              rmm::cuda_stream_view stream)
{
  if (rows.num_survivors == 0) { return; }
  if (!rows.valid() || out == nullptr) {
    throw std::runtime_error("chunk_row_set_convert: expand from an invalid row set");
  }
  to_indices_kernel<<<static_cast<unsigned>(rows.num_touched), kBlock, 0, stream.value()>>>(
    rows.chunk_ids, rows.block_offsets, rows.in_chunk_rows, out);
  throw_on_cuda(cudaPeekAtLastError(), "to_indices launch");
}

void row_set_to_mask(chunk_row_set const& rows,
                     std::uint32_t* mask_words,
                     std::uint32_t* all_chunk_offsets,
                     rmm::cuda_stream_view stream,
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
      mask_words, 0, static_cast<std::size_t>(num_words) * sizeof(std::uint32_t), stream.value()),
    "mask clear");

  rmm::device_buffer counts(
    static_cast<std::size_t>(num_chunks) * sizeof(std::uint32_t), stream, mr);
  throw_on_cuda(cudaMemsetAsync(counts.data(),
                                0,
                                static_cast<std::size_t>(num_chunks) * sizeof(std::uint32_t),
                                stream.value()),
                "counts clear");

  if (rows.num_survivors > 0) {
    if (!rows.valid()) {
      throw std::runtime_error("chunk_row_set_convert: mask expansion from an invalid row set");
    }
    to_mask_kernel<<<static_cast<unsigned>(rows.num_touched), kBlock, 0, stream.value()>>>(
      rows.chunk_ids,
      rows.block_offsets,
      rows.in_chunk_rows,
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
                                              stream.value()),
                "chunk offsets scan probe");
  rmm::device_buffer tmp(tmp_bytes, stream, mr);
  throw_on_cuda(cub::DeviceScan::ExclusiveSum(tmp.data(),
                                              tmp_bytes,
                                              static_cast<std::uint32_t const*>(counts.data()),
                                              all_chunk_offsets,
                                              static_cast<int>(num_chunks),
                                              stream.value()),
                "chunk offsets scan");
  offsets_tail_kernel<<<1, 1, 0, stream.value()>>>(
    static_cast<std::uint32_t const*>(counts.data()), num_chunks, all_chunk_offsets);
  throw_on_cuda(cudaPeekAtLastError(), "chunk offsets tail launch");
}

}  // namespace sirius::codegen
