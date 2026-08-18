// SPDX-License-Identifier: Apache-2.0
//
// selection_wave.cu — device helpers for the row-selection wave
// (SIRIUS_EXP_FUSED_SCAN_FILTER). Three pieces, all mask-shaped (1 bit/row,
// 32 uint32 words per 1024-row chunk, tail bits zero):
//
//   * combine_masks_and : AND k per-column ballot masks into one batch mask.
//   * run_selection_cnt : per-chunk popcount + CUB exclusive scan ->
//                         chunk_offsets (the mask walk's per-chunk output bases) +
//                         survivor_count D2H (the one host sync — it gates
//                         wave-2 allocations).
//   * mask_to_row_indices : mask -> ascending int32 survivor rows (the TierB
//                         cudf::gather map, built once per batch).
//
// CNT's shape is deliberate and was measured: one block per chunk is the
// obvious form and the slow one; it is word-per-thread grid-stride with a warp
// reduce instead, which works because a warp's 32 lanes cover exactly one
// chunk's 32 mask words.
//
// WHY THESE ARE NOT cudf CALLS. cudf has an apparent equivalent for each
// (bitmask_and, bools_to_mask, segmented_count_set_bits, stream compaction) and
// they were evaluated; none fits, for one structural reason. These four are
// IN-PLACE, CHUNK-SEGMENTED operations over a caller-owned arena, and cudf's
// bitmask API is allocating and whole-column:
//
//   * bitmask_and and bools_to_mask return a fresh device_buffer. Every call
//     here writes into a pre-planned slot — and the combine deliberately
//     aliases its destination onto source 0 — so using them would add a
//     per-batch allocation on the hot path.
//   * they also size by bit count, not by our padded strip, so the tail words
//     a mask must own (and must have zeroed) would be neither allocated nor
//     written. That invariant is load-bearing: the count and the gather map
//     read the full strip.
//   * segmented_count_set_bits returns a HOST vector, i.e. a D2H of every
//     per-chunk count. run_selection_cnt keeps the counts and their scan on
//     device and copies back 4 bytes.
//   * a thrust::copy_if for the row ids would redo the prefix sum that
//     chunk_offsets already holds.
//
// The closest call is combine_masks_and: cudf::detail::inplace_bitmask_and does
// write into a caller-owned destination. It is an internal detail header
// (Sirius depends on cudf/detail in only three files), it takes per-mask bit
// offsets we never use and returns a set-bit count we do not want, and it would
// replace twenty lines of uint4-vectorized code. Judged not worth the coupling
// — but it is the one worth revisiting if cudf promotes it.

#include "codegen/selection/selection.hpp"

#include <rmm/device_buffer.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::codegen {

namespace {

constexpr int kBlock         = 256;  // threads per block, all kernels here
constexpr int kWordsPerChunk = 32;   // 1024 rows / 32 bits
constexpr unsigned kFullWarp = 0xffffffffu;

inline void throw_on_cuda(cudaError_t err, char const* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("selection_wave: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

inline int grid_for(int64_t items, int per_block)
{
  int64_t g = (items + per_block - 1) / per_block;
  if (g < 1) g = 1;
  if (g > 4096) g = 4096;  // grid-stride covers the rest
  return static_cast<int>(g);
}

// Up-to-8 source masks, passed by value as kernel params.
struct mask_src_ptrs {
  uint32_t const* p[8];
};

// dst = AND of n srcs, uint4-vectorized (num_words is a multiple of 4: masks
// are 32 words per chunk). Grid-stride.
//
// dst is NOT __restrict__: combine_masks_and's caller aliases it against
// srcs.p[0] by design (an in-place AND-into-the-first-source), so promising
// no-alias here would be a lie the compiler is free to act on. Each thread
// only ever reads then writes its own quad index, so the aliasing is benign
// in practice, but the qualifier's absence keeps that true by construction
// rather than by accident of today's codegen.
__global__ void mask_and_combine_kernel(uint32_t* dst,
                                        mask_src_ptrs srcs,
                                        int num_srcs,
                                        int64_t num_quads)
{
  auto* d4             = reinterpret_cast<uint4*>(dst);
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t q = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; q < num_quads;
       q += stride) {
    uint4 v = reinterpret_cast<uint4 const*>(srcs.p[0])[q];
    for (int s = 1; s < num_srcs; ++s) {
      uint4 const w = reinterpret_cast<uint4 const*>(srcs.p[s])[q];
      v.x &= w.x;
      v.y &= w.y;
      v.z &= w.z;
      v.w &= w.w;
    }
    d4[q] = v;
  }
}

// Per-chunk popcount: word-per-thread, one warp covers one chunk's 32 words,
// warp shfl reduce, lane 0 writes counts[chunk]. Grid-stride over chunks.
__global__ void chunk_popcount_kernel(uint32_t const* __restrict__ words,
                                      int64_t num_chunks,
                                      uint32_t* __restrict__ counts)
{
  int const lane               = threadIdx.x & 31;
  int64_t const warps_per_grid = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5;
  int64_t warp                 = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  for (; warp < num_chunks; warp += warps_per_grid) {
    unsigned pc = __popc(words[warp * kWordsPerChunk + lane]);
    for (int o = 16; o; o >>= 1)
      pc += __shfl_down_sync(kFullWarp, pc, o);
    if (lane == 0) counts[warp] = pc;
  }
}

// Sentinel: chunk_offsets[nc] = chunk_offsets[nc-1] + counts[nc-1]. One thread;
// avoids a D2H round-trip for the tail (same idiom as bp_offsets_tail_kernel
// in src/bridge/offsets_cumsum.cu).
__global__ void chunk_offsets_tail_kernel(uint32_t const* __restrict__ counts,
                                          int64_t num_chunks,
                                          uint32_t* __restrict__ offsets)
{
  offsets[num_chunks] = offsets[num_chunks - 1] + counts[num_chunks - 1];
}

// mask -> ascending survivor row ids. One warp per chunk: lane l owns word l,
// warp-exclusive shfl_up prefix of popcounts gives each word's output base
// inside the chunk; bits are drained in ascending order so the global output
// is fully ordered (cudf gather map contract). Ported from the microbench's
// load_mask_scan/k2a idiom, flattened to warp shuffles (no smem, no block sync).
__global__ void mask_to_indices_kernel(uint32_t const* __restrict__ words,
                                       uint32_t const* __restrict__ chunk_offsets,
                                       int64_t num_chunks,
                                       int32_t* __restrict__ out)
{
  int const lane               = threadIdx.x & 31;
  int64_t const warps_per_grid = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5;
  int64_t warp                 = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  for (; warp < num_chunks; warp += warps_per_grid) {
    uint32_t wv  = words[warp * kWordsPerChunk + lane];
    int const pc = __popc(wv);
    int x        = pc;
    for (int o = 1; o < 32; o <<= 1) {
      int const y = __shfl_up_sync(kFullWarp, x, o);
      if (lane >= o) x += y;
    }
    int64_t base       = chunk_offsets[warp] + static_cast<int64_t>(x - pc);
    int32_t const row0 = static_cast<int32_t>(warp * SELECTION_CHUNK_ROWS) + lane * 32;
    while (wv) {
      int const b = __ffs(wv) - 1;
      wv &= wv - 1u;
      out[base++] = row0 + b;
    }
  }
}

// BOOL8 flags -> packed mask words. One warp per word: lane l tests row
// w*32+l, ballot packs the word, lane 0 stores it. Grid-stride over the FULL
// padded strip; rows beyond num_rows ballot to 0 (tail-zero invariant).
__global__ void mask_from_bool8_kernel(uint8_t const* __restrict__ flags,
                                       int64_t num_rows,
                                       int64_t num_words,
                                       uint32_t* __restrict__ words)
{
  int const lane               = threadIdx.x & 31;
  int64_t const warps_per_grid = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5;
  int64_t w                    = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  for (; w < num_words; w += warps_per_grid) {
    int64_t const r  = w * 32 + lane;
    bool const p     = (r < num_rows) && (flags[r] != 0);
    uint32_t const b = __ballot_sync(kFullWarp, p);
    if (lane == 0) words[w] = b;
  }
}

}  // namespace

void combine_masks_and(uint32_t* dst_words,
                       uint32_t const* const* src_words,
                       int num_srcs,
                       int64_t num_words,
                       rmm::cuda_stream_view stream)
{
  if (num_srcs < 1 || num_srcs > 8)
    throw std::runtime_error("selection_wave: combine_masks_and needs 1..8 sources");
  if ((num_words & 3) != 0)
    throw std::runtime_error("selection_wave: num_words must be a multiple of 4");
  if (num_words == 0) return;
  mask_src_ptrs srcs{};
  for (int s = 0; s < num_srcs; ++s)
    srcs.p[s] = src_words[s];
  int64_t const num_quads = num_words / 4;
  mask_and_combine_kernel<<<grid_for(num_quads, kBlock), kBlock, 0, stream.value()>>>(
    dst_words, srcs, num_srcs, num_quads);
  throw_on_cuda(cudaPeekAtLastError(), "mask_and_combine launch");
}

int64_t run_selection_cnt(selection_mask& mask,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  if (mask.words == nullptr || mask.chunk_offsets == nullptr || mask.num_rows <= 0)
    throw std::runtime_error("selection_wave: run_selection_cnt on an unbound mask");
  int64_t const nc = selection_mask::ChunksFor(mask.num_rows);

  // Per-chunk survivor counts (transient, stream-ordered, freed on return —
  // safe: the final D2H copy below host-syncs the stream).
  rmm::device_buffer counts_buf(static_cast<std::size_t>(nc) * sizeof(uint32_t), stream, mr);
  auto* counts = static_cast<uint32_t*>(counts_buf.data());

  int const warps_per_block = kBlock / 32;
  chunk_popcount_kernel<<<grid_for(nc, warps_per_block), kBlock, 0, stream.value()>>>(
    mask.words, nc, counts);
  throw_on_cuda(cudaPeekAtLastError(), "chunk_popcount launch");

  // CUB two-call exclusive scan counts -> chunk_offsets[0..nc).
  std::size_t tmp_bytes = 0;
  throw_on_cuda(
    cub::DeviceScan::ExclusiveSum(
      nullptr, tmp_bytes, counts, mask.chunk_offsets, static_cast<int>(nc), stream.value()),
    "chunk_offsets scan probe");
  rmm::device_buffer tmp(tmp_bytes, stream, mr);
  throw_on_cuda(
    cub::DeviceScan::ExclusiveSum(
      tmp.data(), tmp_bytes, counts, mask.chunk_offsets, static_cast<int>(nc), stream.value()),
    "chunk_offsets scan");
  chunk_offsets_tail_kernel<<<1, 1, 0, stream.value()>>>(counts, nc, mask.chunk_offsets);
  throw_on_cuda(cudaPeekAtLastError(), "chunk_offsets tail launch");

  // The one host sync of the selection wave: survivor_count gates wave-2
  // allocations (compacted TierA columns, the TierB gather map).
  uint32_t total = 0;
  throw_on_cuda(
    cudaMemcpyAsync(
      &total, mask.chunk_offsets + nc, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.value()),
    "survivor_count D2H");
  throw_on_cuda(cudaStreamSynchronize(stream.value()), "survivor_count sync");

  mask.survivor_count = static_cast<int64_t>(total);
  return mask.survivor_count;
}

void mask_from_bool8(uint8_t const* flags,
                     int64_t num_rows,
                     uint32_t* mask_words,
                     rmm::cuda_stream_view stream)
{
  if (flags == nullptr || mask_words == nullptr || num_rows <= 0)
    throw std::runtime_error("selection_wave: mask_from_bool8 on unbound buffers");
  int64_t const num_words   = selection_mask::WordsFor(num_rows);
  int const warps_per_block = kBlock / 32;
  mask_from_bool8_kernel<<<grid_for(num_words, warps_per_block), kBlock, 0, stream.value()>>>(
    flags, num_rows, num_words, mask_words);
  throw_on_cuda(cudaPeekAtLastError(), "mask_from_bool8 launch");
}

void mask_to_row_indices(selection_mask const& mask,
                         int32_t* out_indices,
                         rmm::cuda_stream_view stream)
{
  if (mask.survivor_count < 0 || mask.chunk_offsets == nullptr)
    throw std::runtime_error("selection_wave: mask_to_row_indices before CNT ran");
  if (mask.survivor_count == 0) return;
  int64_t const nc          = selection_mask::ChunksFor(mask.num_rows);
  int const warps_per_block = kBlock / 32;
  mask_to_indices_kernel<<<grid_for(nc, warps_per_block), kBlock, 0, stream.value()>>>(
    mask.words, mask.chunk_offsets, nc, out_indices);
  throw_on_cuda(cudaPeekAtLastError(), "mask_to_indices launch");
}

}  // namespace sirius::codegen
