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
// FSST string codec. On-disk segment layout (DuckDB):
//
//   offset 0                                                       seg_end
//   +--------+--------------------+---------------+------------------------+
//   | header | compressed_lengths |  symbol table |  FSST-compressed bytes |
//   |  16B   |  bitpacked per row |  opaque blob  |  rows packed in        |
//   |        |  -> comp_len       |  imported on  |  REVERSE order ending  |
//   |        |                    |  device       |  at dict_end           |
//   +--------+--------------------+---------------+------------------------+
//   0        16                   ^                                        ^
//                                 hdr.fsst_symbol_table_offset             hdr.dict_end
//
// Decode pipeline (in launch order):
//   kernel_build_fsst_decoders               — parse opaque symtab into
//                                              fsst_decoder_compact on device
//   kernel_compute_compressed_offsets_fsst   — Pass-1 A+B: per-segment
//                                              prefix sum of compressed
//                                              lengths -> d_comp_offsets
//   kernel_compute_decompressed_lengths_fsst — Pass-1 C: per-row byte-walk
//                                              to compute decoded length
//   kernel_gather_fsst_chunked               — Pass-2: per-row decode +
//                                              emit to d_chars
//
// Pass-1 A+B owns per-segment prefix-sum state (one CTA per segment);
// Pass-1 C and Pass-2 partition each segment across CTAs into chunks.
// The on-device decode core (detail/fsst.cuh) is shared with DICT_FSST.
//===----------------------------------------------------------------------===//

#include "cuda/scan/detail/fsst.cuh"
#include "cuda/scan/strings/fsst.cuh"
#include "cuda/scan/unpack_value.cuh"

#include <cub/cub.cuh>
#include <cuda/cmath>
#include <cuda/std/algorithm>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace sirius::cuda::scan {

namespace {

using detail::device_fsst_import;
using detail::FSST_SCRATCH_U32_PER_WARP;
using detail::FSST_WARPS_PER_CTA;
using detail::warp_compute_decomp_len;
using detail::warp_decode_fsst;

//! On-disk FSST header.
struct fsst_header_t {
  uint32_t dict_size;
  uint32_t dict_end;
  uint32_t bitpacking_width;
  uint32_t fsst_symbol_table_offset;
};

/**
 * @brief Copy FSST header @p hdr into @p base, bounded by the buffer size @p limit.
 * @return true if the header was successfully copied (i.e. the header fits within the buffer), and
 * if the header metadata is valid; false otherwise.
 */
__device__ __forceinline__ bool parse_fsst_header(uint8_t const* base,
                                                  uint32_t limit,
                                                  fsst_header_t* hdr)
{
  if (limit < sizeof(fsst_header_t)) return false;
  memcpy(hdr, base, sizeof(fsst_header_t));
  return hdr->dict_end <= limit && hdr->fsst_symbol_table_offset < hdr->dict_end &&
         hdr->bitpacking_width <= MAX_BITPACKING_WIDTH;
}

/**
 * @brief Build per-segment FSST decoders from the on-disk symbol table blob.
 *
 * Coalesced load of the symbol table blob for the segment into shared memory, then single-threaded
 * parse into the fsst_decoder_compact format in global memory. One CTA per segment.
 */
__global__ __launch_bounds__(BLOCK_DIM) void kernel_build_fsst_decoders(
  string_chunk_desc const* __restrict__ descs,
  int num_segments,
  fsst_decoder_compact* __restrict__ d_decoders)
{
  auto const seg_idx = blockIdx.x;
  auto const desc    = descs[seg_idx];

  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;
  __shared__ uint8_t sm_symtab[FSST_SYMTAB_MAX_BYTES];

  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(desc.d_bytes, desc.bytes_size, &sm_hdr); }
  __syncthreads();

  if (!sm_ok) {
    for (int i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
      d_decoders[seg_idx].len[i]    = 0;
      d_decoders[seg_idx].symbol[i] = 0;
    }
    return;
  }

  auto const symtab_off = sm_hdr.fsst_symbol_table_offset;
  auto const symtab_size =
    ::cuda::std::min(desc.bytes_size - symtab_off, uint32_t{FSST_SYMTAB_MAX_BYTES});
  auto const* sym_src = desc.d_bytes + symtab_off;
  for (int i = threadIdx.x; i < symtab_size; i += blockDim.x) {
    sm_symtab[i] = sym_src[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) { device_fsst_import(sm_symtab, &d_decoders[seg_idx]); }
}

/**
 * @brief Compute the per-row offsets into the FSST compressed byte stream for an FSST segment.
 *
 * One CTA per segment; in-CTA cub::BlockScan over per-thread chunk sums produces the inclusive
 * prefix sum the gather kernel reads.
 */
__global__ void kernel_compute_compressed_offsets_fsst(
  uint32_t* __restrict__ d_comp_offsets,
  string_chunk_desc const* __restrict__ descs,
  uint32_t const* __restrict__ d_fsst_row_starts,
  int num_segments)
{
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_DIM>;

  __shared__ typename BlockScanT::TempStorage scan_temp;
  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;

  auto const seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) return;
  auto const desc  = descs[seg_idx];
  auto const* base = desc.d_bytes;

  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(base, desc.bytes_size, &sm_hdr); }
  __syncthreads();

  auto const segment_base      = d_fsst_row_starts[seg_idx];
  auto const segment_row_count = desc.row_count;
  auto* segment_comp_offsets   = d_comp_offsets + segment_base;

  if (!sm_ok) {
    // Zero the offsets for the segment so phase-C emits empty strings.
    for (int i = threadIdx.x; i < segment_row_count; i += blockDim.x)
      segment_comp_offsets[i] = 0;
    return;
  }

  // Phase A: unpack compressed lengths from the bitpacked length stream after the header.
  auto const* packed = reinterpret_cast<uint32_t const*>(base + sizeof(fsst_header_t));
  for (int i = threadIdx.x; i < segment_row_count; i += blockDim.x) {
    segment_comp_offsets[i] = unpack_value<uint32_t>(packed, i, sm_hdr.bitpacking_width);
  }
  __syncthreads();

  // Phase B: per-thread sequential scan + BlockScan over per-thread totals.
  auto const max_rows_per_thread = ::cuda::ceil_div(segment_row_count, blockDim.x);
  auto const start               = threadIdx.x * max_rows_per_thread;
  auto const end                 = ::cuda::std::min(start + max_rows_per_thread, segment_row_count);
  uint32_t thread_sum            = 0;
  for (int i = start; i < end; ++i) {
    /// WARNING: uncoalesced GMEM access pattern
    thread_sum += segment_comp_offsets[i];
    segment_comp_offsets[i] = thread_sum;
  }
  uint32_t exclusive_sum = 0;
  BlockScanT(scan_temp).ExclusiveSum(thread_sum, exclusive_sum);
  if (exclusive_sum > 0) {
    for (int i = start; i < end; ++i) {
      /// WARNING: uncoalesced GMEM access pattern
      segment_comp_offsets[i] += exclusive_sum;
    }
  }
}

/**
 * @brief Compute the decompressed lengths for each row in the FSST compressed stream.
 *
 * Adaptive per CTA: thread-per-row for short rows (lower warp-coord tax dominates), warp-per-row
 * for longer rows (coalesced LDG dominates). 2 × WARP_THREADS = 64 B / row crossover matches the
 * empirical sweet spot.
 */
__global__ __launch_bounds__(BLOCK_DIM) void kernel_compute_decompressed_lengths_fsst(
  fsst_chunk_desc const* __restrict__ descs,
  uint32_t* __restrict__ d_lengths,
  uint32_t const* __restrict__ d_comp_offsets,
  fsst_decoder_compact const* __restrict__ d_decoders,
  int num_chunks)
{
  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc  = descs[chunk_id];
  auto const* base = desc.d_bytes;

  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;
  __shared__ uint8_t sm_len[FSST_NUM_SYMBOLS];
  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(base, desc.bytes_size, &sm_hdr); }
  __syncthreads();

  // Zero-fill lengths for the chunk if the metadata is malformed.
  if (!sm_ok) {
    for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      d_lengths[desc.global_row_start + i] = 0;
    }
    return;
  }

  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (int i = threadIdx.x; i < FSST_NUM_SYMBOLS; i += blockDim.x) {
    sm_len[i] = dec.len[i];
  }
  __syncthreads();

  auto const* dict_end_ptr          = base + sm_hdr.dict_end;
  auto const* compressed_cumsum_ptr = d_comp_offsets + desc.fsst_row_start;

  // Compute the chunk's average compressed-byte/row to dispatch between
  // thread-per-row and warp-per-row strategies.
  auto const start           = desc.is_first_chunk ? 0 : *(compressed_cumsum_ptr - 1);
  auto const end             = compressed_cumsum_ptr[desc.row_count - 1];
  auto const avg_comp_length = (end - start) / desc.row_count;

  if (avg_comp_length >= 2 * WARP_THREADS) {
    // Warp-per-row: 32 lanes coalesce LDG over a row's compressed bytes.
    auto const lane          = threadIdx.x % WARP_THREADS;
    auto const warp_id       = threadIdx.x / WARP_THREADS;
    auto const warps_per_cta = blockDim.x / WARP_THREADS;
    for (int i = warp_id; i < desc.row_count; i += warps_per_cta) {
      auto const cumsum      = compressed_cumsum_ptr[i];
      auto const prev_cumsum = (i > 0) ? compressed_cumsum_ptr[i - 1] : start;
      if (cumsum > sm_hdr.dict_end || prev_cumsum > cumsum) {
        if (lane == 0) { d_lengths[desc.global_row_start + i] = 0u; }
        continue;
      }
      auto const comp_len = cumsum - prev_cumsum;
      if (comp_len == 0) {
        if (lane == 0) { d_lengths[desc.global_row_start + i] = 0; }
        continue;
      }
      auto const* comp_length_ptr = dict_end_ptr - cumsum;
      int decomp_len = warp_compute_decomp_len(comp_length_ptr, comp_len, sm_len, lane);
      if (lane == 0) { d_lengths[desc.global_row_start + i] = decomp_len; }
    }
  } else {
    // Thread-per-row: short rows; warp-coord tax of cooperative scan dominates.
    for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      auto const cumsum      = compressed_cumsum_ptr[i];
      auto const prev_cumsum = (i > 0) ? compressed_cumsum_ptr[i - 1] : start;
      if (cumsum > sm_hdr.dict_end || prev_cumsum > cumsum) {
        d_lengths[desc.global_row_start + i] = 0u;
        continue;
      }
      auto const comp_len = cumsum - prev_cumsum;
      if (comp_len == 0) {
        d_lengths[desc.global_row_start + i] = 0u;
        continue;
      }
      auto const* comp_ptr = dict_end_ptr - cumsum;
      int decomp_len       = 0;
      int pos              = 0;
      while (pos < comp_len) {
        auto const code = comp_ptr[pos++];
        if (code < FSST_ESC) {
          decomp_len += sm_len[code];
        } else if (pos < comp_len) {
          // Trailing escape (corrupt input) is dropped — stay in sync with gather.
          ++pos;
          ++decomp_len;
        }
      }
      d_lengths[desc.global_row_start + i] = decomp_len;
    }
  }
}

/**
 * @brief Gather kernel for FSST-compressed segments, with the same chunking as phase-C. Each warp
 * walks the compressed byte stream for its row, decodes on the fly into a per-warp scratch buffer,
 * and flushes to global when the next chunk's worst-case emit would overflow scratch.
 */
__global__ __launch_bounds__(BLOCK_DIM) void kernel_gather_fsst_chunked(
  fsst_chunk_desc const* __restrict__ descs,
  int32_t const* __restrict__ d_offsets,
  uint8_t* __restrict__ d_chars,
  uint32_t const* __restrict__ d_comp_offsets,
  fsst_decoder_compact const* __restrict__ d_decoders,
  int num_chunks)
{
  __shared__ bool sm_ok;
  __shared__ fsst_header_t sm_hdr;
  __shared__ uint8_t sm_len[FSST_SIZE];
  __shared__ uint32_t sm_sym_lo[FSST_SIZE];  ///< The lower 4B of the symbol
  __shared__ uint32_t sm_sym_hi[FSST_SIZE];  ///< The upper 4B of the symbol (only used if len > 4)
  __shared__ uint32_t
    sm_scratch_u32[FSST_WARPS_PER_CTA]
                  [FSST_SCRATCH_U32_PER_WARP];  ///< Per-warp scratch for decoding symbols before
                                                ///< flushing to global

  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc          = descs[chunk_id];
  auto const* segment_base = desc.d_bytes;

  // Parse the segment header.
  if (threadIdx.x == 0) { sm_ok = parse_fsst_header(segment_base, desc.bytes_size, &sm_hdr); }
  __syncthreads();
  if (!sm_ok) return;  // pass-1 emitted zero lengths → nothing to gather

  // Load the symbol table into SMEM.
  fsst_decoder_compact const& dec = d_decoders[desc.seg_decoder_idx];
  for (int i = threadIdx.x; i < FSST_SIZE; i += blockDim.x) {
    sm_len[i]          = (i < FSST_NUM_SYMBOLS) ? dec.len[i] : 0;
    uint64_t const sym = (i < FSST_NUM_SYMBOLS) ? dec.symbol[i] : 0;
    sm_sym_lo[i]       = static_cast<uint32_t>(sym);
    sm_sym_hi[i]       = static_cast<uint32_t>(sym >> 32);
  }
  __syncthreads();

  // Warp-per-row gather + decode.
  auto const* dict_end_ptr  = segment_base + sm_hdr.dict_end;
  auto const* my_cumsum_ptr = d_comp_offsets + desc.fsst_row_start;
  auto const lane           = threadIdx.x % WARP_THREADS;
  auto const warp_id        = threadIdx.x / WARP_THREADS;
  auto const warps_per_cta  = blockDim.x / WARP_THREADS;

  for (int i = warp_id; i < desc.row_count; i += warps_per_cta) {
    auto const my_cumsum = my_cumsum_ptr[i];
    auto const prev_cumsum =
      (i > 0) ? my_cumsum_ptr[i - 1] : (desc.is_first_chunk ? 0 : *(my_cumsum_ptr - 1));
    auto const comp_len = my_cumsum - prev_cumsum;
    if (comp_len == 0) continue;

    auto const* comp_ptr       = dict_end_ptr - my_cumsum;
    auto const out_base_offset = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);
    warp_decode_fsst(comp_ptr,
                     comp_len,
                     d_chars + out_base_offset,
                     sm_len,
                     sm_sym_lo,
                     sm_sym_hi,
                     &sm_scratch_u32[warp_id][0],
                     lane);
  }
}

}  // namespace

prepared_fsst prepare_fsst(gpu_string_codec_run const& run)
{
  prepared_fsst out;
  out.total_fsst_row_count = 0;
  out.length_descs.reserve(run.segments.size());
  out.row_starts.reserve(run.segments.size());

  // Segment descriptors for pass-1 A+B.
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    out.row_starts.push_back(out.total_fsst_row_count);
    out.total_fsst_row_count += seg.row_count;
    out.length_descs.push_back(
      {seg.d_bytes, seg.bytes_size, seg.row_count, seg.row_offset, seg.seg_row_start});
  }
  auto const segment_count = out.length_descs.size();
  // decoders are populated on-device by kernel_build_fsst_decoders; here we
  // just allocate the slots so the upload + kernel see the right size.
  out.decoders.resize(segment_count);

  // Chunked descriptors for phase-C + gather. Split per-segment only when
  // total segments < target_ctas (else one-chunk-per-segment fills SMs already).
  auto const target_ctas         = get_target_ctas();
  uint32_t target_rows_per_chunk = 0;
  if (segment_count < target_ctas && out.total_fsst_row_count > 0) {
    target_rows_per_chunk = std::max(out.total_fsst_row_count / target_ctas, MIN_ROWS_PER_CHUNK);
    target_rows_per_chunk = (target_rows_per_chunk / 32) * 32;  // multiple of 32 for coalescing
    if (target_rows_per_chunk == 0) target_rows_per_chunk = 32;
  }
  for (uint32_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
    auto const& seg          = out.length_descs[segment_idx];
    auto const fsst_base_row = out.row_starts[segment_idx];
    if (target_rows_per_chunk == 0) {
      // No chunking: one CTA per segment.
      out.gather_chunks.push_back({seg.d_bytes,
                                   seg.bytes_size,
                                   seg.row_count,
                                   seg.global_row_start,
                                   fsst_base_row,
                                   segment_idx,
                                   1,
                                   {0, 0, 0}});
    } else {
      // Chunking: one CTA per chunk (slice of a segment).
      auto remaining                    = seg.row_count;
      uint32_t offset                   = 0;
      uint8_t is_first_chunk_in_segment = 1;
      while (remaining > 0) {
        auto const chunk_row_count = std::min(remaining, target_rows_per_chunk);
        out.gather_chunks.push_back({seg.d_bytes,
                                     seg.bytes_size,
                                     chunk_row_count,
                                     seg.global_row_start + offset,
                                     fsst_base_row + offset,
                                     segment_idx,
                                     is_first_chunk_in_segment,
                                     {0, 0, 0}});
        offset += chunk_row_count;
        remaining -= chunk_row_count;
        is_first_chunk_in_segment = 0;
      }
    }
  }
  return out;
}

void launch_fsst_lengths(fsst_decoder_compact* d_decoders,
                         uint32_t* d_comp_offsets,
                         uint32_t* d_lengths,
                         string_chunk_desc const* d_length_descs,
                         uint32_t const* d_row_starts,
                         fsst_chunk_desc const* d_gather_chunks,
                         uint32_t n_segments,
                         uint32_t n_chunks,
                         rmm::cuda_stream_view stream)
{
  if (n_segments == 0) return;
  // On-device symbol-table parse: one CTA per segment, coalesced symtab load into shmem then
  // serial host-style parse. Replaces the per-segment sync D2H + duckdb_fsst_import that used to
  // happen in prepare_fsst.
  kernel_build_fsst_decoders<<<n_segments, BLOCK_DIM, 0, stream.value()>>>(
    d_length_descs, n_segments, d_decoders);
  // A+B per-segment (prefix-sum state lives in one CTA); C per-chunk.
  kernel_compute_compressed_offsets_fsst<<<n_segments, BLOCK_DIM, 0, stream.value()>>>(
    d_comp_offsets, d_length_descs, d_row_starts, n_segments);
  kernel_compute_decompressed_lengths_fsst<<<n_chunks, BLOCK_DIM, 0, stream.value()>>>(
    d_gather_chunks, d_lengths, d_comp_offsets, d_decoders, n_chunks);
}

void launch_fsst_gather(fsst_chunk_desc const* d_gather_chunks,
                        int32_t const* d_offsets,
                        uint8_t* d_chars,
                        uint32_t const* d_comp_offsets,
                        fsst_decoder_compact const* d_decoders,
                        uint32_t n_chunks,
                        rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  kernel_gather_fsst_chunked<<<n_chunks, BLOCK_DIM, 0, stream.value()>>>(
    d_gather_chunks, d_offsets, d_chars, d_comp_offsets, d_decoders, n_chunks);
}

}  // namespace sirius::cuda::scan
