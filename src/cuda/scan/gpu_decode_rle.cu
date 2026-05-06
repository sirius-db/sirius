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
// RLE decode.
//
// Per `decode_rle_data` call: D2H each segment's 8-byte header, sync; D2H
// each segment's counts area, sync; build per-segment inclusive prefix sums
// on host; concatenate, H2D; launch one batched kernel.
//
// Each CTA expands RLE_ROWS_PER_CHUNK output rows of one segment via per-row
// upper_bound on the cumsum. For ec <= RLE_SMEM_MAX_ENTRIES the cumsum is
// staged into shmem; otherwise the binary search reads gmem directly. The
// two paths are duplicated (not unified through a single pointer) so the
// inlined upper_bound emits `ld.shared` / `ld.global` rather than the
// slower generic load.
//
// Malformed segments (bad rle_count_offset, count walk doesn't sum to
// row_count, zero count) are flagged with `entry_count = 0` on the host; the
// kernel zero-fills those chunks deterministically rather than leaving the
// output buffer carrying prior device contents.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_rle.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

constexpr uint32_t BLOCK_DIM             = 256;
constexpr uint32_t RLE_ROWS_PER_CHUNK    = 2048;
constexpr uint32_t VPT                   = RLE_ROWS_PER_CHUNK / BLOCK_DIM;
constexpr uint32_t RLE_SMEM_MAX_ENTRIES  = 4096;  // 16 KiB shmem cap
constexpr uint32_t RLE_SMEM_BYTES        = RLE_SMEM_MAX_ENTRIES * sizeof(uint32_t);
static_assert(BLOCK_DIM * VPT == RLE_ROWS_PER_CHUNK);

/// One descriptor per CTA. Chunks of the same segment share `d_values` and
/// `d_cumsum`. `entry_count == 0` flags a malformed segment; the kernel
/// zero-fills that chunk and skips the cumsum/values pointers.
struct rle_chunk_desc {
  uint8_t const* d_values;
  uint32_t const* d_cumsum;
  uint32_t entry_count;
  uint32_t base_global_row;
  uint32_t local_row_start;
  uint32_t chunk_rows;
};

/// First index where `cumsum[idx] > key`. For row `r`, the result is the RLE
/// entry that produced it.
__device__ __forceinline__ uint32_t rle_upper_bound(uint32_t const* __restrict__ cumsum,
                                                    uint32_t n,
                                                    uint32_t key)
{
  uint32_t lo = 0;
  uint32_t hi = n;
  while (lo < hi) {
    uint32_t mid = lo + ((hi - lo) >> 1);
    if (cumsum[mid] <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename T>
__global__ void kernel_decode_rle(rle_chunk_desc const* __restrict__ descs,
                                  T* __restrict__ d_output,
                                  uint32_t num_chunks)
{
  uint32_t const cid = blockIdx.x;
  if (cid >= num_chunks) return;

  auto const desc = descs[cid];
  uint32_t const rc = desc.chunk_rows;
  T* out_chunk      = d_output + desc.base_global_row + desc.local_row_start;

  // Zero-fill malformed chunk. d_output is uninitialised by the dispatcher,
  // so skipping this would leak prior device contents.
  if (desc.entry_count == 0) {
    for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
      __stwt(out_chunk + i, T(0));
    }
    return;
  }

  uint32_t const ec       = desc.entry_count;
  T const* values         = reinterpret_cast<T const*>(desc.d_values);
  uint32_t const lr_start = desc.local_row_start;

  // The two paths are duplicated (not unified through one pointer) so the
  // inlined upper_bound emits ld.shared / ld.global instead of generic loads.
  // The `entry >= ec` clamp is a defensive backstop for malformed tails —
  // the unrolled loop runs the binary search even for masked-off threads
  // whose `local_row` may exceed cumsum's last entry.
  if (ec <= RLE_SMEM_MAX_ENTRIES) {
    extern __shared__ uint32_t s_cumsum[];
    for (uint32_t i = threadIdx.x; i < ec; i += blockDim.x) {
      s_cumsum[i] = desc.d_cumsum[i];
    }
    __syncthreads();

#pragma unroll
    for (uint32_t v = 0; v < VPT; ++v) {
      uint32_t const i = v * blockDim.x + threadIdx.x;
      if (i >= rc) break;
      uint32_t const local_row = lr_start + i;
      uint32_t entry           = rle_upper_bound(s_cumsum, ec, local_row);
      if (entry >= ec) entry = ec - 1;
      __stwt(out_chunk + i, __ldg(values + entry));
    }
    return;
  }

#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t const i = v * blockDim.x + threadIdx.x;
    if (i >= rc) break;
    uint32_t const local_row = lr_start + i;
    uint32_t entry           = rle_upper_bound(desc.d_cumsum, ec, local_row);
    if (entry >= ec) entry = ec - 1;
    __stwt(out_chunk + i, __ldg(values + entry));
  }
}

/// Per-segment parse result. Empty `cumsum` flags a malformed segment.
struct seg_parsed {
  std::vector<uint32_t> cumsum;
  uint8_t const* d_values;
  uint32_t base_global_row;
  uint32_t row_count;
};

/// Walk counts → inclusive prefix sums. Returns empty on any malformed
/// shape: zero count (DuckDB never emits these), or sum != row_count.
std::vector<uint32_t> walk_counts_to_cumsum(uint16_t const* counts,
                                            size_t counts_capacity_entries,
                                            uint32_t row_count)
{
  std::vector<uint32_t> cumsum;
  cumsum.reserve(std::min<size_t>(counts_capacity_entries, 256));

  uint32_t total = 0;
  for (size_t i = 0; i < counts_capacity_entries; ++i) {
    uint16_t c = counts[i];
    if (c == 0) return {};
    total += c;
    cumsum.push_back(total);
    if (total >= row_count) break;
  }

  if (total != row_count) return {};
  return cumsum;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public entry.
//===----------------------------------------------------------------------===//

void decode_rle_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type /*type*/,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  // Bit-level expand is signedness-independent; dispatch by size alone.
  // INT128 (HUGEINT / DECIMAL128) is refused upstream; throw as a backstop.
  if (type_size != 1 && type_size != 2 && type_size != 4 && type_size != 8) {
    throw std::runtime_error(
      "gpu_decode_table: viability invariant violated — RLE type_size " +
      std::to_string(type_size));
  }

  std::vector<gpu_segment_desc const*> live;
  live.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    live.push_back(&seg);
  }
  if (live.empty()) return;

  size_t const n = live.size();

  // Phase 1: D2H each segment's 8-byte rle_count_offset header.
  std::vector<uint64_t> h_offsets(n, 0);
  for (size_t i = 0; i < n; ++i) {
    auto const& seg = *live[i];
    if (seg.bytes_size < RLE_HEADER_SIZE) continue;
    RMM_CUDA_TRY(cudaMemcpyAsync(&h_offsets[i],
                                 seg.d_bytes,
                                 sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  // Phase 2: D2H each segment's counts area. Skip segments whose offset
  // doesn't point strictly between the header and the segment end.
  std::vector<std::vector<uint8_t>> h_count_buffers(n);
  for (size_t i = 0; i < n; ++i) {
    auto const& seg = *live[i];
    if (seg.bytes_size < RLE_HEADER_SIZE) continue;
    uint64_t off = h_offsets[i];
    if (off < RLE_HEADER_SIZE || off >= seg.bytes_size) continue;
    size_t counts_bytes = size_t{seg.bytes_size} - off;
    h_count_buffers[i].resize(counts_bytes);
    RMM_CUDA_TRY(cudaMemcpyAsync(h_count_buffers[i].data(),
                                 seg.d_bytes + off,
                                 counts_bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  // Build per-segment cumsums; concatenate into one column-wide buffer.
  std::vector<seg_parsed> parsed(n);
  std::vector<uint32_t> h_concat_cumsum;
  std::vector<size_t> seg_cumsum_offset(n, 0);

  for (size_t i = 0; i < n; ++i) {
    auto const& seg          = *live[i];
    parsed[i].base_global_row = seg.row_offset;
    parsed[i].row_count       = seg.row_count;
    parsed[i].d_values        = seg.d_bytes + RLE_HEADER_SIZE;

    if (h_count_buffers[i].empty()) {
      seg_cumsum_offset[i] = h_concat_cumsum.size();
      continue;
    }

    auto const* counts =
      reinterpret_cast<uint16_t const*>(h_count_buffers[i].data());
    size_t capacity_entries = h_count_buffers[i].size() / sizeof(uint16_t);

    parsed[i].cumsum =
      walk_counts_to_cumsum(counts, capacity_entries, seg.row_count);

    // Refuse if the values area is too small to hold the entries the count
    // walk produced — values_bytes derives from rle_count_offset, while
    // cumsum.size() comes from walking counts; an inconsistent segment
    // would let the kernel read past the values area.
    if (!parsed[i].cumsum.empty()) {
      uint64_t off          = h_offsets[i];
      size_t values_bytes   = static_cast<size_t>(off) - RLE_HEADER_SIZE;
      size_t needed_values  = parsed[i].cumsum.size() * size_t{type_size};
      if (values_bytes < needed_values) parsed[i].cumsum.clear();
    }

    seg_cumsum_offset[i] = h_concat_cumsum.size();
    h_concat_cumsum.insert(h_concat_cumsum.end(),
                           parsed[i].cumsum.begin(),
                           parsed[i].cumsum.end());
  }

  // H2D cumsum before building descriptors so each descriptor can carry
  // its real device pointer.
  rmm::device_uvector<uint32_t> d_concat_cumsum(h_concat_cumsum.size(), stream, mr);
  if (!h_concat_cumsum.empty()) {
    RMM_CUDA_TRY(cudaMemcpyAsync(d_concat_cumsum.data(),
                                 h_concat_cumsum.data(),
                                 h_concat_cumsum.size() * sizeof(uint32_t),
                                 cudaMemcpyHostToDevice,
                                 stream.value()));
  }

  std::vector<rle_chunk_desc> h_descs;
  h_descs.reserve(n * 2);

  for (size_t i = 0; i < n; ++i) {
    uint32_t const rc       = parsed[i].row_count;
    uint32_t const num_cnks = (rc + RLE_ROWS_PER_CHUNK - 1) / RLE_ROWS_PER_CHUNK;
    bool const malformed    = parsed[i].cumsum.empty();
    uint32_t const* d_cs    = malformed ? nullptr
                                        : d_concat_cumsum.data() + seg_cumsum_offset[i];

    for (uint32_t c = 0; c < num_cnks; ++c) {
      uint32_t local_start = c * RLE_ROWS_PER_CHUNK;
      uint32_t this_rows   = (c + 1u < num_cnks) ? RLE_ROWS_PER_CHUNK
                                                 : rc - local_start;

      rle_chunk_desc d;
      d.d_values        = malformed ? nullptr : parsed[i].d_values;
      d.d_cumsum        = d_cs;
      d.entry_count     = malformed ? 0u : static_cast<uint32_t>(parsed[i].cumsum.size());
      d.base_global_row = parsed[i].base_global_row;
      d.local_row_start = local_start;
      d.chunk_rows      = this_rows;
      h_descs.push_back(d);
    }
  }

  if (h_descs.empty()) return;

  rmm::device_uvector<rle_chunk_desc> d_descs(h_descs.size(), stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs.data(),
                               h_descs.size() * sizeof(rle_chunk_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  uint32_t const grid = static_cast<uint32_t>(h_descs.size());
  switch (type_size) {
    case 1:
      kernel_decode_rle<uint8_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), d_output, grid);
      break;
    case 2:
      kernel_decode_rle<uint16_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), reinterpret_cast<uint16_t*>(d_output), grid);
      break;
    case 4:
      kernel_decode_rle<uint32_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), reinterpret_cast<uint32_t*>(d_output), grid);
      break;
    case 8:
      kernel_decode_rle<uint64_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), reinterpret_cast<uint64_t*>(d_output), grid);
      break;
    default:
      break;
  }
}

}  // namespace sirius::cuda::scan
