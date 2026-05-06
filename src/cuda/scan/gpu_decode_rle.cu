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
// Two device kernels per `decode_rle_data` call:
//   1. kernel_build_cumsum: one CTA per segment. Reads the segment header,
//      cooperatively scans counts via cub::BlockScan into a device-resident
//      cumsum buffer. Writes per-segment entry_count (0 = malformed). Host
//      pays no D2H sync.
//   2. kernel_decode_rle:   one CTA per RLE_ROWS_PER_CHUNK-row slice. Reads
//      entry_count from the per-segment array; loads cumsum into shmem and
//      expands rows via per-row upper_bound + value gather.
//
// Segments with > RLE_BUILD_MAX_ENTRIES entries are flagged malformed by
// the build kernel and zero-filled by the expand kernel. Realistic DuckDB
// RLE columns stay well below this cap; high-entry-count segments need the
// hierarchical-cumsum follow-up.
//
// Malformed conditions detected by the build kernel: rle_count_offset out
// of range, count == 0, walk doesn't sum to row_count, walk exceeds the
// build cap, values area can't hold entry_count values. All collapse to
// entry_count = 0.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_rle.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_scan.cuh>
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
constexpr uint32_t RLE_SMEM_MAX_ENTRIES  = 4096;
constexpr uint32_t RLE_SMEM_BYTES        = RLE_SMEM_MAX_ENTRIES * sizeof(uint32_t);
static_assert(BLOCK_DIM * VPT == RLE_ROWS_PER_CHUNK);

// Build-kernel cap: how many counts a single CTA can scan in one BlockScan.
// 4096 entries × 4 B cumsum slot = 16 KiB device-buffer per segment, plus
// 16 KiB BlockScan working set. Bumping this requires either tile-iterating
// the BlockScan or going to a hierarchical scan.
constexpr uint32_t RLE_BUILD_MAX_ENTRIES = 4096;
constexpr uint32_t VPT_BUILD             = RLE_BUILD_MAX_ENTRIES / BLOCK_DIM;
static_assert(BLOCK_DIM * VPT_BUILD == RLE_BUILD_MAX_ENTRIES);

constexpr uint32_t MALFORMED_FLAG = 0u;

/// Per-segment input to the build kernel.
struct rle_build_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t row_count;
};

/// Per-CTA input to the expand kernel. d_cumsum points into the device-
/// resident concatenated cumsum buffer (one MAX_ENTRIES-slot slice per
/// segment). entry_count is read indirectly via seg_id from a per-segment
/// array (build kernel writes it; host can't observe it without a sync).
struct rle_chunk_desc {
  uint8_t const* d_values;
  uint32_t const* d_cumsum;
  uint32_t base_global_row;
  uint32_t local_row_start;
  uint32_t chunk_rows;
  uint32_t seg_id;
};

//===----------------------------------------------------------------------===//
// Build kernel: device-side cumsum.
//===----------------------------------------------------------------------===//

__global__ void kernel_build_cumsum(rle_build_desc const* __restrict__ descs,
                                    uint32_t* __restrict__ d_cumsums,
                                    uint32_t cumsum_stride_entries,
                                    uint32_t* __restrict__ d_entry_counts,
                                    uint32_t type_size,
                                    uint32_t num_segs)
{
  using BlockScan = cub::BlockScan<uint32_t, BLOCK_DIM>;
  __shared__ typename BlockScan::TempStorage scan_temp;
  __shared__ uint32_t s_offset;
  __shared__ uint32_t s_capacity;
  __shared__ uint32_t s_malformed;
  __shared__ uint32_t s_first_match;

  uint32_t const sid = blockIdx.x;
  if (sid >= num_segs) return;
  auto const desc = descs[sid];

  if (threadIdx.x == 0) {
    s_malformed = 0;
    s_first_match = ~0u;
    if (desc.bytes_size < RLE_HEADER_SIZE) {
      s_malformed = 1;
    } else {
      uint64_t off64 = 0;
      memcpy(&off64, desc.d_bytes, sizeof(uint64_t));
      if (off64 < RLE_HEADER_SIZE || off64 >= desc.bytes_size) {
        s_malformed = 1;
      } else {
        s_offset   = static_cast<uint32_t>(off64);
        s_capacity = (desc.bytes_size - s_offset) / sizeof(uint16_t);
      }
    }
  }
  __syncthreads();

  if (s_malformed) {
    if (threadIdx.x == 0) d_entry_counts[sid] = MALFORMED_FLAG;
    return;
  }

  uint32_t const row_count   = desc.row_count;
  uint16_t const* counts     = reinterpret_cast<uint16_t const*>(desc.d_bytes + s_offset);
  uint32_t const capacity    = s_capacity < RLE_BUILD_MAX_ENTRIES
                                 ? s_capacity : RLE_BUILD_MAX_ENTRIES;
  uint32_t* cumsum_out       = d_cumsums + size_t{sid} * cumsum_stride_entries;

  // Each thread loads VPT_BUILD counts (blocked layout).
  uint32_t my_counts[VPT_BUILD];
  uint32_t local_zero_seen = 0;
#pragma unroll
  for (uint32_t v = 0; v < VPT_BUILD; ++v) {
    uint32_t const i = threadIdx.x * VPT_BUILD + v;
    if (i < capacity) {
      uint16_t c   = counts[i];
      my_counts[v] = c;
      if (c == 0) local_zero_seen = 1;
    } else {
      my_counts[v] = 0;
    }
  }

  // Per-thread inclusive prefix.
  uint32_t my_aggregate = 0;
  uint32_t my_local_cumsum[VPT_BUILD];
#pragma unroll
  for (uint32_t v = 0; v < VPT_BUILD; ++v) {
    my_aggregate += my_counts[v];
    my_local_cumsum[v] = my_aggregate;
  }

  // Block scan over per-thread aggregates → my exclusive prefix.
  uint32_t my_prefix = 0;
  BlockScan(scan_temp).ExclusiveSum(my_aggregate, my_prefix);

  // Detect a zero count anywhere in the block.
  if (local_zero_seen) atomicOr(&s_malformed, 1u);

  // Write cumsum + find the entry that lands exactly on row_count.
#pragma unroll
  for (uint32_t v = 0; v < VPT_BUILD; ++v) {
    uint32_t const i  = threadIdx.x * VPT_BUILD + v;
    uint32_t const cs = my_prefix + my_local_cumsum[v];
    if (i < capacity) cumsum_out[i] = cs;
    if (i < capacity && cs == row_count) atomicMin(&s_first_match, i + 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t ec = s_first_match;
    if (s_malformed || ec == ~0u) {
      d_entry_counts[sid] = MALFORMED_FLAG;
      return;
    }
    // Values area must hold ec values (rle_count_offset - header == values bytes).
    uint32_t values_bytes = s_offset - RLE_HEADER_SIZE;
    if (values_bytes < ec * type_size) {
      d_entry_counts[sid] = MALFORMED_FLAG;
      return;
    }
    d_entry_counts[sid] = ec;
  }
}

//===----------------------------------------------------------------------===//
// Expand kernel.
//===----------------------------------------------------------------------===//

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
                                  uint32_t const* __restrict__ d_entry_counts,
                                  T* __restrict__ d_output,
                                  uint32_t num_chunks)
{
  uint32_t const cid = blockIdx.x;
  if (cid >= num_chunks) return;

  auto const desc = descs[cid];
  uint32_t const rc = desc.chunk_rows;
  T* out_chunk      = d_output + desc.base_global_row + desc.local_row_start;

  uint32_t const ec = __ldg(d_entry_counts + desc.seg_id);

  if (ec == MALFORMED_FLAG) {
    for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
      __stwt(out_chunk + i, T(0));
    }
    return;
  }

  T const* values         = reinterpret_cast<T const*>(desc.d_values);
  uint32_t const lr_start = desc.local_row_start;

  extern __shared__ uint32_t s_cumsum[];
  for (uint32_t i = threadIdx.x; i < ec; i += blockDim.x) {
    s_cumsum[i] = desc.d_cumsum[i];
  }
  __syncthreads();

  // Long-run fast path: when the segment's average run length is at least a
  // warp-width, lane 0 searches once per warp and broadcasts. Skipping this
  // for short-run shapes — the bound check fails almost every iteration
  // and the lane-0 search becomes pure overhead.
  bool const long_runs_heuristic =
    rc / 32u >= ec || (rc >= ec && (rc / ec) >= 32u);
  uint32_t const lane = threadIdx.x & 31u;

  if (long_runs_heuristic) {
#pragma unroll
    for (uint32_t v = 0; v < VPT; ++v) {
      uint32_t const i = v * blockDim.x + threadIdx.x;
      if (i >= rc) break;
      uint32_t const warp_first = lr_start + (i & ~31u);

      uint32_t entry_warp = 0;
      if (lane == 0) {
        entry_warp = rle_upper_bound(s_cumsum, ec, warp_first);
        if (entry_warp >= ec) entry_warp = ec - 1;
      }
      entry_warp = __shfl_sync(0xFFFFFFFFu, entry_warp, 0);

      uint32_t cumsum_at_entry = 0;
      if (lane == 0) cumsum_at_entry = s_cumsum[entry_warp];
      cumsum_at_entry = __shfl_sync(0xFFFFFFFFu, cumsum_at_entry, 0);

      uint32_t const warp_last = warp_first + 31u;

      if (warp_last < cumsum_at_entry) {
        __stwt(out_chunk + i, __ldg(values + entry_warp));
      } else {
        uint32_t const local_row = lr_start + i;
        uint32_t entry = rle_upper_bound(s_cumsum, ec, local_row);
        if (entry >= ec) entry = ec - 1;
        __stwt(out_chunk + i, __ldg(values + entry));
      }
    }
    return;
  }

#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t const i = v * blockDim.x + threadIdx.x;
    if (i >= rc) break;
    uint32_t const local_row = lr_start + i;
    uint32_t entry           = rle_upper_bound(s_cumsum, ec, local_row);
    if (entry >= ec) entry = ec - 1;
    __stwt(out_chunk + i, __ldg(values + entry));
  }
  (void)lane;
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

  // Build per-segment build descriptors.
  std::vector<rle_build_desc> h_build(n);
  for (size_t i = 0; i < n; ++i) {
    h_build[i] = {live[i]->d_bytes, live[i]->bytes_size, live[i]->row_count};
  }

  // Allocate device-side cumsum buffer (one MAX_ENTRIES slot per segment)
  // and entry-count array.
  rmm::device_uvector<uint32_t> d_cumsums(n * RLE_BUILD_MAX_ENTRIES, stream, mr);
  rmm::device_uvector<uint32_t> d_entry_counts(n, stream, mr);
  rmm::device_uvector<rle_build_desc> d_build_descs(n, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_build_descs.data(),
                               h_build.data(),
                               n * sizeof(rle_build_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  kernel_build_cumsum<<<static_cast<uint32_t>(n), BLOCK_DIM, 0, stream.value()>>>(
    d_build_descs.data(),
    d_cumsums.data(),
    RLE_BUILD_MAX_ENTRIES,
    d_entry_counts.data(),
    type_size,
    static_cast<uint32_t>(n));

  // Build per-CTA chunk descriptors. Cumsum slice + values pointer can be
  // computed on host without observing the build kernel's output.
  std::vector<rle_chunk_desc> h_descs;
  h_descs.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    auto const& seg          = *live[i];
    uint32_t const rc        = seg.row_count;
    uint32_t const num_cnks  = (rc + RLE_ROWS_PER_CHUNK - 1) / RLE_ROWS_PER_CHUNK;
    uint8_t const* d_values  = seg.d_bytes + RLE_HEADER_SIZE;
    uint32_t const* d_cumsum = d_cumsums.data() + i * RLE_BUILD_MAX_ENTRIES;

    for (uint32_t c = 0; c < num_cnks; ++c) {
      uint32_t const local_start = c * RLE_ROWS_PER_CHUNK;
      uint32_t const this_rows   = (c + 1u < num_cnks) ? RLE_ROWS_PER_CHUNK
                                                       : rc - local_start;
      h_descs.push_back({d_values,
                         d_cumsum,
                         seg.row_offset,
                         local_start,
                         this_rows,
                         static_cast<uint32_t>(i)});
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
          d_descs.data(), d_entry_counts.data(), d_output, grid);
      break;
    case 2:
      kernel_decode_rle<uint16_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), d_entry_counts.data(),
          reinterpret_cast<uint16_t*>(d_output), grid);
      break;
    case 4:
      kernel_decode_rle<uint32_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), d_entry_counts.data(),
          reinterpret_cast<uint32_t*>(d_output), grid);
      break;
    case 8:
      kernel_decode_rle<uint64_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), d_entry_counts.data(),
          reinterpret_cast<uint64_t*>(d_output), grid);
      break;
    default:
      break;
  }
}

}  // namespace sirius::cuda::scan
