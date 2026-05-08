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
// RLE decode. On-disk segment layout (DuckDB):
//
//   byte 0    8                                rle_count_offset    seg_end
//   │         │                                │                   │
//   ├─uint64──┼─── values: T[entry_count] ────┼─ 0-padded ──────┼─ counts: u16[ec] ──
//   │  header │                                │                   │
//   └─offset where counts begin                                    └─per-entry run lens
//
// `entry_count` (ec) is implicit — walk `counts[]` summing until the total
// hits `descriptor.row_count`. Padding is real (DuckDB's `AlignValue<8>`):
// trust the header offset, don't compute it from values size. `rle_count_t
// = uint16_t` so a single run is ≤ 65535 rows; longer runs split into
// consecutive same-value entries. DuckDB never emits zero counts.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_rle.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_scan.cuh>
#include <cuda/cmath>
#include <cuda/warp>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

constexpr uint32_t BLOCK_DIM          = 256;
constexpr uint32_t RLE_ROWS_PER_CHUNK = 2048;
constexpr uint32_t VALUES_PER_THREAD  = RLE_ROWS_PER_CHUNK / BLOCK_DIM;
// Catch silent integer truncation if BLOCK_DIM stops dividing the chunk.
static_assert(BLOCK_DIM * VALUES_PER_THREAD == RLE_ROWS_PER_CHUNK);

// Build kernel processes counts in tiles of BUILD_TILE_ENTRIES; running
// total propagates across tiles.
constexpr uint32_t BUILD_VALUES_PER_THREAD = 16;
constexpr uint32_t BUILD_TILE_ENTRIES      = BLOCK_DIM * BUILD_VALUES_PER_THREAD;  // 4096
constexpr uint32_t RLE_SMEM_MAX_ENTRIES    = BUILD_TILE_ENTRIES;

// Worst-case ec = (block_size - header) / (sizeof(T) + sizeof(rle_count_t))
// = (256 KiB - 8) / 3 for T=int8. Rounded up to a tile multiple.
constexpr uint32_t RLE_DUCKDB_MAX_EC = 87376;
constexpr uint32_t RLE_BUILD_MAX_ENTRIES =
  ::cuda::ceil_div(RLE_DUCKDB_MAX_EC, BUILD_TILE_ENTRIES) * BUILD_TILE_ENTRIES;  // 90112

constexpr uint32_t MALFORMED_FLAG = 0u;

/// Per-segment input to the build kernel.
struct rle_build_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t row_count;
};

/// Per-CTA input to the expand kernel. d_cumsum points into the device
/// concatenated cumsum buffer; entry_count is read via seg_id from the
/// per-segment array the build kernel writes.
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
  // Smallest entry idx where cumsum == row_count = count of valid entries;
  // ~0u sentinel at end-of-walk ⇒ counts didn't sum to row_count.
  __shared__ uint32_t s_first_match;
  __shared__ uint32_t s_running_total;

  uint32_t const sid = blockIdx.x;
  if (sid >= num_segs) return;
  auto const desc = descs[sid];

  if (threadIdx.x == 0) {
    s_malformed     = 0;
    s_first_match   = ~0u;
    s_running_total = 0;
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

  uint32_t const row_count = desc.row_count;
  uint16_t const* counts   = reinterpret_cast<uint16_t const*>(desc.d_bytes + s_offset);
  uint32_t const max_entries =
    s_capacity < RLE_BUILD_MAX_ENTRIES ? s_capacity : RLE_BUILD_MAX_ENTRIES;
  uint32_t* cumsum_out = d_cumsums + size_t{sid} * cumsum_stride_entries;

  // Iterate tiles of BUILD_TILE_ENTRIES counts; running sum propagates
  // through s_running_total. Stops early once we hit row_count.
  for (uint32_t tile = 0; tile < max_entries; tile += BUILD_TILE_ENTRIES) {
    uint32_t my_counts[BUILD_VALUES_PER_THREAD];
    uint32_t local_zero_seen = 0;
#pragma unroll
    for (uint32_t v = 0; v < BUILD_VALUES_PER_THREAD; ++v) {
      uint32_t const i = tile + threadIdx.x * BUILD_VALUES_PER_THREAD + v;
      if (i < max_entries) {
        uint16_t c   = counts[i];
        my_counts[v] = c;
        if (c == 0) local_zero_seen = 1;
      } else {
        my_counts[v] = 0;
      }
    }

    uint32_t my_aggregate = 0;
    uint32_t my_local_cumsum[BUILD_VALUES_PER_THREAD];
#pragma unroll
    for (uint32_t v = 0; v < BUILD_VALUES_PER_THREAD; ++v) {
      my_aggregate += my_counts[v];
      my_local_cumsum[v] = my_aggregate;
    }

    uint32_t my_prefix = 0;
    BlockScan(scan_temp).ExclusiveSum(my_aggregate, my_prefix);

    if (local_zero_seen) atomicOr(&s_malformed, 1u);

#pragma unroll
    for (uint32_t v = 0; v < BUILD_VALUES_PER_THREAD; ++v) {
      uint32_t const i  = tile + threadIdx.x * BUILD_VALUES_PER_THREAD + v;
      uint32_t const cs = s_running_total + my_prefix + my_local_cumsum[v];
      if (i < max_entries) cumsum_out[i] = cs;
      if (i < max_entries && cs == row_count) atomicMin(&s_first_match, i + 1);
    }

    // Last thread's last cumsum is the new running total for next tile.
    __syncthreads();
    if (threadIdx.x == BLOCK_DIM - 1) {
      s_running_total += my_prefix + my_local_cumsum[BUILD_VALUES_PER_THREAD - 1];
    }
    __syncthreads();

    if (s_first_match != ~0u) break;          // already covered all rows
    if (s_running_total >= row_count) break;  // overshot or just reached
  }

  if (threadIdx.x == 0) {
    uint32_t ec = s_first_match;
    if (s_malformed || ec == ~0u) {
      d_entry_counts[sid] = MALFORMED_FLAG;
      return;
    }
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

/// First index in [lo, hi) where `cumsum[idx] > key`.
__device__ __forceinline__ uint32_t rle_upper_bound(uint32_t const* __restrict__ cumsum,
                                                    uint32_t lo,
                                                    uint32_t hi,
                                                    uint32_t key)
{
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

/// Expand-kernel body, parameterised on the cumsum pointer's address space.
/// `entry_count`: per-segment ec. `lr_start`: segment-local row index of
/// this chunk's first row. `rc`: chunk row count.
template <typename T>
__device__ __forceinline__ void rle_decode_chunk_body(uint32_t const* __restrict__ cumsum,
                                                      T const* __restrict__ values,
                                                      T* __restrict__ out_chunk,
                                                      uint32_t entry_count,
                                                      uint32_t lr_start,
                                                      uint32_t rc)
{
  // Thread 0 binary-searches the chunk's first and last rows once to find
  // [band_lo, band_hi); per-thread search depth then drops from
  // log2(entry_count) to log2(band_hi - band_lo).
  __shared__ uint32_t s_band_lo;
  __shared__ uint32_t s_band_hi;
  if (threadIdx.x == 0) {
    uint32_t const last_row = lr_start + rc - 1u;
    uint32_t const lo       = rle_upper_bound(cumsum, 0, entry_count, lr_start);
    uint32_t hi             = rle_upper_bound(cumsum, 0, entry_count, last_row);
    if (hi >= entry_count) hi = entry_count - 1;
    s_band_lo = lo;
    s_band_hi = hi + 1u;
  }
  __syncthreads();
  uint32_t const band_lo = s_band_lo;
  uint32_t const band_hi = s_band_hi;

  // Avg run ≥ warpSize ⇒ most warps cover one entry; lane 0 searches once
  // and broadcasts (cheaper than 32 searches). Short-run shapes skip — the
  // bound check would miss most iterations.
  bool const long_runs_heuristic = rc / 32u >= entry_count;

  if (long_runs_heuristic) {
    uint32_t const lane = threadIdx.x & 31u;  // 32 = warpSize
#pragma unroll
    for (uint32_t v = 0; v < VALUES_PER_THREAD; ++v) {
      uint32_t const i = v * blockDim.x + threadIdx.x;
      if (i >= rc) break;
      uint32_t const warp_first = lr_start + (i & ~31u);

      uint32_t entry_warp = 0;
      if (lane == 0) {
        entry_warp = rle_upper_bound(cumsum, band_lo, band_hi, warp_first);
        if (entry_warp >= entry_count) entry_warp = entry_count - 1;
      }
      entry_warp = ::cuda::device::warp_shuffle_idx(entry_warp, 0);

      uint32_t cumsum_at_entry = 0;
      if (lane == 0) cumsum_at_entry = cumsum[entry_warp];
      cumsum_at_entry = ::cuda::device::warp_shuffle_idx(cumsum_at_entry, 0);

      uint32_t const warp_last = warp_first + 31u;

      if (warp_last < cumsum_at_entry) {
        // Whole warp on entry_warp: __ldg goes through the read-only cache
        // (one broadcast load); __stwt streams past L1 to keep cumsum hot.
        __stwt(out_chunk + i, __ldg(values + entry_warp));
      } else {
        // Warp straddles an entry boundary — per-lane fallback.
        uint32_t const local_row = lr_start + i;
        uint32_t entry           = rle_upper_bound(cumsum, band_lo, band_hi, local_row);
        if (entry >= entry_count) entry = entry_count - 1;  // never fires on well-formed input
        __stwt(out_chunk + i, __ldg(values + entry));
      }
    }
    return;
  }

#pragma unroll
  for (uint32_t v = 0; v < VALUES_PER_THREAD; ++v) {
    uint32_t const i = v * blockDim.x + threadIdx.x;
    if (i >= rc) break;
    uint32_t const local_row = lr_start + i;
    uint32_t entry           = rle_upper_bound(cumsum, band_lo, band_hi, local_row);
    if (entry >= entry_count) entry = entry_count - 1;
    __stwt(out_chunk + i, __ldg(values + entry));
  }
}

template <typename T>
__global__ void kernel_decode_rle(rle_chunk_desc const* __restrict__ descs,
                                  uint32_t const* __restrict__ d_entry_counts,
                                  T* __restrict__ d_output,
                                  uint32_t num_chunks)
{
  uint32_t const cid = blockIdx.x;
  if (cid >= num_chunks) return;

  auto const desc   = descs[cid];
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

  __shared__ uint32_t s_cumsum[RLE_SMEM_MAX_ENTRIES];

  // Two branches keep each cumsum pointer in a statically-known address
  // space (ld.shared / ld.global). A merged path would force ld.generic.
  if (ec <= RLE_SMEM_MAX_ENTRIES) {
    for (uint32_t i = threadIdx.x; i < ec; i += blockDim.x) {
      s_cumsum[i] = desc.d_cumsum[i];
    }
    __syncthreads();
    rle_decode_chunk_body<T>(s_cumsum, values, out_chunk, ec, lr_start, rc);
  } else {
    // L2 caches the gmem cumsum across CTAs of the same segment.
    rle_decode_chunk_body<T>(desc.d_cumsum, values, out_chunk, ec, lr_start, rc);
  }
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
    throw std::runtime_error("gpu_decode_table: viability invariant violated — RLE type_size " +
                             std::to_string(type_size));
  }

  std::vector<gpu_segment_desc const*> live;
  live.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    live.push_back(&seg);
  }
  if (live.empty()) return;

  size_t const num_live_segments = live.size();

  // Build per-segment build descriptors.
  std::vector<rle_build_desc> h_build(num_live_segments);
  for (size_t i = 0; i < num_live_segments; ++i) {
    h_build[i] = {live[i]->d_bytes, live[i]->bytes_size, live[i]->row_count};
  }

  // Worst-case allocation: 352 KiB/segment (90112 × uint32). Typical ec
  // is far smaller, but this lets host compute each segment's slice base
  // without a D2H sync.
  rmm::device_uvector<uint32_t> d_cumsums(num_live_segments * RLE_BUILD_MAX_ENTRIES, stream, mr);
  rmm::device_uvector<uint32_t> d_entry_counts(num_live_segments, stream, mr);
  rmm::device_uvector<rle_build_desc> d_build_descs(num_live_segments, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_build_descs.data(),
                               h_build.data(),
                               num_live_segments * sizeof(rle_build_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  kernel_build_cumsum<<<static_cast<uint32_t>(num_live_segments), BLOCK_DIM, 0, stream.value()>>>(
    d_build_descs.data(),
    d_cumsums.data(),
    RLE_BUILD_MAX_ENTRIES,
    d_entry_counts.data(),
    type_size,
    static_cast<uint32_t>(num_live_segments));

  // Per-CTA chunk descriptors; cumsum slice + values pointer computable on host.
  size_t total_chunks = 0;
  for (auto const* seg : live) {
    total_chunks += ::cuda::ceil_div(seg->row_count, RLE_ROWS_PER_CHUNK);
  }
  std::vector<rle_chunk_desc> h_descs;
  h_descs.reserve(total_chunks);
  for (size_t i = 0; i < num_live_segments; ++i) {
    auto const& seg          = *live[i];
    uint32_t const rc        = seg.row_count;
    uint32_t const num_cnks  = ::cuda::ceil_div(rc, RLE_ROWS_PER_CHUNK);
    uint8_t const* d_values  = seg.d_bytes + RLE_HEADER_SIZE;
    uint32_t const* d_cumsum = d_cumsums.data() + i * RLE_BUILD_MAX_ENTRIES;

    for (uint32_t c = 0; c < num_cnks; ++c) {
      uint32_t const local_start = c * RLE_ROWS_PER_CHUNK;
      uint32_t const this_rows   = (c + 1u < num_cnks) ? RLE_ROWS_PER_CHUNK : rc - local_start;
      h_descs.push_back(
        {d_values, d_cumsum, seg.row_offset, local_start, this_rows, static_cast<uint32_t>(i)});
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
      kernel_decode_rle<uint8_t><<<grid, BLOCK_DIM, 0, stream.value()>>>(
        d_descs.data(), d_entry_counts.data(), d_output, grid);
      break;
    case 2:
      kernel_decode_rle<uint16_t><<<grid, BLOCK_DIM, 0, stream.value()>>>(
        d_descs.data(), d_entry_counts.data(), reinterpret_cast<uint16_t*>(d_output), grid);
      break;
    case 4:
      kernel_decode_rle<uint32_t><<<grid, BLOCK_DIM, 0, stream.value()>>>(
        d_descs.data(), d_entry_counts.data(), reinterpret_cast<uint32_t*>(d_output), grid);
      break;
    case 8:
      kernel_decode_rle<uint64_t><<<grid, BLOCK_DIM, 0, stream.value()>>>(
        d_descs.data(), d_entry_counts.data(), reinterpret_cast<uint64_t*>(d_output), grid);
      break;
    default:
      // Unreachable — guarded by the type_size check at function entry.
      break;
  }
}

}  // namespace sirius::cuda::scan
