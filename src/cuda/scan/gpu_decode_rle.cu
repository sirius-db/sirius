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
//   byte 0      8                                       rle_count_offset           seg_end
//   │           │                                       │                          │
//   ├─ uint64 ──┼── values: T[entry_count] (0-padded) ──┼── counts: u16[ec] ───────┤
//   │  header   │                                       │                          │
//      └─ stores rle_count_offset                           └─ per-entry run lengths
//
// - `entry_count` (ec) is implicit — walk `counts[]` summing until the total
//    hits `descriptor.row_count`.
// - `rle_count_t' = uint16_t, so a single run is ≤ 65535 rows
//
// Decoding is performed in 2 passes
// Pass 1: Compute per-segment prefix sums over the run lengths (counts) to determine the offsets to
//         which to scatter run values. Work partitioning is 1 CTA per segment. ('Build' kernel)
// Pass 2: Scatter run values to output positions by binary searching the segment-level prefix sums
//         to find the value index from which to copy. Work partitioning is 1 CTA per chunk, where a
//         chunk is a pre-defined slice of a segment for statically balancing the stores across
//         CTAs. ('Expand' kernel)
//===----------------------------------------------------------------------===//

#include "cuda/scan/detail/warp.cuh"
#include "cuda/scan/gpu_decode_rle.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda/cmath>
#include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

using rle_count_t = uint16_t;

constexpr uint32_t BLOCK_DIM          = 256;
constexpr uint32_t RLE_ROWS_PER_CHUNK = 2048;
constexpr uint32_t VALUES_PER_THREAD  = RLE_ROWS_PER_CHUNK / BLOCK_DIM;
// Catch silent integer truncation if BLOCK_DIM stops dividing the chunk.
static_assert(BLOCK_DIM * VALUES_PER_THREAD == RLE_ROWS_PER_CHUNK);

// Build kernel processes counts in tiles of BUILD_TILE_ENTRIES; running
// total propagates across tiles.
constexpr uint32_t BUILD_VALUES_PER_THREAD = 8;
constexpr uint32_t BUILD_TILE_ENTRIES      = BLOCK_DIM * BUILD_VALUES_PER_THREAD;  // 2048
constexpr uint32_t RLE_SMEM_MAX_ENTRIES    = BUILD_TILE_ENTRIES;

// Worst-case ec = (block_size - header) / (sizeof(T) + sizeof(rle_count_t))
// = (256 KiB - 8) / 3 for T=int8. Rounded up to a tile multiple.
constexpr uint32_t RLE_DUCKDB_MAX_EC = 87376;
constexpr uint32_t RLE_BUILD_MAX_ENTRIES =
  ::cuda::ceil_div(RLE_DUCKDB_MAX_EC, BUILD_TILE_ENTRIES) * BUILD_TILE_ENTRIES;  // 90112

constexpr uint32_t MALFORMED_FLAG = 0u;
using detail::FULL_MASK;

//! @brief Per-CTA input for segment-level prefix sum kernel over counts array.
struct rle_segment_desc {
  uint8_t const* d_bytes;
  uint32_t bytes_size;
  uint32_t segment_row_count;
  /// Per-value byte width. DuckDB's `bytes_size` includes zero-padded slack
  /// when the segment is the last in its block; `parse_rle_metadata` uses
  /// `type_size` to bound iteration at the real-count boundary instead.
  uint32_t type_size;
};

//! @brief Per-CTA input for chunk-level value filling kernel into the output column.
struct rle_chunk_desc {
  uint8_t const* d_values;
  uint32_t const* d_prefix_sums;
  uint32_t segment_row_start;
  uint32_t chunk_row_start;
  uint32_t chunk_row_count;
  uint32_t segment_id;
};

//===----------------------------------------------------------------------===//
// Segment-level prefix sum kernel over counts array.
//===----------------------------------------------------------------------===//

//! @brief Stateful prefix functor that carries running total across successive `cub::BlockScan`
//! invocations; used to scan a single segment in tiles.
struct BlockPrefixCallbackOp {
  uint32_t running_total;
  __device__ uint32_t operator()(uint32_t block_aggregate)
  {
    auto const old = running_total;
    running_total += block_aggregate;
    return old;
  }
};

//! @brief Result of parsing a segment's RLE header.
//!
//! `is_malformed` is true if any header-level invariant fails:
//!  - segment smaller than the 8-byte header
//!  - counts_offset out of range
//! When malformed, the pointer/count fields are unspecified and the caller must zero-fill rather
//! than dereference.
struct rle_parsed_metadata {
  rle_count_t const* counts_ptr;
  /// Real entry count of the counts[] region, computed from the values
  /// region length (RLE has a 1:1 invariant between values and counts).
  /// Stops iteration short of any zero-padded slack at the block tail.
  uint32_t n_counts_real;
  bool is_malformed;
};

//! @brief Decode an RLE segment's 8-byte header and bound the counts[] region.
//! Intended for single-thread execution and shared memory broadcast.
__device__ __forceinline__ rle_parsed_metadata parse_rle_metadata(rle_segment_desc const& desc)
{
  rle_parsed_metadata md{};
  md.is_malformed = false;

  if (desc.bytes_size < RLE_HEADER_SIZE) {
    md.is_malformed = true;
    return md;
  }

  uint64_t counts_offset = 0;
  memcpy(&counts_offset, desc.d_bytes, sizeof(uint64_t));
  if (counts_offset < RLE_HEADER_SIZE || counts_offset >= desc.bytes_size) {
    md.is_malformed = true;
    return md;
  }
  // Real entry count = values_region_bytes / type_size (1:1 invariant). If
  // type_size is unset or doesn't divide cleanly, fall back to the bytes-
  // derived bound and rely on the malformed-vs-match swap below to recover.
  const uint64_t values_region_bytes = counts_offset - RLE_HEADER_SIZE;
  const uint64_t counts_region_bytes = desc.bytes_size - counts_offset;
  uint64_t n_real                    = 0;
  if (desc.type_size > 0 && (values_region_bytes % desc.type_size) == 0) {
    n_real = values_region_bytes / desc.type_size;
  } else {
    n_real = counts_region_bytes / sizeof(rle_count_t);
  }

  md.counts_ptr    = reinterpret_cast<rle_count_t const*>(desc.d_bytes + counts_offset);
  md.n_counts_real = static_cast<uint32_t>(n_real);
  return md;
}

//! @brief Compute the prefix sums over the counts in the segment to determine the positions in the
//! output columns into which to fill values.
//!
//! The work partitioning is 1 CTA per segment.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int PREFIX_SUM_BUFFER_ENTRIES_PER_SEGMENT>
__global__ void kernel_build_rle(rle_segment_desc const* __restrict__ segment_descriptors,
                                 int num_segments,
                                 uint32_t* __restrict__ d_prefix_sums,
                                 uint32_t* __restrict__ d_counts)
{
  using BlockLoad =
    cub::BlockLoad<uint16_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
  using BlockScan   = cub::BlockScan<uint32_t, BLOCK_THREADS>;
  using BlockReduce = cub::BlockReduce<uint32_t, BLOCK_THREADS>;
  using BlockStore =
    cub::BlockStore<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;
  using SumOp = ::cuda::std::plus<uint32_t>;
  using MinOp = ::cuda::minimum<uint32_t>;

  auto constexpr ITEMS_PER_TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
  auto constexpr NO_MATCH       = ::cuda::std::numeric_limits<uint32_t>::max();

  // Each tile uses Load → upcast → Scan → Store → Reduce in sequence. The CUB temp_storages are
  // alive only during their respective collective call and are __syncthreads-bounded; union them
  // to cut static SMEM. `prefix_op` carries cross-tile state so it stays separate.
  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockReduce::TempStorage reduce;
    typename BlockStore::TempStorage store;
  } temp;
  __shared__ BlockPrefixCallbackOp prefix_op;
  __shared__ rle_parsed_metadata s_md;
  __shared__ bool s_is_malformed;
  __shared__ uint32_t s_first_match;

  uint16_t thread_counts[ITEMS_PER_THREAD];
  uint32_t thread_sums[ITEMS_PER_THREAD];
  auto const segment_id         = blockIdx.x;
  auto const segment_descriptor = segment_descriptors[segment_id];

  //===----------Metadata parsing----------===//
  if (threadIdx.x == 0) {
    s_md                    = parse_rle_metadata(segment_descriptor);
    s_is_malformed          = s_md.is_malformed;
    prefix_op.running_total = 0;
  }
  __syncthreads();

  if (s_is_malformed) {
    if (threadIdx.x == 0) { d_counts[segment_id] = MALFORMED_FLAG; }
    return;
  }

  auto const n_segment_rows = segment_descriptor.segment_row_count;
  auto const* counts_ptr    = s_md.counts_ptr;
  // Bound by `n_counts_real` (skips block-tail zero slack) and clamp to the
  // prefix-sum slot so BlockStore can't write past it.
  auto const n_counts_max = ::cuda::std::min(
    s_md.n_counts_real, static_cast<uint32_t>(PREFIX_SUM_BUFFER_ENTRIES_PER_SEGMENT));
  auto* prefix_sum_ptr =
    d_prefix_sums + static_cast<size_t>(segment_id) * PREFIX_SUM_BUFFER_ENTRIES_PER_SEGMENT;

  //===----------Divide segment into tiles----------===//
  auto const n_tiles = ::cuda::ceil_div(n_counts_max, ITEMS_PER_TILE);
  for (int tile = 0; tile < n_tiles; ++tile) {
    auto const tile_offset          = tile * ITEMS_PER_TILE;
    auto const n_counts_max_current = n_counts_max - tile_offset;
    //===----------Process tile----------===//
    // 1) Load the count data (guarded on partial tile)
    // 2) Upcast counts to uint32 for prefix sum (and flag any zero count
    //    inside the valid-entry portion of this tile — DuckDB never emits
    //    count == 0, so its presence within `[0, n_counts_max)` is malformed).
    // 3) Inclusive-scan with the stateful prefix callback (carries across tiles)
    // 4) Store the sums (guarded on partial tile)
    // 5) Early-exit if we've covered every row in the segment
    if (n_counts_max_current < ITEMS_PER_TILE) {
      BlockLoad(temp.load).Load(counts_ptr + tile_offset, thread_counts, n_counts_max_current, 0);
    } else {
      BlockLoad(temp.load).Load(counts_ptr + tile_offset, thread_counts);
    }

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      auto const local_idx = threadIdx.x * ITEMS_PER_THREAD + i;
      auto const count     = static_cast<uint32_t>(thread_counts[i]);
      // Zero counts inside the valid range are malformed.
      if (count == 0 && local_idx < n_counts_max_current) {
        // monotonic update of malformed
        s_is_malformed = true;
      }
      thread_sums[i] = count;
    }

    __syncthreads();
    BlockScan(temp.scan).InclusiveScan(thread_sums, thread_sums, SumOp{}, prefix_op);

    __syncthreads();
    if (n_counts_max_current < ITEMS_PER_TILE) {
      BlockStore(temp.store).Store(prefix_sum_ptr + tile_offset, thread_sums, n_counts_max_current);
    } else {
      BlockStore(temp.store).Store(prefix_sum_ptr + tile_offset, thread_sums);
    }

    // Find this thread's earliest entry whose prefix sum equals row_count exactly.
    // A prefix that *overshoots* row_count means counts didn't sum to row_count at any entry —
    // segment is malformed.
    auto my_first_match = NO_MATCH;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      auto const local_idx = threadIdx.x * ITEMS_PER_THREAD + i;
      if (local_idx >= n_counts_max_current) break;
      if (thread_sums[i] == n_segment_rows) {
        my_first_match = tile_offset + local_idx + 1;
        break;
      }
      if (thread_sums[i] > n_segment_rows) {
        s_is_malformed = true;
        break;
      }
    }
    // Reduce per-thread candidates to the block-wide minimum: the smallest
    // entry index across all threads where prefix sum first crossed the threshold.
    __syncthreads();
    auto const tile_first_match = BlockReduce(temp.reduce).Reduce(my_first_match, MinOp{});
    if (threadIdx.x == 0) { s_first_match = tile_first_match; }
    __syncthreads();

    // Match wins over malformed: once a tile entry's prefix sum hits
    // n_segment_rows, the segment is well-formed up to that point. A zero
    // count past that match comes from block-tail slack (encoder never
    // emits count == 0 in real data), not corruption.
    if (s_first_match != NO_MATCH) {
      if (threadIdx.x == 0) { d_counts[segment_id] = s_first_match; }
      return;
    }
    if (s_is_malformed) {
      if (threadIdx.x == 0) { d_counts[segment_id] = MALFORMED_FLAG; }
      return;
    }
  }

  // Counts walk finished without summing to row_count — segment is malformed.
  if (threadIdx.x == 0) { d_counts[segment_id] = MALFORMED_FLAG; }
}

//===----------------------------------------------------------------------===//
// Expand kernel.
//===----------------------------------------------------------------------===//

//! @brief Fill the values into the output array based on the corresponding output position
//! determined by the segment-local prefix sum. @p segment_entry_count is the per-segment number of
//! counts, @p chunk_rows is the number of rows in this segment chunk, and @p chunk_first_row is the
//! segment-local row index of this chunk's first row.
template <int ITEMS_PER_THREAD, typename T>
__device__ __forceinline__ void decode_chunk_rle(uint32_t const* __restrict__ prefix_sums,
                                                 T const* __restrict__ values,
                                                 T* __restrict__ out_chunk,
                                                 uint32_t segment_entry_count,
                                                 uint32_t chunk_first_row,
                                                 uint32_t chunk_rows)
{
  // Return the entry into the vals array given the row idx
  auto find_entry = [&](uint32_t low, uint32_t high, uint32_t row) -> uint32_t {
    auto lo = low;
    auto hi = high;
    while (lo < hi) {
      auto const mid = lo + ((hi - lo) / 2);
      if (prefix_sums[mid] <= row) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return ::cuda::std::min(lo, high - 1);
  };

  // Thread 0 binary-searches the chunk's first and last rows once to find [chunk_entry_lo,
  // chunk_entry_hi) to narrow the search space in the prefix sums array for other threads in this
  // chunk.
  __shared__ uint32_t s_chunk_entry_lo;
  __shared__ uint32_t s_chunk_entry_hi;
  if (threadIdx.x == 0) {
    auto const chunk_last_row = chunk_first_row + chunk_rows - 1;
    auto const lo             = find_entry(0, segment_entry_count, chunk_first_row);
    auto const hi             = find_entry(lo, segment_entry_count, chunk_last_row);
    s_chunk_entry_lo          = lo;
    s_chunk_entry_hi          = hi + 1;
  }
  __syncthreads();
  auto const chunk_entry_lo = s_chunk_entry_lo;
  auto const chunk_entry_hi = s_chunk_entry_hi;

  // Heuristic: if the average number of rows per warp in this chunk is >= the number of RLE entries
  // in the chunk, try to use a single search into the prefix_sums array for each warp in case all
  // the threads in the warp fill the same value. This is an optimization for long runs.
  if (chunk_rows >= cub::detail::warp_threads * (chunk_entry_hi - chunk_entry_lo)) {
    auto const lane = threadIdx.x % cub::detail::warp_threads;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      auto const chunk_row        = i * blockDim.x + threadIdx.x;
      auto const lane_0_chunk_row = chunk_row - (chunk_row % cub::detail::warp_threads);
      if (lane_0_chunk_row >= chunk_rows) break;
      auto const lane_0_segment_row = chunk_first_row + lane_0_chunk_row;

      uint32_t lane_0_entry      = 0;
      uint32_t lane_0_prefix_sum = 0;
      if (lane == 0) {
        lane_0_entry      = find_entry(chunk_entry_lo, chunk_entry_hi, lane_0_segment_row);
        lane_0_prefix_sum = prefix_sums[lane_0_entry];
      }
      lane_0_entry = cub::ShuffleIndex<cub::detail::warp_threads>(lane_0_entry, 0, FULL_MASK);
      lane_0_prefix_sum =
        cub::ShuffleIndex<cub::detail::warp_threads>(lane_0_prefix_sum, 0, FULL_MASK);

      auto const lane_31_row = lane_0_segment_row + cub::detail::warp_threads - 1;

      if (lane_31_row < lane_0_prefix_sum) {
        cub::ThreadStore<cub::STORE_WT>(out_chunk + chunk_row,
                                        cub::ThreadLoad<cub::LOAD_LDG>(values + lane_0_entry));
      } else {
        // Warp straddles an entry boundary — per-lane fallback.
        if (chunk_row >= chunk_rows) { break; }
        auto const segment_row = chunk_first_row + chunk_row;
        auto const entry       = find_entry(chunk_entry_lo, chunk_entry_hi, segment_row);
        cub::ThreadStore<cub::STORE_WT>(out_chunk + chunk_row,
                                        cub::ThreadLoad<cub::LOAD_LDG>(values + entry));
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      auto const chunk_row = i * blockDim.x + threadIdx.x;
      if (chunk_row >= chunk_rows) break;
      auto const segment_row = chunk_first_row + chunk_row;
      auto const entry       = find_entry(chunk_entry_lo, chunk_entry_hi, segment_row);
      cub::ThreadStore<cub::STORE_WT>(out_chunk + chunk_row,
                                      cub::ThreadLoad<cub::LOAD_LDG>(values + entry));
    }
  }
}

//! @brief Wrapper kernel for `decode_chunk_rle` that loads the chunk descriptor and stages the
//! segment prefix sums in shared memory, if possible, for faster binary search.
template <int ITEMS_PER_THREAD, typename T>
__global__ void kernel_expand_rle(rle_chunk_desc const* __restrict__ chunk_descs,
                                  int num_chunks,
                                  uint32_t const* __restrict__ d_segment_entry_counts,
                                  T* __restrict__ d_output)
{
  __shared__ uint32_t s_segment_prefix_sums[RLE_SMEM_MAX_ENTRIES];

  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;

  auto const chunk_desc          = chunk_descs[chunk_id];
  auto const segment_entry_count = d_segment_entry_counts[chunk_desc.segment_id];
  auto const chunk_rows          = chunk_desc.chunk_row_count;
  auto const chunk_row_start     = chunk_desc.chunk_row_start;
  auto* out_chunk_values = d_output + chunk_desc.segment_row_start + chunk_desc.chunk_row_start;
  auto const* in_segment_values = reinterpret_cast<T const*>(chunk_desc.d_values);

  if (segment_entry_count == MALFORMED_FLAG) {
    // Zero-fill the output
    for (int i = threadIdx.x; i < chunk_rows; i += blockDim.x) {
      out_chunk_values[i] = 0;
    }
    return;
  }

  // Try to use SMEM for the binary search space for a chunk
  if (segment_entry_count <= RLE_SMEM_MAX_ENTRIES) {
    for (int i = threadIdx.x; i < segment_entry_count; i += blockDim.x) {
      s_segment_prefix_sums[i] = chunk_desc.d_prefix_sums[i];
    }
    __syncthreads();
    decode_chunk_rle<ITEMS_PER_THREAD>(s_segment_prefix_sums,
                                       in_segment_values,
                                       out_chunk_values,
                                       segment_entry_count,
                                       chunk_row_start,
                                       chunk_rows);
  } else {
    decode_chunk_rle<ITEMS_PER_THREAD>(chunk_desc.d_prefix_sums,
                                       in_segment_values,
                                       out_chunk_values,
                                       segment_entry_count,
                                       chunk_row_start,
                                       chunk_rows);
  }
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public entry.
//===----------------------------------------------------------------------===//

//! @brief Decode all RLE-encoded segments in @p run into @p d_output.
//!
//! Runs the two-pass pipeline: a per-segment prefix-sum build kernel that
//! computes the cumulative row counts and the per-segment entry count,
//! followed by a per-chunk expand kernel that scatters run values to their
//! output positions. Malformed segments are zero-filled.
//!
//! @param run        Column of segments sharing the RLE codec.
//! @param d_output   Device pointer to the output column buffer; must hold
//!                   at least `sum(segment.row_count) * type_size` bytes.
//! @param type       Cudf type of the output column (unused; kept for API
//!                   parity with the dispatcher's other codec entry points).
//! @param type_size  Width of one output value in bytes. Must be 1, 2, 4, or
//!                   8; otherwise the function throws.
//! @param stream     Stream on which all kernels and allocations are issued.
//! @param mr         Memory resource for the transient prefix-sum / descriptor
//!                   buffers allocated during decode.
void decode_rle_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type /*type*/,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  if (type_size != 1 && type_size != 2 && type_size != 4 && type_size != 8) {
    throw std::runtime_error("[gpu_decode_table]: viability invariant violated — RLE type_size " +
                             std::to_string(type_size));
  }

  std::vector<gpu_segment_desc const*> live_segments;
  live_segments.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    live_segments.push_back(&seg);
  }
  if (live_segments.empty()) return;
  auto const num_live_segments = live_segments.size();

  // Build per-segment build descriptors.
  std::vector<rle_segment_desc> h_build_descs(num_live_segments);
  for (size_t i = 0; i < num_live_segments; ++i) {
    h_build_descs[i] = {live_segments[i]->d_bytes,
                        live_segments[i]->bytes_size,
                        live_segments[i]->row_count,
                        type_size};
  }

  // Worst-case allocation: 352 KiB/segment.
  rmm::device_uvector<uint32_t> d_segment_prefix_sums(
    num_live_segments * RLE_BUILD_MAX_ENTRIES, stream, mr);
  rmm::device_uvector<uint32_t> d_entry_counts(num_live_segments, stream, mr);
  rmm::device_uvector<rle_segment_desc> d_build_descs(num_live_segments, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_build_descs.data(),
                               h_build_descs.data(),
                               num_live_segments * sizeof(rle_segment_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  //===----------Pass 1: Build----------===//
  kernel_build_rle<BLOCK_DIM, BUILD_VALUES_PER_THREAD, RLE_BUILD_MAX_ENTRIES>
    <<<static_cast<uint32_t>(num_live_segments), BLOCK_DIM, 0, stream.value()>>>(
      d_build_descs.data(),
      static_cast<uint32_t>(num_live_segments),
      d_segment_prefix_sums.data(),
      d_entry_counts.data());

  // Build per-chunk expand descriptors.
  size_t total_chunks = 0;
  for (auto const* seg : live_segments) {
    total_chunks += ::cuda::ceil_div(seg->row_count, RLE_ROWS_PER_CHUNK);
  }
  std::vector<rle_chunk_desc> h_expand_descs;
  h_expand_descs.reserve(total_chunks);
  for (size_t i = 0; i < num_live_segments; ++i) {
    auto const& seg                  = *live_segments[i];
    auto const segment_row_count     = seg.row_count;
    auto const n_chunks              = ::cuda::ceil_div(segment_row_count, RLE_ROWS_PER_CHUNK);
    auto const* d_segment_values     = seg.d_bytes + RLE_HEADER_SIZE;
    auto const* d_segment_prefix_sum = d_segment_prefix_sums.data() + i * RLE_BUILD_MAX_ENTRIES;

    for (uint32_t c = 0; c < n_chunks; ++c) {
      auto const local_start = c * RLE_ROWS_PER_CHUNK;
      auto const chunk_row_count =
        (c < n_chunks - 1) ? RLE_ROWS_PER_CHUNK : segment_row_count - local_start;
      h_expand_descs.push_back({d_segment_values,
                                d_segment_prefix_sum,
                                seg.row_offset,
                                local_start,
                                chunk_row_count,
                                static_cast<uint32_t>(i)});
    }
  }
  if (h_expand_descs.empty()) return;

  rmm::device_uvector<rle_chunk_desc> d_expand_descs(h_expand_descs.size(), stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_expand_descs.data(),
                               h_expand_descs.data(),
                               h_expand_descs.size() * sizeof(rle_chunk_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  auto const n_ctas = static_cast<uint32_t>(h_expand_descs.size());

  //===----------Pass 2: Expand----------===//
  switch (type_size) {
    case 1:
      kernel_expand_rle<VALUES_PER_THREAD, uint8_t><<<n_ctas, BLOCK_DIM, 0, stream.value()>>>(
        d_expand_descs.data(), n_ctas, d_entry_counts.data(), d_output);
      break;
    case 2:
      kernel_expand_rle<VALUES_PER_THREAD, uint16_t>
        <<<n_ctas, BLOCK_DIM, 0, stream.value()>>>(d_expand_descs.data(),
                                                   n_ctas,
                                                   d_entry_counts.data(),
                                                   reinterpret_cast<uint16_t*>(d_output));
      break;
    case 4:
      kernel_expand_rle<VALUES_PER_THREAD, uint32_t>
        <<<n_ctas, BLOCK_DIM, 0, stream.value()>>>(d_expand_descs.data(),
                                                   n_ctas,
                                                   d_entry_counts.data(),
                                                   reinterpret_cast<uint32_t*>(d_output));
      break;
    case 8:
      kernel_expand_rle<VALUES_PER_THREAD, uint64_t>
        <<<n_ctas, BLOCK_DIM, 0, stream.value()>>>(d_expand_descs.data(),
                                                   n_ctas,
                                                   d_entry_counts.data(),
                                                   reinterpret_cast<uint64_t*>(d_output));
      break;
    default:
      // Unreachable — guarded by the type_size check at function entry.
      break;
  }
}

}  // namespace sirius::cuda::scan
