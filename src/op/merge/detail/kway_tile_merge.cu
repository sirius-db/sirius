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

// A merge is sequential, so its parallelism is the number of independent runs. The caller's tiles
// alone leave most of the device idle on a small round, so every tile is split into sub-tiles,
// one thread each: the thread first finds the co-ranks of the end of its sub-tile by
// multi-sequence selection inside the tile's sub-ranges, then merges its rows with a tournament
// tree. All of a thread's state is in shared memory, transposed (element e of thread t at
// [e * blockDim.x + t]) so the lanes of a warp that touch the same element hit different banks.
//
// Limit: k <= kMaxKwayMergeInputs (4096), where one thread's state (40 KiB) still fits the default
// 48 KiB of shared memory of a CTA; tile_merge throws std::invalid_argument for a larger k.

#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/detail/kway_comparator.cuh"
#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::op::merge::detail {

namespace {

/// Shared memory a CTA may use.
constexpr std::size_t kTileMergeSharedBytes = 40 * 1024;
/// Shared memory a CTA may use when there are more CTAs than SMs: above it fewer CTAs fit an SM.
/// Measured on 188 SMs at fan-in 64 (640 B per thread), rounds of 1.94M and 1.92M rows:
/// 51 threads, 32,640 B per CTA: 3.47 s; 52 threads, 33,280 B: 4.33 s; flat on either side.
constexpr std::size_t kTileMergeDenseSharedBytes = 32 * 1024;
/// A tile is split only when the sub-range of an input fits the 16-bit selection scratch.
constexpr cudf::size_type kTileMergeMaxSplitTileRows = 65535;

/// Shared-memory bytes of one thread: lo and hi (32-bit, k entries) and the tree (16-bit, kp).
constexpr std::size_t thread_state_bytes(std::int32_t padded_inputs)
{
  return static_cast<std::size_t>(padded_inputs) *
         (2 * sizeof(cudf::size_type) + sizeof(std::uint16_t));
}

// The public fan-in limit is what one thread's state and the 16-bit tree entries allow.
static_assert(thread_state_bytes(static_cast<std::int32_t>(kMaxKwayMergeInputs)) <=
              kTileMergeSharedBytes);
static_assert(kMaxKwayMergeInputs <= UINT16_MAX);

/// Thread `threadIdx.x` of block `blockIdx.x` merges sub-tile `threadIdx.x % sub_tiles` of tile
/// `blockIdx.x * (blockDim.x / sub_tiles) + threadIdx.x / sub_tiles`. The shared memory holds
/// blockDim.x * (2 * padded_inputs 32-bit words + padded_inputs 16-bit words).
template <typename View>
__global__ void tile_merge_kernel(View keys,
                                  const cudf::size_type* tile_coranks,
                                  std::int32_t num_inputs,
                                  std::int32_t padded_inputs,
                                  cudf::size_type tiles,
                                  cudf::size_type target,
                                  cudf::size_type tile_rows,
                                  std::int32_t sub_tiles,
                                  cudf::size_type sub_rows,
                                  std::int32_t* out_input,
                                  cudf::size_type* out_row)
{
  extern __shared__ cudf::size_type shared_state[];

  const std::int32_t k           = num_inputs;
  const std::int32_t kp          = padded_inputs;
  const std::uint32_t stride     = blockDim.x;
  const std::uint32_t tid        = threadIdx.x;
  cudf::size_type* const lo_base = shared_state + tid;
  cudf::size_type* const hi_base = shared_state + static_cast<std::size_t>(kp) * stride + tid;
  std::uint16_t* const tree_base =
    reinterpret_cast<std::uint16_t*>(shared_state + 2 * static_cast<std::size_t>(kp) * stride) +
    tid;
  auto lo   = [&](std::int32_t i) -> cudf::size_type& { return lo_base[i * stride]; };
  auto hi   = [&](std::int32_t i) -> cudf::size_type& { return hi_base[i * stride]; };
  auto tree = [&](std::int32_t n) -> std::uint16_t& { return tree_base[n * stride]; };

  const std::int32_t sub  = static_cast<std::int32_t>(tid) % sub_tiles;
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.x) * (stride / sub_tiles) +
                            static_cast<std::int32_t>(tid) / sub_tiles;
  // This sub-tile is the tile-local ranks [sub_begin, sub_end).
  const std::int64_t tile_begin = tile * tile_rows;
  const std::int64_t tile_left  = tile < tiles ? target - tile_begin : 0;
  const std::int64_t tile_size  = tile_left < tile_rows ? tile_left : tile_rows;
  const std::int64_t sub_begin  = static_cast<std::int64_t>(sub) * sub_rows;
  const std::int64_t sub_end = sub_begin + sub_rows < tile_size ? sub_begin + sub_rows : tile_size;
  // Threads past the last tile, or past the last row of a short tile, only pass the barrier.
  const bool active = sub_begin < tile_size;

  const cudf::size_type* begin = tile_coranks + (active ? tile * k : 0);
  const cudf::size_type* end   = begin + k;

  // Leave lo == hi == the co-ranks of the end of this sub-tile.
  if (active && sub_end == tile_size) {
    for (std::int32_t i = 0; i < k; ++i) {
      lo(i) = end[i];
      hi(i) = end[i];
    }
  } else if (active) {
    // Invariant: lo[i] <= c[i] <= hi[i] for the co-ranks c of rank sub_end. `need` rows are
    // still to be selected out of the `open` rows inside the intervals.
    std::int64_t need         = sub_end;
    std::int64_t open         = 0;
    std::int32_t widest       = 0;
    cudf::size_type widest_by = -1;
    for (std::int32_t i = 0; i < k; ++i) {
      const cudf::size_type width = end[i] - begin[i];
      lo(i)                       = begin[i];
      hi(i)                       = end[i];
      open += width;
      if (width > widest_by) {
        widest_by = width;
        widest    = i;
      }
    }
    while (need > 0 && need < open) {
      const std::int32_t pivot_input  = widest;
      const cudf::size_type pivot_row = lo(pivot_input) + (hi(pivot_input) - lo(pivot_input)) / 2;
      // The tree is unused until the merge: it holds the rows of each interval before the pivot.
      std::int64_t before = 0;
      for (std::int32_t i = 0; i < k; ++i) {
        const cudf::size_type first = lo(i);
        cudf::size_type left        = first;
        if (i == pivot_input) {
          left = pivot_row;
        } else {
          cudf::size_type right = hi(i);
          while (left < right) {
            const cudf::size_type mid = left + (right - left) / 2;
            if (keys.less(i, mid, pivot_input, pivot_row)) {
              left = mid + 1;
            } else {
              right = mid;
            }
          }
        }
        tree(i) = static_cast<std::uint16_t>(left - first);
        before += left - first;
      }
      // Searching only inside [lo, hi) clamps the counts, which cannot change this decision.
      const bool pivot_selected = before < need;
      widest_by                 = -1;
      for (std::int32_t i = 0; i < k; ++i) {
        const cudf::size_type split = lo(i) + tree(i);
        if (pivot_selected) {
          lo(i) = split + (i == pivot_input ? 1 : 0);
        } else {
          hi(i) = split;
        }
        const cudf::size_type width = hi(i) - lo(i);
        if (width > widest_by) {
          widest_by = width;
          widest    = i;
        }
      }
      if (pivot_selected) {
        need -= before + 1;
        open -= before + 1;
      } else {
        open = before;
      }
    }
    // Either no open row is selected (the co-ranks are lo) or all of them are (hi).
    for (std::int32_t i = 0; i < k; ++i) {
      if (need == 0) {
        hi(i) = lo(i);
      } else {
        lo(i) = hi(i);
      }
    }
  }
  if (sub_tiles > 1) { __syncthreads(); }
  if (!active) { return; }

  // The sub-tile begins where its left neighbour ends; nothing writes a hi after the barrier.
  for (std::int32_t i = 0; i < k; ++i) {
    lo(i) = sub == 0 ? begin[i] : (&hi(i))[-1];
  }

  auto alive = [&](std::int32_t i) { return i < k && lo(i) < hi(i); };
  // True when the next row of input a is emitted before the next row of input b (a != b).
  auto wins = [&](std::int32_t a, std::int32_t b) {
    if (!alive(b)) { return true; }
    if (!alive(a)) { return false; }
    return keys.less(a, lo(a), b, lo(b));
  };
  // Winner of the subtree of child node c; nodes kp .. 2kp-1 are the leaves.
  auto winner_of = [&](std::int32_t c) {
    return c >= kp ? c - kp : static_cast<std::int32_t>(tree(c));
  };
  auto play = [&](std::int32_t node) {
    const std::int32_t left  = winner_of(2 * node);
    const std::int32_t right = winner_of(2 * node + 1);
    tree(node)               = static_cast<std::uint16_t>(wins(left, right) ? left : right);
  };

  for (std::int32_t node = kp - 1; node >= 1; --node) {
    play(node);
  }

  const std::int64_t out_end = tile_begin + sub_end;
  for (std::int64_t out = tile_begin + sub_begin; out < out_end; ++out) {
    const std::int32_t winner = tree(1);
    // Only when the co-ranks do not cover the tile; never read past a sub-range.
    if (!alive(winner)) { break; }
    out_input[out] = winner;
    out_row[out]   = lo(winner)++;
    for (std::int32_t node = (winner + kp) >> 1; node >= 1; node >>= 1) {
      play(node);
    }
  }
}

/// What the launch shape depends on, of the calling thread's current device.
struct device_traits {
  int sm_count{0};                  ///< 0: not queried yet
  int compute_capability_major{0};
};

/// The traits of the calling thread's current device. The attributes are queried once per device.
device_traits current_device_traits()
{
  static std::mutex mutex;
  static std::vector<device_traits> traits;  // by device ordinal
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  const auto slot = static_cast<std::size_t>(device);
  const std::lock_guard lock(mutex);
  if (traits.size() <= slot) { traits.resize(slot + 1); }
  if (traits[slot].sm_count == 0) {
    int sm_count = 0;
    int major    = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    traits[slot] = device_traits{std::max(sm_count, 1), major};
  }
  return traits[slot];
}

/// Compute capability major version of the devices the dense shape was measured on (GB200).
constexpr int kDenseShapeComputeCapabilityMajor = 10;

/// The launch shape whose CTAs use at most @p shared_limit bytes of shared memory; a single
/// thread whose state is larger keeps it.
tile_merge_shape shape_within(std::size_t shared_limit,
                              cudf::size_type tiles,
                              cudf::size_type tile_rows,
                              std::int32_t padded_inputs,
                              int sm_count,
                              std::int32_t requested_sub_tiles,
                              const tile_merge_policy& policy)
{
  const std::size_t state_bytes = thread_state_bytes(padded_inputs);
  const auto block_limit        = static_cast<std::size_t>(std::max(policy.max_block_threads, 1));
  const auto max_block_threads  =
    static_cast<std::int32_t>(std::clamp<std::size_t>(shared_limit / state_bytes, 1, block_limit));

  // Sub-tiles per tile that bring the merge threads to about policy.threads_per_sm per SM.
  const std::int64_t threads  = sm_count * std::max<std::int64_t>(policy.threads_per_sm, 1);
  const auto device_sub_tiles = static_cast<std::int32_t>(
    std::min<std::int64_t>((threads - 1) / tiles + 1, static_cast<std::int64_t>(block_limit)));

  // A short last tile keeps the sub-tile size of the full tiles and leaves its last threads idle.
  const std::int32_t max_sub_tiles =
    tile_rows > kTileMergeMaxSplitTileRows
      ? 1
      : std::clamp(
          tile_rows / std::max<cudf::size_type>(policy.min_sub_tile_rows, 1), 1, max_block_threads);
  tile_merge_shape shape{};
  shape.sub_tiles =
    std::clamp(requested_sub_tiles > 0 ? requested_sub_tiles : device_sub_tiles, 1, max_sub_tiles);
  shape.sub_rows      = (tile_rows - 1) / shape.sub_tiles + 1;
  shape.block_tiles   = std::min<std::int32_t>(max_block_threads / shape.sub_tiles, tiles);
  shape.block_threads = shape.block_tiles * shape.sub_tiles;
  shape.blocks        = static_cast<std::uint32_t>((tiles - 1) / shape.block_tiles + 1);
  shape.shared_bytes  = state_bytes * static_cast<std::size_t>(shape.block_threads);
  return shape;
}

}  // namespace

bool dense_tile_shape_applies(bool specialized_key,
                              std::int32_t num_inputs,
                              int compute_capability_major)
{
  return specialized_key && num_inputs <= kDenseShapeMaxInputs &&
         compute_capability_major == kDenseShapeComputeCapabilityMajor;
}

tile_merge_policy tile_merge_policy_for(bool specialized_key,
                                        std::int32_t num_inputs,
                                        int compute_capability_major)
{
  tile_merge_policy policy{};
  if (dense_tile_shape_applies(specialized_key, num_inputs, compute_capability_major)) {
    policy.threads_per_sm    = 384;
    policy.max_block_threads = 128;
  }
  return policy;
}

cudf::size_type bounded_merge_tile_rows(const kway_key_comparator& cmp)
{
  const auto tuning = current_kway_tuning();
  if (tuning.tile_rows > 0) { return tuning.tile_rows; }
  if (tuning.baseline_shape != 0) { return kDefaultTileRows; }
  return dense_tile_shape_applies(
           cmp.specialized(), cmp.num_inputs(), current_device_traits().compute_capability_major)
           ? kDenseTileRows
           : kDefaultTileRows;
}

tile_merge_shape tile_merge_launch_shape(cudf::size_type tiles,
                                         cudf::size_type tile_rows,
                                         std::int32_t num_inputs,
                                         int sm_count,
                                         std::int32_t requested_sub_tiles,
                                         const tile_merge_policy& policy)
{
  std::int32_t padded_inputs = 2;
  while (padded_inputs < num_inputs) {
    padded_inputs *= 2;
  }
  sm_count   = std::max(sm_count, 1);
  auto shape = shape_within(
    kTileMergeSharedBytes, tiles, tile_rows, padded_inputs, sm_count, requested_sub_tiles, policy);
  // With an SM for every CTA no CTA waits for another, whatever its shared memory.
  if (policy.dense_shared_cap && shape.shared_bytes > kTileMergeDenseSharedBytes &&
      shape.blocks > static_cast<std::uint32_t>(sm_count)) {
    shape = shape_within(kTileMergeDenseSharedBytes,
                         tiles,
                         tile_rows,
                         padded_inputs,
                         sm_count,
                         requested_sub_tiles,
                         policy);
  }
  return shape;
}

namespace {

template <typename View>
void launch_tile_merge(View view,
                       const tile_merge_shape& shape,
                       const cudf::size_type* tile_coranks,
                       std::int32_t k,
                       std::int32_t padded_inputs,
                       cudf::size_type tiles,
                       cudf::size_type target,
                       cudf::size_type tile_rows,
                       std::int32_t* out_input,
                       cudf::size_type* out_row,
                       rmm::cuda_stream_view stream)
{
  tile_merge_kernel<View>
    <<<shape.blocks, static_cast<unsigned int>(shape.block_threads), shape.shared_bytes,
       stream.value()>>>(view,
                         tile_coranks,
                         k,
                         padded_inputs,
                         tiles,
                         target,
                         tile_rows,
                         shape.sub_tiles,
                         shape.sub_rows,
                         out_input,
                         out_row);
}

/// tile_merge with @p requested_sub_tiles sub-tiles per tile (0: chosen from the device); the
/// count is capped as in tile_merge_launch_shape.
void tile_merge_impl(const kway_key_comparator& cmp,
                     cudf::device_span<const cudf::size_type> lengths,
                     cudf::device_span<const cudf::size_type> tile_coranks,
                     cudf::size_type target,
                     cudf::size_type tile_rows,
                     cudf::device_span<std::int32_t> out_input,
                     cudf::device_span<cudf::size_type> out_row,
                     std::int32_t requested_sub_tiles,
                     rmm::cuda_stream_view stream)
{
  const std::int32_t k = cmp.num_inputs();
  if (static_cast<std::size_t>(k) > kMaxKwayMergeInputs) {
    throw std::invalid_argument("tile_merge: " + std::to_string(k) +
                                " inputs exceed the supported fan-in of " +
                                std::to_string(kMaxKwayMergeInputs));
  }
  if (lengths.size() != static_cast<std::size_t>(k)) {
    throw std::invalid_argument("tile_merge: lengths needs one entry per input");
  }
  if (tile_rows <= 0) { throw std::invalid_argument("tile_merge: tile_rows must be positive"); }
  if (target < 0) { throw std::invalid_argument("tile_merge: target must not be negative"); }
  if (target == 0) { return; }

  const cudf::size_type tiles = (target - 1) / tile_rows + 1;
  if (tile_coranks.size() < (static_cast<std::size_t>(tiles) + 1) * static_cast<std::size_t>(k)) {
    throw std::invalid_argument("tile_merge: tile_coranks needs (tiles + 1) * k entries");
  }
  if (out_input.size() < static_cast<std::size_t>(target) ||
      out_row.size() < static_cast<std::size_t>(target)) {
    throw std::invalid_argument("tile_merge: out_input and out_row need target entries");
  }

  std::int32_t padded_inputs = 2;
  while (padded_inputs < k) {
    padded_inputs *= 2;
  }
  const auto tuning = current_kway_tuning();
  const auto device = current_device_traits();
  auto policy       = tuning.baseline_shape != 0
                        ? tile_merge_policy{}
                        : tile_merge_policy_for(cmp.specialized(), k, device.compute_capability_major);
  // Measurement overrides of the policy.
  if (tuning.threads_per_sm > 0) { policy.threads_per_sm = tuning.threads_per_sm; }
  if (tuning.max_block_threads > 0) { policy.max_block_threads = tuning.max_block_threads; }
  if (tuning.min_sub_tile_rows > 0) { policy.min_sub_tile_rows = tuning.min_sub_tile_rows; }
  if (requested_sub_tiles == 0 && tuning.sub_tiles > 0) { requested_sub_tiles = tuning.sub_tiles; }
  const auto shape =
    tile_merge_launch_shape(tiles, tile_rows, k, device.sm_count, requested_sub_tiles, policy);

  with_key_view(cmp, [&](auto view) {
    launch_tile_merge(view,
                      shape,
                      tile_coranks.data(),
                      k,
                      padded_inputs,
                      tiles,
                      target,
                      tile_rows,
                      out_input.data(),
                      out_row.data(),
                      stream);
  });
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace

void tile_merge(const kway_key_comparator& cmp,
                cudf::device_span<const cudf::size_type> lengths,
                cudf::device_span<const cudf::size_type> tile_coranks,
                cudf::size_type target,
                cudf::size_type tile_rows,
                cudf::device_span<std::int32_t> out_input,
                cudf::device_span<cudf::size_type> out_row,
                rmm::cuda_stream_view stream)
{
  tile_merge_impl(cmp, lengths, tile_coranks, target, tile_rows, out_input, out_row, 0, stream);
}

void tile_merge_with_sub_tiles_for_test(const kway_key_comparator& cmp,
                                        cudf::device_span<const cudf::size_type> lengths,
                                        cudf::device_span<const cudf::size_type> tile_coranks,
                                        cudf::size_type target,
                                        cudf::size_type tile_rows,
                                        cudf::device_span<std::int32_t> out_input,
                                        cudf::device_span<cudf::size_type> out_row,
                                        std::int32_t sub_tiles,
                                        rmm::cuda_stream_view stream)
{
  tile_merge_impl(
    cmp, lengths, tile_coranks, target, tile_rows, out_input, out_row, sub_tiles, stream);
}

}  // namespace sirius::op::merge::detail
