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

// Tile merge: every output tile is merged independently by one thread that owns a tournament
// tree in shared memory.
//
// Design: a tile's merge is sequential (each output row depends on the previous winner), so the
// parallelism comes from the tiles. A CTA holds as many tiles as fit its shared-memory budget, one
// thread per tile, so the lanes of a warp stay busy: every thread runs the same loop (one output
// row, then one replay of log2(k) comparisons) on its own tile. Per tile the shared state is
// - head[i]: the next unmerged row of input i (kp 32-bit words), and
// - tree[n]: the input that wins the subtree of node n (kp 16-bit words, node 0 unused),
// where kp is k rounded up to a power of two (at least 2). Leaves past k and inputs whose
// sub-range is used up lose every comparison. Each output row costs log2(kp) calls of
// kway_key_device_view::less, whose tie-break (input, then row) makes the merged order unique.
//
// Limit: k <= kTileMergeMaxInputs (4096). At that fan-in one tile's state is 24 KiB, which still
// fits the default 48 KiB of shared memory of a CTA; tile_merge throws std::invalid_argument for a
// larger k.

#include "op/merge/detail/kway_comparator.cuh"
#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::op::merge::detail {

namespace {

/// Largest fan-in tile_merge accepts.
constexpr std::int32_t kTileMergeMaxInputs = 4096;
/// Shared memory a CTA may use for the tiles it merges.
constexpr std::size_t kTileMergeSharedBytes = 32 * 1024;
/// Most tiles (threads) per CTA.
constexpr std::int32_t kTileMergeMaxBlockThreads = 32;

/// Shared-memory bytes of one tile: head (32-bit) and tree (16-bit) entries for kp leaves.
constexpr std::size_t tile_state_bytes(std::int32_t padded_inputs)
{
  return static_cast<std::size_t>(padded_inputs) * (sizeof(cudf::size_type) + sizeof(std::int16_t));
}

static_assert(tile_state_bytes(kTileMergeMaxInputs) <= kTileMergeSharedBytes);
static_assert(kTileMergeMaxInputs <= INT16_MAX);

/// One thread per tile; thread `threadIdx.x` of block `blockIdx.x` merges tile
/// `blockIdx.x * blockDim.x + threadIdx.x`.
__global__ void tile_merge_kernel(kway_key_device_view keys,
                                  const cudf::size_type* tile_coranks,
                                  std::int32_t num_inputs,
                                  std::int32_t padded_inputs,
                                  cudf::size_type tiles,
                                  cudf::size_type target,
                                  cudf::size_type tile_rows,
                                  std::int32_t* out_input,
                                  cudf::size_type* out_row)
{
  // The heads of every tile of the block come first, then the trees, so both stay aligned.
  extern __shared__ cudf::size_type shared_state[];

  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile >= tiles) { return; }

  const std::int32_t k  = num_inputs;
  const std::int32_t kp = padded_inputs;
  cudf::size_type* head = shared_state + static_cast<std::size_t>(threadIdx.x) * kp;
  std::int16_t* tree    = reinterpret_cast<std::int16_t*>(shared_state + blockDim.x * kp) +
                       static_cast<std::size_t>(threadIdx.x) * kp;
  const cudf::size_type* begin = tile_coranks + tile * k;
  const cudf::size_type* end   = begin + k;

  for (std::int32_t i = 0; i < k; ++i) {
    head[i] = begin[i];
  }

  auto alive = [&](std::int32_t i) { return i < k && head[i] < end[i]; };
  // True when the next row of input a is emitted before the next row of input b (a != b).
  auto wins = [&](std::int32_t a, std::int32_t b) {
    if (!alive(b)) { return true; }
    if (!alive(a)) { return false; }
    return keys.less(a, head[a], b, head[b]);
  };
  // Winner of the subtree of child node c; nodes kp .. 2kp-1 are the leaves.
  auto winner_of = [&](std::int32_t c) {
    return c >= kp ? c - kp : static_cast<std::int32_t>(tree[c]);
  };
  auto play = [&](std::int32_t node) {
    const std::int32_t left  = winner_of(2 * node);
    const std::int32_t right = winner_of(2 * node + 1);
    tree[node]               = static_cast<std::int16_t>(wins(left, right) ? left : right);
  };

  for (std::int32_t node = kp - 1; node >= 1; --node) {
    play(node);
  }

  const cudf::size_type out_begin = static_cast<cudf::size_type>(tile) * tile_rows;
  const cudf::size_type out_end   = out_begin + min(tile_rows, target - out_begin);
  for (cudf::size_type out = out_begin; out < out_end; ++out) {
    const std::int32_t winner = tree[1];
    // Only when the co-ranks do not cover the tile; never read past a sub-range.
    if (!alive(winner)) { break; }
    out_input[out] = winner;
    out_row[out]   = head[winner]++;
    for (std::int32_t node = (winner + kp) >> 1; node >= 1; node >>= 1) {
      play(node);
    }
  }
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
  const std::int32_t k = cmp.num_inputs();
  if (k > kTileMergeMaxInputs) {
    throw std::invalid_argument("tile_merge: " + std::to_string(k) +
                                " inputs exceed the supported fan-in of " +
                                std::to_string(kTileMergeMaxInputs));
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
  const std::size_t state_bytes = tile_state_bytes(padded_inputs);
  const auto block_threads      = static_cast<std::int32_t>(
    std::clamp<std::size_t>(kTileMergeSharedBytes / state_bytes, 1, kTileMergeMaxBlockThreads));
  const auto blocks = static_cast<unsigned int>((tiles - 1) / block_threads + 1);

  tile_merge_kernel<<<blocks,
                      static_cast<unsigned int>(block_threads),
                      state_bytes* static_cast<std::size_t>(block_threads),
                      stream.value()>>>(device_view_of(cmp),
                                        tile_coranks.data(),
                                        k,
                                        padded_inputs,
                                        tiles,
                                        target,
                                        tile_rows,
                                        out_input.data(),
                                        out_row.data());
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace sirius::op::merge::detail
