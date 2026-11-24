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

#include "op/merge/detail/kway_comparator.cuh"
#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/reduce.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::op::merge::detail {

namespace {

constexpr int kWarpSize = 32;

/// lo + the number of rows in [lo, hi) of window @p input that order before row @p pivot_row of
/// window @p pivot_input. The rows of a window are sorted, so those rows are a prefix of the range.
template <typename View>
__device__ cudf::size_type lower_bound(const View& view,
                                       int32_t input,
                                       cudf::size_type lo,
                                       cudf::size_type hi,
                                       int32_t pivot_input,
                                       cudf::size_type pivot_row) noexcept
{
  while (lo < hi) {
    const cudf::size_type mid = lo + (hi - lo) / 2;
    if (view.less(input, mid, pivot_input, pivot_row)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * Multi-sequence selection, one warp (one block of 32 threads) per rank. Lane l owns the inputs
 * l, l + 32, ...
 *
 * Invariant: lo[j] <= c_j <= hi[j], where c_j is the answer. Each iteration takes the middle row
 * of the widest remaining range as the pivot, counts in every input the rows that order before
 * the pivot (one binary search per input, restricted to [lo[j], hi[j])) and compares the sum R
 * with the rank:
 * - R < rank: the pivot and every row before it are selected, so lo[j] rises to the count;
 * - R > rank: the pivot and every row after it are unselected, so hi[j] drops to the count;
 * - R == rank: the counts are the answer.
 * Restricting the searches keeps the three cases exact: a count clamped to [lo[j], hi[j]] stays
 * on the same side of c_j as the true count.
 *
 * The pivot's range at least halves each iteration, so there are at most k * log2(n) iterations
 * of k binary searches each (32 at a time); on typical data the other ranges shrink with it and
 * the iteration count is close to log2(k * n).
 *
 * lo lives in the output; hi and count are scratch arrays with the same layout.
 */
template <typename View>
__global__ void corank_kernel(View view,
                              const cudf::size_type* lengths,
                              const cudf::size_type* ranks,
                              cudf::size_type* out,
                              cudf::size_type* hi_scratch,
                              cudf::size_type* count_scratch)
{
  __shared__ cudf::size_type s_width[kWarpSize];
  __shared__ int32_t s_input[kWarpSize];
  __shared__ cudf::size_type s_row[kWarpSize];
  __shared__ int64_t s_lo[kWarpSize];
  __shared__ int64_t s_hi[kWarpSize];
  __shared__ int64_t s_count[kWarpSize];

  const int lane             = static_cast<int>(threadIdx.x);
  const int32_t k            = view.num_inputs;
  const int64_t rank         = ranks[blockIdx.x];
  const std::size_t offset   = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(k);
  cudf::size_type* const lo  = out + offset;
  cudf::size_type* const hi  = hi_scratch + offset;
  cudf::size_type* const cnt = count_scratch + offset;

  for (int32_t j = lane; j < k; j += kWarpSize) {
    lo[j] = 0;
    hi[j] = lengths[j];
  }

  while (true) {
    // Widest remaining range and the sums of the bounds.
    cudf::size_type width = 0;
    int32_t pivot_input   = -1;
    cudf::size_type pivot = 0;
    int64_t sum_lo        = 0;
    int64_t sum_hi        = 0;
    for (int32_t j = lane; j < k; j += kWarpSize) {
      const cudf::size_type w = hi[j] - lo[j];
      if (w > width) {
        width       = w;
        pivot_input = j;
        pivot       = lo[j] + w / 2;
      }
      sum_lo += lo[j];
      sum_hi += hi[j];
    }
    s_width[lane] = width;
    s_input[lane] = pivot_input;
    s_row[lane]   = pivot;
    s_lo[lane]    = sum_lo;
    s_hi[lane]    = sum_hi;
    __syncthreads();
    width  = 0;
    sum_lo = 0;
    sum_hi = 0;
    for (int l = 0; l < kWarpSize; ++l) {
      if (s_width[l] > width) {
        width       = s_width[l];
        pivot_input = s_input[l];
        pivot       = s_row[l];
      }
      sum_lo += s_lo[l];
      sum_hi += s_hi[l];
    }
    __syncthreads();

    // Every lane computed the same sums, so the whole warp leaves the loop together.
    // width == 0 only with sum_lo == rank, unless the rank is out of range (> Σ lengths).
    if (sum_lo == rank || width == 0) { break; }
    if (sum_hi == rank) {
      for (int32_t j = lane; j < k; j += kWarpSize) {
        lo[j] = hi[j];
      }
      break;
    }

    int64_t before = 0;
    for (int32_t j = lane; j < k; j += kWarpSize) {
      cnt[j] = j == pivot_input ? pivot : lower_bound(view, j, lo[j], hi[j], pivot_input, pivot);
      before += cnt[j];
    }
    s_count[lane] = before;
    __syncthreads();
    before = 0;
    for (int l = 0; l < kWarpSize; ++l) {
      before += s_count[l];
    }
    __syncthreads();

    if (before == rank) {
      for (int32_t j = lane; j < k; j += kWarpSize) {
        lo[j] = cnt[j];
      }
      break;
    }
    if (before < rank) {
      for (int32_t j = lane; j < k; j += kWarpSize) {
        lo[j] = j == pivot_input ? pivot + 1 : cnt[j];
      }
    } else {
      for (int32_t j = lane; j < k; j += kWarpSize) {
        hi[j] = cnt[j];
      }
    }
  }
}

/**
 * The same selection for a fan-in of at most 32, one warp (one block of 32 threads) per rank:
 * lane l owns input l and keeps its bounds in registers, and the warp agrees on the pivot and on
 * the sums with shuffles, so an iteration touches no memory but the keys it compares. The lanes
 * past the fan-in own an empty range. The answer is the co-rank vector of the rank, which is
 * unique, so it does not depend on the pivots chosen.
 */
template <typename View>
__global__ void corank_warp_kernel(View view,
                                   const cudf::size_type* lengths,
                                   const cudf::size_type* ranks,
                                   cudf::size_type* out)
{
  constexpr unsigned kAllLanes = 0xFFFF'FFFFu;
  const int lane               = static_cast<int>(threadIdx.x);
  const int32_t k              = view.num_inputs;
  const int64_t rank           = ranks[blockIdx.x];
  const bool owner             = lane < k;
  cudf::size_type lo           = 0;
  cudf::size_type hi           = owner ? lengths[lane] : 0;

  while (true) {
    // The widest range (the lowest lane among equals) and the sums of the bounds.
    std::uint64_t widest =
      (static_cast<std::uint64_t>(hi - lo) << 32) | static_cast<std::uint32_t>(kWarpSize - 1 - lane);
    int64_t sum_lo = lo;
    int64_t sum_hi = hi;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      const std::uint64_t other = __shfl_xor_sync(kAllLanes, widest, offset);
      widest                    = other > widest ? other : widest;
      sum_lo += __shfl_xor_sync(kAllLanes, sum_lo, offset);
      sum_hi += __shfl_xor_sync(kAllLanes, sum_hi, offset);
    }
    const auto width        = static_cast<cudf::size_type>(widest >> 32);
    const int32_t pivot_input =
      kWarpSize - 1 - static_cast<int32_t>(widest & static_cast<std::uint64_t>(kWarpSize - 1));

    // Every lane holds the same sums, so the whole warp leaves the loop together.
    // width == 0 only with sum_lo == rank, unless the rank is out of range (> Σ lengths).
    if (sum_lo == rank || width == 0) { break; }
    if (sum_hi == rank) {
      lo = hi;
      break;
    }

    const cudf::size_type pivot = __shfl_sync(kAllLanes, lo + (hi - lo) / 2, pivot_input);
    const cudf::size_type count =
      !owner ? 0 : lane == pivot_input ? pivot : lower_bound(view, lane, lo, hi, pivot_input, pivot);
    int64_t before = count;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      before += __shfl_xor_sync(kAllLanes, before, offset);
    }

    if (before == rank) {
      lo = count;
      break;
    }
    if (before < rank) {
      lo = lane == pivot_input ? pivot + 1 : count;
    } else {
      hi = count;
    }
  }
  if (owner) { out[static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(k) + lane] = lo; }
}

/// The two device-side terms of the safe target.
struct target_bounds {
  int64_t r_safe;  ///< min over truncated windows of (rank of the window's last row + 1)
  int64_t total;   ///< Σ lengths
};

/// One warp (one block of 32 threads) per window i. Lane l counts, in the inputs l, l + 32, ...,
/// the rows that order before the last row of window i; the warp adds the counts up. A window
/// that is not truncated does not bound the target.
template <typename View>
__global__ void window_bounds_kernel(View view,
                                     const cudf::size_type* lengths,
                                     const std::uint8_t* truncated,
                                     target_bounds* out)
{
  const int lane            = static_cast<int>(threadIdx.x);
  const auto i              = static_cast<int32_t>(blockIdx.x);
  const cudf::size_type len = lengths[i];
  // Both early exits depend on the window only, so the whole warp takes them together.
  if (truncated[i] == 0) {
    if (lane == 0) { out[i] = {std::numeric_limits<int64_t>::max(), len}; }
    return;
  }
  // Nothing can be emitted before a truncated window with no rows is refilled.
  if (len == 0) {
    if (lane == 0) { out[i] = {0, 0}; }
    return;
  }
  int64_t before = 0;
  for (int32_t j = lane; j < view.num_inputs; j += kWarpSize) {
    if (j != i) { before += lower_bound(view, j, 0, lengths[j], i, len - 1); }
  }
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    before += __shfl_down_sync(0xFFFF'FFFFu, before, offset);
  }
  // The rank of the last row is (len - 1) + before.
  if (lane == 0) { out[i] = {before + len, len}; }
}

struct combine_bounds_fn {
  __device__ target_bounds operator()(const target_bounds& a, const target_bounds& b) const noexcept
  {
    return {a.r_safe < b.r_safe ? a.r_safe : b.r_safe, a.total + b.total};
  }
};

void check_lengths(const kway_key_comparator& cmp,
                   cudf::device_span<const cudf::size_type> lengths,
                   const char* what)
{
  if (lengths.size() != static_cast<std::size_t>(cmp.num_inputs())) {
    throw std::invalid_argument(std::string(what) + ": one length per key window is required");
  }
}

template <typename View>
void launch_corank(View view,
                   cudf::device_span<const cudf::size_type> lengths,
                   cudf::device_span<const cudf::size_type> ranks,
                   cudf::device_span<cudf::size_type> out,
                   bool warp_kernel,
                   rmm::cuda_stream_view stream)
{
  const auto blocks = static_cast<unsigned>(ranks.size());
  if (warp_kernel) {
    corank_warp_kernel<View>
      <<<blocks, kWarpSize, 0, stream.value()>>>(view, lengths.data(), ranks.data(), out.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    return;
  }
  // hi and count of every (rank, input), laid out like out.
  rmm::device_uvector<cudf::size_type> scratch(
    2 * out.size(), stream, cudf::get_current_device_resource_ref());
  corank_kernel<View>
    <<<blocks, kWarpSize, 0, stream.value()>>>(
      view, lengths.data(), ranks.data(), out.data(), scratch.data(), scratch.data() + out.size());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

template <typename View>
void launch_window_bounds(View view,
                          cudf::device_span<const cudf::size_type> lengths,
                          cudf::device_span<const std::uint8_t> truncated,
                          target_bounds* out,
                          rmm::cuda_stream_view stream)
{
  window_bounds_kernel<View>
    <<<static_cast<unsigned>(lengths.size()), kWarpSize, 0, stream.value()>>>(
      view, lengths.data(), truncated.data(), out);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace

void corank_split(const kway_key_comparator& cmp,
                  cudf::device_span<const cudf::size_type> lengths,
                  cudf::device_span<const cudf::size_type> ranks,
                  cudf::device_span<cudf::size_type> out,
                  rmm::cuda_stream_view stream)
{
  check_lengths(cmp, lengths, "corank_split");
  if (out.size() != ranks.size() * lengths.size()) {
    throw std::invalid_argument("corank_split: out needs ranks.size() * k entries");
  }
  if (ranks.empty()) { return; }
  if (ranks.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("corank_split: too many ranks");
  }

  // The warp kernel has one lane per input. It was measured with specialised comparators; the
  // generic comparator keeps the general kernel unless the switch asks for it.
  const auto warp_switch = current_kway_tuning().warp_corank;
  const bool warp_kernel = lengths.size() <= static_cast<std::size_t>(kWarpSize) &&
                           (warp_switch >= 2 || (warp_switch == 1 && cmp.specialized()));
  with_key_view(
    cmp, [&](auto view) { launch_corank(view, lengths, ranks, out, warp_kernel, stream); });
}

// A truncated window with no rows forces the target to 0: the next row of its run is unknown, so
// no row of any window is safe to emit. The caller must refill such a window before the round.
cudf::size_type safe_target(const kway_key_comparator& cmp,
                            cudf::device_span<const cudf::size_type> lengths,
                            cudf::device_span<const std::uint8_t> truncated,
                            cudf::size_type max_rows,
                            rmm::cuda_stream_view stream)
{
  check_lengths(cmp, lengths, "safe_target");
  if (truncated.size() != lengths.size()) {
    throw std::invalid_argument("safe_target: one truncated flag per key window is required");
  }

  // One warp per window, one binary search per (truncated window, other window) pair. The
  // result of the reduction is the only device-to-host copy.
  const auto temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<target_bounds> window_bounds(lengths.size(), stream, temp_mr);
  with_key_view(cmp, [&](auto view) {
    launch_window_bounds(view, lengths, truncated, window_bounds.data(), stream);
  });
  const auto bounds = thrust::reduce(rmm::exec_policy_nosync(stream, temp_mr),
                                     window_bounds.begin(),
                                     window_bounds.end(),
                                     target_bounds{std::numeric_limits<int64_t>::max(), 0},
                                     combine_bounds_fn{});
  return static_cast<cudf::size_type>(
    std::min({static_cast<int64_t>(max_rows), bounds.r_safe, bounds.total}));
}

}  // namespace sirius::op::merge::detail
