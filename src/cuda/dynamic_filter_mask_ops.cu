/*
 * Copyright 2026, Sirius Contributors.
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

#include <rmm/error.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/dynamic_filter_fused_probe.cuh>
#include <cuda_runtime.h>

#include <helper/utils.hpp>
#include <op/dynamic_filter/dynamic_filter_mask_ops.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <telemetry/nvtx.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace sirius::op {
namespace {
constexpr int block_size                   = 256;
constexpr std::size_t fused_round_capacity = fused_membership_round_capacity;
constexpr std::size_t fused_block_limit    = 4096;

/// @brief A round of fused membership probes. Each round can contain up to `fused_round_capacity`
/// probes. For `fused_round_capacity` + 1 filters:
///  - Round 1: residual -> filter 1 -> filter 2 -> filter 3 -> filter 4
///  - Round 2: previous running mask -> filter 5
struct fused_round {
  std::array<detail::fused_probe, fused_round_capacity>
    probes;              ///< Array of filter descriptions for one round
  bool* running;         ///< Running mask of combined probe results
  bool const* residual;  ///< Optional residual mask from remaining query predicate
  detail::probe_validity
    residual_valid;  ///< Null-validity information for `residual`, including offset
  cudf::size_type rows;
  int active_steps;  ///< Number of active probes in this round, up to `fused_round_capacity`
  cudf::size_type* residual_count;  ///< Optional GPU pointer to count of rows that pass the
                                    ///< residual before membership probes. Supplied only for the
                                    ///< first round when a residual exists and counting is enabled
  cudf::size_type*
    step_counts;    ///< Optional GPU pointer to this round's counters -- one per active filter
  bool initialize;  ///< Whether this is the first round: if true, start from `residual` or
                    ///< all-true, otherwise start from `running`
};

static_assert(std::is_trivially_copyable_v<fused_round>);
// Kernel arguments are limited to 4 KiB.
static_assert(sizeof(fused_round) <= 4096);

using block_reduction = cub::BlockReduce<cudf::size_type, block_size>;

/// @brief Whether @p row survives an optional BOOL8 mask: true without a mask, otherwise only when
/// the row is valid and set. A null row is rejected.
__device__ __forceinline__ bool mask_keeps(bool const* mask,
                                           detail::probe_validity valid,
                                           cudf::size_type row) noexcept
{
  return !mask || (valid(row) && mask[row]);
}

/// @brief Adds the block's sum of @p value to @p counter. Every thread of the block must call it;
/// it synchronizes before returning so the caller can reuse @p scratch.
__device__ __forceinline__ void add_block_count(block_reduction::TempStorage& scratch,
                                                cudf::size_type value,
                                                cudf::size_type* counter)
{
  auto const total = block_reduction(scratch).Sum(value);
  if (threadIdx.x == 0 && total != 0) { atomicAdd(counter, total); }
  __syncthreads();
}

/// @brief Fused membership probe kernel.
/// @note `__grid_constant__` lets the probe lookups take references into the launch argument
/// without the compiler copying the whole round into per-thread local memory.
__global__ void fused_membership_mask_kernel(__grid_constant__ fused_round const args)
{
  __shared__ block_reduction::TempStorage scratch;
  cudf::size_type local[fused_round_capacity + 1]{};
  auto const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < static_cast<std::size_t>(args.rows);
       row += stride) {
    auto const index = static_cast<cudf::size_type>(row);
    bool keep =
      args.initialize ? mask_keeps(args.residual, args.residual_valid, index) : args.running[row];
    if (args.residual_count) { local[0] += keep; }
#pragma unroll
    for (int step = 0; step < static_cast<int>(fused_round_capacity); ++step) {
      if (step < args.active_steps) {
        if (keep) { keep = detail::fused_probe_keeps(args.probes[step], index); }
        if (args.step_counts) { local[step + 1] += keep; }
      }
    }
    args.running[row] = keep;
  }
  if (!args.step_counts) { return; }
  if (args.residual_count) { add_block_count(scratch, local[0], args.residual_count); }
#pragma unroll
  for (int step = 0; step < static_cast<int>(fused_round_capacity); ++step) {
    if (step < args.active_steps) {
      add_block_count(scratch, local[step + 1], args.step_counts + step);
    }
  }
}

/// @brief Folds an incoming mask into a running mask, optionally counting the number of rows
/// that pass the combined mask. If @p initialize is true, the running mask is initialized from the
/// incoming mask, otherwise it is combined with the incoming mask.
__global__ void fold_mask_kernel(bool* running,
                                 bool const* incoming,
                                 cudf::bitmask_type const* validity,
                                 cudf::size_type offset,
                                 cudf::size_type rows,
                                 cudf::size_type* count,
                                 bool initialize)
{
  __shared__ block_reduction::TempStorage scratch;
  auto const row       = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  cudf::size_type kept = 0;
  if (row < static_cast<std::size_t>(rows)) {
    auto const index = static_cast<cudf::size_type>(row);
    bool const value = (initialize || running[row]) &&
                       mask_keeps(incoming, detail::probe_validity{validity, offset}, index);
    running[row] = value;
    kept         = value;
  }
  if (count != nullptr) { add_block_count(scratch, kept, count); }
}
}  // namespace

void try_fused_membership_mask(std::span<fused_membership_input const> steps,
                               int device_id,
                               cudf::mutable_column_view running,
                               std::optional<cudf::column_view> residual,
                               cudf::device_span<cudf::size_type> counts,
                               std::size_t& submitted_rounds,
                               ::cuda::stream_ref stream)
{
  nvtx_scoped_range range{"dynfilter::try_fused_membership_mask"};
  submitted_rounds = 0;
  auto const rows  = running.size();
  if (running.type().id() != cudf::type_id::BOOL8 || running.nullable() ||
      (residual && (residual->type().id() != cudf::type_id::BOOL8 || residual->size() != rows)) ||
      (!counts.empty() &&
       counts.size() != steps.size() + static_cast<std::size_t>(residual.has_value()))) {
    throw std::invalid_argument("invalid fused membership buffers");
  }
  for (auto const& step : steps) {
    if (!step.filter || step.probe.size() != rows) {
      throw std::invalid_argument("invalid fused membership input");
    }
  }
  std::vector<detail::fused_probe> probes(steps.size());
  for (std::size_t i = 0; i < steps.size(); ++i) {
    if (!steps[i].filter->prepare_fused_probe(steps[i].probe, device_id, probes[i])) { return; }
  }
  if (rows == 0 || steps.empty()) { return; }

  auto const blocks =
    std::min(utils::ceil_div(static_cast<std::size_t>(rows), static_cast<std::size_t>(block_size)),
             fused_block_limit);
  for (std::size_t first = 0; first < probes.size(); first += fused_round_capacity) {
    auto const size = std::min(fused_round_capacity, probes.size() - first);
    fused_round args{
      .running        = running.data<bool>(),
      .residual       = first == 0 && residual ? residual->data<bool>() : nullptr,
      .residual_valid = first == 0 && residual
                          ? detail::probe_validity{residual->null_mask(), residual->offset()}
                          : detail::probe_validity{},
      .rows           = rows,
      .active_steps   = static_cast<int>(size),
      .residual_count = !counts.empty() && first == 0 && residual ? counts.data() : nullptr,
      .step_counts    = counts.empty()
                          ? nullptr
                          : counts.data() + first + static_cast<std::size_t>(residual.has_value()),
      .initialize     = first == 0};
    std::copy_n(probes.begin() + static_cast<std::ptrdiff_t>(first), size, args.probes.data());
    fused_membership_mask_kernel<<<blocks, block_size, 0, stream.get()>>>(args);
    RMM_CUDA_TRY(cudaPeekAtLastError());
    ++submitted_rounds;
  }
}

void fold_dynamic_filter_mask(cudf::mutable_column_view running,
                              std::optional<cudf::column_view> incoming,
                              cudf::size_type* count,
                              bool initialize,
                              ::cuda::stream_ref stream)
{
  if (running.type().id() != cudf::type_id::BOOL8 || running.nullable() ||
      (incoming &&
       (incoming->type().id() != cudf::type_id::BOOL8 || incoming->size() != running.size()))) {
    throw std::invalid_argument("invalid dynamic-filter fold buffers");
  }
  if (running.size() == 0) { return; }
  auto const blocks =
    utils::ceil_div(static_cast<std::size_t>(running.size()), static_cast<std::size_t>(block_size));
  fold_mask_kernel<<<blocks, block_size, 0, stream.get()>>>(
    running.data<bool>(),
    incoming ? incoming->data<bool>() : nullptr,
    incoming ? incoming->null_mask() : nullptr,
    incoming ? incoming->offset() : 0,
    running.size(),
    count,
    initialize);
  RMM_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace sirius::op
