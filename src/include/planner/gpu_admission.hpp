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

#pragma once

#include "helper/logical_type.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::planner {

/**
 * @brief Collect every GPU_SCAN operator reachable from @p node.
 *
 * Mirrors sirius_physical_plan_generator::set_parent_ops, the canonical full-tree descent.
 * RESULT_COLLECTOR holds its child in `plan` and DELIM JOIN holds `join`/`distinct_root`,
 * both outside `children[]`; missing either under-counts scans and admits too few GPUs.
 */
void collect_gpu_scans(const op::sirius_physical_operator& node,
                       std::vector<const op::sirius_physical_operator*>& out);

/**
 * @brief Narrow an active-GPU list to the first @p cap entries.
 *
 * @p cap is `topology.gpus_per_query`; 0, negative (rejected at config load), or a value
 * at or above the list size all mean "no cap", so asking for more GPUs than exist yields
 * every GPU rather than failing. Taking a prefix of the sorted list is stable and
 * reproducible but not NUMA-aware — a future allocator should prefer GPUs sharing a NUMA
 * domain over ids 0..k-1.
 */
inline std::vector<int> apply_gpu_cap(std::vector<int> gpu_ids, int cap)
{
  if (cap > 0 && static_cast<std::size_t>(cap) < gpu_ids.size()) {
    gpu_ids.resize(static_cast<std::size_t>(cap));
  }
  return gpu_ids;
}

/**
 * @brief Estimated GPU bytes per row for a projected column list.
 *
 * Fixed-width types defer to logical_type::fixed_width_byte_size(), which already tracks the
 * cuDF carrier each one lands in. Variable-width types have no static width and cost
 * @p avg_var_bytes.
 *
 * Reads *logical* types, so it misses the narrower physical carriers compressed
 * materialization can install, over-charging those columns and biasing admission upward.
 */
inline uint64_t estimate_bytes_per_row(const std::vector<sirius::logical_type>& types,
                                       uint64_t avg_var_bytes)
{
  constexpr auto k_max = std::numeric_limits<uint64_t>::max();
  uint64_t total       = 0;
  for (const auto& t : types) {
    auto const width = t.is_fixed_width() ? uint64_t{t.fixed_width_byte_size()} : avg_var_bytes;
    // A wide enough avg_var_bytes wraps the row width to something small, which reads as a
    // tiny query and admits it onto one GPU. Saturate, as the scan totals do.
    if (total > k_max - width) { return k_max; }
    total += width;
  }
  return total;
}

/**
 * @brief Add `rows * bytes_per_row` to @p running, saturating instead of wrapping.
 *
 * estimated_cardinality is a planner estimate with no bound, and a wrapped total would read
 * as a small query — admitting the largest onto one GPU. Saturate and let
 * gpu_count_for_bytes clamp to the full fleet instead.
 */
inline uint64_t accumulate_scan_bytes(uint64_t running, uint64_t rows, uint64_t bytes_per_row)
{
  constexpr auto k_max = std::numeric_limits<uint64_t>::max();
  if (rows == 0 || bytes_per_row == 0) { return running; }
  if (rows > k_max / bytes_per_row) { return k_max; }
  auto const scan_bytes = rows * bytes_per_row;
  if (running > k_max - scan_bytes) { return k_max; }
  return running + scan_bytes;
}

/// Row count and per-row width of one scan, read off the plan.
struct scan_estimate {
  uint64_t rows;
  uint64_t bytes_per_row;
};

/**
 * @brief Total projected scan-output bytes, or nullopt when the plan cannot be sized.
 *
 * Zero rows is ambiguous — provably empty, or simply unestimated — and the two want
 * opposite treatments. Sirius already reads it as "cannot size" (sirius_physical_sort_sample
 * gates on `estimated_cardinality > 0`), so follow that: one unsized scan, or no scans at
 * all, makes the whole estimate unavailable and admission keeps the full capped fleet
 * rather than sizing from partial information.
 */
inline std::optional<uint64_t> total_scan_bytes(const std::vector<scan_estimate>& scans)
{
  if (scans.empty()) { return std::nullopt; }
  uint64_t total = 0;
  for (auto const& s : scans) {
    if (s.rows == 0) { return std::nullopt; }
    total = accumulate_scan_bytes(total, s.rows, s.bytes_per_row);
  }
  return total;
}

/**
 * @brief Smallest GPU count that keeps @p total_bytes under @p bytes_per_gpu per GPU.
 *
 * Result is clamped to [1, @p n_gpus]: a query always gets at least one GPU and never
 * more than are available. Returns @p n_gpus when @p bytes_per_gpu is 0 (estimation
 * disabled) or when @p total_bytes is 0 (nothing to size against).
 */
inline int gpu_count_for_bytes(uint64_t total_bytes, uint64_t bytes_per_gpu, int n_gpus)
{
  if (bytes_per_gpu == 0 || total_bytes == 0 || n_gpus <= 1) { return std::max(n_gpus, 1); }
  // 1 + (x-1)/y rather than (x+y-1)/y: the latter wraps once x nears UINT64_MAX, and a
  // wrapped quotient narrowed to int can go negative and clamp down to a single GPU —
  // admitting the largest queries onto the least hardware. Clamp in 64-bit before narrowing.
  auto const required = 1ULL + (total_bytes - 1) / bytes_per_gpu;
  return static_cast<int>(std::min<uint64_t>(required, static_cast<uint64_t>(n_gpus)));
}

}  // namespace sirius::planner
