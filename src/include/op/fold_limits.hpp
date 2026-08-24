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

#pragma once

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

/**
 * @file fold_limits.hpp
 * @brief The row limit on a single folded cuDF table, and the arithmetic three layers share.
 *
 * A "fold" is what `sirius_physical_concat` does when it merges a whole partition into one batch:
 * `gpu_merge_impl::concat` calls `cudf::concatenate`, which produces exactly one `cudf::table`.
 * This file states the invariant that makes such a fold legal:
 *
 * > **INV-FOLD.** Every batch group a CONCAT forms holds at most `k_fold_row_limit` rows.
 *
 * Three places cooperate to keep it, and they all read the definitions here so they cannot drift
 * apart:
 *   - `compute_hash_join_partition_strategy` raises the partition count until the measured
 *     folding side fits `fold_row_target` rows per partition, via `fold_partition_floor`.
 *   - `gpu_merge_impl::concat` refuses an over-limit fold via `check_fold_row_limit`, so the
 *     engine reports the violation instead of cuDF.
 *   - `sirius_physical_concat::execute` attributes that refusal to a specific CONCAT and join.
 *
 * The functions are pure and free of engine state so they can be unit-pinned directly.
 */

namespace sirius::op {

/// Rows cuDF can address in ONE table: `cudf::size_type` is `int32_t`, so `cudf::concatenate`
/// throws "Total number of concatenated rows exceeds the column size limit" beyond this. A
/// hardware/library truth, not a tuning target.
inline constexpr uint64_t k_fold_row_limit =
  static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max());

/// Planning target for one folded partition: half of @p limit. Hash partitioning is not perfectly
/// uniform, so a partition count chosen at the hard limit still overflows under moderate key skew.
/// Mirrors `k_max_distinct_count_rows` in `sirius_physical_hash_join.cpp`, which halves the same
/// limit for the same reason.
[[nodiscard]] constexpr uint64_t fold_row_target(uint64_t limit) noexcept
{
  return limit < 2 ? 1 : limit / 2;
}

/// Smallest partition count that keeps `ceil(rows / count)` at or under @p target. Returns 1 when
/// @p rows already fits, and treats a zero @p target as 1 rather than dividing by zero. Saturates
/// at `INT_MAX`, which is unreachable in practice: it needs more than 2^62 rows.
[[nodiscard]] constexpr int fold_partition_floor(uint64_t rows, uint64_t target) noexcept
{
  uint64_t const divisor = target == 0 ? 1 : target;
  uint64_t const count   = rows / divisor + static_cast<uint64_t>(rows % divisor != 0);
  if (count <= 1) { return 1; }
  constexpr auto int_max = static_cast<uint64_t>(std::numeric_limits<int>::max());
  return static_cast<int>(count < int_max ? count : int_max);
}

/**
 * @brief Enforce INV-FOLD on a group about to be folded into one cuDF table.
 *
 * @param total_rows Rows summed over every batch in the group.
 * @param num_batches Batches in the group, reported so a log reader can tell an over-limit fold
 *                    of many small batches from one of a few huge ones.
 * @param limit Largest addressable row count, normally `k_fold_row_limit`.
 * @throws std::runtime_error when @p total_rows exceeds @p limit. The message carries the stable
 *         marker `[fold_limit]`, so log analysis can classify the failure without matching on
 *         cuDF's internal wording.
 */
void check_fold_row_limit(uint64_t total_rows, std::size_t num_batches, uint64_t limit);

}  // namespace sirius::op
