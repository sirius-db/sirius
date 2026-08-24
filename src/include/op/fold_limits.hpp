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
#include <stdexcept>
#include <string>

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
 *     folding side fits one fold, via `fold_partition_count`.
 *   - `gpu_merge_impl` refuses an over-limit fold via `check_fold_row_limit`, so the engine
 *     reports the violation instead of cuDF.
 *   - `sirius_physical_concat::execute` attributes that refusal to a specific CONCAT and join.
 *
 * The guard covers every fold `gpu_merge_impl` performs, which is every CONCAT fold and the two
 * aggregate merges. Only the CONCAT fold has a partition-count floor above it: MERGE_GROUP_BY
 * folds a whole partition too and sizes by bytes alone, so it is guarded but not floored
 * (residual R4, `docs/super-sirius/operators.md`).
 *
 * The functions are pure and free of engine state so they can be unit-pinned directly.
 */

namespace sirius::op {

/// Rows `cudf::concatenate` accepts in ONE table, for the widest column kind a fold may contain.
/// A library truth, not a tuning target.
///
/// cuDF applies two separate limits, and this is the smaller of the two. Rows are capped at
/// `cudf::size_type`'s maximum. Offsets are capped at the same maximum, but a variable-width
/// column carries one more offset than it has rows, so `cudf::concatenate` requires
/// `sum(rows) + 1 <= size_type::max` for every STRING and LIST column -- one row lower. Sirius
/// folds string columns routinely, and the offsets check is the one that fires first, so the
/// conservative constant is the correct one to plan against: it costs a single row and needs no
/// schema inspection. It also keeps INV-FOLD violations inside `check_fold_row_limit`, which
/// reports them attributably, instead of surfacing as a cuDF `std::overflow_error` that carries
/// no marker.
inline constexpr uint64_t k_fold_row_limit =
  static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()) - 1;

/// Per-partition row target used ONLY when a measured side has to be split: half of @p limit.
/// Splitting means hash-partitioning, which is not perfectly uniform, so a count computed against
/// the hard limit still overflows under moderate key skew; the halving is that skew margin and
/// nothing else. It must not be charged to a side that is not being split, where the fold size is
/// exactly the measurement and the skew term is identically zero -- see `fold_partition_count`.
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

/// Partitions a measured side of @p rows rows needs before each partition folds within @p limit.
///
/// One partition whenever the whole measurement already fits @p limit: that fold is the
/// measurement itself, so there is no distribution to be skewed and no margin to reserve. Only
/// once a split is genuinely required does the count get computed against `fold_row_target`, where
/// the skew margin is real.
///
/// The distinction is load-bearing, not cosmetic. Charging the margin at a count of one would
/// split -- or refuse outright -- sides between `fold_row_target(limit)` and @p limit rows, every
/// one of which cuDF addresses perfectly well in a single table. Splitting a build costs the
/// single-partition dynamic filter (see `sirius_physical_partition`'s `build_arrives_whole`); a
/// refusal costs the query.
[[nodiscard]] constexpr int fold_partition_count(uint64_t rows, uint64_t limit) noexcept
{
  if (rows <= limit) { return 1; }
  return fold_partition_floor(rows, fold_row_target(limit));
}

/// Thrown by `check_fold_row_limit`, and by nothing else. A fold can fail for reasons that have
/// nothing to do with the row limit -- device OOM from `cudf::concatenate`, a batch that is not
/// GPU-resident -- and a handler that cannot tell those apart cannot classify anything. Carrying a
/// distinct type is what lets `sirius_physical_concat::execute` stamp the `[fold_limit]` marker on
/// exactly the INV-FOLD violations.
class fold_row_limit_exceeded : public std::runtime_error {
 public:
  explicit fold_row_limit_exceeded(const std::string& what) : std::runtime_error(what) {}
};

/**
 * @brief Enforce INV-FOLD on a group about to be folded into one cuDF table.
 *
 * @param total_rows Rows summed over every batch in the group.
 * @param num_batches Batches in the group, reported so a log reader can tell an over-limit fold
 *                    of many small batches from one of a few huge ones.
 * @param limit Largest addressable row count, normally `k_fold_row_limit`.
 * @throws fold_row_limit_exceeded when @p total_rows exceeds @p limit. The message carries the
 *         stable marker `[fold_limit]`, so log analysis can classify the failure without matching
 *         on cuDF's internal wording.
 */
void check_fold_row_limit(uint64_t total_rows, std::size_t num_batches, uint64_t limit);

}  // namespace sirius::op
