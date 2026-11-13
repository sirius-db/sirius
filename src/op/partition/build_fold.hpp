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

/**
 * @file
 * @brief The build-fold rule: whether a hash join's single-partition build becomes one batch.
 *
 * A STANDARD or MIXED_JOIN hash join builds one hash table per (probe batch, build batch) pair
 * task. Folding the build, that is switching its build `sirius_physical_concat` to `concat_all`,
 * keeps the total hash-build work the same but divides the number of probe passes and pair tasks by
 * the number of build batches. In exchange, every task's hash table covers the complete build, the
 * CONCAT merge holds the build twice for a moment, and the first pair task waits for the whole
 * build pipeline.
 *
 * `bind_partition_inputs` asks `decide_build_fold` once the build side has sized the join to a
 * single partition without choosing BUILD_PROBE. BUILD_PROBE folds, and join types whose build
 * CONCAT is `concat_all` from planning (`join_concat_all`), do not depend on this rule.
 */

#pragma once

#include "op/partition/config.hpp"

#include <cstdint>
#include <optional>

namespace sirius::op {

enum class HASH_JOIN_MODE;  // Defined in op/sirius_physical_hash_join.hpp.

/**
 * @brief Measurements of a complete single-partition build, taken when the join is sized.
 */
struct build_fold_inputs {
  /** @brief Exact build rows; empty when some build batch records no row count. */
  std::optional<std::uint64_t> rows;
  /** @brief Decoded build bytes, which a fold keeps resident as one batch. */
  std::uint64_t resident_bytes;
  /** @brief Group-size target of the build CONCAT when it does not fold. */
  std::uint64_t concat_batch_bytes;
  /**
   * @brief GPUs admitted to the join. No policy reads it yet; it lets a multi-GPU limit restrict
   * folding without changing this interface.
   */
  int num_gpus;
  /** @brief `HASH_JOIN_MODE::STANDARD` or `HASH_JOIN_MODE::MIXED_JOIN`. No policy reads it yet. */
  HASH_JOIN_MODE mode;
};

/**
 * @brief Byte limits on a folded build.
 */
struct build_fold_limits {
  /** @brief The join's `max_build_probe_resident_bytes`, which also bounds BUILD_PROBE folds. */
  std::uint64_t max_resident_bytes;
};

/**
 * @brief Decides whether a single-partition STANDARD or MIXED_JOIN build folds into one batch.
 *
 * `build_fold_policy::bounded` folds if and only if all of these hold:
 * - `inputs.rows` is known and at most the cuDF row limit (`INT32_MAX`);
 * - `inputs.resident_bytes < limits.max_resident_bytes`;
 * - the estimated hash table, `inputs.rows` times `config::BUILD_PROBE_HASH_TABLE_BYTES_PER_ROW`,
 * is at most `inputs.resident_bytes`, so it fits the reservation that the folded input raises; a
 * single-partition build is at most `hash_partition_bytes`, which therefore caps the table too;
 * - `inputs.resident_bytes > inputs.concat_batch_bytes`, because a build at or under the CONCAT
 * target is already one group.
 *
 * A build narrower than `config::BUILD_PROBE_HASH_TABLE_BYTES_PER_ROW` bytes per row therefore
 * never folds under `build_fold_policy::bounded`. The rule never reads
 * `max_build_hash_table_bytes`, so disabling BUILD_PROBE with a zero budget does not change it.
 *
 * @param inputs Measurements of the complete build
 * @param limits Byte limits on the folded build
 * @param policy Rule to apply
 * @param publishes_filters Whether the join publishes dynamic filters; read only by
 * `build_fold_policy::legacy_filter_coupled`
 * @return Whether the build CONCAT should switch to `concat_all`
 */
[[nodiscard]] bool decide_build_fold(build_fold_inputs const& inputs,
                                     build_fold_limits limits,
                                     build_fold_policy policy,
                                     bool publishes_filters) noexcept;

}  // namespace sirius::op
