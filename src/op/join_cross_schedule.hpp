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

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op {

//===----------------------------------------------------------------------===//
// Incremental batch-pair scheduling shared by hash and nested loop joins.
//
// Under a partial barrier the build and probe batches for a partition arrive progressively. For a
// partition we must join every build batch against every probe batch. The upstream concat operator
// folds whichever side must be seen whole (LEFT/SEMI/ANTI and NLJ MARK -> build; RIGHT-family ->
// probe; OUTER -> both) down to a single batch, so this per-partition cross product is correct for
// every join type (see docs/super-sirius/operators.md). This structure tracks, per partition, the
// batch IDs discovered on each side and which (probe,build) pairs have already been scheduled, so
// scheduling tolerates the ID lists growing across calls and each batch is freed once it can no
// longer be needed. probe = "default"/left, build = "build"/right.
//===----------------------------------------------------------------------===//
struct partition_cross_schedule {
  std::vector<uint64_t> probe_ids;               ///< probe batch IDs, first-seen order
  std::vector<uint64_t> build_ids;               ///< build batch IDs, first-seen order
  std::unordered_set<uint64_t> probe_id_seen;    ///< dedup guard while re-polling
  std::unordered_set<uint64_t> build_id_seen;    ///< dedup guard while re-polling
  std::unordered_set<uint64_t> scheduled_pairs;  ///< key = (probe_idx << 32) | build_idx
  std::vector<uint32_t> probe_paired_count;      ///< parallel to probe_ids: #build batches paired
  std::vector<uint32_t> build_paired_count;      ///< parallel to build_ids: #probe batches paired
  std::unordered_set<uint64_t> probe_popped;     ///< probe IDs already freed from the repo
  std::unordered_set<uint64_t> build_popped;     ///< build IDs already freed from the repo
};

/// Merge newly discovered IDs into a partition, preserving first-seen order and existing pairing
/// counts. Seen sets prevent duplicates, including IDs already popped from the repositories.
void merge_cross_schedule_batch_ids(partition_cross_schedule& cross,
                                    std::vector<uint64_t> const& probe_ids,
                                    std::vector<uint64_t> const& build_ids);

/// A fully-consumed batch that can be freed from its repository.
struct cross_schedule_discard {
  std::size_t partition = 0;
  bool is_build         = false;  ///< true: "build" port; false: "default"/probe port
  uint64_t batch_id     = 0;
};

enum class cross_schedule_kind : std::uint8_t { emit_pair, wait_build, wait_probe, done };

/// The next (probe,build) pair to schedule, or why none is schedulable.
struct cross_schedule_pair {
  cross_schedule_kind kind = cross_schedule_kind::done;
  std::size_t partition    = 0;  ///< valid iff kind == emit_pair
  std::size_t probe_idx    = 0;  ///< index into partition's probe_ids; valid iff emit_pair
  std::size_t build_idx    = 0;  ///< index into partition's build_ids; valid iff emit_pair
};

/// Encode / decode a (probe_idx, build_idx) pair into the `scheduled_pairs` key. Per-partition
/// batch counts are far below 2^32, so a single uint64_t key is safe.
[[nodiscard]] inline uint64_t encode_cross_pair(std::size_t probe_idx, std::size_t build_idx)
{
  return (static_cast<uint64_t>(probe_idx) << 32) | static_cast<uint64_t>(build_idx);
}

/// Collect (and mark popped) every batch that is now fully consumed: a probe batch once the build
/// producer is finished and the batch has been paired with every known build batch (build side has
/// >= 1 batch), and symmetrically for build batches. Idempotent.
[[nodiscard]] std::vector<cross_schedule_discard> collect_cross_schedule_discards(
  std::vector<partition_cross_schedule>& cross, bool probe_finished, bool build_finished);

/// Reference arithmetic for a pairing-weighted probe-byte denominator: sum
/// `probe bytes * completed pairings / build batches`. Unused — the STANDARD/MIXED estimate is
/// withheld — but kept with its tests for the eventual fix. See
/// data-size-estimation.md#why-standard-and-mixed_join-are-not-estimated.
[[nodiscard]] std::size_t pairing_weighted_probe_bytes(
  std::vector<partition_cross_schedule> const& cross,
  std::unordered_map<uint64_t, std::size_t> const& probe_bytes);

/// Find and claim the next unscheduled (probe,build) pair across partitions, marking it scheduled
/// and bumping the paired counts. If none is schedulable now, reports whether to wait on the build
/// or probe producer, or that the operator is done. Pure over `cross`.
[[nodiscard]] cross_schedule_pair next_cross_schedule_pair(
  std::vector<partition_cross_schedule>& cross, bool probe_finished, bool build_finished);

/// A surviving batch of a partition whose OPPOSITE side finished with zero batches (empty table).
/// It is emitted as a single-side "orphan" task so execute() can produce the correct NULL-padded /
/// passthrough output, and popping it drains the repository so the pipeline can complete.
struct cross_schedule_orphan {
  bool found            = false;
  std::size_t partition = 0;
  bool present_is_build = false;  ///< the surviving side is the build side (opposite probe empty)
  uint64_t batch_id     = 0;
};

/// Find and claim (mark popped) the next surviving batch whose opposite side finished empty, once
/// BOTH producers are finished. Returns `{found=false}` when no such batch remains. Only fires when
/// the opposite side truly has zero batches, so it never races the normal pair scheduler (which
/// covers partitions with data on both sides). Pure over `cross`.
[[nodiscard]] cross_schedule_orphan next_cross_schedule_orphan(
  std::vector<partition_cross_schedule>& cross, bool probe_finished, bool build_finished);

/// True when a partition has a surviving batch whose opposite side finished empty and that batch
/// has not yet been emitted/popped. Used by peek_cross_schedule_kind and by get_next_task_hint.
/// Pure.
[[nodiscard]] bool has_pending_cross_orphan(std::vector<partition_cross_schedule> const& cross,
                                            bool probe_finished,
                                            bool build_finished);

/// Non-mutating classification for get_next_task_hint: READY (emit_pair, including pending orphan
/// work), WAITING (wait_build / wait_probe), or complete (done).
[[nodiscard]] cross_schedule_kind peek_cross_schedule_kind(
  std::vector<partition_cross_schedule> const& cross, bool probe_finished, bool build_finished);

}  // namespace sirius::op
