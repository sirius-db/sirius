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

#include "cudf/cudf_utils.hpp"
#include "cudf/join/distinct_hash_join.hpp"
#include "cudf/join/filtered_join.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "expression/ast/node.hpp"  // complete sirius::ast::node for join_condition's destructor
#include "expression/join_condition.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "sirius_config.hpp"
#include "utils.hpp"

#include <cudf/types.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

class sirius_dynamic_filter_set;

// STANDARD uses cudf APIs where the build and probe is a single operation.
// BUILD_PROBE builds the hash table in one step and then probes it in a separate step, which allows
// for better pipelining with other operators, and allows reusing the hash table. MIXED_JOIN uses
// cudf's mixed_join API for joins with both equality and inequality conditions.
enum class HASH_JOIN_MODE { STANDARD, BUILD_PROBE, MIXED_JOIN };
enum class BUILD_HASH_TABLE_STATE { NOT_BUILT, SCHEDULING, SCHEDULED, BUILT, DESTROYED };

//===----------------------------------------------------------------------===//
// BUILD_PROBE scheduling helpers - BUILD_PROBE keeps one hash table per partition.
//===----------------------------------------------------------------------===//

/// The action the BUILD_PROBE state machine should take next.
enum class build_probe_action {
  schedule_build,  ///< Build the hash table for `partition` (and probe its first batch).
  schedule_probe,  ///< Probe an already-built `partition` with its next probe batch.
  wait_for_build,  ///< No schedulable work; a partition is still awaiting its build batch.
  wait_for_probe,  ///< No schedulable work; awaiting probe batches (or the op is draining).
  none             ///< No partitions exist / all are destroyed.
};

/// A single partition's observable state for the scheduling decision.
struct build_probe_slot_view {
  BUILD_HASH_TABLE_STATE state = BUILD_HASH_TABLE_STATE::NOT_BUILT;
  bool has_build_batch         = false;  ///< The build repo holds this partition's build batch.
  bool has_probe_batch         = false;  ///< The probe repo holds >=1 batch for this partition.
};

struct build_probe_decision {
  build_probe_action action = build_probe_action::none;
  // Set only for schedule_build / schedule_probe — the partition to act on. nullopt for
  // wait_for_build / wait_for_probe / none, which name no partition (the caller waits on the port's
  // single upstream producer, shared by every partition).
  std::optional<std::size_t> partition;
};

/// Decide the next BUILD_PROBE action from a per-partition snapshot. Prefers scheduling a build for
/// the first NOT_BUILT partition that has both its build and a probe batch, then probing the first
/// BUILT partition with probe data; otherwise reports whether it is waiting on build or probe
/// input.
[[nodiscard]] build_probe_decision select_build_probe_action(
  std::vector<build_probe_slot_view> const& slots);

/// Pure decision for how a PARTITION operator should partition its input for a hash join, folding
/// the natural-count, broadcast-candidacy, and BUILD_PROBE-eligibility logic into one place.
///
/// Only the build side drives broadcast / build-probe, so when `is_build_side` is false (right-
/// family joins are probe-driven) the result is the plain STANDARD-mode natural count.
///
/// MARK joins cannot be hash-partitioned across batches (build_has_null must be globally
/// consistent), so they are clamped to one partition on a single GPU and forced to broadcast on
/// multi-GPU.
///
/// BUILD_PROBE eligibility requires: the build folds to one hash table per GPU within
/// `max_build_hash_table_bytes` (a broadcast join charges the FULL build to every GPU; a
/// hash-partitioned build charges the per-GPU average), the build folds to one batch, and the
/// join is not right-family, mixed, or full-outer (those over-emit build rows on the streamed
/// path). `join_mode` distinguishes MIXED_JOIN; `join_type` supplies the rest.
///
/// Broadcast candidacy: a build is replicate-worthy when it is below the small-table threshold, OR
/// when it is below `max_broadcast_join_size` AND the probe side is large relative to the build
/// (`estimated_probe_to_build_ratio >= num_gpus * 1.25`) — replicating a medium build avoids
/// shuffling a much larger probe across GPUs.
[[nodiscard]] partition_strategy compute_hash_join_partition_strategy(
  uint64_t total_bytes,
  bool is_build_side,
  bool build_foldable,
  int num_gpus,
  uint64_t hash_partition_bytes,
  uint64_t max_build_hash_table_bytes,
  uint64_t max_broadcast_join_size,
  duckdb::JoinType join_type,
  HASH_JOIN_MODE join_mode,
  double estimated_probe_to_build_ratio);

/// Which broadcast slots to discard. In a broadcast join the build table is replicated to every
/// slot but the probe side is unpartitioned, so a slot may hold build data yet never receive probe
/// data. Once the probe upstream is finished (`probe_finished`), any slot that is still NOT_BUILT
/// with a build batch but no probe batch will never be probed and is returned for discard. Returns
/// empty while the probe side may still deliver data. Pure/unit-testable counterpart of
/// discard_build_only_slots_if_probe_complete.
[[nodiscard]] std::vector<std::size_t> broadcast_slots_to_discard(
  std::vector<build_probe_slot_view> const& slots, bool probe_finished);

//===----------------------------------------------------------------------===//
// STANDARD / MIXED_JOIN partial-barrier scheduling helpers.
//
// Under a partial barrier the build and probe batches for a partition arrive progressively. For a
// partition we must join every build batch against every probe batch. The upstream concat operator
// folds whichever side must be seen whole (LEFT/SEMI/ANTI -> build; RIGHT-family -> probe; OUTER ->
// both) down to a single batch, so this per-partition cross product is correct for every join type
// (see docs/super-sirius/operators.md). This structure tracks, per partition, the batch IDs
// discovered on each side and which (probe,build) pairs have already been scheduled, so scheduling
// tolerates the ID lists growing across calls and each batch is freed once it can no longer be
// needed. probe = "default"/left, build = "build"/right.
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

class sirius_physical_hash_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::HASH_JOIN;

  struct join_projection_columns {
    std::vector<cudf::size_type> col_idxs;
    duckdb::vector<sirius::logical_type> col_types;
  };

 public:
  sirius_physical_hash_join(
    duckdb::LogicalOperator& op,
    duckdb::unique_ptr<sirius_physical_operator> left,
    duckdb::unique_ptr<sirius_physical_operator> right,
    duckdb::vector<sirius::join_condition> cond,
    duckdb::JoinType join_type,
    const duckdb::vector<std::size_t>& left_projection_map,
    const duckdb::vector<std::size_t>& right_projection_map,
    duckdb::vector<sirius::logical_type> delim_types,
    std::size_t estimated_cardinality,
    uint64_t max_build_hash_table_bytes             = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
    dynamic_filter_publish_plan dynamic_filter_plan = {},
    uint64_t hash_partition_bytes                   = config::DEFAULT_HASH_PARTITION_BYTES,
    uint64_t max_broadcast_join_size                = config::DEFAULT_MAX_BROADCAST_JOIN_SIZE,
    dynamic_filter_stats* dynamic_filter_stats_sink = {});

  duckdb::vector<sirius::join_condition> conditions;
  //! The types of the join keys
  duckdb::vector<sirius::logical_type> condition_types;
  //! The type of the join
  duckdb::JoinType join_type;

  //! The indices/types of the payload columns
  join_projection_columns payload_columns;
  //! The indices/types of the lhs columns that need to be output
  join_projection_columns lhs_output_columns;
  //! The indices/types of the rhs columns that need to be output
  join_projection_columns rhs_output_columns;

  //! Duplicate eliminated types; only used for delim_joins (i.e. correlated subqueries)
  duckdb::vector<sirius::logical_type> delim_types;

  mutable bool unique_build_keys = false;

  mutable bool unique_probe_keys = false;

  //! When the planner could not *prove* build-key uniqueness, test it at runtime instead (one hash
  //! pass over the build keys) and, if the keys are in fact distinct, take the single-pass
  //! cudf::distinct_hash_join path rather than the general two-pass multiset path. Proving
  //! uniqueness statically needs a declared PRIMARY KEY on a catalog table. Set from
  //! operator_params at planning time.
  bool runtime_distinct_build_probe = config::DEFAULT_ENABLE_RUNTIME_DISTINCT_BUILD_PROBE;

  //! Row-count ratio gate for switching STANDARD-mode MARK joins to cudf::mark_join (build on the
  //! left/output side) instead of filtered_join (build on the right side). Switch when
  //! right_rows >= ratio * left_rows; 0 disables. Set from operator_params at planning time.
  double mark_join_build_switch_ratio = config::DEFAULT_MARK_JOIN_BUILD_SWITCH_RATIO;

  //! Join Keys statistics (optional)
  duckdb::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> join_stats;

  void restrict_dynamic_filter_replicas(std::vector<int> const& admitted_gpu_ids)
  {
    _dynamic_filter_plan.restrict_replicas_to(admitted_gpu_ids);
  }

  static void build_join_pipelines(pipeline::sirius_pipeline& current,
                                   pipeline::sirius_meta_pipeline& meta_pipeline,
                                   sirius_physical_operator& op);

  //! Whether the execute dispatch has an arm for @p join_type. SINGLE does not;
  //! are_conditions_supported() folds this in so the planner never builds one.
  static bool is_join_type_supported(duckdb::JoinType join_type);

  /**
   * @brief Returns true if the given join type and conditions can be handled by this operator.
   *
   * Requires an executable join type (is_join_type_supported) and at least one equality
   * condition. For mixed joins (equality + inequality), also requires that no column referenced
   * by an equality condition appears in any inequality condition on the same side — cuDF's
   * mixed_join API requires disjoint equality and conditional table columns. A MARK join mixing
   * null-safe and plain keys is rejected; see mark_join_mixes_null_safe_keys.
   *
   * @param join_type Used to exclude MARK joins from null-safe routing.
   */
  static bool are_conditions_supported(duckdb::vector<sirius::join_condition>& conditions,
                                       duckdb::JoinType join_type);

  [[nodiscard]] bool is_right_family() const
  {
    return join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::RIGHT_SEMI ||
           join_type == duckdb::JoinType::RIGHT_ANTI;
  }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  /// @brief Called by the upstream PARTITION operator to decide how it should partition its input
  /// for this join. Computes the partition count / broadcast flag (via
  /// compute_hash_join_partition_strategy), then applies the join-side side effects atomically
  /// under op_state_mutex: switching to BUILD_PROBE mode (allocating the per-partition build
  /// states), marking broadcast, and pre-sizing the build/probe input repositories. Returns the
  /// decision so the partition can finish its own wiring (e.g. enabling build-side concat_all).
  partition_strategy get_partition_strategy(const partition_sizing_input& in) override;

  [[nodiscard]] bool publishes_dynamic_filters() const;

  [[nodiscard]] uint64_t max_build_hash_table_bytes() const noexcept
  {
    return _max_build_hash_table_bytes;
  }

  // Publication may claim only a delivery known to contain the whole build.
  void set_build_arrives_whole(bool arrives_whole);

  /// @brief True when this join runs in build-then-probe mode (see `get_partition_strategy`).
  [[nodiscard]] bool is_build_probe_mode();

  std::unique_ptr<operator_data> get_next_task_input_data_for_build_probe();
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /// STANDARD / MIXED_JOIN only. Re-poll both input ports for newly-arrived batch IDs, extend
  /// `_cross`, enforce the non-INNER whole-side invariant (the must-be-whole side stays a single
  /// concat-folded batch), and free every fully-consumed batch from its repository. Must be called
  /// with op_state_mutex held. Returns the (probe_finished, build_finished) producer state so the
  /// caller can reuse it without re-querying.
  std::pair<bool, bool> refresh_cross_schedule();

  /// Snapshot each partition's build state and per-partition data availability for the BUILD_PROBE
  /// scheduler (`select_build_probe_action`). Must be called with `op_state_mutex` held.
  std::vector<build_probe_slot_view> snapshot_build_probe_slots();

  /// Broadcast-mode cleanup: once the probe upstream is finished, any NOT_BUILT slot that holds a
  /// (replicated) build batch but never received probe data is discarded — its build batch is freed
  /// on its own GPU and the slot is marked DESTROYED so the operator can complete. No-op unless
  /// `_broadcast`. Must be called with `op_state_mutex` held.
  void discard_build_only_slots_if_probe_complete();

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 protected:
  // double get_progress(duckdb::ClientContext &context, duckdb::GlobalSourceState &gstate) const
  // override;

  //! Becomes a source when it is an external join
  bool is_source() const override { return true; }

  std::mutex op_state_mutex;
  // STANDARD / MIXED_JOIN partial-barrier schedule: one growable cross-product tracker per
  // partition. Lazily sized to num_partitions on first use. Guarded by op_state_mutex.
  std::vector<partition_cross_schedule> _cross;

  bool is_all_inequality_join = true;

  HASH_JOIN_MODE _join_mode            = HASH_JOIN_MODE::STANDARD;
  uint64_t _max_build_hash_table_bytes = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES;
  // Maximum build-side bytes eligible for a broadcast join (see get_partition_strategy). Set from
  // operator_params at construction.
  uint64_t _max_broadcast_join_size = config::DEFAULT_MAX_BROADCAST_JOIN_SIZE;
  // _num_gpus lives on sirius_physical_partition_consumer_operator (set via set_num_gpus).

  // Broadcast (small build table) BUILD_PROBE join: the build side is replicated to every slot and
  // the probe side is streamed unpartitioned.
  bool _broadcast = false;

  bool _build_arrives_whole = false;

  // Guarded by op_state_mutex.
  bool _build_not_whole_reported = false;

  // Whether any build-side join key column contains a NULL. Used exclusively for MARK join
  // three-valued logic. Sentinel -1 = unset, 0 = false, 1 = true. Join-wide (not per-partition)
  // because MARK joins are forced to a single partition / broadcast, so all build batches agree.
  std::atomic<int> _build_has_null{-1};

  // Per-partition build/probe state for BUILD_PROBE mode. Each partition owns one cuco hash table
  // that lives entirely on one GPU . A partition is built once — its
  // single SCHEDULED build task release-stores BUILT — and then probed by many streamed probe tasks
  // that only read the table. `build_state` is atomic so get_next_task_hint can observe a slot's
  // progress without holding op_state_mutex while execute() flips it.
  struct per_partition_build_state {
    std::atomic<BUILD_HASH_TABLE_STATE> build_state{BUILD_HASH_TABLE_STATE::NOT_BUILT};
    std::unique_ptr<cudf::hash_join> hash_table;  // general path (INNER/LEFT/OUTER)
    std::unique_ptr<cudf::distinct_hash_join>
      distinct_hash_table;  // used instead of hash_table when build keys are proven unique
    std::unique_ptr<cudf::filtered_join>
      filtered_table;  // reusable build-on-right object for MARK/SEMI/ANTI joins
    std::optional<::cucascade::read_only_data_batch>
      build_table;  // owned build table, to materialize build-side results at probe time
    std::vector<std::unique_ptr<cudf::column>>
      built_table_cast_columns;  // scope holder for columns cast for the build table's lifetime
    int device_id = -1;          // GPU this slot's table was built on; guards teardown frees
  };
  // Sized to num_partitions when BUILD_PROBE is entered (see get_partition_strategy). Elements are
  // non-movable (atomic member), so the vector is default-constructed at the target size and only
  // ever whole-move-assigned — never resized or push_back'd — so element moves are never required.
  std::vector<per_partition_build_state> _partition_build_states;
  //
  // Number of equality conditions after reordering; inequality conditions follow at higher indices.
  std::size_t num_equality_conditions = 0;
  std::vector<cudf::size_type> left_key_col_indices;
  std::vector<cudf::size_type> right_key_col_indices;
  bool cast_necessary = false;

  /// Null-matching flag for the equi-keys passed to cuDF joins, cached at
  /// construction (conditions and join_type are fixed thereafter). cuDF applies one
  /// flag to all key columns, so it is EQUAL (null-safe -- NULL matches NULL) only
  /// when EVERY equi-key is IS NOT DISTINCT FROM; a plain `=` key (including mixed
  /// joins such as delim joins) forces UNEQUAL. A MARK join follows the same rule:
  /// all-null-safe keys use EQUAL and emit definite marks (see mark_is_null_safe),
  /// anything containing a plain key uses UNEQUAL and the IN/EXISTS three-valued
  /// result logic. A MARK join *mixing* the two is rejected at plan time.
  cudf::null_equality compare_nulls() const { return compare_nulls_; }
  cudf::null_equality compare_nulls_ = cudf::null_equality::UNEQUAL;

  /// A MARK join whose every condition is IS NOT DISTINCT FROM: the predicate is never UNKNOWN,
  /// so every mark is definite and the result carries no null mask.
  [[nodiscard]] bool mark_is_null_safe() const { return mark_is_null_safe_; }
  bool mark_is_null_safe_ = false;

 public:
  //! Per-key cast info: whether each join key needs a cast before comparison
  struct key_cast_info {
    bool cast_left  = false;
    bool cast_right = false;
    cudf::data_type left_target_type{cudf::type_id::EMPTY};
    cudf::data_type right_target_type{cudf::type_id::EMPTY};
  };

 protected:
  std::vector<key_cast_info> key_casts;

  // Requires PUBLISHING and leaves FINISHED or FAILED. Device OOM is contained; other failures
  // propagate.
  void publish_dynamic_filters(cudf::table_view const& build_view, rmm::cuda_stream_view stream);

  enum class dynamic_filter_publication_state : std::uint8_t {
    OPEN,
    PUBLISHING,
    FINISHED,
    FAILED,  ///< Terminal: a failed window is never reopened for a sibling retry.
    CLOSED
  };

  // Narrowed before execution; immutable during execution.
  dynamic_filter_publish_plan _dynamic_filter_plan;
  // Non-owning; SiriusContext outlives the plan.
  dynamic_filter_stats* _dynamic_filter_stats = nullptr;
  std::atomic<dynamic_filter_publication_state> _dynamic_filter_publication_state{
    dynamic_filter_publication_state::OPEN};

 public:
  void push_data_batch_partitioned(std::string_view port_id,
                                   std::shared_ptr<::cucascade::data_batch> batch,
                                   std::size_t partition_idx) override;

  [[nodiscard]] dynamic_filter_publish_plan const& dynamic_filter_plan() const noexcept
  {
    return _dynamic_filter_plan;
  }

 public:
  //! True when this HJ is the internal `delim.join` of a RIGHT_DELIM_JOIN (set in its
  //! constructor). The delim join owns its execution: `is_sink()` returns false and
  //! `build_join_pipelines` skips build-side externalization.
  [[nodiscard]] bool is_delim_join_inner() const noexcept { return _is_delim_join_inner; }
  void set_delim_join_inner(bool value) noexcept { _is_delim_join_inner = value; }

  // Sink Interface
  //! The inner join of a RIGHT_DELIM_JOIN is never a sink; otherwise the base rule
  //! applies (sink iff parent is PARTITION or RIGHT_DELIM_JOIN).
  bool is_sink() const override
  {
    if (_is_delim_join_inner) { return false; }
    return sirius_physical_operator::is_sink();
  }

  void on_finalize_operator() override;

 protected:
  bool _is_delim_join_inner = false;
};

}  // namespace op
}  // namespace sirius
