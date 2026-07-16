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
#include "op/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter_replica_space.hpp"
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

/// Whether a join is eligible to run in BUILD_PROBE mode. Mirrors the gate in
/// update_join_exec_mode: at most one partition per GPU, each GPU's hash table fits a single hash
/// table, the build side folds to one batch per partition, and the join is not a right-family,
/// mixed, or full-outer join.
///
/// `is_broadcast` selects how the build size is charged: a broadcast join replicates the ENTIRE
/// build to every GPU, so each GPU's hash table is the full `build_side_bytes`; a hash-partitioned
/// build splits across partitions, so the per-partition average applies.
///
/// `is_full_outer` (OUTER) is excluded: BUILD_PROBE streams probe batches and calls `full_join`
/// per batch, emitting unmatched build rows on every batch (and, under broadcast/partitioning, on
/// every GPU) with no global accumulation — so it over-emits build-side rows. Full outer joins run
/// on the STANDARD path instead.
[[nodiscard]] bool build_probe_mode_eligible(int num_partitions,
                                             uint64_t build_side_bytes,
                                             bool build_foldable_to_single_batch,
                                             bool is_right_family,
                                             bool is_mixed_join,
                                             int num_gpus,
                                             uint64_t max_build_hash_table_bytes,
                                             bool is_full_outer = false,
                                             bool is_broadcast  = false);

/// Which broadcast slots to discard. In a broadcast join the build table is replicated to every
/// slot but the probe side is unpartitioned, so a slot may hold build data yet never receive probe
/// data. Once the probe upstream is finished (`probe_finished`), any slot that is still NOT_BUILT
/// with a build batch but no probe batch will never be probed and is returned for discard. Returns
/// empty while the probe side may still deliver data. Pure/unit-testable counterpart of
/// discard_build_only_slots_if_probe_complete.
[[nodiscard]] std::vector<std::size_t> broadcast_slots_to_discard(
  std::vector<build_probe_slot_view> const& slots, bool probe_finished);

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
    duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info,
    uint64_t max_build_hash_table_bytes             = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
    dynamic_filter_publish_plan dynamic_filter_plan = {});

  sirius_physical_hash_join(
    duckdb::LogicalOperator& op,
    duckdb::unique_ptr<sirius_physical_operator> left,
    duckdb::unique_ptr<sirius_physical_operator> right,
    duckdb::vector<sirius::join_condition> cond,
    duckdb::JoinType join_type,
    std::size_t estimated_cardinality,
    uint64_t max_build_hash_table_bytes = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES);

  duckdb::vector<sirius::join_condition> conditions;
  //! Scans where we should push generated filters into (if any)
  duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> filter_pushdown;

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

  //! Row-count ratio gate for switching STANDARD-mode MARK joins to cudf::mark_join (build on the
  //! left/output side) instead of filtered_join (build on the right side). Switch when
  //! right_rows >= ratio * left_rows; 0 disables. Set from operator_params at planning time.
  double mark_join_build_switch_ratio = config::DEFAULT_MARK_JOIN_BUILD_SWITCH_RATIO;

  //! Join Keys statistics (optional)
  duckdb::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> join_stats;

  static void build_join_pipelines(pipeline::sirius_pipeline& current,
                                   pipeline::sirius_meta_pipeline& meta_pipeline,
                                   sirius_physical_operator& op);

  /**
   * @brief Returns true if the given join conditions can be handled by this operator.
   *
   * Requires at least one equality condition. For mixed joins (equality + inequality), also
   * requires that no column referenced by an equality condition appears in any inequality
   * condition on the same side — cuDF's mixed_join API requires disjoint equality and
   * conditional table columns.
   */
  static bool are_conditions_supported(duckdb::vector<sirius::join_condition>& conditions);

  [[nodiscard]] bool is_right_family() const
  {
    return join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::RIGHT_SEMI ||
           join_type == duckdb::JoinType::RIGHT_ANTI;
  }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  /// @brief This is called by the partition operator to inform the hash join of the number of
  /// partitions that will be produced by the partition operator, which can be used to make
  /// decisions about the join execution strategy (e.g., whether to switch to a build-probe strategy
  /// for small datasets).
  /// @param num_partitions
  /// @param build_side_bytes
  /// @param build_foldable_to_single_batch True when the upstream pipeline can guarantee the
  ///        build side will arrive as exactly one batch (typically because a downstream
  ///        build-side CONCAT was configured with concat_all). BUILD_PROBE mode requires
  ///        the build side to fold into a single batch — when this guarantee is absent the
  ///        runtime-side build-batch invariant in get_next_task_input_data_for_build_probe
  ///        would throw on otherwise-valid small-build joins that are still split into
  ///        multiple batches, so BUILD_PROBE is not entered.
  /// @param is_broadcast_candidate True when the partition operator intends to replicate the whole
  ///        (small) build table to every GPU. The build size is then charged in full against
  ///        max_build_hash_table_bytes (each GPU builds the entire table), not per-partition.
  void update_join_exec_mode(int num_partitions,
                             uint64_t build_side_bytes,
                             bool build_foldable_to_single_batch,
                             bool is_broadcast_candidate = false);

  /// @brief Inform the join how many GPUs the query runs on. Set at plan time.
  void set_num_gpus(int num_gpus) { _num_gpus = num_gpus; }

  /// @brief Mark this join as a broadcast (small-build-table) BUILD_PROBE join. Set at runtime.
  void set_broadcast(bool broadcast) { _broadcast = broadcast; }

  /// @brief True when this join runs in build-then-probe mode (see `update_join_exec_mode`).
  [[nodiscard]] bool is_build_probe_mode();

  std::unique_ptr<operator_data> get_next_task_input_data_for_build_probe();
  std::unique_ptr<operator_data> get_next_task_input_data() override;

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
  std::size_t current_partition_index = 0;
  std::size_t num_batches_to_process  = 0;
  std::vector<std::vector<uint64_t>> left_batch_ids;
  std::vector<std::vector<uint64_t>> right_batch_ids;

  bool is_all_inequality_join = true;

  HASH_JOIN_MODE _join_mode            = HASH_JOIN_MODE::STANDARD;
  uint64_t _max_build_hash_table_bytes = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES;

  // Number of GPUs the query runs on (set at plan time via set_num_gpus).
  int _num_gpus = 1;

  // Broadcast (small build table) BUILD_PROBE join: the build side is replicated to every slot and
  // the probe side is streamed unpartitioned.
  bool _broadcast = false;

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
      filtered_table;             // reusable build-on-right object for MARK/SEMI/ANTI joins
    bool build_has_null = false;  // whether the build/right side has a NULL in any join key column;
                                  // needed for MARK three-valued logic at probe time
    std::optional<::cucascade::read_only_data_batch>
      build_table;  // owned build table, to materialize build-side results at probe time
    std::vector<std::unique_ptr<cudf::column>>
      built_table_cast_columns;  // scope holder for columns cast for the build table's lifetime
    int device_id = -1;          // GPU this slot's table was built on; guards teardown frees
  };
  // Sized to num_partitions when BUILD_PROBE is entered (see update_join_exec_mode). Elements are
  // non-movable (atomic member), so the vector is default-constructed at the target size and only
  // ever whole-move-assigned — never resized or push_back'd — so element moves are never required.
  std::vector<per_partition_build_state> _partition_build_states;
  //
  // Number of equality conditions after reordering; inequality conditions follow at higher indices.
  std::size_t num_equality_conditions = 0;
  std::vector<cudf::size_type> left_key_col_indices;
  std::vector<cudf::size_type> right_key_col_indices;
  bool cast_necessary = false;

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

  //===----------------------------------------------------------------------===//
  // Dynamic Filters
  //===----------------------------------------------------------------------===//
  /// @brief Claim and perform this join's one dynamic-filter publication attempt.
  ///
  /// The only publishing caller is @ref push_data_batch_partitioned: it publishes as soon as the
  /// single, concat-folded build batch reaches the build port, before any probe batch is required.
  ///
  /// The caller that changes @c OPEN to @c PUBLISHING owns construction, device replication,
  /// and channel fan-out. GPU work runs without holding @ref op_state_mutex. A successful attempt
  /// ends in @c FINISHED even when selectivity gates or drained targets cause it to emit no
  /// filters. @ref on_finalize_operator never publishes; it only changes an unclaimed @c OPEN
  /// window to @c CLOSED before releasing BUILD_PROBE state.
  ///
  /// @param build_view The build side to reduce / build membership over.
  /// @param stream     Durable build-memory-space stream used for filter construction.
  void publish_dynamic_filters(cudf::table_view const& build_view, rmm::cuda_stream_view stream);

  enum class dynamic_filter_publication_state : std::uint8_t {
    OPEN,        ///< The publication hook has not claimed the build table.
    PUBLISHING,  ///< The claiming caller owns construction, replication, and fan-out.
    FINISHED,    ///< The one publication attempt completed successfully (possibly emitting none).
    FAILED,      ///< The claimed attempt threw; the uncertain state must not be retried.
    CLOSED       ///< Finalization closed the window before the hook claimed it.
  };

  /// Complete plan-time routing, policy, and replica-space description; immutable at runtime.
  dynamic_filter_publish_plan const _dynamic_filter_plan;
  /// Exactly-once arbitration between the publication hook and finalization.
  std::atomic<dynamic_filter_publication_state> _dynamic_filter_publication_state{
    dynamic_filter_publication_state::OPEN};
  //===----------------------------------------------------------------------===//

 public:
  /// @brief Route a partitioned batch and publish dynamic filters from an eligible build batch.
  ///
  /// For the @c build port of a wired @c BUILD_PROBE join, the single concat-folded batch is the
  /// publication point. Its read-only accessor is acquired BEFORE the batch is routed: once
  /// deposited into a repository the batch becomes a downgrade candidate, and holding the shared
  /// lock across the deposit pins its GPU representation until publication completes. A stream
  /// borrowed from the build memory space then waits on the batch writer event and builds and
  /// replicates filters from the build keys without requiring a probe batch or a built hash table.
  ///
  /// Other ports and join modes only route. This synchronous hook completes before this join's
  /// immediate probe producer is scheduled. A scan target reached through an intervening join is
  /// not gated by that edge.
  void push_data_batch_partitioned(std::string_view port_id,
                                   std::shared_ptr<::cucascade::data_batch> batch,
                                   std::size_t partition_idx) override;

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
