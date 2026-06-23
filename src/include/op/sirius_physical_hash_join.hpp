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
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "expression/ast/node.hpp"  // complete sirius::ast::node for join_condition's destructor
#include "expression/join_condition.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "sirius_config.hpp"
#include "utils.hpp"

#include <cstddef>
#include <cstdint>

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

// STANDARD uses cudf APIs where the build and probe is a single operation.
// BUILD_PROBE builds the hash table in one step and then probes it in a separate step, which allows
// for better pipelining with other operators, and allows reusing the hash table. MIXED_JOIN uses
// cudf's mixed_join API for joins with both equality and inequality conditions.
enum class HASH_JOIN_MODE { STANDARD, BUILD_PROBE, MIXED_JOIN };
enum class BUILD_HASH_TABLE_STATE { NOT_BUILT, SCHEDULING, SCHEDULED, BUILT, DESTROYED };

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
    uint64_t max_build_hash_table_bytes = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES);

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

  static void build_join_pipelines(pipeline::sirius_pipeline& current,
                                   pipeline::sirius_meta_pipeline& meta_pipeline,
                                   sirius_physical_operator& op,
                                   bool build_rhs = true);

  /**
   * @brief Returns true if the given join conditions can be handled by this operator.
   *
   * Requires at least one equality condition. For mixed joins (equality + inequality), also
   * requires that no column referenced by an equality condition appears in any inequality
   * condition on the same side — cuDF's mixed_join API requires disjoint equality and
   * conditional table columns.
   */
  static bool are_conditions_supported(duckdb::vector<sirius::join_condition>& conditions);
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
  void update_join_exec_mode(int num_partitions,
                             uint64_t build_side_bytes,
                             bool build_foldable_to_single_batch);

  /// @brief True when this join runs in build-then-probe mode (see `update_join_exec_mode`).
  [[nodiscard]] bool is_build_probe_mode();

  std::unique_ptr<operator_data> get_next_task_input_data_for_build_probe();
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::optional<task_creation_hint> get_next_task_hint(
    std::optional<std::size_t> downstream_request = std::nullopt) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! Join Keys statistics (optional)
  duckdb::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> join_stats;

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

  HASH_JOIN_MODE _join_mode                      = HASH_JOIN_MODE::STANDARD;
  BUILD_HASH_TABLE_STATE _hash_table_build_state = BUILD_HASH_TABLE_STATE::NOT_BUILT;
  uint64_t _max_build_hash_table_bytes           = config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES;
  std::unique_ptr<cudf::hash_join> _hash_table;  // hash object to be used in BUILD_PROBE mode
  std::unique_ptr<cudf::distinct_hash_join>
    _distinct_hash_table;  // used instead of _hash_table when build keys are proven unique
  std::optional<::cucascade::read_only_data_batch>
    _build_table;  // owned build table for BUILD_PROBE mode, to materialize build side results
  std::vector<std::unique_ptr<cudf::column>>
    _built_table_cast_columns;  // scope holder for any columns that may have had to be cast for the
                                // build table
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

 public:
  // Sink Interface
  bool is_sink() const override { return true; }

  void on_finalize_operator() override;
};

}  // namespace op
}  // namespace sirius
