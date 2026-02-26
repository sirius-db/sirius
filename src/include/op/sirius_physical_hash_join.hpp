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

#include "cudf_utils.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "utils.hpp"

#include <cstddef>
#include <cstdint>

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

class sirius_physical_hash_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::HASH_JOIN;

  struct join_projection_columns {
    std::vector<cudf::size_type> col_idxs;
    duckdb::vector<duckdb::LogicalType> col_types;
  };

 public:
  sirius_physical_hash_join(duckdb::LogicalOperator& op,
                            duckdb::unique_ptr<sirius_physical_operator> left,
                            duckdb::unique_ptr<sirius_physical_operator> right,
                            duckdb::vector<duckdb::JoinCondition> cond,
                            duckdb::JoinType join_type,
                            const duckdb::vector<duckdb::idx_t>& left_projection_map,
                            const duckdb::vector<duckdb::idx_t>& right_projection_map,
                            duckdb::vector<duckdb::LogicalType> delim_types,
                            duckdb::idx_t estimated_cardinality,
                            duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info);
  sirius_physical_hash_join(duckdb::LogicalOperator& op,
                            duckdb::unique_ptr<sirius_physical_operator> left,
                            duckdb::unique_ptr<sirius_physical_operator> right,
                            duckdb::vector<duckdb::JoinCondition> cond,
                            duckdb::JoinType join_type,
                            duckdb::idx_t estimated_cardinality);

  duckdb::vector<duckdb::JoinCondition> conditions;
  //! Scans where we should push generated filters into (if any)
  duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> filter_pushdown;

  //! Initialize HT for this operator
  void initialize_hash_table(duckdb::ClientContext& context) const;

  //! The types of the join keys
  duckdb::vector<duckdb::LogicalType> condition_types;
  //! The type of the join
  duckdb::JoinType join_type;

  //! The indices/types of the payload columns
  join_projection_columns payload_columns;
  //! The indices/types of the lhs columns that need to be output
  join_projection_columns lhs_output_columns;
  //! The indices/types of the rhs columns that need to be output
  join_projection_columns rhs_output_columns;

  //! Duplicate eliminated types; only used for delim_joins (i.e. correlated subqueries)
  duckdb::vector<duckdb::LogicalType> delim_types;

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
  static bool are_conditions_supported(duckdb::vector<duckdb::JoinCondition>& conditions);
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! Join Keys statistics (optional)
  duckdb::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> join_stats;

 protected:
  // double get_progress(duckdb::ClientContext &context, duckdb::GlobalSourceState &gstate) const
  // override;

  //! Becomes a source when it is an external join
  bool is_source() const override { return true; }

  std::mutex batches_to_processed_mutex;
  std::size_t current_partition_index = 0;
  std::size_t num_batches_to_process  = 0;
  std::vector<std::vector<uint64_t>> left_batch_ids;
  std::vector<std::vector<uint64_t>> right_batch_ids;

  bool is_all_inequality_join = true;
  // True when conditions contain both equality conditions (hashed) and inequality conditions
  // (evaluated via cuDF mixed_join binary predicate).
  bool is_mixed_join = false;
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
};

}  // namespace op
}  // namespace sirius
