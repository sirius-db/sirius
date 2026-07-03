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

#include <cstdint>

namespace cudf {
class table_view;
}  // namespace cudf

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

/**
 * @brief Marker input for a zero-side nested-loop-join task
 *        (s3-shape-c-zero-side-join-plan.md §8).
 *
 * Handed out by sirius_physical_nested_loop_join::get_next_task_input_data when exactly one
 * join input side died (its source pipeline finished without ever delivering a batch) while
 * the other FULL port still holds data. Carries the ONE surviving batch popped from the live
 * port plus which side died; execute() recognizes it and emits the join-type-correct output
 * for that batch (NULL padding, pass-through, or empty) without evaluating the join
 * condition. Plain pipelineable data, matching the operator's regular task inputs.
 */
class nested_loop_join_zero_side_input : public pipelineable_operator_data {
 public:
  enum class side : uint8_t { PROBE, BUILD };

  nested_loop_join_zero_side_input(std::shared_ptr<::cucascade::data_batch> surviving_batch,
                                   side dead_side)
    : pipelineable_operator_data(
        std::vector<std::shared_ptr<::cucascade::data_batch>>{std::move(surviving_batch)}),
      dead_side(dead_side)
  {
  }

  side dead_side;
};

//! sirius_physical_nested_loop_join represents a nested loop join between two tables
class sirius_physical_nested_loop_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::NESTED_LOOP_JOIN;

 public:
  sirius_physical_nested_loop_join(
    duckdb::LogicalOperator& op,
    duckdb::unique_ptr<sirius_physical_operator> left,
    duckdb::unique_ptr<sirius_physical_operator> right,
    duckdb::vector<sirius::join_condition> cond,
    duckdb::JoinType join_type,
    std::size_t estimated_cardinality,
    duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info_p);

  sirius_physical_nested_loop_join(duckdb::LogicalOperator& op,
                                   duckdb::unique_ptr<sirius_physical_operator> left,
                                   duckdb::unique_ptr<sirius_physical_operator> right,
                                   duckdb::vector<sirius::join_condition> cond,
                                   duckdb::JoinType join_type,
                                   std::size_t estimated_cardinality);

  sirius_physical_nested_loop_join(duckdb::LogicalOperator& op,
                                   duckdb::unique_ptr<sirius_physical_operator> left,
                                   duckdb::unique_ptr<sirius_physical_operator> right,
                                   duckdb::vector<sirius::join_condition> cond,
                                   duckdb::JoinType join_type,
                                   std::size_t estimated_cardinality,
                                   duckdb::vector<std::size_t> left_projection_map,
                                   duckdb::vector<std::size_t> right_projection_map);

  duckdb::vector<sirius::join_condition> conditions;
  //! The types of the join keys
  duckdb::vector<sirius::logical_type> condition_types;
  //! The type of the join
  duckdb::JoinType join_type;

  //! The indices for getting the payload columns
  duckdb::vector<std::size_t> payload_column_idxs;
  //! The types of the payload columns
  duckdb::vector<sirius::logical_type> payload_types;

  //! Positions of the RHS columns that need to output
  duckdb::vector<std::size_t> rhs_output_columns;
  //! The types of the output
  duckdb::vector<sirius::logical_type> rhs_output_types;

  //! Output column order: indices into left table columns (empty = identity 0,1,...,left_cols-1)
  duckdb::vector<std::size_t> left_output_col_idxs;
  //! Output column order: indices into right table columns (empty = identity 0,1,...,right_cols-1)
  duckdb::vector<std::size_t> right_output_col_idxs;

  //! Duplicate eliminated types; only used for delim_joins (i.e. correlated subqueries)
  duckdb::vector<sirius::logical_type> delim_types;

  duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> filter_pushdown;

 protected:
  // CachingOperator Interface

  static void build_join_pipelines(pipeline::sirius_pipeline& current,
                                   pipeline::sirius_meta_pipeline& meta_pipeline,
                                   sirius_physical_operator& op,
                                   bool build_rhs = true);
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

 public:
  // Source interface
  bool is_source() const override { return duckdb::PropagatesBuildSide(join_type); }

 public:
  // Sink Interface
  bool is_sink() const override { return true; }

  static bool is_supported(const duckdb::vector<sirius::join_condition>& conditions,
                           duckdb::JoinType join_type);

 public:
  //! Returns a list of the types of the join conditions
  duckdb::vector<sirius::logical_type> get_join_types() const;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  /// @brief Emit the join-type-correct output for one surviving batch whose other input side
  /// died (s3-shape-c-zero-side-join-plan.md §3/§8). Builds gather maps on the task stream
  /// (iota for keep-all, -1 padding for the NULLed dead side, empty for empty-correct cells)
  /// and publishes through the same output path as a normal join task. The join condition is
  /// never evaluated.
  std::unique_ptr<operator_data> execute_zero_side(const nested_loop_join_zero_side_input& input,
                                                   rmm::cuda_stream_view stream);

  /// @brief Join-type-correct output when one input side has no rows. Shared by the zero-side
  /// task (dead side synthesized as a 0-row table) and the regular execute path receiving a
  /// real 0-row batch (e.g. an all-pruned scan under the empty-split fallback): the preserved
  /// side's rows are padded, kept, or marked false per join type; only the condition
  /// evaluation is skipped.
  std::unique_ptr<operator_data> emit_one_side_empty_result(const cudf::table_view& left,
                                                            const cudf::table_view& right,
                                                            bool left_side_empty,
                                                            cucascade::memory::memory_space& space,
                                                            rmm::cuda_stream_view stream);

 protected:
  std::mutex batches_to_processed_mutex;
  std::size_t current_partition_index = 0;
  std::size_t num_batches_to_process  = 0;
  std::vector<std::vector<uint64_t>> left_batch_ids;
  std::vector<std::vector<uint64_t>> right_batch_ids;

  //! Set under batches_to_processed_mutex exactly where the port snapshot observes batches on
  //! each input side. A side whose flag never flips while every source pipeline finishes is
  //! "dead"; zero_side_pending_locked() then offers the zero-side task for the surviving
  //! port's batches (s3-shape-c-zero-side-join-plan.md §8).
  bool _saw_probe_input = false;
  bool _saw_build_input = false;

  //! True when exactly one input side is dead (never saw input, port empty, source pipeline
  //! finished), the other FULL port still holds data, and BOTH source pipelines finished.
  //! Caller must hold batches_to_processed_mutex.
  bool zero_side_pending_locked();
};

}  // namespace op
}  // namespace sirius
