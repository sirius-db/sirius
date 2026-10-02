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
#include "sirius_config.hpp"

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

//! Task input of a cross product: a left and a right batch, and which of `num_slices` equal row
//! ranges of the left batch to join with the right batch.
class cross_join_slice_data : public pipelineable_operator_data {
 public:
  cross_join_slice_data(std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches,
                        std::size_t slice,
                        std::size_t num_slices)
    : pipelineable_operator_data(std::move(data_batches)), slice(slice), num_slices(num_slices)
  {
  }

  [[nodiscard]] std::unique_ptr<pipelineable_operator_data> with_data_batches(
    std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches) const override
  {
    return std::make_unique<cross_join_slice_data>(std::move(data_batches), slice, num_slices);
  }

  std::size_t slice;
  std::size_t num_slices;
};

//! sirius_physical_nested_loop_join represents a nested loop join between two tables
class sirius_physical_nested_loop_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::NESTED_LOOP_JOIN;

 public:
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

 protected:
  // CachingOperator Interface

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

 public:
  // Source interface
  //! Always a source: every join emits output.
  bool is_source() const override { return true; }

 public:
  //! True when this NLJ is the internal `delim.join` of a RIGHT_DELIM_JOIN; see the
  //! identical field on `sirius_physical_hash_join`.
  [[nodiscard]] bool is_delim_join_inner() const noexcept { return _is_delim_join_inner; }
  void set_delim_join_inner(bool value) noexcept { _is_delim_join_inner = value; }
  [[nodiscard]] std::string_view input_port_for(
    sirius_physical_operator const& producer) const override;

  // Sink Interface
  //! The inner join of a RIGHT_DELIM_JOIN is never a sink; otherwise the base rule
  //! applies. Mirrors HJ.
  bool is_sink() const override
  {
    if (_is_delim_join_inner) { return false; }
    return sirius_physical_operator::is_sink();
  }

 protected:
  bool _is_delim_join_inner = false;

 public:
  static bool is_supported(const duckdb::vector<sirius::join_condition>& conditions,
                           duckdb::JoinType join_type);

  //! Whether `execute` has an arm for @p join_type. DuckDB's PhysicalNestedLoopJoin::IsSupported
  //! is broader -- RIGHT_SEMI / RIGHT_ANTI / SINGLE pass it -- so the planner screens on this
  //! too, turning a mid-query throw into a plan-time rejection that falls back to CPU.
  static bool is_join_type_supported(duckdb::JoinType join_type);

 public:
  //! Returns a list of the types of the join conditions
  duckdb::vector<sirius::logical_type> get_join_types() const;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Output bytes a cross product task aims for. A pair of input batches whose cross join is
  //! larger is split into tasks over row ranges of the left batch.
  void set_cross_join_task_bytes(uint64_t bytes) noexcept { cross_join_task_bytes = bytes; }

  //! A nested-loop join always runs on a single partition (its build side is not hash-partitioned),
  //! so it never broadcasts or enters build-probe.
  partition_strategy get_partition_strategy(const partition_sizing_input& in) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  /// @brief Join-type-correct output when one input side has no rows. Invoked by the regular
  /// execute path when it receives a real 0-row batch (e.g. an all-pruned scan under the
  /// empty-split fallback): the preserved side's rows are padded, kept, or marked false per
  /// join type; only the condition evaluation is skipped.
  std::unique_ptr<operator_data> emit_one_side_empty_result(const cudf::table_view& left,
                                                            const cudf::table_view& right,
                                                            bool left_side_empty,
                                                            cucascade::memory::memory_space& space,
                                                            ::cuda::stream_ref stream);

  //! Left table restricted to left_output_col_idxs (the plan's left projection map). Applied
  //! by the left-only output paths (SEMI/ANTI/MARK), whose result must match op.types;
  //! selection drops columns only, so row indices from joins on the full table stay valid.
  cudf::table_view select_left_output(const cudf::table_view& left) const;

 protected:
  std::mutex batches_to_processed_mutex;
  std::size_t current_partition_index = 0;
  std::size_t num_batches_to_process  = 0;
  std::vector<std::vector<uint64_t>> left_batch_ids;
  std::vector<std::vector<uint64_t>> right_batch_ids;

  //! One task: a pair of input batches and, for a cross product, a slice of the left batch.
  struct pair_task {
    std::size_t partition;
    std::size_t left;   //!< index into left_batch_ids[partition]
    std::size_t right;  //!< index into right_batch_ids[partition]
    std::size_t slice;
    std::size_t num_slices;
  };
  //! Every task, in the order they are handed out. Built with the batch ID snapshot.
  std::vector<pair_task> pair_tasks;
  uint64_t cross_join_task_bytes = config::DEFAULT_BATCH_SIZE;
};

}  // namespace op
}  // namespace sirius
