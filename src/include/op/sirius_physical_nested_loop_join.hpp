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
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "op/sirius_physical_operator.hpp"

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

//! sirius_physical_nested_loop_join represents a nested loop join between two tables
class sirius_physical_nested_loop_join : public sirius_physical_operator {
 public:
  static constexpr const duckdb::PhysicalOperatorType TYPE =
    duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN;

 public:
  sirius_physical_nested_loop_join(
    duckdb::LogicalOperator& op,
    duckdb::unique_ptr<sirius_physical_operator> left,
    duckdb::unique_ptr<sirius_physical_operator> right,
    duckdb::vector<duckdb::JoinCondition> cond,
    duckdb::JoinType join_type,
    duckdb::idx_t estimated_cardinality,
    duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info_p);

  sirius_physical_nested_loop_join(duckdb::LogicalOperator& op,
                                   duckdb::unique_ptr<sirius_physical_operator> left,
                                   duckdb::unique_ptr<sirius_physical_operator> right,
                                   duckdb::vector<duckdb::JoinCondition> cond,
                                   duckdb::JoinType join_type,
                                   duckdb::idx_t estimated_cardinality);

  duckdb::vector<duckdb::JoinCondition> conditions;
  //! The types of the join keys
  duckdb::vector<duckdb::LogicalType> condition_types;
  //! The type of the join
  duckdb::JoinType join_type;

  //! The indices for getting the payload columns
  duckdb::vector<duckdb::idx_t> payload_column_idxs;
  //! The types of the payload columns
  duckdb::vector<duckdb::LogicalType> payload_types;

  //! Positions of the RHS columns that need to output
  duckdb::vector<duckdb::idx_t> rhs_output_columns;
  //! The types of the output
  duckdb::vector<duckdb::LogicalType> rhs_output_types;

  //! Duplicate eliminated types; only used for delim_joins (i.e. correlated subqueries)
  duckdb::vector<duckdb::LogicalType> delim_types;

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
  bool parallel_source() const override { return true; }

 public:
  // Sink Interface
  bool is_sink() const override { return true; }
  bool parallel_sink() const override { return true; }

  static bool is_supported(const duckdb::vector<duckdb::JoinCondition>& conditions,
                           duckdb::JoinType join_type);

 public:
  //! Returns a list of the types of the join conditions
  duckdb::vector<duckdb::LogicalType> get_join_types() const;

 private:
  // // resolve joins that output max N elements (SEMI, ANTI, MARK)
  //   void resolve_simple_join(sirius_intermediate_relation& input_relation,
  //                          sirius_intermediate_relation& output_relation) const;
  // // resolve joins that can potentially output N*M elements (INNER, LEFT, FULL)
  //   duckdb::OperatorResultType resolve_complex_join(sirius_intermediate_relation& input_relation,
  //                                         sirius_intermediate_relation& output_relation) const;
};

}  // namespace op
}  // namespace sirius
