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

#include "op/sirius_physical_hash_join.hpp"

#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {
namespace op {

void reorder_join_conditions(duckdb::vector<duckdb::JoinCondition>& conditions)
{
  bool is_ordered     = true;
  bool seen_non_equal = false;
  for (auto& cond : conditions) {
    if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
        cond.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      if (seen_non_equal) {
        is_ordered = false;
        break;
      }
    } else {
      seen_non_equal = true;
    }
  }
  if (is_ordered) { return; }
  duckdb::vector<duckdb::JoinCondition> equal_conditions;
  duckdb::vector<duckdb::JoinCondition> other_conditions;
  for (auto& cond : conditions) {
    if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
        cond.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      equal_conditions.push_back(std::move(cond));
    } else {
      other_conditions.push_back(std::move(cond));
    }
  }
  conditions.clear();
  for (auto& cond : equal_conditions) {
    conditions.push_back(std::move(cond));
  }
  for (auto& cond : other_conditions) {
    conditions.push_back(std::move(cond));
  }
}

sirius_physical_hash_join::sirius_physical_hash_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<duckdb::JoinCondition> cond,
  duckdb::JoinType join_type,
  const duckdb::vector<duckdb::idx_t>& left_projection_map,
  const duckdb::vector<duckdb::idx_t>& right_projection_map,
  duckdb::vector<duckdb::LogicalType> delim_types,
  duckdb::idx_t estimated_cardinality,
  duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::HASH_JOIN, op.types, estimated_cardinality),
    join_type(join_type),
    conditions(std::move(cond)),
    delim_types(std::move(delim_types))
{
  reorder_join_conditions(conditions);

  filter_pushdown = std::move(pushdown_info_p);

  children.push_back(std::move(left));
  children.push_back(std::move(right));

  duckdb::unordered_map<duckdb::idx_t, duckdb::idx_t> build_columns_in_conditions;
  for (duckdb::idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    auto& condition = conditions[cond_idx];
    condition_types.push_back(condition.left->return_type);
    if (condition.right->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
      build_columns_in_conditions.emplace(
        condition.right->Cast<duckdb::BoundReferenceExpression>().index, cond_idx);
    }
  }

  auto& lhs_input_types = children[0]->get_types();

  lhs_output_columns.col_idxs = left_projection_map;
  if (lhs_output_columns.col_idxs.empty()) {
    lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
    for (duckdb::idx_t i = 0; i < lhs_input_types.size(); i++) {
      lhs_output_columns.col_idxs.emplace_back(i);
    }
  }

  for (auto& lhs_col : lhs_output_columns.col_idxs) {
    auto& lhs_col_type = lhs_input_types[lhs_col];
    lhs_output_columns.col_types.push_back(lhs_col_type);
  }

  if (join_type == duckdb::JoinType::ANTI || join_type == duckdb::JoinType::SEMI ||
      join_type == duckdb::JoinType::MARK) {
    // materialized_build_key =
    //   duckdb::make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
    // hash_table_result =
    //   duckdb::make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
    return;
  }

  auto& rhs_input_types = children[1]->get_types();

  auto right_projection_map_copy = right_projection_map;
  if (right_projection_map_copy.empty()) {
    right_projection_map_copy.reserve(rhs_input_types.size());
    for (duckdb::idx_t i = 0; i < rhs_input_types.size(); i++) {
      right_projection_map_copy.emplace_back(i);
    }
  }

  for (auto& rhs_col : right_projection_map_copy) {
    auto& rhs_col_type = rhs_input_types[rhs_col];

    auto it = build_columns_in_conditions.find(rhs_col);
    if (it == build_columns_in_conditions.end()) {
      payload_columns.col_idxs.push_back(rhs_col);
      payload_columns.col_types.push_back(rhs_col_type);
      rhs_output_columns.col_idxs.push_back(condition_types.size() +
                                            payload_columns.col_types.size() - 1);
    } else {
      rhs_output_columns.col_idxs.push_back(it->second);
    }
    rhs_output_columns.col_types.push_back(rhs_col_type);
  }

  // hash_table_result =
  // duckdb::make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size() +
  //                                                              payload_columns.col_idxs.size());
  // materialized_build_key =
  //   duckdb::make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
};

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_hash_join::build_join_pipelines(pipeline::sirius_pipeline& current,
                                                     pipeline::sirius_meta_pipeline& meta_pipeline,
                                                     sirius_physical_operator& op,
                                                     bool build_rhs)
{
  op.op_state.reset();
  op.sink_state.reset();

  auto& state = meta_pipeline.get_state();
  state.add_pipeline_operator(current, op);

  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> pipelines_so_far;
  meta_pipeline.get_pipelines(pipelines_so_far, false);
  auto& last_pipeline = *pipelines_so_far.back();

  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> dependencies;
  duckdb::optional_ptr<pipeline::sirius_meta_pipeline> last_child_ptr;
  if (build_rhs) {
    // on the RHS (build side), we construct a child MetaPipeline with this operator as its sink
    auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, op);
    child_meta_pipeline.build(*op.children[1]);
    // if (op.children[1].get().CanSaturateThreads(current.GetClientContext())) {
    // 	// if the build side can saturate all available threads,
    // 	// we don't just make the LHS pipeline depend on the RHS, but recursively all LHS children
    // too.
    // 	// this prevents breadth-first plan evaluation
    // 	child_meta_pipeline.GetPipelines(dependencies, false);
    // 	last_child_ptr = meta_pipeline.GetLastChild();
    // }
  }

  op.children[0]->build_pipelines(current, meta_pipeline);

  // if (last_child_ptr) {
  // 	// the pointer was set, set up the dependencies
  // 	meta_pipeline.add_recursive_dependencies(dependencies, *last_child_ptr);
  // }

  switch (op.type) {
    case SiriusPhysicalOperatorType::POSITIONAL_JOIN:
      throw duckdb::NotImplementedException("POSITIONAL_JOIN is not implemented yet");
      meta_pipeline.create_child_pipeline(current, op, last_pipeline);
      return;
    case SiriusPhysicalOperatorType::CROSS_PRODUCT:
      throw duckdb::NotImplementedException("CROSS_PRODUCT is not implemented yet");
      return;
    default: break;
  }

  bool add_child_pipeline = false;
  auto& join_op           = op.Cast<sirius_physical_hash_join>();
  if (join_op.is_source()) { add_child_pipeline = true; }

  if (add_child_pipeline) { meta_pipeline.create_child_pipeline(current, op, last_pipeline); }
}

void sirius_physical_hash_join::build_pipelines(pipeline::sirius_pipeline& current,
                                                pipeline::sirius_meta_pipeline& meta_pipeline)
{
  sirius_physical_hash_join::build_join_pipelines(current, meta_pipeline, *this);
}

}  // namespace op
}  // namespace sirius
