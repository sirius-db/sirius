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

#include "op/sirius_physical_nested_loop_join.hpp"

#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/nested_loop_join.hpp"
#include "duckdb/execution/operator/join/outer_join_marker.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {
namespace op {

void reorder_conditions(duckdb::vector<duckdb::JoinCondition>& conditions)
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

sirius_physical_nested_loop_join::sirius_physical_nested_loop_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<duckdb::JoinCondition> cond,
  duckdb::JoinType join_type,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN, op.types, estimated_cardinality),
    join_type(join_type),
    conditions(std::move(cond))
{
  // conditions.resize(cond.size());
  // duckdb::idx_t equal_position = 0;
  // duckdb::idx_t other_position = cond.size() - 1;
  // for (duckdb::idx_t i = 0; i < cond.size(); i++) {
  //   if (cond[i].comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
  //       cond[i].comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
  //     conditions[equal_position++] = std::move(cond[i]);
  //   } else {
  //     conditions[other_position--] = std::move(cond[i]);
  //   }
  // }
  reorder_conditions(conditions);
  children.push_back(std::move(left));
  children.push_back(std::move(right));

  // right_temp_data =
  // duckdb::make_shared_ptr<GPUIntermediateRelation>(children[1]->get_types().size());
}

sirius_physical_nested_loop_join::sirius_physical_nested_loop_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<duckdb::JoinCondition> cond,
  duckdb::JoinType join_type,
  duckdb::idx_t estimated_cardinality,
  duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info_p)
  : sirius_physical_operator(
      duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN, op.types, estimated_cardinality),
    join_type(join_type),
    conditions(std::move(cond))
{
  // conditions.resize(cond.size());
  // duckdb::idx_t equal_position = 0;
  // duckdb::idx_t other_position = cond.size() - 1;
  // for (duckdb::idx_t i = 0; i < cond.size(); i++) {
  //   if (cond[i].comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
  //       cond[i].comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
  //     conditions[equal_position++] = std::move(cond[i]);
  //   } else {
  //     conditions[other_position--] = std::move(cond[i]);
  //   }
  // }
  reorder_conditions(conditions);
  filter_pushdown = std::move(pushdown_info_p);
  children.push_back(std::move(left));
  children.push_back(std::move(right));
  // right_temp_data =
  // duckdb::make_shared_ptr<GPUIntermediateRelation>(children[1]->get_types().size());
}

bool sirius_physical_nested_loop_join::is_supported(
  const duckdb::vector<duckdb::JoinCondition>& conditions, duckdb::JoinType join_type)
{
  if (join_type == duckdb::JoinType::MARK) { return true; }
  for (auto& cond : conditions) {
    if (cond.left->return_type.InternalType() == duckdb::PhysicalType::STRUCT ||
        cond.left->return_type.InternalType() == duckdb::PhysicalType::LIST ||
        cond.left->return_type.InternalType() == duckdb::PhysicalType::ARRAY) {
      return false;
    }
  }
  if (join_type == duckdb::JoinType::SEMI || join_type == duckdb::JoinType::ANTI) {
    return conditions.size() == 1;
  }
  return true;
}

duckdb::vector<duckdb::LogicalType> sirius_physical_nested_loop_join::get_join_types() const
{
  duckdb::vector<duckdb::LogicalType> result;
  for (auto& op : conditions) {
    result.push_back(op.right->return_type);
  }
  return result;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_nested_loop_join::build_join_pipelines(
  pipeline::sirius_pipeline& current,
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
    case duckdb::PhysicalOperatorType::POSITIONAL_JOIN:
      throw duckdb::NotImplementedException("POSITIONAL_JOIN is not implemented yet");
      meta_pipeline.create_child_pipeline(current, op, last_pipeline);
      return;
    case duckdb::PhysicalOperatorType::CROSS_PRODUCT:
      throw duckdb::NotImplementedException("CROSS_PRODUCT is not implemented yet");
      return;
    default: break;
  }

  bool add_child_pipeline = false;
  auto& join_op           = op.Cast<sirius_physical_nested_loop_join>();
  if (join_op.is_source()) { add_child_pipeline = true; }

  if (add_child_pipeline) { meta_pipeline.create_child_pipeline(current, op, last_pipeline); }
}

void sirius_physical_nested_loop_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  sirius_physical_nested_loop_join::build_join_pipelines(current, meta_pipeline, *this);
}

}  // namespace op
}  // namespace sirius
