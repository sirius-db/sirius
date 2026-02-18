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

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
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

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <unordered_map>

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
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, op.types, estimated_cardinality),
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
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, op.types, estimated_cardinality),
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
  auto& join_op           = op.Cast<sirius_physical_nested_loop_join>();
  if (join_op.is_source()) { add_child_pipeline = true; }

  if (add_child_pipeline) { meta_pipeline.create_child_pipeline(current, op, last_pipeline); }
}

void sirius_physical_nested_loop_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  sirius_physical_nested_loop_join::build_join_pipelines(current, meta_pipeline, *this);
}

std::unique_ptr<operator_data> sirius_physical_nested_loop_join::get_next_task_input_data()
{
  size_t batch_index = 0;
  {
    std::lock_guard<std::mutex> lg(batches_to_processed_mutex);
    if (left_batch_ids.empty() && right_batch_ids.empty()) {
      auto* default_port = get_port("default");
      auto* build_port   = get_port("build");
      if (!default_port || !default_port->repo || !build_port || !build_port->repo) {
        return nullptr;
      }
      if (default_port->repo->num_partitions() != build_port->repo->num_partitions()) {
        throw std::runtime_error(
          "sirius_physical_nested_loop_join: number of partitions for default and build ports must "
          "match");
      }
      left_batch_ids.reserve(default_port->repo->num_partitions());
      right_batch_ids.reserve(build_port->repo->num_partitions());
      for (size_t i = 0; i < default_port->repo->num_partitions(); i++) {
        left_batch_ids.push_back(default_port->repo->get_batch_ids(i));
        right_batch_ids.push_back(build_port->repo->get_batch_ids(i));
        num_batches_to_process += left_batch_ids[i].size() * right_batch_ids[i].size();
      }
    }
    if (current_partition_index < num_batches_to_process) {
      batch_index = current_partition_index;
      current_partition_index++;
    } else {
      return nullptr;
    }
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
  input_batch.reserve(2);
  size_t counter     = 0;
  auto* default_port = get_port("default");
  auto* build_port   = get_port("build");
  for (size_t partition_idx = 0; partition_idx < left_batch_ids.size(); partition_idx++) {
    size_t left_counter = 0;
    for (auto& left_batch_id : left_batch_ids[partition_idx]) {
      size_t right_counter = 0;
      for (auto& right_batch_id : right_batch_ids[partition_idx]) {
        if (counter == batch_index) {
          if (right_counter == right_batch_ids[partition_idx].size() - 1) {
            input_batch.push_back(default_port->repo->pop_data_batch_by_id(
              left_batch_id, cucascade::batch_state::task_created, partition_idx));
          } else {
            input_batch.push_back(default_port->repo->get_data_batch_by_id(
              left_batch_id, cucascade::batch_state::task_created, partition_idx));
          }
          if (left_counter == left_batch_ids[partition_idx].size() - 1) {
            input_batch.push_back(build_port->repo->pop_data_batch_by_id(
              right_batch_id, cucascade::batch_state::task_created, partition_idx));
          } else {
            input_batch.push_back(build_port->repo->get_data_batch_by_id(
              right_batch_id, cucascade::batch_state::task_created, partition_idx));
          }
          return std::make_unique<operator_data>(input_batch);
        }
        right_counter++;
        counter++;
      }
      left_counter++;
    }
  }
  return nullptr;
}

namespace {

cudf::ast::ast_operator to_ast_operator(duckdb::ExpressionType comparison)
{
  switch (comparison) {
    case duckdb::ExpressionType::COMPARE_EQUAL: return cudf::ast::ast_operator::EQUAL;
    case duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM:
      return cudf::ast::ast_operator::NULL_EQUAL;
    case duckdb::ExpressionType::COMPARE_NOTEQUAL:
    case duckdb::ExpressionType::COMPARE_DISTINCT_FROM: return cudf::ast::ast_operator::NOT_EQUAL;
    case duckdb::ExpressionType::COMPARE_LESSTHAN: return cudf::ast::ast_operator::LESS;
    case duckdb::ExpressionType::COMPARE_GREATERTHAN: return cudf::ast::ast_operator::GREATER;
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
      return cudf::ast::ast_operator::LESS_EQUAL;
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      return cudf::ast::ast_operator::GREATER_EQUAL;
    default:
      throw std::runtime_error("sirius_physical_nested_loop_join: unsupported comparison type");
  }
}

// Resolve left-table column index: BOUND_REF or BOUND_CAST(BOUND_REF).
bool get_left_column_index(const duckdb::Expression& expr, cudf::size_type& out_idx)
{
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_REF) {
    out_idx = static_cast<cudf::size_type>(expr.Cast<duckdb::BoundReferenceExpression>().index);
    return true;
  }
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    const auto& cast_expr = expr.Cast<duckdb::BoundCastExpression>();
    if (cast_expr.child->expression_class == duckdb::ExpressionClass::BOUND_REF) {
      out_idx = static_cast<cudf::size_type>(
        cast_expr.child->Cast<duckdb::BoundReferenceExpression>().index);
      return true;
    }
  }
  return false;
}

// Resolve right-table column index: BOUND_REF, BOUND_CAST(BOUND_REF), or BOUND_SUBQUERY (scalar
// subquery result = single column, index 0).
bool get_right_column_index(const duckdb::Expression& expr, cudf::size_type& out_idx)
{
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_REF) {
    out_idx = static_cast<cudf::size_type>(expr.Cast<duckdb::BoundReferenceExpression>().index);
    return true;
  }
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    const auto& cast_expr = expr.Cast<duckdb::BoundCastExpression>();
    if (cast_expr.child->expression_class == duckdb::ExpressionClass::BOUND_REF) {
      out_idx = static_cast<cudf::size_type>(
        cast_expr.child->Cast<duckdb::BoundReferenceExpression>().index);
      return true;
    }
  }
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_SUBQUERY) {
    out_idx = 0;
    return true;
  }
  return false;
}

}  // namespace

std::unique_ptr<operator_data> sirius_physical_nested_loop_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  const auto& input_batches = input_data.get_data_batches();
  size_t pipeline_id = (this->get_pipeline() != nullptr) ? this->get_pipeline()->get_pipeline_id()
                                                         : static_cast<size_t>(-1);
  SIRIUS_LOG_DEBUG(
    "Pipeline {}: nested loop join, {} input batches", pipeline_id, input_batches.size());

  if (input_batches.size() != 2) {
    throw std::runtime_error(
      "sirius_physical_nested_loop_join expects 2 input batches (left, right), got " +
      std::to_string(input_batches.size()));
  }

  auto left_batch  = input_batches[0];
  auto right_batch = input_batches[1];
  if (!left_batch || !right_batch) {
    SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 0 output batches", pipeline_id);
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  cudf::table_view left                  = get_cudf_table_view(*left_batch);
  cudf::table_view right                 = get_cudf_table_view(*right_batch);
  cucascade::memory::memory_space* space = left_batch->get_memory_space();
  if (!space) {
    SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 0 output batches", pipeline_id);
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto mr = space->get_default_allocator();

  if (left.num_rows() == 0 || right.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    for (cudf::size_type c = 0; c < left.num_columns(); c++) {
      empty_cols.push_back(cudf::make_empty_column(left.column(c).type()));
    }
    for (cudf::size_type c = 0; c < right.num_columns(); c++) {
      empty_cols.push_back(cudf::make_empty_column(right.column(c).type()));
    }
    auto empty_table = std::make_unique<cudf::table>(std::move(empty_cols), stream, mr);
    SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{
      make_data_batch(std::move(empty_table), *space)});
  }

  std::unique_ptr<cudf::table> result_table;

  if (conditions.empty()) {
    result_table = cudf::cross_join(left, right, stream, mr);
  } else {
    // Resolve column indices and target types so AST predicate operands match (cudf requires
    // matching types). Columns used in conditions may be cast to the expression return type.
    std::vector<cudf::ast::column_reference> left_refs;
    std::vector<cudf::ast::column_reference> right_refs;
    std::vector<cudf::ast::operation> cond_ops;
    std::unordered_map<cudf::size_type, cudf::data_type> left_target_type;
    std::unordered_map<cudf::size_type, cudf::data_type> right_target_type;
    for (const auto& cond : conditions) {
      cudf::size_type left_idx  = 0;
      cudf::size_type right_idx = 0;
      if (!get_left_column_index(*cond.left, left_idx)) {
        throw std::runtime_error(
          "sirius_physical_nested_loop_join: left side of condition must be a column reference or "
          "CAST(column) (got: " +
          cond.left->ToString() + ")");
      }
      if (!get_right_column_index(*cond.right, right_idx)) {
        throw std::runtime_error(
          "sirius_physical_nested_loop_join: right side of condition must be a column reference, "
          "CAST(column), or scalar SUBQUERY (got: " +
          cond.right->ToString() + ")");
      }
      left_target_type[left_idx]   = duckdb::GetCudfType(cond.left->return_type);
      right_target_type[right_idx] = duckdb::GetCudfType(cond.right->return_type);
      left_refs.emplace_back(left_idx, cudf::ast::table_reference::LEFT);
      right_refs.emplace_back(right_idx, cudf::ast::table_reference::RIGHT);
      cond_ops.emplace_back(to_ast_operator(cond.comparison), left_refs.back(), right_refs.back());
    }

    // Build left/right table views with cast columns where type != target (so AST operands match).
    std::vector<cudf::column_view> left_col_views;
    std::vector<cudf::column_view> right_col_views;
    std::vector<std::unique_ptr<cudf::column>> owned_left_casts;
    std::vector<std::unique_ptr<cudf::column>> owned_right_casts;
    left_col_views.reserve(left.num_columns());
    right_col_views.reserve(right.num_columns());
    for (cudf::size_type c = 0; c < left.num_columns(); c++) {
      auto it = left_target_type.find(c);
      if (it != left_target_type.end() && left.column(c).type() != it->second) {
        owned_left_casts.push_back(cudf::cast(left.column(c), it->second, stream));
        left_col_views.push_back(owned_left_casts.back()->view());
      } else {
        left_col_views.push_back(left.column(c));
      }
    }
    for (cudf::size_type c = 0; c < right.num_columns(); c++) {
      auto it = right_target_type.find(c);
      if (it != right_target_type.end() && right.column(c).type() != it->second) {
        owned_right_casts.push_back(cudf::cast(right.column(c), it->second, stream));
        right_col_views.push_back(owned_right_casts.back()->view());
      } else {
        right_col_views.push_back(right.column(c));
      }
    }
    cudf::table_view left_effective(left_col_views);
    cudf::table_view right_effective(right_col_views);

    std::vector<cudf::ast::operation> and_chain;
    and_chain.push_back(std::move(cond_ops[0]));
    for (size_t i = 1; i < cond_ops.size(); i++) {
      and_chain.emplace_back(cudf::ast::ast_operator::BITWISE_AND, and_chain.back(), cond_ops[i]);
    }
    const cudf::ast::expression& predicate = and_chain.back();

    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
      join_result;

    switch (join_type) {
      case duckdb::JoinType::INNER:
        join_result = cudf::conditional_inner_join(
          left_effective, right_effective, predicate, std::nullopt, stream, mr);
        break;
      case duckdb::JoinType::LEFT:
        join_result = cudf::conditional_left_join(
          left_effective, right_effective, predicate, std::nullopt, stream, mr);
        break;
      case duckdb::JoinType::RIGHT:
        join_result = cudf::conditional_left_join(
          right_effective, left_effective, predicate, std::nullopt, stream, mr);
        std::swap(join_result.first, join_result.second);
        break;
      case duckdb::JoinType::SEMI: {
        auto left_indices = cudf::conditional_left_semi_join(
          left_effective, right_effective, predicate, std::nullopt, stream, mr);
        auto left_map = cudf::column_view(cudf::data_type(cudf::type_id::INT32),
                                          left_indices->size(),
                                          left_indices->data(),
                                          nullptr,
                                          0,
                                          0,
                                          {});
        auto gathered =
          cudf::gather(left, left_map, cudf::out_of_bounds_policy::NULLIFY, stream, mr);
        SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
        return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{
          make_data_batch(std::move(gathered), *space)});
      }
      case duckdb::JoinType::ANTI: {
        auto left_indices = cudf::conditional_left_anti_join(
          left_effective, right_effective, predicate, std::nullopt, stream, mr);
        auto left_map = cudf::column_view(cudf::data_type(cudf::type_id::INT32),
                                          left_indices->size(),
                                          left_indices->data(),
                                          nullptr,
                                          0,
                                          0,
                                          {});
        auto gathered =
          cudf::gather(left, left_map, cudf::out_of_bounds_policy::NULLIFY, stream, mr);
        SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
        return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{
          make_data_batch(std::move(gathered), *space)});
      }
      case duckdb::JoinType::OUTER:
        join_result =
          cudf::conditional_full_join(left_effective, right_effective, predicate, stream, mr);
        break;
      default:
        throw std::runtime_error("sirius_physical_nested_loop_join: unsupported join type: " +
                                 duckdb::JoinTypeToString(join_type));
    }

    std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices =
      std::move(join_result.first);
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_indices =
      std::move(join_result.second);
    cudf::column_view left_map_view(cudf::data_type(cudf::type_id::INT32),
                                    left_indices->size(),
                                    left_indices->data(),
                                    nullptr,
                                    0,
                                    0,
                                    {});
    cudf::column_view right_map_view(cudf::data_type(cudf::type_id::INT32),
                                     right_indices->size(),
                                     right_indices->data(),
                                     nullptr,
                                     0,
                                     0,
                                     {});
    auto left_out_of_bounds =
      (join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::OUTER)
        ? cudf::out_of_bounds_policy::NULLIFY
        : cudf::out_of_bounds_policy::DONT_CHECK;
    auto right_out_of_bounds =
      (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::OUTER)
        ? cudf::out_of_bounds_policy::NULLIFY
        : cudf::out_of_bounds_policy::DONT_CHECK;

    auto left_gathered  = cudf::gather(left, left_map_view, left_out_of_bounds, stream, mr);
    auto right_gathered = cudf::gather(right, right_map_view, right_out_of_bounds, stream, mr);
    std::vector<std::unique_ptr<cudf::column>> out_cols;
    auto left_released  = left_gathered->release();
    auto right_released = right_gathered->release();
    for (auto& col : left_released) {
      out_cols.push_back(std::move(col));
    }
    for (auto& col : right_released) {
      out_cols.push_back(std::move(col));
    }
    result_table = std::make_unique<cudf::table>(std::move(out_cols), stream, mr);
  }

  SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
  return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{
    make_data_batch(std::move(result_table), *space)});
}

}  // namespace op
}  // namespace sirius
