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
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cstdio>
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
  reorder_conditions(conditions);

  children.push_back(std::move(left));
  children.push_back(std::move(right));
  auto& lhs_types = children[0]->get_types();
  auto& rhs_types = children[1]->get_types();
  left_output_col_idxs.reserve(lhs_types.size());
  for (duckdb::idx_t i = 0; i < lhs_types.size(); i++) {
    left_output_col_idxs.push_back(i);
  }
  right_output_col_idxs.reserve(rhs_types.size());
  for (duckdb::idx_t i = 0; i < rhs_types.size(); i++) {
    right_output_col_idxs.push_back(i);
  }
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
  reorder_conditions(conditions);
  children.push_back(std::move(left));
  children.push_back(std::move(right));
  auto& lhs_types = children[0]->get_types();
  auto& rhs_types = children[1]->get_types();
  left_output_col_idxs.reserve(lhs_types.size());
  for (duckdb::idx_t i = 0; i < lhs_types.size(); i++) {
    left_output_col_idxs.push_back(i);
  }
  right_output_col_idxs.reserve(rhs_types.size());
  for (duckdb::idx_t i = 0; i < rhs_types.size(); i++) {
    right_output_col_idxs.push_back(i);
  }
  filter_pushdown = std::move(pushdown_info_p);
}

sirius_physical_nested_loop_join::sirius_physical_nested_loop_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<duckdb::JoinCondition> cond,
  duckdb::JoinType join_type,
  duckdb::idx_t estimated_cardinality,
  duckdb::vector<duckdb::idx_t> left_projection_map,
  duckdb::vector<duckdb::idx_t> right_projection_map)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, op.types, estimated_cardinality),
    join_type(join_type),
    conditions(std::move(cond))
{
  reorder_conditions(conditions);
  children.push_back(std::move(left));
  children.push_back(std::move(right));
  auto& lhs_types = children[0]->get_types();
  auto& rhs_types = children[1]->get_types();
  if (left_projection_map.empty()) {
    for (duckdb::idx_t i = 0; i < lhs_types.size(); i++) {
      left_output_col_idxs.push_back(i);
    }
  } else {
    for (duckdb::idx_t idx : left_projection_map) {
      if (idx < lhs_types.size()) { left_output_col_idxs.push_back(idx); }
    }
  }
  if (right_projection_map.empty()) {
    for (duckdb::idx_t i = 0; i < rhs_types.size(); i++) {
      right_output_col_idxs.push_back(i);
    }
  } else {
    for (duckdb::idx_t idx : right_projection_map) {
      if (idx < rhs_types.size()) { right_output_col_idxs.push_back(idx); }
    }
  }
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

// Resolve table column index: BOUND_REF, BOUND_CAST(BOUND_REF), or BOUND_SUBQUERY (scalar
// subquery result = single column, index 0).
bool get_column_index(const duckdb::Expression& expr, cudf::size_type& out_idx)
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
  nvtx3::scoped_range nvtx_range{"sirius_physical_nested_loop_join::execute"};
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

  cudf::table_view left  = get_cudf_table_view(*left_batch);
  cudf::table_view right = get_cudf_table_view(*right_batch);

  cucascade::memory::memory_space* space = left_batch->get_memory_space();
  if (!space) {
    SIRIUS_LOG_DEBUG(
      "Pipeline {}: nested loop join, 0 output batches because left batch had no memory space",
      pipeline_id);
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto mr = space->get_default_allocator();

  if (left.num_rows() == 0 || right.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    empty_cols.reserve(left_output_col_idxs.size() + right_output_col_idxs.size());
    for (duckdb::idx_t idx : left_output_col_idxs) {
      if (idx < static_cast<duckdb::idx_t>(left.num_columns())) {
        empty_cols.push_back(cudf::make_empty_column(left.column(idx).type()));
      }
    }
    for (duckdb::idx_t idx : right_output_col_idxs) {
      if (idx < static_cast<duckdb::idx_t>(right.num_columns())) {
        empty_cols.push_back(cudf::make_empty_column(right.column(idx).type()));
      }
    }
    auto empty_table = std::make_unique<cudf::table>(std::move(empty_cols), stream, mr);
    SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 empty output batches", pipeline_id);
    return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{
      make_data_batch(std::move(empty_table), *space)});
  }

  std::unique_ptr<cudf::table> result_table;

  if (conditions.empty()) {
    auto cross         = cudf::cross_join(left, right, stream, mr);
    auto left_released = cross->release();
    const auto left_n  = static_cast<duckdb::idx_t>(left.num_columns());
    const auto right_n = static_cast<duckdb::idx_t>(right.num_columns());
    std::vector<std::unique_ptr<cudf::column>> out_cols;
    out_cols.reserve(left_output_col_idxs.size() + right_output_col_idxs.size());
    for (duckdb::idx_t idx : left_output_col_idxs) {
      if (idx < left_n && idx < left_released.size()) {
        out_cols.push_back(std::move(left_released[idx]));
      }
    }
    for (duckdb::idx_t idx : right_output_col_idxs) {
      if (idx < right_n && left_n + idx < left_released.size()) {
        out_cols.push_back(std::move(left_released[left_n + idx]));
      }
    }
    result_table = std::make_unique<cudf::table>(std::move(out_cols), stream, mr);
  } else {
    // Resolve column indices and target types so AST predicate operands match (cudf requires
    // matching types). Columns used in conditions may be cast to the expression return type.
    // Reserve to the exact number of conditions to prevent reallocation.
    // cudf::ast::operation stores operands as reference_wrapper<expression const> — any
    // reallocation of these vectors invalidates the stored references and causes UB/segfault.

    std::map<uint64_t, cudf::size_type> left_expressions_to_idx;
    std::map<uint64_t, cudf::size_type> right_expressions_to_idx;
    std::vector<cudf::ast::column_reference> left_refs;
    std::vector<cudf::ast::column_reference> right_refs;
    std::vector<cudf::ast::operation> cond_ops;
    std::vector<cudf::ast::operation> and_chain;
    left_refs.reserve(conditions.size());
    right_refs.reserve(conditions.size());
    cond_ops.reserve(conditions.size());
    and_chain.reserve(conditions.size() > 1 ? conditions.size() - 1 : 0);
    std::vector<cudf::column_view> left_col_views, right_col_views;
    std::vector<std::unique_ptr<cudf::column>> intermediates_scope_holder;
    std::vector<std::shared_ptr<cucascade::data_batch>> expression_res_scope_hodler;
    left_col_views.reserve(left.num_columns());
    right_col_views.reserve(right.num_columns());

    // Resolves one side of a join condition to a column index in col_views, evaluating or casting
    // as needed. Returns the index to use as the cudf::ast::column_reference offset.
    auto resolve_join_col = [&](const duckdb::Expression& expr,
                                std::map<uint64_t, cudf::size_type>& expr_to_idx,
                                const std::shared_ptr<cucascade::data_batch>& batch,
                                const cudf::table_view& table,
                                std::vector<cudf::column_view>& col_views,
                                const char* side) -> cudf::size_type {
      auto cond_hash = expr.Hash();
      auto it        = expr_to_idx.find(cond_hash);
      if (it != expr_to_idx.end()) { return it->second; }
      cudf::size_type join_input_index = static_cast<cudf::size_type>(col_views.size());
      expr_to_idx[cond_hash]           = join_input_index;
      cudf::size_type source_idx       = 0;
      if (!get_column_index(expr, source_idx)) {
        duckdb::sirius::GpuExpressionExecutor executor(expr, mr);
        auto expr_result_batch = executor.execute(batch, stream);
        auto& expr_table =
          expr_result_batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
        auto expr_view = expr_table.view();
        if (expr_view.num_columns() != 1) {
          throw std::runtime_error(std::string("sirius_physical_nested_loop_join: expression on ") +
                                   side + " should produce one column");
        }
        if (expr_view.num_rows() != table.num_rows()) {
          throw std::runtime_error(
            std::string(
              "sirius_physical_nested_loop_join: expression result row count must match ") +
            side + " table");
        }
        col_views.push_back(expr_view.column(0));
        expression_res_scope_hodler.push_back(std::move(expr_result_batch));
      } else {
        auto target_type = duckdb::GetCudfType(expr.return_type);

        // now lets see if we have to cast
        if (table.column(source_idx).type() != target_type) {
          if (expr.expression_class != duckdb::ExpressionClass::BOUND_CAST) {
            // We might want to just change this to an ASSERT
            throw std::runtime_error(
              "sirius_physical_nested_loop_join: unexpected, column type does not match, yet "
              "there "
              "is no BOUND_CAST");
          }
          intermediates_scope_holder.push_back(
            cudf::cast(table.column(source_idx), target_type, stream));
          col_views.push_back(intermediates_scope_holder.back()->view());
        } else {
          col_views.push_back(table.column(source_idx));
        }
      }
      return join_input_index;
    };

    for (const auto& cond : conditions) {
      cudf::size_type left_join_input_index = resolve_join_col(
        *cond.left, left_expressions_to_idx, left_batch, left, left_col_views, "left");
      cudf::size_type right_join_input_index = resolve_join_col(
        *cond.right, right_expressions_to_idx, right_batch, right, right_col_views, "right");

      left_refs.emplace_back(left_join_input_index, cudf::ast::table_reference::LEFT);
      right_refs.emplace_back(right_join_input_index, cudf::ast::table_reference::RIGHT);
      cond_ops.emplace_back(to_ast_operator(cond.comparison), left_refs.back(), right_refs.back());
    }

    cudf::table_view left_effective(left_col_views);
    cudf::table_view right_effective(right_col_views);

    // Build a left-associative AND chain referencing cond_ops elements directly — never copying
    // operations, matching the cuDF test pattern. and_chain holds exactly (N-1) LOGICAL_AND nodes
    // for N conditions; cond_ops[0] is the left leaf of the first AND node, not copied into
    // and_chain.
    for (size_t i = 1; i < cond_ops.size(); i++) {
      const cudf::ast::expression& lhs =
        (i == 1) ? static_cast<const cudf::ast::expression&>(cond_ops[0]) : and_chain.back();
      and_chain.emplace_back(cudf::ast::ast_operator::LOGICAL_AND,
                             lhs,
                             static_cast<const cudf::ast::expression&>(cond_ops[i]));
    }
    const cudf::ast::expression& predicate =
      and_chain.empty() ? static_cast<const cudf::ast::expression&>(cond_ops[0]) : and_chain.back();

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
    out_cols.reserve(left_output_col_idxs.size() + right_output_col_idxs.size());
    for (duckdb::idx_t idx : left_output_col_idxs) {
      if (idx < left_released.size()) { out_cols.push_back(std::move(left_released[idx])); }
    }
    for (duckdb::idx_t idx : right_output_col_idxs) {
      if (idx < right_released.size()) { out_cols.push_back(std::move(right_released[idx])); }
    }
    result_table = std::make_unique<cudf::table>(std::move(out_cols), stream, mr);
  }

  SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
  return std::make_unique<operator_data>(std::vector<std::shared_ptr<cucascade::data_batch>>{
    make_data_batch(std::move(result_table), *space)});
}

}  // namespace op
}  // namespace sirius
