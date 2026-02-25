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

#include "cudf/copying.hpp"
#include "cudf/join/filtered_join.hpp"
#include "cudf/join/join.hpp"
#include "cudf/join/mixed_join.hpp"
#include "cudf/table/table_view.hpp"
#include "cudf/types.hpp"
#include "cudf/unary.hpp"
#include "cudf/utilities/memory_resource.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "expression_executor/gpu_expression_translator.hpp"
#include "log/logging.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cstdio>
#include <unordered_set>

namespace sirius {
namespace op {

/// Recursively collect all BoundReferenceExpression indices from an expression tree.
static void collect_bound_ref_indices(duckdb::Expression& expr,
                                      std::unordered_set<duckdb::idx_t>& indices)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    indices.insert(expr.Cast<duckdb::BoundReferenceExpression>().index);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expr, [&](duckdb::Expression& child) { collect_bound_ref_indices(child, indices); });
}

bool sirius_physical_hash_join::are_conditions_supported(
  duckdb::vector<duckdb::JoinCondition>& conditions)
{
  // Must have at least one equality condition for a hash-based join.
  bool has_equality = false;
  for (auto const& cond : conditions) {
    if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
        cond.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      has_equality = true;
      break;
    }
  }
  if (!has_equality) { return false; }

  // Pure equality join: always supported.
  bool has_inequality = false;
  for (auto const& cond : conditions) {
    if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL &&
        cond.comparison != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      has_inequality = true;
      break;
    }
  }
  if (!has_inequality) { return true; }

  // Mixed join: collect the column indices used on each side of the equality conditions.
  std::unordered_set<duckdb::idx_t> equality_left_cols, equality_right_cols;
  for (auto const& cond : conditions) {
    if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL &&
        cond.comparison != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      continue;
    }
    collect_bound_ref_indices(*cond.left, equality_left_cols);
    collect_bound_ref_indices(*cond.right, equality_right_cols);
  }

  // For each inequality condition, verify that its left/right column references don't overlap
  // with the equality key columns on the same side. cuDF's mixed_join API requires the equality
  // and conditional table columns to be disjoint.
  for (auto const& cond : conditions) {
    if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
        cond.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      continue;
    }
    std::unordered_set<duckdb::idx_t> ineq_left_cols, ineq_right_cols;
    collect_bound_ref_indices(*cond.left, ineq_left_cols);
    collect_bound_ref_indices(*cond.right, ineq_right_cols);
    for (auto const idx : ineq_left_cols) {
      if (equality_left_cols.count(idx) > 0) { return false; }
    }
    for (auto const idx : ineq_right_cols) {
      if (equality_right_cols.count(idx) > 0) { return false; }
    }
  }

  return true;
}

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
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::HASH_JOIN, op.types, estimated_cardinality),
    join_type(join_type),
    conditions(std::move(cond)),
    delim_types(std::move(delim_types))
{
  reorder_join_conditions(conditions);

  filter_pushdown = std::move(pushdown_info_p);

  children.push_back(std::move(left));
  children.push_back(std::move(right));

  auto& lhs_input_types = children[0]->get_types();

  if (left_projection_map.empty()) {
    lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
    for (duckdb::idx_t i = 0; i < lhs_input_types.size(); i++) {
      lhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(i));
    }
  } else {
    lhs_output_columns.col_idxs.reserve(left_projection_map.size());
    for (auto& col_idx : left_projection_map) {
      if (col_idx < lhs_input_types.size()) {
        lhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(col_idx));
      } else {
        printf("WARNING:In sirius_physical_hash_join: left_projection_map index out of range");
      }
    }
  }

  for (auto& lhs_col : lhs_output_columns.col_idxs) {
    auto& lhs_col_type = lhs_input_types[lhs_col];
    lhs_output_columns.col_types.push_back(lhs_col_type);
  }

  auto& rhs_input_types = children[1]->get_types();

  if (right_projection_map.empty()) {
    rhs_output_columns.col_idxs.reserve(rhs_input_types.size());
    for (duckdb::idx_t i = 0; i < rhs_input_types.size(); i++) {
      rhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(i));
    }
  } else {
    rhs_output_columns.col_idxs.reserve(right_projection_map.size());
    for (auto& col_idx : right_projection_map) {
      if (col_idx < rhs_input_types.size()) {
        rhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(col_idx));
      } else {
        printf("WARNING:In sirius_physical_hash_join: right_projection_map index out of range");
      }
    }
  }

  for (auto& rhs_col : rhs_output_columns.col_idxs) {
    auto& rhs_col_type = rhs_input_types[rhs_col];
    rhs_output_columns.col_types.push_back(rhs_col_type);
  }

  for (duckdb::idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    auto& condition = conditions[cond_idx];
    const bool is_equality =
      (condition.comparison == duckdb::ExpressionType::COMPARE_EQUAL ||
       condition.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM);

    if (!is_equality) {
      // Inequality conditions are handled at execute time via the cuDF mixed_join binary predicate.
      // No key index extraction is needed here.
      continue;
    }

    is_all_inequality_join = false;
    num_equality_conditions++;

    // Extract left key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    key_cast_info cast_info;
    auto left_class  = condition.left->GetExpressionClass();
    auto right_class = condition.right->GetExpressionClass();

    if (left_class == duckdb::ExpressionClass::BOUND_REF) {
      left_key_col_indices.push_back(
        condition.left->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (left_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = condition.left->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error(
          "Unsupported join condition: BOUND_CAST child is not BOUND_REF (left)");
      }
      left_key_col_indices.push_back(
        bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_left        = true;
      cast_info.left_target_type = duckdb::GetCudfType(condition.left->return_type);
      cast_necessary             = true;
    } else {
      throw std::runtime_error("Unsupported join condition left expression");
    }

    // Extract right key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    if (right_class == duckdb::ExpressionClass::BOUND_REF) {
      right_key_col_indices.push_back(
        condition.right->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (right_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = condition.right->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error(
          "Unsupported join condition: BOUND_CAST child is not BOUND_REF (right)");
      }
      right_key_col_indices.push_back(
        bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_right        = true;
      cast_info.right_target_type = duckdb::GetCudfType(condition.right->return_type);
      cast_necessary              = true;
    } else {
      throw std::runtime_error("Unsupported join condition right expression");
    }

    key_casts.push_back(cast_info);
  }

  // Mixed join: has at least one equality condition (for hashing) and at least one inequality
  // condition (for the binary predicate).
  is_mixed_join = !is_all_inequality_join && (num_equality_conditions < conditions.size());
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

std::unique_ptr<operator_data> sirius_physical_hash_join::get_next_task_input_data()
{
  size_t batch_index = 0;
  {
    std::lock_guard<std::mutex> lg(batches_to_processed_mutex);
    if (left_batch_ids.empty() && right_batch_ids.empty()) {
      if (ports["default"]->repo->num_partitions() != ports["build"]->repo->num_partitions()) {
        throw std::runtime_error(
          "In sirius_physical_hash_join:Number of partitions for left and right ports must be the "
          "same");
      }

      left_batch_ids.reserve(ports["default"]->repo->num_partitions());
      right_batch_ids.reserve(ports["build"]->repo->num_partitions());
      for (size_t i = 0; i < ports["default"]->repo->num_partitions(); i++) {
        left_batch_ids.push_back(ports["default"]->repo->get_batch_ids(i));
        right_batch_ids.push_back(ports["build"]->repo->get_batch_ids(i));
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
  size_t counter = 0;
  for (size_t partition_idx = 0; partition_idx < left_batch_ids.size(); partition_idx++) {
    size_t left_counter = 0;
    for (auto& left_batch_id : left_batch_ids[partition_idx]) {
      size_t right_counter = 0;
      for (auto& right_batch_id : right_batch_ids[partition_idx]) {
        if (counter == batch_index) {
          if (right_counter == right_batch_ids[partition_idx].size() - 1) {
            input_batch.push_back(ports["default"]->repo->pop_data_batch_by_id(
              left_batch_id, cucascade::batch_state::task_created, partition_idx));
          } else {
            input_batch.push_back(ports["default"]->repo->get_data_batch_by_id(
              left_batch_id, cucascade::batch_state::task_created, partition_idx));
          }
          if (left_counter == left_batch_ids[partition_idx].size() - 1) {
            input_batch.push_back(ports["build"]->repo->pop_data_batch_by_id(
              right_batch_id, cucascade::batch_state::task_created, partition_idx));
          } else {
            input_batch.push_back(ports["build"]->repo->get_data_batch_by_id(
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
  if (input_batch.size() == 0) {
    return nullptr;
  } else {
    throw std::runtime_error("Expected to have returned already or received nothing, but got " +
                             std::to_string(input_batch.size()) + " input batches for hash join");
  }
}

/// Result of prepare_join_keys: the key table views and any cast columns that must remain alive.
struct join_keys_result {
  // Owned cast columns - kept alive so the table views referencing them remain valid
  std::vector<std::unique_ptr<cudf::column>> owned_cast_columns;
  cudf::table_view left_keys;
  cudf::table_view right_keys;
  // Storage for column views used to build the table_views (must outlive the table_views)
  std::vector<cudf::column_view> left_key_views;
  std::vector<cudf::column_view> right_key_views;
};

/// Build the left and right key table views for the join.
/// If cast_necessary is false, this simply selects the key columns from the input batches.
/// If cast_necessary is true, each key column that requires a cast is cast to its target type
/// via cudf::cast before being included in the key table.
static join_keys_result prepare_join_keys(
  const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches,
  const std::vector<cudf::size_type>& left_key_col_indices,
  const std::vector<cudf::size_type>& right_key_col_indices,
  bool cast_necessary,
  const std::vector<sirius_physical_hash_join::key_cast_info>& key_casts,
  rmm::cuda_stream_view stream)
{
  join_keys_result result;

  if (!cast_necessary) {
    // Fast path: no casts needed, just select columns directly
    result.left_keys  = get_cudf_table_view(*input_batches[0]).select(left_key_col_indices);
    result.right_keys = get_cudf_table_view(*input_batches[1]).select(right_key_col_indices);
    return result;
  }

  // Slow path: iterate over key columns and cast where needed
  cudf::table_view left_table  = get_cudf_table_view(*input_batches[0]);
  cudf::table_view right_table = get_cudf_table_view(*input_batches[1]);

  for (size_t i = 0; i < left_key_col_indices.size(); i++) {
    const auto& cast_info = key_casts[i];

    // Left key column
    cudf::column_view left_col = left_table.column(left_key_col_indices[i]);
    if (cast_info.cast_left) {
      auto cast_col = cudf::cast(left_col, cast_info.left_target_type, stream);
      result.left_key_views.push_back(cast_col->view());
      result.owned_cast_columns.push_back(std::move(cast_col));
    } else {
      result.left_key_views.push_back(left_col);
    }

    // Right key column
    cudf::column_view right_col = right_table.column(right_key_col_indices[i]);
    if (cast_info.cast_right) {
      auto cast_col = cudf::cast(right_col, cast_info.right_target_type, stream);
      result.right_key_views.push_back(cast_col->view());
      result.owned_cast_columns.push_back(std::move(cast_col));
    } else {
      result.right_key_views.push_back(right_col);
    }
  }

  // Build table_views from the column_view vectors
  result.left_keys  = cudf::table_view(result.left_key_views);
  result.right_keys = cudf::table_view(result.right_key_views);
  return result;
}

/// @brief the MARK join output from the semi_join matching row indices.
///
/// Copies all left output columns (all rows pass through, no gather), then creates a BOOL8 mark
/// column initialized to false and scatters true at every position in semi_indices.
///
/// @param semi_indices  Device vector of left-side row indices that matched the join condition,
///                      as returned by cuDF's semi-join. Used as the scatter map for the mark
///                      column.
/// @param left_full     Full left-side table view (all columns, all rows) before output projection.
/// @param lhs_output_col_idxs  Column indices within @p left_full to include in the output.
///                             Drives the projection of the left side.
/// @param left_batch    The original left-side data batch; used to propagate memory space metadata
///                      to the returned operator_data.
/// @param stream        CUDA stream on which all device operations are launched.
static std::unique_ptr<operator_data> resolve_mark_join_result(
  rmm::device_uvector<cudf::size_type> const& semi_indices,
  cudf::table_view const& left_full,
  std::vector<cudf::size_type> const& lhs_output_col_idxs,
  std::shared_ptr<::cucascade::data_batch> const& left_batch,
  rmm::cuda_stream_view stream)
{
  cudf::table_view left_cols_to_output = left_full.select(lhs_output_col_idxs);
  auto num_left_rows                   = left_cols_to_output.num_rows();

  std::vector<std::unique_ptr<cudf::column>> mark_out_cols;
  for (cudf::size_type i = 0; i < left_cols_to_output.num_columns(); i++) {
    mark_out_cols.push_back(std::make_unique<cudf::column>(left_cols_to_output.column(i), stream));
  }

  // Create BOOL8 mark column: start all-false, scatter true at matching positions
  cudf::numeric_scalar<bool> false_scalar(false, true, stream);
  auto mark_column = cudf::make_column_from_scalar(false_scalar, num_left_rows, stream);

  if (semi_indices.size() > 0) {
    cudf::numeric_scalar<bool> true_scalar(true, true, stream);
    cudf::column_view scatter_map(cudf::data_type(cudf::type_id::INT32),
                                  static_cast<cudf::size_type>(semi_indices.size()),
                                  semi_indices.data(),
                                  nullptr,
                                  0,
                                  0,
                                  {});
    // The scatter API is a bit confusing when it says: the number of elements in first arg i.e.
    // the vector should have same number of columns in the target table. It is essentially a
    // row-scatter operation. For our use case, we have only column i.e. target mark column;
    // therefore we are good. The scalar is broadcasted to respective positions provided by the
    // scatter map.
    auto scattered = cudf::scatter({std::ref(static_cast<cudf::scalar const&>(true_scalar))},
                                   scatter_map,
                                   cudf::table_view({mark_column->view()}),
                                   stream);
    mark_column    = std::move(scattered->release()[0]);
  }

  mark_out_cols.push_back(std::move(mark_column));
  auto output_cudf_table = std::make_unique<cudf::table>(std::move(mark_out_cols), stream);
  return std::make_unique<operator_data>(std::vector<std::shared_ptr<::cucascade::data_batch>>{
    make_data_batch(std::move(output_cudf_table), *left_batch->get_memory_space())});
}

std::unique_ptr<operator_data> sirius_physical_hash_join::execute(const operator_data& input_data,
                                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_hash_join::execute"};
  const auto& input_batches = input_data.get_data_batches();
  if (input_batches.size() != 2) {
    throw std::runtime_error("Expected 2 input batches for hash join, got " +
                             std::to_string(input_batches.size()) + " input batches");
  }
  if (is_all_inequality_join) {
    throw std::runtime_error(
      "Error sirius_physical_hash_join being asked to do all inequality join of type: " +
      duckdb::JoinTypeToString(join_type));
  }

  // Full input table views used as both gather sources and (for mixed joins) conditional views.
  // Hoisted here so both join paths and the shared gather tail can reference them.
  cudf::table_view left_full  = get_cudf_table_view(*input_batches[0]);
  cudf::table_view right_full = get_cudf_table_view(*input_batches[1]);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices, right_indices;

  if (is_mixed_join) {
    // Mixed join: equality conditions drive the hash table; inequality conditions are evaluated
    // via a cuDF AST binary predicate on the full input tables.
    auto keys                 = prepare_join_keys(input_batches,
                                  left_key_col_indices,
                                  right_key_col_indices,
                                  cast_necessary,
                                  key_casts,
                                  stream);
    cudf::table_view left_eq  = keys.left_keys;
    cudf::table_view right_eq = keys.right_keys;

    sirius::gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto pred =
      translator.translate_join_conditions(conditions, num_equality_conditions, conditions.size());
    if (!pred) {
      throw std::runtime_error(
        "In sirius_physical_hash_join: failed to translate mixed join inequality conditions to "
        "cuDF AST predicate");
    }

    if (join_type == duckdb::JoinType::MARK) {
      auto semi_indices = cudf::mixed_left_semi_join(left_eq,
                                                     right_eq,
                                                     left_full,
                                                     right_full,
                                                     pred->back(),
                                                     cudf::null_equality::UNEQUAL,
                                                     stream);
      return resolve_mark_join_result(
        *semi_indices, left_full, lhs_output_columns.col_idxs, input_batches[0], stream);
    } else if (join_type == duckdb::JoinType::INNER) {
      auto result   = cudf::mixed_inner_join(left_eq,
                                           right_eq,
                                           left_full,
                                           right_full,
                                           pred->back(),
                                           cudf::null_equality::UNEQUAL,
                                             {},
                                           stream);
      left_indices  = std::move(result.first);
      right_indices = std::move(result.second);
    } else if (join_type == duckdb::JoinType::LEFT) {
      auto result   = cudf::mixed_left_join(left_eq,
                                          right_eq,
                                          left_full,
                                          right_full,
                                          pred->back(),
                                          cudf::null_equality::UNEQUAL,
                                            {},
                                          stream);
      left_indices  = std::move(result.first);
      right_indices = std::move(result.second);
    } else if (join_type == duckdb::JoinType::RIGHT) {
      // Implement as a swapped left join: right becomes the probe side, left becomes the build
      // side. The predicate is rebuilt with LEFT/RIGHT table references flipped to match.
      auto swapped_pred = translator.translate_join_conditions(
        conditions, num_equality_conditions, conditions.size(), /*swap_sides=*/true);
      if (!swapped_pred) {
        throw std::runtime_error(
          "In sirius_physical_hash_join: failed to translate swapped predicate for RIGHT mixed "
          "join");
      }
      auto result   = cudf::mixed_left_join(right_eq,
                                          left_eq,
                                          right_full,
                                          left_full,
                                          swapped_pred->back(),
                                          cudf::null_equality::UNEQUAL,
                                            {},
                                          stream);
      right_indices = std::move(result.first);
      left_indices  = std::move(result.second);
    } else if (join_type == duckdb::JoinType::OUTER) {
      auto result   = cudf::mixed_full_join(left_eq,
                                          right_eq,
                                          left_full,
                                          right_full,
                                          pred->back(),
                                          cudf::null_equality::UNEQUAL,
                                            {},
                                          stream);
      left_indices  = std::move(result.first);
      right_indices = std::move(result.second);
    } else if (join_type == duckdb::JoinType::SEMI) {
      left_indices = cudf::mixed_left_semi_join(left_eq,
                                                right_eq,
                                                left_full,
                                                right_full,
                                                pred->back(),
                                                cudf::null_equality::UNEQUAL,
                                                stream);
    } else if (join_type == duckdb::JoinType::ANTI) {
      left_indices = cudf::mixed_left_anti_join(left_eq,
                                                right_eq,
                                                left_full,
                                                right_full,
                                                pred->back(),
                                                cudf::null_equality::UNEQUAL,
                                                stream);
    } else if (join_type == duckdb::JoinType::RIGHT_SEMI) {
      auto swapped_pred = translator.translate_join_conditions(
        conditions, num_equality_conditions, conditions.size(), /*swap_sides=*/true);
      if (!swapped_pred) {
        throw std::runtime_error(
          "In sirius_physical_hash_join: failed to translate swapped predicate for RIGHT_SEMI "
          "mixed join");
      }
      right_indices = cudf::mixed_left_semi_join(right_eq,
                                                 left_eq,
                                                 right_full,
                                                 left_full,
                                                 swapped_pred->back(),
                                                 cudf::null_equality::UNEQUAL,
                                                 stream);
    } else if (join_type == duckdb::JoinType::RIGHT_ANTI) {
      auto swapped_pred = translator.translate_join_conditions(
        conditions, num_equality_conditions, conditions.size(), /*swap_sides=*/true);
      if (!swapped_pred) {
        throw std::runtime_error(
          "In sirius_physical_hash_join: failed to translate swapped predicate for RIGHT_ANTI "
          "mixed join");
      }
      right_indices = cudf::mixed_left_anti_join(right_eq,
                                                 left_eq,
                                                 right_full,
                                                 left_full,
                                                 swapped_pred->back(),
                                                 cudf::null_equality::UNEQUAL,
                                                 stream);
    } else {
      throw std::runtime_error("Unsupported join type for mixed join: " +
                               duckdb::JoinTypeToString(join_type));
    }
  } else {
    auto keys                   = prepare_join_keys(input_batches,
                                  left_key_col_indices,
                                  right_key_col_indices,
                                  cast_necessary,
                                  key_casts,
                                  stream);
    cudf::table_view left_keys  = keys.left_keys;
    cudf::table_view right_keys = keys.right_keys;

    if (join_type == duckdb::JoinType::INNER) {
      auto join_result =
        cudf::inner_join(left_keys, right_keys, cudf::null_equality::UNEQUAL, stream);
      left_indices  = std::move(join_result.first);
      right_indices = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::LEFT) {
      auto join_result =
        cudf::left_join(left_keys, right_keys, cudf::null_equality::UNEQUAL, stream);
      left_indices  = std::move(join_result.first);
      right_indices = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::RIGHT) {
      auto join_result =
        cudf::left_join(right_keys, left_keys, cudf::null_equality::UNEQUAL, stream);
      right_indices = std::move(join_result.first);
      left_indices  = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::SEMI) {
      auto filtered_join_object = cudf::filtered_join(
        right_keys, cudf::null_equality::UNEQUAL, cudf::set_as_build_table::RIGHT, stream);
      left_indices = filtered_join_object.semi_join(left_keys, stream);
    } else if (join_type == duckdb::JoinType::RIGHT_SEMI) {
      auto filtered_join_object = cudf::filtered_join(
        left_keys, cudf::null_equality::UNEQUAL, cudf::set_as_build_table::RIGHT, stream);
      right_indices = filtered_join_object.semi_join(right_keys, stream);
    } else if (join_type == duckdb::JoinType::ANTI) {
      auto filtered_join_object = cudf::filtered_join(
        right_keys, cudf::null_equality::UNEQUAL, cudf::set_as_build_table::RIGHT, stream);
      left_indices = filtered_join_object.anti_join(left_keys, stream);
    } else if (join_type == duckdb::JoinType::RIGHT_ANTI) {
      auto filtered_join_object = cudf::filtered_join(
        left_keys, cudf::null_equality::UNEQUAL, cudf::set_as_build_table::RIGHT, stream);
      right_indices = filtered_join_object.anti_join(right_keys, stream);
    } else if (join_type == duckdb::JoinType::MARK) {
      // MARK join: output ALL left rows + a BOOL8 column indicating match presence.
      // Use semi join to find which left rows have matches in the right table.
      auto filtered_join_object = cudf::filtered_join(
        right_keys, cudf::null_equality::UNEQUAL, cudf::set_as_build_table::RIGHT, stream);
      auto semi_indices = filtered_join_object.semi_join(left_keys, stream);
      return resolve_mark_join_result(
        *semi_indices, left_full, lhs_output_columns.col_idxs, input_batches[0], stream);
    } else if (join_type == duckdb::JoinType::OUTER) {
      auto join_result =
        cudf::full_join(left_keys, right_keys, cudf::null_equality::UNEQUAL, stream);
      left_indices  = std::move(join_result.first);
      right_indices = std::move(join_result.second);
    } else {
      throw std::runtime_error("Unsupported join type: " + duckdb::JoinTypeToString(join_type));
    }
  }

  // Shared tail: which sides to collect, out-of-bounds nullification policy, and gather.
  // collect_left/right are purely a function of join type and apply to both mixed and non-mixed.
  bool collect_left =
    (join_type != duckdb::JoinType::RIGHT_SEMI && join_type != duckdb::JoinType::RIGHT_ANTI);
  bool collect_right = (join_type != duckdb::JoinType::SEMI && join_type != duckdb::JoinType::ANTI);

  cudf::out_of_bounds_policy left_oob  = cudf::out_of_bounds_policy::DONT_CHECK;
  cudf::out_of_bounds_policy right_oob = cudf::out_of_bounds_policy::DONT_CHECK;
  if (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::OUTER ||
      join_type == duckdb::JoinType::SEMI) {
    right_oob = cudf::out_of_bounds_policy::NULLIFY;
  }
  if (join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::OUTER ||
      join_type == duckdb::JoinType::RIGHT_SEMI) {
    left_oob = cudf::out_of_bounds_policy::NULLIFY;
  }

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  if (collect_left) {
    cudf::table_view left_cols_to_gather = left_full.select(lhs_output_columns.col_idxs);
    cudf::column_view left_map_view(cudf::data_type(cudf::type_id::INT32),
                                    left_indices->size(),
                                    left_indices->data(),
                                    nullptr,
                                    0,
                                    0,
                                    {});
    auto left_result = cudf::gather(left_cols_to_gather, left_map_view, left_oob, stream);
    out_cols         = left_result->release();
  }
  if (collect_right) {
    cudf::table_view right_cols_to_gather = right_full.select(rhs_output_columns.col_idxs);
    cudf::column_view right_map_view(cudf::data_type(cudf::type_id::INT32),
                                     right_indices->size(),
                                     right_indices->data(),
                                     nullptr,
                                     0,
                                     0,
                                     {});
    auto right_result   = cudf::gather(right_cols_to_gather, right_map_view, right_oob, stream);
    auto right_out_cols = right_result->release();
    for (auto& col : right_out_cols) {
      out_cols.push_back(std::move(col));
    }
  }

  auto output_cudf_table = std::make_unique<cudf::table>(std::move(out_cols), stream);
  return std::make_unique<operator_data>(std::vector<std::shared_ptr<::cucascade::data_batch>>{
    make_data_batch(std::move(output_cudf_table), *input_batches[0]->get_memory_space())});
}

}  // namespace op
}  // namespace sirius
