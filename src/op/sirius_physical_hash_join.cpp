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
#include "data/data_batch_utils.hpp"

#include "duckdb/common/enums/physical_operator_type.hpp"
#include <cstdio>
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include "cudf/copying.hpp"
#include "cudf/join/join.hpp"
#include "cudf/types.hpp"
#include "cudf/table/table_view.hpp"
#include "cudf/unary.hpp"

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

  // duckdb::unordered_map<duckdb::idx_t, duckdb::idx_t> build_columns_in_conditions;
  // for (duckdb::idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
  //   auto& condition = conditions[cond_idx];
  //   condition_types.push_back(condition.left->return_type);
  //   if (condition.right->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
  //     build_columns_in_conditions.emplace(
  //       condition.right->Cast<duckdb::BoundReferenceExpression>().index, cond_idx);
  //   }
  // }

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

  if (join_type == duckdb::JoinType::ANTI  ||
      join_type == duckdb::JoinType::MARK) {
        throw std::runtime_error("Unsupported join type: " + duckdb::JoinTypeToString(join_type));
    // materialized_build_key =
    //   duckdb::make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
    // hash_table_result =
    //   duckdb::make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
    return;
  }

  auto& rhs_input_types = children[1]->get_types();

  // std::vector<cudf::size_type> right_projection_cudf(right_projection_map.begin(),
  //                                                   right_projection_map.end());
  // if (right_projection_cudf.empty()) {
  //   right_projection_cudf.reserve(rhs_input_types.size());
  //   for (duckdb::idx_t i = 0; i < rhs_input_types.size(); i++) {
  //     right_projection_cudf.emplace_back(static_cast<cudf::size_type>(i));
  //   }
  // }

  // for (auto rhs_col : right_projection_cudf) {
  //   auto& rhs_col_type = rhs_input_types[rhs_col];

  //   auto it = build_columns_in_conditions.find(static_cast<duckdb::idx_t>(rhs_col));
  //   if (it == build_columns_in_conditions.end()) {
  //     payload_columns.col_idxs.push_back(rhs_col);
  //     payload_columns.col_types.push_back(rhs_col_type);
  //     rhs_output_columns.col_idxs.push_back(static_cast<cudf::size_type>(
  //       condition_types.size() + payload_columns.col_types.size() - 1));
  //   } else {
  //     rhs_output_columns.col_idxs.push_back(static_cast<cudf::size_type>(it->second));
  //   }
  //   rhs_output_columns.col_types.push_back(rhs_col_type);
  // }

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
    if (condition.comparison != duckdb::ExpressionType::COMPARE_EQUAL &&
        condition.comparison != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      printf("Unsupported non-equality condition comparison: %d\n", static_cast<int>(condition.comparison));
      printf("    left: %s\n", condition.left->ToString().c_str());
      printf("    right: %s\n", condition.right->ToString().c_str());
      is_equality_join = false;
      break;
    }

    // Extract left key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    key_cast_info cast_info;
    auto left_class = condition.left->GetExpressionClass();
    auto right_class = condition.right->GetExpressionClass();

    if (left_class == duckdb::ExpressionClass::BOUND_REF) {
      left_key_col_indices.push_back(condition.left->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (left_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = condition.left->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error("Unsupported join condition: BOUND_CAST child is not BOUND_REF (left)");
      }
      left_key_col_indices.push_back(bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_left = true;
      cast_info.left_target_type = duckdb::GetCudfType(condition.left->return_type);
      cast_necessary = true;
    } else {
      printf("Unsupported left expression class: %s\n", condition.left->ToString().c_str());
      throw std::runtime_error("Unsupported join condition left expression");
    }

    // Extract right key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    if (right_class == duckdb::ExpressionClass::BOUND_REF) {
      right_key_col_indices.push_back(condition.right->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (right_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = condition.right->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error("Unsupported join condition: BOUND_CAST child is not BOUND_REF (right)");
      }
      right_key_col_indices.push_back(bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_right = true;
      cast_info.right_target_type = duckdb::GetCudfType(condition.right->return_type);
      cast_necessary = true;
    } else {
      printf("Unsupported right expression class: %s\n", condition.right->ToString().c_str());
      throw std::runtime_error("Unsupported join condition right expression");
    }

    key_casts.push_back(cast_info);
  }

  printf("sirius_physical_hash_join::sirius_physical_hash_join\n");
  printf("  class: sirius_physical_hash_join\n");
  printf("  left: %s\n", left ? left->get_name().c_str() : "(null)");
  printf("  right: %s\n", right ? right->get_name().c_str() : "(null)");
  printf("  join_type: %d\n", static_cast<int>(join_type));
  printf("  conditions.size(): %zu\n", conditions.size());
  printf("  delim_types.size(): %zu\n", this->delim_types.size());
  printf("  estimated_cardinality: %llu\n", static_cast<unsigned long long>(estimated_cardinality));
  printf("  pushdown_info: %s\n", pushdown_info_p ? "non-null" : "null");
  printf("  is_equality_join: %s\n", is_equality_join ? "true" : "false");
  printf("  cast_necessary: %s\n", cast_necessary ? "true" : "false");
  printf("  left_key_col_indices: [");
  for (size_t i = 0; i < left_key_col_indices.size(); i++) {
    printf("%s%zu", i ? ", " : "", static_cast<size_t>(left_key_col_indices[i]));
  }
  printf("] (size=%zu)\n", left_key_col_indices.size());
  printf("  right_key_col_indices: [");
  for (size_t i = 0; i < right_key_col_indices.size(); i++) {
    printf("%s%zu", i ? ", " : "", static_cast<size_t>(right_key_col_indices[i]));
  }
  printf("] (size=%zu)\n", right_key_col_indices.size());
  printf("  payload_columns: col_idxs=[");
  for (size_t i = 0; i < payload_columns.col_idxs.size(); i++) {
    printf("%s%zu", i ? ", " : "", static_cast<size_t>(payload_columns.col_idxs[i]));
  }
  printf("] (size=%zu), col_types.size()=%zu\n",
         payload_columns.col_idxs.size(),
         payload_columns.col_types.size());
  printf("  lhs_output_columns: col_idxs=[");
  for (size_t i = 0; i < lhs_output_columns.col_idxs.size(); i++) {
    printf("%s%zu", i ? ", " : "", static_cast<size_t>(lhs_output_columns.col_idxs[i]));
  }
  printf("] (size=%zu), col_types.size()=%zu\n",
         lhs_output_columns.col_idxs.size(),
         lhs_output_columns.col_types.size());
  printf("  rhs_output_columns: col_idxs=[");
  for (size_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
    printf("%s%zu", i ? ", " : "", static_cast<size_t>(rhs_output_columns.col_idxs[i]));
  }
  printf("] (size=%zu), col_types.size()=%zu\n",
         rhs_output_columns.col_idxs.size(),
         rhs_output_columns.col_types.size());
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






std::optional<std::vector<std::shared_ptr<::cucascade::data_batch>>> sirius_physical_hash_join::get_next_task_input_batch()
{
  printf("sirius_physical_hash_join::get_next_task_input_batch 0\n");
  // print all the port names
  // printf("  ports: ");
  // for (auto& [port_name, port_ptr] : ports) {
  //   printf("%s ", port_name.c_str());
  // }
  // printf("\n");
  size_t batch_index = 0;
  {
    std::lock_guard<std::mutex> lg(batches_to_processed_mutex);
    if (left_batch_ids.empty() && right_batch_ids.empty()) {
      if (ports["default"]->repo->num_partitions() != ports["build"]->repo->num_partitions() ) {
        throw std::runtime_error("In sirius_physical_hash_join:Number of partitions for left and right ports must be the same");
      }

      left_batch_ids.reserve(ports["default"]->repo->num_partitions());
      right_batch_ids.reserve(ports["build"]->repo->num_partitions());
      printf("ports[\"default\"]->repo->num_partitions(): %zu\n", ports["default"]->repo->num_partitions());
      for (size_t i = 0; i < ports["default"]->repo->num_partitions(); i++) {
        left_batch_ids.push_back(ports["default"]->repo->get_batch_ids(i));
        right_batch_ids.push_back(ports["build"]->repo->get_batch_ids(i));
        num_batches_to_process += left_batch_ids[i].size() * right_batch_ids[i].size();
        printf("partition %zu: left_batch_ids.size(): %zu, right_batch_ids.size(): %zu\n", i, left_batch_ids[i].size(), right_batch_ids[i].size());
        // the contents of left_batch_ids and right_batch_ids are the batch ids, print the contents of left_batch_ids and right_batch_ids
        printf("left_batch_ids[%zu]: ", i);
        for (auto& batch_id : left_batch_ids[i]) {
          printf("%zu ", batch_id);
        }
        printf("\n");
        printf("right_batch_ids[%zu]: ", i);
        for (auto& batch_id : right_batch_ids[i]) {
          printf("%zu ", batch_id);
        }
        printf("\n");
      }
      printf("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ  num_batches_to_process: %zu\n", num_batches_to_process);
    }
    if (current_partition_index < num_batches_to_process) {
      batch_index = current_partition_index;
      current_partition_index++;
    } else {
      return std::nullopt;
    }
  }
  printf("sirius_physical_hash_join::get_next_task_input_batch batch_index: %zu\n", batch_index);

  std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
  input_batch.reserve(2);
  size_t counter = 0;
  for (size_t partition_idx = 0; partition_idx < left_batch_ids.size(); partition_idx++) {
    size_t left_counter = 0;
    for (auto& left_batch_id : left_batch_ids[partition_idx]) {
      printf("starting loop for left_batch_id: %zu\n", left_batch_id);
      size_t right_counter = 0;
      for (auto& right_batch_id : right_batch_ids[partition_idx]) {
        if (counter == batch_index) {  
          if (right_counter == right_batch_ids[partition_idx].size() - 1) {
            printf("popping left batch %zu\n", left_batch_id);
            input_batch.push_back(ports["default"]->repo->pop_data_batch_by_id(left_batch_id, cucascade::batch_state::task_created, partition_idx));  
          } else {
            printf("getting left batch %zu\n", left_batch_id);
            input_batch.push_back(ports["default"]->repo->get_data_batch_by_id(left_batch_id, cucascade::batch_state::task_created, partition_idx));
          }
          if (left_counter == left_batch_ids[partition_idx].size() - 1) {
            printf("popping right batch %zu\n", right_batch_id);
            input_batch.push_back(ports["build"]->repo->pop_data_batch_by_id(right_batch_id, cucascade::batch_state::task_created, partition_idx));
          } else {
            printf("getting right batch %zu\n", right_batch_id);
            input_batch.push_back(ports["build"]->repo->get_data_batch_by_id(right_batch_id, cucascade::batch_state::task_created, partition_idx));
          }
          printf("returning left_counter: %zu, right_counter: %zu\n", left_counter, right_counter);
          printf("input_batch[0]->get_data()->get_size_in_bytes(): %zu\n", input_batch[0]->get_data()->get_size_in_bytes());
          printf("input_batch[1]->get_data()->get_size_in_bytes(): %zu\n", input_batch[1]->get_data()->get_size_in_bytes());
          return input_batch;
        }
        right_counter++;
        counter++;
      }      
      left_counter++;
      printf("left_counter: %zu\n", left_counter);
    }
    printf("counter: %zu\n", counter);    
  }  
  if (input_batch.size() == 0) {
    printf("input_batch.size() == 0\n");
    return std::nullopt;
  } else {
    throw std::runtime_error("Expected to have returned already or received nothing, but got " + std::to_string(input_batch.size()) + " input batches for hash join");
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
    result.left_keys = get_cudf_table_view(*input_batches[0]).select(left_key_col_indices);
    result.right_keys = get_cudf_table_view(*input_batches[1]).select(right_key_col_indices);
    return result;
  }

  // Slow path: iterate over key columns and cast where needed
  cudf::table_view left_table = get_cudf_table_view(*input_batches[0]);
  cudf::table_view right_table = get_cudf_table_view(*input_batches[1]);

  for (size_t i = 0; i < left_key_col_indices.size(); i++) {
    const auto& cast_info = key_casts[i];

    // Left key column
    cudf::column_view left_col = left_table.column(left_key_col_indices[i]);
    if (cast_info.cast_left) {
      printf("  casting left key column %zu from %s to %s\n",
             i,
             cudf::type_to_name(left_col.type()).c_str(),
             cudf::type_to_name(cast_info.left_target_type).c_str());
      auto cast_col = cudf::cast(left_col, cast_info.left_target_type, stream);
      result.left_key_views.push_back(cast_col->view());
      result.owned_cast_columns.push_back(std::move(cast_col));
    } else {
      result.left_key_views.push_back(left_col);
    }

    // Right key column
    cudf::column_view right_col = right_table.column(right_key_col_indices[i]);
    if (cast_info.cast_right) {
      printf("  casting right key column %zu from %s to %s\n",
             i,
             cudf::type_to_name(right_col.type()).c_str(),
             cudf::type_to_name(cast_info.right_target_type).c_str());
      auto cast_col = cudf::cast(right_col, cast_info.right_target_type, stream);
      result.right_key_views.push_back(cast_col->view());
      result.owned_cast_columns.push_back(std::move(cast_col));
    } else {
      result.right_key_views.push_back(right_col);
    }
  }

  // Build table_views from the column_view vectors
  result.left_keys = cudf::table_view(result.left_key_views);
  result.right_keys = cudf::table_view(result.right_key_views);
  return result;
}

std::vector<std::shared_ptr<::cucascade::data_batch>> sirius_physical_hash_join::execute(
  const std::vector<std::shared_ptr<::cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  // printf("sirius_physical_hash_join::execute 0\n");
  // printf("  input_batches.size(): %zu\n", input_batches.size());

  if (input_batches.size() != 2) {
    throw std::runtime_error("Expected 2 input batches for hash join, got " + std::to_string(input_batches.size()));
  }
  if (!is_equality_join) {
    throw std::runtime_error("Unsupported non-equality join of type type: " + duckdb::JoinTypeToString(join_type));
  }
  if (join_type == duckdb::JoinType::INNER || join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::RIGHT 
      || join_type == duckdb::JoinType::OUTER || join_type == duckdb::JoinType::SEMI || join_type == duckdb::JoinType::RIGHT_SEMI){

    auto keys = prepare_join_keys(input_batches, left_key_col_indices, right_key_col_indices,
                                  cast_necessary, key_casts, stream);
    cudf::table_view left_keys = keys.left_keys;
    cudf::table_view right_keys = keys.right_keys;
    // printf("left keys size: %zu\n", left_keys.num_rows());
    // printf("right keys size: %zu\n", right_keys.num_rows());
    // std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
    //       std::unique_ptr<rmm::device_uvector<size_type>>> join_result;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices, right_indices;
    bool collect_left = true;
    bool collect_right = true;
    if (join_type == duckdb::JoinType::INNER) {
      auto join_result = cudf::inner_join(left_keys, right_keys, cudf::null_equality::UNEQUAL, stream);
      left_indices = std::move(join_result.first);
      right_indices = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::SEMI || 
       join_type == duckdb::JoinType::RIGHT_SEMI) {
      auto join_result = cudf::left_join(left_keys, right_keys, cudf::null_equality::UNEQUAL, stream);
      left_indices = std::move(join_result.first);
      right_indices = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::RIGHT) {
      auto join_result = cudf::left_join(right_keys, left_keys, cudf::null_equality::UNEQUAL, stream);
      right_indices = std::move(join_result.first);
      left_indices = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::OUTER) {
      auto join_result = cudf::full_join(left_keys, right_keys, cudf::null_equality::UNEQUAL, stream);
      left_indices = std::move(join_result.first);
      right_indices = std::move(join_result.second);
    }
    if (join_type == duckdb::JoinType::SEMI || join_type == duckdb::JoinType::RIGHT_SEMI) {
      collect_right = false;
    }

    cudf::out_of_bounds_policy left_out_of_bounds_policy = cudf::out_of_bounds_policy::DONT_CHECK;
    cudf::out_of_bounds_policy right_out_of_bounds_policy = cudf::out_of_bounds_policy::DONT_CHECK;
    if (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::OUTER ||
      join_type == duckdb::JoinType::SEMI || join_type == duckdb::JoinType::RIGHT_SEMI) {
      right_out_of_bounds_policy = cudf::out_of_bounds_policy::NULLIFY;
    }
    if (join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::OUTER) {
      left_out_of_bounds_policy = cudf::out_of_bounds_policy::NULLIFY;
    }
      
    std::vector<std::unique_ptr<cudf::column>> out_cols;
    if (collect_left) {
      printf("collect left\n");
      // print lhs_output_columns.col_idxs
      printf("  lhs_output_columns.col_idxs: [");
      for (size_t i = 0; i < lhs_output_columns.col_idxs.size(); i++) {
        printf("%s%zu", i ? ", " : "", static_cast<size_t>(lhs_output_columns.col_idxs[i]));
      }
      printf("] (size=%zu)\n", lhs_output_columns.col_idxs.size());
      cudf::table_view left_cols_to_gather = get_cudf_table_view(*input_batches[0]).select(lhs_output_columns.col_idxs);
      cudf::column_view left_map_view(cudf::data_type(cudf::type_id::INT32),
                                    left_indices->size(),
                                    left_indices->data(),
                                    nullptr,
                                    0,
                                    0,
                                    {});
      // printf("before left gather\n");
      auto left_result = cudf::gather(left_cols_to_gather, left_map_view, left_out_of_bounds_policy, stream);
      // printf("after left gather\n");
      out_cols = left_result->release();
    }
    if (collect_right) {
      printf("collect right\n");
      // print rhs_output_columns.col_idxs
      printf("  rhs_output_columns.col_idxs: [");
      for (size_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
        printf("%s%zu", i ? ", " : "", static_cast<size_t>(rhs_output_columns.col_idxs[i]));
      }
      printf("] (size=%zu)\n", rhs_output_columns.col_idxs.size());
      cudf::table_view right_cols_to_gather = get_cudf_table_view(*input_batches[1]).select(rhs_output_columns.col_idxs);
      cudf::column_view right_map_view(cudf::data_type(cudf::type_id::INT32),
                                      right_indices->size(),
                                      right_indices->data(),
                                      nullptr,
                                      0,
                                      0,
                                      {});
      // printf("before right gather\n");
      auto right_result = cudf::gather(right_cols_to_gather, right_map_view, right_out_of_bounds_policy, stream);
      // printf("after right gather\n");
      auto right_out_cols = right_result->release();
      for (auto& col : right_out_cols) {
        out_cols.push_back(std::move(col));
      }
    }
    
    auto output_cudf_table = std::make_unique<cudf::table>(std::move(out_cols), stream);
    // printf("output cudf table size: %zu\n", output_cudf_table->num_rows());
    return std::vector<std::shared_ptr<cucascade::data_batch>>{
        make_data_batch(std::move(output_cudf_table), *input_batches[0]->get_memory_space())};
    
  
  // } else if (join_type == duckdb::JoinType::ANTI) {
  //   return std::vector<std::shared_ptr<::cucascade::data_batch>>{};
  // } else if (join_type == duckdb::JoinType::MARK) {
  //   return std::vector<std::shared_ptr<::cucascade::data_batch>>{};
  // } else if (join_type == duckdb::JoinType::SINGLE) {
  //   return std::vector<std::shared_ptr<::cucascade::data_batch>>{};
  // } else if (join_type == duckdb::JoinType::RIGHT_SEMI) {
  //   return std::vector<std::shared_ptr<::cucascade::data_batch>>{};
  // } else if (join_type == duckdb::JoinType::RIGHT_ANTI) {
  //   return std::vector<std::shared_ptr<::cucascade::data_batch>>{};
  } else {
    throw std::runtime_error("Unsupported join type: " + duckdb::JoinTypeToString(join_type));
  }

}

}  // namespace op
}  // namespace sirius
