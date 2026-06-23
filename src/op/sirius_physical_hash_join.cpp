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
#include "cudf/join/distinct_hash_join.hpp"
#include "cudf/join/filtered_join.hpp"
#include "cudf/join/join.hpp"
#include "cudf/join/mixed_join.hpp"
#include "cudf/table/table_view.hpp"
#include "cudf/types.hpp"
#include "cudf/unary.hpp"
#include "cudf/utilities/memory_resource.hpp"
#include "cudf/version_config.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/joinside.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression_executor/gpu_expression_translator_internal.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cstdio>
#include <unordered_set>

namespace sirius {
namespace op {

/// Recursively collect all BoundReferenceExpression indices from an expression tree.
static void collect_bound_ref_indices(const duckdb::Expression& expr,
                                      std::unordered_set<std::size_t>& indices)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    indices.insert(expr.Cast<duckdb::BoundReferenceExpression>().index);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expr, [&](const duckdb::Expression& child) { collect_bound_ref_indices(child, indices); });
}

static bool is_equality(sirius::comparison_type c)
{
  return c == sirius::comparison_type::equal || c == sirius::comparison_type::not_distinct_from;
}

static cudf::filtered_join make_right_filtered_join(cudf::table_view const& right_keys,
                                                    rmm::cuda_stream_view stream)
{
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 8)
  return cudf::filtered_join(right_keys, cudf::null_equality::UNEQUAL, stream);
#else
  return cudf::filtered_join(
    right_keys, cudf::null_equality::UNEQUAL, cudf::set_as_build_table::RIGHT, stream);
#endif
}

bool sirius_physical_hash_join::are_conditions_supported(
  duckdb::vector<sirius::join_condition>& conditions)
{
  // Must have at least one equality condition for a hash-based join.
  bool has_equality = false;
  for (auto const& cond : conditions) {
    if (is_equality(cond.comparison)) {
      has_equality = true;
      break;
    }
  }
  if (!has_equality) { return false; }

  // Pure equality join: always supported.
  bool has_inequality = false;
  for (auto const& cond : conditions) {
    if (!is_equality(cond.comparison)) {
      has_inequality = true;
      break;
    }
  }
  if (!has_inequality) { return true; }

  // Mixed join: collect the column indices used on each side of the equality conditions.
  std::unordered_set<std::size_t> equality_left_cols, equality_right_cols;
  for (auto const& cond : conditions) {
    if (!is_equality(cond.comparison)) { continue; }
    auto left_owned  = sirius::ast::to_duckdb(*cond.left);
    auto right_owned = sirius::ast::to_duckdb(*cond.right);
    collect_bound_ref_indices(*left_owned, equality_left_cols);
    collect_bound_ref_indices(*right_owned, equality_right_cols);
  }

  // For each inequality condition, verify that its left/right column references don't overlap
  // with the equality key columns on the same side. cuDF's mixed_join API requires the equality
  // and conditional table columns to be disjoint.
  for (auto const& cond : conditions) {
    if (is_equality(cond.comparison)) { continue; }
    std::unordered_set<std::size_t> ineq_left_cols, ineq_right_cols;
    auto left_owned  = sirius::ast::to_duckdb(*cond.left);
    auto right_owned = sirius::ast::to_duckdb(*cond.right);
    collect_bound_ref_indices(*left_owned, ineq_left_cols);
    collect_bound_ref_indices(*right_owned, ineq_right_cols);
    for (auto const idx : ineq_left_cols) {
      if (equality_left_cols.count(idx) > 0) { return false; }
    }
    for (auto const idx : ineq_right_cols) {
      if (equality_right_cols.count(idx) > 0) { return false; }
    }
  }

  return true;
}

void reorder_join_conditions(duckdb::vector<sirius::join_condition>& conditions)
{
  bool is_ordered     = true;
  bool seen_non_equal = false;
  for (auto& cond : conditions) {
    if (is_equality(cond.comparison)) {
      if (seen_non_equal) {
        is_ordered = false;
        break;
      }
    } else {
      seen_non_equal = true;
    }
  }
  if (is_ordered) { return; }
  duckdb::vector<sirius::join_condition> equal_conditions;
  duckdb::vector<sirius::join_condition> other_conditions;
  for (auto& cond : conditions) {
    if (is_equality(cond.comparison)) {
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
  duckdb::vector<sirius::join_condition> cond,
  duckdb::JoinType join_type,
  const duckdb::vector<std::size_t>& left_projection_map,
  const duckdb::vector<std::size_t>& right_projection_map,
  duckdb::vector<sirius::logical_type> delim_types,
  std::size_t estimated_cardinality,
  duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> pushdown_info_p,
  uint64_t max_build_hash_table_bytes)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::HASH_JOIN,
                                                sirius::from_duckdb_vec(op.types),
                                                estimated_cardinality),
    conditions(std::move(cond)),
    join_type(join_type),
    delim_types(std::move(delim_types))
{
  _max_build_hash_table_bytes = max_build_hash_table_bytes;
  reorder_join_conditions(conditions);

  filter_pushdown = std::move(pushdown_info_p);

  children.push_back(std::move(left));
  children.push_back(std::move(right));

  auto& lhs_input_types = children[0]->get_types();

  if (left_projection_map.empty()) {
    lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
    for (std::size_t i = 0; i < lhs_input_types.size(); i++) {
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
    for (std::size_t i = 0; i < rhs_input_types.size(); i++) {
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

  for (std::size_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    auto& condition        = conditions[cond_idx];
    auto left_owned        = sirius::ast::to_duckdb(*condition.left);
    auto right_owned       = sirius::ast::to_duckdb(*condition.right);
    auto const* left_expr  = left_owned.get();
    auto const* right_expr = right_owned.get();

    if (!is_equality(condition.comparison)) {
      // Inequality conditions are handled at execute time via the cuDF mixed_join binary predicate.
      // No key index extraction is needed here.
      continue;
    }

    is_all_inequality_join = false;
    num_equality_conditions++;

    // Extract left key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    key_cast_info cast_info;
    auto left_class  = left_expr->GetExpressionClass();
    auto right_class = right_expr->GetExpressionClass();

    if (left_class == duckdb::ExpressionClass::BOUND_REF) {
      left_key_col_indices.push_back(left_expr->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (left_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = left_expr->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error(
          "Unsupported join condition: BOUND_CAST child is not BOUND_REF (left)");
      }
      left_key_col_indices.push_back(
        bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_left        = true;
      cast_info.left_target_type = duckdb::GetCudfType(left_expr->return_type);
      cast_necessary             = true;
    } else {
      throw std::runtime_error("Unsupported join condition left expression");
    }

    // Extract right key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    if (right_class == duckdb::ExpressionClass::BOUND_REF) {
      right_key_col_indices.push_back(right_expr->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (right_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = right_expr->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error(
          "Unsupported join condition: BOUND_CAST child is not BOUND_REF (right)");
      }
      right_key_col_indices.push_back(
        bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_right        = true;
      cast_info.right_target_type = duckdb::GetCudfType(right_expr->return_type);
      cast_necessary              = true;
    } else {
      throw std::runtime_error("Unsupported join condition right expression");
    }

    key_casts.push_back(cast_info);
  }

  // Mixed join: has at least one equality condition (for hashing) and at least one inequality
  // condition (for the binary predicate).
  if (!is_all_inequality_join && (num_equality_conditions < conditions.size())) {
    _join_mode = HASH_JOIN_MODE::MIXED_JOIN;
  }
};

sirius_physical_hash_join::sirius_physical_hash_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<sirius::join_condition> cond,
  duckdb::JoinType join_type,
  std::size_t estimated_cardinality,
  uint64_t max_build_hash_table_bytes)
  : sirius_physical_hash_join(op,
                              std::move(left),
                              std::move(right),
                              std::move(cond),
                              join_type,
                              {},
                              {},
                              {},
                              estimated_cardinality,
                              nullptr,
                              max_build_hash_table_bytes)
{
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_hash_join::build_join_pipelines(pipeline::sirius_pipeline& current,
                                                     pipeline::sirius_meta_pipeline& meta_pipeline,
                                                     sirius_physical_operator& op,
                                                     bool build_rhs)
{
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
      throw not_implemented_exception("POSITIONAL_JOIN is not implemented yet");
      meta_pipeline.create_child_pipeline(current, op, last_pipeline);
      return;
    case SiriusPhysicalOperatorType::CROSS_PRODUCT:
      throw not_implemented_exception("CROSS_PRODUCT is not implemented yet");
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

void sirius_physical_hash_join::update_join_exec_mode(int num_partitions,
                                                      uint64_t build_side_bytes,
                                                      bool build_foldable_to_single_batch)
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  if (num_partitions == 1 && build_side_bytes < _max_build_hash_table_bytes &&
      build_foldable_to_single_batch && join_type != duckdb::JoinType::SEMI &&
      join_type != duckdb::JoinType::RIGHT_SEMI && join_type != duckdb::JoinType::ANTI &&
      join_type != duckdb::JoinType::RIGHT_ANTI && join_type != duckdb::JoinType::RIGHT &&
      join_type != duckdb::JoinType::MARK && _join_mode != HASH_JOIN_MODE::MIXED_JOIN) {
    // Switch to a more efficient join strategy for small datasets. The
    // build_foldable_to_single_batch gate matches the runtime invariant in
    // get_next_task_input_data_for_build_probe — BUILD_PROBE requires the
    // build port to deliver exactly one batch, so we refuse to enter the
    // mode when the upstream pipeline cannot guarantee that.
    _join_mode = HASH_JOIN_MODE::BUILD_PROBE;
    SIRIUS_LOG_DEBUG(
      "sirius_physical_hash_join id {} switching to BUILD_PROBE mode with {} partitions and build "
      "side size {} bytes",
      this->get_operator_id(),
      num_partitions,
      build_side_bytes);
  }
}

bool sirius_physical_hash_join::is_build_probe_mode()
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  return _join_mode == HASH_JOIN_MODE::BUILD_PROBE;
}

std::optional<task_creation_hint> sirius_physical_hash_join::get_next_task_hint(
  std::optional<std::size_t> downstream_request)
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    // In build-probe mode, we want the first task to be with one batch from either side.
    // In the first batch we will build the hash table, then we only need batches from the probe
    // side.
    auto* build_port = get_port("build");
    auto* probe_port = get_port("default");
    if (!build_port || !probe_port) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:get_next_task_hint: missing expected ports in operator " +
        std::to_string(this->get_operator_id()));
    }
    auto build_size = build_port->repo->total_size();
    auto probe_size = probe_port->repo->total_size();
    if (_hash_table_build_state == BUILD_HASH_TABLE_STATE::NOT_BUILT) {
      if (build_size > 0 && probe_size > 0) {
        // Only a single build task fires here — it consumes one batch from each side and
        // transitions the state machine. Subsequent probe tasks happen after BUILT. Downstream's
        // request is ignored here because the build-task count is structurally fixed at 1.
        _hash_table_build_state = BUILD_HASH_TABLE_STATE::SCHEDULING;
        return task_creation_hint{TaskCreationHint::READY, this, 1};
      } else if (build_size == 0) {
        // No build batch available yet, hint to wait for build input data. Build side is FAN_IN
        // into the hash table; do not promote with downstream_request.
        auto* producer = &build_port->src_pipeline->get_operators()[0].get();
        return task_creation_hint{
          TaskCreationHint::WAITING_FOR_INPUT_DATA, producer, tasks_for_barrier(build_port->type)};
      } else {
        // Build batch is available but no probe batch yet, hint to wait for probe input data.
        // Probe side is PASSTHROUGH in BUILT mode, so we propagate the downstream request when
        // larger than the port's local barrier default.
        auto* producer = &probe_port->src_pipeline->get_operators()[0].get();
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                                  producer,
                                  combine_upstream_request(tasks_for_barrier(probe_port->type),
                                                           TaskCountRelation::PASSTHROUGH,
                                                           downstream_request)};
      }
    } else if (_hash_table_build_state == BUILD_HASH_TABLE_STATE::SCHEDULING ||
               _hash_table_build_state == BUILD_HASH_TABLE_STATE::SCHEDULED) {
      // Hash table is currently being built; we can't fire probe tasks yet. Asking the upstream
      // for zero tasks lets task_creator skip creating speculative probe tasks during this window;
      // the join will be re-scheduled when the build completes.
      auto* producer = &probe_port->src_pipeline->get_operators()[0].get();
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer, 0};
    } else if (_hash_table_build_state == BUILD_HASH_TABLE_STATE::BUILT) {
      // Hash table is built, we can process probe only batches. In this state each probe batch
      // produces one output batch (PASSTHROUGH); cap by downstream's explicit request when given.
      if (ports["default"]->repo->total_size() > 0) {
        std::size_t cap = downstream_request.value_or(task_creation_hint::ALL_TASKS);
        return task_creation_hint{TaskCreationHint::READY, this, cap};
      } else {
        // No probe batch available yet, hint to wait for probe input data.
        auto* producer = &ports["default"]->src_pipeline->get_operators()[0].get();
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                                  producer,
                                  combine_upstream_request(tasks_for_barrier(probe_port->type),
                                                           TaskCountRelation::PASSTHROUGH,
                                                           downstream_request)};
      }
    } else {
      // If we are here, then this operator is actually complete.
      return std::nullopt;
    }
  } else {
    return sirius_physical_operator::get_next_task_hint(downstream_request);
  }
}

std::unique_ptr<operator_data> sirius_physical_hash_join::get_next_task_input_data_for_build_probe()
{
  auto* build_port = get_port("build");
  auto* probe_port = get_port("default");
  if (!build_port || !probe_port) {
    throw std::runtime_error(
      "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: missing expected "
      "ports in operator " +
      std::to_string(this->get_operator_id()));
  }
  if (_hash_table_build_state == BUILD_HASH_TABLE_STATE::SCHEDULING) {
    if (build_port->repo->num_partitions() != 1 || build_port->repo->size(0) != 1 ||
        probe_port->repo->num_partitions() != 1) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: expected exactly 1 "
        "partition and 1 batch in default (build) port in operator " +
        std::to_string(this->get_operator_id()));
    }
    // When the hash table is not build yet, we will send both the build and probe side. To build
    // the hash table and perform the first join.
    std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
    auto probe_batch = probe_port->repo->pop_next_data_batch();
    auto build_batch = build_port->repo->pop_next_data_batch();
    input_batch.push_back(std::move(probe_batch));
    input_batch.push_back(std::move(build_batch));
    _hash_table_build_state = BUILD_HASH_TABLE_STATE::SCHEDULED;
    // BUILD_PROBE mode uses a single cuco hash table shared across every probe
    // task for this join. All such tasks must land on the same GPU, so we tag
    // them with operator_id as the partition index (hash joins get spread
    // across GPUs at the query level, but each individual join stays pinned).
    return std::make_unique<partitioned_operator_data>(std::move(input_batch),
                                                       this->get_operator_id());

  } else if (_hash_table_build_state == BUILD_HASH_TABLE_STATE::BUILT) {
    if (probe_port->repo->num_partitions() != 1) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: expected exactly 1 "
        "partition in operator " +
        std::to_string(this->get_operator_id()));
    }
    // If the hash table has already been build, we only send the probe side. The hash table should
    // already be built and we can perform the join with the probe side batches.
    std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
    auto batch = probe_port->repo->pop_next_data_batch();
    if (batch) {
      input_batch.push_back(std::move(batch));
    } else {
      SIRIUS_LOG_WARN(
        "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: expected to pop a "
        "batch from the default port but got none in operator " +
        std::to_string(this->get_operator_id()));
    }
    // Subsequent probe-only tasks share the hash table built under the initial
    // SCHEDULING task. They MUST run on the same GPU — tag with operator_id.
    return std::make_unique<partitioned_operator_data>(std::move(input_batch),
                                                       this->get_operator_id());
  } else {
    SIRIUS_LOG_WARN(fmt::format(
      "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: invalid hash table "
      "build state {} in operator {}",
      static_cast<int>(_hash_table_build_state),
      this->get_operator_id()));
    return nullptr;
  }
}

std::unique_ptr<operator_data> sirius_physical_hash_join::get_next_task_input_data()
{
  // Hold the mutex for the entire operation to prevent concurrent pop/get races.
  // A pop on one thread must not remove a batch that another thread's get expects to find.
  std::lock_guard<std::mutex> lg(op_state_mutex);

  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    return get_next_task_input_data_for_build_probe();
  }

  // One-time initialization: snapshot all batch IDs from both ports.
  if (left_batch_ids.empty() && right_batch_ids.empty()) {
    if (ports["default"]->repo->num_partitions() != ports["build"]->repo->num_partitions()) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:Number of partitions for left and right ports must be the "
        "same in operator " +
        std::to_string(this->get_operator_id()));
    }

    left_batch_ids.reserve(ports["default"]->repo->num_partitions());
    right_batch_ids.reserve(ports["build"]->repo->num_partitions());
    for (size_t i = 0; i < ports["default"]->repo->num_partitions(); i++) {
      left_batch_ids.push_back(ports["default"]->repo->get_batch_ids(i));
      right_batch_ids.push_back(ports["build"]->repo->get_batch_ids(i));
      num_batches_to_process += left_batch_ids[i].size() * right_batch_ids[i].size();
    }
  }

  if (current_partition_index >= num_batches_to_process) { return nullptr; }

  size_t batch_index = current_partition_index++;

  // Walk the partition × left × right grid to find the (left, right) pair for this batch_index.
  std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
  input_batch.reserve(2);
  size_t counter = 0;
  for (size_t partition_idx = 0; partition_idx < left_batch_ids.size(); partition_idx++) {
    size_t left_counter = 0;
    for (auto& left_batch_id : left_batch_ids[partition_idx]) {
      size_t right_counter = 0;
      for (auto& right_batch_id : right_batch_ids[partition_idx]) {
        if (counter == batch_index) {
          bool pop_left  = (right_counter == right_batch_ids[partition_idx].size() - 1);
          bool pop_right = (left_counter == left_batch_ids[partition_idx].size() - 1);
          if (pop_left) {
            input_batch.push_back(
              ports["default"]->repo->pop_data_batch_by_id(left_batch_id, partition_idx));
          } else {
            input_batch.push_back(
              ports["default"]->repo->get_data_batch_by_id(left_batch_id, partition_idx));
          }
          if (pop_right) {
            input_batch.push_back(
              ports["build"]->repo->pop_data_batch_by_id(right_batch_id, partition_idx));
          } else {
            input_batch.push_back(
              ports["build"]->repo->get_data_batch_by_id(right_batch_id, partition_idx));
          }
          // MIXED_JOIN distributes per-partition tasks across GPUs by
          // partition_idx % num_gpus. Tag with the partition index so the
          // scheduler can route by partition.
          return std::make_unique<partitioned_operator_data>(std::move(input_batch), partition_idx);
        }
        right_counter++;
        counter++;
      }
      left_counter++;
    }
  }

  if (input_batch.empty()) {
    return nullptr;
  } else {
    throw std::runtime_error("Expected to have returned already or received nothing, but got " +
                             std::to_string(input_batch.size()) + " input batches for hash join");
  }
}

/// Result of prepare_join_keys for a single join side: the key table view and any cast columns
/// that must remain alive.
struct join_side_keys_result {
  // Owned cast columns - kept alive so the table view referencing them remains valid
  std::vector<std::unique_ptr<cudf::column>> owned_cast_columns;
  cudf::table_view keys;
  // Storage for column views used to build the table_view (must outlive the table_view)
  std::vector<cudf::column_view> key_views;
};

/// Build the key table view for one side of the join.
/// If cast_necessary is false, this simply selects the key columns from the input batch.
/// If cast_necessary is true, each key column that requires a cast is cast to its target type
/// via cudf::cast before being included in the key table.
/// @param is_left_side  If true, uses cast_left/left_target_type from key_casts; otherwise uses
///                      cast_right/right_target_type.
static join_side_keys_result prepare_join_keys(
  const ::cucascade::read_only_data_batch& input_batch,
  const std::vector<cudf::size_type>& key_col_indices,
  bool cast_necessary,
  const std::vector<sirius_physical_hash_join::key_cast_info>& key_casts,
  bool is_left_side,
  rmm::cuda_stream_view stream)
{
  join_side_keys_result result;

  cudf::table_view table = get_cudf_table_view(input_batch);

  if (!cast_necessary) {
    // INVARIANT: every entry in key_col_indices must address a column in
    // `table`. PR #732 closed the only known violator — DuckDB's
    // LATE_MATERIALIZATION optimizer was rewriting `ORDER BY ... LIMIT N`
    // into a self-RIGHT_SEMI_JOIN keyed on parquet virtual columns
    // (file_index / file_row_number) that Sirius's scan path silently
    // drops, leaving the join with key_col_indices entries pointing past
    // the physical batch. Disabling that pass in
    // src/transparent/sirius_optimizer_extension.cpp removed the bad
    // emitter. If you hit this throw, a new emitter has been introduced —
    // fix it at the source rather than reintroducing the historical
    // silent filter (see PR #732 comment 3242605041 for the prior shape).
    auto const num_cols = table.num_columns();
    for (auto idx : key_col_indices) {
      if (idx >= num_cols) {
        throw std::out_of_range("prepare_join_keys: key_col_indices entry " + std::to_string(idx) +
                                " is >= input table column count " + std::to_string(num_cols) +
                                " (is_left_side=" + (is_left_side ? "true" : "false") +
                                "). The upstream emitter wired a join key that does not exist in "
                                "the physical batch — fix the emitter, do not paper over it here.");
      }
    }
    result.keys = table.select(key_col_indices);
    return result;
  }

  // Slow path: iterate over key columns and cast where needed
  for (size_t i = 0; i < key_col_indices.size(); i++) {
    const auto& cast_info = key_casts[i];
    cudf::column_view col = table.column(key_col_indices[i]);
    bool needs_cast       = is_left_side ? cast_info.cast_left : cast_info.cast_right;
    cudf::data_type target_type =
      is_left_side ? cast_info.left_target_type : cast_info.right_target_type;

    if (needs_cast) {
      auto cast_col = cudf::cast(col, target_type, stream);
      result.key_views.push_back(cast_col->view());
      result.owned_cast_columns.push_back(std::move(cast_col));
    } else {
      result.key_views.push_back(col);
    }
  }

  result.keys = cudf::table_view(result.key_views);
  return result;
}

/// Gather output columns from both sides of a join using row index vectors, then assemble the
/// result into an operator_data. Handles collect/oob policy selection based on join type.
/// @param left_indices   Row indices into left_full; may be null if the left side is not collected.
/// @param right_indices  Row indices into right_full; may be null if the right side is not
///                       collected.
/// @param memory_space   Memory space of the input batch used to tag the output data batch.
static std::unique_ptr<operator_data> gather_join_output(
  duckdb::JoinType join_type,
  cudf::table_view left_full,
  cudf::table_view right_full,
  std::vector<cudf::size_type> const& lhs_col_idxs,
  std::vector<cudf::size_type> const& rhs_col_idxs,
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices,
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_indices,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream)
{
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
    cudf::table_view left_cols_to_gather = left_full.select(lhs_col_idxs);
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
    cudf::table_view right_cols_to_gather = right_full.select(rhs_col_idxs);
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
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{
      make_data_batch(std::move(output_cudf_table), memory_space, stream)});
}

/// Assemble output for a distinct_hash_join left_join.
/// distinct_hash_join::left_join returns only build indices (one per probe row, in probe order).
/// Left (probe) columns are copied directly; right (build) columns are gathered with NULLIFY.
static std::unique_ptr<operator_data> gather_distinct_left_join_output(
  cudf::table_view left_full,
  cudf::table_view right_full,
  std::vector<cudf::size_type> const& lhs_col_idxs,
  std::vector<cudf::size_type> const& rhs_col_idxs,
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_indices,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<cudf::column>> out_cols;

  // Left (probe): all rows appear in order — copy selected columns directly.
  cudf::table_view left_cols = left_full.select(lhs_col_idxs);
  for (cudf::size_type i = 0; i < left_cols.num_columns(); i++) {
    out_cols.push_back(std::make_unique<cudf::column>(left_cols.column(i), stream));
  }

  // Right (build): gather using build_indices; unmatched entries are JoinNoneValue → NULLIFY.
  cudf::table_view right_cols = right_full.select(rhs_col_idxs);
  cudf::column_view right_map(cudf::data_type(cudf::type_id::INT32),
                              static_cast<cudf::size_type>(build_indices->size()),
                              build_indices->data(),
                              nullptr,
                              0,
                              0,
                              {});
  auto right_result =
    cudf::gather(right_cols, right_map, cudf::out_of_bounds_policy::NULLIFY, stream);
  for (auto& col : right_result->release()) {
    out_cols.push_back(std::move(col));
  }

  auto output_cudf_table = std::make_unique<cudf::table>(std::move(out_cols), stream);
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{
      make_data_batch(std::move(output_cudf_table), memory_space, stream)});
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
  ::cucascade::read_only_data_batch const& left_batch,
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
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{
      make_data_batch(std::move(output_cudf_table), *left_batch.get_memory_space(), stream)});
}

std::unique_ptr<operator_data> sirius_physical_hash_join::execute(const operator_data& input_data,
                                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_hash_join::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  if (is_all_inequality_join) {
    throw std::runtime_error(
      "Error sirius_physical_hash_join being asked to do all inequality join of type: " +
      duckdb::JoinTypeToString(join_type));
  }

  cudf::table_view left_full, right_full;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices, right_indices;

  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    if (_hash_table_build_state == BUILD_HASH_TABLE_STATE::SCHEDULED) {
      if (input_batches.size() != 2) {
        throw std::runtime_error(
          "In sirius_physical_hash_join::execute: BUILD_PROBE SCHEDULED expects probe + build "
          "batch, got " +
          std::to_string(input_batches.size()) + " batches in operator " +
          std::to_string(this->get_operator_id()));
      }
      auto const& build_batch_ro  = input_batches[1];
      auto build_keys_result      = prepare_join_keys(build_batch_ro,
                                                 right_key_col_indices,
                                                 cast_necessary,
                                                 key_casts,
                                                 /*is_left_side=*/false,
                                                 stream);
      cudf::table_view build_keys = build_keys_result.keys;
      {
        std::lock_guard<std::mutex> lg(op_state_mutex);
        _built_table_cast_columns = std::move(build_keys_result.owned_cast_columns);
        _build_table              = build_batch_ro;
        if (unique_build_keys &&
            (join_type == duckdb::JoinType::INNER || join_type == duckdb::JoinType::LEFT)) {
          _distinct_hash_table = std::make_unique<cudf::distinct_hash_join>(
            build_keys, cudf::null_equality::UNEQUAL, 0.5, stream);
          SIRIUS_LOG_DEBUG(
            "sirius_physical_hash_join id {}: using distinct_hash_join (BUILD_PROBE)",
            this->get_operator_id());
        } else {
          _hash_table =
            std::make_unique<cudf::hash_join>(build_keys, cudf::null_equality::UNEQUAL, stream);
        }
        stream.synchronize();  // Ensure the hash table is fully built before we allow any probe
                               // batches to proceed.
        _hash_table_build_state = BUILD_HASH_TABLE_STATE::BUILT;
      }
    }
    if (_hash_table_build_state == BUILD_HASH_TABLE_STATE::BUILT) {
      // Hash table is built, we can process probe batches. The probe-side keys will be processed in
      // the same way as the mixed join path, but with an equality-only predicate.
      auto probe_keys_result      = prepare_join_keys(input_batches[0],
                                                 left_key_col_indices,
                                                 cast_necessary,
                                                 key_casts,
                                                 /*is_left_side=*/true,
                                                 stream);
      cudf::table_view probe_keys = probe_keys_result.keys;

      left_full  = get_cudf_table_view(input_batches[0]);
      right_full = _build_table.value()
                     .get_data()
                     ->cast<cucascade::gpu_table_representation>()
                     .get_table_view();

      if (_distinct_hash_table) {
        // Distinct hash join path (unique build keys, INNER or LEFT only).
        if (join_type == duckdb::JoinType::INNER) {
          auto result   = _distinct_hash_table->inner_join(probe_keys, stream);
          left_indices  = std::move(result.first);
          right_indices = std::move(result.second);
        } else {
          // LEFT: returns only build indices; probe indices are implicit [0..N-1].
          auto build_indices = _distinct_hash_table->left_join(probe_keys, stream);
          return gather_distinct_left_join_output(left_full,
                                                  right_full,
                                                  lhs_output_columns.col_idxs,
                                                  rhs_output_columns.col_idxs,
                                                  std::move(build_indices),
                                                  *input_batches[0].get_memory_space(),
                                                  stream);
        }
      } else {
        if (join_type == duckdb::JoinType::INNER) {
          auto result   = _hash_table->inner_join(probe_keys, {}, stream);
          left_indices  = std::move(result.first);
          right_indices = std::move(result.second);
        } else if (join_type == duckdb::JoinType::LEFT) {
          auto result   = _hash_table->left_join(probe_keys, {}, stream);
          left_indices  = std::move(result.first);
          right_indices = std::move(result.second);
        } else if (join_type == duckdb::JoinType::OUTER) {
          auto result   = _hash_table->full_join(probe_keys, {}, stream);
          left_indices  = std::move(result.first);
          right_indices = std::move(result.second);
        } else {
          throw std::runtime_error("Unsupported join type in BUILD_PROBE mode: " +
                                   duckdb::JoinTypeToString(join_type));
        }
      }

    } else {
      throw std::runtime_error(fmt::format(
        "In sirius_physical_hash_join::execute: invalid hash table build state {} in BUILD_PROBE "
        "mode for operator id {}",
        static_cast<int>(_hash_table_build_state),
        this->get_operator_id()));
    }

  } else if (_join_mode == HASH_JOIN_MODE::MIXED_JOIN) {
    if (input_batches.size() != 2) {
      throw std::runtime_error("Expected 2 input batches for hash join, got " +
                               std::to_string(input_batches.size()) + " input batches");
    }
    left_full  = get_cudf_table_view(input_batches[0]);
    right_full = get_cudf_table_view(input_batches[1]);
    // Mixed join: equality conditions drive the hash table; inequality conditions are evaluated
    // via a cuDF AST binary predicate on the full input tables.
    auto left_keys_result     = prepare_join_keys(input_batches[0],
                                              left_key_col_indices,
                                              cast_necessary,
                                              key_casts,
                                              /*is_left_side=*/true,
                                              stream);
    auto right_keys_result    = prepare_join_keys(input_batches[1],
                                               right_key_col_indices,
                                               cast_necessary,
                                               key_casts,
                                               /*is_left_side=*/false,
                                               stream);
    cudf::table_view left_eq  = left_keys_result.keys;
    cudf::table_view right_eq = right_keys_result.keys;

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
  } else {  // STANDARD HASH JOIN
    if (input_batches.size() != 2) {
      throw std::runtime_error("Expected 2 input batches for hash join, got " +
                               std::to_string(input_batches.size()) + " input batches");
    }
    left_full                   = get_cudf_table_view(input_batches[0]);
    right_full                  = get_cudf_table_view(input_batches[1]);
    auto left_keys_result       = prepare_join_keys(input_batches[0],
                                              left_key_col_indices,
                                              cast_necessary,
                                              key_casts,
                                              /*is_left_side=*/true,
                                              stream);
    auto right_keys_result      = prepare_join_keys(input_batches[1],
                                               right_key_col_indices,
                                               cast_necessary,
                                               key_casts,
                                               /*is_left_side=*/false,
                                               stream);
    cudf::table_view left_keys  = left_keys_result.keys;
    cudf::table_view right_keys = right_keys_result.keys;

    if (unique_build_keys &&
        (join_type == duckdb::JoinType::INNER || join_type == duckdb::JoinType::LEFT)) {
      // Distinct hash join: build on right (build) keys, probe with left (probe) keys.
      cudf::distinct_hash_join dht(right_keys, cudf::null_equality::UNEQUAL, 0.5, stream);
      SIRIUS_LOG_DEBUG("sirius_physical_hash_join id {}: using distinct_hash_join (STANDARD)",
                       this->get_operator_id());
      if (join_type == duckdb::JoinType::INNER) {
        auto result   = dht.inner_join(left_keys, stream);
        left_indices  = std::move(result.first);
        right_indices = std::move(result.second);
      } else {
        // LEFT: returns only build indices; probe indices are implicit.
        auto build_indices = dht.left_join(left_keys, stream);
        return gather_distinct_left_join_output(left_full,
                                                right_full,
                                                lhs_output_columns.col_idxs,
                                                rhs_output_columns.col_idxs,
                                                std::move(build_indices),
                                                *input_batches[0].get_memory_space(),
                                                stream);
      }
    } else if (join_type == duckdb::JoinType::INNER) {
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
      auto filtered_join_object = make_right_filtered_join(right_keys, stream);
      left_indices              = filtered_join_object.semi_join(left_keys, stream);
    } else if (join_type == duckdb::JoinType::RIGHT_SEMI) {
      auto filtered_join_object = make_right_filtered_join(left_keys, stream);
      right_indices             = filtered_join_object.semi_join(right_keys, stream);
    } else if (join_type == duckdb::JoinType::ANTI) {
      auto filtered_join_object = make_right_filtered_join(right_keys, stream);
      left_indices              = filtered_join_object.anti_join(left_keys, stream);
    } else if (join_type == duckdb::JoinType::RIGHT_ANTI) {
      auto filtered_join_object = make_right_filtered_join(left_keys, stream);
      right_indices             = filtered_join_object.anti_join(right_keys, stream);
    } else if (join_type == duckdb::JoinType::MARK) {
      // MARK join: output ALL left rows + a BOOL8 column indicating match presence.
      // Use semi join to find which left rows have matches in the right table.
      auto filtered_join_object = make_right_filtered_join(right_keys, stream);
      auto semi_indices         = filtered_join_object.semi_join(left_keys, stream);
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

  return gather_join_output(join_type,
                            left_full,
                            right_full,
                            lhs_output_columns.col_idxs,
                            rhs_output_columns.col_idxs,
                            std::move(left_indices),
                            std::move(right_indices),
                            *input_batches[0].get_memory_space(),
                            stream);
}

void sirius_physical_hash_join::on_finalize_operator()
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    _hash_table.reset();
    _distinct_hash_table.reset();
    _build_table = std::nullopt;
    _built_table_cast_columns.clear();
    _hash_table_build_state = BUILD_HASH_TABLE_STATE::DESTROYED;
  }
}

}  // namespace op
}  // namespace sirius
