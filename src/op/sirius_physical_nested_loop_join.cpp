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

#include "config.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "expression_evaluator/gpu_expression_translator_internal.hpp"
#include "helper/numeric_narrowing.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cstdio>
#include <span>

namespace sirius {
namespace op {

static bool nlj_is_equality(sirius::comparison_type c)
{
  return c == sirius::comparison_type::equal || c == sirius::comparison_type::not_distinct_from;
}

// Null-safe comparisons treat NULL as an ordinary value: a NULL operand yields a definite
// TRUE/FALSE, never UNKNOWN.
static bool nlj_is_null_safe(sirius::comparison_type c)
{
  return c == sirius::comparison_type::distinct_from ||
         c == sirius::comparison_type::not_distinct_from;
}

void reorder_conditions(duckdb::vector<sirius::join_condition>& conditions)
{
  bool is_ordered     = true;
  bool seen_non_equal = false;
  for (auto& cond : conditions) {
    if (nlj_is_equality(cond.comparison)) {
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
    if (nlj_is_equality(cond.comparison)) {
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

bool sirius_physical_nested_loop_join::is_join_type_supported(duckdb::JoinType join_type)
{
  // Keep in lockstep with the `switch (join_type)` in execute() and emit_one_side_empty_result().
  switch (join_type) {
    case duckdb::JoinType::INNER:
    case duckdb::JoinType::LEFT:
    case duckdb::JoinType::RIGHT:
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::MARK:
    case duckdb::JoinType::OUTER: return true;
    // RIGHT_SEMI / RIGHT_ANTI would need the predicate rebuilt with the table references
    // swapped, SINGLE the matches deduplicated to one right row per left row; neither exists.
    default: return false;
  }
}

// Backstop for a construction site that skipped the planner's screen: throwing here still lands
// in plan generation, which falls back to CPU, rather than aborting the query from execute().
static void require_supported_join_type(duckdb::JoinType join_type)
{
  if (!sirius_physical_nested_loop_join::is_join_type_supported(join_type)) {
    throw duckdb::NotImplementedException(
      "sirius_physical_nested_loop_join: unsupported join type: " +
      duckdb::JoinTypeToString(join_type));
  }
}

sirius_physical_nested_loop_join::sirius_physical_nested_loop_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<sirius::join_condition> cond,
  duckdb::JoinType join_type,
  std::size_t estimated_cardinality)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::NESTED_LOOP_JOIN,
                                                sirius::from_duckdb_vec(op.types),
                                                estimated_cardinality),
    conditions(std::move(cond)),
    join_type(join_type)
{
  require_supported_join_type(join_type);
  reorder_conditions(conditions);

  children.push_back(std::move(left));
  children.push_back(std::move(right));
  auto& lhs_types = children[0]->get_types();
  auto& rhs_types = children[1]->get_types();
  left_output_col_idxs.reserve(lhs_types.size());
  for (std::size_t i = 0; i < lhs_types.size(); i++) {
    left_output_col_idxs.push_back(i);
  }
  right_output_col_idxs.reserve(rhs_types.size());
  for (std::size_t i = 0; i < rhs_types.size(); i++) {
    right_output_col_idxs.push_back(i);
  }
}

sirius_physical_nested_loop_join::sirius_physical_nested_loop_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<sirius::join_condition> cond,
  duckdb::JoinType join_type,
  std::size_t estimated_cardinality,
  duckdb::vector<std::size_t> left_projection_map,
  duckdb::vector<std::size_t> right_projection_map)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::NESTED_LOOP_JOIN,
                                                sirius::from_duckdb_vec(op.types),
                                                estimated_cardinality),
    conditions(std::move(cond)),
    join_type(join_type)
{
  require_supported_join_type(join_type);
  reorder_conditions(conditions);
  children.push_back(std::move(left));
  children.push_back(std::move(right));
  auto& lhs_types = children[0]->get_types();
  auto& rhs_types = children[1]->get_types();
  if (left_projection_map.empty()) {
    for (std::size_t i = 0; i < lhs_types.size(); i++) {
      left_output_col_idxs.push_back(i);
    }
  } else {
    for (std::size_t idx : left_projection_map) {
      if (idx < lhs_types.size()) { left_output_col_idxs.push_back(idx); }
    }
  }
  if (right_projection_map.empty()) {
    for (std::size_t i = 0; i < rhs_types.size(); i++) {
      right_output_col_idxs.push_back(i);
    }
  } else {
    for (std::size_t idx : right_projection_map) {
      if (idx < rhs_types.size()) { right_output_col_idxs.push_back(idx); }
    }
  }
}

bool sirius_physical_nested_loop_join::is_supported(
  const duckdb::vector<sirius::join_condition>& conditions, duckdb::JoinType join_type)
{
  if (!is_join_type_supported(join_type)) { return false; }
  if (join_type == duckdb::JoinType::MARK) { return true; }
  for (auto& cond : conditions) {
    auto left_expr = sirius::ast::to_duckdb(*cond.left);
    if (left_expr->return_type.InternalType() == duckdb::PhysicalType::STRUCT ||
        left_expr->return_type.InternalType() == duckdb::PhysicalType::LIST ||
        left_expr->return_type.InternalType() == duckdb::PhysicalType::ARRAY) {
      return false;
    }
  }
  if (join_type == duckdb::JoinType::SEMI || join_type == duckdb::JoinType::ANTI) {
    return conditions.size() == 1;
  }
  return true;
}

partition_strategy sirius_physical_nested_loop_join::get_partition_strategy(
  const partition_sizing_input& /*in*/)
{
  // A nested-loop join is never hash-partitioned: it runs on a single partition and streams both
  // sides through the cross-product, so it never broadcasts or enters build-probe.
  return {/*num_partitions=*/1, /*broadcast=*/false, /*build_probe=*/false};
}

duckdb::vector<sirius::logical_type> sirius_physical_nested_loop_join::get_join_types() const
{
  duckdb::vector<sirius::logical_type> result;
  for (auto& op : conditions) {
    auto right_expr = sirius::ast::to_duckdb(*op.right);
    result.push_back(sirius::from_duckdb(right_expr->return_type));
  }
  return result;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_nested_loop_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Mirrors sirius_physical_hash_join::build_pipelines.
  pipeline::sirius_meta_pipeline* host_meta;
  pipeline::sirius_pipeline* host_current;
  if (is_sink()) {
    auto& sink_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
    host_meta       = &sink_meta;
    host_current    = sink_meta.get_base_pipeline().get();
  } else {
    meta_pipeline.get_state().add_pipeline_operator(current, *this);
    host_meta    = &meta_pipeline;
    host_current = &current;
  }

  D_ASSERT(children.size() == 2);
  auto& build_child = *children[1];
  D_ASSERT(build_child.is_sink());
  D_ASSERT(!build_child.children.empty());
  auto& build_meta = host_meta->create_child_meta_pipeline(*host_current, build_child);
  build_meta.build(*build_child.children[0]);

  auto& probe_child = *children[0];
  D_ASSERT(probe_child.is_sink());
  D_ASSERT(!probe_child.children.empty());
  auto& probe_meta = host_meta->create_child_meta_pipeline(*host_current, probe_child);
  probe_meta.build(*probe_child.children[0]);
}

std::unique_ptr<operator_data> sirius_physical_nested_loop_join::get_next_task_input_data()
{
  // Hold the mutex for the entire operation to prevent concurrent pop/get races.
  // A pop on one thread must not remove a batch that another thread's get expects to find.
  std::lock_guard<std::mutex> lg(batches_to_processed_mutex);

  // One-time initialization: snapshot all batch IDs from both ports.
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

  if (current_partition_index >= num_batches_to_process) { return nullptr; }

  size_t batch_index = current_partition_index++;

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
            input_batch.push_back(
              default_port->repo->pop_data_batch_by_id(left_batch_id, partition_idx));
          } else {
            input_batch.push_back(
              default_port->repo->get_data_batch_by_id(left_batch_id, partition_idx));
          }
          if (left_counter == left_batch_ids[partition_idx].size() - 1) {
            input_batch.push_back(
              build_port->repo->pop_data_batch_by_id(right_batch_id, partition_idx));
          } else {
            input_batch.push_back(
              build_port->repo->get_data_batch_by_id(right_batch_id, partition_idx));
          }
          return std::make_unique<pipelineable_operator_data>(input_batch);
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

cudf::ast::ast_operator to_ast_operator(sirius::comparison_type comparison)
{
  switch (comparison) {
    using enum sirius::comparison_type;
    using enum cudf::ast::ast_operator;
    case equal: return EQUAL;
    case not_distinct_from: return NULL_EQUAL;
    case not_equal: return NOT_EQUAL;
    case distinct_from: break;  // built as NOT(NULL_EQUAL) by the caller
    case lt: return LESS;
    case gt: return GREATER;
    case le: return LESS_EQUAL;
    case ge: return GREATER_EQUAL;
  }
  throw std::runtime_error("sirius_physical_nested_loop_join: unsupported comparison type");
}

/// @brief Left-associative LOGICAL_AND over @p terms; @p chain owns the AND nodes and must be
/// pre-reserved to terms.size()-1.
const cudf::ast::expression& fold_logical_and(
  std::span<const std::reference_wrapper<const cudf::ast::expression>> terms,
  std::vector<cudf::ast::operation>& chain)
{
  for (size_t i = 1; i < terms.size(); i++) {
    const cudf::ast::expression& lhs = (i == 1) ? terms[0].get() : chain.back();
    chain.emplace_back(cudf::ast::ast_operator::LOGICAL_AND, lhs, terms[i].get());
  }
  return chain.empty() ? terms[0].get() : chain.back();
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

cudf::table_view sirius_physical_nested_loop_join::select_left_output(
  const cudf::table_view& left) const
{
  std::vector<cudf::size_type> sel;
  sel.reserve(left_output_col_idxs.size());
  for (std::size_t idx : left_output_col_idxs) {
    if (idx < static_cast<std::size_t>(left.num_columns())) {
      sel.push_back(static_cast<cudf::size_type>(idx));
    }
  }
  return left.select(sel);
}

static std::unique_ptr<cudf::column> scatter_bool(
  std::unique_ptr<cudf::column> column,
  const rmm::device_uvector<cudf::size_type>& indices,
  bool value,
  rmm::cuda_stream_view stream)
{
  if (indices.size() == 0) { return column; }
  cudf::numeric_scalar<bool> scalar(value, true, stream);
  cudf::column_view scatter_map(cudf::data_type(cudf::type_id::INT32),
                                static_cast<cudf::size_type>(indices.size()),
                                indices.data(),
                                nullptr,
                                0,
                                0,
                                {});
  auto scattered = cudf::scatter({std::ref(static_cast<const cudf::scalar&>(scalar))},
                                 scatter_map,
                                 cudf::table_view({column->view()}),
                                 stream);
  return std::move(scattered->release()[0]);
}

/// @brief MARK join output with SQL three-valued logic: every row of @p left_view passes through,
/// plus a BOOL8 mark that is true at @p true_indices, NULL at rows in @p maybe_indices (the
/// "predicate IS NOT FALSE" semi-join) but not in @p true_indices, and false elsewhere.
///
/// Callers pass the projection-selected left view; the index sets index original left rows, which
/// stay valid because selection drops columns only. @p telemetry_info links the emitted batch into
/// the query's telemetry lineage.
static std::unique_ptr<operator_data> resolve_mark_join_result(
  const rmm::device_uvector<cudf::size_type>& true_indices,
  const rmm::device_uvector<cudf::size_type>& maybe_indices,
  const cudf::table_view& left_view,
  cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(left_view.num_columns() + 1);
  for (cudf::size_type i = 0; i < left_view.num_columns(); i++) {
    out_cols.push_back(std::make_unique<cudf::column>(left_view.column(i), stream));
  }

  auto num_rows = left_view.num_rows();

  cudf::numeric_scalar<bool> false_scalar(false, true, stream);
  auto mark_column = cudf::make_column_from_scalar(false_scalar, num_rows, stream);
  mark_column      = scatter_bool(std::move(mark_column), true_indices, true, stream);

  // validity == matched OR NOT maybe; false cells become NULL via bools_to_mask.
  cudf::numeric_scalar<bool> true_scalar(true, true, stream);
  auto validity_col = cudf::make_column_from_scalar(true_scalar, num_rows, stream);
  validity_col      = scatter_bool(std::move(validity_col), maybe_indices, false, stream);
  validity_col      = scatter_bool(std::move(validity_col), true_indices, true, stream);

  auto [null_mask, null_count] = cudf::bools_to_mask(validity_col->view(), stream);
  if (null_count > 0) { mark_column->set_null_mask(std::move(*null_mask), null_count); }

  out_cols.push_back(std::move(mark_column));

  auto output_table = std::make_unique<cudf::table>(std::move(out_cols));
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{
      make_data_batch(std::move(output_table), space, stream, telemetry_info)});
}

std::unique_ptr<operator_data> sirius_physical_nested_loop_join::emit_one_side_empty_result(
  const cudf::table_view& left,
  const cudf::table_view& right,
  bool left_side_empty,
  cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream)
{
  auto mr                       = space.get_default_allocator();
  auto const num_surviving_rows = left_side_empty ? right.num_rows() : left.num_rows();

  if (join_type == duckdb::JoinType::MARK) {
    // Empty left emits 0 rows (schema kept); empty right marks every left row false — the OR over
    // an empty set of right rows is FALSE, never NULL, so both index sets stay empty.
    rmm::device_uvector<cudf::size_type> no_matches(0, stream);
    rmm::device_uvector<cudf::size_type> no_maybe(0, stream);
    return resolve_mark_join_result(
      no_matches, no_maybe, select_left_output(left), space, stream, batch_telemetry());
  }

  // Gather maps per the §3 semantics table, filled on the task stream (cudf::sequence /
  // make_column_from_scalar run the device fill; this TU is host-compiled, so raw thrust
  // device algorithms are not available here). -1 entries gathered from the 0-row empty-side
  // table become NULL rows under the NULLIFY policy.
  auto iota = [&]() -> std::unique_ptr<cudf::column> {
    cudf::numeric_scalar<cudf::size_type> init(0, true, stream);
    return cudf::sequence(num_surviving_rows, init, stream, mr);
  };
  auto pad = [&]() -> std::unique_ptr<cudf::column> {
    cudf::numeric_scalar<cudf::size_type> minus_one(-1, true, stream);
    return cudf::make_column_from_scalar(minus_one, num_surviving_rows, stream, mr);
  };
  auto none = [&]() -> std::unique_ptr<cudf::column> {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  };

  std::unique_ptr<cudf::column> left_map, right_map;
  switch (join_type) {
    case duckdb::JoinType::LEFT:
      left_map  = left_side_empty ? none() : iota();
      right_map = left_side_empty ? none() : pad();
      break;
    case duckdb::JoinType::RIGHT:
      left_map  = left_side_empty ? pad() : none();
      right_map = left_side_empty ? iota() : none();
      break;
    case duckdb::JoinType::OUTER:
      left_map  = left_side_empty ? pad() : iota();
      right_map = left_side_empty ? iota() : pad();
      break;
    case duckdb::JoinType::INNER:
      left_map  = none();
      right_map = none();
      break;
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI: {
      // Mirror execute(): SEMI/ANTI output the projected left columns. An empty side
      // means no matches — SEMI keeps nothing; ANTI keeps every left row when the right
      // side is the empty one.
      bool const keep_all = (join_type == duckdb::JoinType::ANTI) && !left_side_empty;
      auto left_map_col   = keep_all ? iota() : none();
      auto gathered       = cudf::gather(select_left_output(left),
                                   left_map_col->view(),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   mr);
      return std::make_unique<pipelineable_operator_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{
          make_data_batch(std::move(gathered), space, stream, batch_telemetry())});
    }
    default:
      throw std::runtime_error("sirius_physical_nested_loop_join: unsupported join type: " +
                               duckdb::JoinTypeToString(join_type));
  }

  auto left_out_of_bounds =
    (join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::OUTER)
      ? cudf::out_of_bounds_policy::NULLIFY
      : cudf::out_of_bounds_policy::DONT_CHECK;
  auto right_out_of_bounds =
    (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::OUTER)
      ? cudf::out_of_bounds_policy::NULLIFY
      : cudf::out_of_bounds_policy::DONT_CHECK;

  auto left_gathered  = cudf::gather(left, left_map->view(), left_out_of_bounds, stream, mr);
  auto right_gathered = cudf::gather(right, right_map->view(), right_out_of_bounds, stream, mr);
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  auto left_released  = left_gathered->release();
  auto right_released = right_gathered->release();
  out_cols.reserve(left_output_col_idxs.size() + right_output_col_idxs.size());
  for (std::size_t idx : left_output_col_idxs) {
    if (idx < left_released.size()) { out_cols.push_back(std::move(left_released[idx])); }
  }
  for (std::size_t idx : right_output_col_idxs) {
    if (idx < right_released.size()) { out_cols.push_back(std::move(right_released[idx])); }
  }
  auto result_table = std::make_unique<cudf::table>(std::move(out_cols));
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{
      make_data_batch(std::move(result_table), space, stream, batch_telemetry())});
}

std::unique_ptr<operator_data> sirius_physical_nested_loop_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_nested_loop_join::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  size_t pipeline_id = (this->get_pipeline() != nullptr) ? this->get_pipeline()->get_pipeline_id()
                                                         : static_cast<size_t>(-1);
  SIRIUS_LOG_DEBUG(
    "Pipeline {}: nested loop join, {} input batches", pipeline_id, input_batches.size());

  if (input_batches.size() != 2) {
    throw std::runtime_error(
      "sirius_physical_nested_loop_join expects 2 input batches (left, right), got " +
      std::to_string(input_batches.size()));
  }

  auto const& left_batch  = input_batches[0];
  auto const& right_batch = input_batches[1];

  cudf::table_view left  = get_cudf_table_view(left_batch);
  cudf::table_view right = get_cudf_table_view(right_batch);

  cucascade::memory::memory_space* space = left_batch.get_memory_space();
  if (!space) {
    SIRIUS_LOG_DEBUG(
      "Pipeline {}: nested loop join, 0 output batches because left batch had no memory space",
      pipeline_id);
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  auto mr = space->get_default_allocator();

  if (left.num_rows() == 0 || right.num_rows() == 0) {
    // A real 0-row batch on one side (e.g. an all-pruned scan under the empty-split fallback)
    // must produce the same join-type-correct output as a dead side — LEFT/RIGHT/OUTER pad the
    // preserved rows, ANTI keeps them, MARK marks them false — not an unconditionally empty
    // table.
    SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, one input side empty", pipeline_id);
    return emit_one_side_empty_result(left, right, left.num_rows() == 0, *space, stream);
  }

  std::unique_ptr<cudf::table> result_table;

  if (conditions.empty()) {
    auto cross         = cudf::cross_join(left, right, stream, mr);
    auto left_released = cross->release();
    const auto left_n  = static_cast<std::size_t>(left.num_columns());
    const auto right_n = static_cast<std::size_t>(right.num_columns());
    std::vector<std::unique_ptr<cudf::column>> out_cols;
    out_cols.reserve(left_output_col_idxs.size() + right_output_col_idxs.size());
    for (std::size_t idx : left_output_col_idxs) {
      if (idx < left_n && idx < left_released.size()) {
        out_cols.push_back(std::move(left_released[idx]));
      }
    }
    for (std::size_t idx : right_output_col_idxs) {
      if (idx < right_n && left_n + idx < left_released.size()) {
        out_cols.push_back(std::move(left_released[left_n + idx]));
      }
    }
    result_table = std::make_unique<cudf::table>(std::move(out_cols));
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
    std::vector<cudf::ast::operation> distinct_inner_ops;  // NULL_EQUAL nodes under distinct_from
    std::vector<cudf::ast::operation> and_chain;
    left_refs.reserve(conditions.size());
    right_refs.reserve(conditions.size());
    cond_ops.reserve(conditions.size());
    distinct_inner_ops.reserve(conditions.size());
    and_chain.reserve(conditions.size() > 1 ? conditions.size() - 1 : 0);
    std::vector<cudf::column_view> left_col_views, right_col_views;
    std::vector<std::unique_ptr<cudf::column>> intermediates_scope_holder;
    std::vector<std::unique_ptr<cudf::table>> expression_res_scope_hodler;
    left_col_views.reserve(left.num_columns());
    right_col_views.reserve(right.num_columns());

    // Resolves one side of a join condition to a column index in col_views, evaluating or casting
    // as needed. Returns the index to use as the cudf::ast::column_reference offset.
    auto resolve_join_col = [&](const duckdb::Expression& expr,
                                const sirius::ast::node& ast_expr,
                                std::map<uint64_t, cudf::size_type>& expr_to_idx,
                                const ::cucascade::read_only_data_batch& batch,
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
        sirius::expression_evaluator evaluator(&ast_expr, mr, stream);
        auto expr_result_table = evaluator.evaluate(table);
        auto expr_view         = expr_result_table->view();
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
        expression_res_scope_hodler.push_back(std::move(expr_result_table));
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
            sirius::cast_through_rep(table.column(source_idx), target_type, stream));
          col_views.push_back(intermediates_scope_holder.back()->view());
        } else {
          col_views.push_back(table.column(source_idx));
        }
      }
      return join_input_index;
    };

    for (const auto& cond : conditions) {
      auto const* left_node                 = cond.left.get();
      auto const* right_node                = cond.right.get();
      auto left_owned                       = sirius::ast::to_duckdb(*left_node);
      auto right_owned                      = sirius::ast::to_duckdb(*right_node);
      cudf::size_type left_join_input_index = resolve_join_col(
        *left_owned, *left_node, left_expressions_to_idx, left_batch, left, left_col_views, "left");
      cudf::size_type right_join_input_index = resolve_join_col(*right_owned,
                                                                *right_node,
                                                                right_expressions_to_idx,
                                                                right_batch,
                                                                right,
                                                                right_col_views,
                                                                "right");

      left_refs.emplace_back(left_join_input_index, cudf::ast::table_reference::LEFT);
      right_refs.emplace_back(right_join_input_index, cudf::ast::table_reference::RIGHT);
      if (cond.comparison == sirius::comparison_type::distinct_from) {
        // IS DISTINCT FROM is null-safe (NULL vs 5 is TRUE, NULL vs NULL is FALSE) but cuDF's
        // NOT_EQUAL is null-propagating, so build NOT(NULL_EQUAL(l, r)) instead.
        distinct_inner_ops.emplace_back(
          cudf::ast::ast_operator::NULL_EQUAL, left_refs.back(), right_refs.back());
        cond_ops.emplace_back(cudf::ast::ast_operator::NOT, distinct_inner_ops.back());
      } else {
        cond_ops.emplace_back(
          to_ast_operator(cond.comparison), left_refs.back(), right_refs.back());
      }
    }

    cudf::table_view left_effective(left_col_views);
    cudf::table_view right_effective(right_col_views);

    const std::vector<std::reference_wrapper<const cudf::ast::expression>> cond_terms(
      cond_ops.begin(), cond_ops.end());
    const cudf::ast::expression& predicate = fold_logical_and(cond_terms, and_chain);

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
        auto gathered = cudf::gather(
          select_left_output(left), left_map, cudf::out_of_bounds_policy::NULLIFY, stream, mr);
        SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
        return std::make_unique<pipelineable_operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{
            make_data_batch(std::move(gathered), *space, stream, batch_telemetry())});
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
        auto gathered = cudf::gather(
          select_left_output(left), left_map, cudf::out_of_bounds_policy::NULLIFY, stream, mr);
        SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
        return std::make_unique<pipelineable_operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{
            make_data_batch(std::move(gathered), *space, stream, batch_telemetry())});
      }
      case duckdb::JoinType::MARK: {
        auto true_indices = cudf::conditional_left_semi_join(
          left_effective, right_effective, predicate, std::nullopt, stream, mr);

        // Three-valued MARK: an unmatched row is NULL only when the predicate was UNKNOWN (never
        // TRUE) for some right row; a second semi-join on "predicate IS NOT FALSE" finds those.
        // (AND ci) IS NOT FALSE == AND_i (ci IS NOT FALSE), where a null-propagating ci becomes
        // ci OR IS_NULL(left) OR IS_NULL(right) (Kleene NULL_LOGICAL_OR) and a null-safe ci is
        // never UNKNOWN, so it stays ci itself.
        std::vector<cudf::ast::operation> isnull_left_ops;
        std::vector<cudf::ast::operation> isnull_right_ops;
        std::vector<cudf::ast::operation> notfalse_inner_ops;
        std::vector<cudf::ast::operation> notfalse_or_ops;
        std::vector<cudf::ast::operation> notfalse_and_chain;
        std::vector<std::reference_wrapper<const cudf::ast::expression>> notfalse_terms;
        isnull_left_ops.reserve(cond_ops.size());
        isnull_right_ops.reserve(cond_ops.size());
        notfalse_inner_ops.reserve(cond_ops.size());
        notfalse_or_ops.reserve(cond_ops.size());
        notfalse_and_chain.reserve(cond_ops.size() > 1 ? cond_ops.size() - 1 : 0);
        notfalse_terms.reserve(cond_ops.size());
        for (size_t i = 0; i < cond_ops.size(); i++) {
          if (nlj_is_null_safe(conditions[i].comparison)) {
            notfalse_terms.emplace_back(cond_ops[i]);
            continue;
          }
          isnull_left_ops.emplace_back(cudf::ast::ast_operator::IS_NULL, left_refs[i]);
          isnull_right_ops.emplace_back(cudf::ast::ast_operator::IS_NULL, right_refs[i]);
          notfalse_inner_ops.emplace_back(
            cudf::ast::ast_operator::NULL_LOGICAL_OR, cond_ops[i], isnull_left_ops.back());
          notfalse_or_ops.emplace_back(cudf::ast::ast_operator::NULL_LOGICAL_OR,
                                       notfalse_inner_ops.back(),
                                       isnull_right_ops.back());
          notfalse_terms.emplace_back(notfalse_or_ops.back());
        }
        const cudf::ast::expression& not_false_predicate =
          fold_logical_and(notfalse_terms, notfalse_and_chain);
        auto maybe_indices = cudf::conditional_left_semi_join(
          left_effective, right_effective, not_false_predicate, std::nullopt, stream, mr);

        SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
        return resolve_mark_join_result(*true_indices,
                                        *maybe_indices,
                                        select_left_output(left),
                                        *space,
                                        stream,
                                        batch_telemetry());
      }
      case duckdb::JoinType::OUTER:
        join_result =
          cudf::conditional_full_join(left_effective, right_effective, predicate, stream, mr);
        break;
      // Unreachable: is_join_type_supported() screens these out at plan time and at construction.
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
    for (std::size_t idx : left_output_col_idxs) {
      if (idx < left_released.size()) { out_cols.push_back(std::move(left_released[idx])); }
    }
    for (std::size_t idx : right_output_col_idxs) {
      if (idx < right_released.size()) { out_cols.push_back(std::move(right_released[idx])); }
    }
    result_table = std::make_unique<cudf::table>(std::move(out_cols));
  }

  SIRIUS_LOG_DEBUG("Pipeline {}: nested loop join, 1 output batches", pipeline_id);
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{
      make_data_batch(std::move(result_table), *space, stream, batch_telemetry())});
}

}  // namespace op
}  // namespace sirius
