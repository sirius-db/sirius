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
#include "expression_evaluator/expression_evaluator.hpp"
#include "expression_evaluator/gpu_expression_translator_internal.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/cross_join_slicing.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

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
std::string_view sirius_physical_nested_loop_join::input_port_for(
  sirius_physical_operator const& producer) const
{
  if (producer.type == SiriusPhysicalOperatorType::CONCAT) {
    return producer.Cast<sirius_physical_concat>().is_build_concat() ? "build" : "default";
  }
  return sirius_physical_operator::input_port_for(producer);
}

bool sirius_physical_nested_loop_join::is_supported(
  const duckdb::vector<sirius::join_condition>& conditions, duckdb::JoinType join_type)
{
  if (!is_join_type_supported(join_type)) { return false; }
  if (join_type == duckdb::JoinType::MARK) { return true; }
  for (auto& cond : conditions) {
    auto const id = cond.left->return_type().id();
    if (id == sirius::type_id::STRUCT || id == sirius::type_id::LIST ||
        id == sirius::type_id::ARRAY) {
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
  // sides through the cross-product, so it never broadcasts or enters build-probe. No GPU state
  // is shared across tasks; leave the CONCATs and join tasks free to follow input locality.
  return partition_strategy{/*num_partitions=*/1,
                            /*broadcast=*/false,
                            /*build_probe=*/false,
                            partition_placement::unpinned(1)};
}

duckdb::vector<sirius::logical_type> sirius_physical_nested_loop_join::get_join_types() const
{
  duckdb::vector<sirius::logical_type> result;
  for (auto& op : conditions) {
    result.push_back(op.right->return_type());
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
  auto* default_port = get_port("default");
  auto* build_port   = get_port("build");

  // One-time initialization: snapshot all batch IDs from both ports.
  if (left_batch_ids.empty() && right_batch_ids.empty()) {
    if (!default_port || !default_port->repo || !build_port || !build_port->repo) {
      return nullptr;
    }
    if (default_port->repo->num_partitions() != build_port->repo->num_partitions()) {
      throw std::runtime_error(
        "sirius_physical_nested_loop_join: number of partitions for default and build ports must "
        "match");
    }
    auto const sizes = [&](cucascade::shared_data_repository& repo,
                           std::vector<uint64_t> const& ids,
                           std::size_t partition_idx) {
      std::vector<batch_rows_and_bytes> result;
      result.reserve(ids.size());
      for (uint64_t id : ids) {
        auto batch = repo.get_data_batch_by_id(id, partition_idx);
        // Takes a read lock on the batch while batches_to_processed_mutex is held. It runs before
        // the join has tasks, so the only exclusive holder can be a spill, which never takes this
        // mutex, and the wait is at most one spill copy. A non-blocking read would leave the pair
        // unsplit.
        result.push_back(batch ? get_batch_rows_and_bytes(*batch) : batch_rows_and_bytes{});
      }
      return result;
    };
    auto const num_partitions = default_port->repo->num_partitions();
    left_batch_ids.reserve(num_partitions);
    right_batch_ids.reserve(num_partitions);
    for (size_t p = 0; p < num_partitions; p++) {
      left_batch_ids.push_back(default_port->repo->get_batch_ids(p));
      right_batch_ids.push_back(build_port->repo->get_batch_ids(p));
      // Only a cross product is split, so only its batch sizes are read.
      if (conditions.empty()) {
        left_batch_sizes.push_back(sizes(*default_port->repo, left_batch_ids[p], p));
        right_batch_sizes.push_back(sizes(*build_port->repo, right_batch_ids[p], p));
      }
    }
  }

  // Skip partitions without a pair of batches.
  while (
    next_task.partition < left_batch_ids.size() &&
    (left_batch_ids[next_task.partition].empty() || right_batch_ids[next_task.partition].empty())) {
    next_task.partition++;
  }
  if (next_task.partition >= left_batch_ids.size()) { return nullptr; }

  auto const& left_ids  = left_batch_ids[next_task.partition];
  auto const& right_ids = right_batch_ids[next_task.partition];
  if (next_task.num_slices == 0) {
    next_task.num_slices =
      conditions.empty()
        ? cross_join_num_slices(left_batch_sizes[next_task.partition][next_task.left],
                                right_batch_sizes[next_task.partition][next_task.right],
                                cross_join_task_bytes)
        : std::size_t{1};
  }
  auto const task = next_task;
  if (++next_task.slice == next_task.num_slices) {
    next_task.slice      = 0;
    next_task.num_slices = 0;
    if (++next_task.right == right_ids.size()) {
      next_task.right = 0;
      if (++next_task.left == left_ids.size()) {
        next_task.left = 0;
        next_task.partition++;
      }
    }
  }

  // Tasks are handed out in order, so a left batch is last used by the last slice of its pair with
  // the last right batch, and a right batch by the last slice of its pair with the last left batch.
  bool const last_slice = task.slice + 1 == task.num_slices;
  bool const pop_left   = last_slice && task.right + 1 == right_ids.size();
  bool const pop_right  = last_slice && task.left + 1 == left_ids.size();

  std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
  input_batch.reserve(2);
  input_batch.push_back(
    pop_left ? default_port->repo->pop_data_batch_by_id(left_ids[task.left], task.partition)
             : default_port->repo->get_data_batch_by_id(left_ids[task.left], task.partition));
  input_batch.push_back(
    pop_right ? build_port->repo->pop_data_batch_by_id(right_ids[task.right], task.partition)
              : build_port->repo->get_data_batch_by_id(right_ids[task.right], task.partition));
  if (conditions.empty()) {
    return std::make_unique<cross_join_slice_data>(
      std::move(input_batch), task.slice, task.num_slices);
  }
  return std::make_unique<pipelineable_operator_data>(input_batch);
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
  ::cuda::stream_ref stream)
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
  ::cuda::stream_ref stream,
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
  ::cuda::stream_ref stream)
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
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_nested_loop_join::execute"};
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
    // The task joins one slice of the left batch with the right batch.
    auto const all_left_rows = static_cast<std::size_t>(left.num_rows());
    if (auto const* slice_data = dynamic_cast<const cross_join_slice_data*>(&input_data);
        slice_data != nullptr && slice_data->num_slices > 1) {
      auto const begin = all_left_rows * slice_data->slice / slice_data->num_slices;
      auto const end   = all_left_rows * (slice_data->slice + 1) / slice_data->num_slices;
      left =
        cudf::slice(
          left, {static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(end)}, stream)
          .front();
    }
    auto const left_rows  = static_cast<std::size_t>(left.num_rows());
    auto const right_rows = static_cast<std::size_t>(right.num_rows());

    // An output that exceeds the memory space can never be allocated, so fail instead of
    // rescheduling the task on every out-of-memory error until the retry limit. The output holds
    // every left column once per right row and every right column once per left row.
    auto const left_bytes =
      memory::saturating_mul(left_batch.get_data()->get_size_in_bytes(), left_rows) / all_left_rows;
    auto const output_bytes = memory::saturating_add(
      memory::saturating_mul(left_bytes, right_rows),
      memory::saturating_mul(right_batch.get_data()->get_size_in_bytes(), left_rows));
    auto const max_bytes = space->get_max_memory();
    if (max_bytes > 0 && output_bytes > max_bytes) {
      throw sirius::not_implemented_exception(
        "Cross join of {} x {} rows needs {} bytes, more than the {} bytes of GPU memory",
        left.num_rows(),
        right.num_rows(),
        output_bytes,
        max_bytes);
    }
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
    // Evaluate condition operands before building the cuDF predicate (cuDF requires matching
    // operand types). Keep the evaluated tables alive until the join completes.
    // Reserve to the exact number of conditions to prevent reallocation.
    // cudf::ast::operation stores operands as reference_wrapper<expression const> — any
    // reallocation of these vectors invalidates the stored references and causes UB/segfault.

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
    std::vector<std::unique_ptr<cudf::table>> evaluated_operands;
    left_col_views.reserve(conditions.size());
    right_col_views.reserve(conditions.size());
    evaluated_operands.reserve(2 * conditions.size());

    // Each condition owns its evaluated operands. Reuse a direct column only when it already
    // satisfies the reference's physical contract; narrowed carriers and every operation go
    // through native evaluation. Hash-only expression reuse cannot establish equivalence.
    auto resolve_join_col = [&](const sirius::ast::node& expr,
                                const cudf::table_view& table,
                                std::vector<cudf::column_view>& col_views,
                                const char* side) -> cudf::size_type {
      auto const join_input_index = static_cast<cudf::size_type>(col_views.size());
      if (expr.holds<sirius::ast::reference>()) {
        auto const& ref   = expr.get<sirius::ast::reference>();
        auto const column = table.column(ref.column_index);
        if (column.type() == sirius::get_cudf_type(ref.return_type())) {
          col_views.push_back(column);
          return join_input_index;
        }
      }
      sirius::expression_evaluator evaluator(&expr,
                                             mr,
                                             stream,
                                             strategy_from_config(),
                                             expression_evaluator::default_min_ast_size,
                                             like_swar_fastpath_enabled(),
                                             like_cache());
      auto result     = evaluator.evaluate(table);
      auto const view = result->view();
      if (view.num_columns() != 1 || view.num_rows() != table.num_rows()) {
        throw std::runtime_error(std::string("sirius_physical_nested_loop_join: operand on ") +
                                 side + " must produce one column with the input row count");
      }
      col_views.push_back(view.column(0));
      evaluated_operands.push_back(std::move(result));
      return join_input_index;
    };

    for (const auto& cond : conditions) {
      auto const left_join_input_index = resolve_join_col(*cond.left, left, left_col_views, "left");
      auto const right_join_input_index =
        resolve_join_col(*cond.right, right, right_col_views, "right");

      // RIGHT is executed as a left join with the input tables swapped below. Keep
      // each operand attached to its original table; swapping the result maps alone
      // does not preserve an asymmetric predicate such as left.x < right.y.
      const bool swap_sides = join_type == duckdb::JoinType::RIGHT;
      left_refs.emplace_back(
        left_join_input_index,
        swap_sides ? cudf::ast::table_reference::RIGHT : cudf::ast::table_reference::LEFT);
      right_refs.emplace_back(
        right_join_input_index,
        swap_sides ? cudf::ast::table_reference::LEFT : cudf::ast::table_reference::RIGHT);
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
