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

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "late_mat/pin_uniqueness.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/dynamic_filter/detail/accumulated_bloom_builder.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/dynamic_filter/build_filter_evidence.hpp"
#include "planner/dynamic_filter/build_key_domain.hpp"
#include "planner/dynamic_filter/dynamic_filter_key_admission.hpp"
#include "planner/dynamic_filter/dynamic_filter_target_discovery.hpp"
#include "planner/join_implied_filters.hpp"
#include "planner/join_key_nullability.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_join_utils.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_hash_join> make_hash_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> left,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> right,
  duckdb::vector<sirius::join_condition> conditions,
  duckdb::JoinType join_type,
  sirius::operator_params const& params,
  hash_join_options options)
{
  auto join =
    duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(op,
                                                             std::move(left),
                                                             std::move(right),
                                                             std::move(conditions),
                                                             join_type,
                                                             options.left_projection_map,
                                                             options.right_projection_map,
                                                             std::move(options.delim_types),
                                                             op.estimated_cardinality,
                                                             params.max_build_hash_table_bytes,
                                                             std::move(options.dynamic_filter_plan),
                                                             params.hash_partition_bytes,
                                                             params.max_broadcast_join_size,
                                                             options.dynamic_filter_stats_sink);
  join->mark_join_build_switch_ratio   = params.mark_join_build_switch_ratio;
  join->runtime_distinct_build_probe   = params.enable_runtime_distinct_build_probe;
  join->max_build_probe_resident_bytes = params.max_build_probe_resident_bytes;
  return join;
}

/// Whether @p get is a plain parquet scan whose output columns are the files' own columns, read by
/// name. A pin proves something about those columns, so a scan that maps, renames, merges or
/// synthesizes columns (iceberg_scan resolves them by field id and fills in added columns and
/// partition values; union_by_name and an explicit schema merge and remap the files') may return
/// different values under the same name. An allowlist, so a scan this does not know is not proven.
static bool returns_raw_parquet_columns(duckdb::LogicalGet const& get)
{
  auto const function_name = get.function.GetName().GetIdentifierName();
  if (function_name != "read_parquet" && function_name != "parquet_scan") { return false; }
  auto const* multi_file = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get());
  if (multi_file == nullptr) { return false; }
  auto const& options = multi_file->file_options;
  return !options.union_by_name && options.schema.empty() && options.bound_schema.empty() &&
         options.mapping == duckdb::MultiFileColumnMappingMode::BY_NAME;
}

/// Alternative unique keys of an operator's output: each set of output column indices is a key on
/// its own, so a join whose equality keys include any one of them has unique keys on that side.
using unique_keys = std::vector<std::unordered_set<duckdb::idx_t>>;

/// Whether @p columns include every column of at least one of @p keys.
static bool includes_unique_key(unique_keys const& keys,
                                std::unordered_set<duckdb::idx_t> const& columns)
{
  return std::any_of(keys.begin(), keys.end(), [&](auto const& key) {
    return !key.empty() &&
           std::all_of(key.begin(), key.end(), [&](auto col) { return columns.contains(col); });
  });
}

/// Remaps every key through @p map_column (child position to output position, or nullopt when the
/// column does not reach the output), keeping only the keys whose columns all reach the output.
template <typename MapColumn>
static unique_keys remap_unique_keys(unique_keys const& child_keys, MapColumn&& map_column)
{
  unique_keys mapped;
  for (auto const& key : child_keys) {
    std::unordered_set<duckdb::idx_t> mapped_key;
    for (auto const col : key) {
      auto const out = map_column(col);
      if (!out) { break; }
      mapped_key.insert(*out);
    }
    if (mapped_key.size() == key.size()) { mapped.push_back(std::move(mapped_key)); }
  }
  return mapped;
}

/// Output position of the first output column produced from child column @p child_col through
/// @p projection_map (output[i] = child[projection_map[i]]), or nullopt.
template <typename ProjectionMap>
static std::optional<duckdb::idx_t> through_projection_map(ProjectionMap const& projection_map,
                                                           duckdb::idx_t child_col)
{
  for (duckdb::idx_t i = 0; i < projection_map.size(); i++) {
    if (projection_map[i] == child_col) { return i; }
  }
  return std::nullopt;
}

/// Output positions of a parquet scan's columns that the pinned copy of the same files proved
/// distinct at pin time (see late_mat::unique_probe: no nulls and no repeated value across the
/// whole table), each a single-column key. Only the scan's own files count, so a filter or
/// projection above the scan keeps the fact valid: a subset of distinct values is distinct.
static unique_keys prove_unique_from_pin(duckdb::SiriusContext* sirius_context,
                                         duckdb::LogicalGet& get)
{
  if (sirius_context == nullptr || !returns_raw_parquet_columns(get)) { return {}; }
  auto const files = resolve_parquet_scan_file_paths(
    get.function.GetName().GetIdentifierName(), get.bind_data.get(), get.parameters);
  if (files.empty()) { return {}; }
  auto const entry = sirius_context->get_scan_manager().find_pinned_entry_for_parquet_files(files);
  if (!entry) { return {}; }
  auto const& pinned_names = entry->cache_info.names;
  auto const proven_unique = [&](std::string const& name) {
    for (std::size_t entry_idx = 0; entry_idx < pinned_names.size(); ++entry_idx) {
      if (pinned_names[entry_idx] == name) {
        return entry_idx < entry->proven_unique_columns.size() &&
               entry->proven_unique_columns[entry_idx];
      }
    }
    return false;
  };

  // A Hive partition column takes its value from the directory name, even when the files also hold
  // a column of that name, so what the pin proved about the file column says nothing about it.
  std::unordered_set<duckdb::idx_t> hive_partition_columns;
  if (auto const* multi_file =
        dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get())) {
    for (auto const& partition : multi_file->reader_bind.hive_partitioning_indexes) {
      hive_partition_columns.insert(partition.index);
    }
  }

  auto const& column_ids  = get.GetColumnIds();
  auto const& proj_ids    = get.projection_ids;
  auto const output_count = proj_ids.empty() ? column_ids.size() : proj_ids.size();
  unique_keys keys;
  for (duckdb::idx_t output_idx = 0; output_idx < output_count; ++output_idx) {
    auto const column_idx           = proj_ids.empty() ? output_idx : proj_ids[output_idx];
    auto const primary_column_index = column_ids[column_idx].GetPrimaryIndex();
    // Virtual columns (row ids and the like) have no pinned counterpart.
    if (primary_column_index >= get.names.size()) { continue; }
    if (hive_partition_columns.contains(primary_column_index)) { continue; }
    auto const& column_name = get.names[primary_column_index].GetIdentifierName();
    if (proven_unique(column_name)) {
      SIRIUS_LOG_DEBUG("[join] column '{}' proven distinct by the pinned table", column_name);
      keys.push_back({output_idx});
    }
  }
  return keys;
}

/// Returns the unique keys proven for the given logical operator subtree's output, or none if
/// uniqueness cannot be proven.
static unique_keys prove_unique_columns(duckdb::SiriusContext* sirius_context,
                                        duckdb::LogicalOperator& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET: {
      auto& get  = op.Cast<duckdb::LogicalGet>();
      auto table = get.GetTable();
      if (!table) { return prove_unique_from_pin(sirius_context, get); }

      const auto& constraints = table->GetConstraints();
      const auto& column_ids  = get.GetColumnIds();
      const auto& proj_ids    = get.projection_ids;

      // Build map: table column logical index → LogicalGet output position.
      std::unordered_map<duckdb::idx_t, duckdb::idx_t> col_to_output;
      if (proj_ids.empty()) {
        for (duckdb::idx_t i = 0; i < column_ids.size(); i++) {
          col_to_output[column_ids[i].GetPrimaryIndex()] = i;
        }
      } else {
        for (duckdb::idx_t i = 0; i < proj_ids.size(); i++) {
          col_to_output[column_ids[proj_ids[i]].GetPrimaryIndex()] = i;
        }
      }

      // Find the smallest PRIMARY KEY constraint whose columns are all present in the output.
      // Only PK is used (not plain UNIQUE) because PK guarantees NOT NULL — a nullable UNIQUE
      // column can contain duplicate NULLs which would violate distinct_hash_join's contract.
      std::unordered_set<duckdb::idx_t> best;
      for (const auto& constraint : constraints) {
        if (constraint->type != duckdb::ConstraintType::UNIQUE) { continue; }
        auto& unique = constraint->Cast<duckdb::UniqueConstraint>();
        if (!unique.IsPrimaryKey()) { continue; }
        auto logical_indexes = unique.GetLogicalIndexes(table->GetColumns());

        std::unordered_set<duckdb::idx_t> candidate;
        bool all_present = true;
        for (const auto& idx : logical_indexes) {
          auto it = col_to_output.find(idx.index);
          if (it == col_to_output.end()) {
            all_present = false;
            break;
          }
          candidate.insert(it->second);
        }
        if (all_present && !candidate.empty() && (best.empty() || candidate.size() < best.size())) {
          best = std::move(candidate);
        }
      }
      if (best.empty()) { return {}; }
      return {std::move(best)};
    }

    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
      auto& aggr = op.Cast<duckdb::LogicalAggregate>();
      // Only single grouping set — no CUBE/ROLLUP/GROUPING SETS.
      if (aggr.grouping_sets.size() > 1 || aggr.groups.empty()) { return {}; }
      std::unordered_set<duckdb::idx_t> unique_set;
      for (duckdb::idx_t i = 0; i < aggr.groups.size(); i++) {
        unique_set.insert(i);
      }
      return {std::move(unique_set)};
    }

    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET: {
      // A delim scan replays the enclosing delim join's duplicate-eliminated chunk, which
      // Sirius materializes with a grouped aggregate over ALL delim columns
      // (plan_delim_join builds the DISTINCT; wrap_delim_distinct globalizes it via
      // PARTITION_DISTINCT → DISTINCT_MERGE), so the full row set is unique by
      // construction — under EQUAL null semantics too, since GROUP BY collapses NULL keys
      // into one group. Only the complete column set is a key; a subset can repeat.
      auto& delim_get = op.Cast<duckdb::LogicalDelimGet>();
      if (delim_get.chunk_types.empty()) { return {}; }
      std::unordered_set<duckdb::idx_t> unique_set;
      for (duckdb::idx_t i = 0; i < delim_get.chunk_types.size(); i++) {
        unique_set.insert(i);
      }
      return {std::move(unique_set)};
    }

    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N: {
      // These operators only truncate rows — no column remapping.
      if (op.children.empty()) { return {}; }
      return prove_unique_columns(sirius_context, *op.children[0]);
    }

    case duckdb::LogicalOperatorType::LOGICAL_FILTER: {
      if (op.children.empty()) { return {}; }
      auto child_unique = prove_unique_columns(sirius_context, *op.children[0]);
      if (child_unique.empty()) { return {}; }
      auto& filter = op.Cast<duckdb::LogicalFilter>();
      if (!filter.HasProjectionMap()) { return child_unique; }
      return remap_unique_keys(child_unique, [&](duckdb::idx_t col) {
        return through_projection_map(filter.projection_map, col);
      });
    }

    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: {
      if (op.children.empty()) { return {}; }
      auto child_unique = prove_unique_columns(sirius_context, *op.children[0]);
      if (child_unique.empty()) { return {}; }
      auto& order = op.Cast<duckdb::LogicalOrder>();
      if (!order.HasProjectionMap()) { return child_unique; }
      return remap_unique_keys(child_unique, [&](duckdb::idx_t col) {
        return through_projection_map(order.projection_map, col);
      });
    }

    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
      if (op.children.empty()) { return {}; }
      auto child_unique = prove_unique_columns(sirius_context, *op.children[0]);
      if (child_unique.empty()) { return {}; }

      auto& proj = op.Cast<duckdb::LogicalProjection>();
      // Only direct BoundReferenceExpression pass-throughs are safe.
      return remap_unique_keys(
        child_unique, [&](duckdb::idx_t col) -> std::optional<duckdb::idx_t> {
          for (duckdb::idx_t i = 0; i < proj.expressions.size(); i++) {
            auto& expr = proj.expressions[i];
            if (expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF &&
                expr->Cast<duckdb::BoundReferenceExpression>().Index() == col) {
              return i;
            }
          }
          return std::nullopt;
        });
    }

    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
      if (op.children.size() != 2) { return {}; }
      auto& join = op.Cast<duckdb::LogicalComparisonJoin>();

      auto left_unique  = prove_unique_columns(sirius_context, *op.children[0]);
      auto right_unique = prove_unique_columns(sirius_context, *op.children[1]);
      if (left_unique.empty() && right_unique.empty()) { return {}; }

      // Collect equality key columns on each side (only direct column refs).
      std::unordered_set<duckdb::idx_t> left_eq_keys, right_eq_keys;
      for (const auto& c : join.conditions) {
        if (!c.IsComparison() || c.GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
          continue;
        }
        if (c.LeftReference()->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          left_eq_keys.insert(c.LeftReference()->Cast<duckdb::BoundReferenceExpression>().Index());
        }
        if (c.RightReference()->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          right_eq_keys.insert(
            c.RightReference()->Cast<duckdb::BoundReferenceExpression>().Index());
        }
      }

      bool right_keys_unique = includes_unique_key(right_unique, right_eq_keys);
      bool left_keys_unique  = includes_unique_key(left_unique, left_eq_keys);

      // Which side's row-level uniqueness survives the join?
      bool left_preserved  = false;
      bool right_preserved = false;
      switch (join.join_type) {
        case duckdb::JoinType::INNER:
          // Each left row matches ≤1 right row iff right keys are unique (and vice versa).
          left_preserved  = right_keys_unique;
          right_preserved = left_keys_unique;
          break;
        case duckdb::JoinType::LEFT: left_preserved = right_keys_unique; break;
        case duckdb::JoinType::RIGHT: right_preserved = left_keys_unique; break;
        case duckdb::JoinType::SEMI:
        case duckdb::JoinType::ANTI:
          left_preserved = !left_unique.empty();  // output ⊆ left rows
          break;
        case duckdb::JoinType::RIGHT_SEMI:
        case duckdb::JoinType::RIGHT_ANTI: right_preserved = !right_unique.empty(); break;
        case duckdb::JoinType::MARK:
        case duckdb::JoinType::SINGLE: left_preserved = !left_unique.empty(); break;
        default: return {};  // FULL OUTER: NULL-padding can duplicate key values
      }
      if (!left_preserved && !right_preserved) { return {}; }

      // Remap child unique keys through a projection map to output positions.
      auto remap = [](unique_keys const& child_unique,
                      const duckdb::vector<duckdb::ProjectionIndex>& proj_map,
                      duckdb::idx_t offset) {
        return remap_unique_keys(child_unique,
                                 [&](duckdb::idx_t col) -> std::optional<duckdb::idx_t> {
                                   if (proj_map.empty()) { return offset + col; }
                                   auto const out = through_projection_map(proj_map, col);
                                   if (!out) { return std::nullopt; }
                                   return offset + *out;
                                 });
      };

      bool only_left =
        join.join_type == duckdb::JoinType::SEMI || join.join_type == duckdb::JoinType::ANTI;
      bool only_right = join.join_type == duckdb::JoinType::RIGHT_SEMI ||
                        join.join_type == duckdb::JoinType::RIGHT_ANTI;

      duckdb::idx_t left_output_count = join.left_projection_map.empty()
                                          ? op.children[0]->types.size()
                                          : join.left_projection_map.size();

      unique_keys keys;
      if (left_preserved && !only_right) { keys = remap(left_unique, join.left_projection_map, 0); }
      if (right_preserved && !only_left) {
        duckdb::idx_t right_offset = only_right ? 0 : left_output_count;
        auto mapped                = remap(right_unique, join.right_projection_map, right_offset);
        keys.insert(keys.end(),
                    std::make_move_iterator(mapped.begin()),
                    std::make_move_iterator(mapped.end()));
      }
      return keys;
    }

    default: return {};
  }
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_comparison_join(duckdb::LogicalComparisonJoin& op)
{
  // now visit the children
  D_ASSERT(op.children.size() == 2);

  // Inner-join residuals use the unprojected left/right schema in DuckDB v2.
  // Evaluate them before applying the join's output projection. Outer/semi/anti joins
  // must evaluate residuals while matching rows. DELIM planning requires a bare join.
  if (std::any_of(op.conditions.begin(), op.conditions.end(), [](auto const& condition) {
        return !condition.IsComparison();
      })) {
    if (op.type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
        op.join_type != duckdb::JoinType::INNER) {
      throw duckdb::NotImplementedException(
        "Sirius does not support arbitrary join conditions for this join type");
    }
    auto conjunction = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
      duckdb::ExpressionType::CONJUNCTION_AND);
    duckdb::vector<duckdb::JoinCondition> comparisons;
    for (auto& condition : op.conditions) {
      if (condition.IsComparison()) {
        comparisons.push_back(std::move(condition));
      } else {
        reject_nested_column_operation(condition.GetJoinExpression(), "a join predicate");
        conjunction->GetChildrenMutable().push_back(std::move(condition.JoinExpressionReference()));
      }
    }
    auto predicate = sirius::ast::from_duckdb(*conjunction);
    if (!predicate) {
      throw duckdb::NotImplementedException("Unsupported inner-join residual predicate");
    }
    auto output_types = sirius::from_duckdb_vec(op.types);
    std::vector<cudf::size_type> output_indices;
    auto const left_width        = op.children[0]->types.size();
    auto const append_projection = [&](auto const& map, std::size_t width, std::size_t offset) {
      if (map.empty()) {
        for (std::size_t i = 0; i < width; ++i) {
          output_indices.push_back(static_cast<cudf::size_type>(offset + i));
        }
      } else {
        for (auto index : map) {
          output_indices.push_back(static_cast<cudf::size_type>(offset + index));
        }
      }
    };
    append_projection(op.left_projection_map, left_width, 0);
    append_projection(op.right_projection_map, op.children[1]->types.size(), left_width);
    op.conditions = std::move(comparisons);
    op.left_projection_map.clear();
    op.right_projection_map.clear();
    op.types = op.children[0]->types;
    op.types.insert(op.types.end(), op.children[1]->types.begin(), op.children[1]->types.end());
    auto plan   = plan_comparison_join(op);
    auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(std::move(output_types),
                                                                        std::move(predicate),
                                                                        op.estimated_cardinality,
                                                                        std::move(output_indices));
    filter->children.push_back(std::move(plan));
    return filter;
  }

  // Reject nested join keys before planning either child.
  for (auto const& condition : op.conditions) {
    D_ASSERT(condition.IsComparison());
    reject_nested_column_operation(*condition.LeftReference(), "a join condition");
    reject_nested_column_operation(*condition.RightReference(), "a join condition");
  }

  // A MARK join mixing null-safe (IS NOT DISTINCT FROM) and plain keys has no correct GPU
  // encoding (see mark_join_mixes_null_safe_keys in sirius_physical_hash_join.cpp), and the NLJ
  // is no escape hatch: its MARK arm models the mixture but evaluates each (left batch x right
  // batch) pair independently, so a left side spread over several right batches emits one partial
  // mark per pair. Reject here so both routes are closed and DuckDB's CPU mark join answers.
  if (op.join_type == duckdb::JoinType::MARK) {
    bool has_null_safe = false;
    bool all_null_safe = !op.conditions.empty();
    for (auto const& condition : op.conditions) {
      if (condition.GetComparisonType() == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        has_null_safe = true;
      } else {
        all_null_safe = false;
      }
    }
    if (has_null_safe && !all_null_safe) {
      throw duckdb::NotImplementedException(
        "MARK join mixing a null-safe (IS NOT DISTINCT FROM) key with a plain key not supported "
        "in GPU");
    }
  }

  normalize_non_null_join_keys(op);
  planner::drop_join_implied_filters(op);

  std::size_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
  std::size_t rhs_cardinality = op.children[1]->EstimateCardinality(context);

  auto sirius_context = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  bool const dynamic_filter_enabled =
    sirius_context && duckdb::query_operator_options(context)->enable_dynamic_filter;

  bool const build_filtered = dynamic_filter_enabled && build_subtree_is_filtering(*op.children[1]);
  bool const build_opaque   = dynamic_filter_enabled && build_relation_is_opaque(*op.children[1]);
  bool const build_evidence = build_filtered || build_opaque;

  // Capture domain and uniqueness evidence before create_plan() moves the logical children.
  auto condition_domains =
    build_evidence ? build_key_domain_cardinalities(op, duckdb_base_table_cardinality{context})
                   : std::vector<std::size_t>{};

  // Pin-time uniqueness facts count as proofs only with pin_table_prove_unique_keys; without a
  // context, prove_unique_columns considers declared PRIMARY KEYs alone.
  auto* const pin_facts_context =
    sirius_context && duckdb::query_operator_options(context)->prove_unique_keys
      ? sirius_context.get()
      : nullptr;
  auto build_side_unique_keys  = prove_unique_columns(pin_facts_context, *op.children[1]);
  auto left                    = create_plan(*op.children[0]);
  auto right                   = create_plan(*op.children[1]);
  left->estimated_cardinality  = lhs_cardinality;
  right->estimated_cardinality = rhs_cardinality;

  if (op.conditions.empty()) {
    // No conditions: a cross product, as LogicalCrossProduct lowers to. DuckDB emits this for a
    // correlated scalar count(*): a LEFT join against a one-row default. On two non-empty inputs
    // INNER/LEFT/RIGHT/OUTER all equal the cross product and the nested loop join pads a 0-row
    // side per join type; SEMI/ANTI/MARK/SINGLE would need rows filtered or marked, which the
    // condition-less path does not do.
    switch (op.join_type) {
      case duckdb::JoinType::INNER:
      case duckdb::JoinType::LEFT:
      case duckdb::JoinType::RIGHT:
      case duckdb::JoinType::OUTER:
        op.estimated_cardinality = memory::saturating_mul(lhs_cardinality, rhs_cardinality);
        return make_cross_product_join(
          op,
          std::move(left),
          std::move(right),
          op.join_type,
          op.estimated_cardinality,
          duckdb::vector<std::size_t>(op.left_projection_map.begin(), op.left_projection_map.end()),
          duckdb::vector<std::size_t>(op.right_projection_map.begin(),
                                      op.right_projection_map.end()));
      default:
        throw duckdb::NotImplementedException("Cross product with join type " +
                                              duckdb::JoinTypeToString(op.join_type) +
                                              " not supported in GPU");
    }
  }

  // Preserve key shape before materialization makes computed keys look like direct references.
  auto condition_key_shapes = classify_join_key_shapes(op.conditions);

  std::size_t has_range              = 0;
  [[maybe_unused]] bool has_equality = op.HasEquality(has_range);
  bool can_merge                     = has_range > 0;
  bool can_iejoin                    = has_range >= 2 && recursive_cte_tables.empty();
  switch (op.join_type) {
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::RIGHT_ANTI:
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::MARK:
      can_merge  = can_merge && op.conditions.size() == 1;
      can_iejoin = false;
      break;
    default: break;
  }
  //	TODO: Extend PWMJ to handle all comparisons and projection maps
  bool prefer_range_joins = duckdb::Settings::Get<duckdb::PreferRangeJoinsSetting>(context);
  prefer_range_joins      = prefer_range_joins && can_iejoin;

  // Check DuckDB's NLJ IsSupported here because it needs the raw `op.conditions`; wrapping the
  // conditions below drains them.
  const bool nlj_conditions_supported =
    duckdb::PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type);
  // DuckDB's check accepts join types the Sirius NLJ execute switch has no arm for
  // (RIGHT_SEMI / RIGHT_ANTI / SINGLE), which would throw mid-query instead of falling back.
  // The hash join screens the same way via are_conditions_supported. SINGLE is implemented by
  // the hash join only, so a SINGLE join whose conditions need the NLJ (no hashable equality)
  // still falls back.
  const bool nlj_join_type_supported =
    sirius::op::sirius_physical_nested_loop_join::is_join_type_supported(op.join_type);
  const bool nlj_is_supported = nlj_conditions_supported && nlj_join_type_supported;

  // Wrap once — subsequent checks and ctors consume from the wrapped vector.
  duckdb::vector<sirius::join_condition> conditions =
    sirius::wrap_join_conditions(std::move(op.conditions));
  duckdb::vector<std::size_t> left_projection_map(op.left_projection_map.begin(),
                                                  op.left_projection_map.end());
  duckdb::vector<std::size_t> right_projection_map(op.right_projection_map.begin(),
                                                   op.right_projection_map.end());
  sirius::op::sirius_physical_hash_join::materialize_expression_join_keys(
    conditions, left, right, left_projection_map, right_projection_map);

  bool is_supported_by_hash_join =
    sirius::op::sirius_physical_hash_join::are_conditions_supported(conditions, op.join_type);
  if (is_supported_by_hash_join && !prefer_range_joins) {
    auto options          = duckdb::query_operator_options(context);
    const auto& op_params = *options;
    auto* const stats     = &sirius_context->get_dynamic_filter_stats();

    // Resolve placements before registering any producer channel.
    auto& memory_manager  = sirius_context->get_memory_manager();
    auto const gpu_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    auto const host_spaces =
      memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);

    // Only a single-column proof can establish per-key uniqueness; take one that is a build key.
    std::optional<std::size_t> build_side_unique_column;
    for (auto const& condition : conditions) {
      if (!condition.right->is_reference()) { continue; }
      auto const col = static_cast<duckdb::idx_t>(condition.right->as_reference().column_index);
      if (includes_unique_key(build_side_unique_keys, {col})) {
        build_side_unique_column = static_cast<std::size_t>(col);
        break;
      }
    }
    auto admitted_keys = admit_dynamic_filter_keys(
      conditions, condition_key_shapes, condition_domains, build_side_unique_column);

    // Prefer scan binding; each key uses one route.
    std::vector<sirius::op::dynamic_filter_publish_plan::probe_target> targets;
    std::size_t scan_target_count = 0;
    bool const needs_early_publication =
      op.join_type == duckdb::JoinType::RIGHT_SEMI || op.join_type == duckdb::JoinType::RIGHT_ANTI;
    bool const discovery_runs =
      build_evidence && op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
      !gpu_spaces.empty() && !host_spaces.empty() &&
      (!needs_early_publication || (op_params.enable_dynamic_filter_multi_partition &&
                                    op_params.max_dynamic_filter_bloom_bytes_per_gpu > 0));
    if (discovery_runs) {
      bool const scan_bind_armed = probe_filter_join_type_admissible(op.join_type);
      std::unordered_map<sirius::op::sirius_physical_operator const*, std::size_t> target_by_scan;
      for (std::size_t key_index = 0; key_index < admitted_keys.size(); ++key_index) {
        auto const& key       = admitted_keys[key_index];
        auto const& condition = conditions[key.planner_condition_index];
        // Right-family partitioning waits for the probe. Only register keys that the
        // pre-partition accumulator can publish before that scan drains.
        if (needs_early_publication &&
            !sirius::op::detail::accumulated_bloom_builder::supports(key.storage_type)) {
          continue;
        }
        descent_policy const policy{.descend_build_blocks = true,
                                    .preserve_nulls       = key.preserve_nulls};
        // Terminal ordinals are local to each terminal, not the probe entry schema.
        auto const terminals =
          trace_probe_key(*left, static_cast<std::size_t>(key.probe_key_ordinal), policy);
        bool scan_bound = false;
        for (auto const& terminal : terminals) {
          if (!scan_bind_armed ||
              terminal.node->type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
            continue;
          }
          auto& scan = terminal.node->Cast<sirius::op::sirius_physical_table_scan>();
          if (terminal.ordinal >= scan.types.size()) {
            SIRIUS_LOG_WARN(
              "[sirius_plan_comparison_join] dynamic filter key {}: scan-route terminal at scan "
              "'{}' carries exit ordinal {} outside the scan's {} output columns; skipping this "
              "binding.",
              key_index,
              scan.function.GetName().GetIdentifierName(),
              terminal.ordinal,
              scan.types.size());
            continue;
          }
          auto const [entry, inserted] = target_by_scan.try_emplace(terminal.node, targets.size());
          if (inserted) {
            // Reuse the scan channel so multiple producers share the consumer.
            if (!scan.sirius_dynamic_filters) {
              scan.sirius_dynamic_filters =
                std::make_shared<sirius::op::sirius_dynamic_filter_set>();
            }
            targets.push_back({.filter_set  = scan.sirius_dynamic_filters,
                               .route_class = sirius::op::dynamic_filter_route_class::scan,
                               .accepts_zone_map_filters = true,
                               .key_bindings             = {}});
            ++scan_target_count;
          }
          targets[entry->second].key_bindings.push_back(
            {.admitted_key_index   = key_index,
             .channel_push_ordinal = terminal.ordinal,
             .probe_storage_type   = sirius::get_cudf_type(scan.types[terminal.ordinal])
                                     .value_or(cudf::data_type{cudf::type_id::EMPTY})});
          scan_bound = true;
        }
        if (scan_bound) { continue; }
        // Admission proves build-block safety; placement preserves the reported site ordinal.
        if (!direct_route_admissible(op.join_type,
                                     condition.comparison,
                                     key.key_shape,
                                     key.probe_storage_type,
                                     key.storage_type)) {
          continue;
        }
        std::vector<std::shared_ptr<sirius::op::sirius_dynamic_filter_set>> site_channels;
        auto placed = place_endpoint(
          std::move(left),
          static_cast<std::size_t>(key.probe_key_ordinal),
          policy,
          [&site_channels, &op_params, stats](sirius::op::sirius_physical_operator const& site)
            -> duckdb::unique_ptr<sirius::op::sirius_physical_operator> {
            auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
            auto consumer = sirius::op::scan::make_dynamic_filter_consumer(
              channel,
              sirius::op::scan::dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
              op_params.dynamic_filter_keep_threshold,
              /*decode_bindings=*/{},
              stats);
            auto endpoint = duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
              site.types, site.estimated_cardinality, std::move(consumer));
            site_channels.push_back(std::move(channel));
            return endpoint;
          });
        left = std::move(placed.subtree);
        if (site_channels.size() != placed.site_ordinals.size()) {
          SIRIUS_LOG_WARN(
            "[sirius_plan_comparison_join] dynamic filter key {}: direct-route walk placed {} "
            "endpoints but recorded {} site ordinals; skipping this key's bindings.",
            key_index,
            site_channels.size(),
            placed.site_ordinals.size());
          continue;
        }
        for (std::size_t site = 0; site < site_channels.size(); ++site) {
          targets.push_back({.filter_set  = std::move(site_channels[site]),
                             .route_class = sirius::op::dynamic_filter_route_class::direct,
                             .accepts_zone_map_filters = false,
                             .key_bindings             = {{.admitted_key_index   = key_index,
                                                           .channel_push_ordinal = placed.site_ordinals[site],
                                                           .probe_storage_type   = key.probe_storage_type}}});
        }
      }
    }
    if (!targets.empty()) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_comparison_join] Wired hash join with {} dynamic-filter probe "
        "target(s) ({} scan-bound, {} join-edge; evidence={}; build est {} rows).",
        targets.size(),
        scan_target_count,
        targets.size() - scan_target_count,
        build_filtered ? (build_opaque ? "filtered+opaque" : "filtered") : "opaque",
        rhs_cardinality);
    } else if (build_evidence && op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
               !admitted_keys.empty() && (gpu_spaces.empty() || host_spaces.empty())) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_comparison_join] Not wiring dynamic filter(s): a GPU and HOST "
        "memory space are required for device-local replicas.");
    } else if (dynamic_filter_enabled && !build_evidence &&
               op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
               !admitted_keys.empty()) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_comparison_join] Not wiring dynamic filter(s): build side carries "
        "neither filter nor opaque-build evidence (build est {} rows).",
        rhs_cardinality);
    }

    // Prefer a NUMA-local HOST staging space for each GPU and fall back to the first HOST space.
    std::vector<sirius::op::dynamic_filter_replica_space> filter_replica_spaces;
    if (!targets.empty()) {
      auto const& topology = sirius_context->get_config().get_hw_topology();
      for (auto const* gpu_space : gpu_spaces) {
        auto const gpu_id = gpu_space->get_device_id();
        auto const gpu_info =
          std::find_if(topology.gpus.begin(), topology.gpus.end(), [gpu_id](auto const& gpu) {
            return static_cast<int>(gpu.id) == gpu_id;
          });
        auto const numa_node = gpu_info == topology.gpus.end() ? -1 : gpu_info->numa_node;
        auto const local_host =
          std::find_if(host_spaces.begin(), host_spaces.end(), [numa_node](auto const* host) {
            return numa_node >= 0 && host->get_device_id() == numa_node;
          });
        auto const* host_space =
          local_host == host_spaces.end() ? host_spaces.front() : *local_host;
        auto* mutable_gpu_space =
          memory_manager.get_memory_space(cucascade::memory::Tier::GPU, gpu_id);
        if (mutable_gpu_space == nullptr) {
          throw std::logic_error(
            "[sirius_plan_comparison_join] Dynamic-filter GPU space disappeared during "
            "plan construction");
        }
        filter_replica_spaces.emplace_back(*mutable_gpu_space, *host_space);
      }
    }

    sirius::op::dynamic_filter_publish_plan filter_plan{
      std::move(admitted_keys),
      std::move(targets),
      std::move(filter_replica_spaces),
      {.emit_zone_map_filters     = op_params.enable_dynamic_zone_map_filter,
       .domain_coverage_threshold = op_params.dynamic_filter_domain_coverage_threshold,
       .inlist_max_l2_fraction    = op_params.dynamic_filter_inlist_max_l2_fraction,
       .enable_multi_partition    = op_params.enable_dynamic_filter_multi_partition,
       .max_bloom_bytes_per_gpu   = op_params.max_dynamic_filter_bloom_bytes_per_gpu}};

    auto join = make_hash_join(op,
                               std::move(left),
                               std::move(right),
                               std::move(conditions),
                               op.join_type,
                               op_params,
                               {.left_projection_map       = std::move(left_projection_map),
                                .right_projection_map      = std::move(right_projection_map),
                                .delim_types               = sirius::from_duckdb_vec(op.mark_types),
                                .dynamic_filter_plan       = std::move(filter_plan),
                                .dynamic_filter_stats_sink = stats});
    auto& hj  = *join;
    hj.set_single_join_error_on_multiple_rows(
      duckdb::Settings::Get<duckdb::ScalarSubqueryErrorOnMultipleRowsSetting>(context));

    // --- Detect build-side key uniqueness ---
    // Gate: only for pure equal conditions (not_distinct_from needs null_equality::EQUAL).
    bool all_compare_equal = true;
    for (const auto& c : hj.conditions) {
      if (c.comparison == sirius::comparison_type::equal) { continue; }
      if (c.comparison == sirius::comparison_type::not_distinct_from) {
        all_compare_equal = false;
        break;
      }
      // Inequality conditions are fine — only equality conditions drive the hash table.
    }
    if (all_compare_equal) {
      // Extract build-side (right) column indices from equality conditions.
      // Only accept direct BoundReferenceExpression (skip if any has a cast).
      std::unordered_set<duckdb::idx_t> build_key_cols;
      bool keys_extractable = true;
      for (const auto& c : hj.conditions) {
        if (c.comparison != sirius::comparison_type::equal) { continue; }
        if (!c.right->holds<sirius::ast::reference>()) {
          keys_extractable = false;
          break;
        }
        build_key_cols.insert(c.right->get<sirius::ast::reference>().column_index);
      }
      if (keys_extractable && !build_key_cols.empty()) {
        // build_side_unique_keys was computed before create_plan (which moves logical node data).
        // Some proven key ⊆ build_key_cols  →  build keys are unique.
        if (includes_unique_key(build_side_unique_keys, build_key_cols)) {
          hj.unique_build_keys = true;
        }
      }
    }

    return join;
  }

  // D_ASSERT(op.left_projection_map.empty());
  // std::size_t nested_loop_join_threshold =
  //   duckdb::DBConfig::GetSetting<duckdb::NestedLoopJoinThresholdSetting>(context);
  // if (left->estimated_cardinality < nested_loop_join_threshold ||
  //     right->estimated_cardinality < nested_loop_join_threshold) {
  //   can_iejoin = false;
  //   can_merge  = false;
  // }

  // if (can_merge && can_iejoin) {
  //   std::size_t merge_join_threshold =
  //     duckdb::DBConfig::GetSetting<duckdb::MergeJoinThresholdSetting>(context);
  //   if (left->estimated_cardinality < merge_join_threshold ||
  //       right->estimated_cardinality < merge_join_threshold) {
  //     can_iejoin = false;
  //   }
  // }

  // if (can_iejoin) {
  //   throw duckdb::NotImplementedException("InequalityJoin not supported in GPU");
  //   // return Make<PhysicalIEJoin>(op, left, right, std::move(op.conditions), op.join_type,
  //   // op.estimated_cardinality,
  //   //                             std::move(op.filter_pushdown));
  // }
  // if (can_merge) {
  //   throw duckdb::NotImplementedException("Piecewise merge join not supported in GPU");
  //   // range join: use piecewise merge join
  //   // return Make<PhysicalPiecewiseMergeJoin>(op, left, right, std::move(op.conditions),
  //   // op.join_type,
  //   //                                         op.estimated_cardinality,
  //   //                                         std::move(op.filter_pushdown));
  // }
  if (nlj_is_supported) {
    // inequality join: use nested loop; pass projection maps so output column order matches plan
    auto join = duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(
      op,
      std::move(left),
      std::move(right),
      std::move(conditions),
      op.join_type,
      op.estimated_cardinality,
      std::move(left_projection_map),
      std::move(right_projection_map));
    return join;
  }

  // Neither GPU join implements this join type, so say so plainly: the generic "blockwise nested
  // loop" message below would misattribute the rejection to the conditions.
  if (!nlj_join_type_supported) {
    throw duckdb::NotImplementedException("Join type " + duckdb::JoinTypeToString(op.join_type) +
                                          " not supported in GPU");
  }

  throw duckdb::NotImplementedException("Blockwise nested loop join not supported in GPU");
  // for (auto &cond : op.conditions) {
  // 	RewriteJoinCondition(cond.RightReference(), left.types.size());
  // }
  // auto condition = JoinCondition::CreateExpression(std::move(op.conditions));
  // return Make<PhysicalBlockwiseNLJoin>(op, left, right, std::move(condition), op.join_type,
  // op.estimated_cardinality);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalComparisonJoin& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_ASOF_JOIN:
      // return plan_asof_join(op);
      throw duckdb::NotImplementedException("Asof join not supported in GPU");
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: return plan_comparison_join(op);
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN: return plan_delim_join(op);
    default:
      throw duckdb::InternalException("Unrecognized operator type for LogicalComparisonJoin");
  }
}

}  // namespace sirius::planner
