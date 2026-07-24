/*
 * Copyright 2026, Sirius Contributors.
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

#include "planner/sirius_plan_compressed_schema.hpp"

#include "duckdb/common/assert.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/utils.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"

#include <cudf/cudf_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

std::optional<std::vector<cudf::data_type>> try_native_physical_schema(
  sirius::op::sirius_physical_operator const& op)
{
  std::vector<cudf::data_type> schema;
  schema.reserve(op.types.size());
  for (auto const& type : op.types) {
    auto const native = sirius::try_get_cudf_type(type);
    if (!native) { return std::nullopt; }
    schema.push_back(*native);
  }
  return schema;
}

std::vector<cudf::data_type> native_physical_schema(sirius::op::sirius_physical_operator const& op)
{
  auto schema = try_native_physical_schema(op);
  return schema ? std::move(*schema) : std::vector<cudf::data_type>{};
}

bool compressed_schema_tree_is_mappable(sirius::op::sirius_physical_operator const& op)
{
  if (!try_native_physical_schema(op)) { return false; }
  for (auto const& child : op.children) {
    if (child && !compressed_schema_tree_is_mappable(*child)) { return false; }
  }

  if (op.type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      op.type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto const& delim = op.Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join && !compressed_schema_tree_is_mappable(*delim.join)) { return false; }
    if (delim.distinct_root && !compressed_schema_tree_is_mappable(*delim.distinct_root)) {
      return false;
    }
  }
  return true;
}

void clear_compressed_schema_tree(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  if (!slot) { return; }
  slot->set_physical_types({});
  for (auto& child : slot->children) {
    clear_compressed_schema_tree(child);
  }

  if (slot->type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      slot->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = slot->Cast<sirius::op::sirius_physical_delim_join>();
    clear_compressed_schema_tree(delim.join);
    clear_compressed_schema_tree(delim.distinct_root);
  }
}

std::vector<cudf::data_type> output_physical_schema(sirius::op::sirius_physical_operator const& op)
{
  return op.has_physical_overrides() ? op.get_physical_types() : native_physical_schema(op);
}

void install_physical_schema(sirius::op::sirius_physical_operator& op,
                             std::vector<cudf::data_type> schema)
{
  auto const native = native_physical_schema(op);
  op.set_physical_types(schema == native ? std::vector<cudf::data_type>{} : std::move(schema));
}

//! Wrap @p slot in a projection that casts every column selected by @p should_restore whose
//! carrier differs from native back to its logical type, forwarding all other columns as bare
//! references. The projection's output sidecar keeps the unselected columns' carriers (an
//! all-native result normalizes to the empty sidecar); a no-op when no selected column differs.
template <typename ShouldRestore>
void restore_columns_matching(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                              ShouldRestore const& should_restore)
{
  if (!slot || !slot->has_physical_overrides()) { return; }

  auto const physical = output_physical_schema(*slot);
  auto const native   = native_physical_schema(*slot);
  bool needs_restore  = false;
  for (std::size_t column_idx = 0; column_idx < physical.size(); ++column_idx) {
    if (should_restore(column_idx) && physical[column_idx] != native[column_idx]) {
      needs_restore = true;
      break;
    }
  }
  if (!needs_restore) { return; }

  auto input         = std::move(slot);
  auto output_schema = physical;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions;
  expressions.reserve(input->types.size());
  for (std::size_t column_idx = 0; column_idx < input->types.size(); ++column_idx) {
    if (should_restore(column_idx) && physical[column_idx] != native[column_idx]) {
      auto reference = std::make_unique<sirius::ast::node>(
        sirius::ast::reference{static_cast<std::uint32_t>(column_idx)});
      expressions.push_back(std::make_unique<sirius::ast::node>(
        sirius::ast::cast{std::move(reference), input->types[column_idx], /*try_cast=*/false}));
      output_schema[column_idx] = native[column_idx];
    } else {
      expressions.push_back(std::make_unique<sirius::ast::node>(
        sirius::ast::reference{static_cast<std::uint32_t>(column_idx), input->types[column_idx]}));
    }
  }

  auto projection = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
    input->types, std::move(expressions), input->estimated_cardinality);
  projection->children.push_back(std::move(input));
  install_physical_schema(*projection, std::move(output_schema));
  slot = std::move(projection);
}

void restore_native_columns(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                            std::unordered_set<std::size_t> const& columns)
{
  restore_columns_matching(
    slot, [&columns](std::size_t column_idx) { return columns.contains(column_idx); });
}

void restore_native_output_in_place(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  if (!slot) { return; }
  for (auto& child : slot->children) {
    restore_native_schema(child);
  }
  slot->set_physical_types({});
}

//! Derive @p op 's output sidecar from @p child_schema: a bare-reference output forwards the
//! referenced child carrier, every other output keeps its native carrier. Callers guarantee the
//! projection's select_list arity matches its native output schema.
void derive_projection_sidecar(sirius::op::sirius_physical_operator& op,
                               std::vector<cudf::data_type> const& child_schema)
{
  auto const& projection = op.Cast<sirius::op::sirius_physical_projection>();
  auto schema            = native_physical_schema(op);
  for (std::size_t output_idx = 0; output_idx < projection.select_list.size(); ++output_idx) {
    auto const& expression = projection.select_list[output_idx];
    if (!expression || !expression->holds<sirius::ast::reference>()) { continue; }
    auto const input_idx = expression->get<sirius::ast::reference>().column_index;
    if (input_idx < child_schema.size()) { schema[output_idx] = child_schema[input_idx]; }
  }
  install_physical_schema(op, std::move(schema));
}

//! Return whether @p op is a projection consisting solely of bare column references. Such a
//! projection forwards column views without materializing a batch, so a narrow carrier crossing
//! it saves no bandwidth.
bool is_pure_reference_projection(sirius::op::sirius_physical_operator const& op)
{
  if (op.type != sirius::op::SiriusPhysicalOperatorType::PROJECTION) { return false; }
  auto const& projection = op.Cast<sirius::op::sirius_physical_projection>();
  if (projection.select_list.size() != op.types.size()) { return false; }
  for (auto const& expression : projection.select_list) {
    if (!expression || !expression->holds<sirius::ast::reference>()) { return false; }
  }
  return true;
}

}  // namespace

void restore_native_schema(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  restore_columns_matching(slot, [](std::size_t) { return true; });
}

void propagate_compressed_schema(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  if (!slot) { return; }
  for (auto& child : slot->children) {
    propagate_compressed_schema(child);
  }

  if (slot->type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      slot->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = slot->Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join) {
      propagate_compressed_schema(delim.join);
      restore_native_output_in_place(delim.join);
    }
    if (delim.distinct_root) {
      propagate_compressed_schema(delim.distinct_root);
      restore_native_output_in_place(delim.distinct_root);
    }
  }

  switch (slot->type) {
    case sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN: {
      // The scan planner installed the residency-gated physical schema. A dynamic-filter producer
      // publishes literals in the key's native carrier and the filter runs before any join-input
      // restore projection, so every planned target column advertises native output here; unrelated
      // payload columns keep their carriers. A producer that declared no targets is conservative by
      // construction: every column is a potential target, so the whole sidecar goes native.
      auto& scan = slot->Cast<sirius::op::sirius_physical_table_scan>();
      if (!scan.sirius_dynamic_filters || !scan.sirius_dynamic_filters->has_producers()) { return; }
      if (!slot->has_physical_overrides()) { return; }
      if (scan.sirius_dynamic_filters->has_unscoped_producer()) {
        slot->set_physical_types({});
        return;
      }
      // Translate each planned target column (column_ids space) to its scan output position:
      // output i reads column_ids position `projection_ids.empty() ? i : projection_ids[i]`, the
      // same convention scan_physical_schema uses. A planned target with no output position marks
      // nothing (such a filter is rejected by the consumer remap at publish time).
      auto const targets = scan.sirius_dynamic_filters->planned_target_columns();
      auto const native  = native_physical_schema(*slot);
      auto physical      = slot->get_physical_types();
      for (std::size_t output_idx = 0; output_idx < physical.size(); ++output_idx) {
        if (!scan.projection_ids.empty() && output_idx >= scan.projection_ids.size()) { continue; }
        auto const ids_position =
          scan.projection_ids.empty() ? output_idx : scan.projection_ids[output_idx];
        if (std::ranges::find(targets, ids_position) != targets.end()) {
          physical[output_idx] = native[output_idx];
        }
      }
      install_physical_schema(*slot, std::move(physical));
      return;
    }

    case sirius::op::SiriusPhysicalOperatorType::FILTER: {
      if (slot->children.size() != 1) { break; }
      auto const child_schema = output_physical_schema(*slot->children[0]);
      auto const& filter      = slot->Cast<sirius::op::sirius_physical_filter>();
      std::vector<cudf::data_type> schema;
      if (std::holds_alternative<sirius::op::passthrough>(filter.output_columns)) {
        if (child_schema.size() != slot->types.size()) { break; }
        schema = child_schema;
      } else {
        auto const& output_indices = std::get<std::vector<cudf::size_type>>(filter.output_columns);
        if (output_indices.size() != slot->types.size()) { break; }
        schema.reserve(output_indices.size());
        for (auto const input_idx : output_indices) {
          if (input_idx < 0 || static_cast<std::size_t>(input_idx) >= child_schema.size()) {
            schema.clear();
            break;
          }
          schema.push_back(child_schema[static_cast<std::size_t>(input_idx)]);
        }
        if (schema.size() != slot->types.size()) { break; }
      }
      install_physical_schema(*slot, std::move(schema));
      return;
    }

    case sirius::op::SiriusPhysicalOperatorType::PROJECTION: {
      if (slot->children.size() != 1) { break; }
      auto const& projection = slot->Cast<sirius::op::sirius_physical_projection>();
      if (projection.select_list.size() != native_physical_schema(*slot).size()) { break; }
      derive_projection_sidecar(*slot, output_physical_schema(*slot->children[0]));
      return;
    }

    case sirius::op::SiriusPhysicalOperatorType::LIMIT:
    case sirius::op::SiriusPhysicalOperatorType::STREAMING_LIMIT: {
      if (slot->children.size() != 1 || slot->children[0]->types != slot->types) { break; }
      install_physical_schema(*slot, output_physical_schema(*slot->children[0]));
      return;
    }

    case sirius::op::SiriusPhysicalOperatorType::HASH_JOIN: {
      if (slot->children.size() != 2) { break; }
      auto& join = slot->Cast<sirius::op::sirius_physical_hash_join>();
      std::unordered_set<std::size_t> left_keys;
      std::unordered_set<std::size_t> right_keys;
      for (auto const& condition : join.conditions) {
        if (condition.left) {
          sirius::ast::visit_references(*condition.left, [&](sirius::ast::reference const& ref) {
            left_keys.insert(ref.column_index);
          });
        }
        if (condition.right) {
          sirius::ast::visit_references(*condition.right, [&](sirius::ast::reference const& ref) {
            right_keys.insert(ref.column_index);
          });
        }
      }
      restore_native_columns(slot->children[0], left_keys);
      restore_native_columns(slot->children[1], right_keys);

      auto const left_schema  = output_physical_schema(*slot->children[0]);
      auto const right_schema = output_physical_schema(*slot->children[1]);
      auto schema             = native_physical_schema(*slot);
      std::size_t output_idx  = 0;
      bool const collect_left = join.join_type != duckdb::JoinType::RIGHT_SEMI &&
                                join.join_type != duckdb::JoinType::RIGHT_ANTI;
      bool const collect_right = join.join_type != duckdb::JoinType::SEMI &&
                                 join.join_type != duckdb::JoinType::ANTI &&
                                 join.join_type != duckdb::JoinType::MARK;
      if (collect_left) {
        for (auto const input_idx : join.lhs_output_columns.col_idxs) {
          if (output_idx >= schema.size()) { break; }
          if (input_idx >= 0 && static_cast<std::size_t>(input_idx) < left_schema.size()) {
            schema[output_idx] = left_schema[static_cast<std::size_t>(input_idx)];
          }
          ++output_idx;
        }
      }
      if (collect_right) {
        for (auto const input_idx : join.rhs_output_columns.col_idxs) {
          if (output_idx >= schema.size()) { break; }
          if (input_idx >= 0 && static_cast<std::size_t>(input_idx) < right_schema.size()) {
            schema[output_idx] = right_schema[static_cast<std::size_t>(input_idx)];
          }
          ++output_idx;
        }
      }
      install_physical_schema(*slot, std::move(schema));
      return;
    }

    case sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY: {
      // Bare-reference group keys may stay narrow: cudf::groupby receives them as raw views and
      // grouping is pure equality, which narrowing preserves (same values, family, and decimal
      // scale). Aggregate inputs must be native — the aggregation kernels accumulate at the input
      // width, so a narrow carrier could overflow where the native type would not. Any shape the
      // simplified key extraction does not cover (multiple grouping sets, grouping functions, an
      // AVG/count-distinct partial layout wider than the declared output) falls through to the
      // native boundary below.
      if (slot->children.size() != 1) { break; }
      auto& aggregate = slot->Cast<sirius::op::sirius_physical_grouped_aggregate>();
      if (aggregate.grouping_sets.size() > 1) { break; }
      // AVG decomposes into SUM + COUNT_VALID partial columns and COUNT(DISTINCT) keeps a LIST
      // partial column, so the partial batch layout deviates from the declared `types` shape a
      // sidecar describes; those shapes keep the native boundary.
      if (aggregate.has_avg || aggregate.has_count_distinct) { break; }
      // Grouping functions append output columns the operator does not compute through group_idx /
      // aggregate_slots; the arity check rejects that shape along with any other layout drift.
      if (slot->types.size() != aggregate.group_idx.size() + aggregate.aggregate_slots.size()) {
        break;
      }
      auto const child_width = slot->children[0]->types.size();
      bool consistent        = true;
      for (auto const key_idx : aggregate.group_idx) {
        if (key_idx < 0 || static_cast<std::size_t>(key_idx) >= child_width) {
          consistent = false;
          break;
        }
      }
      if (!consistent) { break; }

      // Every child column referenced as an aggregate input must be native. COUNT_ALL carries a
      // placeholder input index and reads no values, so it constrains nothing.
      std::unordered_set<std::size_t> aggregate_inputs;
      for (std::size_t i = 0; i < aggregate.cudf_aggregates.size(); ++i) {
        if (aggregate.cudf_aggregates[i] == cudf::aggregation::Kind::COUNT_ALL) { continue; }
        if (i < aggregate.cudf_aggregate_struct_col_indices.size() &&
            !aggregate.cudf_aggregate_struct_col_indices[i].empty()) {
          for (auto const struct_idx : aggregate.cudf_aggregate_struct_col_indices[i]) {
            if (struct_idx >= 0) { aggregate_inputs.insert(static_cast<std::size_t>(struct_idx)); }
          }
          continue;
        }
        if (i < aggregate.cudf_aggregate_idx.size() && aggregate.cudf_aggregate_idx[i] >= 0) {
          aggregate_inputs.insert(static_cast<std::size_t>(aggregate.cudf_aggregate_idx[i]));
        }
      }

      std::unordered_set<std::size_t> eligible_keys;
      for (auto const key_idx : aggregate.group_idx) {
        auto const key = static_cast<std::size_t>(key_idx);
        if (!aggregate_inputs.contains(key)) { eligible_keys.insert(key); }
      }

      std::unordered_set<std::size_t> columns_to_restore;
      for (std::size_t column_idx = 0; column_idx < child_width; ++column_idx) {
        if (!eligible_keys.contains(column_idx)) { columns_to_restore.insert(column_idx); }
      }
      restore_native_columns(slot->children[0], columns_to_restore);

      // Output prefix 0..group_idx.size()-1 holds the keys (get_output_grouping_indices is the
      // iota over it) and mirrors the child key carriers; aggregate outputs are cast to their
      // declared return types by the aggregation implementation and stay native. No restore is
      // inserted above: downstream boundaries widen the keys on the small grouped output.
      auto const child_schema = output_physical_schema(*slot->children[0]);
      auto schema             = native_physical_schema(*slot);
      for (std::size_t key_pos = 0; key_pos < aggregate.group_idx.size(); ++key_pos) {
        if (key_pos >= schema.size()) { break; }
        auto const child_idx = static_cast<std::size_t>(aggregate.group_idx[key_pos]);
        if (child_idx < child_schema.size()) { schema[key_pos] = child_schema[child_idx]; }
      }
      install_physical_schema(*slot, std::move(schema));
      return;
    }

    default: break;
  }

  // Pipeline wrapper operators (PARTITION, CONCAT, MERGE_*, SORT_PARTITION, SORT_SAMPLE,
  // GPU_SCAN, DYNAMIC_FILTER) are inserted by insert_gpu_pipeline_operators after these passes
  // run; their carrier contracts are established at wrap time from the finished sidecars.
  D_ASSERT(slot->type != sirius::op::SiriusPhysicalOperatorType::PARTITION &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::CONCAT &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::MERGE_SORT &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::MERGE_GROUP_BY &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::MERGE_TOP_N &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::MERGE_AGGREGATE &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::SORT_PARTITION &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::SORT_SAMPLE &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::GPU_SCAN &&
           slot->type != sirius::op::SiriusPhysicalOperatorType::DYNAMIC_FILTER);

  // Joins, aggregates, ordering, and all other operators retain their existing native-type
  // contracts. Restore any narrowed child immediately before crossing that boundary.
  restore_native_output_in_place(slot);
}

void prune_immediate_scan_restores(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  if (!slot) { return; }
  for (auto& child : slot->children) {
    prune_immediate_scan_restores(child);
  }
  if (slot->type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      slot->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = slot->Cast<sirius::op::sirius_physical_delim_join>();
    prune_immediate_scan_restores(delim.join);
    prune_immediate_scan_restores(delim.distinct_root);
  }

  if (slot->type != sirius::op::SiriusPhysicalOperatorType::PROJECTION ||
      slot->children.size() != 1) {
    return;
  }

  // Walk through zero-copy pure-reference projections to the scan this restore covers.
  std::vector<sirius::op::sirius_physical_operator*> chain;
  auto* node = slot->children[0].get();
  while (node != nullptr && node->children.size() == 1 && is_pure_reference_projection(*node)) {
    chain.push_back(node);
    node = node->children[0].get();
  }
  if (node == nullptr || node->type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN ||
      !node->has_physical_overrides()) {
    return;
  }
  auto* scan = node;

  // Map an input index of the restore projection through the reference chain to a scan column.
  auto map_to_scan_index = [&](std::uint32_t index) -> std::optional<std::uint32_t> {
    for (auto const* link : chain) {
      auto const& projection = link->Cast<sirius::op::sirius_physical_projection>();
      if (index >= projection.select_list.size()) { return std::nullopt; }
      index = projection.select_list[index]->get<sirius::ast::reference>().column_index;
    }
    if (index >= scan->types.size()) { return std::nullopt; }
    return index;
  };

  auto& projection = slot->Cast<sirius::op::sirius_physical_projection>();
  if (projection.select_list.size() != slot->types.size()) { return; }

  // Pruning must not change this projection's output physical schema — ancestors' sidecars were
  // derived from it. A bare-reference output forwards the scan column's carrier, so any scan
  // column it resolves to must keep its narrowing; only columns whose sole appearances here are
  // restore casts (already-native outputs) may flip native. Non-reference, non-cast expressions
  // need no guard: their outputs are native before and after, and their nested references adapt
  // to the actual carrier at runtime.
  std::unordered_set<std::uint32_t> carrier_forwarded_scan_columns;
  for (auto const& expression : projection.select_list) {
    if (!expression || !expression->holds<sirius::ast::reference>()) { continue; }
    auto const scan_idx = map_to_scan_index(expression->get<sirius::ast::reference>().column_index);
    if (scan_idx) { carrier_forwarded_scan_columns.insert(*scan_idx); }
  }

  auto const native = native_physical_schema(*scan);
  auto physical     = scan->get_physical_types();
  bool pruned_any   = false;
  for (auto& expression : projection.select_list) {
    if (!expression || !expression->holds<sirius::ast::cast>()) { continue; }
    auto const& cast_expr = expression->get<sirius::ast::cast>();
    if (cast_expr.try_cast || !cast_expr.child ||
        !cast_expr.child->holds<sirius::ast::reference>()) {
      continue;
    }
    auto const input_idx = cast_expr.child->get<sirius::ast::reference>().column_index;
    auto const scan_idx  = map_to_scan_index(input_idx);
    if (!scan_idx || cast_expr.target_type != scan->types[*scan_idx] ||
        physical[*scan_idx] == native[*scan_idx] ||
        carrier_forwarded_scan_columns.contains(*scan_idx)) {
      continue;
    }
    // A narrowing restored before any narrow batch write: emit natively at the scan instead.
    physical[*scan_idx] = native[*scan_idx];
    expression =
      std::make_unique<sirius::ast::node>(sirius::ast::reference{input_idx, cast_expr.target_type});
    pruned_any = true;
  }
  if (!pruned_any) { return; }

  install_physical_schema(*scan, std::move(physical));

  // Re-derive the sidecar of each chain projection bottom-up, then of the restore projection
  // itself, through the same derivation the PROJECTION propagation case uses.
  auto child_schema = output_physical_schema(*scan);
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    derive_projection_sidecar(**it, child_schema);
    child_schema = output_physical_schema(**it);
  }

  auto& restore_child = *slot->children[0];
  derive_projection_sidecar(*slot, output_physical_schema(restore_child));

  // A restore projection whose casts were all pruned is a positional identity over its child —
  // drop it to give the pipeline stage back.
  bool identity = slot->types == restore_child.types;
  for (std::size_t output_idx = 0; identity && output_idx < projection.select_list.size();
       ++output_idx) {
    auto const& expression = projection.select_list[output_idx];
    identity               = expression && expression->holds<sirius::ast::reference>() &&
               expression->get<sirius::ast::reference>().column_index == output_idx;
  }
  if (identity) {
    auto child = std::move(slot->children[0]);
    slot       = std::move(child);
  }
}

void apply_compressed_schema_passes(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan)
{
  if (compressed_schema_tree_is_mappable(*plan)) {
    propagate_compressed_schema(plan);
    restore_native_schema(plan);
    prune_immediate_scan_restores(plan);
  } else {
    clear_compressed_schema_tree(plan);
  }
}

}  // namespace sirius::planner
