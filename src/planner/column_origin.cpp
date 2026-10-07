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

#include "planner/column_origin.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include <filesystem>

namespace sirius::planner {

void column_origin_resolver::resolve(duckdb::LogicalOperator& op)
{
  // Children first: an operator's own bindings are expressed in terms of the
  // bindings its children produce.
  for (auto& child : op.children) {
    resolve(*child);
  }

  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET: record_get(op); break;
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: record_projection(op); break;
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: record_aggregate(op); break;
    default:
      // Everything else (filter, join, order, limit, ...) re-exposes its
      // children's bindings unchanged, so there is nothing new to record.
      break;
  }
}

namespace {

/// Table name for a file-backed scan, taken from the directory holding the files.
///
/// `read_parquet` is a table function, so there is no catalog entry to name — but
/// the datasets these plans are written for are laid out one directory per table
/// (`<root>/lineitem/part.0.parquet`), which is also the layout `pin_table(name=…)`
/// is given. The *file* stem is useless and actively dangerous here: every file is
/// called `part.N.parquet`, and `part` is itself a TPC-H table.
///
/// Returns empty when there is no file list to read, in which case the scan simply
/// contributes no lineage.
std::string table_name_from_files(duckdb::LogicalGet& get)
{
  if (!get.bind_data) { return {}; }
  auto const* multi = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get());
  if (multi == nullptr || !multi->file_list || multi->file_list->IsEmpty()) { return {}; }

  auto const files = multi->file_list->GetAllFiles();
  if (files.empty()) { return {}; }
  return std::filesystem::path(files.front().path).parent_path().filename().string();
}

}  // namespace

void column_origin_resolver::record_get(duckdb::LogicalOperator& op)
{
  auto& get  = op.Cast<duckdb::LogicalGet>();
  auto table = get.GetTable();

  // Base-table scans name themselves; file-backed scans (read_parquet) have no
  // catalog entry, so fall back to the directory holding their files.
  const std::string table_name = table ? table->name : table_name_from_files(get);
  if (table_name.empty()) { return; }

  // Only a real catalog table can bound the column index against a schema; for a
  // file scan the scan's own column list is the schema.
  const auto logical_column_count =
    table ? table->GetColumns().LogicalColumnCount() : get.names.size();

  // Output position i maps to a table column through column_ids, or through
  // projection_ids first when the scan projects a subset.
  const auto& column_ids = get.GetColumnIds();
  const auto& proj_ids   = get.projection_ids;
  const auto count       = proj_ids.empty() ? column_ids.size() : proj_ids.size();

  for (duckdb::idx_t i = 0; i < count; ++i) {
    const auto ids_index = proj_ids.empty() ? i : proj_ids[i];
    if (ids_index >= column_ids.size()) { continue; }
    const auto primary = column_ids[ids_index].GetPrimaryIndex();
    // Row-id and other virtual columns index past the real schema.
    if (primary >= logical_column_count) { continue; }

    _origins[duckdb::ColumnBinding(get.table_index, i)] =
      column_origin{table_name, static_cast<std::size_t>(primary)};
  }
}

void column_origin_resolver::record_projection(duckdb::LogicalOperator& op)
{
  auto& proj                = op.Cast<duckdb::LogicalProjection>();
  const auto child_bindings = child_output_bindings(op);
  for (duckdb::idx_t i = 0; i < proj.expressions.size(); ++i) {
    // A projection that merely forwards a column keeps its origin; anything
    // computed (arithmetic, a cast, a function) does not have one.
    if (auto origin = resolve_expression(*proj.expressions[i], child_bindings)) {
      _origins[duckdb::ColumnBinding(proj.table_index, i)] = *origin;
    }
  }
}

void column_origin_resolver::record_aggregate(duckdb::LogicalOperator& op)
{
  auto& agg = op.Cast<duckdb::LogicalAggregate>();
  // Group keys are usually plain column references and keep their origin. The
  // aggregate results (sums, counts) are new values with no base column, so they
  // are simply left unrecorded.
  const auto child_bindings = child_output_bindings(op);
  for (duckdb::idx_t i = 0; i < agg.groups.size(); ++i) {
    if (auto origin = resolve_expression(*agg.groups[i], child_bindings)) {
      _origins[duckdb::ColumnBinding(agg.group_index, i)] = *origin;
    }
  }
}

std::vector<duckdb::ColumnBinding> column_origin_resolver::child_output_bindings(
  duckdb::LogicalOperator& op)
{
  if (op.children.empty() || !op.children[0]) { return {}; }
  return op.children[0]->GetColumnBindings();
}

std::optional<column_origin> column_origin_resolver::resolve_expression(
  const duckdb::Expression& expr, const std::vector<duckdb::ColumnBinding>& child_bindings) const
{
  switch (expr.GetExpressionType()) {
    case duckdb::ExpressionType::BOUND_COLUMN_REF:
      return lookup(expr.Cast<duckdb::BoundColumnRefExpression>().binding);

    case duckdb::ExpressionType::BOUND_REF: {
      // The plan reaching Sirius has already been binding-resolved, so a
      // pass-through column arrives as a *positional* reference into the child's
      // output rather than as a named binding. Map the position back through the
      // child's bindings to reach the base column.
      const auto index = expr.Cast<duckdb::BoundReferenceExpression>().index;
      if (index >= child_bindings.size()) { return std::nullopt; }
      return lookup(child_bindings[index]);
    }

    default:
      // Computed: arithmetic, cast, function call. No single base column.
      return std::nullopt;
  }
}

std::optional<column_origin> column_origin_resolver::lookup(const duckdb::ColumnBinding& b) const
{
  auto it = _origins.find(b);
  if (it == _origins.end()) { return std::nullopt; }
  return it->second;
}

column_origins column_origin_resolver::origins_of(duckdb::LogicalOperator& op) const
{
  // Lineage is an optimization for the spill compressor: it lets a spilled batch
  // reuse its base table's offline plan instead of exploring for one mid-query.
  // Nothing about correctness depends on it, so a failure to derive it must never
  // fail the query. GetColumnBindings() can throw on an operator whose children a
  // planner has already moved out, and until this guard existed that took the
  // whole GPU plan down and sent the query to DuckDB CPU.
  duckdb::vector<duckdb::ColumnBinding> bindings;
  try {
    bindings = op.GetColumnBindings();
  } catch (const std::exception&) {
    return {};
  }
  column_origins out;
  out.reserve(bindings.size());
  for (auto const& b : bindings) {
    out.push_back(lookup(b));
  }
  return out;
}

}  // namespace sirius::planner
