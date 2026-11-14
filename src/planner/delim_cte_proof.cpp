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

#include "planner/delim_cte_proof.hpp"

#include "planner/connector_registry.hpp"

#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/expression_barrier.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_cteref.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_materialized_cte.hpp>
#include <duckdb/planner/operator/logical_recursive_cte.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

namespace sirius::planner {

namespace {

using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;

// DuckDB counts only native tables as repeatable inputs. A Parquet file does not change while a
// query reads it, so Sirius counts Parquet scans too.
bool is_repeatable_scan(duckdb::LogicalGet const& get, duckdb::ClientContext& context)
{
  if (get.GetTable()) { return true; }
  auto const* source = lookup_connector(get, context);
  return source != nullptr &&
         (source->function_name == "read_parquet" || source->function_name == "parquet_scan");
}

// DuckDB's HasPartitionedRecursiveReferences (delim_join_cte_rewriter.cpp).
bool has_partitioned_recursive_references(LogicalOperator& op,
                                          duckdb::LogicalRecursiveCTE const& cte)
{
  if (op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
    auto const& ref = op.Cast<duckdb::LogicalCTERef>();
    if (ref.cte_index == cte.table_index &&
        ref.correlated_columns != cte.correlated_columns.size()) {
      return false;
    }
  }
  for (auto& child : op.children) {
    if (!has_partitioned_recursive_references(*child, cte)) { return false; }
  }
  return true;
}

// DuckDB's HasPartitionedRecursiveKeys (delim_join_cte_rewriter.cpp): decorrelation appends the
// correlation columns to the recursive rows and to the USING KEY targets.
bool has_partitioned_recursive_keys(duckdb::LogicalRecursiveCTE& cte)
{
  if (cte.key_targets.empty() && !cte.ref_recurring) { return true; }
  if (cte.correlated_columns.empty() || cte.column_count < cte.correlated_columns.size()) {
    return false;
  }
  auto const offset = cte.column_count - cte.correlated_columns.size();
  for (duckdb::idx_t i = offset; i < cte.column_count; i++) {
    bool found = false;
    for (auto const& key : cte.key_targets) {
      if (key->GetExpressionClass() == duckdb::ExpressionClass::BOUND_COLUMN_REF &&
          key->Cast<duckdb::BoundColumnRefExpression>().Binding() ==
            duckdb::ColumnBinding(cte.table_index, duckdb::ProjectionIndex(i))) {
        found = true;
        break;
      }
    }
    if (!found) { return false; }
  }
  return has_partitioned_recursive_references(*cte.children[1], cte);
}

// DuckDB's CanRestrictCorrelationDomain (delim_join_cte_rewriter.cpp), with Parquet scans counted
// as repeatable.
bool can_restrict_domain(LogicalOperator& op,
                         std::unordered_set<duckdb::idx_t>& local_ctes,
                         duckdb::ClientContext& context)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET:
      if (!is_repeatable_scan(op.Cast<duckdb::LogicalGet>(), context)) { return false; }
      break;
    case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
      auto& cte = op.Cast<duckdb::LogicalRecursiveCTE>();
      if (!has_partitioned_recursive_keys(cte)) { return false; }
      local_ctes.insert(cte.table_index.index);
      break;
    }
    case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
      local_ctes.insert(op.Cast<duckdb::LogicalMaterializedCTE>().table_index.index);
      break;
    case LogicalOperatorType::LOGICAL_CTE_REF:
      if (!local_ctes.count(op.Cast<duckdb::LogicalCTERef>().cte_index.index)) { return false; }
      break;
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    case LogicalOperatorType::LOGICAL_PROJECTION:
    case LogicalOperatorType::LOGICAL_FILTER:
    case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
    case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
    case LogicalOperatorType::LOGICAL_DISTINCT:
    case LogicalOperatorType::LOGICAL_WINDOW:
    case LogicalOperatorType::LOGICAL_LIMIT:
    case LogicalOperatorType::LOGICAL_ORDER_BY:
    case LogicalOperatorType::LOGICAL_UNION:
    case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
    case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
    case LogicalOperatorType::LOGICAL_EXPRESSION_GET: break;
    default: return false;
  }
  bool repeatable = true;
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(
    op, [&](duckdb::unique_ptr<duckdb::Expression>* expr) {
      repeatable &= !(*expr)->IsVolatile() && !duckdb::ExpressionBarrier::Contains(**expr);
      duckdb::ExpressionIterator::VisitExpression<duckdb::BoundFunctionExpression>(
        **expr, [&](duckdb::BoundFunctionExpression const& func) {
          repeatable &= !func.Function().RequiresOrderedExecution();
        });
    });
  if (!repeatable) { return false; }
  for (auto& child : op.children) {
    if (!can_restrict_domain(*child, local_ctes, context)) { return false; }
  }
  return true;
}

LogicalOperator* skip_projections(LogicalOperator* op)
{
  while (op->type == LogicalOperatorType::LOGICAL_PROJECTION && op->children.size() == 1) {
    op = op->children[0].get();
  }
  return op;
}

duckdb::LogicalCTERef* as_ref_of(LogicalOperator* op, duckdb::idx_t cte_table_index)
{
  if (op == nullptr || op->type != LogicalOperatorType::LOGICAL_CTE_REF) { return nullptr; }
  auto& ref = op->Cast<duckdb::LogicalCTERef>();
  return ref.cte_index.index == cte_table_index ? &ref : nullptr;
}

// DuckDB's rewrite names each CTE it generates after the CTE's own table index, so a user CTE that
// only borrows the prefix does not match.
bool is_generated(duckdb::LogicalMaterializedCTE const& cte, std::string_view prefix)
{
  return cte.ctename.GetIdentifierName() ==
         std::string(prefix) + std::to_string(cte.table_index.index);
}

// The parts of one delim CTE as DuckDB's delim-to-CTE rewrite produces it:
//   CTE __duckdb_delim_N (children[0] = outer rows,
//     children[1] = CTE __duckdb_delim_dedup_M (children[0] = DISTINCT over a read of N,
//       children[1] = the join-back: probe = a read of N, build = the subquery)).
struct delim_cte_parts {
  duckdb::LogicalMaterializedCTE* dedup    = nullptr;
  duckdb::LogicalComparisonJoin* join_back = nullptr;
  duckdb::LogicalCTERef* outer_row_read    = nullptr;
  duckdb::LogicalCTERef* key_read          = nullptr;
};

std::optional<delim_cte_parts> match_delim_cte(duckdb::LogicalMaterializedCTE& cte)
{
  if (!is_generated(cte, "__duckdb_delim_") || cte.children.size() != 2) { return std::nullopt; }
  auto* const dedup_op = skip_projections(cte.children[1].get());
  if (dedup_op->type != LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) { return std::nullopt; }
  auto& dedup = dedup_op->Cast<duckdb::LogicalMaterializedCTE>();
  if (!is_generated(dedup, "__duckdb_delim_dedup_") || dedup.children.size() != 2) {
    return std::nullopt;
  }
  // The rewrite's dedup body is a bare DISTINCT over a read of the delim CTE. SQL cannot produce
  // one, since the binder plans a projection above every aggregate; so a user CTE, whose subquery
  // need not be partitioned by the keys, never matches.
  auto& distinct = *dedup.children[0];
  if (distinct.type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY ||
      distinct.children.size() != 1) {
    return std::nullopt;
  }
  auto const& aggregate = distinct.Cast<duckdb::LogicalAggregate>();
  if (!aggregate.expressions.empty() || !aggregate.grouping_functions.empty()) {
    return std::nullopt;
  }
  auto* const key_read = as_ref_of(distinct.children[0].get(), cte.table_index.index);
  auto* const join_op  = skip_projections(dedup.children[1].get());
  if (key_read == nullptr || join_op->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return std::nullopt;
  }
  auto& join_back = join_op->Cast<duckdb::LogicalComparisonJoin>();
  if (join_back.children.size() != 2) { return std::nullopt; }
  auto* const outer_row_read = as_ref_of(join_back.children[0].get(), cte.table_index.index);
  if (outer_row_read == nullptr) { return std::nullopt; }
  return delim_cte_parts{.dedup          = &dedup,
                         .join_back      = &join_back,
                         .outer_row_read = outer_row_read,
                         .key_read       = key_read};
}

// DuckDB's conditions (MaterializeDelimJoinAsCTE): the join-back keeps an outer row only through
// that row, and the subquery depends only on repeatable inputs partitioned by the domain.
bool can_restrict(delim_cte_parts const& parts, duckdb::ClientContext& context)
{
  using duckdb::JoinType;
  auto const type              = parts.join_back->join_type;
  bool const requires_left_row = type == JoinType::INNER || type == JoinType::LEFT ||
                                 type == JoinType::SEMI || type == JoinType::ANTI ||
                                 type == JoinType::MARK;
  std::unordered_set<duckdb::idx_t> local_ctes{parts.dedup->table_index.index};
  return requires_left_row &&
         can_restrict_domain(*parts.join_back->children[1], local_ctes, context);
}

template <class Fn>
void for_each_delim_cte(LogicalOperator& op, Fn const& fn)
{
  if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
    auto& cte = op.Cast<duckdb::LogicalMaterializedCTE>();
    if (auto const parts = match_delim_cte(cte)) { fn(cte, *parts); }
  }
  for (auto& child : op.children) {
    for_each_delim_cte(*child, fn);
  }
}

}  // namespace

delim_cte_proofs collect_delim_cte_proofs(duckdb::ClientContext& context, LogicalOperator& plan)
{
  delim_cte_proofs proofs;
  for_each_delim_cte(plan, [&](duckdb::LogicalMaterializedCTE&, delim_cte_parts const& parts) {
    if (!can_restrict(parts, context)) { return; }
    proofs.reader_roles[parts.outer_row_read->table_index.index] =
      sirius::op::cte_reader_role::outer_row;
    proofs.reader_roles[parts.key_read->table_index.index] = sirius::op::cte_reader_role::key;
  });
  return proofs;
}

std::size_t restore_cte_filter_dependencies(duckdb::ClientContext& context, LogicalOperator& plan)
{
  std::size_t restored = 0;
  for_each_delim_cte(plan, [&](duckdb::LogicalMaterializedCTE& cte, delim_cte_parts const& parts) {
    if (cte.filter_dependency || !can_restrict(parts, context)) { return; }
    cte.filter_dependency = duckdb::make_uniq<duckdb::CTEFilterDependency>(
      parts.outer_row_read->table_index, parts.key_read->table_index);
    ++restored;
  });
  return restored;
}

}  // namespace sirius::planner
