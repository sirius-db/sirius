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

#pragma once

#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::planner {

/**
 * @brief Where an operator output column came from in the base tables.
 *
 * A spilled batch's columns are usually verbatim base-table columns — joins,
 * filters, partitions and sorts pass them through untouched. Knowing which base
 * column a spill column *is* lets the compressor reuse the plan already explored
 * offline for that table, instead of running a beam search inside the query.
 */
struct column_origin {
  /// Base table this column came from.
  std::string table_name;
  /// Index of the column in that table's schema (not in the operator's output).
  std::size_t table_column_index{0};
};

/// Per-output-column origin; nullopt where the column is computed (an aggregate
/// result, an arithmetic expression) and so has no single base column.
using column_origins = std::vector<std::optional<column_origin>>;

/**
 * @brief Resolves each logical operator's output columns back to base tables.
 *
 * DuckDB's `ColumnBinding`s already form the lineage graph: every output column
 * of every operator is a `(table_index, column_index)` pair naming the operator
 * that produced it. This walks the plan once, seeding at each `LogicalGet` with
 * the real table and column, and propagating through the operators that
 * introduce their own table_index (projections, aggregates) by following their
 * expressions.
 *
 * Must run *after* `ColumnBindingResolver`, so bindings are resolved.
 */
class column_origin_resolver {
 public:
  /// Walk @p op and every descendant, recording what each binding resolves to.
  void resolve(duckdb::LogicalOperator& op);

  /// Origins for @p op's output columns, in output order. Entries are nullopt
  /// for columns that do not trace back to a single base column.
  [[nodiscard]] column_origins origins_of(duckdb::LogicalOperator& op) const;

 private:
  /// Record what a projection's / aggregate's own output bindings resolve to.
  void record_projection(duckdb::LogicalOperator& op);
  void record_aggregate(duckdb::LogicalOperator& op);
  void record_get(duckdb::LogicalOperator& op);

  /// Resolve a single binding, or nullopt when it is not a base column.
  [[nodiscard]] std::optional<column_origin> lookup(const duckdb::ColumnBinding& b) const;

  /// Output bindings of @p op's first child, against which positional
  /// (`BOUND_REF`) expressions are resolved.
  [[nodiscard]] static std::vector<duckdb::ColumnBinding> child_output_bindings(
    duckdb::LogicalOperator& op);

  /// Follow an expression to a base column when it merely forwards one, whether
  /// it does so by name (`BOUND_COLUMN_REF`) or by position (`BOUND_REF`, which
  /// is what the already-resolved plans reaching Sirius actually contain).
  [[nodiscard]] std::optional<column_origin> resolve_expression(
    const duckdb::Expression& expr, const std::vector<duckdb::ColumnBinding>& child_bindings) const;

  struct binding_hash {
    std::size_t operator()(const duckdb::ColumnBinding& b) const noexcept
    {
      return std::hash<duckdb::idx_t>{}(b.table_index) ^
             (std::hash<duckdb::idx_t>{}(b.column_index) << 1);
    }
  };
  struct binding_eq {
    bool operator()(const duckdb::ColumnBinding& a, const duckdb::ColumnBinding& b) const noexcept
    {
      return a.table_index == b.table_index && a.column_index == b.column_index;
    }
  };

  std::unordered_map<duckdb::ColumnBinding, column_origin, binding_hash, binding_eq> _origins;
};

}  // namespace sirius::planner
