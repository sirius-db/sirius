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

#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace duckdb {
class ClientContext;
class LogicalAggregate;
class LogicalGet;
}  // namespace duckdb

/// A tree of inner joins, cross products, filters and projections over leaves is the product of
/// its leaves filtered by the conjunction of every predicate in it, whatever the join order.
/// Flattening puts a subtree into that form: its leaves, its predicates, and each output column
/// as an expression over leaf columns. Expressions name a leaf column canonically as
/// `ColumnBinding(leaf ordinal, column)`. A scan's column is its base-table column, so the same
/// column of two scans of one source has the same number, and two flattened subtrees whose leaves
/// read the same rows can be compared predicate by predicate.
namespace sirius::planner::flat {

enum class leaf_kind { scan, cte_ref, keys, opaque };

struct leaf {
  leaf_kind kind;
  duckdb::LogicalOperator* op;
};

struct flat_tree {
  std::vector<leaf> leaves;
  /// Output binding to its value over the leaf columns
  duckdb::column_binding_map_t<duckdb::unique_ptr<duckdb::Expression>> columns;
  std::vector<duckdb::unique_ptr<duckdb::Expression>> predicates;
};

/// Where a subquery gets the outer rows' distinct keys from. Set exactly one of the three kinds;
/// `flatten` then records the key relation it finds as a `keys` leaf.
///  - `delim_get`: a one-column DELIM_GET.
///  - `cte`: a SEMI join against the distinct values of one column of a CTE, the form an inlined
///    dedup CTE takes in an optimized plan.
///  - `dedup_ref`: a SEMI join against a scan of the CTE `cte_index`, the form DuckDB's
///    delim-to-CTE rewrite produces before the optimizer runs. The keys leaf has that CTE's
///    columns.
struct key_source {
  bool delim_get = false;
  bool cte       = false;
  bool dedup_ref = false;
  std::optional<std::size_t> leaf;
  std::optional<duckdb::TableIndex> cte_index;
  duckdb::idx_t cte_column = 0;
};

duckdb::unique_ptr<duckdb::Expression> column(duckdb::LogicalType const& type,
                                              duckdb::ColumnBinding binding);
bool is_column(duckdb::Expression const& expression);
duckdb::ColumnBinding binding_of(duckdb::Expression const& expression);
duckdb::ColumnBinding canonical(std::size_t leaf, duckdb::idx_t column);

/// @p expression over the tree's leaf columns, or null when it reads a column the tree does not
/// output or is volatile.
duckdb::unique_ptr<duckdb::Expression> resolve(duckdb::Expression const& expression,
                                               flat_tree const& tree);

/// A plain GROUP BY of its one group.
bool grouped_by_its_group(duckdb::LogicalAggregate const& aggregate);

/// Flatten @p op into @p tree. Operators other than inner joins, cross products, filters,
/// projections, scans and (with @p keys) the key source become opaque leaves. False when a
/// predicate or column cannot be expressed over the leaves.
bool flatten(duckdb::LogicalOperator& op, flat_tree& tree, key_source* keys);

/// Whether two scans read the same view of the same source, including what selects its rows (two
/// Iceberg snapshots of the same files are different sources).
bool same_source(duckdb::ClientContext& context,
                 duckdb::LogicalGet const& a,
                 duckdb::LogicalGet const& b);

/// Whether two leaves produce the same rows: scans of one view, or scans of one CTE.
bool same_rows(duckdb::ClientContext& context, leaf const& a, leaf const& b);

std::vector<duckdb::ColumnBinding> columns_read(duckdb::Expression const& expression);

/// Equal, or the same comparison with its operands swapped.
bool same_predicate(duckdb::Expression const& a, duckdb::Expression const& b);

/// `x = y` between two plain columns.
std::optional<std::pair<duckdb::ColumnBinding, duckdb::ColumnBinding>> column_equality(
  duckdb::Expression const& predicate);

}  // namespace sirius::planner::flat
