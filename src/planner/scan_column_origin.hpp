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

#include <cstddef>
#include <optional>

namespace duckdb {
class LogicalGet;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/// A base-scan column that an operator output carries.
struct scan_column_origin {
  duckdb::LogicalGet const* get;
  std::size_t ordinal;  ///< position in the scan's output
};

/// What an output column must preserve of the scan column it descends from.
enum class origin_policy {
  /// Every output row is a scan row at most once, so the scan's row count bounds the output
  /// (dynamic-filter domain coverage): a join contributes only a side it cannot duplicate, and an
  /// aggregate with several grouping sets is refused.
  row_subset,
  /// Only the values are unchanged (magnitude bounds): either side of any join or cross product
  /// qualifies, since row multiplication and NULL padding add no values.
  value_preserving,
};

/**
 * @brief Traces @p output_ordinal of @p subtree to the base-scan column it carries, or nullopt
 *
 * Follows projections of bare references, filters, sorts, limits, DISTINCT and GROUP BY keys, and
 * joins as @p policy allows. Stops at a computed column, an unmodelled operator, a table in-out
 * function, or an ordinal past the scan's width.
 *
 * @pre Types and column bindings are resolved on @p subtree.
 */
[[nodiscard]] std::optional<scan_column_origin> resolve_scan_column_origin(
  duckdb::LogicalOperator const& subtree, std::size_t output_ordinal, origin_policy policy);

}  // namespace sirius::planner
