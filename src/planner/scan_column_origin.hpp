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

/// Trace unchanged columns through row subsets to their base scan and output ordinal.
/// Returns nullopt for computed columns, unmodelled operators, or joins that can repeat rows.
/// Types and column bindings must be resolved on the subtree.
[[nodiscard]] std::optional<scan_column_origin> resolve_scan_column_origin(
  duckdb::LogicalOperator const& subtree, std::size_t output_ordinal) noexcept;

}  // namespace sirius::planner
