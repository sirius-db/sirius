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

#include "duckdb/common/types.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"

namespace duckdb {
class ClientContext;
}  // namespace duckdb

namespace sirius::planner {

/// Bind DuckDB's built-in aggregate @p name over one @p argument of type @p type, with an optional
/// FILTER, as the binder would for a query that calls it. The function comes from the host's
/// system catalog. Throws when the aggregate has no overload for @p type.
duckdb::unique_ptr<duckdb::BoundAggregateExpression> bind_builtin_aggregate(
  duckdb::ClientContext& context,
  char const* name,
  duckdb::LogicalType const& type,
  duckdb::unique_ptr<duckdb::Expression> argument,
  duckdb::unique_ptr<duckdb::Expression> filter = nullptr);

/// Whether @p aggregate has no DISTINCT, FILTER, ORDER BY or exported state: the parts a rewrite
/// loses when it keeps only the aggregate's function and arguments.
bool is_plain_aggregate(duckdb::BoundAggregateExpression const& aggregate);

}  // namespace sirius::planner
