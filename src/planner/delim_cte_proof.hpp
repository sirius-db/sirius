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

#include "op/cte_reader_role.hpp"

#include <duckdb/common/common.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>
#include <unordered_map>

namespace duckdb {
class ClientContext;
}

namespace sirius::planner {

/// Reads of delim CTEs whose rows may be restricted, keyed by the `LogicalCTERef::table_index` of
/// the read: the outer-row read, and the key read that only computes those rows' domain.
///
/// DuckDB records the same fact in `LogicalMaterializedCTE::filter_dependency`, but only for native
/// tables, and `Optimizer::Optimize` clears it before every pre-optimize extension runs. So this
/// re-derives it with DuckDB's conditions (`CanRestrictCorrelationDomain`), counting Parquet scans
/// as repeatable as well.
struct delim_cte_proofs {
  std::unordered_map<duckdb::idx_t, sirius::op::cte_reader_role> reader_roles;

  [[nodiscard]] sirius::op::cte_reader_role role_of(duckdb::idx_t cte_ref_table_index) const
  {
    auto const found = reader_roles.find(cte_ref_table_index);
    return found == reader_roles.end() ? sirius::op::cte_reader_role::unknown : found->second;
  }
};

/// Collect the proofs from a plan before DuckDB's built-in optimizers run (they may inline or
/// restructure the delim CTEs). Returns an empty set when the plan holds no provable delim CTE.
delim_cte_proofs collect_delim_cte_proofs(duckdb::ClientContext& context,
                                          duckdb::LogicalOperator& plan);

/// Give back the `filter_dependency` DuckDB cleared before Sirius's pre-optimize hook, on every
/// delim CTE the rule above proves. DuckDB's CTE filter pushers (`CTEFilterPusher`,
/// `CTEJoinFilterPusher`) then work as they do without Sirius loaded, and also over Parquet, which
/// DuckDB on its own never proves. Only valid while the plan is still the one DuckDB's rewrite
/// produced, so call it last in the hook. Returns the number of CTEs given a dependency.
std::size_t restore_cte_filter_dependencies(duckdb::ClientContext& context,
                                            duckdb::LogicalOperator& plan);

}  // namespace sirius::planner
