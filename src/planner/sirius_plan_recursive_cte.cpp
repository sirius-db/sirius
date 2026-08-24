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

#include "duckdb/planner/operator/logical_cteref.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalCTERef& op)
{
  D_ASSERT(op.children.empty());

  // Check if this LogicalCTERef is supposed to scan a materialized CTE.
  // Lookup if there is a materialized CTE for the cte_index.
  auto materialized_cte = materialized_ctes.find(op.cte_index);

  // If this check fails, this is a reference to a materialized recursive CTE.
  if (materialized_cte != materialized_ctes.end()) {
    auto chunk_scan = duckdb::make_uniq<sirius::op::sirius_physical_column_data_scan>(
      sirius::from_duckdb_vec(op.chunk_types),
      sirius::op::SiriusPhysicalOperatorType::CTE_SCAN,
      op.estimated_cardinality,
      op.cte_index);

    auto cte = recursive_cte_tables.find(op.cte_index);
    if (cte == recursive_cte_tables.end()) {
      throw duckdb::InvalidInputException("Referenced materialized CTE does not exist.");
    }

    chunk_scan->collection = cte->second.get();

    materialized_cte->second.push_back(*chunk_scan.get());

    return std::move(chunk_scan);
  }

  throw duckdb::NotImplementedException("Recursive CTE is not implemented");
}

}  // namespace sirius::planner
