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

  // A materialized CTE registers itself before its consumer is planned. A reference to anything
  // else reads a recursive CTE's working table, which Sirius does not plan.
  auto materialized_cte = materialized_ctes.find(op.cte_index);
  if (materialized_cte == materialized_ctes.end()) {
    throw duckdb::NotImplementedException("Recursive CTE is not implemented");
  }

  // The scan holds no data: the CTE's batches are routed to this scan's parent.
  auto cte_scan = duckdb::make_uniq<sirius::op::sirius_physical_column_data_scan>(
    sirius::from_duckdb_vec(op.chunk_types),
    sirius::op::SiriusPhysicalOperatorType::CTE_SCAN,
    op.estimated_cardinality,
    op.cte_index.index);
  if (cte_reader_proofs) {
    cte_scan->reader_role = cte_reader_proofs->role_of(op.table_index.index);
  }

  materialized_cte->second.push_back(*cte_scan.get());

  return std::move(cte_scan);
}

}  // namespace sirius::planner
