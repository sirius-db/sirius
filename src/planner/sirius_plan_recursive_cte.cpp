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

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/execution/perfect_aggregate_hashtable.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_recursive_cte.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_cte.hpp"
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
      op.chunk_types,
      duckdb::PhysicalOperatorType::CTE_SCAN,
      op.estimated_cardinality,
      op.cte_index);

    auto cte = recursive_cte_tables.find(op.cte_index);
    if (cte == recursive_cte_tables.end()) {
      throw duckdb::InvalidInputException("Referenced materialized CTE does not exist.");
    }

    chunk_scan->collection = cte->second.get();
    // materialized_cte->second.push_back(*chunk_scan.get())

    // auto gpu_cte = gpu_recursive_cte_tables.find(op.cte_index);
    // if (gpu_cte == gpu_recursive_cte_tables.end()) {
    //   throw duckdb::InvalidInputException("Referenced materialized CTE does not exist.");
    // }
    // chunk_scan->intermediate_relation = gpu_cte->second;

    materialized_cte->second.push_back(*chunk_scan.get());

    return std::move(chunk_scan);
  }

  throw duckdb::NotImplementedException("Recursive CTE is not implemented");

  auto cte = recursive_cte_tables.find(op.cte_index);
  if (cte == recursive_cte_tables.end()) {
    throw duckdb::InvalidInputException("Referenced recursive CTE does not exist.");
  }

  // If we found a recursive CTE and we want to scan the recurring table, we search for it,
  if (op.is_recurring) {
    cte = recurring_cte_tables.find(op.cte_index);
    if (cte == recurring_cte_tables.end()) {
      throw duckdb::InvalidInputException(
        "RECURRING can only be used with USING KEY in recursive CTE.");
    }
  }

  auto& types     = cte->second.get()->Types();
  auto op_type    = op.is_recurring ? duckdb::PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN
                                    : duckdb::PhysicalOperatorType::RECURSIVE_CTE_SCAN;
  auto chunk_scan = duckdb::make_uniq<sirius::op::sirius_physical_column_data_scan>(
    cte->second.get()->Types(),
    duckdb::PhysicalOperatorType::RECURSIVE_CTE_SCAN,
    op.estimated_cardinality,
    op.cte_index);

  chunk_scan->collection = cte->second.get();
  return std::move(chunk_scan);
}

}  // namespace sirius::planner
