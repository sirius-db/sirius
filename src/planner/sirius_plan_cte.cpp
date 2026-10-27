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
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "op/sirius_physical_cte.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <unordered_map>

namespace sirius::planner {

namespace {

using cte_read_cache =
  std::unordered_map<const duckdb::LogicalOperator*, std::unordered_map<duckdb::idx_t, bool>>;

// Whether `op` will plan a CTE_SCAN of CTE `cte_index`. A nested CTE that is itself unread
// keeps no definition, so references inside it do not count. Cache each (operator, CTE) pair:
// checking nested CTE liveness otherwise revisits the same consumer exponentially often.
bool reads_cte(const duckdb::LogicalOperator& op,
               duckdb::TableIndex cte_index,
               cte_read_cache& cache)
{
  auto& results = cache[&op];
  if (auto it = results.find(cte_index.index); it != results.end()) { return it->second; }

  bool found = false;
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_CTE_REF) {
    found = op.Cast<duckdb::LogicalCTERef>().cte_index.index == cte_index.index;
  } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE &&
             !reads_cte(
               *op.children[1], op.Cast<duckdb::LogicalMaterializedCTE>().table_index, cache)) {
    found = reads_cte(*op.children[1], cte_index, cache);
  } else {
    for (auto const& child : op.children) {
      if (reads_cte(*child, cte_index, cache)) {
        found = true;
        break;
      }
    }
  }
  results[cte_index.index] = found;
  return found;
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalMaterializedCTE& op)
{
  D_ASSERT(op.children.size() == 2);

  // Optimizers can fold every reference away (e.g. statistics proving the CTE empty) after
  // unused-CTE removal has run. Plan only the consumer: a CTE pipeline with no CTE_SCAN has
  // no outgoing edge, so the active task creator never reaches the consumer's sources.
  cte_read_cache cache;
  if (!reads_cte(*op.children[1], op.table_index, cache)) {
    // DuckDB retains unreferenced modifying CTEs so they execute once. Sirius cannot plan their
    // writes; reject the GPU plan so DuckDB executes the original statement on the CPU.
    if (op.children[0]->HasSideEffects()) {
      throw duckdb::NotImplementedException("Side-effecting materialized CTE not supported");
    }
    return create_plan(*op.children[1]);
  }

  // Create the working_table that the PhysicalCTE will use for evaluation.
  auto working_table =
    duckdb::make_shared_ptr<duckdb::ColumnDataCollection>(context, op.children[0]->types);

  // Add the ColumnDataCollection to the context of this PhysicalPlanGenerator
  recursive_cte_tables[op.table_index] = working_table;
  materialized_ctes[op.table_index] =
    duckdb::vector<duckdb::const_reference<sirius::op::sirius_physical_operator>>();

  // Create the plan for the left side. This is the materialization.
  auto left = create_plan(*op.children[0]);
  // Initialize an empty vector to collect the scan operators.
  auto right = create_plan(*op.children[1]);

  // V2 uses materialized CTEs for correlated joins: parents need the consumer schema,
  // while the producer-forwarding stage opts out of the runtime output-schema check.
  auto result_types = right->types;
  duckdb::unique_ptr<sirius::op::sirius_physical_cte> cte;
  cte = duckdb::make_uniq<sirius::op::sirius_physical_cte>(op.ctename.GetIdentifierName(),
                                                           op.table_index.index,
                                                           std::move(result_types),
                                                           std::move(left),
                                                           std::move(right),
                                                           op.estimated_cardinality);
  cte->working_table = working_table;
  // cte->working_table_gpu = working_table_gpu;
  cte->cte_scans = materialized_ctes[op.table_index];

  return std::move(cte);
}

}  // namespace sirius::planner
