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
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"

#include "gpu_physical_cte.hpp"
#include "gpu_physical_plan_generator.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalMaterializedCTE &op) {
	D_ASSERT(op.children.size() == 2);

	// Create the working_table that the PhysicalCTE will use for evaluation.
	auto working_table = make_shared_ptr<ColumnDataCollection>(context, op.children[0]->types);
	auto working_table_gpu = make_shared_ptr<GPUIntermediateRelation>(op.children[0]->types.size());

	// Add the ColumnDataCollection to the context of this PhysicalPlanGenerator
	recursive_cte_tables[op.table_index] = working_table;
	gpu_recursive_cte_tables[op.table_index] = working_table_gpu;
	materialized_ctes[op.table_index] = vector<const_reference<GPUPhysicalOperator>>();

	// Create the plan for the left side. This is the materialization.
	auto left = CreatePlan(*op.children[0]);
	// Initialize an empty vector to collect the scan operators.
	auto right = CreatePlan(*op.children[1]);

	unique_ptr<GPUPhysicalCTE> cte;
	cte = make_uniq<GPUPhysicalCTE>(op.ctename, op.table_index, right->types, std::move(left), std::move(right),
	                             op.estimated_cardinality);
	cte->working_table = working_table;
	cte->working_table_gpu = working_table_gpu;
	cte->cte_scans = materialized_ctes[op.table_index];

	return std::move(cte);
}

} // namespace duckdb