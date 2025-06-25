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

#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"

#include "gpu_physical_dummy_scan.hpp"
#include "gpu_physical_plan_generator.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalDummyScan &op) {
	D_ASSERT(op.children.size() == 0);
	return make_uniq<GPUPhysicalDummyScan>(op.types, op.estimated_cardinality);
}

} // namespace duckdb
