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

#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_order.hpp"

#include "gpu_physical_order.hpp"
#include "gpu_physical_plan_generator.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalOrder &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);
	if (!op.orders.empty()) {
		vector<idx_t> projection_map;
		if (op.HasProjectionMap()) {
			projection_map = std::move(op.projection_map);
		} else {
			for (idx_t i = 0; i < plan->types.size(); i++) {
				projection_map.push_back(i);
			}
		}
		auto order = make_uniq<GPUPhysicalOrder>(op.types, std::move(op.orders), std::move(projection_map),
		                                      op.estimated_cardinality);
		order->children.push_back(std::move(plan));
		plan = std::move(order);
	}
	return plan;
}


} // namespace duckdb
