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

#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

#include "gpu_physical_plan_generator.hpp"
#include "gpu_physical_limit.hpp"

namespace duckdb {

// bool UseBatchLimit(BoundLimitNode &limit_val, BoundLimitNode &offset_val) {
// #ifdef DUCKDB_ALTERNATIVE_VERIFY
// 	return true;
// #else
// 	// we only use batch limit when we are computing a small amount of values
// 	// as the batch limit materializes this many rows PER thread
// 	static constexpr const idx_t BATCH_LIMIT_THRESHOLD = 10000;

// 	if (limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
// 		return false;
// 	}
// 	if (offset_val.Type() == LimitNodeType::EXPRESSION_VALUE) {
// 		return false;
// 	}
// 	idx_t total_offset = limit_val.GetConstantValue();
// 	if (offset_val.Type() == LimitNodeType::CONSTANT_VALUE) {
// 		total_offset += offset_val.GetConstantValue();
// 	}
// 	return total_offset <= BATCH_LIMIT_THRESHOLD;
// #endif
// }

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalLimit &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	unique_ptr<GPUPhysicalOperator> limit;
	switch (op.limit_val.Type()) {
	case LimitNodeType::EXPRESSION_PERCENTAGE:
	case LimitNodeType::CONSTANT_PERCENTAGE:
		throw NotImplementedException("Percentage limit not supported in GPU");
		// limit = make_uniq<PhysicalLimitPercent>(op.types, std::move(op.limit_val), std::move(op.offset_val),
		//                                         op.estimated_cardinality);
		break;
	default:
		// if (!PreserveInsertionOrder(*plan)) {
			// use parallel streaming limit if insertion order is not important
			limit = make_uniq<GPUPhysicalStreamingLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
			                                          op.estimated_cardinality, true);
		// } else {
		// 	// maintaining insertion order is important
		// 	if (UseBatchIndex(*plan) && UseBatchLimit(op.limit_val, op.offset_val)) {
		// 		// source supports batch index: use parallel batch limit
		// 		throw NotImplementedException("Batch limit not supported in GPU");
		// 		// limit = make_uniq<PhysicalLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
		// 		//                                  op.estimated_cardinality);
		// 	} else {
		// 		// source does not support batch index: use a non-parallel streaming limit
		// 		limit = make_uniq<GPUPhysicalStreamingLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
		// 		                                          op.estimated_cardinality, false);
		// 	}
		// }
		break;
	}

	limit->children.push_back(std::move(plan));
	return limit;
}

} // namespace duckdb
