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
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "op/sirius_physical_limit.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

// bool use_batch_limit(duckdb::BoundLimitNode &limit_val, duckdb::BoundLimitNode &offset_val) {
// #ifdef DUCKDB_ALTERNATIVE_VERIFY
// 	return true;
// #else
// 	// we only use batch limit when we are computing a small amount of values
// 	// as the batch limit materializes this many rows PER thread
// 	static constexpr const duckdb::idx_t BATCH_LIMIT_THRESHOLD = 10000;

// 	if (limit_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
// 		return false;
// 	}
// 	if (offset_val.Type() == duckdb::LimitNodeType::EXPRESSION_VALUE) {
// 		return false;
// 	}
// 	duckdb::idx_t total_offset = limit_val.GetConstantValue();
// 	if (offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
// 		total_offset += offset_val.GetConstantValue();
// 	}
// 	return total_offset <= BATCH_LIMIT_THRESHOLD;
// #endif
// }

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalLimit& op)
{
  D_ASSERT(op.children.size() == 1);

  auto plan = create_plan(*op.children[0]);

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> limit;
  switch (op.limit_val.Type()) {
    case duckdb::LimitNodeType::EXPRESSION_PERCENTAGE:
    case duckdb::LimitNodeType::CONSTANT_PERCENTAGE:
      throw duckdb::NotImplementedException("Percentage limit not supported in GPU");
      // limit = duckdb::make_uniq<PhysicalLimitPercent>(op.types, std::move(op.limit_val),
      // std::move(op.offset_val),
      //                                         op.estimated_cardinality);
      break;
    default:
      if (!preserve_insertion_order(*plan)) {
        // use parallel streaming limit if insertion order is not important
        limit =
          duckdb::make_uniq<sirius::op::sirius_physical_streaming_limit>(op.types,
                                                                         std::move(op.limit_val),
                                                                         std::move(op.offset_val),
                                                                         op.estimated_cardinality,
                                                                         true);
      } else {
        throw duckdb::NotImplementedException(
          "Streaming limit with insertion order preservation not supported in GPU");
        // maintaining insertion order is important
        // 	if (use_batch_index(*plan) && use_batch_limit(op.limit_val, op.offset_val)) {
        // 		// source supports batch index: use parallel batch limit
        // 		throw duckdb::NotImplementedException("Batch limit not supported in GPU");
        // 		// limit = duckdb::make_uniq<PhysicalLimit>(op.types, std::move(op.limit_val),
        // std::move(op.offset_val),
        // 		//                                  op.estimated_cardinality);
        // 	} else {
        // 		// source does not support batch index: use a non-parallel streaming limit
        // 		limit = duckdb::make_uniq<sirius::op::sirius_physical_streaming_limit>(op.types,
        // std::move(op.limit_val), std::move(op.offset_val), op.estimated_cardinality, false);
        // 	}
      }
      break;
  }

  limit->children.push_back(std::move(plan));
  return limit;
}

}  // namespace sirius::planner
