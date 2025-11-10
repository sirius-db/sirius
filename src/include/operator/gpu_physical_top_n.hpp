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

#pragma once

#include "gpu_physical_operator.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/bound_query_node.hpp"

namespace duckdb {
struct DynamicFilterData;

//! Represents a physical ordering of the data. Note that this will not change
//! the data but only add a selection vector.
class GPUPhysicalTopN : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::TOP_N;

public:
	GPUPhysicalTopN(vector<LogicalType> types_p, vector<BoundOrderByNode> orders, idx_t limit, idx_t offset,
	             shared_ptr<DynamicFilterData> dynamic_filter, idx_t estimated_cardinality);
	~GPUPhysicalTopN() override;

	vector<BoundOrderByNode> orders;
	idx_t limit;
	idx_t offset;
	//! Dynamic table filter (if any)
	shared_ptr<DynamicFilterData> dynamic_filter;
	shared_ptr<GPUIntermediateRelation> sort_result;
public:
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return true;
	}
	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::FIXED_ORDER;
	}

public:
    SinkResultType Sink(GPUIntermediateRelation& input_relation) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}

};

} // namespace duckdb
