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
void cudf_orderby(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& projection, uint64_t num_keys, uint64_t num_projections, OrderByType* order_by_type, idx_t num_results = 0); 

void orderByString(uint8_t** col_keys, uint64_t** col_offsets, int* sort_orders, uint64_t* col_num_bytes, uint64_t num_rows, uint64_t num_cols);

class GPUPhysicalOrder : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::ORDER_BY;
	
public:
    GPUPhysicalOrder(vector<LogicalType> types, vector<BoundOrderByNode> orders, vector<idx_t> projections_p,
	              idx_t estimated_cardinality);

	//! Input data
	vector<BoundOrderByNode> orders;
	vector<idx_t> projections;
	shared_ptr<GPUIntermediateRelation> sort_result;

public:
	// Source interface
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::FIXED_ORDER;
	}

public:
	// Sink interface
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}
	bool SinkOrderDependent() const override {
		return false;
	}

};
} // namespace duckdb