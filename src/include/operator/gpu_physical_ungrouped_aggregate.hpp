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
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/common/unordered_map.hpp"

namespace duckdb {

// template <typename T>
// void ungroupedAggregate(uint8_t **a, uint8_t **result, uint64_t N, int* agg_mode, int num_aggregates);

void cudf_aggregate(vector<shared_ptr<GPUColumn>>& column, uint64_t num_aggregates, AggregationType* agg_mode);

class GPUPhysicalUngroupedAggregate : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::UNGROUPED_AGGREGATE;

public:
    GPUPhysicalUngroupedAggregate(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

	//! The aggregates that have to be computed
	vector<unique_ptr<Expression>> aggregates;
	unique_ptr<DistinctAggregateData> distinct_data;
	unique_ptr<DistinctAggregateCollectionInfo> distinct_collection_info;
	shared_ptr<GPUIntermediateRelation> aggregation_result;

// public:
// 	// Source interface
	// SourceResultType GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const override;
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return true;
	}

public:
	// Sink interface
	// SinkResultType Sink(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input) const override;
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	// SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	//                           OperatorSinkFinalizeInput &input) const override;

	// unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	// string ParamsToString() const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	// bool SinkOrderDependent() const override;

// private:
	// //! Finalize the distinct aggregates
	// SinkFinalizeType FinalizeDistinct(Pipeline &pipeline, Event &event, ClientContext &context,
	//                                   GlobalSinkState &gstate) const;
	// //! Combine the distinct aggregates
	// void CombineDistinct(ExecutionContext &context, OperatorSinkCombineInput &input) const;
	// //! Sink the distinct aggregates
	// void SinkDistinct(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input) const;
	void SinkDistinct(GPUIntermediateRelation &input_relation) const;
    
};
} // namespace duckdb