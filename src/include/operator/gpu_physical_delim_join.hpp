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

#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalGroupedAggregate;

//! PhysicalDelimJoin represents a join where either the LHS or RHS will be duplicate eliminated and pushed into a
//! PhysicalColumnDataScan in the other side. Implementations are PhysicalLeftDelimJoin and PhysicalRightDelimJoin
class GPUPhysicalDelimJoin : public GPUPhysicalOperator {
public:
	GPUPhysicalDelimJoin(PhysicalOperatorType type, vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
	                  vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality, optional_idx delim_idx);

	unique_ptr<GPUPhysicalOperator> join;
	unique_ptr<GPUPhysicalGroupedAggregate> distinct;
	vector<const_reference<GPUPhysicalOperator>> delim_scans;

	optional_idx delim_idx;

public:
	// vector<const_reference<GPUPhysicalOperator>> GetChildren() const override;

	bool IsSink() const override {
		return true;
	}
	// bool ParallelSink() const override {
	// 	return true;
	// }

	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}
	bool SinkOrderDependent() const override {
		return false;
	}

	// InsertionOrderPreservingMap<string> ParamsToString() const override;
};


//! PhysicalRightDelimJoin represents a join where the RHS will be duplicate eliminated and pushed into a
//! PhysicalColumnDataScan in the LHS.
class GPUPhysicalRightDelimJoin : public GPUPhysicalDelimJoin {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::RIGHT_DELIM_JOIN;

public:
	GPUPhysicalRightDelimJoin(vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
	                       vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality,
						   optional_idx delim_idx);

// public:
	// unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	// unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// SinkResultType Sink(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input) const override;
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	// void PrepareFinalize(ClientContext &context, GlobalSinkState &sink_state) const override;
	// SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	//                           OperatorSinkFinalizeInput &input) const override;

public:
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;
};


class GPUPhysicalLeftDelimJoin : public GPUPhysicalDelimJoin {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::LEFT_DELIM_JOIN;

public:
	GPUPhysicalLeftDelimJoin(vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
	                      vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality,
						  optional_idx delim_idx);

	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;

public:
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;
};

} // namespace duckdb
