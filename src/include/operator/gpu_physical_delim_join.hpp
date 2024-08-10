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
	                  vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality);

	unique_ptr<GPUPhysicalOperator> join;
	unique_ptr<GPUPhysicalGroupedAggregate> distinct;
	vector<const_reference<GPUPhysicalOperator>> delim_scans;

public:
	vector<const_reference<GPUPhysicalOperator>> GetChildren() const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}
	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}
	bool SinkOrderDependent() const override {
		return false;
	}

	string ParamsToString() const override;
};


//! PhysicalRightDelimJoin represents a join where the RHS will be duplicate eliminated and pushed into a
//! PhysicalColumnDataScan in the LHS.
class GPUPhysicalRightDelimJoin : public GPUPhysicalDelimJoin {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::RIGHT_DELIM_JOIN;

public:
	GPUPhysicalRightDelimJoin(vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
	                       vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality);

public:
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
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
	                      vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality);

public:
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	// SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	//                           OperatorSinkFinalizeInput &input) const override;

public:
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;
};

} // namespace duckdb
