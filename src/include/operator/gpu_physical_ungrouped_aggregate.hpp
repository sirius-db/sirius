#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalUngroupedAggregate : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::UNGROUPED_AGGREGATE;

public:
    GPUPhysicalUngroupedAggregate(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

	//! The aggregates that have to be computed
	vector<unique_ptr<Expression>> aggregates;
// 	unique_ptr<DistinctAggregateData> distinct_data;
// 	unique_ptr<DistinctAggregateCollectionInfo> distinct_collection_info;

// public:
// 	// Source interface
// 	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

public:
	// Sink interface
	// SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
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
	// void SinkDistinct(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const;
    
};
} // namespace duckdb