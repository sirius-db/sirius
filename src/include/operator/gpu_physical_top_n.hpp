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
	// Source interface
	// unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	// SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return true;
	}
	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::FIXED_ORDER;
	}

public:
	// SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
    SinkResultType Sink(GPUIntermediateRelation& input_relation) const override;
	// SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	//                           OperatorSinkFinalizeInput &input) const override;
	// unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	// unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}

	// InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
