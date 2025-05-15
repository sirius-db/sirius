//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/set/physical_cte.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

// class RecursiveCTEState;

class GPUPhysicalCTE : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::CTE;

public:
	GPUPhysicalCTE(string ctename, idx_t table_index, vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> top,
	            unique_ptr<GPUPhysicalOperator> bottom, idx_t estimated_cardinality);
	~GPUPhysicalCTE() override;

	vector<const_reference<GPUPhysicalOperator>> cte_scans;

	shared_ptr<ColumnDataCollection> working_table;

	shared_ptr<GPUIntermediateRelation> working_table_gpu;

	idx_t table_index;
	string ctename;

public:
	// Sink interface
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;

	// unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	// unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;

	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	bool SinkOrderDependent() const override {
		return false;
	}

	// InsertionOrderPreservingMap<string> ParamsToString() const override;

public:
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;

	vector<const_reference<GPUPhysicalOperator>> GetSources() const override;
};

} // namespace duckdb
