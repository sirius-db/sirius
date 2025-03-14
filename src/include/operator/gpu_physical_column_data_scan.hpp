#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

//! The PhysicalColumnDataScan scans a ColumnDataCollection
class GPUPhysicalColumnDataScan : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

public:
	GPUPhysicalColumnDataScan(vector<LogicalType> types, PhysicalOperatorType op_type, idx_t estimated_cardinality,
	                       optionally_owned_ptr<ColumnDataCollection> collection);

	GPUPhysicalColumnDataScan(vector<LogicalType> types, PhysicalOperatorType op_type, idx_t estimated_cardinality,
	                       idx_t cte_index);

	//! (optionally owned) column data collection to scan
	optionally_owned_ptr<ColumnDataCollection> collection;

	idx_t cte_index;
	optional_idx delim_index;

	GPUIntermediateRelation* intermediate_relation;

public:
	// unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	// SourceResultType GetData(ExecutionContext &context, GPUIntermediateRelation& output_relation, OperatorSourceInput &input) const override;
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;


	bool IsSource() const override {
		return true;
	}

	// string ParamsToString() const override;

// public:
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;
};

} // namespace duckdb
