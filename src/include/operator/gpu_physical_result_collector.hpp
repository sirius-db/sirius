#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPreparedStatementData;

class GPUPhysicalResultCollector : public GPUPhysicalOperator {
// public:
//     GPUPhysicalResultCollector(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

public:
	explicit GPUPhysicalResultCollector(GPUPreparedStatementData &data);

	StatementType statement_type;
	StatementProperties properties;
	GPUPhysicalOperator &plan;
	vector<string> names;

public:
	bool IsSink() const override {
		return true;
	}

public:
	vector<const_reference<GPUPhysicalOperator>> GetChildren() const override;
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;

	bool IsSource() const override {
		return true;
	}

};
} // namespace duckdb