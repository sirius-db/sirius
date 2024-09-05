#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalEmptyResult : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EMPTY_RESULT;

public:
	explicit GPUPhysicalEmptyResult(vector<LogicalType> types, idx_t estimated_cardinality)
	    : GPUPhysicalOperator(PhysicalOperatorType::EMPTY_RESULT, std::move(types), estimated_cardinality) {
	}

public:
	// SourceResultType GetData(ExecutionContext &context, GPUIntermediateRelation& output_relation, OperatorSourceInput &input) const override;
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return true;
	}
};
} // namespace duckdb
