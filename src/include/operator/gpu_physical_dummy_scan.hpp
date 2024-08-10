#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalDummyScan : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::DUMMY_SCAN;

public:
	explicit GPUPhysicalDummyScan(vector<LogicalType> types, idx_t estimated_cardinality)
	    : GPUPhysicalOperator(PhysicalOperatorType::DUMMY_SCAN, std::move(types), estimated_cardinality) {
	}

public:
	// SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}
};
} // namespace duckdb