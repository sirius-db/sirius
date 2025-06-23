#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

//! PhysicalFilter represents a filter operator. It removes non-matching tuples
//! from the result. Note that it does not physically change the data, it only
//! adds a selection vector to the chunk.
class GPUPhysicalFilter : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::FILTER;

public:
	// GPUPhysicalFilter(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

    GPUPhysicalFilter(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

// 	//! The filter expression
	unique_ptr<Expression> expression;

  // GPUExpressionExecutor* gpu_expression_executor;

// public:
// 	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

// 	bool ParallelOperator() const override {
// 		return true;
// 	}

// 	string ParamsToString() const override;

	// OperatorResultType Execute(ExecutionContext &context, GPUIntermediateRelation &input, GPUIntermediateRelation &chunk,
	//                                    GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorResultType Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const override;
};
} // namespace duckdb