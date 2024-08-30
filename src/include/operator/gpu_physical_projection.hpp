#pragma once

#include "gpu_physical_operator.hpp"
// #include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "gpu_expression_executor.hpp"

namespace duckdb {

class GPUPhysicalProjection : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::PROJECTION;

public:
	GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
	                   idx_t estimated_cardinality);

	vector<unique_ptr<Expression>> select_list;

	OperatorResultType Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
										GlobalOperatorState &gstate, OperatorState &state) const override;

	GPUExpressionExecutor* gpu_expression_executor;

public:
	// unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	// OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	//                            GlobalOperatorState &gstate, OperatorState &state) const override;

	// bool ParallelOperator() const override {
	// 	return true;
	// }

	// string ParamsToString() const override;

	// static unique_ptr<PhysicalOperator>
	// CreateJoinProjection(vector<LogicalType> proj_types, const vector<LogicalType> &lhs_types,
	//                      const vector<LogicalType> &rhs_types, const vector<idx_t> &left_projection_map,
	//                      const vector<idx_t> &right_projection_map, const idx_t estimated_cardinality);


};
} // namespace duckdb