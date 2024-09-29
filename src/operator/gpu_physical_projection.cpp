#include "operator/gpu_physical_projection.hpp"

namespace duckdb {

GPUPhysicalProjection::GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
      select_list(std::move(select_list)) {

    gpu_expression_executor = new GPUExpressionExecutor();
}

// OperatorResultType
// GPUPhysicalProjection::Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, 
// 	GPUIntermediateRelation &output_relation, GlobalOperatorState &gstate, OperatorState &state) const {
OperatorResultType
GPUPhysicalProjection::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
    printf("Executing projection\n");
    for (int expr = 0; expr < select_list.size(); expr++) {
        printf("Executing expression: %s\n", select_list[expr]->ToString().c_str());
        gpu_expression_executor->ProjectionRecursiveExpression(input_relation, output_relation, *select_list[expr], expr, 0);
    }
    return OperatorResultType::FINISHED;
}

} // namespace duckdb