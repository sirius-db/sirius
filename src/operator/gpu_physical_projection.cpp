#include "operator/gpu_physical_projection.hpp"

namespace duckdb {

GPUPhysicalProjection::GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
      select_list(std::move(select_list)) {

    gpu_expression_executor = new GPUExpressionExecutor();
}

OperatorResultType
GPUPhysicalProjection::Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, 
	GPUIntermediateRelation &output_relation, GlobalOperatorState &gstate, OperatorState &state) const {
    printf("Executing projection\n");
    for (int i = 0; i < select_list.size(); i++) {
        printf("Executing expression ");
        select_list[i]->Print();
        gpu_expression_executor->ProjectionRecursiveExpression(input_relation, output_relation, *select_list[i], i, 0);
    }
}

} // namespace duckdb