#include "operator/gpu_physical_projection.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"

namespace duckdb {

GPUPhysicalProjection::GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
      select_list(std::move(select_list)) {

    //gpu_expression_executor = new GPUExpressionExecutor();
}

OperatorResultType
GPUPhysicalProjection::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
    printf("Executing projection\n");
    auto start = std::chrono::high_resolution_clock::now();

    // The old executor...
    // GPUExpressionExecutor old_gpu_expression_executor();
    // for (int idx = 0; idx < select_list.size(); idx++) {
    //     printf("Executing expression: %s\n", select_list[idx]->ToString().c_str());
    //     old_gpu_expression_executor->ProjectionRecursiveExpression(input_relation,
    //     output_relation, *select_list[idx], idx, 0);
    // }

    // The new executor...
    sirius::GpuExpressionExecutor gpu_expression_executor(select_list);
    gpu_expression_executor.Execute(input_relation, output_relation);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Projection time: %.2f ms\n", duration.count()/1000.0);
    return OperatorResultType::FINISHED;
}

} // namespace duckdb