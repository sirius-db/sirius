#include "operator/gpu_physical_filter.hpp"
#include "operator/gpu_physical_string_matching.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>

namespace duckdb {

GPUPhysicalFilter::GPUPhysicalFilter(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                               idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality) {

	D_ASSERT(select_list.size() > 0);
	if (select_list.size() > 1) {
		// create a big AND out of the expressions
		auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
		for (auto &expr : select_list) {
			conjunction->children.push_back(std::move(expr));
		}
		expression = std::move(conjunction);
	} else {
		expression = std::move(select_list[0]);
	}

	GPUExpressionExecutor* gpu_expression_executor = new GPUExpressionExecutor();

}

// OperatorResultType 
// GPUPhysicalFilter::Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
// 	                                   GlobalOperatorState &gstate, OperatorState &state) const {
OperatorResultType 
GPUPhysicalFilter::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
	std::cout << "Executing GPUPhysicalFilter for expression " << expression->ToString() << std::endl;
    gpu_expression_executor->FilterRecursiveExpression(input_relation, output_relation, *expression, 0);
	return OperatorResultType::FINISHED;
}



} // namespace duckdb