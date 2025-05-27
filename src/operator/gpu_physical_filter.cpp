#include "operator/gpu_physical_filter.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"

namespace duckdb {

GPUPhysicalFilter::GPUPhysicalFilter(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                               idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality) {

	D_ASSERT(select_list.size() > 0);
	if (select_list.size() > 1) {
    // KEVIN: I don't think this code path is ever entered
		// create a big AND out of the expressions
		auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
		for (auto &expr : select_list) {
			conjunction->children.push_back(std::move(expr));
		}
		expression = std::move(conjunction);
	} else {
		expression = std::move(select_list[0]);
	}
}

OperatorResultType 
GPUPhysicalFilter::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
	printf("Executing expression %s\n", expression->ToString().c_str());
  auto start = std::chrono::high_resolution_clock::now();

  // The old executor...
  // GPUExpressionExecutor old_gpu_expression_executor();
  // old_gpu_expression_executor->FilterRecursiveExpression(input_relation, output_relation, *expression, 0);

  // The new executor...
  sirius::GpuExpressionExecutor gpu_expression_executor(*expression.get());
  gpu_expression_executor.Select(input_relation, output_relation);
  
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Filter time: %.2f ms\n", duration.count()/1000.0);
	return OperatorResultType::FINISHED;
}

} // namespace duckdb