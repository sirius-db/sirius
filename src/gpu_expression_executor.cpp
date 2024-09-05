#include "gpu_expression_executor.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

void 
GPUExpressionExecutor::FilterRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int depth) {
    for (int i = 1; i < depth; i++) {
        printf("  ");
    }
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN: {
        auto &bound_between = expr.Cast<BoundBetweenExpression>();
        printf("Executing between expression\n");
        FilterRecursiveExpression(input_relation, output_relation, *(bound_between.input), depth + 1);
        FilterRecursiveExpression(input_relation, output_relation, *(bound_between.lower), depth + 1);
        FilterRecursiveExpression(input_relation, output_relation, *(bound_between.upper), depth + 1);
		break;
    } case ExpressionClass::BOUND_REF: {
        auto &bound_ref = expr.Cast<BoundReferenceExpression>();
        printf("Reading column index %ld\n", bound_ref.index);
		input_relation.checkLateMaterialization(bound_ref.index);
        output_relation.columns[bound_ref.index]->row_ids = new uint64_t[1];
        output_relation.columns[bound_ref.index] = input_relation.columns[bound_ref.index];
        printf("Overwrite row ids with column %ld\n", bound_ref.index);
		break;
	} case ExpressionClass::BOUND_CASE: {
        auto &bound_case = expr.Cast<BoundCaseExpression>();
        printf("Executing case expression\n");
        for (idx_t i = 0; i < bound_case.case_checks.size(); i++) {
            FilterRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].when_expr), depth + 1);
            FilterRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].then_expr), depth + 1);
        }
        FilterRecursiveExpression(input_relation, output_relation, *(bound_case.else_expr), depth + 1);
		break;
	} case ExpressionClass::BOUND_CAST: {
        auto &bound_cast = expr.Cast<BoundCastExpression>();
        printf("Executing cast expression\n");
        FilterRecursiveExpression(input_relation, output_relation, *(bound_cast.child), depth + 1);
		break;
	} case ExpressionClass::BOUND_COMPARISON: {
        auto &bound_comparison = expr.Cast<BoundComparisonExpression>();
        printf("Executing comparison expression\n");
        FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.left), depth + 1);
        FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.right), depth + 1);
		break;
	} case ExpressionClass::BOUND_CONJUNCTION: {
        auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
		printf("Executing conjunction expression\n");
        for (auto &child : bound_conjunction.children) {
            FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_CONSTANT: {
        printf("Reading value %s\n", expr.Cast<BoundConstantExpression>().value.ToString().c_str());
		break;
	} case ExpressionClass::BOUND_FUNCTION: {
        auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
        for (auto &child : bound_conjunction.children) {
            FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_OPERATOR: {
		throw NotImplementedException("Operator expression is not supported");
		break;
	} case ExpressionClass::BOUND_PARAMETER: {
		throw NotImplementedException("Parameter expression is not supported");
		break;
	} default: {
		throw InternalException("Attempting to execute expression of unknown type!");
	}
    }
}


void 
GPUExpressionExecutor::ProjectionRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int output_idx, int depth) {
    for (int i = 1; i < depth; i++) {
        printf("  ");
    }
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN: {
        auto &bound_between = expr.Cast<BoundBetweenExpression>();
        printf("Executing between expression\n");
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.input), output_idx, depth + 1);
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.lower), output_idx, depth + 1);
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_between.upper), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_REF: {
        auto &bound_ref = expr.Cast<BoundReferenceExpression>();
        printf("Reading column index %ld\n", bound_ref.index);
		input_relation.checkLateMaterialization(bound_ref.index);
		break;
	} case ExpressionClass::BOUND_CASE: {
        auto &bound_case = expr.Cast<BoundCaseExpression>();
        printf("Executing case expression\n");
        for (idx_t i = 0; i < bound_case.case_checks.size(); i++) {
            ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].when_expr), output_idx, depth + 1);
            ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.case_checks[i].then_expr), output_idx, depth + 1);
        }
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_case.else_expr), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_CAST: {
        auto &bound_cast = expr.Cast<BoundCastExpression>();
        printf("Executing cast expression\n");
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_cast.child), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_COMPARISON: {
        auto &bound_comparison = expr.Cast<BoundComparisonExpression>();
        printf("Executing comparison expression\n");
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_comparison.left), output_idx, depth + 1);
        ProjectionRecursiveExpression(input_relation, output_relation, *(bound_comparison.right), output_idx, depth + 1);
		break;
	} case ExpressionClass::BOUND_CONJUNCTION: {
        auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
		printf("Executing conjunction expression\n");
        for (auto &child : bound_conjunction.children) {
            ProjectionRecursiveExpression(input_relation, output_relation, *child, output_idx, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_CONSTANT: {
        printf("Reading value %s\n", expr.Cast<BoundConstantExpression>().value.ToString().c_str());
		break;
	} case ExpressionClass::BOUND_FUNCTION: {
        auto &bound_conjunction = expr.Cast<BoundConjunctionExpression>();
        for (auto &child : bound_conjunction.children) {
            ProjectionRecursiveExpression(input_relation, output_relation, *child, output_idx, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_OPERATOR: {
		throw NotImplementedException("Operator expression is not supported");
		break;
	} case ExpressionClass::BOUND_PARAMETER: {
		throw NotImplementedException("Parameter expression is not supported");
		break;
	} default: {
		throw InternalException("Attempting to execute expression of unknown type!");
	}
    }
    printf("Writing projection result to idx %ld\n", output_idx);
    if (depth == 0) {
        if (expr.expression_class == ExpressionClass::BOUND_REF) {
            output_relation.columns[output_idx] = input_relation.columns[expr.Cast<BoundReferenceExpression>().index];
        } else {
            uint8_t* fake_data = new uint8_t[1];
            output_relation.columns[output_idx] = new GPUColumn(1, ColumnType::INT32, fake_data);
        }
    }
}


} // namespace duckdb