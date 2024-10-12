#include "gpu_expression_executor.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"

namespace duckdb {

void 
GPUExpressionExecutor::FilterRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int depth) {
    printf("Expression class %d\n", expr.expression_class);
    uint64_t* comparison_idx = nullptr;
    uint64_t* count = nullptr;

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
        printf("Executing bound reference expression\n");
        printf("Reading column index %ld\n", bound_ref.index);
		input_relation.checkLateMaterialization(bound_ref.index);
        // printf("output_relation.columns.size() %ld\n", output_relation.columns.size());
        // printf("input_relation.columns.size() %ld\n", input_relation.columns.size());
        // printf("bound_ref.index %ld\n", bound_ref.index);
        output_relation.columns[bound_ref.index] = input_relation.columns[bound_ref.index];
        output_relation.columns[bound_ref.index]->row_ids = new uint64_t[1];
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
        // FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.left), depth + 1);
        // FilterRecursiveExpression(input_relation, output_relation, *(bound_comparison.right), depth + 1);

        auto &bound_ref1 = bound_comparison.left->Cast<BoundReferenceExpression>();
        auto &bound_ref2 = bound_comparison.right->Cast<BoundConstantExpression>();
        size_t size;

        uint64_t* a;
        if (input_relation.checkLateMaterialization(bound_ref1.index)) {
            uint64_t* temp = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref1.index]->data_wrapper.data);
            uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref1.index]->row_ids);
            a = gpuBufferManager->customCudaMalloc<uint64_t>(input_relation.columns[bound_ref1.index]->row_id_count, 0, 0);
            materializeExpression<uint64_t>(temp, a, row_ids_input, input_relation.columns[bound_ref1.index]->row_id_count);
            size = input_relation.columns[bound_ref1.index]->row_id_count;
        } else {
            a = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref1.index]->data_wrapper.data);
            size = input_relation.columns[bound_ref1.index]->column_length;
        }

        count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
        uint64_t b = bound_ref2.value.GetValue<uint64_t>();
        comparisonConstantExpression<uint64_t>(a, b, comparison_idx, count, (uint64_t) size, 0);
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
        printf("Executing function expression\n");
        auto &bound_function = expr.Cast<BoundFunctionExpression>();
        for (auto &child : bound_function.children) {
            FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_OPERATOR: {
        printf("Executing IN expression\n");
        auto &bound_operator = expr.Cast<BoundOperatorExpression>();
        for (auto &child : bound_operator.children) {
            FilterRecursiveExpression(input_relation, output_relation, *child, depth + 1);
        }
		break;
	} case ExpressionClass::BOUND_PARAMETER: {
		throw NotImplementedException("Parameter expression is not supported");
		break;
	} default: {
		throw InternalException("Attempting to execute expression of unknown type!");
	}
    }

    if (depth == 0) {
        printf("Writing filter result\n");
        for (int i = 0; i < input_relation.columns.size(); i++) {
            output_relation.columns[i] = input_relation.columns[i];
            // output_relation.columns[i]->row_ids = new uint64_t[1];
            if (comparison_idx) {
                if (input_relation.columns[i]->row_ids == nullptr) {
                    output_relation.columns[i]->row_ids = comparison_idx;
                } else {
                    uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[i]->row_ids);
                    uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
                    materializeExpression<uint64_t>(row_ids_input, new_row_ids, comparison_idx, count[0]);
                    output_relation.columns[i]->row_ids = new_row_ids;
                }
            }
            if (count) output_relation.columns[i]->row_id_count = count[0];
        }
    }
}


void 
GPUExpressionExecutor::ProjectionRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int output_idx, int depth) {
    printf("Expression class %d\n", expr.expression_class);
    GPUColumn* result;

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
        printf("Executing bound reference expression\n");
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
        printf("Executing function expression\n");
        auto &bound_function = expr.Cast<BoundFunctionExpression>();
        // for (auto &child : bound_function.children) {
        //     ProjectionRecursiveExpression(input_relation, output_relation, *child, output_idx, depth + 1);
        // }

        auto &bound_ref1 = bound_function.children[0]->Cast<BoundReferenceExpression>();
        auto &bound_ref2 = bound_function.children[1]->Cast<BoundReferenceExpression>();
        size_t size = input_relation.columns[bound_ref1.index]->column_length;
        double* ptr_double = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);

        double* a, *b;
        if (input_relation.checkLateMaterialization(bound_ref1.index)) {
            double* temp = reinterpret_cast<double*> (input_relation.columns[bound_ref1.index]->data_wrapper.data);
            uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref1.index]->row_ids);
            a = gpuBufferManager->customCudaMalloc<double>(input_relation.columns[bound_ref1.index]->row_id_count, 0, 0);
            materializeExpression<double>(temp, a, row_ids_input, input_relation.columns[bound_ref1.index]->row_id_count);
        } else {
            a = reinterpret_cast<double*> (input_relation.columns[bound_ref1.index]->data_wrapper.data);
        }

        if (input_relation.checkLateMaterialization(bound_ref2.index)) {
            double* temp = reinterpret_cast<double*> (input_relation.columns[bound_ref2.index]->data_wrapper.data);
            uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[bound_ref2.index]->row_ids);
            b = gpuBufferManager->customCudaMalloc<double>(input_relation.columns[bound_ref2.index]->row_id_count, 0, 0);
            materializeExpression<double>(temp, b, row_ids_input, input_relation.columns[bound_ref2.index]->row_id_count);
            size = input_relation.columns[bound_ref2.index]->row_id_count;
        } else {
            b = reinterpret_cast<double*> (input_relation.columns[bound_ref2.index]->data_wrapper.data);
            size = input_relation.columns[bound_ref2.index]->column_length;
        }
        binaryExpression<double>(a, b, ptr_double, (uint64_t) size, 0);
        result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(ptr_double));
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
            if (result) {
                output_relation.columns[output_idx] = result;
                // output_relation.length = result->column_length;
                output_relation.columns[output_idx]->row_ids = nullptr;
                output_relation.columns[output_idx]->row_id_count = 0;
                // uint8_t* host_data = new uint8_t[output_relation.length * 8];
                // callCudaMemcpyDeviceToHost<uint8_t>(host_data, output_relation.columns[output_idx]->data_wrapper.data, output_relation.length * 8, 0);
            } else {
                output_relation.columns[output_idx] = new GPUColumn(1, ColumnType::INT32, fake_data);
            }
        }
    }
}


} // namespace duckdb