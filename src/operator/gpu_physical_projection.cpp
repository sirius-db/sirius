#include "operator/gpu_physical_projection.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

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

    for (int idx = 0; idx < select_list.size(); idx++) {
        printf("Executing expression: %s\n", select_list[idx]->ToString().c_str());



		if (select_list[idx]->type == ExpressionType::CASE_EXPR) {
			BoundCaseExpression& expr = select_list[idx]->Cast<BoundCaseExpression>();
			if (expr.case_checks[0].then_expr->type == ExpressionType::BOUND_REF) {
				//case when nation = 1 then volume else 0 
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundComparisonExpression>();
				auto nation = when_expr.left->Cast<BoundReferenceExpression>().index;

				auto &then_expr = expr.case_checks[0].then_expr->Cast<BoundReferenceExpression>();
				auto volume = then_expr.index;

                commonCaseExpression(nation, volume, result, N, 0);
                size_t size = input_relation.columns[nation]->column_length;
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


			} else if (expr.case_checks[0].then_expr->type == ExpressionType::BOUND_FUNCTION) {
				//case when (p_type >= 125 and p_type < 150) then l_extendedprice * (1 - l_discount) else 0.0
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundConjunctionExpression>();
				auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
				auto p_type = compare_expr.left->Cast<BoundReferenceExpression>().index;

				auto &then_expr = expr.case_checks[0].then_expr->Cast<BoundFunctionExpression>();
				auto l_extendedprice = then_expr.children[0]->Cast<BoundReferenceExpression>().index;
				auto &substract_expr = then_expr.children[1]->Cast<BoundFunctionExpression>();
				auto l_discount = substract_expr.children[1]->Cast<BoundReferenceExpression>().index;

                uint64_t p_type_value1 = 125;
                uint64_t p_type_value2 = 150;

			} else if (expr.case_checks[0].then_expr->expression_class == ExpressionClass::CONSTANT) {
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundConjunctionExpression>();
				if (when_expr.type == ExpressionType::CONJUNCTION_AND) {
					// case when o_orderpriority <> 0 and o_orderpriority <> 1 then 1 else 0
					auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
					auto o_orderpriority = compare_expr.left->Cast<BoundReferenceExpression>().index;
                    
				} else if (when_expr.type == ExpressionType::CONJUNCTION_OR) {
					// case when o_orderpriority = 0 or o_orderpriority = 1 then 1 else 0
					auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
					auto o_orderpriority = compare_expr.left->Cast<BoundReferenceExpression>().index;
				}
			}
		}

		if (select_list[idx]->expression_class == ExpressionClass::BOUND_FUNCTION) {
			auto &expr = select_list[idx]->Cast<BoundFunctionExpression>();
			printf("Function name %s\n", expr.function.name.c_str());

			if (expr.children[0]->type == ExpressionType::OPERATOR_CAST) {
				//(CAST(O_ORDERDATE AS DOUBLE) / 10000.0)
				auto &cast_expr = expr.children[0]->Cast<BoundCastExpression>();
				auto date = cast_expr.child->Cast<BoundReferenceExpression>().index;
			} else if (expr.children[1]->type == ExpressionType::BOUND_FUNCTION) {
				
				if (expr.children[0]->type == ExpressionType::BOUND_REF) {
					//#4 * (1 + l_tax)
					//l_extendedprice * (1 - l_discount)
					auto &ref_expr = expr.children[0]->Cast<BoundReferenceExpression>();
					auto l_extendedprice_or_4 = ref_expr.index;

					if (expr.function.name.compare("+") == 0) {
						//#4 * (1 + l_tax)
						auto &function_expr = expr.children[1]->Cast<BoundFunctionExpression>();
						auto l_tax = function_expr.children[1]->Cast<BoundReferenceExpression>().index;
					}  else if (expr.function.name.compare("-") == 0) {
						//l_extendedprice * (1 - l_discount)
						auto &function_expr = expr.children[1]->Cast<BoundFunctionExpression>();
						auto l_discount = function_expr.children[1]->Cast<BoundReferenceExpression>().index;
					}

				} else if (expr.children[0]->type == ExpressionType::BOUND_FUNCTION) {
                        //((L_EXTENDEDPRICE * (1.0 - L_DISCOUNT)) - (PS_SUPPLYCOST * L_QUANTITY))
						auto &left_function_expr = expr.children[0]->Cast<BoundFunctionExpression>();
						auto l_extendedprice = left_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
						auto &left_right_function_expr = left_function_expr.children[1]->Cast<BoundFunctionExpression>();
						auto l_discount = left_right_function_expr.children[1]->Cast<BoundReferenceExpression>().index;

						auto &right_function_expr = expr.children[1]->Cast<BoundFunctionExpression>();
						auto ps_supplycost = right_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
						auto l_quantity = right_function_expr.children[1]->Cast<BoundReferenceExpression>().index;
				}

			} else if (expr.children[1]->type == ExpressionType::BOUND_REF) {
				if (expr.children[0]->type == ExpressionType::BOUND_FUNCTION) {
					//((sum(CASE  WHEN (((P_TYPE >= CAST(125 AS BIGINT)) AND (P_TYPE < CAST(150 AS BIGINT)))) THEN ((L_EXTENDEDPRICE * (CAST(1 AS DOUBLE) - L_DISCOUNT))) ELSE CAST(0.0 AS DOUBLE) END) * 100.0) / sum((L_EXTENDEDPRICE * (CAST(1 AS DOUBLE) - L_DISCOUNT))))
					auto &left_function_expr = expr.children[0]->Cast<BoundFunctionExpression>();
					auto sum_case = left_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
					auto value = left_function_expr.children[1]->Cast<BoundConstantExpression>().value.GetValue<double>();
					auto sum = expr.children[1]->Cast<BoundReferenceExpression>().index;
				} else if (expr.children[0]->type == ExpressionType::BOUND_REF) {
					//l_extendedprice * l_discount
					auto left = expr.children[0]->Cast<BoundReferenceExpression>().index;
					auto right = expr.children[1]->Cast<BoundReferenceExpression>().index;
				}
			} else if (expr.children[1]->expression_class == ExpressionClass::BOUND_CONSTANT) {
				// sum(l_extendedprice) / 7.0
				auto left = expr.children[0]->Cast<BoundReferenceExpression>().index;
				auto value = expr.children[1]->Cast<BoundConstantExpression>().value.GetValue<double>();
			}
		}


        gpu_expression_executor->ProjectionRecursiveExpression(input_relation, output_relation, *select_list[idx], idx, 0);
    }
    return OperatorResultType::FINISHED;
}

} // namespace duckdb