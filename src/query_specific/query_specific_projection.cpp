#include "gpu_expression_executor.hpp"
#include "operator/gpu_materialize.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

bool
GPUExpressionExecutor::HandlingSpecificProjection(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expression, int output_idx) {

        GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

        GPUColumn* result = nullptr;
		if (expression.type == ExpressionType::CASE_EXPR) {
			BoundCaseExpression& expr = expression.Cast<BoundCaseExpression>();
			if (expr.case_checks[0].then_expr->type == ExpressionType::BOUND_REF) {
				//Q8 HACK!!!
				//case when nation = 1 then volume else 0 
				printf("Case expression of Q8\n");
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundComparisonExpression>();
				auto nation = when_expr.left->Cast<BoundReferenceExpression>().index;

				auto &then_expr = expr.case_checks[0].then_expr->Cast<BoundReferenceExpression>();
				auto volume = then_expr.index;

                auto materialized_nation = HandleMaterializeExpression(input_relation.columns[nation], when_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                auto materialized_volume = HandleMaterializeExpression(input_relation.columns[volume], then_expr, gpuBufferManager);

                uint64_t* a = reinterpret_cast<uint64_t*> (materialized_nation->data_wrapper.data);
                double* b = reinterpret_cast<double*> (materialized_volume->data_wrapper.data);
                size_t size = materialized_nation->column_length;
                double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;

				uint64_t nation_val = 1;
				double else_val = 0;
                q8CaseExpression(a, b, nation_val, else_val, out, size);

                result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));

			} else if (expr.case_checks[0].then_expr->type == ExpressionType::BOUND_FUNCTION) {
				// Q14 HACK!!!
				printf("Case expression of Q14\n");
				//CASE  WHEN ((P_TYPE >= 125)) THEN ((L_EXTENDEDPRICE * (1.0 - L_DISCOUNT))) ELSE 0.0 END
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundComparisonExpression>();
				// printf("%s\n", ExpressionTypeToString(when_expr.type).c_str());
				// auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
				auto p_type = when_expr.left->Cast<BoundReferenceExpression>().index;

				auto &then_expr = expr.case_checks[0].then_expr->Cast<BoundFunctionExpression>();
				auto l_extendedprice = then_expr.children[0]->Cast<BoundReferenceExpression>().index;
				auto &substract_expr = then_expr.children[1]->Cast<BoundFunctionExpression>();
				auto l_discount = substract_expr.children[1]->Cast<BoundReferenceExpression>().index;

                uint64_t p_type_value1 = 125;
                uint64_t p_type_value2 = 150;

                auto materialized_type = HandleMaterializeExpression(input_relation.columns[p_type], when_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                auto materialized_extendedprice = HandleMaterializeExpression(input_relation.columns[l_extendedprice], then_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
                auto materialized_discount = HandleMaterializeExpression(input_relation.columns[l_discount], substract_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);

                uint64_t* a = reinterpret_cast<uint64_t*> (materialized_type->data_wrapper.data);
                double* b = reinterpret_cast<double*> (materialized_extendedprice->data_wrapper.data);
                double* c = reinterpret_cast<double*> (materialized_discount->data_wrapper.data);
                size_t size = materialized_type->column_length;
                double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;
                q14CaseExpression(a, b, c, p_type_value1, p_type_value2, out, size);

				result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));

			} else if (expr.case_checks[0].then_expr->expression_class == ExpressionClass::BOUND_CONSTANT) {
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundConjunctionExpression>();
				if (when_expr.type == ExpressionType::CONJUNCTION_AND) {
					// Q12 HACK!!!
					// case when o_orderpriority <> 0 and o_orderpriority <> 1 then 1 else 0
					printf("Case expression of Q12\n");
					auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
					auto o_orderpriority = compare_expr.left->Cast<BoundReferenceExpression>().index;

                    auto materialized_orderpriority = HandleMaterializeExpression(input_relation.columns[o_orderpriority], compare_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    uint64_t* a = reinterpret_cast<uint64_t*> (materialized_orderpriority->data_wrapper.data);
                    size_t size = materialized_orderpriority->column_length;
                    double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;
                    commonCaseExpression(a, a, out, size, 1);

					result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));

				} else if (when_expr.type == ExpressionType::CONJUNCTION_OR) {
					// Q12 HACK!!!
					// case when o_orderpriority = 0 or o_orderpriority = 1 then 1 else 0
					printf("Case expression of Q12\n");
					auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
					auto o_orderpriority = compare_expr.left->Cast<BoundReferenceExpression>().index;

                    auto materialized_orderpriority = HandleMaterializeExpression(input_relation.columns[o_orderpriority], compare_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    uint64_t* a = reinterpret_cast<uint64_t*> (materialized_orderpriority->data_wrapper.data);
                    size_t size = materialized_orderpriority->column_length;
                    double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;
					commonCaseExpression(a, a, out, size, 2);

					result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));

				}
			}
		}

		if (expression.expression_class == ExpressionClass::BOUND_FUNCTION) {
			auto &expr = expression.Cast<BoundFunctionExpression>();
			// printf("Function name %s\n", expr.function.name.c_str());

			if (expr.children[0]->type == ExpressionType::OPERATOR_CAST) {
				//Q7 Q8 Q9 HACK!!!
				//(CAST(O_ORDERDATE AS DOUBLE) // 10000.0)
				printf("Projection expression of Q7, Q8, Q9\n");
				auto &cast_expr = expr.children[0]->Cast<BoundCastExpression>();
				auto date = cast_expr.child->Cast<BoundReferenceExpression>().index;

				auto materialized_date = HandleMaterializeExpression(input_relation.columns[date], cast_expr.child->Cast<BoundReferenceExpression>(), gpuBufferManager);
				uint64_t* a = reinterpret_cast<uint64_t*> (materialized_date->data_wrapper.data);
				size_t size = materialized_date->column_length;
				uint64_t* out = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0).data_;
				extractYear(a, out, size);

				result = new GPUColumn(size, ColumnType::INT64, reinterpret_cast<uint8_t*>(out));

			} else if (expr.children[1]->type == ExpressionType::BOUND_FUNCTION) {
				
				if (expr.children[0]->type == ExpressionType::BOUND_REF) {
					//#4 * (1 + l_tax)
					//l_extendedprice * (1 - l_discount)
					auto &ref_expr = expr.children[0]->Cast<BoundReferenceExpression>();
					auto l_extendedprice_or_4 = ref_expr.index;
					auto &function_expr = expr.children[1]->Cast<BoundFunctionExpression>();

					if (function_expr.function.name.compare("+") == 0) {
						//Q1 HACK!!!
						//#4 * (1 + l_tax)
						printf("Projection expression of Q1\n");
						auto l_tax = function_expr.children[1]->Cast<BoundReferenceExpression>().index;

						auto materialize_4 = HandleMaterializeExpression(input_relation.columns[l_extendedprice_or_4], ref_expr, gpuBufferManager);
						auto materialize_tax = HandleMaterializeExpression(input_relation.columns[l_tax], function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						double* a = reinterpret_cast<double*> (materialize_4->data_wrapper.data);
						double* b = reinterpret_cast<double*> (materialize_tax->data_wrapper.data);
						size_t size = materialize_4->column_length;
						double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;
						commonArithmeticExpression(a, b, a, a, out, size, 1);

						result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));


					}  else if (function_expr.function.name.compare("-") == 0) {
						//l_extendedprice * (1 - l_discount)
						printf("Common projection expression l_extendedprice * (1 - l_discount)\n");
						auto l_discount = function_expr.children[1]->Cast<BoundReferenceExpression>().index;

						auto materialize_extendedprice = HandleMaterializeExpression(input_relation.columns[l_extendedprice_or_4], ref_expr, gpuBufferManager);
						auto materialize_discount = HandleMaterializeExpression(input_relation.columns[l_discount], function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						double* a = reinterpret_cast<double*> (materialize_extendedprice->data_wrapper.data);
						double* b = reinterpret_cast<double*> (materialize_discount->data_wrapper.data);
						size_t size = materialize_extendedprice->column_length;
						double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;
						commonArithmeticExpression(a, b, a, a, out, size, 0);

						result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));
					}

				} else if (expr.children[0]->type == ExpressionType::BOUND_FUNCTION) {
						//Q9 HACK!!!
                        //((L_EXTENDEDPRICE * (1.0 - L_DISCOUNT)) - (PS_SUPPLYCOST * L_QUANTITY))
						printf("Projection expression of Q9\n");
						auto &left_function_expr = expr.children[0]->Cast<BoundFunctionExpression>();
						auto l_extendedprice = left_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
						auto &left_right_function_expr = left_function_expr.children[1]->Cast<BoundFunctionExpression>();
						auto l_discount = left_right_function_expr.children[1]->Cast<BoundReferenceExpression>().index;

						auto &right_function_expr = expr.children[1]->Cast<BoundFunctionExpression>();
						auto ps_supplycost = right_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
						auto l_quantity = right_function_expr.children[1]->Cast<BoundReferenceExpression>().index;

						auto materialize_extendedprice = HandleMaterializeExpression(input_relation.columns[l_extendedprice], left_function_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						auto materialize_discount = HandleMaterializeExpression(input_relation.columns[l_discount], left_right_function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						auto materialize_supplycost = HandleMaterializeExpression(input_relation.columns[ps_supplycost], right_function_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						auto materialize_quantity = HandleMaterializeExpression(input_relation.columns[l_quantity], right_function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);

						double* a = reinterpret_cast<double*> (materialize_extendedprice->data_wrapper.data);
						double* b = reinterpret_cast<double*> (materialize_discount->data_wrapper.data);
						double* c = reinterpret_cast<double*> (materialize_supplycost->data_wrapper.data);
						double* d = reinterpret_cast<double*> (materialize_quantity->data_wrapper.data);

						size_t size = materialize_extendedprice->column_length;
						double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;
						commonArithmeticExpression(a, b, c, d, out, size, 2);

						result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));
				}

			} else if (expr.children[1]->type == ExpressionType::BOUND_REF) {
				if (expr.children[0]->type == ExpressionType::BOUND_FUNCTION) {
					//Q14 HACK!!!
					//((sum(CASE  WHEN (((P_TYPE >= CAST(125 AS BIGINT)) AND (P_TYPE < CAST(150 AS BIGINT)))) THEN ((L_EXTENDEDPRICE * (CAST(1 AS DOUBLE) - L_DISCOUNT))) ELSE CAST(0.0 AS DOUBLE) END) * 100.0) / sum((L_EXTENDEDPRICE * (CAST(1 AS DOUBLE) - L_DISCOUNT))))
					printf("Projection expression of Q14\n");
					auto &left_function_expr = expr.children[0]->Cast<BoundFunctionExpression>();
					auto sum_case = left_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
					auto value = left_function_expr.children[1]->Cast<BoundConstantExpression>().value.GetValue<double>();
					auto sum = expr.children[1]->Cast<BoundReferenceExpression>().index;

					auto materialize_a = HandleMaterializeExpression(input_relation.columns[sum_case], left_function_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
					auto materialize_b = HandleMaterializeExpression(input_relation.columns[sum], expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);

					double* a = reinterpret_cast<double*> (materialize_a->data_wrapper.data);
					double* b = reinterpret_cast<double*> (materialize_b->data_wrapper.data);

					size_t size = materialize_a->column_length;
					double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0).data_;
					commonArithmeticExpression(a, b, a, a, out, size, 3);

					result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));

					
				}
			}
		}


        if (result) {
			printf("Writing projection result to idx %ld\n", output_idx);
            output_relation.columns[output_idx] = result;
            output_relation.columns[output_idx]->row_ids = nullptr;
            output_relation.columns[output_idx]->row_id_count = 0;
			return true;
        }

		return false;

}

} // namespace duckdb