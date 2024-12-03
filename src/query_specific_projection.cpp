#include "gpu_expression_executor.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

template <typename T>
GPUColumn* ResolveTypeMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager) {
    // GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    size_t size;
    T* a;
    if (column->row_ids != nullptr) {
        T* temp = reinterpret_cast<T*> (column->data_wrapper.data);
        uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (column->row_ids);
        a = gpuBufferManager->customCudaMalloc<T>(column->row_id_count, 0, 0);
        materializeExpression<T>(temp, a, row_ids_input, column->row_id_count);
        size = column->row_id_count;
    } else {
        a = reinterpret_cast<T*> (column->data_wrapper.data);
        size = column->column_length;
    }
    GPUColumn* result = new GPUColumn(size, column->data_wrapper.type, reinterpret_cast<uint8_t*>(a));
    // printf("size %ld\n", size);
    return result;
}

GPUColumn* HandleMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager) {
    switch(column->data_wrapper.type) {
        case ColumnType::INT32:
            return ResolveTypeMaterializeExpression<int>(column, bound_ref, gpuBufferManager);
        case ColumnType::INT64:
            return ResolveTypeMaterializeExpression<uint64_t>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT32:
            return ResolveTypeMaterializeExpression<float>(column, bound_ref, gpuBufferManager);
        case ColumnType::FLOAT64:
            return ResolveTypeMaterializeExpression<double>(column, bound_ref, gpuBufferManager);
        default:
            throw NotImplementedException("query_specific_project HandleMaterializeExpression Unsupported column type");
    }
}

void HandlingSpecificProjection(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expression, int output_idx) {

        GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

        GPUColumn* result;
		if (expression.type == ExpressionType::CASE_EXPR) {
			BoundCaseExpression& expr = expression.Cast<BoundCaseExpression>();
			if (expr.case_checks[0].then_expr->type == ExpressionType::BOUND_REF) {
				//case when nation = 1 then volume else 0 
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundComparisonExpression>();
				auto nation = when_expr.left->Cast<BoundReferenceExpression>().index;

				auto &then_expr = expr.case_checks[0].then_expr->Cast<BoundReferenceExpression>();
				auto volume = then_expr.index;

                auto materialized_nation = HandleMaterializeExpression(input_relation.columns[nation], when_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                auto materialized_volume = HandleMaterializeExpression(input_relation.columns[volume], then_expr, gpuBufferManager);

                uint64_t* a = reinterpret_cast<uint64_t*> (materialized_nation->data_wrapper.data);
                uint64_t* b = reinterpret_cast<uint64_t*> (materialized_volume->data_wrapper.data);
                size_t size = materialized_nation->column_length;
                uint64_t* out = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
                // commonCaseExpression(a, b, out, size, op_mode);

                result = new GPUColumn(size, ColumnType::INT64, reinterpret_cast<uint8_t*>(out));

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

                auto materialized_type = HandleMaterializeExpression(input_relation.columns[p_type], compare_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                auto materialized_extendedprice = HandleMaterializeExpression(input_relation.columns[l_extendedprice], then_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
                auto materialized_discount = HandleMaterializeExpression(input_relation.columns[l_discount], substract_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);

                uint64_t* a = reinterpret_cast<uint64_t*> (materialized_type->data_wrapper.data);
                double* b = reinterpret_cast<double*> (materialized_extendedprice->data_wrapper.data);
                double* c = reinterpret_cast<double*> (materialized_discount->data_wrapper.data);
                size_t size = materialized_type->column_length;
                double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);
                // q14CaseExpression(a, b, c, p_type_value1, p_type_value2, out, size);

				result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));

			} else if (expr.case_checks[0].then_expr->expression_class == ExpressionClass::CONSTANT) {
				auto &when_expr = expr.case_checks[0].when_expr->Cast<BoundConjunctionExpression>();
				if (when_expr.type == ExpressionType::CONJUNCTION_AND) {
					// case when o_orderpriority <> 0 and o_orderpriority <> 1 then 1 else 0
					auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
					auto o_orderpriority = compare_expr.left->Cast<BoundReferenceExpression>().index;

                    auto materialized_orderpriority = HandleMaterializeExpression(input_relation.columns[o_orderpriority], compare_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    uint64_t* a = reinterpret_cast<uint64_t*> (materialized_orderpriority->data_wrapper.data);
                    size_t size = materialized_orderpriority->column_length;
                    uint64_t* out = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
                    // commonCaseExpression(a, a, out, size, op_mode);

					result = new GPUColumn(size, ColumnType::INT64, reinterpret_cast<uint8_t*>(out));

				} else if (when_expr.type == ExpressionType::CONJUNCTION_OR) {
					// case when o_orderpriority = 0 or o_orderpriority = 1 then 1 else 0
					auto &compare_expr = when_expr.children[0]->Cast<BoundComparisonExpression>();
					auto o_orderpriority = compare_expr.left->Cast<BoundReferenceExpression>().index;

                    auto materialized_orderpriority = HandleMaterializeExpression(input_relation.columns[o_orderpriority], compare_expr.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    uint64_t* a = reinterpret_cast<uint64_t*> (materialized_orderpriority->data_wrapper.data);
                    size_t size = materialized_orderpriority->column_length;
                    uint64_t* out = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);

					result = new GPUColumn(size, ColumnType::INT64, reinterpret_cast<uint8_t*>(out));

				}
			}
		}

		if (expression.expression_class == ExpressionClass::BOUND_FUNCTION) {
			auto &expr = expression.Cast<BoundFunctionExpression>();
			printf("Function name %s\n", expr.function.name.c_str());

			if (expr.children[0]->type == ExpressionType::OPERATOR_CAST) {
				//(CAST(O_ORDERDATE AS DOUBLE) / 10000.0)
				auto &cast_expr = expr.children[0]->Cast<BoundCastExpression>();
				auto date = cast_expr.child->Cast<BoundReferenceExpression>().index;

				auto materialized_date = HandleMaterializeExpression(input_relation.columns[date], cast_expr.child->Cast<BoundReferenceExpression>(), gpuBufferManager);
				uint64_t* a = reinterpret_cast<uint64_t*> (materialized_date->data_wrapper.data);
				size_t size = materialized_date->column_length;
				double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);
				// commonCaseExpression(a, a, out, size, op_mode);

				result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));

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

						auto materialize_4 = HandleMaterializeExpression(input_relation.columns[l_extendedprice_or_4], ref_expr, gpuBufferManager);
						auto materialize_tax = HandleMaterializeExpression(input_relation.columns[l_tax], function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						double* a = reinterpret_cast<double*> (materialize_4->data_wrapper.data);
						double* b = reinterpret_cast<double*> (materialize_tax->data_wrapper.data);
						size_t size = materialize_4->column_length;
						double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);

						result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));


					}  else if (expr.function.name.compare("-") == 0) {
						//l_extendedprice * (1 - l_discount)
						auto &function_expr = expr.children[1]->Cast<BoundFunctionExpression>();
						auto l_discount = function_expr.children[1]->Cast<BoundReferenceExpression>().index;

						auto materialize_extendedprice = HandleMaterializeExpression(input_relation.columns[l_extendedprice_or_4], ref_expr, gpuBufferManager);
						auto materialize_discount = HandleMaterializeExpression(input_relation.columns[l_discount], function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						double* a = reinterpret_cast<double*> (materialize_extendedprice->data_wrapper.data);
						double* b = reinterpret_cast<double*> (materialize_discount->data_wrapper.data);
						size_t size = materialize_extendedprice->column_length;
						double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);

						result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));
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

						auto materialize_extendedprice = HandleMaterializeExpression(input_relation.columns[l_extendedprice], left_function_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						auto materialize_discount = HandleMaterializeExpression(input_relation.columns[l_discount], left_right_function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						auto materialize_supplycost = HandleMaterializeExpression(input_relation.columns[ps_supplycost], right_function_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
						auto materialize_quantity = HandleMaterializeExpression(input_relation.columns[l_quantity], right_function_expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);

						double* a = reinterpret_cast<double*> (materialize_extendedprice->data_wrapper.data);
						double* b = reinterpret_cast<double*> (materialize_discount->data_wrapper.data);
						double* c = reinterpret_cast<double*> (materialize_supplycost->data_wrapper.data);
						double* d = reinterpret_cast<double*> (materialize_quantity->data_wrapper.data);

						size_t size = materialize_extendedprice->column_length;
						double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);

						result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));
				}

			} else if (expr.children[1]->type == ExpressionType::BOUND_REF) {
				if (expr.children[0]->type == ExpressionType::BOUND_FUNCTION) {
					//((sum(CASE  WHEN (((P_TYPE >= CAST(125 AS BIGINT)) AND (P_TYPE < CAST(150 AS BIGINT)))) THEN ((L_EXTENDEDPRICE * (CAST(1 AS DOUBLE) - L_DISCOUNT))) ELSE CAST(0.0 AS DOUBLE) END) * 100.0) / sum((L_EXTENDEDPRICE * (CAST(1 AS DOUBLE) - L_DISCOUNT))))
					auto &left_function_expr = expr.children[0]->Cast<BoundFunctionExpression>();
					auto sum_case = left_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
					auto value = left_function_expr.children[1]->Cast<BoundConstantExpression>().value.GetValue<double>();
					auto sum = expr.children[1]->Cast<BoundReferenceExpression>().index;

					auto materialize_a = HandleMaterializeExpression(input_relation.columns[sum_case], left_function_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);
					auto materialize_b = HandleMaterializeExpression(input_relation.columns[sum], expr.children[1]->Cast<BoundReferenceExpression>(), gpuBufferManager);

					double* a = reinterpret_cast<double*> (materialize_a->data_wrapper.data);
					double* b = reinterpret_cast<double*> (materialize_b->data_wrapper.data);

					size_t size = materialize_a->column_length;
					double* out = gpuBufferManager->customCudaMalloc<double>(size, 0, 0);

					result = new GPUColumn(size, ColumnType::FLOAT64, reinterpret_cast<uint8_t*>(out));
					
				} else if (expr.children[0]->type == ExpressionType::BOUND_REF) {
					//l_extendedprice * l_discount
					// auto left = expr.children[0]->Cast<BoundReferenceExpression>().index;
					// auto right = expr.children[1]->Cast<BoundReferenceExpression>().index;
				}
			} else if (expr.children[1]->expression_class == ExpressionClass::BOUND_CONSTANT) {
				// sum(l_extendedprice) / 7.0
				// auto left = expr.children[0]->Cast<BoundReferenceExpression>().index;
				// auto value = expr.children[1]->Cast<BoundConstantExpression>().value.GetValue<double>();
			}
		}


        if (result) {
            output_relation.columns[output_idx] = result;
            output_relation.columns[output_idx]->row_ids = nullptr;
            output_relation.columns[output_idx]->row_id_count = 0;
        } else {
            output_relation.columns[output_idx] = new GPUColumn(1, ColumnType::INT32, fake_data);
        }

}

} // namespace duckdb