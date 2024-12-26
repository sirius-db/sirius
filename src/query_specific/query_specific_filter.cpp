#include "gpu_expression_executor.hpp"
#include "operator/gpu_materialize.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"

namespace duckdb {

bool 
GPUExpressionExecutor::HandlingSpecificFilter(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expression) {

        GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

        uint64_t* count = nullptr;
        uint64_t* comparison_idx = nullptr;

        if (expression.type == ExpressionType::CONJUNCTION_OR) {
            auto &expr = expression.Cast<BoundConjunctionExpression>();

            //Q19 HACK!!!
            if (expr.children.size() == 3 && expr.children[2]->type == ExpressionType::CONJUNCTION_AND) {
                auto &expr2 = expr.children[2]->Cast<BoundConjunctionExpression>();
                if (expr2.children.size() == 5 && expr2.children[0]->type == ExpressionType::COMPARE_EQUAL && expr2.children[1]->type == ExpressionType::COMPARE_LESSTHANOREQUALTO && \
                            expr2.children[2]->type == ExpressionType::COMPARE_GREATERTHANOREQUALTO && expr2.children[3]->type == ExpressionType::COMPARE_LESSTHANOREQUALTO && expr2.children[4]->type == ExpressionType::COMPARE_IN) {
                    auto &expr3 = expr2.children[4]->Cast<BoundOperatorExpression>();
                    if (expr3.children.size() == 5) {
                            printf("Filter expression of Q19\n");
                            // string t = "(((P_BRAND = 12) AND (L_QUANTITY <= 11) AND (P_SIZE <= 5) AND (P_CONTAINER IN (0, 1, 4, 5))) OR ((P_BRAND = 23) AND (L_QUANTITY >= 10) AND (L_QUANTITY <= 20) AND (P_SIZE <= 10) AND (P_CONTAINER IN (17, 18, 20, 21))) OR ((P_BRAND = 34) AND (L_QUANTITY >= 20) AND (L_QUANTITY <= 30) AND (P_SIZE <= 15) AND (P_CONTAINER IN (8, 9, 12, 13))))";
                            string t = "(((P_BRAND = 12) AND (P_SIZE <= 5) AND (L_QUANTITY <= 11.0) AND (P_CONTAINER IN (0, 1, 4, 5))) OR ((P_BRAND = 23) AND (P_SIZE <= 10) AND (L_QUANTITY >= 10.0) AND (L_QUANTITY <= 20.0) AND (P_CONTAINER IN (17, 18, 20, 21))) OR ((P_BRAND = 34) AND (P_SIZE <= 15) AND (L_QUANTITY >= 20.0) AND (L_QUANTITY <= 30.0) AND (P_CONTAINER IN (8, 9, 12, 13))))";
                            if (!expression.ToString().compare(t)) {

                                    BoundComparisonExpression& first = expr2.children[0]->Cast<BoundComparisonExpression>();
                                    auto p_brand = first.left->Cast<BoundReferenceExpression>().index;
                                    BoundComparisonExpression& second = expr2.children[1]->Cast<BoundComparisonExpression>();
                                    auto p_size = second.left->Cast<BoundReferenceExpression>().index;
                                    BoundComparisonExpression& third = expr2.children[2]->Cast<BoundComparisonExpression>();
                                    auto l_quantity = third.left->Cast<BoundReferenceExpression>().index;
                                    BoundOperatorExpression& fourth = expr2.children[4]->Cast<BoundOperatorExpression>();
                                    auto p_container = fourth.children[0]->Cast<BoundReferenceExpression>().index;

                                    uint64_t p_brand_val[3] = {12, 23, 34};
                                    double l_quantity_val[6] = {1, 11, 10, 20, 20, 30};
                                    uint64_t p_size_val[3] = {5, 10, 15};
                                    uint64_t p_container_val[12] = {0, 1, 4, 5, 17, 18, 20, 21, 8, 9, 12, 13};

                                    GPUColumn* materialized_brand = HandleMaterializeExpression(input_relation.columns[p_brand], first.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                                    GPUColumn* materialized_size = HandleMaterializeExpression(input_relation.columns[p_size], second.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                                    GPUColumn* materialized_quantity = HandleMaterializeExpression(input_relation.columns[l_quantity], third.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                                    GPUColumn* materialized_container = HandleMaterializeExpression(input_relation.columns[p_container], fourth.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);

                                    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                                    uint64_t* a = reinterpret_cast<uint64_t*> (materialized_brand->data_wrapper.data);
                                    double* b = reinterpret_cast<double*> (materialized_quantity->data_wrapper.data);
                                    uint64_t* c = reinterpret_cast<uint64_t*> (materialized_size->data_wrapper.data);
                                    uint64_t* d = reinterpret_cast<uint64_t*> (materialized_container->data_wrapper.data);

                                    size_t size = materialized_brand->column_length;

                                    q19FilterExpression(a, b, c, d, p_brand_val, l_quantity_val, p_size_val, p_container_val, comparison_idx, count, size);

                            }

                    }
                }
            }

            //Q7 HACK!!!
            if (expr.children.size() == 2 && expr.children[0]->type == ExpressionType::CONJUNCTION_AND && expr.children[1]->type == ExpressionType::CONJUNCTION_AND) {
                auto &expr2 = expr.children[0]->Cast<BoundConjunctionExpression>();
                if (expr2.children.size() == 2 && expr2.children[0]->type == ExpressionType::COMPARE_EQUAL && expr2.children[1]->type == ExpressionType::COMPARE_EQUAL) {
                    printf("Filter expression of Q7\n");
                    string t = "(((N_NATIONKEY = 6) AND (N_NATIONKEY = 7)) OR ((N_NATIONKEY = 7) AND (N_NATIONKEY = 6)))";
                    if (!expression.ToString().compare(t)) {
                        
                            BoundComparisonExpression& first = expr2.children[0]->Cast<BoundComparisonExpression>();
                            auto n_nationkey1 = first.left->Cast<BoundReferenceExpression>().index;
                            BoundComparisonExpression& second = expr2.children[1]->Cast<BoundComparisonExpression>();
                            auto n_nationkey2 = second.left->Cast<BoundReferenceExpression>().index;

                            uint64_t val[4] = {6, 7, 7, 6};

                            GPUColumn* materialized_nkey1 = HandleMaterializeExpression(input_relation.columns[n_nationkey1], first.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                            GPUColumn* materialized_nkey2 = HandleMaterializeExpression(input_relation.columns[n_nationkey2], second.left->Cast<BoundReferenceExpression>(), gpuBufferManager);

                            count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                            uint64_t* a = reinterpret_cast<uint64_t*> (materialized_nkey1->data_wrapper.data);
                            uint64_t* b = reinterpret_cast<uint64_t*> (materialized_nkey2->data_wrapper.data);

                            size_t size = materialized_nkey1->column_length;
                            q7FilterExpression(a, b, val[0], val[1], val[2], val[3], comparison_idx, count, size);

                    }

                }
            }
        } else if (expression.type == ExpressionType::CONJUNCTION_AND) {
            auto &expr = expression.Cast<BoundConjunctionExpression>();
            //Q16 HACK!!!
            if (expr.children.size() == 3 && expr.children[0]->type == ExpressionType::COMPARE_NOTEQUAL && expr.children[1]->type == ExpressionType::CONJUNCTION_OR && expr.children[2]->type == ExpressionType::COMPARE_IN) {
                printf("Filter expression of Q16\n");
                string t = "((P_BRAND != 45) AND ((P_TYPE < 65) OR (P_TYPE >= 70)) AND (P_SIZE IN (49, 14, 23, 45, 19, 3, 36, 9)))";
                if (!expression.ToString().compare(t)) {
                    BoundComparisonExpression& first = expr.children[0]->Cast<BoundComparisonExpression>();
                    auto p_brand = first.left->Cast<BoundReferenceExpression>().index;
                    BoundConjunctionExpression& second = expr.children[1]->Cast<BoundConjunctionExpression>();
                    BoundComparisonExpression& second_temp = second.children[0]->Cast<BoundComparisonExpression>();
                    auto p_type = second_temp.left->Cast<BoundReferenceExpression>().index;
                    BoundOperatorExpression& third = expr.children[2]->Cast<BoundOperatorExpression>();
                    auto p_size = third.children[0]->Cast<BoundReferenceExpression>().index;

                    uint64_t p_brand_val = 45;
                    uint64_t p_type_val[2] = {65, 70};
                    uint64_t p_size_val[8] = {49, 14, 23, 45, 19, 3, 36, 9};

                    GPUColumn* materialized_brand = HandleMaterializeExpression(input_relation.columns[p_brand], first.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    GPUColumn* materialized_type = HandleMaterializeExpression(input_relation.columns[p_type], second_temp.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    GPUColumn* materialized_size = HandleMaterializeExpression(input_relation.columns[p_size], third.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);

                    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                    uint64_t* a = reinterpret_cast<uint64_t*> (materialized_brand->data_wrapper.data);
                    uint64_t* b = reinterpret_cast<uint64_t*> (materialized_type->data_wrapper.data);
                    uint64_t* c = reinterpret_cast<uint64_t*> (materialized_size->data_wrapper.data);

                    size_t size = materialized_brand->column_length;
                    q16FilterExpression(a, b, c, p_brand_val, p_type_val[0], p_type_val[1], p_size_val, comparison_idx, count, size);
                }

            }


            //Q12 HACK!!!
            if (expr.children.size() == 3 && expr.children[0]->type == ExpressionType::COMPARE_LESSTHAN && expr.children[1]->type == ExpressionType::COMPARE_LESSTHAN && expr.children[2]->type == ExpressionType::COMPARE_IN) {
                printf("Filter expression of Q12\n");
                string t = "((L_COMMITDATE < L_RECEIPTDATE) AND (L_SHIPDATE < L_COMMITDATE) AND (L_SHIPMODE IN (4, 6)))";
                if (!expression.ToString().compare(t)) {
                    BoundConjunctionExpression& expr = expression.Cast<BoundConjunctionExpression>();

                    BoundComparisonExpression& first = expr.children[0]->Cast<BoundComparisonExpression>();
                    auto l_commitdate = first.left->Cast<BoundReferenceExpression>().index;
                    auto l_receiptdate = first.right->Cast<BoundReferenceExpression>().index;
                    BoundComparisonExpression& second = expr.children[1]->Cast<BoundComparisonExpression>();
                    auto l_shipdate = second.left->Cast<BoundReferenceExpression>().index;
                    // auto l_commitdate = second.right->Cast<BoundReferenceExpression>().index;
                    BoundOperatorExpression& third = expr.children[2]->Cast<BoundOperatorExpression>();
                    auto l_shipmode = third.children[0]->Cast<BoundReferenceExpression>().index;

                    uint64_t l_shipmode_val[2] = {4, 6};

                    GPUColumn* materialized_commitdate = HandleMaterializeExpression(input_relation.columns[l_commitdate], first.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    GPUColumn* materialized_receiptdate = HandleMaterializeExpression(input_relation.columns[l_receiptdate], first.right->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    GPUColumn* materialized_shipdate = HandleMaterializeExpression(input_relation.columns[l_shipdate], second.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
                    GPUColumn* materialized_shipmode = HandleMaterializeExpression(input_relation.columns[l_shipmode], third.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);

                    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                    uint64_t* a = reinterpret_cast<uint64_t*> (materialized_commitdate->data_wrapper.data);
                    uint64_t* b = reinterpret_cast<uint64_t*> (materialized_receiptdate->data_wrapper.data);
                    uint64_t* c = reinterpret_cast<uint64_t*> (materialized_shipdate->data_wrapper.data);
                    uint64_t* d = reinterpret_cast<uint64_t*> (materialized_shipmode->data_wrapper.data);

                    size_t size = materialized_commitdate->column_length;

                    q12FilterExpression(a, b, c, d, l_shipmode_val[0], l_shipmode_val[1], comparison_idx, count, size);
                }

            }

        }

        //Q2 HACK!!!
        if (expression.type == ExpressionType::COMPARE_EQUAL) {
            printf("Filter expression of Q2\n");
            string t = "(((P_TYPE + 3) % 5) = 0)";
            if (!expression.ToString().compare(t)) {
                auto &expr = expression.Cast<BoundComparisonExpression>();	
                if (expr.left->type == ExpressionType::BOUND_FUNCTION) {
                    auto& function_expr = expr.left->Cast<BoundFunctionExpression>();
                    if (function_expr.function.name.compare("%") == 0) {
                        auto& left_function_expr = function_expr.children[0]->Cast<BoundFunctionExpression>();
                        if (left_function_expr.function.name.compare("+") == 0) {
                                auto p_type = left_function_expr.children[0]->Cast<BoundReferenceExpression>().index;
                                uint64_t p_type_val = 0;

                                GPUColumn* materialized_type = HandleMaterializeExpression(input_relation.columns[p_type], left_function_expr.children[0]->Cast<BoundReferenceExpression>(), gpuBufferManager);

                                count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                                uint64_t* a = reinterpret_cast<uint64_t*> (materialized_type->data_wrapper.data);

                                size_t size = materialized_type->column_length;

                                q2FilterExpression(a, p_type_val, comparison_idx, count, size);

                        }
                    }
                }
            }
        }
       
        if (expression.type == ExpressionType::COMPARE_IN) {
             BoundOperatorExpression& in_expr = expression.Cast<BoundOperatorExpression>();
             if (in_expr.children.size() == 8 || in_expr.children.size() == 4) {
                string t = "(substr(C_PHONE, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17'))";
                if (!expression.ToString().compare(t)) {
                    auto &bound_function = in_expr.children[0]->Cast<BoundFunctionExpression>();
                    auto &bound_ref = bound_function.children[0]->Cast<BoundReferenceExpression>();
                    GPUColumn* input_column = input_relation.columns[bound_ref.index];
                    uint64_t start_idx = 0;
                    uint64_t length = 2;
                    GPUColumn* materialized_column = HandleMaterializeExpression(input_column, bound_ref, gpuBufferManager);
                    size_t size = materialized_column->column_length;
                    uint8_t* a = materialized_column->data_wrapper.data;
                    uint64_t* offset = materialized_column->data_wrapper.offset;
                    string c_phone_val_str = "13312329301817";
                    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                    q22FilterExpression(a, offset, start_idx, length, c_phone_val_str, comparison_idx, count, size, 8);
                }
                t = "(substr(C_PHONE, 1, 2) IN ('13', '31', '23'))";
                if (!expression.ToString().compare(t)) {
                    auto &bound_function = in_expr.children[0]->Cast<BoundFunctionExpression>();
                    auto &bound_ref = bound_function.children[0]->Cast<BoundReferenceExpression>();
                    GPUColumn* input_column = input_relation.columns[bound_ref.index];
                    uint64_t start_idx = 0;
                    uint64_t length = 2;
                    GPUColumn* materialized_column = HandleMaterializeExpression(input_column, bound_ref, gpuBufferManager);
                    size_t size = materialized_column->column_length;
                    uint8_t* a = materialized_column->data_wrapper.data;
                    uint64_t* offset = materialized_column->data_wrapper.offset;
                    string c_phone_val_str = "133123";
                    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                    q22FilterExpression(a, offset, start_idx, length, c_phone_val_str, comparison_idx, count, size, 8);
                }
             } else {
                throw NotImplementedException("IN expression not supported");
             }
        }

        if (count && comparison_idx) {
            if (count[0] == 0) throw NotImplementedException("No match found");
            HandleMaterializeRowIDs(input_relation, output_relation, count[0], comparison_idx, gpuBufferManager);
            // for (int i = 0; i < input_relation.columns.size(); i++) {
            //     output_relation.columns[i] = input_relation.columns[i];
            //     if (input_relation.columns[i]->row_ids == nullptr) {
            //         output_relation.columns[i]->row_ids = comparison_idx;
            //     } else {
            //         uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[i]->row_ids);
            //         uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
            //         materializeExpression<uint64_t>(row_ids_input, new_row_ids, comparison_idx, count[0]);
            //         output_relation.columns[i]->row_ids = new_row_ids;
            //     }
            //     output_relation.columns[i]->row_id_count = count[0];
            // }
            return true;
        }

        return false;

}

} // namespace duckdb