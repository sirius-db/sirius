/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "gpu_columns.hpp"
#include "duckdb/planner/expression.hpp"
#include "gpu_buffer_manager.hpp"
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

// Declaration of the CUDA kernel
// void doubleRoundExpression(double *a, double *result, int decimal_places, uint64_t N);
// void floatRoundExpression(float *a, float *result, int decimal_places, uint64_t N);
// template <typename T> void binaryExpression(T *a, T *b, T *result, uint64_t N, int op_mode);
// template <typename T> void binaryConstantExpression(T *a, T b, T *result, uint64_t N, int op_mode);
template <typename T> void comparisonConstantExpression(T *a, T b, T c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template <typename T> void comparisonExpression(T *a, T* b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
void comparisonStringBetweenExpression(char* char_data, uint64_t num_chars, uint64_t* str_indices, uint64_t num_strings, std::string lower_string, std::string upper_string, 
    bool is_lower_inclusive, bool is_upper_inclusive, uint64_t* &row_id, uint64_t* &count);
void comparisonStringExpression(char* char_data, uint64_t num_chars, uint64_t* str_indices, uint64_t num_strings, std::string comparison_string, int op_mode, uint64_t* &row_id, uint64_t* &count);

//query specific projection
// void commonArithmeticExpression(double *a, double *b, double* c, double* d, double *result, uint64_t N, int op_mode);
// void extractYear(uint64_t *date, uint64_t *year, uint64_t N);
// void commonCaseExpression(uint64_t *a, uint64_t *b, double *result, uint64_t N, int op_mode);
// void q14CaseExpression(uint64_t *p_type, double *l_extendedprice, double *l_discount, uint64_t p_type_val1, uint64_t p_type_val2, double *result, uint64_t N);
// void q8CaseExpression(uint64_t *nation, double *volume, uint64_t nation_val, double else_val, double *result, uint64_t N);

//query specific filter
// void q7FilterExpression(uint64_t *n1_nationkey, uint64_t *n2_nationkey, uint64_t val1, uint64_t val2, uint64_t val3, uint64_t val4, 
//                                 uint64_t* &row_ids, uint64_t* &count, uint64_t N);
// void q7FilterExpression2(uint64_t *n1_nationkey, uint64_t *n2_nationkey, uint64_t val1, uint64_t val2, 
//                                 uint64_t* &row_ids, uint64_t* &count, uint64_t N);
// void q2FilterExpression(uint64_t *p_type, uint64_t p_type_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N);
// void q12FilterExpression(uint64_t *l_commitdate, uint64_t *l_receiptdate, uint64_t *l_shipdate, uint64_t *l_shipmode, uint64_t l_shipmode_val1, uint64_t l_shipmode_val2, uint64_t* &row_ids, uint64_t* &count, uint64_t N);
// // void q16FilterExpression(uint64_t *p_brand, uint64_t *p_type, uint64_t *p_size, uint64_t p_brand_val, uint64_t p_type_val1, uint64_t p_type_val2, uint64_t *p_size_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N);
// void q16FilterExpression(uint64_t *p_type, uint64_t *p_size, uint64_t p_type_val1, uint64_t p_type_val2, uint64_t *p_size_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N);
// void q19FilterExpression(uint64_t *p_brand, double *l_quantity, uint64_t *p_size, uint64_t* p_container, uint64_t *p_brand_val, double *l_quantity_val, uint64_t *p_size_val, uint64_t* p_container_val, 
//             uint64_t* &row_ids, uint64_t* &count, uint64_t N);
// void q22FilterExpression(uint8_t *a, uint64_t* offset, uint64_t start_idx, uint64_t length, string c_phone_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int num_predicates);

class GPUExpressionExecutor {
public:
    vector<uint64_t> projected_columns;

    void FilterRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int depth = 0);
    void ProjectionRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int output_idx, int depth = 0);
    
    template <typename T> void ResolveTypeComparisonConstantExpression (shared_ptr<GPUColumn> column, BoundConstantExpression& expr1, BoundConstantExpression& expr2, uint64_t* &count, uint64_t* & row_ids, ExpressionType expression_type);
    void HandleComparisonConstantExpression(shared_ptr<GPUColumn> column, BoundConstantExpression& expr1, BoundConstantExpression& expr2, uint64_t* &count, uint64_t* &row_ids, ExpressionType expression_type);
    
    template <typename T> void ResolveTypeComparisonExpression (shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, uint64_t* &count, uint64_t* & row_ids, ExpressionType expression_type);
    void HandleComparisonExpression(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, uint64_t* &count, uint64_t* &row_ids, ExpressionType expression_type);

    template <typename T> shared_ptr<GPUColumn> ResolveTypeBinaryConstantExpression (shared_ptr<GPUColumn> column, T constant, GPUBufferManager* gpuBufferManager, string function_name);
    shared_ptr<GPUColumn> HandleBinaryConstantExpression(shared_ptr<GPUColumn> column, BoundConstantExpression& expr, GPUBufferManager* gpuBufferManager, string function_name);

    template <typename T> shared_ptr<GPUColumn> ResolveTypeBinaryExpression (shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager, string function_name);
    shared_ptr<GPUColumn> HandleBinaryExpression(shared_ptr<GPUColumn> column1, shared_ptr<GPUColumn> column2, GPUBufferManager* gpuBufferManager, string function_name);

    bool HandlingSpecificProjection(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expression, int output_idx);
    bool HandlingSpecificFilter(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expression);

    shared_ptr<GPUColumn> HandleRoundExpression(shared_ptr<GPUColumn> column, int decimal_places);

};

} // namespace duckdb