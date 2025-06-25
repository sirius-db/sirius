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

#include "operator/gpu_physical_projection.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"

namespace duckdb {

GPUPhysicalProjection::GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
      select_list(std::move(select_list)) {
}

OperatorResultType
GPUPhysicalProjection::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
    SIRIUS_LOG_DEBUG("Executing projection");
    auto start = std::chrono::high_resolution_clock::now();

    // The new executor...
    sirius::GpuExpressionExecutor gpu_expression_executor(select_list);
    gpu_expression_executor.Execute(input_relation, output_relation);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_DEBUG("Projection time: {:.2f} ms", duration.count()/1000.0);
    return OperatorResultType::FINISHED;
}

} // namespace duckdb