#pragma once

#include "gpu_columns.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

class GPUExpressionExecutor {
public:
    GPUExpressionExecutor() = default;
    void FilterRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int depth = 0);
    void ProjectionRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int output_idx, int depth = 0);
};

} // namespace duckdb