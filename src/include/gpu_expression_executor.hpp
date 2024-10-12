#pragma once

#include "gpu_columns.hpp"
#include "duckdb/planner/expression.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

// Declaration of the CUDA kernel
template <typename T> void binaryExpression(T *a, T *b, T *result, uint64_t N, int op_mode);
template <typename T> void comparisonConstantExpression(T *a, T b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);

class GPUExpressionExecutor {
public:
    GPUExpressionExecutor() {
        gpuBufferManager = &(GPUBufferManager::GetInstance());
    };
    GPUBufferManager* gpuBufferManager;
    void FilterRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int depth = 0);
    void ProjectionRecursiveExpression(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, Expression& expr, int output_idx, int depth = 0);
};

} // namespace duckdb