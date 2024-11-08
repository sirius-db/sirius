#pragma once

#include "gpu_columns.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

template <typename T>
GPUColumn* 
ResolveTypeMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager);

GPUColumn* 
HandleMaterializeExpression(GPUColumn* column, BoundReferenceExpression& bound_ref, GPUBufferManager* gpuBufferManager);

} // namespace duckdb