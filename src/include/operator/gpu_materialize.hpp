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

// GPUColumn* 
// HandleMaterializeRowIDs(GPUColumn* in_column, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager);

// void
// HandleMaterializeRowIDs(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager);

// void
// HandleMaterializeRowIDsRHS(GPUIntermediateRelation& hash_table_result, GPUIntermediateRelation& output_relation, vector<idx_t> rhs_output_columns, size_t offset, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager);

void
HandleMaterializeRowIDs(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique);

void
HandleMaterializeRowIDsRHS(GPUIntermediateRelation& hash_table_result, GPUIntermediateRelation& output_relation, vector<idx_t> rhs_output_columns, size_t offset, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique);

} // namespace duckdb