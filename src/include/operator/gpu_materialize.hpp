#pragma once

#include "gpu_columns.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

void materializeStringColumnToDuckdbFormat(shared_ptr<GPUColumn> column, char* column_char_write_buffer, string_t* column_string_write_buffer);

template <typename T>
shared_ptr<GPUColumn> 
ResolveTypeMaterializeExpression(shared_ptr<GPUColumn> column, GPUBufferManager* gpuBufferManager);

shared_ptr<GPUColumn> 
HandleMaterializeExpression(shared_ptr<GPUColumn> column, GPUBufferManager* gpuBufferManager);

void
HandleMaterializeRowIDs(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique);

void
HandleMaterializeRowIDsRHS(GPUIntermediateRelation& hash_table_result, GPUIntermediateRelation& output_relation, vector<idx_t> rhs_output_columns, size_t offset, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique);

void
HandleMaterializeRowIDsLHS(GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation, vector<idx_t> lhs_output_columns, uint64_t count, uint64_t* row_ids, GPUBufferManager* gpuBufferManager, bool maintain_unique);
} // namespace duckdb