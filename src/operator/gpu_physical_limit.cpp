#include "operator/gpu_physical_limit.hpp"
#include "operator/gpu_materialize.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

GPUPhysicalStreamingLimit::GPUPhysicalStreamingLimit(vector<LogicalType> types, BoundLimitNode limit_val_p,
                                               BoundLimitNode offset_val_p, idx_t estimated_cardinality, bool parallel)
    : GPUPhysicalOperator(PhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
      limit_val(std::move(limit_val_p)), offset_val(std::move(offset_val_p)), parallel(parallel) {
}

OperatorResultType 
GPUPhysicalStreamingLimit::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {

	printf("Executing streaming limit\n");
  printf("Limit value %ld\n", limit_val.GetConstantValue());
  auto limit_const = limit_val.GetConstantValue();
  auto offset_const = offset_val.GetConstantValue();
  if (offset_const > 0) {
    throw NotImplementedException("Streaming limit with offset not implemented");
  }
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	for (int col_idx = 0; col_idx < output_relation.columns.size(); col_idx++) {
    BoundReferenceExpression& bound_ref = *new BoundReferenceExpression(LogicalType::INTEGER, col_idx);
    GPUColumn* materialize_column = HandleMaterializeExpression(input_relation.columns[col_idx], bound_ref, gpuBufferManager);

    limit_const = min(limit_const, materialize_column->column_length);
    output_relation.columns[col_idx] = new GPUColumn(limit_const, materialize_column->data_wrapper.type, materialize_column->data_wrapper.data,
                          materialize_column->data_wrapper.offset, materialize_column->data_wrapper.num_bytes, materialize_column->data_wrapper.is_string_data);
    output_relation.columns[col_idx]->is_unique = materialize_column->is_unique;
    if (limit_const > 0 && output_relation.columns[col_idx]->data_wrapper.type == ColumnType::VARCHAR) {
      Allocator& allocator = Allocator::DefaultAllocator();
			uint64_t* new_num_bytes = reinterpret_cast<uint64_t*>(allocator.AllocateData(sizeof(uint64_t)));
			callCudaMemcpyDeviceToHost<uint64_t>(new_num_bytes, materialize_column->data_wrapper.offset + limit_const, 1, 0);
      output_relation.columns[col_idx]->data_wrapper.num_bytes = new_num_bytes[0];
    }
    printf("Column %d has %ld rows\n", col_idx, output_relation.columns[col_idx]->column_length);
	}

  return OperatorResultType::FINISHED;
}
} // namespace duckdb
