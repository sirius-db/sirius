#include "operator/gpu_physical_limit.hpp"

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
	for (int col_idx = 0; col_idx < output_relation.columns.size(); col_idx++) {
		// output_relation.columns[col_idx] = input_relation.columns[col_idx];
    output_relation.columns[col_idx] = new GPUColumn(limit_const, input_relation.columns[col_idx]->data_wrapper.type, input_relation.columns[col_idx]->data_wrapper.data);
	}

  return OperatorResultType::FINISHED;
}
} // namespace duckdb
