#include "operator/gpu_physical_union.hpp"

#include "gpu_meta_pipeline.hpp"
#include "gpu_pipeline.hpp"

namespace duckdb {
GPUPhysicalUnion::GPUPhysicalUnion(vector<LogicalType> types, idx_t estimated_cardinality)
  : GPUPhysicalOperator(TYPE, std::move(types), estimated_cardinality)
{
}
void GPUPhysicalUnion::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline)
{
  op_state             = nullptr;
  auto& union_pipeline = meta_pipeline.CreateUnionPipeline(current, false);
  children[0]->BuildPipelines(current, meta_pipeline);
  children[1]->BuildPipelines(union_pipeline, meta_pipeline);
}
}  // namespace duckdb
