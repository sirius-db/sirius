#pragma once
#include "gpu_physical_operator.hpp"
namespace duckdb {
class GPUPhysicalUnion : public GPUPhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::UNION;
  GPUPhysicalUnion(vector<LogicalType> types, idx_t estimated_cardinality);
  void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;
};
}
