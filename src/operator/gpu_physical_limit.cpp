#include "operator/gpu_physical_limit.hpp"

namespace duckdb {

GPUPhysicalStreamingLimit::GPUPhysicalStreamingLimit(vector<LogicalType> types, BoundLimitNode limit_val_p,
                                               BoundLimitNode offset_val_p, idx_t estimated_cardinality, bool parallel)
    : GPUPhysicalOperator(PhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
      limit_val(std::move(limit_val_p)), offset_val(std::move(offset_val_p)), parallel(parallel) {
}

} // namespace duckdb
