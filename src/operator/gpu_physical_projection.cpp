#include "operator/gpu_physical_projection.hpp"

namespace duckdb {

// GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
// 	                   idx_t estimated_cardinality); {
// };

GPUPhysicalProjection::GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
      select_list(std::move(select_list)) {
}

} // namespace duckdb