#include "operator/gpu_physical_delim_join.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"

namespace duckdb {

GPUPhysicalDelimJoin::GPUPhysicalDelimJoin(PhysicalOperatorType type, vector<LogicalType> types,
                                     unique_ptr<GPUPhysicalOperator> original_join,
                                     vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality)
    : GPUPhysicalOperator(type, std::move(types), estimated_cardinality), join(std::move(original_join)),
      delim_scans(std::move(delim_scans)) {
	D_ASSERT(type == PhysicalOperatorType::LEFT_DELIM_JOIN || type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
}

GPUPhysicalRightDelimJoin::GPUPhysicalRightDelimJoin(vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
                                               vector<const_reference<GPUPhysicalOperator>> delim_scans,
                                               idx_t estimated_cardinality)
    : GPUPhysicalDelimJoin(PhysicalOperatorType::RIGHT_DELIM_JOIN, std::move(types), std::move(original_join),
                        std::move(delim_scans), estimated_cardinality) {

}

GPUPhysicalLeftDelimJoin::GPUPhysicalLeftDelimJoin(vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
                                             vector<const_reference<GPUPhysicalOperator>> delim_scans,
                                             idx_t estimated_cardinality)
    : GPUPhysicalDelimJoin(PhysicalOperatorType::LEFT_DELIM_JOIN, std::move(types), std::move(original_join),
                        std::move(delim_scans), estimated_cardinality) {
}

}
