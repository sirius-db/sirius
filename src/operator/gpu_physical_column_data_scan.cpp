#include "operator/gpu_physical_column_data_scan.hpp"

namespace duckdb {

GPUPhysicalColumnDataScan::GPUPhysicalColumnDataScan(vector<LogicalType> types, PhysicalOperatorType op_type,
                                               idx_t estimated_cardinality, optionally_owned_ptr<ColumnDataCollection> collection_p)
    : GPUPhysicalOperator(op_type, std::move(types), estimated_cardinality), collection(std::move(collection_p)) {
}

GPUPhysicalColumnDataScan::GPUPhysicalColumnDataScan(vector<LogicalType> types, PhysicalOperatorType op_type,
                                               idx_t estimated_cardinality, idx_t cte_index)
    : GPUPhysicalOperator(op_type, std::move(types), estimated_cardinality), collection(nullptr), cte_index(cte_index) {
}

} // namespace duckdb