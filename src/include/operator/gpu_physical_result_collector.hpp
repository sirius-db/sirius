#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalResultCollector : public GPUPhysicalOperator {
public:
    GPUPhysicalResultCollector(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

};
} // namespace duckdb