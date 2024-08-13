#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalUngroupedAggregate : public GPUPhysicalOperator {
public:
    GPUPhysicalUngroupedAggregate(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

    vector<unique_ptr<Expression>> aggregates;
    
};
} // namespace duckdb