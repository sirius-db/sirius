#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalUngroupedAggregate : public GPUPhysicalOperator {
public:
    GPUPhysicalUngroupedAggregate(PhysicalOperator op);
    
};
} // namespace duckdb