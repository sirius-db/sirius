#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalGroupedAggregate : public GPUPhysicalOperator {
public:
    GPUPhysicalGroupedAggregate(PhysicalOperator op);

};
} // namespace duckdb