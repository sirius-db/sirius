#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalHashJoin : public GPUPhysicalOperator {
public:
    GPUPhysicalHashJoin(PhysicalOperator op);

};
} // namespace duckdb