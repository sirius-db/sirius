#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalProjection : public GPUPhysicalOperator {
public:
    GPUPhysicalProjection(PhysicalOperator op);

};
} // namespace duckdb