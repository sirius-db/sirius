#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalOrder : public GPUPhysicalOperator {
public:
    GPUPhysicalOrder(PhysicalOperator op);

};
} // namespace duckdb