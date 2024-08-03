#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalResultCollector : public GPUPhysicalOperator {
public:
    GPUPhysicalResultCollector(PhysicalOperator op);

};
} // namespace duckdb