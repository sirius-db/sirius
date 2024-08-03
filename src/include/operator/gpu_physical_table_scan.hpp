#pragma once

#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalTableScan : public GPUPhysicalOperator {
public:
    GPUPhysicalTableScan(PhysicalOperator op);

};
} // namespace duckdb