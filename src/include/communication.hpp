#pragma once

#include "duckdb.hpp"

namespace duckdb {

// Declaration of the CUDA kernel
extern int* sendDataToGPU(int* data, int size);

} // namespace duckdb