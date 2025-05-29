#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "../include/sirius_extension.hpp"
#include "log/logging.hpp"

namespace duckdb {

//Define the CUDA kernel
__global__ void kernelFunction() {
    // Noop
}

// Define the host function that launches the CUDA kernel
void myKernel() {
    SIRIUS_LOG_DEBUG("My kernel");
    kernelFunction<<<1, 1>>>();
    cudaDeviceSynchronize();
}

} // namespace duckdb