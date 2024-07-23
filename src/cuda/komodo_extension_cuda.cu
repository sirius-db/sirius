#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "../include/komodo_extension.hpp"

namespace duckdb {

//Define the CUDA kernel
__global__ void kernelFunction() {
    printf("Hello from CUDA kernel!\n");
}

// Define the host function that launches the CUDA kernel
void myKernel() {
    printf("My kernel\n");
    kernelFunction<<<1, 1>>>();
    cudaDeviceSynchronize();
}

} // namespace duckdb