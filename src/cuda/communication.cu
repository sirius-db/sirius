#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "../include/communication.hpp"

namespace duckdb {

// Define the host function that launches the CUDA kernel
int* sendDataToGPU(int* data, int size) {
    printf("Send data to GPU\n");
    // use cudamemcpy
    int** target = new int*[1];
    cudaMalloc((void**) &target[0], size * sizeof(int));
    cudaMemcpy(target, data, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Done sending data to GPU\n");
    return target[0];
}

} // namespace duckdb