#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "communication.hpp"
#include "gpu_buffer_manager.hpp"
#include "operator/cuda_helper.cuh"

namespace duckdb {

template void
callCudaMemcpyHostToDevice<int>(int* dest, int* src, size_t size, int gpu);

template void
callCudaMemcpyHostToDevice<uint64_t>(uint64_t* dest, uint64_t* src, size_t size, int gpu);

template void
callCudaMemcpyHostToDevice<float>(float* dest, float* src, size_t size, int gpu);

template void
callCudaMemcpyHostToDevice<double>(double* dest, double* src, size_t size, int gpu);

template void
callCudaMemcpyHostToDevice<uint8_t>(uint8_t* dest, uint8_t* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToHost<int>(int* dest, int* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToHost<uint64_t>(uint64_t* dest, uint64_t* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToHost<float>(float* dest, float* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToHost<double>(double* dest, double* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToHost<uint8_t>(uint8_t* dest, uint8_t* src, size_t size, int gpu);

template <typename T> 
void callCudaMemcpyHostToDevice(T* dest, T* src, size_t size, int gpu) {
    CHECK_ERROR();
    if (size == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Send data to GPU\n");
    cudaSetDevice(gpu);
    gpuErrchk(cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    cudaSetDevice(0);
    printf("Done sending data to GPU\n");
}

template <typename T> 
void callCudaMemcpyDeviceToHost(T* dest, T* src, size_t size, int gpu) {
    CHECK_ERROR();
    if (size == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Send data to CPU\n");
    cudaSetDevice(gpu);
    gpuErrchk(cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_ERROR();
    gpuErrchk(cudaDeviceSynchronize());
    cudaSetDevice(0);
    printf("Done sending data to CPU\n");
}

// Define the host function that launches the CUDA kernel
int* sendDataToGPU(int* data, int size) {
    printf("Send data to GPU\n");
    // use cudamemcpy
    int* target;
    cudaMalloc((void**) &target, size * sizeof(int));
    cudaMemcpy(target, data, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Done sending data to GPU\n");
    return target;
}

} // namespace duckdb