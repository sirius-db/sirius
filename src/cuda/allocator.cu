#include <iostream>
#include "operator/cuda_helper.cuh"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

template int*
callCudaMalloc<int>(size_t size, int gpu);

template uint64_t*
callCudaMalloc<uint64_t>(size_t size, int gpu);

template uint8_t*
callCudaMalloc<uint8_t>(size_t size, int gpu);

template float*
callCudaMalloc<float>(size_t size, int gpu);

template double*
callCudaMalloc<double>(size_t size, int gpu);

template bool*
callCudaMalloc<bool>(size_t size, int gpu);

template void
callCudaFree<int>(int* ptr, int gpu);

template void
callCudaFree<uint64_t>(uint64_t* ptr, int gpu);

template void
callCudaFree<uint8_t>(uint8_t* ptr, int gpu);

template void
callCudaFree<float>(float* ptr, int gpu);

template void
callCudaFree<double>(double* ptr, int gpu);

template void
callCudaFree<bool>(bool* ptr, int gpu);

template <typename T>
T* callCudaMalloc(size_t size, int gpu) {
    T* ptr;
    cudaSetDevice(gpu);
    gpuErrchk(cudaMalloc((void**) &ptr, size * sizeof(T)));
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    return ptr;
}

template <typename T>
void callCudaFree(T* ptr, int gpu) {
    cudaSetDevice(gpu);
    gpuErrchk(cudaFree(ptr));
    cudaDeviceSynchronize();
    cudaSetDevice(gpu);
}

} // namespace duckdb