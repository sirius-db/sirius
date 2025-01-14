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

template int*
callCudaHostAlloc<int>(size_t size, bool return_dev_ptr);

template uint64_t*
callCudaHostAlloc<uint64_t>(size_t size, bool return_dev_ptr);

template uint8_t*
callCudaHostAlloc<uint8_t>(size_t size, bool return_dev_ptr);

template float*
callCudaHostAlloc<float>(size_t size, bool return_dev_ptr);

template double*
callCudaHostAlloc<double>(size_t size, bool return_dev_ptr);

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

template <typename T>
T* callCudaMalloc(size_t size, int gpu) {
    T* ptr;
    cudaSetDevice(gpu);
    printf("Allocating %lu bytes on GPU %d\n", size * sizeof(T), gpu);
    gpuErrchk(cudaMalloc((void**) &ptr, size * sizeof(T)));
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    return ptr;
}

template <typename T>
T* callCudaHostAlloc(size_t size, bool return_dev_ptr) {
    T* ptr;
    printf("Allocating %lu bytes on CPU\n", size * sizeof(T));
    gpuErrchk(cudaHostAlloc((void**) &ptr, size * sizeof(T), cudaHostAllocMapped));
    if (return_dev_ptr) {
        T* return_ptr;
        gpuErrchk(cudaHostGetDevicePointer((void **)&return_ptr, ptr, 0));
        return return_ptr;
    }
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