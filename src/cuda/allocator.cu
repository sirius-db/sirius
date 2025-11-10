/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include "operator/cuda_helper.cuh"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

template int*
callCudaMalloc<int>(size_t size, int gpu);

template uint64_t*
callCudaMalloc<uint64_t>(size_t size, int gpu);

template uint8_t*
callCudaMalloc<uint8_t>(size_t size, int gpu);

template uint32_t*
callCudaMalloc<uint32_t>(size_t size, int gpu);

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

template uint32_t*
callCudaHostAlloc<uint32_t>(size_t size, bool return_dev_ptr);

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
callCudaFree<uint32_t>(uint32_t* ptr, int gpu);

template void
callCudaFree<float>(float* ptr, int gpu);

template void
callCudaFree<double>(double* ptr, int gpu);

template <typename T>
T* callCudaMalloc(size_t size, int gpu) {
    T* ptr;
    cudaError_t err = cudaSetDevice(gpu);
    if (err != cudaSuccess) {
        SIRIUS_LOG_ERROR("CUDA initialization error for gpu {}: {}", gpu, cudaGetErrorString(err));
    }
    int nDevices;
    err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) {
        SIRIUS_LOG_ERROR("CUDA error for gpu {}: {}", gpu, cudaGetErrorString(err));
    }

    int driverVersion = 0;
    err = cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        SIRIUS_LOG_ERROR("CUDA driver error: {}", cudaGetErrorString(err));
    }
    SIRIUS_LOG_DEBUG("Number of devices: {}", nDevices);
    CHECK_ERROR();

    SIRIUS_LOG_DEBUG("Allocating {} bytes on GPU {}", size * sizeof(T), gpu);
    gpuErrchk(cudaMalloc((void**) &ptr, size * sizeof(T)));
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    return ptr;
}

template <typename T>
T* callCudaHostAlloc(size_t size, bool return_dev_ptr) {
    T* ptr;
    SIRIUS_LOG_DEBUG("Allocating {} bytes on CPU", size * sizeof(T));
    gpuErrchk(cudaHostAlloc((void**) &ptr, size * sizeof(T), cudaHostAllocMapped));
    if (return_dev_ptr) {
        T* return_ptr;
        gpuErrchk(cudaHostGetDevicePointer((void **)&return_ptr, ptr, 0));
        return return_ptr;
    }
    return ptr;
}

uint8_t* allocatePinnedCPUMemory(size_t size) {
    uint8_t* ptr;
    gpuErrchk(cudaHostAlloc((void**) &ptr, size * sizeof(uint8_t), cudaHostAllocDefault));
    return ptr;
}

uint8_t* allocatePageableCPUMemory(size_t size) {
    return (uint8_t*)malloc(size * sizeof(uint8_t));
}

void freePinnedCPUMemory(uint8_t* ptr) {
    cudaFreeHost(ptr);
}

void freePageableCPUMemory(uint8_t* ptr) {
    free(ptr);
}

template <typename T>
void callCudaFree(T* ptr, int gpu) {
    cudaSetDevice(gpu);
    gpuErrchk(cudaFree(ptr));
    cudaDeviceSynchronize();
    cudaSetDevice(gpu);
}

size_t getFreeGPUMemorySize(int gpu) {
    gpuErrchk(cudaSetDevice(gpu));
    size_t free, total;
    gpuErrchk(cudaMemGetInfo(&free, &total));
    gpuErrchk(cudaSetDevice(0));
    return free;
}

} // namespace duckdb