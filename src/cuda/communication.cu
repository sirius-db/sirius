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
#include <cuda_runtime.h>
#include <cuda.h>
#include "gpu_columns.hpp"
#include "operator/cuda_helper.cuh"
#include "log/logging.hpp"

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

template void
callCudaMemcpyDeviceToHost<char>(char* dest, char* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToHost<string_t>(string_t* dest, string_t* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToDevice<uint8_t>(uint8_t* dest, uint8_t* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToDevice<int>(int* dest, int* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToDevice<uint64_t>(uint64_t* dest, uint64_t* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToDevice<float>(float* dest, float* src, size_t size, int gpu);

template void
callCudaMemcpyDeviceToDevice<double>(double* dest, double* src, size_t size, int gpu);

template <typename T> 
void callCudaMemcpyHostToDevice(T* dest, T* src, size_t size, int gpu) {
    CHECK_ERROR();
    if (size == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Send data to GPU");
    cudaSetDevice(gpu);
    gpuErrchk(cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    cudaSetDevice(0);
    SIRIUS_LOG_DEBUG("Done sending data to GPU");
}

template <typename T> 
void callCudaMemcpyDeviceToHost(T* dest, T* src, size_t size, int gpu) {
    CHECK_ERROR();
    if (size == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    SIRIUS_LOG_DEBUG("Send data to CPU");
    cudaSetDevice(gpu);
    SIRIUS_LOG_DEBUG("Transferred bytes: {}", size * sizeof(T));
    if (src == nullptr) {
        SIRIUS_LOG_DEBUG("src is null");
    }
    if (dest == nullptr) {
        SIRIUS_LOG_DEBUG("dest is null");
    }
    gpuErrchk(cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_ERROR();
    gpuErrchk(cudaDeviceSynchronize());
    cudaSetDevice(0);
    SIRIUS_LOG_DEBUG("Done sending data to CPU");
    STOP_TIMER();
}

template <typename T> 
void callCudaMemcpyDeviceToDevice(T* dest, T* src, size_t size, int gpu) {
    CHECK_ERROR();
    if (size == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    SIRIUS_LOG_DEBUG("Send data within GPU");
    cudaSetDevice(gpu);
    SIRIUS_LOG_DEBUG("Transferred bytes: {}", size * sizeof(T));
    if (src == nullptr) {
        SIRIUS_LOG_DEBUG("src is null");
    }
    if (dest == nullptr) {
        SIRIUS_LOG_DEBUG("dest is null");
    }
    gpuErrchk(cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToDevice));
    CHECK_ERROR();
    gpuErrchk(cudaDeviceSynchronize());
    cudaSetDevice(0);
    SIRIUS_LOG_DEBUG("Done sending data to GPU");
    STOP_TIMER();
}

void callCudaMemset(void* ptr, int value, size_t size, int gpu) {
    CHECK_ERROR();
    if (size == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    SIRIUS_LOG_DEBUG("Setting memory on GPU");
    cudaSetDevice(gpu);
    gpuErrchk(cudaMemset(ptr, value, size));
    CHECK_ERROR();
    gpuErrchk(cudaDeviceSynchronize());
    cudaSetDevice(0);
    SIRIUS_LOG_DEBUG("Done setting memory on GPU");
    STOP_TIMER();
}

} // namespace duckdb