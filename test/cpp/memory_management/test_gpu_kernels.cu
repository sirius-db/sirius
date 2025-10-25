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

#include "test_gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void gpu_memory_work_kernel(uint32_t* data, size_t num_elements, uint64_t* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements
    uint64_t local_sum = 0;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        // Write a pattern to memory
        data[i] = static_cast<uint32_t>(i ^ 0xDEADBEEF ^ threadIdx.x);
        
        // Read back and accumulate (prevents optimization)
        local_sum += data[i];
        
        // Do some additional computation to simulate real work
        if (data[i] % 2 == 0) {
            data[i] = data[i] ^ 0xCAFEBABE;
        } else {
            data[i] = data[i] + threadIdx.x;
        }
        
        local_sum += data[i];
    }
    
    // Use atomics to accumulate results from all threads
    atomicAdd(reinterpret_cast<unsigned long long*>(result), static_cast<unsigned long long>(local_sum));
}

__global__ void gpu_memory_verification_kernel(uint32_t* data, size_t num_elements, uint64_t* checksum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    uint64_t local_checksum = 0;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        // Verify the data has been modified and compute checksum
        local_checksum += data[i];
        
        // Additional verification work
        uint32_t expected_pattern = static_cast<uint32_t>(i ^ 0xDEADBEEF ^ (i % 32)); // threadIdx.x would be up to 31
        if ((data[i] % 2 == 0 && (data[i] & 0xCAFEBABE) != 0) || 
            (data[i] % 2 == 1 && data[i] > expected_pattern)) {
            // Data appears to have been processed correctly
            local_checksum += 1;
        }
    }
    
    atomicAdd(reinterpret_cast<unsigned long long*>(checksum), static_cast<unsigned long long>(local_checksum));
}

extern "C" {

cudaError_t performGpuMemoryWork(void* gpu_ptr, size_t size_bytes, uint64_t* result_checksum) {
    if (!gpu_ptr || size_bytes == 0 || !result_checksum) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate number of elements
    size_t num_elements = size_bytes / sizeof(uint32_t);
    if (num_elements == 0) {
        return cudaErrorInvalidValue;
    }
    
    uint32_t* data = static_cast<uint32_t*>(gpu_ptr);
    
    // Allocate GPU memory for result
    uint64_t* d_result;
    cudaError_t err = cudaMalloc(&d_result, sizeof(uint64_t));
    if (err != cudaSuccess) {
        return err;
    }
    
    // Initialize result to 0
    err = cudaMemset(d_result, 0, sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(d_result);
        return err;
    }
    
    // Calculate grid and block dimensions
    int block_size = 256;
    int num_blocks = std::min(static_cast<int>((num_elements + block_size - 1) / block_size), 65535);
    
    // Launch kernel to do memory work
    gpu_memory_work_kernel<<<num_blocks, block_size>>>(data, num_elements, d_result);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_result);
        return err;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_result);
        return err;
    }
    
    // Copy result back to host
    err = cudaMemcpy(result_checksum, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_result);
    
    return err;
}

cudaError_t verifyGpuMemoryWork(void* gpu_ptr, size_t size_bytes, uint64_t* verification_checksum) {
    if (!gpu_ptr || size_bytes == 0 || !verification_checksum) {
        return cudaErrorInvalidValue;
    }
    
    size_t num_elements = size_bytes / sizeof(uint32_t);
    if (num_elements == 0) {
        return cudaErrorInvalidValue;
    }
    
    uint32_t* data = static_cast<uint32_t*>(gpu_ptr);
    
    // Allocate GPU memory for checksum
    uint64_t* d_checksum;
    cudaError_t err = cudaMalloc(&d_checksum, sizeof(uint64_t));
    if (err != cudaSuccess) {
        return err;
    }
    
    // Initialize checksum to 0
    err = cudaMemset(d_checksum, 0, sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(d_checksum);
        return err;
    }
    
    // Calculate grid and block dimensions
    int block_size = 256;
    int num_blocks = std::min(static_cast<int>((num_elements + block_size - 1) / block_size), 65535);
    
    // Launch verification kernel
    gpu_memory_verification_kernel<<<num_blocks, block_size>>>(data, num_elements, d_checksum);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_checksum);
        return err;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_checksum);
        return err;
    }
    
    // Copy result back to host
    err = cudaMemcpy(verification_checksum, d_checksum, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_checksum);
    
    return err;
}

} // extern "C"
