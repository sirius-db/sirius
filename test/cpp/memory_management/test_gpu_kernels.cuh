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

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs memory work on GPU memory using CUDA kernels.
 * 
 * This function launches CUDA kernels to write patterns to GPU memory,
 * read the data back, and compute a checksum. This ensures actual work
 * is performed on the GPU memory and prevents compiler optimizations.
 * 
 * @param gpu_ptr Pointer to GPU memory (must be device memory)
 * @param size_bytes Size of the memory region in bytes
 * @param result_checksum Output parameter for the computed checksum
 * @return cudaError_t indicating success or failure
 */
cudaError_t performGpuMemoryWork(void* gpu_ptr, size_t size_bytes, uint64_t* result_checksum);

/**
 * @brief Verifies that GPU memory work was performed correctly.
 * 
 * This function launches kernels to verify the memory contains the expected
 * patterns and computes a verification checksum.
 * 
 * @param gpu_ptr Pointer to GPU memory (must be device memory)
 * @param size_bytes Size of the memory region in bytes
 * @param verification_checksum Output parameter for the verification checksum
 * @return cudaError_t indicating success or failure
 */
cudaError_t verifyGpuMemoryWork(void* gpu_ptr, size_t size_bytes, uint64_t* verification_checksum);

#ifdef __cplusplus
}
#endif
