/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

//! @file
//! CUDA Runtime API compatibility shim for ROCm/HIP.
//!
//! Sirius (and its dependencies RMM, cuDF, cuCascade) call the CUDA Runtime
//! API directly: @c cudaMalloc, @c cudaMemcpy, @c cudaError_t, @c cudaSuccess,
//! etc. On a stock ROCm install these symbols do not exist — HIP uses the
//! @c hip* prefix and a different header (<hip/hip_runtime.h>).
//!
//! This shim, placed first on the include path when @c ENABLE_ROCM is ON,
//! pulls in the HIP runtime and then @c #define's every @c cuda* symbol to its
//! @c hip* equivalent. This mirrors what hipDF's own build infrastructure
//! provides: hipDF keeps @c cuda* calls verbatim in its source and relies on
//! a compat layer at build time.
//!
//! @c CUDART_VERSION is defined as 0 so that @c #if CUDART_VERSION >= 12080
//! guards (which gate the CUDA 12.8 batch-memcpy API) route to the serial
//! @c cudaMemcpyAsync fallback — HIP lacks @c cudaMemcpyBatchAsync in the
//! pre-6.0 era and the struct layouts differ in 7.x.

#ifndef SIRIUS_ROCM_COMPAT_CUDA_RUNTIME_H
#define SIRIUS_ROCM_COMPAT_CUDA_RUNTIME_H

// Pull in the full HIP runtime — provides hip*, __shared__, __global__,
// <<<>>>, hipStream_t, hipEvent_t, etc.
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// ---------------------------------------------------------------------------
// CUDART_VERSION — set to 0 so #if CUDART_VERSION >= 12080 guards take the
// serial cudaMemcpyAsync fallback (no cudaMemcpyBatchAsync on HIP).
// ---------------------------------------------------------------------------
#ifndef CUDART_VERSION
#define CUDART_VERSION 0
#endif

// ---------------------------------------------------------------------------
// Type aliases (using, not #define — they are types, not macros)
// ---------------------------------------------------------------------------
using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaStream_t = hipStream_t;
using cudaMemPool_t = hipMemPool_t;
// CUDA names it cudaDeviceProp; HIP names it hipDeviceProp_t.
using cudaDeviceProp = hipDeviceProp_t;
// CUDA and HIP have the same struct names for these (verified on ROCm 7.2.1).
// They are already defined via hip/driver_types.h.

// cudaPointerAttributes → hipPointerAttribute_t. On ROCm 7.2.1, the HIP struct
// has the same field names Sirius uses: .type, .device, .devicePointer,
// .hostPointer. Direct alias — no wrapper needed.
using cudaPointerAttributes = hipPointerAttribute_t;
#define cudaPointerGetAttributes hipPointerGetAttributes

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorIllegalAddress hipErrorIllegalAddress
#define cudaErrorMisalignedAddress hipErrorMisalignedAddress
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorCudartUnloading hipErrorCudartUnloading

// ---------------------------------------------------------------------------
// cudaMemcpy kinds
// ---------------------------------------------------------------------------
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

// ---------------------------------------------------------------------------
// Event flags
// ---------------------------------------------------------------------------
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync

// ---------------------------------------------------------------------------
// Host alloc flags
// ---------------------------------------------------------------------------
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped

// ---------------------------------------------------------------------------
// Device attribute enums — CUDA uses cudaDevAttr*, HIP uses
// hipDeviceAttribute*.  Map the ones Sirius uses.
// ---------------------------------------------------------------------------
#define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiProcessorCount
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor \
  hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define cudaDevAttrMaxThreadsPerMultiprocessor \
  hipDeviceAttributeMaxThreadsPerMultiprocessor

// ---------------------------------------------------------------------------
// MemPool attributes
// ---------------------------------------------------------------------------
#define cudaMemPoolAttrReservedMemCurrent hipMemPoolAttrReservedMemCurrent
#define cudaMemPoolAttrUsedMemCurrent hipMemPoolAttrUsedMemCurrent

// ---------------------------------------------------------------------------
// Memory type / location enums
// ---------------------------------------------------------------------------
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemLocationTypeHost hipMemLocationTypeHost
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice

// ---------------------------------------------------------------------------
// Memcpy batch attributes (only reached if CUDART_VERSION >= 12080, which
// is never true since CUDART_VERSION == 0.  Defined for completeness.)
// ---------------------------------------------------------------------------
#define cudaMemcpySrcAccessOrderStream hipMemcpySrcAccessOrderStream
#define cudaMemcpyFlagDefault hipMemcpyFlagDefault

// ---------------------------------------------------------------------------
// Stream constants
// ---------------------------------------------------------------------------
#define cudaStreamPerThread hipStreamPerThread

// ---------------------------------------------------------------------------
// Runtime API functions — 1:1 cuda→hip prefix swap
// ---------------------------------------------------------------------------
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipMallocHost
#define cudaFreeHost hipFreeHost
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#define cudaMemPoolTrimTo hipMemPoolTrimTo
#define cudaHostAlloc hipHostAlloc
#define cudaHostGetDevicePointer hipHostGetDevicePointer

// ---------------------------------------------------------------------------
// Profiler API — Sirius declares these extern "C" in sirius_extension.cpp:
//   extern "C" int cudaProfilerStart();
//   extern "C" int cudaProfilerStop();
// HIP provides hipProfilerStart/Stop (deprecated but functional in 7.2.1),
// but they return hipError_t (enum), not int. A #define would create a
// redeclaration mismatch (int vs hipError_t). Instead, provide inline
// wrapper functions with C linkage and the correct int return type.
// The `extern "C"` matches Sirius's own declarations, and `inline` makes
// them COMDAT-safe across TUs.
// ---------------------------------------------------------------------------
extern "C" {
inline int cudaProfilerStart() { return static_cast<int>(hipProfilerStart()); }
inline int cudaProfilerStop() { return static_cast<int>(hipProfilerStop()); }
}

#endif  // SIRIUS_ROCM_COMPAT_CUDA_RUNTIME_H
