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
  
    // for (int i = 0; i < nDevices; i++) {
    //     cudaDeviceProp prop;
    //     cudaGetDeviceProperties(&prop, i);
    //     SIRIUS_LOG_DEBUG("Device Number: {}", i);
    //     SIRIUS_LOG_DEBUG("  Device name: {}", prop.name);
    //     SIRIUS_LOG_DEBUG("  Memory Clock Rate (MHz): {}",
    //             prop.memoryClockRate/1024);
    //     SIRIUS_LOG_DEBUG("  Memory Bus Width (bits): {}",
    //             prop.memoryBusWidth);
    //     SIRIUS_LOG_DEBUG("  Peak Memory Bandwidth (GB/s): {:.1f}",
    //             2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    //     SIRIUS_LOG_DEBUG("  Total global memory (Gbytes) {:.1f}",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    //     SIRIUS_LOG_DEBUG("  Shared memory per block (Kbytes) {:.1f}",(float)(prop.sharedMemPerBlock)/1024.0);
    //     SIRIUS_LOG_DEBUG("  minor-major: {}-{}\n", prop.minor, prop.major);
    //     SIRIUS_LOG_DEBUG("  Warp-size: {}", prop.warpSize);
    //     SIRIUS_LOG_DEBUG("  Concurrent kernels: {}", prop.concurrentKernels ? "yes" : "no");
    //     SIRIUS_LOG_DEBUG("  Concurrent computation/communication: {}",prop.deviceOverlap ? "yes" : "no");
    // }
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

void freePinnedCPUMemory(uint8_t* ptr) {
    cudaFreeHost(ptr);
}

template <typename T>
void callCudaFree(T* ptr, int gpu) {
    cudaSetDevice(gpu);
    gpuErrchk(cudaFree(ptr));
    cudaDeviceSynchronize();
    cudaSetDevice(gpu);
}

} // namespace duckdb