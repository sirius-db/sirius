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
    cudaError_t err = cudaSetDevice(gpu);
    if (err != cudaSuccess) {
        printf("CUDA initialization error: %s\n", cudaGetErrorString(err));
    }
    int nDevices;
    err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    int driverVersion = 0;
    err = cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        printf("CUDA driver error: %s\n", cudaGetErrorString(err));
    }
    printf("Number of devices: %d\n", nDevices);
  
    // for (int i = 0; i < nDevices; i++) {
    //     cudaDeviceProp prop;
    //     cudaGetDeviceProperties(&prop, i);
    //     printf("Device Number: %d\n", i);
    //     printf("  Device name: %s\n", prop.name);
    //     printf("  Memory Clock Rate (MHz): %d\n",
    //             prop.memoryClockRate/1024);
    //     printf("  Memory Bus Width (bits): %d\n",
    //             prop.memoryBusWidth);
    //     printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
    //             2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    //     printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    //     printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    //     printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    //     printf("  Warp-size: %d\n", prop.warpSize);
    //     printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    //     printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
    // }
    CHECK_ERROR();

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