#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "communication.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

template <typename T> 
void callCudaMemcpyHostToDevice(T* dest, T* src, size_t size, int gpu) {
    printf("Send data to GPU\n");
    cudaSetDevice(gpu);
    gpuErrchk(cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    cudaSetDevice(0);
    printf("Done sending data to GPU\n");
}

// Define the host function that launches the CUDA kernel
int* sendDataToGPU(int* data, int size) {
    printf("Send data to GPU\n");
    // use cudamemcpy
    int** target = new int*[1];
    cudaMalloc((void**) &target[0], size * sizeof(int));
    cudaMemcpy(target, data, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Done sending data to GPU\n");
    return target[0];
}

} // namespace duckdb