#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gpu_buffer_manager.hpp"
#include "operator/cuda_helper.cuh"

namespace duckdb {

template <typename T>
__global__ void print_gpu_column(T* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < N; i++) {
            printf("a: %.2f ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void print_gpu_column<uint64_t>(uint64_t* a, uint64_t N);
template
__global__ void print_gpu_column<double>(double* a, uint64_t N);
template
__global__ void print_gpu_column<int>(int* a, uint64_t N);
template
__global__ void print_gpu_column<float>(float* a, uint64_t N);
template
__global__ void print_gpu_column<uint8_t>(uint8_t* a, uint64_t N);

template <typename T> 
void printGPUColumn(T* a, size_t N, int gpu) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    T* result_host_temp = new T[1];
    cudaMemcpy(result_host_temp, a, sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Result: %ld and N: %d\n", result_host_temp[0], N);
    printf("N: %ld\n", N);
    print_gpu_column<T><<<1, 1>>>(a, N);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template void printGPUColumn<uint64_t>(uint64_t* a, size_t N, int gpu);
template void printGPUColumn<double>(double* a, size_t N, int gpu);
template void printGPUColumn<int>(int* a, size_t N, int gpu);
template void printGPUColumn<float>(float* a, size_t N, int gpu);

} // namespace duckdb