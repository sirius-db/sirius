#include "../operator/cuda_helper.cuh"
#include "gpu_columns.hpp"

namespace duckdb {

template<int B, int I>
__global__ void convert_uint64_to_int32(uint64_t* data, int32_t* output, size_t N) {
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t val = data[tile_offset + threadIdx.x + ITEM * B];
            output[tile_offset + threadIdx.x + ITEM * B] = static_cast<int32_t>(val);
        }
    }
}

template<int B, int I>
__global__ void convert_int32_to_uint64(int32_t* data, uint64_t* output, size_t N) {
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }
    
    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            int32_t val = data[tile_offset + threadIdx.x + ITEM * B];
            output[tile_offset + threadIdx.x + ITEM * B] = static_cast<uint64_t>(val);
        }
    }
}

int32_t* convertUInt64ToInt32(uint64_t* data, size_t N) {
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    int32_t* output_dev = gpuBufferManager->customCudaMalloc<int32_t>(N, 0, 0);
    convert_uint64_to_int32<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(data, output_dev, N);
    CHECK_ERROR();
    return output_dev;
}

uint64_t* convertInt32ToUInt64(int32_t* data, size_t N) {
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* output_dev = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    convert_int32_to_uint64<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(data, output_dev, N);
    CHECK_ERROR();
    return output_dev;
}

cudf::bitmask_type* createNullMask(size_t size, cudf::mask_state state) {
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    size_t mask_bytes = getMaskBytesSize(size);
    uint8_t* mask = gpuBufferManager->customCudaMalloc<uint8_t>(mask_bytes, 0, 0);
    if (state == cudf::mask_state::ALL_VALID) {
        cudaMemset(mask, 0xFF, mask_bytes);
    }
    return reinterpret_cast<cudf::bitmask_type*>(mask);
}

} // namespace duckdb