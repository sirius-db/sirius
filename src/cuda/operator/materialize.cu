#include "cuda_helper.cuh"
#include "gpu_columns.hpp"

namespace duckdb {

template <typename T, int B, int I>
__global__ void materialize_expression(const T *a, T* result, uint64_t *row_ids, uint64_t N) {

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
            int items_ids = row_ids[tile_offset + threadIdx.x + ITEM * B];
            uint64_t write_offset = tile_offset + threadIdx.x + ITEM * B;
            result[write_offset] = a[items_ids];
        }
    }

}

template
__global__ void materialize_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, int* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, float* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, uint8_t* result, uint64_t *row_ids, uint64_t N);

template <typename T>
void materializeExpression(T *a, T* result, uint64_t *row_ids, uint64_t N) {
    // Nothing to materialize
    if(N == 0) {
        return;
    }
    
    printf("Launching Materialize Kernel\n");
    int num_items = (int) N;
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    int num_blocks = std::max(1, (int) (N + BLOCK_THREADS - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD));
    printf("Items per Thread: %d, Blocks per Thread: %d, Tile Items: %d, N: %d, Num Blocks: %d\n", ITEMS_PER_THREAD, BLOCK_THREADS, tile_items, num_items, num_blocks);

    materialize_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(a, result, row_ids, N);
    cudaDeviceSynchronize();
    CHECK_ERROR();
}

template
void materializeExpression<int>(int *a, int* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<uint64_t>(uint64_t *a, uint64_t* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<float>(float *a, float* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<double>(double *a, double* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<uint8_t>(uint8_t *a, uint8_t* result, uint64_t *row_ids, uint64_t N);

__global__ void set_materalized_offsets(int* materalized_offsets, int* input_offsets, uint64_t* row_ids, size_t num_rows, int* shifted_offset) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < num_rows) {
        int copy_row_id = static_cast<int>(row_ids[tid]);
        int new_length = input_offsets[copy_row_id + 1] - input_offsets[copy_row_id];
        materalized_offsets[tid] = atomicAdd(shifted_offset, new_length);
        printf("SET MATERALIZE: Copy Row Id - %d, New Len - %d, New Offset - %d\n", copy_row_id, new_length, materalized_offsets[tid]);
    }
}

// Method to materalize the offsets
int strMateralizeOffsets(int* materalized_offsets, int* input_offsets, uint64_t* row_ids, size_t num_rows) {
    // Create the offset tracker
    int* shifted_offset = reinterpret_cast<int*>(callCudaMalloc<int>(1, 0));
    cudaMemset(shifted_offset, 0, sizeof(int));

    // Copy over the offsets
    int num_blocks = std::max((int) 1, (int) (num_rows + BLOCK_THREADS - 1)/BLOCK_THREADS);
    set_materalized_offsets<<<num_blocks, BLOCK_THREADS>>>(materalized_offsets, input_offsets, row_ids, num_rows, shifted_offset);
    cudaDeviceSynchronize();
    cudaMemcpy(materalized_offsets + num_rows, shifted_offset, sizeof(int), cudaMemcpyDeviceToDevice);

    int new_chars_len = 0;
    cudaMemcpy(&new_chars_len, shifted_offset, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Got new chars len of " << new_chars_len << std::endl;

    CHECK_ERROR();
    callCudaFree(shifted_offset, 0);
    return new_chars_len;
}

__global__ void perform_chars_copy(uint8_t* materalized_chars, uint8_t* input_chars, int* materalized_offsets, int* input_offsets, uint64_t* row_ids, size_t num_rows) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < num_rows) {
        int copy_row_id = static_cast<int>(row_ids[tid]);
        int input_start_idx = input_offsets[copy_row_id];
        int input_length = input_offsets[copy_row_id + 1] - input_start_idx;
        int output_start_idx = materalized_offsets[tid];
        printf("CHARS COPY: Copy Row Id - %d, Src Start Idx - %d, Src Length - %d, Dst Write Idx - %d\n", copy_row_id, input_start_idx, input_length, output_start_idx);
        memcpy(materalized_chars + output_start_idx, input_chars + input_start_idx, input_length * sizeof(uint8_t));
    }
}

void strMateralizeChars(uint8_t* materalized_chars, uint8_t* input_chars, int* materalized_offsets, int* input_offsets, uint64_t* row_ids, size_t num_rows) {
    int num_blocks = std::max((int) 1, (int) (num_rows + BLOCK_THREADS - 1)/BLOCK_THREADS);
    perform_chars_copy<<<num_blocks, BLOCK_THREADS>>>(materalized_chars, input_chars, materalized_offsets, input_offsets, row_ids, num_rows);

    cudaDeviceSynchronize();
    CHECK_ERROR();
}

} // namespace duckdb