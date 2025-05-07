#include "cuda_helper.cuh"
#include "gpu_columns.hpp"

#include <chrono>
#include <cmath>

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
            // if (N == 3793296 && (items_ids < 0 || items_ids >= 3793296)) printf("items_ids: %d\n", items_ids);
            result[tile_offset + threadIdx.x + ITEM * B] = a[items_ids];
            // cudaAssert(a[items_ids] == 19940101);
            // printf("Result: %ld\n", result[tile_offset + threadIdx.x + ITEM * B]);
        }
    }

}

__global__ void materialize_offset(uint64_t* offset, uint64_t* result_length, uint64_t* row_ids, size_t N) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N) {
        uint64_t copy_row_id = row_ids[tid];
        uint64_t new_length = offset[copy_row_id + 1] - offset[copy_row_id];
        result_length[tid] = new_length;
        // printf("%ld %ld\n", tid, result_length[tid]);
        // printf("SET MATERALIZE: Copy Row Id - %d, New Len - %d, New Offset - %d\n", copy_row_id, new_length, materalized_offsets[tid]);
    }
}

__global__ void materialize_string(uint8_t* data, uint8_t* result, uint64_t* input_offset, uint64_t* materialized_offset, uint64_t* row_ids, size_t num_rows) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < num_rows) {
        uint64_t copy_row_id = row_ids[tid];
        uint64_t input_start_idx = input_offset[copy_row_id];
        uint64_t input_length = input_offset[copy_row_id + 1] - input_offset[copy_row_id];
        uint64_t output_start_idx = materialized_offset[tid];
        // printf("CHARS COPY: Copy Row Id - %ld, Src Start Idx - %ld, Src Length - %ld, Dst Write Idx - %ld\n", copy_row_id, input_start_idx, input_length, output_start_idx);
        memcpy(result + output_start_idx, data + input_start_idx, input_length * sizeof(uint8_t));
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
void materializeExpression(T *a, T*& result, uint64_t *row_ids, uint64_t result_len, uint64_t input_len) {
    CHECK_ERROR();
    if (result_len == 0) {
        printf("result_len is 0\n");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    printf("Launching Materialize Kernel\n");
    // SETUP_TIMING();
    // START_TIMER();
    // printf("result_len: %lu\n", N);
    // testprintmat<T><<<1, 1>>>(a, N);
    // CHECK_ERROR();
    // testprintmat<uint64_t><<<1, 1>>>(row_ids, N);
    // CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    result = gpuBufferManager->customCudaMalloc<T>(result_len, 0, 0);
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    materialize_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(result_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, result, row_ids, result_len);
    CHECK_ERROR();
    // thrust::device_vector<T> sorted(result, result + N);
    // thrust::sort(thrust::device, sorted.begin(), sorted.end());
    // T* raw_sorted = thrust::raw_pointer_cast(sorted.data());
    // cudaMemcpy(result, raw_sorted, N * sizeof(T), cudaMemcpyDeviceToDevice);
    // testprintmat<T><<<1, 1>>>(result, 100);
    cudaDeviceSynchronize();
    STOP_TIMER();
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(a), 0);
}

void materializeString(uint8_t* data, uint64_t* offset, uint8_t* &result, uint64_t* &result_offset, uint64_t* row_ids, uint64_t* &result_bytes, uint64_t result_len, uint64_t input_size, uint64_t input_bytes) {
    CHECK_ERROR();
    if (result_len == 0) {
        printf("result_len is 0\n");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    printf("Launching Materialize String Kernel\n");
    // SETUP_TIMING();
    // START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    //allocate temp memory and copying keys
    uint64_t* temp_len = gpuBufferManager->customCudaMalloc<uint64_t>(result_len + 1, 0, 0);
    result_offset = gpuBufferManager->customCudaMalloc<uint64_t>(result_len + 1, 0, 0);

    cudaMemset(temp_len + result_len, 0, sizeof(uint64_t));
    CHECK_ERROR();

    // Copy over the offsets
    uint64_t num_blocks = std::max((uint64_t) 1, (uint64_t) (result_len + BLOCK_THREADS - 1)/BLOCK_THREADS);
    materialize_offset<<<num_blocks, BLOCK_THREADS>>>(offset, temp_len, row_ids, result_len);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    //cub scan
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp_len, result_offset, result_len + 1);

    // Allocate temporary storage for exclusive prefix sum
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp_len, result_offset, result_len + 1);
    CHECK_ERROR();

    // testprintmat<uint64_t><<<1, 1>>>(result_offset, N + 1);

    result_bytes = new uint64_t[1];
    cudaMemcpy(result_bytes, result_offset + result_len, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    // std::cout << "Got new chars len of " << new_num_bytes[0] << std::endl;

    CHECK_ERROR();

    result = gpuBufferManager->customCudaMalloc<uint8_t>(result_bytes[0], 0, 0);

    materialize_string<<<num_blocks, BLOCK_THREADS>>>(data, result, offset, result_offset, row_ids, result_len);
    cudaDeviceSynchronize();

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(temp_len), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(data), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset), 0);
    CHECK_ERROR();
    STOP_TIMER();
}

__global__ void create_cpu_strings(duckdb_string_type* gpu_strings, char* cpu_chars_buffer, char* gpu_chars, uint64_t* string_offsets, size_t num_strings, size_t inline_threshold) { 
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_strings) {
        uint64_t str_offset = string_offsets[idx];
        uint64_t str_length = string_offsets[idx + 1] - str_offset;
        
        // Use the threshold to determine if we should inline the string or not
        duckdb_string_type& curr_string = gpu_strings[idx];
        if(str_length <= inline_threshold) {
            curr_string.value.inlined.length = str_length;
            char* gpu_data_ptr = gpu_chars + str_offset;
            memcpy(curr_string.value.inlined.inlined, gpu_data_ptr, str_length);
        } else {
            curr_string.value.pointer.length = str_length;
            curr_string.value.pointer.ptr = cpu_chars_buffer + str_offset;
        }
    }
}

void materializeStringColumnToDuckdbFormat(GPUColumn* column, char* column_char_write_buffer, string_t* column_string_write_buffer) {
    // First copy the characters from the GPU to the CPU
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    DataWrapper column_data = column->data_wrapper;

    size_t column_num_chars = column_data.num_bytes;
    size_t column_chars_bytes = column_num_chars * sizeof(char);
    char* gpu_chars = reinterpret_cast<char*>(column_data.data);
    cudaMemcpy(column_char_write_buffer, gpu_chars, column_chars_bytes, cudaMemcpyDeviceToHost);
    
    // Now use the CPU buffer to create the strings on the GPU (using the CPU buffer to set the address)
    size_t num_strings = column_data.size;
    size_t num_blocks = (num_strings + BLOCK_THREADS - 1)/BLOCK_THREADS;
    duckdb_string_type* d_column_strings = gpuBufferManager->customCudaMalloc<duckdb_string_type>(num_strings, 0, 0);
    size_t cpu_str_bytes = num_strings * sizeof(string_t);
    create_cpu_strings<<<num_blocks, BLOCK_THREADS>>>(d_column_strings, column_char_write_buffer, gpu_chars, column_data.offset, num_strings, static_cast<size_t>(string_t::INLINE_LENGTH));

    // Copy over the strings to the CPU
    column->data_wrapper.data = reinterpret_cast<uint8_t*>(column_char_write_buffer);
    cudaMemcpy((uint8_t*) column_string_write_buffer, (uint8_t*) d_column_strings, cpu_str_bytes, cudaMemcpyDeviceToHost);
}

template
void materializeExpression<int>(int *a, int*& result, uint64_t *row_ids, uint64_t result_len, uint64_t input_size);
template
void materializeExpression<uint64_t>(uint64_t *a, uint64_t*& result, uint64_t *row_ids, uint64_t result_len, uint64_t input_size);
template
void materializeExpression<float>(float *a, float*& result, uint64_t *row_ids, uint64_t result_len, uint64_t input_size);
template
void materializeExpression<double>(double *a, double*& result, uint64_t *row_ids, uint64_t result_len, uint64_t input_size);
template
void materializeExpression<uint8_t>(uint8_t *a, uint8_t*& result, uint64_t *row_ids, uint64_t result_len, uint64_t input_size);

} // namespace duckdb