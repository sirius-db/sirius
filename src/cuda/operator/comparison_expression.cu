#include "cuda_helper.cuh"
#include "gpu_expression_executor.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace duckdb {

template <typename T, int B, int I>
__global__ void comparison_expression(const T *a, const T *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T items_a[I];
    T items_b[I];
    int selection_flags[I];

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0; // Number of items selected per thread
    int c_t_count = 0; //Prefix sum of t_count
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            items_a[ITEM] = a[tile_offset + threadIdx.x + ITEM * B];
            items_b[ITEM] = b[tile_offset + threadIdx.x + ITEM * B];
            if (compare_mode == 0) {
                selection_flags[ITEM] = (items_a[ITEM] == items_b[ITEM]);
            } else if (compare_mode == 1) {
                selection_flags[ITEM] = (items_a[ITEM] != items_b[ITEM]);
            } else if (compare_mode == 2) {
                selection_flags[ITEM] = (items_a[ITEM] > items_b[ITEM]);
            } else if (compare_mode == 3) {
                selection_flags[ITEM] = (items_a[ITEM] >= items_b[ITEM]);
            } else if (compare_mode == 4) {
                selection_flags[ITEM] = (items_a[ITEM] < items_b[ITEM]);
            } else if (compare_mode == 5) {
                selection_flags[ITEM] = (items_a[ITEM] <= items_b[ITEM]);
            } else {
                cudaAssert(0);
            }
            if(selection_flags[ITEM]) t_count++;
        }
    }

    //Barrier
    __syncthreads();

    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count); //doing a prefix sum of all the previous threads in the block and store it to c_t_count
    if(threadIdx.x == blockDim.x - 1) { //if the last thread in the block, add the prefix sum of all the prev threads + sum of my threads to global variable total
        block_off = atomicAdd(count, (unsigned long long) t_count+c_t_count); //the previous value of total is gonna be assigned to block_off
    } //block_off does not need to be global (it's just need to be shared), because it will get the previous value from total which is global

    __syncthreads();

    if (is_count) return;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                row_ids[offset] = tile_offset + threadIdx.x + ITEM * B;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void comparison_constant_expression(const T *a, const T b, const T c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T items_a[I];
    int selection_flags[I];

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0; // Number of items selected per thread
    int c_t_count = 0; //Prefix sum of t_count
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            items_a[ITEM] = a[tile_offset + threadIdx.x + ITEM * B];
            if (compare_mode == 0) {
                selection_flags[ITEM] = (items_a[ITEM] == b);
            } else if (compare_mode == 1) {
                selection_flags[ITEM] = (items_a[ITEM] != b);
            } else if (compare_mode == 2) {
                selection_flags[ITEM] = (items_a[ITEM] > b);
            } else if (compare_mode == 3) {
                selection_flags[ITEM] = (items_a[ITEM] >= b);
            } else if (compare_mode == 4) {
                selection_flags[ITEM] = (items_a[ITEM] < b);
            } else if (compare_mode == 5) {
                selection_flags[ITEM] = (items_a[ITEM] <= b);
            } else if (compare_mode == 6) {
                selection_flags[ITEM] = ((items_a[ITEM] >= b) && (items_a[ITEM] <= c));
            } else if (compare_mode == 7) {
                selection_flags[ITEM] = ((items_a[ITEM] < b) || (items_a[ITEM] > c));
            } else if (compare_mode == 8) {
                selection_flags[ITEM] = ((items_a[ITEM] >= b) || (items_a[ITEM] < c));
            } else if (compare_mode == 9) {
                selection_flags[ITEM] = ((items_a[ITEM] > b) || (items_a[ITEM] <= c));
            } else if (compare_mode == 10) {
                selection_flags[ITEM] = ((items_a[ITEM] > b) || (items_a[ITEM] < c));
            } else {
                cudaAssert(0);
            }
            if(selection_flags[ITEM]) {
                t_count++;
            }
        }
    }

    //Barrier
    __syncthreads();

    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count); //doing a prefix sum of all the previous threads in the block and store it to c_t_count
    if(threadIdx.x == blockDim.x - 1) { //if the last thread in the block, add the prefix sum of all the prev threads + sum of my threads to global variable total
        block_off = atomicAdd(count, (unsigned long long) t_count+c_t_count); //the previous value of total is gonna be assigned to block_off
    } //block_off does not need to be global (it's just need to be shared), because it will get the previous value from total which is global

    __syncthreads();

    if (is_count) return;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                row_ids[offset] = tile_offset + threadIdx.x + ITEM * B;
            }
        }
    }
}

template
__global__ void comparison_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, const int *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, const uint64_t *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, const float *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, const double *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, const uint8_t *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);

template
__global__ void comparison_constant_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, const int b, const int c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, const uint64_t b, const uint64_t c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, const float b, const float c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, const double b, const double c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, const uint8_t b, const uint8_t c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);

template <typename T>
void comparisonConstantExpression(T *a, T b, T c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Comparison Expression Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    // size_t openmalloc_full = (gpuBufferManager->processing_size_per_gpu - gpuBufferManager->gpuProcessingPointer[0] - 1024) / sizeof(uint64_t);
    // row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(openmalloc_full, 0, 0);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    comparison_constant_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, c, row_ids, (unsigned long long*) count, N, op_mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    // gpuBufferManager->gpuProcessingPointer[0] = (reinterpret_cast<uint8_t*>(row_ids + h_count[0]) - gpuBufferManager->gpuProcessing[0]);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

template <typename T>
void comparisonExpression(T *a, T *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Comparison Expression Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    // size_t openmalloc_full = (gpuBufferManager->processing_size_per_gpu - gpuBufferManager->gpuProcessingPointer[0] - 1024) / sizeof(uint64_t);
    // row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(openmalloc_full, 0, 0);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    comparison_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, row_ids, (unsigned long long*) count, N, op_mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    // gpuBufferManager->gpuProcessingPointer[0] = (reinterpret_cast<uint8_t*>(row_ids + h_count[0]) - gpuBufferManager->gpuProcessing[0]);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

__global__ void string_comparison_expression(char* char_data, uint64_t num_chars, uint64_t* str_indices, uint64_t num_strings, char* compare_chars, uint64_t compare_length, 
    int op_mode, bool* d_results) {

    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t tile_size = gridDim.x * blockDim.x;

    for(uint64_t i = start_idx; i < num_strings; i += tile_size) {
        // If the result is already false then skip
        if(!d_results[i]) {
            continue;
        }

        // Get the current strings details
        uint64_t start_offset = str_indices[i]; 
        uint64_t end_offset = str_indices[i + 1];
        uint64_t curr_str_length = end_offset - start_offset;
        char* curr_str_chars = char_data + start_offset;

        // First compare the chars
        bool is_valid = true;
        bool found_answer = false;
        if (op_mode == 0 || op_mode == 1) {
            if (op_mode == 0 && compare_length != curr_str_length) {
                is_valid = false;
            } else if (op_mode == 1 && compare_length != curr_str_length) {
                is_valid = true;
            } else {
                for(uint64_t j = 0; j < compare_length; j++) {
                    char curr_str_char = curr_str_chars[j];
                    char compare_char = compare_chars[j];
                    if (op_mode == 0) {
                        if(curr_str_char != compare_char) {
                            is_valid = false;
                            break;
                        }
                    } else if (op_mode == 1) {
                        if(curr_str_char == compare_char) {
                            is_valid = false;
                            break;
                        }
                    }
                }
            }
        } else {
            uint64_t num_compare_chars = min(compare_length, curr_str_length);
            bool is_greater_check;
            bool is_inclusive;
            if (op_mode == 2) {
                is_greater_check = true;
                is_inclusive = false;
            } else if (op_mode == 3) {
                is_greater_check = true;
                is_inclusive = true;
            } else if (op_mode == 4) {
                is_greater_check = false;
                is_inclusive = false;
            } else if (op_mode == 5) {
                is_greater_check = false;
                is_inclusive = true;
            }
            else cudaAssert(0);

            for(uint64_t j = 0; j < num_compare_chars; j++) {
                char curr_str_char = curr_str_chars[j];
                char compare_char = compare_chars[j];
                if(curr_str_char != compare_char) {
                    is_valid = is_greater_check ? curr_str_char > compare_char : curr_str_char < compare_char;
                    found_answer = true;
                    break;
                }
            }

            // If not compare the lengths
            if(!found_answer) {
                is_valid = (is_inclusive && curr_str_length == num_compare_chars) || (is_greater_check && curr_str_length > num_compare_chars) || (!is_greater_check && curr_str_length < num_compare_chars);
            }

        }

        d_results[i] = is_valid;
    }
}

void comparisonStringBetweenExpression(char* char_data, uint64_t num_chars, uint64_t* str_indices, uint64_t num_strings, std::string lower_string, std::string upper_string, 
    bool is_lower_inclusive, bool is_upper_inclusive, uint64_t* &row_id, uint64_t* &count) {

    CHECK_ERROR();
    if (num_strings == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }

    // Allocate the necesary buffers on the GPU
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    uint64_t num_lower_chars = lower_string.length();
    char* d_lower_chars = gpuBufferManager->customCudaMalloc<char>(num_lower_chars, 0, 0);
    cudaMemcpy(d_lower_chars, lower_string.c_str(), num_lower_chars * sizeof(char), cudaMemcpyHostToDevice);

    uint64_t num_upper_chars = upper_string.length();
    char* d_upper_chars = gpuBufferManager->customCudaMalloc<char>(num_upper_chars, 0, 0);
    cudaMemcpy(d_upper_chars, upper_string.c_str(), num_upper_chars * sizeof(char), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    bool* d_is_valid = gpuBufferManager->customCudaMalloc<bool>(num_strings, 0, 0);
    cudaMemset(d_is_valid, 1, num_strings * sizeof(bool));

    // Perform the lower string comparsions
    int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    int num_blocks = (num_strings + items_per_block - 1)/items_per_block;
    int op_mode;
    if (is_lower_inclusive) op_mode = 3;
    else op_mode = 2;
    string_comparison_expression<<<num_blocks, BLOCK_THREADS>>>(char_data, num_chars, str_indices, num_strings, d_lower_chars, num_lower_chars, op_mode, d_is_valid);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Perform the upper string comparsion
    if (is_upper_inclusive) op_mode = 5;
    else op_mode = 4;
    string_comparison_expression<<<num_blocks, BLOCK_THREADS>>>(char_data, num_chars, str_indices, num_strings, d_upper_chars, num_upper_chars, op_mode, d_is_valid);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    cudaMemset(count, 0, sizeof(uint64_t));

    // size_t openmalloc_full = (gpuBufferManager->processing_size_per_gpu - gpuBufferManager->gpuProcessingPointer[0] - 1024) / sizeof(uint64_t);
    // row_id = gpuBufferManager->customCudaMalloc<uint64_t>(openmalloc_full, 0, 0);
    row_id = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);
    compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_is_valid, row_id, (unsigned long long*) count, num_strings, 0, 0);
    CHECK_ERROR();
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    // gpuBufferManager->gpuProcessingPointer[0] = (reinterpret_cast<uint8_t*>(row_id + h_count[0]) - gpuBufferManager->gpuProcessing[0]);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;
    std::cout << "comparisonStringBetweenExpression got count of " << h_count[0] << std::endl;

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_lower_chars), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_upper_chars), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_is_valid), 0);
}

void comparisonStringExpression(char* char_data, uint64_t num_chars, uint64_t* str_indices, uint64_t num_strings, std::string comparison_string, int op_mode, uint64_t* &row_id, uint64_t* &count) {

    CHECK_ERROR();
    if (num_strings == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }

    // Allocate the necesary buffers on the GPU
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    uint64_t num_compare_chars = comparison_string.length();
    char* d_compare_chars = gpuBufferManager->customCudaMalloc<char>(num_compare_chars, 0, 0);
    cudaMemcpy(d_compare_chars, comparison_string.c_str(), num_compare_chars * sizeof(char), cudaMemcpyHostToDevice);

    bool* d_is_valid = gpuBufferManager->customCudaMalloc<bool>(num_strings, 0, 0);
    cudaMemset(d_is_valid, 1, num_strings * sizeof(bool));

    int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    int num_blocks = (num_strings + items_per_block - 1)/items_per_block;
    string_comparison_expression<<<num_blocks, BLOCK_THREADS>>>(char_data, num_chars, str_indices, num_strings, d_compare_chars, num_compare_chars, op_mode, d_is_valid);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    cudaMemset(count, 0, sizeof(uint64_t));
    // size_t openmalloc_full = (gpuBufferManager->processing_size_per_gpu - gpuBufferManager->gpuProcessingPointer[0] - 1024) / sizeof(uint64_t);
    // row_id = gpuBufferManager->customCudaMalloc<uint64_t>(openmalloc_full, 0, 0);

    compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_is_valid, row_id, (unsigned long long*) count, num_strings, 0, 0);
    CHECK_ERROR();

    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    // gpuBufferManager->gpuProcessingPointer[0] = (reinterpret_cast<uint8_t*>(row_id + h_count[0]) - gpuBufferManager->gpuProcessing[0]);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;
    std::cout << "comparisonStringExpression got count of " << h_count[0] << std::endl;

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_compare_chars), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_is_valid), 0);
}

template
void comparisonConstantExpression<int>(int *a, int b, int c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<uint64_t>(uint64_t *a, uint64_t b, uint64_t c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<float>(float *a, float b, float c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<double>(double *a, double b, double c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<uint8_t>(uint8_t *a, uint8_t b, uint8_t c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);


template
void comparisonExpression<int>(int *a, int *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<uint64_t>(uint64_t *a, uint64_t *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<float>(float *a, float *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<double>(double *a, double *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<uint8_t>(uint8_t *a, uint8_t *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);

} // namespace duckdb