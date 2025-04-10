#include "cuda_helper.cuh"
#include "gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace duckdb {

template <typename T>
__device__ bool device_comparison(T item, T compare, CompareType compare_mode) {
    bool flag = 0;
    if (compare_mode == EQUAL) {
        flag = (item == compare);
    } else if (compare_mode == NOTEQUAL) {
        flag = (item != compare);
    } else if (compare_mode == GREATERTHAN) {
        flag = (item > compare);
    } else if (compare_mode == GREATERTHANOREQUALTO) {
        flag = (item >= compare);
    } else if (compare_mode == LESSTHAN) {
        flag = (item < compare);
    } else if (compare_mode == LESSTHANOREQUALTO) {
        flag = (item <= compare);
    } else {
        cudaAssert(0);
    }
    return flag;
}

__device__ int device_comparison_string(char* char_data, uint64_t char_len, char* compare_chars, uint64_t compare_length, CompareType compare_mode) {
    
    // First compare the chars
    bool is_valid = true;
    bool found_answer = false;
    if (compare_mode == EQUAL || compare_mode == NOTEQUAL) {
        if (compare_mode == EQUAL && compare_length != char_len) {
            is_valid = false;
        } else if (compare_mode == NOTEQUAL && compare_length != char_len) {
            is_valid = true;
        } else {
            for(uint64_t j = 0; j < compare_length; j++) {
                char curr_str_char = char_data[j];
                char compare_char = compare_chars[j];
                if (compare_mode == EQUAL) {
                    if(curr_str_char != compare_char) {
                        is_valid = false;
                        break;
                    }
                } else if (compare_mode == NOTEQUAL) {
                    if(curr_str_char == compare_char) {
                        is_valid = false;
                        break;
                    }
                }
            }
        }
    } else {
        uint64_t num_compare_chars = min(compare_length, char_len);
        bool is_greater_check;
        bool is_inclusive;
        if (compare_mode == GREATERTHAN) {
            is_greater_check = true;
            is_inclusive = false;
        } else if (compare_mode == GREATERTHANOREQUALTO) {
            is_greater_check = true;
            is_inclusive = true;
        } else if (compare_mode == LESSTHAN) {
            is_greater_check = false;
            is_inclusive = false;
        } else if (compare_mode == LESSTHANOREQUALTO) {
            is_greater_check = false;
            is_inclusive = true;
        }
        else cudaAssert(0);

        for(uint64_t j = 0; j < num_compare_chars; j++) {
            char curr_str_char = char_data[j];
            char compare_char = compare_chars[j];
            if(curr_str_char != compare_char) {
                is_valid = is_greater_check ? curr_str_char > compare_char : curr_str_char < compare_char;
                found_answer = true;
                break;
            }
        }

        if(!found_answer) {
            is_valid = (is_inclusive && char_len == num_compare_chars) || (is_greater_check && char_len > num_compare_chars) || (!is_greater_check && char_len < num_compare_chars);
        }
    }

    return is_valid;
}

template <int B, int I>
__global__ void table_scan_expression(uint8_t **col, uint64_t** offset, uint8_t *constant_compare, uint64_t *constant_offset, 
        ScanDataType* data_type, uint64_t *row_ids, unsigned long long* count, uint64_t N, CompareType* compare_mode, int is_count, int num_expr) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    bool selection_flags[I];

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
        for (int expr = 0; expr < num_expr; expr++) {
            if (threadIdx.x + ITEM * B < num_tile_items) {

                if (data_type[expr] == INT32) {
                    uint64_t item_idx = tile_offset + threadIdx.x + ITEM * B;
                    int item = (reinterpret_cast<int*>(col[expr]))[item_idx];

                    uint64_t start_constant_offset = constant_offset[expr]; 
                    int constant;
                    memcpy(&constant, constant_compare + start_constant_offset, sizeof(int));

                    selection_flags[ITEM] = device_comparison<int>(item, constant, compare_mode[expr]);

                } else if (data_type[expr] == INT64) {
                    uint64_t item_idx = tile_offset + threadIdx.x + ITEM * B;
                    uint64_t item = (reinterpret_cast<uint64_t*>(col[expr]))[item_idx];

                    uint64_t start_constant_offset = constant_offset[expr]; 
                    uint64_t constant;
                    memcpy(&constant, constant_compare + start_constant_offset, sizeof(uint64_t));
                    
                    selection_flags[ITEM] = device_comparison<uint64_t>(item, constant, compare_mode[expr]);

                } else if (data_type[expr] == FLOAT32) {
                    uint64_t item_idx = tile_offset + threadIdx.x + ITEM * B;
                    float item = (reinterpret_cast<float*>(col[expr]))[item_idx];

                    uint64_t start_constant_offset = constant_offset[expr]; 
                    float constant;
                    memcpy(&constant, constant_compare + start_constant_offset, sizeof(float));
                    
                    selection_flags[ITEM] = device_comparison<float>(item, constant, compare_mode[expr]);
                } else if (data_type[expr] == FLOAT64) {
                    uint64_t item_idx = tile_offset + threadIdx.x + ITEM * B;
                    double item = (reinterpret_cast<double*>(col[expr]))[item_idx];

                    uint64_t start_constant_offset = constant_offset[expr]; 
                    double constant;
                    memcpy(&constant, constant_compare + start_constant_offset, sizeof(double));
                    
                    selection_flags[ITEM] = device_comparison<double>(item, constant, compare_mode[expr]);
                } else if (data_type[expr] == BOOLEAN) {
                    uint64_t item_idx = tile_offset + threadIdx.x + ITEM * B;
                    uint8_t item = (reinterpret_cast<uint8_t*>(col[expr]))[item_idx];

                    uint64_t start_constant_offset = constant_offset[expr]; 
                    uint8_t constant = (reinterpret_cast<uint8_t*>(constant_compare))[start_constant_offset];
                    
                    selection_flags[ITEM] = device_comparison<uint8_t>(item, constant, compare_mode[expr]);
                } else if (data_type[expr] == VARCHAR) {
                    uint64_t item_idx = tile_offset + threadIdx.x + ITEM * B;

                    uint64_t start_offset = offset[expr][item_idx]; 
                    uint64_t end_offset = offset[expr][item_idx + 1];
                    uint64_t curr_str_length = end_offset - start_offset;
                    char* curr_str_chars = (char*) col[expr] + start_offset;

                    uint64_t start_constant_offset = constant_offset[expr]; 
                    uint64_t end_constant_offset = constant_offset[expr + 1];
                    uint64_t compare_length = end_constant_offset - start_constant_offset;
                    char* compare_chars = (char*) constant_compare + start_constant_offset;

                    selection_flags[ITEM] = device_comparison_string(curr_str_chars, curr_str_length, compare_chars, compare_length, compare_mode[expr]);
                } else {
                    cudaAssert(0);
                }
    
                if(!selection_flags[ITEM]) {
                    break;
                }
            }
        }
        
        if (selection_flags[ITEM]) {
            t_count++;
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

__global__ void print_columns(uint8_t **col, uint64_t** offset, uint64_t N) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < 100; i++) {
            printf("%ld\n", (reinterpret_cast<uint64_t*>(col[0]))[i]);
        }
    }
}

void 
tableScanExpression(uint8_t **col, uint64_t** offset, uint8_t *constant_compare, uint64_t *constant_offset, 
        ScanDataType* data_type, uint64_t *&row_ids, uint64_t* &count, uint64_t N, CompareType* compare_mode, int num_expr) {

    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Arbitrary Table Scan Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    uint64_t constant_size = constant_offset[num_expr];
    uint8_t* d_constant_compare = gpuBufferManager->customCudaMalloc<uint8_t>(constant_size, 0, 0);
    uint64_t* d_constant_offset = gpuBufferManager->customCudaMalloc<uint64_t>(num_expr + 1, 0, 0);
    cudaMemcpy(d_constant_compare, constant_compare, constant_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_constant_offset, constant_offset, (num_expr + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);

    CompareType* d_compare_mode = (CompareType*) gpuBufferManager->customCudaMalloc<int>(num_expr, 0, 0);
    cudaMemcpy(d_compare_mode, compare_mode, num_expr * sizeof(int), cudaMemcpyHostToDevice);
    ScanDataType* d_data_type = (ScanDataType*) gpuBufferManager->customCudaMalloc<int>(num_expr, 0, 0);
    cudaMemcpy(d_data_type, data_type, num_expr * sizeof(int), cudaMemcpyHostToDevice);

    // uint8_t** d_col = gpuBufferManager->customCudaMalloc<uint8_t*>(num_expr, 0, 0);
    // uint64_t** d_offset = gpuBufferManager->customCudaMalloc<uint64_t*>(num_expr, 0, 0);
    uint8_t** d_col;
    uint64_t** d_offset;
    cudaMalloc((void**) &d_col, num_expr * sizeof(uint8_t*));
    cudaMalloc((void**) &d_offset, num_expr * sizeof(uint64_t*));
    cudaMemcpy(d_col, col, num_expr * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, offset, num_expr * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    // print_columns<<<1, 1>>>(d_col, d_offset, N);
    // cudaDeviceSynchronize();

    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    table_scan_expression<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            d_col, d_offset, d_constant_compare, d_constant_offset, d_data_type, row_ids, (unsigned long long*) count, N, d_compare_mode, 1, num_expr);
    CHECK_ERROR();
    
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    table_scan_expression<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            d_col, d_offset, d_constant_compare, d_constant_offset, d_data_type, row_ids, (unsigned long long*) count, N, d_compare_mode, 0, num_expr);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]); 
}

} // namespace duckdb
