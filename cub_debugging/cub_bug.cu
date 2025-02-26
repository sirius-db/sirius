#include <chrono>
#include <stdexcept>
#include <cub/cub.cuh>
#include <cstdio>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <cmath>

#define CHECK_ERROR() { \
    cudaDeviceSynchronize(); \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) \
    { \
      printf("CUDA error: %s\n", cudaGetErrorString(error)); \
      exit(-1); \
    } \
  }

#define CHUNK_SIZE 1000
#define THREADS_IN_BLOCK 256
#define ITEMS_PER_THREAD 8

struct custom_key_type {
    uint64_t key_value;

    __host__ __device__ custom_key_type() {}
    __host__ __device__ custom_key_type(uint64_t _value) : key_value(_value) {}

    __host__ __device__ bool operator==(const custom_key_type& other) const {
        uint64_t curr_chunk = key_value/CHUNK_SIZE;
        uint64_t other_chunk = other.key_value/CHUNK_SIZE;
        return curr_chunk == other_chunk;
    }
};

struct CustomReductionOperator {
    uint64_t* records_buffer;
    uint64_t num_records;

    __host__ CustomReductionOperator(uint64_t* _records_buffer, uint64_t _num_records) : 
        records_buffer(_records_buffer), num_records(_num_records) {}

    __device__ __forceinline__ uint64_t operator()(const uint64_t& left, const uint64_t& right) const {
        // Verify the records
        uint64_t lower_value = min(left, right);
        uint64_t upper_value = max(left, right);
        assert(lower_value < num_records);
        assert(upper_value < num_records);

        // Combine the value at the upper index to the value at the lower index
        records_buffer[lower_value] += records_buffer[upper_value];
        return lower_value;
    }
};

__global__ void create_key_records(custom_key_type* key_records, uint64_t num_records) {
    uint64_t tile_size = gridDim.x * blockDim.x;
    uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i = start_idx; i < num_records; i += tile_size) {
        key_records[i] = custom_key_type(i);
    }
}

__global__ void verify_key_records(custom_key_type* key_records) {
    printf("First 5 Key records: ");
    for(uint64_t i = 0; i < 5; i++) {
        printf("%lu ", key_records[i].key_value);
    }
    printf("\n");
}

__global__ void verify_value_records(uint64_t* value_records) {
    printf("First 5 Value records: ");
    for(uint64_t i = 0; i < 5; i++) {
        printf("%lu ", value_records[i]);
    }
    printf("\n");
}

int main() {
    // Create the key records buffer
    const uint64_t num_records = static_cast<uint64_t>(pow(2.0, 20.0));
    std::cout << "Running experiment with num records of " << num_records << std::endl;
    custom_key_type* d_key_records;
    cudaMalloc((void**) &d_key_records, num_records * sizeof(custom_key_type));

    // Populate the key records buffer
    uint64_t records_per_block = THREADS_IN_BLOCK * ITEMS_PER_THREAD;
    uint64_t blocks_needed = (num_records + records_per_block - 1)/records_per_block;
    create_key_records<<<blocks_needed, THREADS_IN_BLOCK>>>(d_key_records, num_records);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Verify the records
    verify_key_records<<<1, 1>>>(d_key_records);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Now create the values buffer
    uint64_t* d_value_records;
    cudaMalloc((void**) &d_value_records, num_records * sizeof(uint64_t));
    cudaMemset(d_value_records, 0, num_records * sizeof(uint64_t));

    verify_value_records<<<1, 1>>>(d_value_records);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Create the records buffer
    uint64_t* d_records_buffer;
    cudaMalloc((void**) &d_records_buffer, num_records * sizeof(uint64_t));
    cudaMemset(d_records_buffer, 0, num_records * sizeof(uint64_t));

    // Create the additional fields for the reduce by key
    custom_key_type* d_result_keys;
    uint64_t* d_result_values;
    uint64_t* d_num_runs;
    CustomReductionOperator reduction_operator(d_records_buffer, num_records);

    cudaMalloc((void**) &d_result_keys, num_records * sizeof(custom_key_type));
    cudaMalloc((void**) &d_result_values, num_records * sizeof(uint64_t));
    cudaMalloc((void**) &d_num_runs, sizeof(uint64_t));
    cudaMemset(d_num_runs, 0, sizeof(uint64_t));

    // Now actually perform the reduction
    void*  d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey<
        custom_key_type*,
        custom_key_type*,
        uint64_t*,
        uint64_t*,
        uint64_t*,
        CustomReductionOperator,
        uint64_t    
    >(
        d_temp_storage, 
        temp_storage_bytes, 
        d_key_records, 
        d_result_keys, 
        d_value_records, 
        d_result_values, 
        d_num_runs, 
        reduction_operator, 
        num_records
    );
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaMemset(d_temp_storage, 0, temp_storage_bytes);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    cub::DeviceReduce::ReduceByKey<
        custom_key_type*,
        custom_key_type*,
        uint64_t*,
        uint64_t*,
        uint64_t*,
        CustomReductionOperator,
        uint64_t    
    >(
        d_temp_storage, 
        temp_storage_bytes, 
        d_key_records, 
        d_result_keys, 
        d_value_records, 
        d_result_values, 
        d_num_runs, 
        reduction_operator, 
        num_records
    );
    cudaDeviceSynchronize();
    CHECK_ERROR();
}