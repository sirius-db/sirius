#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

#include <chrono>

namespace duckdb {

using std::chrono::high_resolution_clock;
using std::chrono::duration;

struct sort_keys_type_string {
  uint64_t row_id;  
  uint64_t* keys;
  uint64_t num_key;

  __host__ __device__ sort_keys_type_string() {}
  __host__ __device__ sort_keys_type_string(uint64_t _row_id, uint64_t* _keys, uint64_t _num_key) : row_id(_row_id), keys(_keys), num_key(_num_key) {}

  __host__ __device__ bool operator<(const sort_keys_type_string& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) {
            return keys[i] < other.keys[i];
        }
      }
      return true;
    }

    __host__ __device__ bool operator==(const sort_keys_type_string& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) return false;
      }
      return true;
    }

    __host__ __device__ bool operator!=(const sort_keys_type_string& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) return true;
      }
      return false;
    }
};

constexpr bool V1_LOG_MODE = false;
__device__ int d_comparator_keys_compared_v1 = 0;
__device__ int d_comparator_num_comparsions_v1 = 0;

struct CustomLessString
{
 __host__ __device__ CustomLessString() {}

  __device__ bool operator()(const sort_keys_type_string &lhs, const sort_keys_type_string &rhs) {
    if constexpr(V1_LOG_MODE) {
        atomicAdd(&d_comparator_num_comparsions_v1, (int) 1);
    }
    
    for (uint64_t i = 0; i < lhs.num_key; i++) {
        if (lhs.keys[i] != rhs.keys[i]) {
            if constexpr(V1_LOG_MODE) {
                atomicAdd(&d_comparator_keys_compared_v1, (int) i);
            }
            return lhs.keys[i] < rhs.keys[i];
        }
    }

    if constexpr(V1_LOG_MODE) {
        atomicAdd(&d_comparator_keys_compared_v1, (int) lhs.num_key);
    }
    return true;
  }
};

struct CustomSumString
{
    template <typename T>
    __host__ __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct CustomMinString
{
    template <typename T>
    __host__ __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct CustomMaxString
{
    template <typename T>
    __host__ __device__ __forceinline__
     T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

template <typename T, int B, int I>
__global__ void fill_offset(uint64_t* offset, uint64_t N) {
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
            offset[tile_offset + threadIdx.x + ITEM * B] = sizeof(T) * (tile_offset + threadIdx.x + ITEM * B);
        }
    }
}

template <int B, int I>
__global__ void columns_to_rows_string(uint8_t **a, uint8_t* result, uint64_t **input_offset, uint64_t* key_length,
            sort_keys_type_string* temp, uint64_t N, uint64_t num_keys) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    uint64_t total_length = 0;
    for (uint64_t key = 0; key < (num_keys - 1); key ++) {
        total_length += key_length[key];
    }
    //add the row ids into the total length
    total_length += sizeof(uint64_t);

    uint64_t meta_num_keys = (total_length + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t total_length_bytes = meta_num_keys * sizeof(uint64_t);

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            uint64_t output_start_idx = offset * total_length_bytes;
            memset(result + output_start_idx, 0, total_length_bytes * sizeof(uint8_t));
            //copy the keys without the row ids
            for (uint64_t key = 0; key < (num_keys - 1); key ++) {
                uint64_t input_length = input_offset[key][offset + 1] - input_offset[key][offset];
                uint64_t input_start_idx = input_offset[key][offset];
                memcpy(result + output_start_idx, a[key] + input_start_idx, input_length * sizeof(uint8_t));
                output_start_idx += key_length[key];
            }
            //copy the row ids
            memcpy(result + (offset * total_length_bytes) + ((meta_num_keys - 1) * sizeof(uint64_t)), a[num_keys - 1] + (offset * sizeof(uint64_t)), sizeof(uint64_t));
            temp[offset] = sort_keys_type_string(offset, reinterpret_cast<uint64_t*>(&result[offset * total_length_bytes]), meta_num_keys);
        }
    }
}

template <int B, int I>
__global__ void compact_string_offset(uint64_t* group_idx, uint64_t** group_byte_offset, uint64_t** result_offset, uint64_t N, uint64_t num_keys) {
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
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            if (offset == N - 1) {
                uint64_t out_idx = group_idx[offset];
                for (uint64_t key = 0; key < num_keys; key ++) {
                    result_offset[key][out_idx] = group_byte_offset[key][offset];
                }
            } else if ((offset < (N - 1)) && (group_idx[offset] != group_idx[offset + 1])) {
                uint64_t out_idx = group_idx[offset];
                for (uint64_t key = 0; key < num_keys; key ++) {
                    cudaAssert(group_byte_offset[key][offset] != group_byte_offset[key][offset + 1]);
                    result_offset[key][out_idx] = group_byte_offset[key][offset];
                }
            }
        }
    }
}

__global__ void print_sort_metadata_v1() {
    float average_compare_values = (1.0 * d_comparator_keys_compared_v1)/d_comparator_num_comparsions_v1;
    printf("STRING GROUP BY V1: Performed %d row comparsions checking an average of %f values\n", d_comparator_num_comparsions_v1, average_compare_values);
}

template <int B, int I>
__global__ void rows_to_columns_string(uint64_t* group_idx, sort_keys_type_string *row_keys, uint8_t** col_keys, uint64_t **group_byte_offset, uint64_t* key_length,
    uint64_t N, uint64_t num_keys) {

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
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            //we should write out the offset
            if (group_idx[offset] != group_idx[offset + 1]) {
                uint64_t out_idx = group_idx[offset];
                uint64_t key_length_bytes = 0;
                for (uint64_t key = 0; key < num_keys; key ++) {
                    cudaAssert(group_byte_offset[key][offset] != group_byte_offset[key][offset + 1]);
                    uint64_t out_offset = group_byte_offset[key][offset];
                    uint64_t actual_key_length = group_byte_offset[key][offset + 1] - group_byte_offset[key][offset];
                    uint8_t* ptr = reinterpret_cast<uint8_t*>(row_keys[out_idx].keys);
                    memcpy(col_keys[key] + out_offset, ptr + key_length_bytes, actual_key_length * sizeof(uint8_t));
                    key_length_bytes += key_length[key];
                }
                // char temp1[5];
                // char temp2[18];
                // memcpy(temp1, col_keys[0] + group_byte_offset[0][offset], 5);
                // memcpy(temp2, col_keys[1] + group_byte_offset[1][offset], 18);
                // printf("String %s %s\n", temp1, temp2);
                // printf("%ld %ld\n", row_keys[out_idx].keys, row_keys[out_idx].keys);
            }
        }
    }
}

template <int B, int I>
__global__ void get_len(uint64_t* offset, uint64_t* len, uint64_t N) {
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
            uint64_t idx = tile_offset + threadIdx.x + ITEM * B;
            len[idx] = offset[idx + 1] - offset[idx];
        }
    }
}


template <int B, int I>
__global__ void distinct_string(uint64_t* distinct_mark, uint64_t* distinct_len, uint64_t* len, sort_keys_type_string *sort_keys, uint64_t N) {
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
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            if (offset == 0 || (offset > 0  && (sort_keys[offset] != sort_keys[offset - 1]))) {
                distinct_mark[offset] = 1;
                distinct_len[offset] = len[offset];
            } else {
                distinct_mark[offset] = 0;
                distinct_len[offset] = 0;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void gather_and_modify(const T *a, T* result, sort_keys_type_string *sort_keys, uint64_t N, uint64_t meta_num_keys) {
    cudaAssert(meta_num_keys > 1);
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
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            uint64_t items_ids = sort_keys[offset].keys[meta_num_keys - 1];
            result[offset] = a[items_ids];
            sort_keys[offset] = sort_keys_type_string(offset, sort_keys[offset].keys, meta_num_keys - 1);
        }
    }
}

template <typename T, int B, int I>
__global__ void gather(const T *a, T* result, sort_keys_type_string *sort_keys, uint64_t N, uint64_t num_keys) {

    cudaAssert(num_keys > 1);
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
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            uint64_t items_ids = sort_keys[offset].keys[num_keys - 1];
            result[offset] = a[items_ids];
        }
    }
}

template <int B, int I>
__global__ void modify(sort_keys_type_string *sort_keys, uint64_t N, uint64_t meta_num_keys) {

    cudaAssert(meta_num_keys > 1);
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
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            sort_keys[offset] = sort_keys_type_string(offset, sort_keys[offset].keys, meta_num_keys - 1);
        }
    }
}

template <int B, int I>
__global__ void sequence(uint64_t* result, uint64_t N) {

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
            result[tile_offset + threadIdx.x + ITEM * B] = tile_offset + threadIdx.x + ITEM * B;
        }
    }
}

template <typename T, int B, int I>
__global__ void divide(T* a, uint64_t* b, T* result, uint64_t N) {

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
            int offset = tile_offset + threadIdx.x + ITEM * B;
            result[offset] = a[offset] / b[offset];
        }
    }
}

template <typename T, int B, int I>
__global__ void fill_n(T* a, T b, uint64_t N) {
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
            a[tile_offset + threadIdx.x + ITEM * B] = b;
        }
    }
}

template <typename T>
__global__ void testprint(T* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < N; i++) {
            printf("%.2f ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, sort_keys_type_string* sort_keys, uint64_t N, uint64_t meta_num_keys);
template
__global__ void gather_and_modify<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, sort_keys_type_string* sort_keys, uint64_t N, uint64_t meta_num_keys);
template
__global__ void gather<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, sort_keys_type_string* sort_keys, uint64_t N, uint64_t num_keys);

template <typename V>
void groupedStringAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }

    printf("Launching String Grouped Aggregate Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;

    //cubmax
    auto preprocess_start_time = high_resolution_clock::now();
    // Get the maximum key length for each key
    uint64_t* key_length = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys, 0, 0); // store the maximum length of each key
    uint64_t** len = new uint64_t*[num_keys];
    for (int key = 0; key < num_keys; key++) {
        len[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);

        if (offset[key] == nullptr) {
            offset[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
            fill_offset<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD), BLOCK_THREADS>>>(offset[key], N+1);
            CHECK_ERROR();
        }

        get_len<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD), BLOCK_THREADS>>>(offset[key], len[key], N);
        CHECK_ERROR();
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;

        if (offset[key] == nullptr) {
            cudaMemcpy(key_length + key, len[key], sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        } else {
            cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, len[key], key_length + key, N);

            // Allocate temporary storage
            d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

            // Run min-reduction
            cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, len[key], key_length + key, N);
        }
    }

    uint64_t* h_key_length = new uint64_t[num_keys];
    cudaMemcpy(h_key_length, key_length, num_keys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();

    uint64_t row_id_size = sizeof(uint64_t);
    uint64_t total_length = 0;
    for (uint64_t key = 0; key < num_keys; key ++) {
        total_length += h_key_length[key];
    }
    //add the row ids into the total length
    total_length += row_id_size;
    uint64_t meta_num_keys = (total_length + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t total_length_bytes = meta_num_keys * sizeof(uint64_t);
    // printf("Total Length: %lu\n", total_length);
    // printf("Total Length Bytes: %lu\n", total_length_bytes);

    //allocate temp memory and copying keys
    uint64_t total_preprocessing_bytes = total_length_bytes * N;
    uint8_t* row_keys = gpuBufferManager->customCudaMalloc<uint8_t>(total_preprocessing_bytes, 0, 0);
    sort_keys_type_string* materialized_temp = reinterpret_cast<sort_keys_type_string*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));

    uint8_t** keys_row_id = new uint8_t*[num_keys + 1];
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_row_id[i] = keys[i];
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t* row_sequence = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    sequence<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(row_sequence, N);
    keys_row_id[num_keys] = reinterpret_cast<uint8_t*> (row_sequence);

    uint8_t** keys_dev;
    cudaMalloc((void**) &keys_dev, (num_keys + 1) * sizeof(uint8_t*));
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + 1) * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    uint64_t** offset_dev;
    cudaMalloc((void**) &offset_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(offset_dev, offset, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    columns_to_rows_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, offset_dev, key_length,
            materialized_temp, N, num_keys + 1);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    std::cout << "STRING GROUP BY V1: Preprocessing requires " << meta_num_keys << " ints per row with " << N << " rows taking " << total_preprocessing_bytes << " bytes" << std::endl;
    std::cout << "STRING GROUP BY V1: Preprocessing took " << preprocess_time_ms << " ms" << std::endl;

    //perform sort-based groupby
    // Determine temporary device storage requirements
    auto sort_start_time = high_resolution_clock::now();
    CustomLessString custom_less;
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_less);

    CHECK_ERROR();

    // Allocate temporary storage
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run sorting operation
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_less);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto sort_end_time = high_resolution_clock::now();
    auto sort_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(sort_end_time - sort_start_time).count();
    
    // Log the results
    print_sort_metadata_v1<<<1, 1>>>();
    std::cout << "STRING GROUP BY V1: Sorting took " << sort_time_ms << " ms" << std::endl;

    auto group_by_start_time = high_resolution_clock::now();
    // printf("Gathering offset\n");
    uint64_t** group_byte_offset = new uint64_t*[num_keys];
    uint64_t* distinct_bound = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    uint64_t* group_idx = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
    uint64_t* d_num_bytes = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys, 0, 0);

    for (uint64_t key = 0; key < num_keys; key++) {
        uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
        group_byte_offset[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
        cudaMemset(group_byte_offset[key] + N, 0, sizeof(uint64_t));

        gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(len[key], temp, materialized_temp, N, meta_num_keys);
        CHECK_ERROR();
        distinct_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(distinct_bound, temp, temp, materialized_temp, N);
        CHECK_ERROR();
        //cub scan
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp, group_byte_offset[key], N + 1);

        // Allocate temporary storage for exclusive prefix sum
        d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

        // Run exclusive prefix sum
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp, group_byte_offset[key], N + 1);
        CHECK_ERROR();

        cudaMemcpy(d_num_bytes + key, group_byte_offset[key] + N, sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        CHECK_ERROR();
    }

    //copy num_bytes over
    cudaMemcpy(num_bytes, d_num_bytes, num_keys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint64_t** group_byte_offset_dev;
    cudaMalloc((void**) &group_byte_offset_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(group_byte_offset_dev, group_byte_offset, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    //cub scan
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, distinct_bound, group_idx, N + 1);

    // Allocate temporary storage for exclusive prefix sum
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, distinct_bound, group_idx, N + 1);
    CHECK_ERROR();

    //gather the aggregates based on the row_sequence
    // printf("Gathering Aggregates\n");
    V** aggregate_keys_temp = new V*[num_aggregates];
    uint64_t** aggregate_star_temp = new uint64_t*[num_aggregates];
    sort_keys_type_string* group_by_rows = reinterpret_cast<sort_keys_type_string*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    uint64_t* h_count = new uint64_t[1];

    for (int agg = 0; agg < num_aggregates; agg++) {
        // printf("Aggregating %d\n", agg);
        cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
        if (agg_mode[agg] == 4 || agg_mode[agg] == 5) { //count_star or count(null) or sum(null)
            aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            if (agg_mode[agg] == 4) {
                fill_n<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], 1, N);
            } else if (agg_mode[agg] == 5) {
                cudaMemset(aggregate_star_temp[agg], 0, N * sizeof(double));
            }

            modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(materialized_temp, N, meta_num_keys);
            CHECK_ERROR();

            //perform reduce_by_key
            uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));

            // printf("Reduce by key count_star\n");
            // Determine temporary device storage requirements
            d_temp_storage = nullptr;
            temp_storage_bytes = 0;
            CustomSumString custom_sum;
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, aggregate_star_temp[agg],
                agg_star_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            // Allocate temporary storage
            d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

            // Run reduce-by-key
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, aggregate_star_temp[agg],
                agg_star_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            count[0] = h_count[0];

            // printf("Count: %lu\n", count[0]);

            CHECK_ERROR();
            aggregate_keys[agg] = reinterpret_cast<uint8_t*> (agg_star_out);
        } else {
            aggregate_keys_temp[agg] = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            V* temp = reinterpret_cast<V*> (aggregate_keys[agg]);
            gather_and_modify<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(temp, aggregate_keys_temp[agg], materialized_temp, N, meta_num_keys);
            CHECK_ERROR();

            V* agg_out = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            cudaMemset(agg_out, 0, N * sizeof(V));

            CHECK_ERROR();
            if (agg_mode[agg] == 0) {
                // printf("Reduce by key sum\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSumString custom_sum;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (agg_out);
                // printf("Count: %lu\n", count[0]);
            } else if (agg_mode[agg] == 1) {
                //Currently typename V has to be a double
                // printf("Reduce by key avg\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSumString custom_sum;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
                fill_n<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], 1, N);

                uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
                cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));
                cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));

                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_star_temp[agg],
                    agg_star_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_star_temp[agg],
                    agg_star_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                V* output = gpuBufferManager->customCudaMalloc<V>(count[0], 0, 0);
                divide<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(agg_out, agg_star_out, output, count[0]);

                CHECK_ERROR();
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (output);
            } else if (agg_mode[agg] == 2) {
                // printf("Reduce by key max\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMaxString custom_max;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_max, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_max, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (agg_out);
            } else if (agg_mode[agg] == 3) {
                // printf("Reduce by key min\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMinString custom_min;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_min, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_min, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (agg_out);
            }
        }
    }
    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto group_by_end_time = high_resolution_clock::now();
    auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
    std::cout << "STRING GROUP BY V1: Group By took " << group_by_time_ms << " ms" << std::endl;

    auto post_processing_start_time = high_resolution_clock::now();
    uint64_t** offset_dev_result;
    cudaMalloc((void**) &offset_dev_result, num_keys * sizeof(uint64_t*));
    for (uint64_t i = 0; i < num_keys; i++) {
        offset[i] = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
    }
    cudaMemcpy(offset_dev_result, offset, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    compact_string_offset<BLOCK_THREADS, ITEMS_PER_THREAD><<<((N + 1) + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            group_idx, group_byte_offset_dev, offset_dev_result, N + 1, num_keys);

    CHECK_ERROR();

    uint8_t** keys_dev_result;
    cudaMalloc((void**) &keys_dev_result, num_keys * sizeof(uint8_t*));
    for (uint64_t i = 0; i < num_keys; i++) {
        uint64_t* temp_num_bytes = new uint64_t[1];
        cudaMemcpy(temp_num_bytes, offset[i] + count[0], sizeof(uint64_t), cudaMemcpyDeviceToHost);
        keys[i] = gpuBufferManager->customCudaMalloc<uint8_t>(temp_num_bytes[0], 0, 0);
    }
    cudaMemcpy(keys_dev_result, keys, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    rows_to_columns_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            group_idx, group_by_rows, keys_dev_result, group_byte_offset_dev, key_length, N, num_keys);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto post_processing_end_time = high_resolution_clock::now();
    auto post_processing_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(post_processing_end_time - post_processing_start_time).count();
    std::cout << "STRING GROUP BY V1: Post Processing took " << post_processing_time_ms << " ms" << std::endl;

    // testprint<uint64_t><<<1, 1>>>(group_idx, N);
    // testprint<double><<<1, 1>>>(reinterpret_cast<double*> (aggregate_keys[0]), N);
    // testprint<uint64_t><<<1, 1>>>(offset[1], N);
    // CHECK_ERROR();

    cudaDeviceSynchronize();
    printf("Count: %lu\n", count[0]);
    throw std::runtime_error("Grouped String Aggregate V1 implementation stop");
}

__global__ void add_offset(uint64_t* a, uint64_t* b, uint64_t offset, uint64_t N) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = b[idx] + offset;
    }
}

void combineStrings(uint8_t* a, uint8_t* b, uint8_t* c, 
        uint64_t* offset_a, uint64_t* offset_b, uint64_t* offset_c, 
        uint64_t num_bytes_a, uint64_t num_bytes_b, uint64_t N_a, uint64_t N_b) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        printf("N is 0\n");
        return;
    }
    cudaMemcpy(c, a, num_bytes_a * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(c + num_bytes_a, b, num_bytes_b * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    cudaMemcpy(offset_c, offset_a, N_a * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    add_offset<<<((N_b + 1) + BLOCK_THREADS - 1)/(BLOCK_THREADS), BLOCK_THREADS>>>(offset_c + N_a, offset_b, num_bytes_a, N_b + 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template
void groupedStringAggregate<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedStringAggregate<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

}