#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

struct sort_keys_type {
  uint64_t* keys;
  uint64_t num_key;

  __host__ __device__ sort_keys_type() {}
  __host__ __device__ sort_keys_type(uint64_t* keys, uint64_t num_key) : keys(keys), num_key(num_key) {}

  __host__ __device__ bool operator<(const sort_keys_type& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) {
            return keys[i] < other.keys[i];
        }
      }
      return true;
    }

    __host__ __device__ bool operator==(const sort_keys_type& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) return false;
      }
      return true;
    }
};

struct CustomLess
{
  __device__ bool operator()(const sort_keys_type &lhs, const sort_keys_type &rhs) {
      for (uint64_t i = 0; i < lhs.num_key; i++) {
        // Handle floating-point comparisons with a tolerance
        if (std::is_floating_point<decltype(lhs.keys[i])>::value) {
            if (fabs(lhs.keys[i] - rhs.keys[i]) > 1e-6) {
                return lhs.keys[i] < rhs.keys[i];
            }
        } else {
            if (lhs.keys[i] != rhs.keys[i]) {
                return lhs.keys[i] < rhs.keys[i];
            }
        }
    }
      return true;
  }
};

struct CustomSum
{
    __device__ int32_t operator()(const int32_t &a, const int32_t &b) {
        return a + b;
    }

    __device__ float operator()(const float &a, const float &b) {
        return a + b;
    }

    __device__ uint64_t operator()(const uint64_t &a, const uint64_t &b) {
        return a + b;
    }

    __device__ double operator()(const double &a, const double &b) {
        return a + b;
    }
};

struct CustomMin
{
    __device__ int32_t operator()(const int32_t &a, const int32_t &b) const {
        return (b < a) ? b : a;
    }

    __device__ float operator()(const float &a, const float &b) const {
        return (b < a) ? b : a;
    }

    __device__ uint64_t operator()(const uint64_t &a, const uint64_t &b) const {
        return (b < a) ? b : a;
    }

    __device__ double operator()(const double &a, const double &b) const {
        return (b < a) ? b : a;
    }
};

struct CustomMax
{
    __device__ int32_t operator()(const int32_t &a, const int32_t &b) const {
        return (b > a) ? b : a;
    }

    __device__ float operator()(const float &a, const float &b) const {
        return (b > a) ? b : a;
    }

    __device__ uint64_t operator()(const uint64_t &a, const uint64_t &b) const {
        return (b > a) ? b : a;
    }

    __device__ double operator()(const double &a, const double &b) const {
        return (b > a) ? b : a;
    }
};

template <int B, int I>
__global__ void columns_to_rows(void** a, void* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys, int key_type) {
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
            for (uint64_t i = 0; i < num_keys; i++) {
                switch(key_type) {
                    case 0: // int32_t
                        reinterpret_cast<int32_t*>(result)[offset * num_keys + i] = reinterpret_cast<int32_t**>(a)[i][offset];
                        break;
                    case 1: // float
                        reinterpret_cast<float*>(result)[offset * num_keys + i] = reinterpret_cast<float**>(a)[i][offset];
                        break;
                    case 2: // uint64_t
                        reinterpret_cast<uint64_t*>(result)[offset * num_keys + i] = reinterpret_cast<uint64_t**>(a)[i][offset];
                        break;
                    case 3: // double
                        reinterpret_cast<double*>(result)[offset * num_keys + i] = reinterpret_cast<double**>(a)[i][offset];
                        break;
                    default:
                        // Handle unsupported key type
                        break;
                }
            }
            temp[offset] = sort_keys_type(&reinterpret_cast<uint64_t*>(result)[offset * num_keys], num_keys);
        }
    }
}

template <int B, int I>
__global__ void rows_to_columns(sort_keys_type *row_keys, void** col_keys, uint64_t N, uint64_t num_keys, int key_type) {
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
            for (int i = 0; i < num_keys; i++) {
                switch(key_type) {
                    case 0: // int32_t
                        reinterpret_cast<int32_t**>(col_keys)[i][offset] = static_cast<int32_t>(row_keys[offset].keys[i]);
                        break;
                    case 1: // float
                        reinterpret_cast<float**>(col_keys)[i][offset] = static_cast<float>(row_keys[offset].keys[i]);
                        break;
                    case 2: // uint64_t
                        reinterpret_cast<uint64_t**>(col_keys)[i][offset] = row_keys[offset].keys[i];
                        break;
                    case 3: // double
                        reinterpret_cast<double**>(col_keys)[i][offset] = static_cast<double>(row_keys[offset].keys[i]);
                        break;
                    default:
                        // Handle unsupported key type
                        break;
                }
            }
        }
    }
}

template <int B, int I>
__global__ void gather_and_modify(const void *a, void* result, sort_keys_type *sort_keys, uint64_t N, uint64_t num_keys, int key_type) {
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
            switch(key_type) {
                case 0: // int32_t
                    reinterpret_cast<int32_t*>(result)[offset] = reinterpret_cast<const int32_t*>(a)[items_ids];
                    break;
                case 1: // float
                    reinterpret_cast<float*>(result)[offset] = reinterpret_cast<const float*>(a)[items_ids];
                    break;
                case 2: // uint64_t
                    reinterpret_cast<uint64_t*>(result)[offset] = reinterpret_cast<const uint64_t*>(a)[items_ids];
                    break;
                case 3: // double
                    reinterpret_cast<double*>(result)[offset] = reinterpret_cast<const double*>(a)[items_ids];
                    break;
                default:
                    // Handle unsupported key type
                    break;
            }
            sort_keys[offset] = sort_keys_type(sort_keys[offset].keys, num_keys - 1);
        }
    }
}

template <int B, int I>
__global__ void modify(sort_keys_type *sort_keys, uint64_t N, uint64_t num_keys) {

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
            sort_keys[offset] = sort_keys_type(sort_keys[offset].keys, num_keys - 1);
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

template <int B, int I>
__global__ void divide(const void* a, const uint64_t* b, void* result, uint64_t N, int key_type) {
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
            switch(key_type) {
                case 0: // int32_t
                    reinterpret_cast<int32_t*>(result)[offset] = reinterpret_cast<const int32_t*>(a)[offset] / static_cast<int32_t>(b[offset]);
                    break;
                case 1: // float
                    reinterpret_cast<float*>(result)[offset] = reinterpret_cast<const float*>(a)[offset] / static_cast<float>(b[offset]);
                    break;
                case 2: // uint64_t
                    reinterpret_cast<uint64_t*>(result)[offset] = reinterpret_cast<const uint64_t*>(a)[offset] / b[offset];
                    break;
                case 3: // double
                    reinterpret_cast<double*>(result)[offset] = reinterpret_cast<const double*>(a)[offset] / static_cast<double>(b[offset]);
                    break;
                default:
                    break;
            }
        }
    }
}

template <int B, int I>
__global__ void fill_n(void* a, void* b, uint64_t N, int key_type) {
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
            switch(key_type) {
                case 0: // int32_t
                    reinterpret_cast<int32_t*>(a)[offset] = *reinterpret_cast<int32_t*>(b);
                    break;
                case 1: // float
                    reinterpret_cast<float*>(a)[offset] = *reinterpret_cast<float*>(b);
                    break;
                case 2: // uint64_t
                    reinterpret_cast<uint64_t*>(a)[offset] = *reinterpret_cast<uint64_t*>(b);
                    break;
                case 3: // double
                    reinterpret_cast<double*>(a)[offset] = *reinterpret_cast<double*>(b);
                    break;
                default:
                    break;
            }
        }
    }
}

// gather_and_modify
template
__global__ void gather_and_modify<BLOCK_THREADS, ITEMS_PER_THREAD>(const void *a, void* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys, int key_type);

// columns_to_rows
template
__global__ void columns_to_rows<BLOCK_THREADS, ITEMS_PER_THREAD>(void **a, void* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys, int key_type);

// rows_to_columns
template
__global__ void rows_to_columns<BLOCK_THREADS, ITEMS_PER_THREAD>(sort_keys_type *row_keys, void** col_keys, uint64_t N, uint64_t num_keys, int key_type);

template <typename V>
void groupedAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, int key_type) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }

    printf("Launching Grouped Aggregate Kernel\n");

    SETUP_TIMING();
    cudaEventRecord(start, 0);
    
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    // T* keys_row_id[num_keys + 1];

    void* row_keys;
    void** keys_row_id = new void*[num_keys + 1];
    void** keys_dev;
    size_t type_size;

    switch(key_type) {
        // int32_t
        case 0:
            type_size = sizeof(int32_t);
            row_keys = gpuBufferManager->customCudaMalloc<int32_t>((num_keys + 1) * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<int32_t*>(keys[i]);
            }
            break;
        // float
        case 1:
            type_size = sizeof(float);
            row_keys = gpuBufferManager->customCudaMalloc<float>((num_keys + 1) * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<float*>(keys[i]);
            }
            break;
        // uint64_t
        case 2:
            type_size = sizeof(uint64_t);
            row_keys = gpuBufferManager->customCudaMalloc<uint64_t>((num_keys + 1) * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<uint64_t*>(keys[i]);
            }
            break;
        // double
        case 3:
            type_size = sizeof(double);
            row_keys = gpuBufferManager->customCudaMalloc<double>((num_keys + 1) * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<double*>(keys[i]);
            }
            break;
        default:
            throw std::runtime_error("Unsupported key type");
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t* row_sequence = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    sequence<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(row_sequence, N);
    keys_row_id[num_keys] = row_sequence;

    cudaMalloc((void**) &keys_dev, (num_keys + 1) * sizeof(void*));
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + 1) * sizeof(void*), cudaMemcpyHostToDevice);

    columns_to_rows<BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>
                (reinterpret_cast<void**>(keys_dev), 
                 reinterpret_cast<void*>(row_keys),
                 materialized_temp, N, static_cast<uint64_t>(num_keys + 1), key_type);

    CHECK_ERROR();
    cudaDeviceSynchronize();

    //perform sort-based groupby
    // Determine temporary device storage requirements
    CustomLess custom_less;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
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

    // sort_keys_type* materialized_temp2 = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    // // Determine temporary device storage requirements
    // CustomLess custom_op;
    // void *d_temp_storage = nullptr;
    // size_t temp_storage_bytes = 0;
    // cub::DeviceRadixSort::SortKeys(
    //     d_temp_storage,
    //     temp_storage_bytes,
    //     materialized_temp,
    //     materialized_temp2,
    //     N);

    // CHECK_ERROR();

    // // Allocate temporary storage
    // d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // // Run sorting operation
    // cub::DeviceRadixSort::SortKeys(
    //     d_temp_storage,
    //     temp_storage_bytes,
    //     materialized_temp,
    //     materialized_temp2,
    //     N);

    // CHECK_ERROR();

    //gather the aggregates based on the row_sequence
    printf("Gathering Aggregates\n");
    V** aggregate_keys_temp = new V*[num_aggregates];
    uint64_t** aggregate_star_temp = new uint64_t*[num_aggregates];
    sort_keys_type* group_by_rows = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
    uint64_t* h_count = new uint64_t[1];

    size_t buffer_size = N * sizeof(uint64_t); // Adjust the type and size as needed
    uint64_t* temp_buffer;
    cudaMalloc(&temp_buffer, buffer_size);
    cudaMemset(temp_buffer, 0, buffer_size);

    for (int agg = 0; agg < num_aggregates; agg++) {
        // printf("Aggregating %d\n", agg);
        if (agg_mode[agg] == 4 || agg_mode[agg] == 5) { //count_star or count(null) or sum(null)
            aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            if (agg_mode[agg] == 4) 
                fill_n<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1) / tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], temp_buffer, N, key_type);
            else if (agg_mode[agg] == 5)
                cudaMemset(aggregate_star_temp[agg], 0, N * sizeof(uint64_t));

            modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(materialized_temp, N, num_keys + 1);

            //perform reduce_by_key
            uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));

            printf("Reduce by key count_star\n");
            // Determine temporary device storage requirements
            d_temp_storage = nullptr;
            temp_storage_bytes = 0;
            CustomSum custom_sum;
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

            printf("Count: %lu\n", count[0]);

            CHECK_ERROR();
            aggregate_keys[agg] = reinterpret_cast<uint8_t*> (agg_star_out);
        } else {
            aggregate_keys_temp[agg] = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            V* temp = reinterpret_cast<V*> (aggregate_keys[agg]);
            gather_and_modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(temp, aggregate_keys_temp[agg], materialized_temp, N, num_keys + 1, key_type);

            V* agg_out = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            cudaMemset(agg_out, 0, N * sizeof(V));

            CHECK_ERROR();
            if (agg_mode[agg] == 0) {
                printf("Reduce by key sum\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSum custom_sum;
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
            } else if (agg_mode[agg] == 1) {
                //Currently typename V has to be a double
                printf("Reduce by key avg\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSum custom_sum;
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
                fill_n<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1) / tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], temp_buffer, N, key_type);

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
                divide<BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(agg_out, agg_star_out, output, count[0], key_type);

                CHECK_ERROR();
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (output);
            } else if (agg_mode[agg] == 2) {
                printf("Reduce by key max\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMax custom_max;
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
                printf("Reduce by key min\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMin custom_min;
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

    cudaFree(temp_buffer);

    void** keys_dev_result;
    void** keys_result = new void*[num_keys];
    cudaMalloc((void**) &keys_dev_result, num_keys * sizeof(void*));

    for (uint64_t i = 0; i < num_keys; i++) {
        switch (key_type) {
            case 0: // int32_t
                keys_result[i] = gpuBufferManager->customCudaMalloc<int32_t>(count[0], 0, 0);
                break;
            case 1: // float
                keys_result[i] = gpuBufferManager->customCudaMalloc<float>(count[0], 0, 0);
                break;
            case 2: // uint64_t
                keys_result[i] = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
                break;
            case 3: // double
                keys_result[i] = gpuBufferManager->customCudaMalloc<double>(count[0], 0, 0);
                break;
            default:
                throw std::runtime_error("Unsupported key type");
        }
    }

    cudaMemcpy(keys_dev_result, keys_result, num_keys * sizeof(void*), cudaMemcpyHostToDevice);

    rows_to_columns<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>
        (group_by_rows, keys_dev_result, count[0], num_keys, key_type);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Count: %lu\n", count[0]);

    for (uint64_t i = 0; i < num_keys; i++) {
        keys[i] = reinterpret_cast<uint8_t*>(keys_result[i]);
    }

    STOP_TIMER();
}


template <typename T>
void groupedWithoutAggregate(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys, int key_type) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }
    printf("Launching Grouped Without Aggregate Kernel\n");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
        // Declare pointers for different types
    void* row_keys;
    void** keys_row_id = new void*[num_keys];
    void** keys_dev;
    size_t type_size;

    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*>(
        gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0)
    );

    // Determine type size and allocate appropriate memory based on key_type
    switch(key_type) {
        case 0: // int32_t
            type_size = sizeof(int32_t);
            row_keys = gpuBufferManager->customCudaMalloc<int32_t>(num_keys * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<int32_t*>(keys[i]);
            }
            break;
        case 1: // float
            type_size = sizeof(float);
            row_keys = gpuBufferManager->customCudaMalloc<float>(num_keys * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<float*>(keys[i]);
            }
            break;
        case 2: // uint64_t
            type_size = sizeof(uint64_t);
            row_keys = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<uint64_t*>(keys[i]);
            }
            break;
        case 3: // double
            type_size = sizeof(double);
            row_keys = gpuBufferManager->customCudaMalloc<double>(num_keys * N, 0, 0);
            for (uint64_t i = 0; i < num_keys; i++) {
                keys_row_id[i] = reinterpret_cast<double*>(keys[i]);
            }
            break;
        default:
            throw std::runtime_error("Unsupported key type");
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    cudaMalloc((void**) &keys_dev, num_keys * sizeof(void*));
    cudaMemcpy(keys_dev, keys_row_id, num_keys * sizeof(void*), cudaMemcpyHostToDevice);


    columns_to_rows<BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>
                (reinterpret_cast<void**>(keys_dev), 
                 reinterpret_cast<void*>(row_keys),
                 materialized_temp, N, num_keys, key_type);

    CHECK_ERROR();

    //perform sort-based groupby
    // Determine temporary device storage requirements
    CustomLess custom_less;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
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

    CHECK_ERROR();

    //gather the aggregates based on the row_sequence
    // printf("Gathering Aggregates\n");
    sort_keys_type* group_by_rows = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
    uint64_t* h_count = new uint64_t[1];

    uint64_t* aggregate_star_temp = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    cudaMemset(aggregate_star_temp, 0, N * sizeof(uint64_t));

    //perform reduce_by_key
    uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));

    // printf("Reduce by key count_star\n");
    // Determine temporary device storage requirements
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    CustomSum custom_sum;
    cub::DeviceReduce::ReduceByKey(
        d_temp_storage, temp_storage_bytes,
        materialized_temp, group_by_rows, aggregate_star_temp,
        agg_star_out, d_num_runs_out, custom_sum, N);

    CHECK_ERROR();

    // Allocate temporary storage
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run reduce-by-key
    cub::DeviceReduce::ReduceByKey(
        d_temp_storage, temp_storage_bytes,
        materialized_temp, group_by_rows, aggregate_star_temp,
        agg_star_out, d_num_runs_out, custom_sum, N);

    CHECK_ERROR();

    cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    count[0] = h_count[0];

    void** keys_dev_result;
    void** keys_result = new void*[num_keys];
    cudaMalloc((void**) &keys_dev_result, num_keys * sizeof(void*));
    for (uint64_t i = 0; i < num_keys; i++) {
        switch(key_type) {
            case 0:
                keys_result[i] = gpuBufferManager->customCudaMalloc<int32_t>(count[0], 0, 0);
                break;
            case 1:
                keys_result[i] = gpuBufferManager->customCudaMalloc<float>(count[0], 0, 0);
                break;
            case 2:
                keys_result[i] = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
                break;
            case 3:
                keys_result[i] = gpuBufferManager->customCudaMalloc<double>(count[0], 0, 0);
                break;
        }
    }
    cudaMemcpy(keys_dev_result, keys_result, num_keys * sizeof(void*), cudaMemcpyHostToDevice);

    rows_to_columns<BLOCK_THREADS, ITEMS_PER_THREAD>
                <<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>
                (group_by_rows, reinterpret_cast<void**>(keys_dev_result), count[0], num_keys, key_type);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Count: %lu\n", count[0]);

    for (uint64_t i = 0; i < num_keys; i++) {
        keys[i] = reinterpret_cast<uint8_t*> (keys_result[i]);
    }

    STOP_TIMER();
}

void combineColumns(void* a, void* b, void* c, uint64_t N_a, uint64_t N_b, int key_type) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Combine Columns Kernel\n");

    switch(key_type) {
        case 0: // int32_t
            cudaMemcpy(reinterpret_cast<int32_t*>(c), reinterpret_cast<int32_t*>(a), N_a * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(reinterpret_cast<int32_t*>(c) + N_a, reinterpret_cast<int32_t*>(b), N_b * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            break;
        case 1: // float
            cudaMemcpy(reinterpret_cast<float*>(c), reinterpret_cast<float*>(a), N_a * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(reinterpret_cast<float*>(c) + N_a, reinterpret_cast<float*>(b), N_b * sizeof(float), cudaMemcpyDeviceToDevice);
            break;
        case 2: // uint64_t
            cudaMemcpy(reinterpret_cast<uint64_t*>(c), reinterpret_cast<uint64_t*>(a), N_a * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(reinterpret_cast<uint64_t*>(c) + N_a, reinterpret_cast<uint64_t*>(b), N_b * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            break;
        case 3: // double
            cudaMemcpy(reinterpret_cast<double*>(c), reinterpret_cast<double*>(a), N_a * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(reinterpret_cast<double*>(c) + N_a, reinterpret_cast<double*>(b), N_b * sizeof(double), cudaMemcpyDeviceToDevice);
            break;
        default:
            throw std::runtime_error("Unsupported key type");
    }

    CHECK_ERROR();
    cudaDeviceSynchronize();
}

// void groupedAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, int key_type) {

template
void groupedAggregate<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, int key_type);

template
void groupedAggregate<float>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, int key_type);

template
void groupedAggregate<int32_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, int key_type);

template
void groupedAggregate<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, int key_type);


template
void groupedWithoutAggregate<uint64_t>(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys, int key_type);

template
void groupedWithoutAggregate<float>(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys, int key_type);

template
void groupedWithoutAggregate<int32_t>(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys, int key_type);

template
void groupedWithoutAggregate<double>(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys, int key_type);

// void combineColumns(void* a, void* b, void* c, uint64_t N_a, uint64_t N_b, int key_type) {

// For int32_t
// combineColumns((int32_t* a), (int32_t* b), (int32_t* c), uint64_t N_a, uint64_t N_b, 0);

// For float
// combineColumns(reinterpret_cast<void*>(float* a), reinterpret_cast<void*>(float* b), reinterpret_cast<void*>(float* c), uint64_t N_a, uint64_t N_b, 1);

// For uint64_t
// combineColumns(reinterpret_cast<void*>(uint64_t* a), reinterpret_cast<void*>(uint64_t* b), reinterpret_cast<void*>(uint64_t* c), uint64_t N_a, uint64_t N_b, 2);

// For double
// combineColumns(reinterpret_cast<void*>(double* a), reinterpret_cast<void*>(double* b), reinterpret_cast<void*>(double* c), uint64_t N_a, uint64_t N_b, 3);

// void combineColumns<uint64_t>(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t N_a, uint64_t N_b);

// void combineColumns<double>(double* a, double* b, double* c, uint64_t N_a, uint64_t N_b);

}