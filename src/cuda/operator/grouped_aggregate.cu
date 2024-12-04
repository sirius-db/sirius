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

template <typename T, int B, int I>
__global__ void columns_to_rows(T **a, T* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys) {

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
                result[offset * num_keys + i] = a[i][offset];
            }
            temp[offset] = sort_keys_type(&result[offset * num_keys], num_keys);
        }
    }
}

template <typename T, int B, int I>
__global__ void rows_to_columns(sort_keys_type *row_keys, T** col_keys, uint64_t N, uint64_t num_keys) {

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
                col_keys[i][offset] = row_keys[offset].keys[i];
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void gather_and_modify(const T *a, T* result, sort_keys_type *sort_keys, uint64_t N, uint64_t num_keys) {

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

__global__ void test(sort_keys_type* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 15; i++) {
            for (uint64_t j = 0; j < a[i].num_key + 1; j++) {
                printf("%ld ", a[i].keys[j]);
            }
        }
        printf("\n");
    }
}

template <typename T>
__global__ void testprint(T* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 100; i++) {
            printf("%.2f ", a[i]);
        }
        printf("\n");
    }
}

__global__ void test3(uint64_t* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 15; i++) {
            printf("%ld ", a[i]);
        }
        printf("\n");
    }
}

__global__ void test2(double* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 15; i++) {
            printf("%.2f ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);
template
__global__ void gather_and_modify<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);

template
__global__ void columns_to_rows<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **a, uint64_t* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);

template
__global__ void rows_to_columns<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sort_keys_type *row_keys, uint64_t** col_keys, uint64_t N, uint64_t num_keys);

template <typename T, typename V>
void groupedAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }
    printf("Launching Grouped Aggregate Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
    T* row_keys = gpuBufferManager->customCudaMalloc<T>((num_keys + 1) * N, 0, 0);
    T* keys_temp = gpuBufferManager->customCudaMalloc<T>((num_keys + 1) * N, 0, 0);
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    // T* keys_row_id[num_keys + 1];
    T** keys_row_id = new T*[num_keys + num_aggregates];
    for (uint64_t i = 0; i < num_keys; i++) {
        cudaMemcpy(keys_temp + i * N, reinterpret_cast<T*> (keys[i]), N * sizeof(T), cudaMemcpyDeviceToDevice);
        keys_row_id[i] = keys_temp + i * N;
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t* row_sequence = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    sequence<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(row_sequence, N);
    keys_row_id[num_keys] = row_sequence;

    T** keys_dev;
    cudaMalloc((void**) &keys_dev, (num_keys + 1) * sizeof(T*));
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + 1) * sizeof(T*), cudaMemcpyHostToDevice);

    columns_to_rows<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, materialized_temp, N, num_keys + 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    //perform sort-based groupby
    thrust::device_vector<sort_keys_type> sorted_keys(materialized_temp, materialized_temp + N);
    thrust::sort(thrust::device, sorted_keys.begin(), sorted_keys.end());
    CHECK_ERROR();

    //gather the aggregates based on the row_sequence
    printf("Gathering Aggregates\n");
    sort_keys_type* gather_idx = thrust::raw_pointer_cast(sorted_keys.data());
    V** aggregate_keys_temp = new V*[num_aggregates];
    uint64_t* aggregate_star_temp[num_aggregates];
    thrust::device_vector<sort_keys_type> group_by_rows(N);

    for (int agg = 0; agg < num_aggregates; agg++) {
        // printf("Aggregating %d\n", agg);
        if (agg_mode[agg] == 4 || agg_mode[agg] == 5) { //count_star or count(null) or sum(null)
            thrust::device_vector<uint64_t> aggregate_star_column(N);
            if (agg_mode[agg] == 4) thrust::fill_n(aggregate_star_column.begin(), aggregate_star_column.size(), 1);
            else if (agg_mode[agg] == 5) thrust::fill_n(aggregate_star_column.begin(), aggregate_star_column.size(), 0);

            modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(gather_idx, N, num_keys + 1);

            //perform reduce_by_key
            thrust::device_vector<uint64_t> agg_star_out(N);
            uint64_t* raw_agg_out = thrust::raw_pointer_cast(agg_star_out.data());
            // cudaMemset(raw_agg_out, 0, N * sizeof(uint64_t));

            thrust::equal_to<sort_keys_type> binary_pred; thrust::plus<uint64_t> binary_op;

            printf("Reduce by key count_star\n");
            auto reduce_result = thrust::reduce_by_key(thrust::device, 
                                        sorted_keys.begin(), 
                                        sorted_keys.end(),
                                        aggregate_star_column.begin(), 
                                        group_by_rows.begin(), 
                                        agg_star_out.begin(), binary_pred, binary_op);

            count[0] = reduce_result.second - agg_star_out.begin();
            aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);

            printf("Count: %lu\n", count[0]);

            CHECK_ERROR();
            cudaMemcpy(aggregate_star_temp[agg], raw_agg_out, count[0] * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            aggregate_keys[agg] = reinterpret_cast<uint8_t*> (aggregate_star_temp[agg]);
        } else {
            aggregate_keys_temp[agg] = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            V* temp = reinterpret_cast<V*> (aggregate_keys[agg]);
            gather_and_modify<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(temp, aggregate_keys_temp[agg], gather_idx, N, num_keys + 1);
            thrust::device_vector<V> aggregate_columns(aggregate_keys_temp[agg], aggregate_keys_temp[agg] + N);

            //perform reduce_by_key
            thrust::device_vector<V> agg_out(N);
            V* raw_agg_out = thrust::raw_pointer_cast(agg_out.data());
            // cudaMemset(raw_agg_out, 0, N * sizeof(V));

            CHECK_ERROR();
            if (agg_mode[agg] == 0) {
                thrust::equal_to<sort_keys_type> binary_pred; thrust::plus<V> binary_op;
                printf("Reduce by key sum\n");
                auto reduce_result = thrust::reduce_by_key(thrust::device, 
                                            sorted_keys.begin(), 
                                            sorted_keys.end(),
                                            aggregate_columns.begin(), 
                                            group_by_rows.begin(), 
                                            agg_out.begin(), binary_pred, binary_op);

                count[0] = reduce_result.second - agg_out.begin();

                CHECK_ERROR();
                cudaMemcpy(aggregate_keys_temp[agg], raw_agg_out, count[0] * sizeof(V), cudaMemcpyDeviceToDevice);
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (aggregate_keys_temp[agg]);
            } else if (agg_mode[agg] == 1) {
                //Currently typename V has to be a double
                thrust::equal_to<sort_keys_type> binary_pred; thrust::plus<V> binary_op;
                printf("Reduce by key avg\n");
                auto reduce_result = thrust::reduce_by_key(thrust::device, 
                                            sorted_keys.begin(), 
                                            sorted_keys.end(),
                                            aggregate_columns.begin(), 
                                            group_by_rows.begin(), 
                                            agg_out.begin(), binary_pred, binary_op);

                count[0] = reduce_result.second - agg_out.begin();

                thrust::device_vector<uint64_t> aggregate_star_column(N);
                thrust::fill_n(aggregate_star_column.begin(), aggregate_star_column.size(), 1);
                thrust::device_vector<uint64_t> agg_star_out(N);
                uint64_t* raw_agg_star_out = thrust::raw_pointer_cast(agg_star_out.data());

                thrust::plus<uint64_t> binary_op_star;
                auto reduce_result_star = thrust::reduce_by_key(thrust::device, 
                                            sorted_keys.begin(), 
                                            sorted_keys.end(),
                                            aggregate_star_column.begin(), 
                                            group_by_rows.begin(), 
                                            agg_star_out.begin(), binary_pred, binary_op_star);

                V* output = gpuBufferManager->customCudaMalloc<V>(count[0], 0, 0);
                divide<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(raw_agg_out, raw_agg_star_out, output, count[0]);

                CHECK_ERROR();
                cudaMemcpy(aggregate_keys_temp[agg], output, count[0] * sizeof(V), cudaMemcpyDeviceToDevice);
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (aggregate_keys_temp[agg]);
            } else if (agg_mode[agg] == 2) {
                thrust::equal_to<sort_keys_type> binary_pred; thrust::maximum<V> binary_op;
                printf("Reduce by key max\n");
                auto reduce_result = thrust::reduce_by_key(thrust::device, 
                                            sorted_keys.begin(), 
                                            sorted_keys.end(),
                                            aggregate_columns.begin(), 
                                            group_by_rows.begin(), 
                                            agg_out.begin(), binary_pred, binary_op);

                count[0] = reduce_result.second - agg_out.begin();

                CHECK_ERROR();
                cudaMemcpy(aggregate_keys_temp[agg], raw_agg_out, count[0] * sizeof(V), cudaMemcpyDeviceToDevice);
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (aggregate_keys_temp[agg]);
            } else if (agg_mode[agg] == 3) {
                thrust::equal_to<sort_keys_type> binary_pred; thrust::minimum<V> binary_op;
                printf("Reduce by key min\n");
                auto reduce_result = thrust::reduce_by_key(thrust::device, 
                                            sorted_keys.begin(), 
                                            sorted_keys.end(),
                                            aggregate_columns.begin(), 
                                            group_by_rows.begin(), 
                                            agg_out.begin(), binary_pred, binary_op);

                count[0] = reduce_result.second - agg_out.begin();

                CHECK_ERROR();
                cudaMemcpy(aggregate_keys_temp[agg], raw_agg_out, count[0] * sizeof(V), cudaMemcpyDeviceToDevice);
                aggregate_keys[agg] = reinterpret_cast<uint8_t*> (aggregate_keys_temp[agg]);
            }
        }
    }
    sort_keys_type* raw_sorted_keys = thrust::raw_pointer_cast(group_by_rows.data());
    rows_to_columns<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(raw_sorted_keys, keys_dev, count[0], num_keys);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Count: %lu\n", count[0]);

    for (uint64_t i = 0; i < num_keys; i++) {
        T* temp = (keys_temp + i * N);
        keys[i] = reinterpret_cast<uint8_t*> (temp);
    }
}


template <typename T>
void groupedWithoutAggregate(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }
    printf("Launching Grouped Aggregate Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
    T* row_keys = gpuBufferManager->customCudaMalloc<T>(num_keys * N, 0, 0);
    T* keys_temp = gpuBufferManager->customCudaMalloc<T>(num_keys * N, 0, 0);
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    T* keys_row_id[num_keys];
    for (uint64_t i = 0; i < num_keys; i++) {
        cudaMemcpy(keys_temp + i * N, reinterpret_cast<T*> (keys[i]), N * sizeof(T), cudaMemcpyDeviceToDevice);
        keys_row_id[i] = keys_temp + i * N;
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    T** keys_dev;
    cudaMalloc((void**) &keys_dev, num_keys * sizeof(T*));
    cudaMemcpy(keys_dev, keys_row_id, num_keys * sizeof(T*), cudaMemcpyHostToDevice);

    columns_to_rows<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, materialized_temp, N, num_keys);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    //perform sort-based groupby
    thrust::device_vector<sort_keys_type> sorted_keys(materialized_temp, materialized_temp + N);
    thrust::sort(thrust::device, sorted_keys.begin(), sorted_keys.end());
    CHECK_ERROR();

    thrust::device_vector<sort_keys_type> group_by_rows(N);

    thrust::device_vector<uint64_t> aggregate_columns(N);
    thrust::fill_n(aggregate_columns.begin(), aggregate_columns.size(), 1);

    //perform reduce_by_key
    thrust::device_vector<uint64_t> agg_out(N);
    thrust::equal_to<sort_keys_type> binary_pred; thrust::plus<uint64_t> binary_op;

    CHECK_ERROR();
    auto reduce_result = thrust::reduce_by_key(thrust::device, 
                                sorted_keys.begin(), 
                                sorted_keys.end(),
                                aggregate_columns.begin(), 
                                group_by_rows.begin(), 
                                agg_out.begin(), binary_pred, binary_op);

    count[0] = reduce_result.second - agg_out.begin();
    CHECK_ERROR();

    sort_keys_type* raw_sorted_keys = thrust::raw_pointer_cast(group_by_rows.data());
    rows_to_columns<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(raw_sorted_keys, keys_dev, count[0], num_keys);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Count: %lu\n", count[0]);

    for (uint64_t i = 0; i < num_keys; i++) {
        T* temp = (keys_temp + i * N);
        keys[i] = reinterpret_cast<uint8_t*> (temp);
    }
}

template<typename T>
void combineColumns(T* a, T* b, T* c, uint64_t N_a, uint64_t N_b) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        printf("N is 0\n");
        return;
    }
    cudaMemcpy(c, a, N_a * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(c + N_a, b, N_b * sizeof(T), cudaMemcpyDeviceToDevice);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template
void groupedAggregate<uint64_t, uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedAggregate<uint64_t, double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedWithoutAggregate<uint64_t>(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys);

template
void combineColumns<uint64_t>(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t N_a, uint64_t N_b);

template
void combineColumns<double>(double* a, double* b, double* c, uint64_t N_a, uint64_t N_b);

}