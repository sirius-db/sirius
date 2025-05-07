#include <cudf/table/table.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/round.hpp>
#include <cudf/unary.hpp>

template <typename T>
__global__ void print_gpu_column(T* a, int32_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < N; i++) {
            printf("a: %d ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void print_gpu_column<uint64_t>(uint64_t* a, int32_t N);
template
__global__ void print_gpu_column<double>(double* a, int32_t N);
template
__global__ void print_gpu_column<int>(int* a, int32_t N);
template
__global__ void print_gpu_column<float>(float* a, int32_t N);
template
__global__ void print_gpu_column<uint8_t>(uint8_t* a, int32_t N);

template <typename T> 
void printGPUColumn(T* a, int32_t N, int gpu) {
    // CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    T* result_host_temp = new T[1];
    cudaMemcpy(result_host_temp, a, sizeof(T), cudaMemcpyDeviceToHost);
    // CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Result: %d and N: %d\n", result_host_temp[0], N);
    printf("N: %ld\n", N);
    print_gpu_column<T><<<1, 1>>>(a, N);
    // CHECK_ERROR();
    cudaDeviceSynchronize();
}

template void printGPUColumn<uint64_t>(uint64_t* a, int32_t N, int gpu);
template void printGPUColumn<double>(double* a, int32_t N, int gpu);
template void printGPUColumn<int>(int* a, int32_t N, int gpu);
template void printGPUColumn<float>(float* a, int32_t N, int gpu);

int main() {

    // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
    // This is the default memory resource for libcudf for allocating device memory.
    rmm::mr::cuda_memory_resource cuda_mr{};
    // Construct a memory pool using the CUDA memory resource
    // Using a memory pool for device memory allocations is important for good performance in libcudf.
    // The pool defaults to allocating half of the available GPU memory.
    rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(10)};

    // Set the pool resource to be used by default for all device memory allocations
    // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
    // it being set as the default
    // Also, call this before the first libcudf API call to ensure all data is allocated by the same
    // memory resource.
    cudf::set_current_device_resource(&mr);

    int num_keys = 2;

    uint64_t* a = new uint64_t[25] {0, 0, 3, 1, 3, 2, 1, 1, 0, 1, 4, 3, 4, 2, 2, 3, 3, 2, 0, 4, 4, 2, 1, 4};
    uint64_t* b = new uint64_t[25] {16, 15, 5, 6, 1, 22, 8, 2, 17, 14, 3, 4, 7, 11, 21, 9, 23, 19, 18, 0, 13, 10, 12, 24, 20};

    uint64_t* a_gpu;
    uint64_t* b_gpu;
    cudaMalloc(&a_gpu, 25 * sizeof(uint64_t));
    cudaMalloc(&b_gpu, 25 * sizeof(uint64_t));
    cudaMemcpy(a_gpu, a, 25 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, 25 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::vector<cudf::column_view> columns_cudf;
    cudf::size_type size = 25;
    auto cudf_column_a = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), 25, reinterpret_cast<void*>(a_gpu), nullptr, 0);
    columns_cudf.push_back(cudf_column_a);

    auto cudf_column_b = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), 25, reinterpret_cast<void*>(b_gpu), nullptr, 0);
    columns_cudf.push_back(cudf_column_b);

    std::vector<cudf::order> orders;
    orders.push_back(cudf::order::ASCENDING);
    orders.push_back(cudf::order::ASCENDING);

    // //copy the projection columns to a new array
    // GPUColumn** projection_columns = new GPUColumn*[num_projections];
    // for (int projection_idx = 0; projection_idx < num_projections; projection_idx++) {
    //     if (projection[projection_idx]->data_wrapper.type == ColumnType::VARCHAR) {
    //         uint64_t* temp_offset = gpuBufferManager->customCudaMalloc<uint64_t>(projection[projection_idx]->column_length, 0, false);
    //         uint8_t* temp_column = gpuBufferManager->customCudaMalloc<uint8_t>(projection[projection_idx]->data_wrapper.num_bytes, 0, false);
    //         callCudaMemcpyDeviceToDevice<uint64_t>(temp_offset, projection[projection_idx]->data_wrapper.offset, projection[projection_idx]->column_length, 0);
    //         callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, projection[projection_idx]->data_wrapper.data, projection[projection_idx]->data_wrapper.num_bytes, 0);
    //         projection_columns[projection_idx] = new GPUColumn(projection[projection_idx]->column_length, projection[projection_idx]->data_wrapper.type, temp_column, temp_offset, projection[projection_idx]->data_wrapper.num_bytes, true);
    //     } else {
    //         uint8_t* temp_column = gpuBufferManager->customCudaMalloc<uint8_t>(projection[projection_idx]->data_wrapper.num_bytes, 0, false);
    //         callCudaMemcpyDeviceToDevice<uint8_t>(temp_column, projection[projection_idx]->data_wrapper.data, projection[projection_idx]->data_wrapper.num_bytes, 0);
    //         projection_columns[projection_idx] = new GPUColumn(projection[projection_idx]->column_length, projection[projection_idx]->data_wrapper.type, temp_column);
    //     }
    // }

    printf("Creating keys table\n");
    auto keys_table = cudf::table_view(columns_cudf);

    printGPUColumn<uint64_t>(const_cast<uint64_t*>(cudf_column_a.data<uint64_t>()), size, 0);
    printGPUColumn<uint64_t>(const_cast<uint64_t*>(cudf_column_b.data<uint64_t>()), size, 0);

    // printf("Sorting keys\n");
    printf("keys table size: %ld\n", keys_table.num_columns());
    printf("orders size: %ld\n", orders.size());
    auto sorted_order = cudf::sorted_order(keys_table, orders);
    auto sorted_order_view = sorted_order->view();

    // printf("keys table num columns: %ld\n", keys_table.num_columns());
    // printf("orders size: %ld\n", orders.size());
    // auto sorted_table = cudf::stable_sort(keys_table, orders);
    // auto sorted_table_view = sorted_table->view();

    printGPUColumn<uint64_t>(const_cast<uint64_t*>(cudf_column_a.data<uint64_t>()), size, 0);
    printGPUColumn<uint64_t>(const_cast<uint64_t*>(cudf_column_b.data<uint64_t>()), size, 0);

    int size = sorted_order_view.size();
    int* data = const_cast<int*>(sorted_order_view.data<int>());
    printGPUColumn<int>(data, size, 0);

    // std::vector<cudf::column_view> projection_cudf;
    // for (int col = 0; col < num_projections; col++) {
    //     auto cudf_column = projection_columns[col]->convertToCudfColumn();
    //     projection_cudf.push_back(cudf_column);
    // }
    // auto projection_table = cudf::table_view(projection_cudf);

    // printf("Gathering projection table\n");
    // auto gathered_table = cudf::gather(projection_table, sorted_order_view);
    // auto gathered_table_view = gathered_table->view();

    printf("Done\n");

    return 0;
}
