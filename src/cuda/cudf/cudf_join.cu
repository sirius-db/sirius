#include <cudf/table/table.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"
namespace duckdb {

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>, std::unique_ptr<rmm::device_uvector<cudf::size_type>>> 
    cudf_probe(void **probe_keys, cudf::hash_join* hash_table, uint64_t N, int num_keys)
{
    // // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
    // // This is the default memory resource for libcudf for allocating device memory.
    // rmm::mr::cuda_memory_resource cuda_mr{};
    // // Construct a memory pool using the CUDA memory resource
    // // Using a memory pool for device memory allocations is important for good performance in libcudf.
    // // The pool defaults to allocating half of the available GPU memory.
    // rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(10)};

    // // Set the pool resource to be used by default for all device memory allocations
    // // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
    // // it being set as the default
    // // Also, call this before the first libcudf API call to ensure all data is allocated by the same
    // // memory resource.
    // cudf::set_current_device_resource(&mr);
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);
    printf("CUDF probe\n");

    std::vector<cudf::column_view> probe_keys_cudf;

    for (int key = 0; key < num_keys; key++) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT64), N, (void*)(probe_keys[key]), nullptr, 0);
        probe_keys_cudf.push_back(column);
    }

    auto probe_table = cudf::table_view(probe_keys_cudf);


    // std::vector<cudf::column_view> build_keys_cudf;

    // for (int key = 0; key < num_keys; key++) {
    //     auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT64), N, (void*)(probe_keys[key]), nullptr, 0);
    //     build_keys_cudf.push_back(column);
    // }

    // auto build_table = cudf::table_view(build_keys_cudf);
    // auto ht = cudf::hash_join(build_table, cudf::null_equality::EQUAL);

    // auto size = ht.inner_join_size(probe_table);
    // printf("%ld\n", size);
    // auto result = ht.inner_join(probe_table);
    // printf("CUDF probe\n");
    auto result = hash_table->inner_join(probe_table);
    // auto result_left = result.first->element_ptr(0);
    // auto result_right = result.second->element_ptr(0);

    // row_ids_left = (uint64_t*)result_left;
    // row_ids_right = (uint64_t*)result_right;

    // auto result_left = result.first->data();
    // auto result_right = result.second->data();
    // uint64_t* count = new uint64_t[1];
    // count[0] = result.first->size();
    // printf("count[0] = %ld\n", count[0]);

    // uint32_t* row_ids_left = (uint32_t*)result_left;
    // uint32_t* row_ids_right = (uint32_t*)result_right;

    // uint32_t* row_ids_left_copy = new uint32_t[count[0]];
    // uint32_t* row_ids_right_copy = new uint32_t[count[0]];
    // cudaMemcpy(row_ids_left_copy, row_ids_left, count[0] * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(row_ids_right_copy, row_ids_right, count[0] * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // for (idx_t i = 0; i < count[0]; i++) {
    //     printf("row_ids_left[%ld] = %ld, row_ids_right[%ld] = %ld\n", i, row_ids_left_copy[i], i, row_ids_right_copy[i]);
    // }
    return result;
    
}

void cudf_build(void **build_keys, cudf::hash_join*& hash_table, uint64_t N, int num_keys) {

    // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
    // This is the default memory resource for libcudf for allocating device memory.
    // rmm::mr::cuda_memory_resource cuda_mr{};
    // Construct a memory pool using the CUDA memory resource
    // Using a memory pool for device memory allocations is important for good performance in libcudf.
    // The pool defaults to allocating half of the available GPU memory.
    // rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(10)};

    // Set the pool resource to be used by default for all device memory allocations
    // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
    // it being set as the default
    // Also, call this before the first libcudf API call to ensure all data is allocated by the same
    // memory resource.
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);

    std::vector<cudf::column_view> build_keys_cudf;
    printf("CUDF build\n");

    for (int key = 0; key < num_keys; key++) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT64), N, (void*)(build_keys[key]), nullptr, 0);
        build_keys_cudf.push_back(column);
    }
    printf("CUDF build\n");


    auto build_table = cudf::table_view(build_keys_cudf);
    printf("CUDF build\n");
    hash_table = new cudf::hash_join(build_table, cudf::null_equality::EQUAL);
    printf("CUDF build\n");
}

} //namespace duckdb