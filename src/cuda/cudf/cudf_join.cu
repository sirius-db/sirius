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

void cudf_probe(void **probe_keys, cudf::hash_join* hash_table, uint64_t N, int num_keys, int32_t*& row_ids_left, int32_t*& row_ids_right, uint64_t*& count)
{
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);
    printf("CUDF probe\n");

    std::vector<cudf::column_view> probe_keys_cudf;

    for (int key = 0; key < num_keys; key++) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT64), N, (void*)(probe_keys[key]), nullptr, 0);
        probe_keys_cudf.push_back(column);
    }

    auto probe_table = cudf::table_view(probe_keys_cudf);

    auto result = hash_table->inner_join(probe_table);
    row_ids_left = (int32_t*)result.first->data();
    row_ids_right = (int32_t*)result.second->data();
    count = new uint64_t[1];
    count[0] = result.first->size();
}

void cudf_build(void **build_keys, cudf::hash_join*& hash_table, uint64_t N, int num_keys) {

    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);

    std::vector<cudf::column_view> build_keys_cudf;

    for (int key = 0; key < num_keys; key++) {
        auto column = cudf::column_view(cudf::data_type(cudf::type_id::INT64), N, (void*)(build_keys[key]), nullptr, 0);
        build_keys_cudf.push_back(column);
    }

    auto build_table = cudf::table_view(build_keys_cudf);
    hash_table = new cudf::hash_join(build_table, cudf::null_equality::EQUAL);
}

void cudf_join(GPUColumn** build_columns, GPUColumn** probe_columns, vector<JoinCondition> &conditions, JoinType join_type) {
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);

    // std::vector<cudf::column_view> build_keys_cudf;
    // std::vector<cudf::column_view> probe_keys_cudf;

    for (int i = 0; i < conditions.size(); i++) {
        
    }
    
}

} //namespace duckdb