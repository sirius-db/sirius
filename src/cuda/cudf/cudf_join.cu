#include "cudf/cudf_utils.hpp"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"
namespace duckdb {

void cudf_probe(void **probe_keys, cudf::hash_join* hash_table, uint64_t N, int num_keys, int32_t*& row_ids_left, int32_t*& row_ids_right, uint64_t*& count)
{
    if (N == 0) {
        printf("N is 0\n");
        return;
    }

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

    if (N == 0) {
        printf("N is 0\n");
        return;
    }

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

void cudf_conditional_join(GPUColumn** build_columns, GPUColumn** probe_columns, vector<JoinCondition> &conditions, JoinType join_type, int32_t*& row_ids_left, int32_t*& row_ids_right, uint64_t*& count) {
    
    if (build_columns[0]->column_length == 0 || probe_columns[0]->column_length == 0) {
        printf("N is 0\n");
        return;
    }

    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);

    std::vector<cudf::column_view> build_keys_cudf;
    std::vector<cudf::column_view> probe_keys_cudf;

    for (int cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
        build_keys_cudf.push_back(build_columns[cond_idx]->convertToCudfColumn());
        probe_keys_cudf.push_back(probe_columns[cond_idx]->convertToCudfColumn());
    }

    auto build_table = cudf::table_view(build_keys_cudf);
    auto probe_table = cudf::table_view(probe_keys_cudf);

    // for (int cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    //     if (conditions[cond_idx].comparison == ExpressionType::COMPARE_EQUAL || conditions[cond_idx].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {


    // auto result = cudf::conditional_inner_join(probe_table, build_table, conditions);

    // row_ids_left = (int32_t*)result.first->data();
    // row_ids_right = (int32_t*)result.second->data();
    // count = new uint64_t[1];
    // count[0] = result.first->size();
}

} //namespace duckdb