#include "cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_physical_order.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

void cudf_orderby(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& projection, uint64_t num_keys, uint64_t num_projections, OrderByType* order_by_type) 
{
    if (keys[0]->column_length == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        for (idx_t col = 0; col < num_projections; col++) {
            bool old_unique = projection[col]->is_unique;
            if (projection[col]->data_wrapper.type == ColumnType::VARCHAR) {
                projection[col] = make_shared_ptr<GPUColumn>(0, projection[col]->data_wrapper.type, projection[col]->data_wrapper.data, projection[col]->data_wrapper.offset, 0, true);
            } else {
                projection[col] = make_shared_ptr<GPUColumn>(0, projection[col]->data_wrapper.type, projection[col]->data_wrapper.data);
            }
            projection[col]->is_unique = old_unique;
        }
        return;
    }

    SIRIUS_LOG_DEBUG("CUDF Order By");
    SIRIUS_LOG_DEBUG("Input size: {}", keys[0]->column_length);
    SETUP_TIMING();
    START_TIMER();

    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);

    // uint64_t* a = new uint64_t[25] {0, 0, 0, 3, 1, 3, 2, 1, 1, 0, 1, 4, 3, 4, 2, 2, 3, 3, 2, 0, 4, 4, 2, 1, 4};
    // uint64_t* b = new uint64_t[25] {16, 15, 5, 6, 1, 22, 8, 2, 17, 14, 3, 4, 7, 11, 21, 9, 23, 19, 18, 0, 13, 10, 12, 24, 20};
    // a: 1 a: 0 a: 8 a: 18 a: 24 a: 7 a: 3 a: 6 a: 22 a: 9 a: 21 a: 13 a: 17 a: 14 a: 5 a: 4 a: 11 a: 2 a: 15 a: 16 a: 19 a: 10 a: 12 a: 20 a: 23 

    // uint64_t* a_gpu = reinterpret_cast<uint64_t*>(gpuBufferManager->mr->allocate(25 * sizeof(uint64_t)));
    // uint64_t* b_gpu = reinterpret_cast<uint64_t*>(gpuBufferManager->mr->allocate(25 * sizeof(uint64_t)));
    // uint64_t* a_gpu; uint64_t* b_gpu;
    // cudaMalloc(&a_gpu, 25 * sizeof(uint64_t));
    // cudaMalloc(&b_gpu, 25 * sizeof(uint64_t));

    // cudaMemcpy(a_gpu, keys[0]->data_wrapper.data, 25 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(b_gpu, keys[1]->data_wrapper.data, 25 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // cudaMemcpy(a_gpu, a, 25 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(b_gpu, b, 25 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // std::vector<cudf::column_view> columns_cudf;
    // cudf::size_type size = 25;
    // auto cudf_column_a = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), 25, reinterpret_cast<void*>(a_gpu), nullptr, 0);
    // columns_cudf.push_back(cudf_column_a);

    // auto cudf_column_b = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), 25, reinterpret_cast<void*>(b_gpu), nullptr, 0);
    // columns_cudf.push_back(cudf_column_b);


    std::vector<cudf::column_view> columns_cudf;
    for (int key = 0; key < num_keys; key++) {
        auto cudf_column_view = keys[key]->convertToCudfColumn();
        // auto cudf_column_view = cudf::column_view(cudf::data_type(cudf::type_id::UINT64), keys[key]->column_length, reinterpret_cast<void*>(keys[key]->data_wrapper.data), nullptr, 0);
        // auto cudf_column = cudf::column(cudf_column_view);
        columns_cudf.push_back(cudf_column_view);
    }

    std::vector<cudf::order> orders;
    for (int i = 0; i < num_keys; i++) {
        if (order_by_type[i] == OrderByType::ASCENDING) {
            orders.push_back(cudf::order::ASCENDING);
        } else {
            orders.push_back(cudf::order::DESCENDING);
        }
    }

    //copy the projection columns to a new array
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

    auto keys_table = cudf::table_view(columns_cudf);

    // for (int col = 0; col < num_keys; col++) {
    //     int size = keys[col]->column_length;
    //     printGPUColumn<uint64_t>(reinterpret_cast<uint64_t*>(keys[col]->data_wrapper.data), size, 0);
    // }

    // printGPUColumn<uint64_t>(a_gpu, 25, 0);
    // printGPUColumn<uint64_t>(b_gpu, 25, 0);

    // SIRIUS_LOG_DEBUG("Sorting keys");
    auto sorted_order = cudf::sorted_order(keys_table, orders);
    auto sorted_order_view = sorted_order->view();
    // SIRIUS_LOG_DEBUG("keys table num columns: {}", keys_table.num_columns());
    // SIRIUS_LOG_DEBUG("orders size: {}", orders.size());
    // auto sorted_table = cudf::stable_sort(keys_table, orders);
    // auto sorted_table_view = sorted_table->view();

    // for (int col = 0; col < num_keys; col++) {
    //     int size = keys[col]->column_length;
    //     printGPUColumn<uint64_t>(reinterpret_cast<uint64_t*>(keys[col]->data_wrapper.data), size, 0);
    // }

    // printGPUColumn<uint64_t>(a_gpu, 25, 0);
    // printGPUColumn<uint64_t>(b_gpu, 25, 0);

    // int size = sorted_order_view.size();
    // int* data = const_cast<int*>(sorted_order_view.data<int>());
    // printGPUColumn<int>(data, size, 0);

    std::vector<cudf::column_view> projection_cudf;
    for (int col = 0; col < num_projections; col++) {
        auto cudf_column = projection[col]->convertToCudfColumn();
        projection_cudf.push_back(cudf_column);
    }
    auto projection_table = cudf::table_view(projection_cudf);

    auto gathered_table = cudf::gather(projection_table, sorted_order_view);

    for (int col = 0; col < num_projections; col++) {
        auto sorted_column = gathered_table->get_column(col);
        projection[col]->setFromCudfColumn(sorted_column, projection[col]->is_unique, nullptr, 0, gpuBufferManager);
        // projection[col] = gpuBufferManager->copyDataFromcuDFColumn(sorted_column, 0);
    }

    SIRIUS_LOG_DEBUG("Order by done");

    STOP_TIMER();
    // throw NotImplementedException("Order by is not implemented");


}

} //namespace duckdb