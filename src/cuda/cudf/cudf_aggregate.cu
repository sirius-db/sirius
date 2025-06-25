/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cudf/cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_physical_ungrouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

template <cudf::reduce_aggregation::Kind kind>
static std::unique_ptr<cudf::reduce_aggregation> make_reduce_aggregation()
{
  switch (kind) {
    case cudf::reduce_aggregation::MIN:
      return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MAX:
      return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MEAN:
      return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::SUM:
      return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::NUNIQUE:
      return cudf::make_nunique_aggregation<cudf::reduce_aggregation>();
    default:
      throw NotImplementedException("Unsupported reduce aggregation");
  }
}

void cudf_aggregate(vector<shared_ptr<GPUColumn>>& column, uint64_t num_aggregates, AggregationType* agg_mode) 
{
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);
    if (column[0]->column_length == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        for (int agg_idx = 0; agg_idx < num_aggregates; agg_idx++) {
            if (agg_mode[agg_idx] == AggregationType::COUNT_STAR || agg_mode[agg_idx] == AggregationType::COUNT) {
                uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                cudaMemset(temp, 0, sizeof(uint64_t));
                column[agg_idx] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp));
            } else {
                column[agg_idx] = make_shared_ptr<GPUColumn>(0, column[agg_idx]->data_wrapper.type, column[agg_idx]->data_wrapper.data);
            }
        }
        return;
    }

    SIRIUS_LOG_DEBUG("CUDF Aggregate");
    SIRIUS_LOG_DEBUG("Input size: {}", column[0]->column_length);
    SETUP_TIMING();
    START_TIMER();

    uint64_t size = 0;
    for (int agg = 0; agg < num_aggregates; agg++) {
        if (column[agg]->data_wrapper.data != nullptr || (column[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT_STAR && column[agg]->column_length > 0)) {
            size = column[agg]->column_length;
            break;
        }
    }

    for (int agg = 0; agg < num_aggregates; agg++) {
        if (column[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT && column[agg]->column_length == 0) {
            uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
            cudaMemset(temp, 0, sizeof(uint64_t));
            column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp));
        } else if (column[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::SUM && column[agg]->column_length == 0) {
            uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
            cudaMemset(temp, 0, sizeof(uint64_t));
            column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp));
        } else if (column[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT_STAR && column[agg]->column_length != 0) {
            uint64_t* res = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
            res[0] = size;
            uint64_t* result_temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
            cudaMemcpy(result_temp, res, sizeof(uint64_t), cudaMemcpyHostToDevice);
            column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(result_temp));
        } else if (agg_mode[agg] == AggregationType::SUM) {
            auto aggregate = make_reduce_aggregation<cudf::reduce_aggregation::SUM>();
            auto cudf_column = column[agg]->convertToCudfColumn();
            auto result = cudf::reduce(cudf_column, *aggregate, cudf_column.type());
            column[agg]->setFromCudfScalar(*result, gpuBufferManager);
        } else if (agg_mode[agg] == AggregationType::AVERAGE) {
            auto aggregate = make_reduce_aggregation<cudf::reduce_aggregation::MEAN>();
            // If aggregate input column is decimal, need to convert to double following duckdb
            if (column[agg]->data_wrapper.type.id() == GPUColumnTypeId::DECIMAL) {
                if (column[agg]->data_wrapper.getColumnTypeSize() != sizeof(int64_t)) {
                    throw NotImplementedException("Only support decimal64 for decimal AVG aggregate");
                }
                auto from_cudf_column_view = column[agg]->convertToCudfColumn();
                auto to_cudf_type = cudf::data_type(cudf::type_id::FLOAT64);
                auto to_cudf_column = cudf::cast(
                    from_cudf_column_view, to_cudf_type, rmm::cuda_stream_default, GPUBufferManager::GetInstance().mr);
                column[agg]->setFromCudfColumn(*to_cudf_column, false, nullptr, 0, gpuBufferManager);
            }
            auto cudf_column = column[agg]->convertToCudfColumn();
            auto result = cudf::reduce(cudf_column, *aggregate, cudf_column.type());
            column[agg]->setFromCudfScalar(*result, gpuBufferManager);
        } else if (agg_mode[agg] == AggregationType::MIN) {
            auto aggregate = make_reduce_aggregation<cudf::reduce_aggregation::MIN>();
            auto cudf_column = column[agg]->convertToCudfColumn();
            auto result = cudf::reduce(cudf_column, *aggregate, cudf_column.type());
            column[agg]->setFromCudfScalar(*result, gpuBufferManager);
        } else if (agg_mode[agg] == AggregationType::MAX) {
            auto aggregate = make_reduce_aggregation<cudf::reduce_aggregation::MAX>();
            auto cudf_column = column[agg]->convertToCudfColumn();
            auto result = cudf::reduce(cudf_column, *aggregate, cudf_column.type());
            column[agg]->setFromCudfScalar(*result, gpuBufferManager);
        } else if (agg_mode[agg] == AggregationType::COUNT) {
            uint64_t* res = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
            res[0] = size;
            uint64_t* result_temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
            cudaMemcpy(result_temp, res, sizeof(uint64_t), cudaMemcpyHostToDevice);
            column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(result_temp));
        } else if (agg_mode[agg] == AggregationType::COUNT_DISTINCT) {
            auto aggregate = make_reduce_aggregation<cudf::reduce_aggregation::NUNIQUE>();
            auto cudf_column = column[agg]->convertToCudfColumn();
            auto result = cudf::reduce(cudf_column, *aggregate, cudf_column.type());
            column[agg]->setFromCudfScalar(*result, gpuBufferManager);
        } else if (agg_mode[agg] == AggregationType::FIRST) {
            if (column[agg]->data_wrapper.type.id() == GPUColumnTypeId::INT64) {
                uint64_t* result_temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
                cudaMemcpy(result_temp, reinterpret_cast<uint64_t*>(column[agg]->data_wrapper.data), sizeof(uint64_t), cudaMemcpyDeviceToDevice);
                column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(result_temp));
            } else if (column[agg]->data_wrapper.type.id() == GPUColumnTypeId::INT32) {
                int32_t* result_temp = gpuBufferManager->customCudaMalloc<int32_t>(1, 0, 0);
                cudaMemcpy(result_temp, reinterpret_cast<int32_t*>(column[agg]->data_wrapper.data), sizeof(int32_t), cudaMemcpyDeviceToDevice);
                column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT32), reinterpret_cast<uint8_t*>(result_temp));
            } else if (column[agg]->data_wrapper.type.id() == GPUColumnTypeId::FLOAT32) {
                float* result_temp = gpuBufferManager->customCudaMalloc<float>(1, 0, 0);
                cudaMemcpy(result_temp, reinterpret_cast<float*>(column[agg]->data_wrapper.data), sizeof(float), cudaMemcpyDeviceToDevice);
                column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::FLOAT32), reinterpret_cast<uint8_t*>(result_temp));
            } else if (column[agg]->data_wrapper.type.id() == GPUColumnTypeId::FLOAT64) {
                double* result_temp = gpuBufferManager->customCudaMalloc<double>(1, 0, 0);
                cudaMemcpy(result_temp, reinterpret_cast<double*>(column[agg]->data_wrapper.data), sizeof(double), cudaMemcpyDeviceToDevice);
                column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::FLOAT64), reinterpret_cast<uint8_t*>(result_temp));
            } else if (column[agg]->data_wrapper.type.id() == GPUColumnTypeId::BOOLEAN) {
                uint8_t* result_temp = gpuBufferManager->customCudaMalloc<uint8_t>(1, 0, 0);
                cudaMemcpy(result_temp, reinterpret_cast<uint8_t*>(column[agg]->data_wrapper.data), sizeof(uint8_t), cudaMemcpyDeviceToDevice);
                column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::BOOLEAN), reinterpret_cast<uint8_t*>(result_temp));
            } else if (column[agg]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
                uint64_t* length = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
                cudaMemcpy(length, column[agg]->data_wrapper.offset + 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);

                char* result_temp = gpuBufferManager->customCudaMalloc<char>(length[0], 0, 0);
                cudaMemcpy(result_temp, reinterpret_cast<char*>(column[agg]->data_wrapper.data), length[0], cudaMemcpyDeviceToDevice);

                uint64_t* new_offset = gpuBufferManager->customCudaMalloc<uint64_t>(2, 0, 0);
                cudaMemcpy(new_offset, column[agg]->data_wrapper.offset, 2 * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

                column[agg] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::VARCHAR), reinterpret_cast<uint8_t*>(result_temp), new_offset, length[0], true);
            }
        } 
        else {
            throw NotImplementedException("Aggregate function not supported");
        }
    }

    STOP_TIMER();

}

} //namespace duckdb