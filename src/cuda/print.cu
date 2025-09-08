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

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "print.hpp"
#include "operator/cuda_helper.cuh"
#include "operator/gpu_physical_result_collector.hpp"
#include "log/logging.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/common/printer.hpp"

namespace duckdb {

template <typename T>
__global__ void print_gpu_column(T* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < N; i++) {
            // FIXME: do this in cpu code using logging
        }
    }
}

template
__global__ void print_gpu_column<uint64_t>(uint64_t* a, uint64_t N);
template
__global__ void print_gpu_column<double>(double* a, uint64_t N);
template
__global__ void print_gpu_column<int>(int* a, uint64_t N);
template
__global__ void print_gpu_column<float>(float* a, uint64_t N);
template
__global__ void print_gpu_column<uint8_t>(uint8_t* a, uint64_t N);

template <typename T> 
void printGPUColumn(T* a, size_t N, int gpu) {
    CHECK_ERROR();
    if (N == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    T result_host_temp;
    cudaMemcpy(&result_host_temp, a, sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    SIRIUS_LOG_DEBUG("Result: {} and N: {}", result_host_temp, N);
    SIRIUS_LOG_DEBUG("Input size: {}", N);
    print_gpu_column<T><<<1, 1>>>(a, N);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template void printGPUColumn<uint64_t>(uint64_t* a, size_t N, int gpu);
template void printGPUColumn<double>(double* a, size_t N, int gpu);
template void printGPUColumn<int>(int* a, size_t N, int gpu);
template void printGPUColumn<float>(float* a, size_t N, int gpu);

void printGPUTable(GPUIntermediateRelation& table, ClientContext &context) {
    // Transfer data from GPU to CPU `gpu_result_collection`
    vector<LogicalType> types;
    for (int i = 0; i < table.column_count; ++i) {
        if (table.columns[i] == nullptr) {
            throw InternalException("Column %d uninitialized in `printGPUTable`", i);
        }
        types.push_back(convertColumnTypeToLogicalType(table.columns[i]->data_wrapper.type));
    }
    auto gpu_result_collection = make_uniq<GPUResultCollection>();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    GPUPhysicalMaterializedCollector::ConvertGPUTableToCPUCollection(
        table, types, gpu_result_collection.get(), gpuBufferManager);
    
    // Make duckdb `MaterializedQueryResult` and print
    auto column_data_collection = make_uniq<ColumnDataCollection>(Allocator::Get(context), types);
    auto chunk = gpu_result_collection->GetNext();
    while (chunk != nullptr) {
        column_data_collection->Append(*chunk);
        chunk = gpu_result_collection->GetNext();
    }
    auto materialized_query_result = make_uniq<MaterializedQueryResult>(
        StatementType::SELECT_STATEMENT, StatementProperties(), table.column_names,
        move(column_data_collection), context.GetClientProperties());
    BoxRendererConfig box_render_config;
    Printer::Print(materialized_query_result->ToBox(context, box_render_config));
}

} // namespace duckdb