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

#include "config.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/common/printer.hpp"
#include "log/logging.hpp"
#include "operator/cuda_helper.cuh"
#include "operator/gpu_physical_result_collector.hpp"
#include "print.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <iostream>
#include <vector>

namespace duckdb {

template <typename T>
__global__ void print_gpu_column(T* a, uint64_t N)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (uint64_t i = 0; i < N; i++) {
      // FIXME: do this in cpu code using logging
    }
  }
}

template __global__ void print_gpu_column<uint64_t>(uint64_t* a, uint64_t N);
template __global__ void print_gpu_column<double>(double* a, uint64_t N);
template __global__ void print_gpu_column<int>(int* a, uint64_t N);
template __global__ void print_gpu_column<float>(float* a, uint64_t N);
template __global__ void print_gpu_column<uint8_t>(uint8_t* a, uint64_t N);

template <typename T>
void printGPUColumn(T* a, size_t N, int gpu)
{
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

void printGPUTable(GPUIntermediateRelation& table, ClientContext& context)
{
  // Transfer data from GPU to CPU `gpu_result_collection`
  vector<LogicalType> types;
  for (int i = 0; i < table.column_count; ++i) {
    if (table.columns[i] == nullptr) {
      throw InternalException("Column %d uninitialized in `printGPUTable`", i);
    }
    types.push_back(convertColumnTypeToLogicalType(table.columns[i]->data_wrapper.type));
  }
  auto gpu_result_collection         = make_uniq<GPUResultCollection>();
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  GPUPhysicalMaterializedCollector::ConvertGPUTableToCPUCollection(
    table, types, gpu_result_collection.get(), gpuBufferManager);

  // Make duckdb `MaterializedQueryResult` and print
  auto column_data_collection = make_uniq<ColumnDataCollection>(Allocator::Get(context), types);
  auto chunk                  = gpu_result_collection->GetNext();
  while (chunk != nullptr) {
    column_data_collection->Append(*chunk);
    chunk = gpu_result_collection->GetNext();
  }
  auto materialized_query_result =
    make_uniq<MaterializedQueryResult>(StatementType::SELECT_STATEMENT,
                                       StatementProperties(),
                                       table.column_names,
                                       move(column_data_collection),
                                       context.GetClientProperties());
  BoxRendererConfig box_render_config;
  box_render_config.max_rows = Config::PRINT_GPU_TABLE_MAX_ROWS;
  Printer::Print(materialized_query_result->ToBox(context, box_render_config));
}

}  // namespace duckdb

namespace sirius {

namespace {

constexpr cudf::size_type kDefaultMaxRows = 20;

template <typename T>
void print_column_values_signed(cudf::column_view const& col, cudf::size_type max_rows)
{
  cudf::size_type n = std::min(col.size(), max_rows);
  if (n <= 0) { return; }
  std::vector<T> host(n);
  cudaError_t err = cudaMemcpy(
    host.data(), col.data<T>(), static_cast<size_t>(n) * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::printf("(cudaMemcpy failed: %s)", cudaGetErrorString(err));
    return;
  }
  for (cudf::size_type i = 0; i < n; ++i) {
    std::printf("%s%" PRId64, i ? ", " : "", static_cast<int64_t>(host[i]));
  }
  if (col.size() > max_rows) { std::printf(", ..."); }
}

template <typename T>
void print_column_values_unsigned(cudf::column_view const& col, cudf::size_type max_rows)
{
  cudf::size_type n = std::min(col.size(), max_rows);
  if (n <= 0) { return; }
  std::vector<T> host(n);
  cudaError_t err = cudaMemcpy(
    host.data(), col.data<T>(), static_cast<size_t>(n) * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::printf("(cudaMemcpy failed: %s)", cudaGetErrorString(err));
    return;
  }
  for (cudf::size_type i = 0; i < n; ++i) {
    std::printf("%s%" PRIu64, i ? ", " : "", static_cast<uint64_t>(host[i]));
  }
  if (col.size() > max_rows) { std::printf(", ..."); }
}

void print_column_values_float(cudf::column_view const& col, cudf::size_type max_rows)
{
  cudf::size_type n = std::min(col.size(), max_rows);
  if (n <= 0) { return; }
  std::vector<float> host(n);
  cudaError_t err = cudaMemcpy(
    host.data(), col.data<float>(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::printf("(cudaMemcpy failed: %s)", cudaGetErrorString(err));
    return;
  }
  for (cudf::size_type i = 0; i < n; ++i) {
    std::printf("%s%.6g", i ? ", " : "", host[i]);
  }
  if (col.size() > max_rows) { std::printf(", ..."); }
}

void print_column_values_double(cudf::column_view const& col, cudf::size_type max_rows)
{
  cudf::size_type n = std::min(col.size(), max_rows);
  if (n <= 0) { return; }
  std::vector<double> host(n);
  cudaError_t err = cudaMemcpy(host.data(),
                               col.data<double>(),
                               static_cast<size_t>(n) * sizeof(double),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::printf("(cudaMemcpy failed: %s)", cudaGetErrorString(err));
    return;
  }
  for (cudf::size_type i = 0; i < n; ++i) {
    std::printf("%s%.6g", i ? ", " : "", host[i]);
  }
  if (col.size() > max_rows) { std::printf(", ..."); }
}

void print_one_column(cudf::column_view const& col, cudf::size_type max_rows, int col_idx)
{
  std::printf("  col[%d] (%s, %zd rows): ",
              col_idx,
              cudf::type_to_name(col.type()).c_str(),
              static_cast<size_t>(col.size()));
  switch (col.type().id()) {
    case cudf::type_id::INT8: print_column_values_signed<int8_t>(col, max_rows); break;
    case cudf::type_id::INT16: print_column_values_signed<int16_t>(col, max_rows); break;
    case cudf::type_id::INT32: print_column_values_signed<int32_t>(col, max_rows); break;
    case cudf::type_id::INT64: print_column_values_signed<int64_t>(col, max_rows); break;
    case cudf::type_id::UINT8: print_column_values_unsigned<uint8_t>(col, max_rows); break;
    case cudf::type_id::UINT16: print_column_values_unsigned<uint16_t>(col, max_rows); break;
    case cudf::type_id::UINT32: print_column_values_unsigned<uint32_t>(col, max_rows); break;
    case cudf::type_id::UINT64: print_column_values_unsigned<uint64_t>(col, max_rows); break;
    case cudf::type_id::FLOAT32: print_column_values_float(col, max_rows); break;
    case cudf::type_id::FLOAT64: print_column_values_double(col, max_rows); break;
    case cudf::type_id::BOOL8: print_column_values_signed<int8_t>(col, max_rows); break;
    default: std::printf("(unprinted type %s)", cudf::type_to_name(col.type()).c_str()); break;
  }
  std::printf("\n");
}

}  // namespace

void print_table_contents(cudf::table_view const& table, cudf::size_type max_rows)
{
  if (max_rows <= 0) { max_rows = kDefaultMaxRows; }
  cudaDeviceSynchronize();
  std::printf("table_view: %zd rows, %d columns\n",
              static_cast<size_t>(table.num_rows()),
              static_cast<int>(table.num_columns()));
  for (cudf::size_type c = 0; c < table.num_columns(); ++c) {
    print_one_column(table.column(c), max_rows, static_cast<int>(c));
  }
}

void print_data_batch_contents(cucascade::data_batch const& batch, cudf::size_type max_rows)
{
  cudf::table_view tv = get_cudf_table_view(batch);
  std::printf("data_batch (id=%llu):\n", static_cast<unsigned long long>(batch.get_batch_id()));
  print_table_contents(tv, max_rows);
}

}  // namespace sirius
