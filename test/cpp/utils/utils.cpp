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

#include "utils.hpp"

#include "catch.hpp"
#include "gpu_materialize.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

namespace duckdb {

std::mt19937_64& global_rng() {
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  return gen;
}

template <typename T>
T rand_int(T low, T high) {
  std::uniform_int_distribution<T> dist(low, high);
  return dist(global_rng());
}

std::string rand_str(int len) {
  static const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::uniform_int_distribution<std::size_t> dist(0, chars.size() - 1);
  std::string result;
  result.reserve(len);
  for (std::size_t i = 0; i < len; ++i) {
      result += chars[dist(global_rng())];
  }
  return result;
}

shared_ptr<GPUIntermediateRelation> create_table(
  GPUBufferManager* gpu_buffer_manager, const vector<GPUColumnType>& types, const int num_rows,
  uint8_t**& host_data, uint64_t**& host_offset) {
  int num_columns = types.size();
  auto table = make_shared_ptr<GPUIntermediateRelation>(num_columns);
  host_data = new uint8_t*[num_columns]();
  host_offset = new uint64_t*[num_columns]();
  for (int c = 0; c < num_columns; ++c) {
    table->column_names[c] = "col_" + to_string(c);
    switch (types[c].id()) {
      case GPUColumnTypeId::INT32: {
        table->columns[c] = make_shared_ptr<GPUColumn>(num_rows, types[c], nullptr, nullptr);
        table->columns[c]->data_wrapper.data = gpu_buffer_manager->customCudaMalloc<uint8_t>(
          num_rows * sizeof(int32_t), 0, false);
        host_data[c] = new uint8_t[num_rows * sizeof(int32_t)];
        for (int r = 0; r < num_rows; ++r) {
          reinterpret_cast<int32_t*>(host_data[c])[r] = rand_int<int32_t>(0, 1000);
        }
        cudaMemcpy(table->columns[c]->data_wrapper.data, host_data[c], num_rows * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        table->columns[c]->data_wrapper.num_bytes = num_rows * sizeof(int32_t);
        break;
      }
      case GPUColumnTypeId::INT64: {
        table->columns[c] = make_shared_ptr<GPUColumn>(num_rows, types[c], nullptr, nullptr);
        table->columns[c]->data_wrapper.data = gpu_buffer_manager->customCudaMalloc<uint8_t>(
          num_rows * sizeof(int64_t), 0, false);
        host_data[c] = new uint8_t[num_rows * sizeof(int64_t)];
        for (int r = 0; r < num_rows; ++r) {
          reinterpret_cast<int64_t*>(host_data[c])[r] = rand_int<int64_t>(0, 1000);
        }
        cudaMemcpy(table->columns[c]->data_wrapper.data, host_data[c], num_rows * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
        table->columns[c]->data_wrapper.num_bytes = num_rows * sizeof(int64_t);
        break;
      }
      case GPUColumnTypeId::VARCHAR: {
        table->columns[c] = make_shared_ptr<GPUColumn>(num_rows, GPUColumnType(GPUColumnTypeId::VARCHAR), nullptr,
                                                       nullptr, 0, true, nullptr);
        host_offset[c] = new uint64_t[num_rows + 1];
        host_offset[c][0] = 0;
        for (int r = 0; r < num_rows; ++r) {
          int len = rand_int<int32_t>(1, 20);
          host_offset[c][r + 1] = host_offset[c][r] + len;
        }
        host_data[c] = new uint8_t[host_offset[c][num_rows]];
        for (int r = 0; r < num_rows; ++r) {
          int len = host_offset[c][r + 1] - host_offset[c][r];
          std::string str = rand_str(len);
          memcpy(host_data[c] + host_offset[c][r], str.data(), len);
        }
        table->columns[c]->data_wrapper.offset = gpu_buffer_manager->customCudaMalloc<uint64_t>(
          num_rows + 1, 0, false);
        table->columns[c]->data_wrapper.data = gpu_buffer_manager->customCudaMalloc<uint8_t>(
          host_offset[c][num_rows], 0, false);
        cudaMemcpy(table->columns[c]->data_wrapper.offset, host_offset[c], (num_rows + 1) * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(table->columns[c]->data_wrapper.data, host_data[c], host_offset[c][num_rows],
                   cudaMemcpyHostToDevice);
        table->columns[c]->data_wrapper.num_bytes = host_offset[c][num_rows];
        break;
      }
      default:
        FAIL("Unsupported GPUColumnTypeId in `create_table`: " << static_cast<int>(types[c].id()));
    }
  }
  return table;
}

void verify_table(GPUBufferManager* gpu_buffer_manager, GPUIntermediateRelation& table,
                  uint8_t** expected_host_data, uint64_t** expected_host_offset) {
  for (int c = 0; c < table.column_count; ++c) {
    auto column = HandleMaterializeExpression(table.columns[c], gpu_buffer_manager);
    switch (column->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32: {
        int32_t* actual_host_data = new int32_t[column->column_length];
        cudaMemcpy(actual_host_data, column->data_wrapper.data, column->column_length * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        for (int r = 0; r < column->column_length; ++r) {
          REQUIRE(actual_host_data[r] == reinterpret_cast<int32_t*>(expected_host_data[c])[r]);
        }
        delete[] actual_host_data;
        break;
      }
      case GPUColumnTypeId::INT64: {
        int64_t* actual_host_data = new int64_t[column->column_length];
        cudaMemcpy(actual_host_data, column->data_wrapper.data, column->column_length * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        for (int r = 0; r < column->column_length; ++r) {
          REQUIRE(actual_host_data[r] == reinterpret_cast<int64_t*>(expected_host_data[c])[r]);
        }
        delete[] actual_host_data;
        break;
      }
      case GPUColumnTypeId::VARCHAR: {
        uint64_t* actual_host_offset = new uint64_t[column->column_length + 1];
        cudaMemcpy(actual_host_offset, column->data_wrapper.offset, (column->column_length + 1) * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        uint8_t* actual_host_data = new uint8_t[actual_host_offset[column->column_length]];
        cudaMemcpy(actual_host_data, column->data_wrapper.data, actual_host_offset[column->column_length],
                   cudaMemcpyDeviceToHost);
        for (int r = 0; r < column->column_length; ++r) {
          std::string actual_str(reinterpret_cast<char*>(actual_host_data) + actual_host_offset[r],
                                 actual_host_offset[r + 1] - actual_host_offset[r]);
          std::string expected_str(reinterpret_cast<char*>(expected_host_data[c]) + expected_host_offset[c][r],
                                   expected_host_offset[c][r + 1] - expected_host_offset[c][r]);
          REQUIRE(actual_str == expected_str);
        }
        delete[] actual_host_offset;
        delete[] actual_host_data;
        break;
      }
      default:
        FAIL("Unsupported GPUColumnTypeId in `verify_table`: " << static_cast<int>(column->data_wrapper.type.id()));
    }
  }
}

void free_cpu_buffer(const vector<GPUColumnType>& types, uint8_t** host_data, uint64_t** host_offset) {
  for (int i = 0; i < types.size(); ++i) {
    delete[] host_data[i];
    if (types[i].id() == GPUColumnTypeId::VARCHAR) {
      delete[] host_offset[i];
    }
  }
  delete[] host_data;
  delete[] host_offset;
}

void verify_cuda_errors(const char *msg) {
    cudaError_t __err = cudaGetLastError();
    if (__err != cudaSuccess) {
        printf("CUDA error: %s (%s at %s:%d)\n",
                msg, cudaGetErrorString(__err),
                __FILE__, __LINE__);
        REQUIRE(1 == 2);
    }   
}

void verify_gpu_buffer_equality(uint8_t* buffer_1, uint8_t* buffer_2, size_t num_bytes) { 
  // If the first buffer is null then verify that the second one is null as well
  if(buffer_1 == nullptr) {
    REQUIRE(buffer_2 == nullptr);
    return;
  }

  // Allocate temporary host buffers to copy the data back
  uint8_t* host_buffer_1 = (uint8_t*) malloc(num_bytes);
  uint8_t* host_buffer_2 = (uint8_t*) malloc(num_bytes);

  // Copy the data back to the host
  cudaMemcpy(host_buffer_1, buffer_1, num_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_buffer_2, buffer_2, num_bytes, cudaMemcpyDeviceToHost);

  // Now compare the two buffers
  REQUIRE(memcmp(host_buffer_1, host_buffer_2, num_bytes) == 0);

  // Free the temporary host buffers
  free(host_buffer_1);
  free(host_buffer_2);
}

void verify_gpu_column_equality(shared_ptr<GPUColumn> col1, shared_ptr<GPUColumn> col2) { 
  // First verify that all of the metadata is the same
  REQUIRE(col1->column_length == col2->column_length);
  REQUIRE(col1->row_id_count == col2->row_id_count);
  REQUIRE(col1->is_unique == col2->is_unique);

  DataWrapper col1_data = col1->data_wrapper;
  DataWrapper col2_data = col2->data_wrapper;
  REQUIRE(col1_data.type.id() == col2_data.type.id());
  REQUIRE(col1_data.size == col2_data.size);
  REQUIRE(col1_data.num_bytes == col2_data.num_bytes);
  REQUIRE(col1_data.is_string_data == col2_data.is_string_data);
  REQUIRE(col1_data.mask_bytes == col2_data.mask_bytes);

  // Now verify all of the buffers are the same
  verify_gpu_buffer_equality((uint8_t*) col1->row_ids, (uint8_t*) col2->row_ids, col1->row_id_count * sizeof(uint64_t));
  verify_gpu_buffer_equality(col1_data.data, col2_data.data, col1_data.num_bytes);
  verify_gpu_buffer_equality((uint8_t*) col1_data.validity_mask, (uint8_t*) col2_data.validity_mask, col1_data.mask_bytes);
  if(col1_data.is_string_data) {
    verify_gpu_buffer_equality((uint8_t*) col1_data.offset, (uint8_t*) col2_data.offset, col1_data.size * sizeof(uint64_t));
  }
}

}
