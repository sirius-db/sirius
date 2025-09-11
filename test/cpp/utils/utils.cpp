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
  GPUBufferManager* gpuBufferManager, const vector<GPUColumnType>& types, const int num_rows,
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
        table->columns[c]->data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(
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
        table->columns[c]->data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(
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
        table->columns[c]->data_wrapper.offset = gpuBufferManager->customCudaMalloc<uint64_t>(num_rows + 1, 0, false);
        table->columns[c]->data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(
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

void verify_table(GPUBufferManager* gpuBufferManager, GPUIntermediateRelation& table,
                  uint8_t** expected_host_data, uint64_t** expected_host_offset) {
  for (int c = 0; c < table.column_count; ++c) {
    auto column = HandleMaterializeExpression(table.columns[c], gpuBufferManager);
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

void free_buffer(GPUBufferManager* gpuBufferManager, const vector<GPUColumnType>& types,
                 uint8_t** host_data, uint64_t** host_offset) {
  gpuBufferManager->ResetBuffer();
  for (int i = 0; i < types.size(); ++i) {
    delete[] host_data[i];
    if (types[i].id() == GPUColumnTypeId::VARCHAR) {
      delete[] host_offset[i];
    }
  }
  delete[] host_data;
  delete[] host_offset;
}

}
