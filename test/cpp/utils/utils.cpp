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

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace duckdb {

template <typename T>
T rand_int(T low, T high)
{
  std::uniform_int_distribution<T> dist(low, high);
  return dist(sirius::global_rng());
}

std::string rand_str(int len)
{
  static const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::uniform_int_distribution<std::size_t> dist(0, chars.size() - 1);
  std::string result;
  result.reserve(len);
  for (std::size_t i = 0; i < len; ++i) {
    result += chars[dist(sirius::global_rng())];
  }
  return result;
}

GPUBufferManager* initialize_test_buffer_manager()
{
  return &(GPUBufferManager::GetInstance(TEST_BUFFER_MANAGER_MEMORY_BYTES,
                                         TEST_BUFFER_MANAGER_MEMORY_BYTES,
                                         TEST_BUFFER_MANAGER_MEMORY_BYTES));
}

void fill_gpu_buffer_with_random_data(uint8_t* gpu_buffer, size_t num_bytes)
{
  // First fill the buffer with random data in chunks of 4 bytes
  int32_t* random_values = reinterpret_cast<int32_t*>(std::malloc(num_bytes));
  size_t num_chunks      = num_bytes / 4;
  for (size_t i = 0; i < num_chunks; i++) {
    random_values[i] = static_cast<uint32_t>(rand());
  }

  // Now copy the data to the GPU and free the temporary buffer
  cudaMemcpy(gpu_buffer, (uint8_t*)random_values, num_bytes, cudaMemcpyHostToDevice);
  std::free(random_values);
}

shared_ptr<GPUIntermediateRelation> create_table(GPUBufferManager* gpu_buffer_manager,
                                                 const vector<GPUColumnType>& types,
                                                 const int num_rows,
                                                 uint8_t**& host_data,
                                                 uint64_t**& host_offset)
{
  int num_columns = types.size();
  auto table      = make_shared_ptr<GPUIntermediateRelation>(num_columns);
  host_data       = new uint8_t*[num_columns]();
  host_offset     = new uint64_t*[num_columns]();
  for (int c = 0; c < num_columns; ++c) {
    table->column_names[c] = "col_" + to_string(c);
    switch (types[c].id()) {
      case GPUColumnTypeId::INT32: {
        table->columns[c] = make_shared_ptr<GPUColumn>(num_rows, types[c], nullptr, nullptr);
        table->columns[c]->data_wrapper.data =
          gpu_buffer_manager->customCudaMalloc<uint8_t>(num_rows * sizeof(int32_t), 0, false);
        host_data[c] = new uint8_t[num_rows * sizeof(int32_t)];
        for (int r = 0; r < num_rows; ++r) {
          reinterpret_cast<int32_t*>(host_data[c])[r] = rand_int<int32_t>(0, 1000);
        }
        cudaMemcpy(table->columns[c]->data_wrapper.data,
                   host_data[c],
                   num_rows * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        table->columns[c]->data_wrapper.num_bytes = num_rows * sizeof(int32_t);
        break;
      }
      case GPUColumnTypeId::INT64: {
        table->columns[c] = make_shared_ptr<GPUColumn>(num_rows, types[c], nullptr, nullptr);
        table->columns[c]->data_wrapper.data =
          gpu_buffer_manager->customCudaMalloc<uint8_t>(num_rows * sizeof(int64_t), 0, false);
        host_data[c] = new uint8_t[num_rows * sizeof(int64_t)];
        for (int r = 0; r < num_rows; ++r) {
          reinterpret_cast<int64_t*>(host_data[c])[r] = rand_int<int64_t>(0, 1000);
        }
        cudaMemcpy(table->columns[c]->data_wrapper.data,
                   host_data[c],
                   num_rows * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
        table->columns[c]->data_wrapper.num_bytes = num_rows * sizeof(int64_t);
        break;
      }
      case GPUColumnTypeId::VARCHAR: {
        table->columns[c] = make_shared_ptr<GPUColumn>(
          num_rows, GPUColumnType(GPUColumnTypeId::VARCHAR), nullptr, nullptr, 0, true, nullptr);
        host_offset[c]    = new uint64_t[num_rows + 1];
        host_offset[c][0] = 0;
        for (int r = 0; r < num_rows; ++r) {
          int len               = rand_int<int32_t>(1, 20);
          host_offset[c][r + 1] = host_offset[c][r] + len;
        }
        host_data[c] = new uint8_t[host_offset[c][num_rows]];
        for (int r = 0; r < num_rows; ++r) {
          int len         = host_offset[c][r + 1] - host_offset[c][r];
          std::string str = rand_str(len);
          memcpy(host_data[c] + host_offset[c][r], str.data(), len);
        }
        table->columns[c]->data_wrapper.offset =
          gpu_buffer_manager->customCudaMalloc<uint64_t>(num_rows + 1, 0, false);
        table->columns[c]->data_wrapper.data =
          gpu_buffer_manager->customCudaMalloc<uint8_t>(host_offset[c][num_rows], 0, false);
        cudaMemcpy(table->columns[c]->data_wrapper.offset,
                   host_offset[c],
                   (num_rows + 1) * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(table->columns[c]->data_wrapper.data,
                   host_data[c],
                   host_offset[c][num_rows],
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

void verify_table(GPUBufferManager* gpu_buffer_manager,
                  GPUIntermediateRelation& table,
                  uint8_t** expected_host_data,
                  uint64_t** expected_host_offset)
{
  for (int c = 0; c < table.column_count; ++c) {
    auto column = HandleMaterializeExpression(table.columns[c], gpu_buffer_manager);
    switch (column->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT32: {
        int32_t* actual_host_data = new int32_t[column->column_length];
        cudaMemcpy(actual_host_data,
                   column->data_wrapper.data,
                   column->column_length * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        for (int r = 0; r < column->column_length; ++r) {
          REQUIRE(actual_host_data[r] == reinterpret_cast<int32_t*>(expected_host_data[c])[r]);
        }
        delete[] actual_host_data;
        break;
      }
      case GPUColumnTypeId::INT64: {
        int64_t* actual_host_data = new int64_t[column->column_length];
        cudaMemcpy(actual_host_data,
                   column->data_wrapper.data,
                   column->column_length * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        for (int r = 0; r < column->column_length; ++r) {
          REQUIRE(actual_host_data[r] == reinterpret_cast<int64_t*>(expected_host_data[c])[r]);
        }
        delete[] actual_host_data;
        break;
      }
      case GPUColumnTypeId::VARCHAR: {
        uint64_t* actual_host_offset = new uint64_t[column->column_length + 1];
        cudaMemcpy(actual_host_offset,
                   column->data_wrapper.offset,
                   (column->column_length + 1) * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        uint8_t* actual_host_data = new uint8_t[actual_host_offset[column->column_length]];
        cudaMemcpy(actual_host_data,
                   column->data_wrapper.data,
                   actual_host_offset[column->column_length],
                   cudaMemcpyDeviceToHost);
        for (int r = 0; r < column->column_length; ++r) {
          std::string actual_str(reinterpret_cast<char*>(actual_host_data) + actual_host_offset[r],
                                 actual_host_offset[r + 1] - actual_host_offset[r]);
          std::string expected_str(
            reinterpret_cast<char*>(expected_host_data[c]) + expected_host_offset[c][r],
            expected_host_offset[c][r + 1] - expected_host_offset[c][r]);
          REQUIRE(actual_str == expected_str);
        }
        delete[] actual_host_offset;
        delete[] actual_host_data;
        break;
      }
      default:
        FAIL("Unsupported GPUColumnTypeId in `verify_table`: "
             << static_cast<int>(column->data_wrapper.type.id()));
    }
  }
}

void free_cpu_buffer(const vector<GPUColumnType>& types,
                     uint8_t** host_data,
                     uint64_t** host_offset)
{
  for (int i = 0; i < types.size(); ++i) {
    delete[] host_data[i];
    if (types[i].id() == GPUColumnTypeId::VARCHAR) { delete[] host_offset[i]; }
  }
  delete[] host_data;
  delete[] host_offset;
}

void verify_cuda_errors(const char* msg)
{
  cudaError_t __err = cudaGetLastError();
  if (__err != cudaSuccess) {
    printf("CUDA error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);
    REQUIRE(1 == 2);
  }
}

void verify_gpu_buffer_equality(uint8_t* buffer_1, uint8_t* buffer_2, size_t num_bytes)
{
  // If the first buffer is null then verify that the second one is null as well
  if (buffer_1 == nullptr) {
    REQUIRE(buffer_2 == nullptr);
    return;
  }

  // Allocate temporary host buffers to copy the data back
  uint8_t* host_buffer_1 = (uint8_t*)malloc(num_bytes);
  uint8_t* host_buffer_2 = (uint8_t*)malloc(num_bytes);

  // Copy the data back to the host
  cudaMemcpy(host_buffer_1, buffer_1, num_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_buffer_2, buffer_2, num_bytes, cudaMemcpyDeviceToHost);

  // Now compare the two buffers
  REQUIRE(memcmp(host_buffer_1, host_buffer_2, num_bytes) == 0);

  // Free the temporary host buffers
  free(host_buffer_1);
  free(host_buffer_2);
}

void verify_gpu_column_equality(shared_ptr<GPUColumn> col1, shared_ptr<GPUColumn> col2)
{
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
  verify_gpu_buffer_equality(
    (uint8_t*)col1->row_ids, (uint8_t*)col2->row_ids, col1->row_id_count * sizeof(uint64_t));
  verify_gpu_buffer_equality(col1_data.data, col2_data.data, col1_data.num_bytes);
  verify_gpu_buffer_equality(
    (uint8_t*)col1_data.validity_mask, (uint8_t*)col2_data.validity_mask, col1_data.mask_bytes);
  if (col1_data.is_string_data) {
    verify_gpu_buffer_equality(
      (uint8_t*)col1_data.offset, (uint8_t*)col2_data.offset, col1_data.size * sizeof(uint64_t));
  }
}

shared_ptr<GPUColumn> create_column_with_random_data(GPUColumnTypeId col_type,
                                                     size_t num_records,
                                                     size_t chars_per_record,
                                                     size_t num_materialize_row_ids,
                                                     bool has_null_mask)
{
  // Initialize an empty column
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  shared_ptr<GPUColumn> column =
    make_shared_ptr<GPUColumn>(num_records, GPUColumnType(col_type), nullptr, nullptr);

  // Populate the data of the column based on whether it is a fixed or variable length type
  size_t num_data_bytes = 0;
  if (col_type == GPUColumnTypeId::VARCHAR) {
    num_data_bytes                      = num_records * chars_per_record;
    column->data_wrapper.is_string_data = true;

    // Also populate the offsets column if is a string column
    size_t num_offset_records = (num_records + 1);
    column->data_wrapper.offset =
      gpuBufferManager->customCudaMalloc<uint64_t>(num_offset_records, 0, 0);
    fill_gpu_buffer_with_random_data((uint8_t*)column->data_wrapper.offset,
                                     num_offset_records * sizeof(uint64_t));
  } else {
    column->data_wrapper.is_string_data = false;
    num_data_bytes                      = num_records * column->data_wrapper.getColumnTypeSize();
  }
  column->data_wrapper.data = gpuBufferManager->customCudaMalloc<uint8_t>(num_data_bytes, 0, 0);
  column->data_wrapper.num_bytes = num_data_bytes;
  fill_gpu_buffer_with_random_data(column->data_wrapper.data, num_data_bytes);

  // If specified then set the null mask
  if (has_null_mask) {
    size_t null_mask_bytes             = getMaskBytesSize(num_records);
    column->data_wrapper.mask_bytes    = null_mask_bytes;
    column->data_wrapper.validity_mask = reinterpret_cast<cudf::bitmask_type*>(
      gpuBufferManager->customCudaMalloc<uint8_t>(null_mask_bytes, 0, 0));
    fill_gpu_buffer_with_random_data((uint8_t*)column->data_wrapper.validity_mask, null_mask_bytes);
  }

  // Also initialize the materialize row ids pointer
  column->row_id_count = num_materialize_row_ids;
  if (num_materialize_row_ids > 0) {
    column->row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(num_materialize_row_ids, 0, 0);
    fill_gpu_buffer_with_random_data((uint8_t*)column->row_ids,
                                     num_materialize_row_ids * sizeof(uint64_t));
  } else {
    column->row_ids = nullptr;
  }

  return column;
}

}  // namespace duckdb

namespace sirius {

std::mt19937_64& global_rng()
{
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  return gen;
}

template <typename T>
std::unique_ptr<cudf::column> create_numeric_column_with_random_data(
  size_t num_rows,
  const cudf::data_type& dtype,
  const std::optional<std::pair<int, int>>& range,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto& gen = global_rng();
  auto col  = cudf::make_numeric_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto view = col->mutable_view();

  auto dist = range.has_value() ? std::uniform_int_distribution<T>(range->first, range->second)
                                : std::uniform_int_distribution<T>(0, 1000);
  std::vector<T> h_data(num_rows);
  for (size_t r = 0; r < num_rows; ++r)
    h_data[r] = dist(gen);

  cudaMemcpy(view.data<T>(), h_data.data(), sizeof(T) * num_rows, cudaMemcpyHostToDevice);
  return col;
}

std::unique_ptr<cudf::table> create_cudf_table_with_random_data(
  size_t num_rows,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<std::optional<std::pair<int, int>>>& ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  bool use_int64_string_offsets)
{
  auto& gen = global_rng();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(column_types.size());

  for (int c = 0; c < column_types.size(); ++c) {
    const auto& dtype = column_types[c];
    switch (dtype.id()) {
      case cudf::type_id::INT32: {
        cols.push_back(std::move(
          create_numeric_column_with_random_data<int32_t>(num_rows, dtype, ranges[c], stream, mr)));
        break;
      }
      case cudf::type_id::INT64: {
        cols.push_back(std::move(
          create_numeric_column_with_random_data<int64_t>(num_rows, dtype, ranges[c], stream, mr)));
        break;
      }
      case cudf::type_id::STRING: {
        auto dist = ranges[c].has_value()
                      ? std::uniform_int_distribution<int>(ranges[c]->first, ranges[c]->second)
                      : std::uniform_int_distribution<int>(0, 1000);
        std::vector<char> h_chars;
        std::vector<int64_t> h_offsets(num_rows + 1, 0);
        for (size_t r = 0; r < num_rows; ++r) {
          std::string h_str = "str_" + std::to_string(dist(gen));
          h_chars.insert(h_chars.end(), h_str.begin(), h_str.end());
          h_offsets[r + 1] = h_offsets[r] + static_cast<int64_t>(h_str.size());
        }

        std::unique_ptr<cudf::column> offsets_col;
        if (use_int64_string_offsets) {
          offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                  h_offsets.size(),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
          cudaMemcpy(offsets_col->mutable_view().data<int64_t>(),
                     h_offsets.data(),
                     sizeof(int64_t) * h_offsets.size(),
                     cudaMemcpyHostToDevice);
        } else {
          std::vector<cudf::size_type> offsets32(h_offsets.size(), 0);
          std::transform(h_offsets.begin(), h_offsets.end(), offsets32.begin(), [](int64_t value) {
            return static_cast<cudf::size_type>(value);
          });
          offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  h_offsets.size(),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
          cudaMemcpy(offsets_col->mutable_view().data<cudf::size_type>(),
                     offsets32.data(),
                     sizeof(cudf::size_type) * offsets32.size(),
                     cudaMemcpyHostToDevice);
        }

        rmm::device_buffer d_chars(h_chars.data(), h_chars.size(), stream, mr);

        auto col = make_strings_column(
          num_rows, std::move(offsets_col), std::move(d_chars), 0, rmm::device_buffer{});
        cols.push_back(std::move(col));
        break;
      }
      default:
        throw std::runtime_error("Unsupported cudf::data_type in `make_random_cudf_table()`");
    }
  }

  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace sirius
