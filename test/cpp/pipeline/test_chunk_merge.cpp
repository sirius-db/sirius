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

#include "catch.hpp"
#include "gpu_combine.hpp"
#include "utils/utils.hpp"

namespace duckdb {

void merge_and_verify(GPUBufferManager* gpu_buffer_manager, const vector<shared_ptr<GPUIntermediateRelation>>& tables,
                      const vector<uint8_t**>& host_data, const vector<uint64_t**>& host_offset) {
  // Do merge
  auto merged_table = CombineChunks(tables);
  int num_columns = merged_table->column_count;
  if (num_columns == 0) {
    return;
  }

  // Get num_rows for each input and output
  int merged_num_rows = merged_table->columns[0]->row_ids == nullptr
    ? merged_table->columns[0]->column_length
    : merged_table->columns[0]->row_id_count;
  int num_input_chunks = tables.size();
  vector<int> num_input_rows{};
  num_input_rows.resize(num_input_chunks);
  for (int i = 0; i < num_input_chunks; ++i) {
    num_input_rows[i] = tables[i]->columns[0]->row_ids == nullptr
      ? tables[i]->columns[0]->column_length
      : tables[i]->columns[0]->row_id_count;
  }

  // Construct expected merged table data
  uint8_t** merged_host_data = new uint8_t*[num_columns]();
  uint64_t** merged_host_offset = new uint64_t*[num_columns]();
  vector<GPUColumnType> types{};
  types.resize(num_columns);
  for (int c = 0; c < num_columns; ++c) {
    types[c] = merged_table->columns[c]->data_wrapper.type;
    switch (types[c].id()) {
      case GPUColumnTypeId::INT32: {
        merged_host_data[c] = new uint8_t[merged_num_rows * sizeof(int32_t)];
        int row_offset = 0;
        for (int i = 0; i < num_input_chunks; ++i) {
          memcpy(merged_host_data[c] + row_offset * sizeof(int32_t), host_data[i][c],
                 num_input_rows[i] * sizeof(int32_t));
          row_offset += num_input_rows[i];
        }
        break;
      }
      case GPUColumnTypeId::INT64: {
        merged_host_data[c] = new uint8_t[merged_num_rows * sizeof(int64_t)];
        int row_offset = 0;
        for (int i = 0; i < num_input_chunks; ++i) {
          memcpy(merged_host_data[c] + row_offset * sizeof(int64_t), host_data[i][c],
                 num_input_rows[i] * sizeof(int64_t));
          row_offset += num_input_rows[i];
        }
        break;
      }
      case GPUColumnTypeId::VARCHAR: {
        merged_host_offset[c] = new uint64_t[merged_num_rows + 1];
        merged_host_offset[c][0] = 0;
        int row_offset = 0;
        for (int i = 0; i < num_input_chunks; ++i) {
          for (int r = 0; r < num_input_rows[i]; ++r) {
            int len = host_offset[i][c][r + 1] - host_offset[i][c][r];
            merged_host_offset[c][row_offset + r + 1] = len + merged_host_offset[c][row_offset + r];
          }
          row_offset += num_input_rows[i];
        }
        merged_host_data[c] = new uint8_t[merged_host_offset[c][merged_num_rows]];
        int data_offset = 0;
        for (int i = 0; i < num_input_chunks; ++i) {
          memcpy(merged_host_data[c] + data_offset, host_data[i][c], host_offset[i][c][num_input_rows[i]]);
          data_offset += host_offset[i][c][num_input_rows[i]];
        }
        break;
      }
      default:
        FAIL("Unsupported GPUColumnTypeId in `merge_and_verify`: " << static_cast<int>(types[c].id()));
    }
  }
  verify_table(gpu_buffer_manager, *merged_table, merged_host_data, merged_host_offset);
  free_cpu_buffer(types, merged_host_data, merged_host_offset);
}

void test_chunk_merge() {
  // Initialize
  static constexpr size_t buffer_size = 1024L * 1024;
  GPUBufferManager* gpu_buffer_manager = &(GPUBufferManager::GetInstance(buffer_size, buffer_size, buffer_size));

  // Prepare input data
  vector<GPUColumnType> types{GPUColumnType(GPUColumnTypeId::INT32),
                              GPUColumnType(GPUColumnTypeId::INT64),
                              GPUColumnType(GPUColumnTypeId::VARCHAR)};
  const int num_rows_per_chunk = 10, num_input_chunks = 10;
  vector<uint8_t**> host_data{num_input_chunks};
  vector<uint64_t**> host_offset{num_input_chunks};
  vector<shared_ptr<GPUIntermediateRelation>> tables{num_input_chunks};
  for (int i = 0; i < num_input_chunks; ++i) {
    tables[i] = create_table(gpu_buffer_manager, types, num_rows_per_chunk, host_data[i], host_offset[i]);
  }

  // Merge and verify result
  merge_and_verify(gpu_buffer_manager, tables, host_data, host_offset);

  // Free
  gpu_buffer_manager->ResetBuffer();
  // free_cpu_buffer(types, merged_host_data, merged_host_offset);
  for (int i = 0; i < num_input_chunks; ++i) {
    free_cpu_buffer(types, host_data[i], host_offset[i]);
  }
}

}

using namespace duckdb;

TEST_CASE("test_chunk_merge_basic", "[pipeline]") {
  test_chunk_merge();
}

TEST_CASE("test_chunk_merge_with_row_id", "[pipeline]") {
  test_chunk_merge();
}

TEST_CASE("test_chunk_merge_with_null", "[pipeline]") {
  test_chunk_merge();
}
