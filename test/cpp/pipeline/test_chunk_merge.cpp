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

using namespace duckdb;

TEST_CASE("test_chunk_merge_basic", "[pipeline]") {
  // Initialize
  static constexpr size_t buffer_size = 1024L * 1024;
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance(buffer_size, buffer_size, buffer_size));

  // Prepare input data
  duckdb::vector<GPUColumnType> types{GPUColumnType(GPUColumnTypeId::INT32),
                                      GPUColumnType(GPUColumnTypeId::INT64),
                                      GPUColumnType(GPUColumnTypeId::VARCHAR)};
  const int num_rows_per_chunk = 10;
  uint8_t** host_data;
  uint64_t** host_offset;
  auto table = create_table(gpuBufferManager, types, num_rows_per_chunk, host_data, host_offset);

  // Verify result
  verify_table(gpuBufferManager, *table, host_data, host_offset);

  // Free
  free_buffer(gpuBufferManager, types, host_data, host_offset);

  // const int num_input_chunks = 10;
  // duckdb::vector<duckdb::shared_ptr<GPUIntermediateRelation>> input;
  // for (int i = 0; i < num_input_chunks; ++i) {
  //   input.push_back(create_table(i, gpuBufferManager, num_columns, num_rows_per_chunk));
  //   printf("Input %d:\n", i);
  //   printGPUTable(*input[i], *con.context);
  // }

  // // Call function
  // auto output = CombineChunks(input);
}

TEST_CASE("test_chunk_merge_with_row_id", "[pipeline]") {

}

TEST_CASE("test_chunk_merge_with_null", "[pipeline]") {

}
