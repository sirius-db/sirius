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
#include "utils/utils.hpp"

#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"
#include "cpu_cache.hpp"

using namespace duckdb;

TEST_CASE("test_cpu_cache_basic_fixed", "[cpu_cache]") {
    // Initialize the buffer manager
    size_t num_records = 1024;
    size_t column_bytes = num_records * sizeof(int32_t);
    size_t memory_buffer_sizes = 2 * column_bytes;
    GPUBufferManager::GetInstance(memory_buffer_sizes, memory_buffer_sizes, memory_buffer_sizes);

    // Create a cpu cache with enough pinned memory and one stream
    MallocCPUCache cpu_cache(memory_buffer_sizes, 1);

    // Now create a GPU column representing a single integer column
    int32_t* column_data = callCudaMalloc<int32_t>(num_records, 0);
    duckdb::shared_ptr<GPUColumn> gpu_column = make_shared_ptr<GPUColumn>(num_records, GPUColumnType(GPUColumnTypeId::INT32), (uint8_t*) column_data, nullptr);
    duckdb::shared_ptr<GPUIntermediateRelation> relationship = make_shared_ptr<GPUIntermediateRelation>(1);
    relationship->columns[0] = gpu_column;

    // Now cache the column to CPU
    uint32_t chunk_id = cpu_cache.moveDataToCPU(relationship);
    REQUIRE(chunk_id == 0);

    // Now load the column back from the CPU cache to the GPU
    duckdb::shared_ptr<GPUIntermediateRelation> loaded_relationship = cpu_cache.moveDataToGPU(chunk_id, true);
    REQUIRE(loaded_relationship->columns.size() == 1);
    duckdb::shared_ptr<GPUColumn> cpu_column = loaded_relationship->columns[0];

    // Verify that we got the expected result
    verify_cuda_errors("CUDA Errors in Caching Test");
    verify_gpu_column_equality(cpu_column, gpu_column);

    // Cleanup all allocated memory
    callCudaFree<int32_t>(column_data, 0);
}