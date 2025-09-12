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
#include "log/logging.hpp"

using namespace duckdb;

constexpr size_t CPU_CACHE_TEST_MEM_SF = 8; // Factor used to determine the initial size of the CPU cache
size_t calculate_test_cpu_cache_size(size_t bytes_to_cache) {
   return std::pow(2.0, std::ceil(std::log2(CPU_CACHE_TEST_MEM_SF * bytes_to_cache)));
}

TEST_CASE("test_cpu_cache_basic_fixed_single_col", "[cpu_cache]") {
    // Initialize the buffer manager
    size_t num_records = 1024;
    GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

    // Create a GPU column representing a single integer column
    duckdb::shared_ptr<GPUColumn> gpu_column = create_column_with_random_data(GPUColumnTypeId::INT32, num_records);
    duckdb::shared_ptr<GPUIntermediateRelation> relationship = make_shared_ptr<GPUIntermediateRelation>(1);
    relationship->columns[0] = gpu_column;

    // Now cache the column to CPU
    size_t cpu_cache_bytes = calculate_test_cpu_cache_size(2 * gpu_column->getTotalColumnSize());
    MallocCPUCache cpu_cache(cpu_cache_bytes, 1);
    uint32_t chunk_id = cpu_cache.moveDataToCPU(relationship);
    REQUIRE(chunk_id == 0);

    // Cache the column again and make sure we get an incremented chunk id
    uint32_t copy_chunk_id = cpu_cache.moveDataToCPU(relationship);
    REQUIRE(copy_chunk_id == 1);

    // Now load the column and the duplicate back from the CPU cache to the GPU
    duckdb::shared_ptr<GPUIntermediateRelation> loaded_relationship = cpu_cache.moveDataToGPU(chunk_id, true);
    REQUIRE(loaded_relationship->columns.size() == 1);
    duckdb::shared_ptr<GPUColumn> reloaded_column = loaded_relationship->columns[0];

    duckdb::shared_ptr<GPUIntermediateRelation> copy_loaded_relationship = cpu_cache.moveDataToGPU(copy_chunk_id, false);
    REQUIRE(copy_loaded_relationship->columns.size() == 1);
    duckdb::shared_ptr<GPUColumn> copy_reloaded_column = copy_loaded_relationship->columns[0];

    // Verify that we got the expected result
    REQUIRE(reloaded_column->segment_id == -1);
    verify_cuda_errors("CUDA Errors in CPU Caching Test");
    verify_gpu_column_equality(reloaded_column, gpu_column);
    verify_gpu_column_equality(copy_reloaded_column, gpu_column);

    // Verify that we get an error trying to move an evicted column but can remove a non evicted column
    REQUIRE_NOTHROW(cpu_cache.moveDataToGPU(copy_chunk_id, true));
    REQUIRE_THROWS(cpu_cache.moveDataToGPU(chunk_id, true));
}

TEST_CASE("test_cpu_cache_basic_string_single_col", "[cpu_cache]") {
    // Initialize the buffer manager
    size_t num_records = 1024;
    size_t chars_per_record = 8;
    GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

    // Now create a GPU column representing a single string column
    duckdb::shared_ptr<GPUColumn> gpu_column = create_column_with_random_data(GPUColumnTypeId::VARCHAR, num_records, chars_per_record);
    duckdb::shared_ptr<GPUIntermediateRelation> gpu_relationship = make_shared_ptr<GPUIntermediateRelation>(1);
    gpu_relationship->columns[0] = gpu_column;

    // Now cache the column to CPU
    size_t cpu_cache_bytes = calculate_test_cpu_cache_size(gpu_column->getTotalColumnSize());
    MallocCPUCache cpu_cache(cpu_cache_bytes, 1);
    uint32_t chunk_id = cpu_cache.moveDataToCPU(gpu_relationship);
    REQUIRE(chunk_id == 0);

    // Now load the column back from the CPU cache to the GPU
    duckdb::shared_ptr<GPUIntermediateRelation> loaded_relationship = cpu_cache.moveDataToGPU(chunk_id, true);
    REQUIRE(loaded_relationship->columns.size() == 1);
    duckdb::shared_ptr<GPUColumn> reloaded_column = loaded_relationship->columns[0];

    // Verify that we got the expected result
    verify_cuda_errors("CUDA Errors in CPU Caching Test");
    verify_gpu_column_equality(reloaded_column, gpu_column);
}

TEST_CASE("test_cpu_cache_multiple_col_diff_streams", "[cpu_cache]") {
    // Initialize the buffer manager
    GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

    // Create a relationship with multiple string and integer columns
    size_t num_records = 1024; size_t chars_per_record = 8;
    size_t num_int_cols = 2; size_t num_string_cols = 2;

    size_t num_total_cols = num_int_cols + num_string_cols;
    duckdb::shared_ptr<GPUIntermediateRelation> gpu_relationship = make_shared_ptr<GPUIntermediateRelation>(num_total_cols);

    size_t total_relationship_bytes = 0;
    for(size_t i = 0; i < num_int_cols; i++) { 
        gpu_relationship->columns[i] = create_column_with_random_data(GPUColumnTypeId::INT32, num_records);
        total_relationship_bytes += gpu_relationship->columns[i]->getTotalColumnSize();
    }
    for(size_t i = 0; i < num_string_cols; i++) { 
        gpu_relationship->columns[num_int_cols + i] = create_column_with_random_data(GPUColumnTypeId::VARCHAR, num_records, chars_per_record);
        total_relationship_bytes += gpu_relationship->columns[num_int_cols + i]->getTotalColumnSize();
    }

    // Create two caches - one with single stream and the other with multiple streams
    size_t cpu_cache_bytes = calculate_test_cpu_cache_size(total_relationship_bytes);
    MallocCPUCache single_stream_cpu_cache(cpu_cache_bytes, 1);
    MallocCPUCache multiple_stream_cpu_cache(cpu_cache_bytes, num_total_cols);

    // Cache the column on both of the caches
    uint32_t single_chunk_id = single_stream_cpu_cache.moveDataToCPU(gpu_relationship);
    REQUIRE(single_chunk_id == 0);
    uint32_t multiple_chunk_id = multiple_stream_cpu_cache.moveDataToCPU(gpu_relationship);
    REQUIRE(multiple_chunk_id == 0); 

    // Now load the column back from the CPU caches to the GPU
    duckdb::shared_ptr<GPUIntermediateRelation> single_loaded_relationship = single_stream_cpu_cache.moveDataToGPU(single_chunk_id, true);
    REQUIRE(single_loaded_relationship->columns.size() == num_total_cols);
    duckdb::shared_ptr<GPUIntermediateRelation> multi_loaded_relationship = multiple_stream_cpu_cache.moveDataToGPU(multiple_chunk_id, true);
    REQUIRE(multi_loaded_relationship->columns.size() == num_total_cols);

    // Verify that we got the expected result by comparing column by column
    verify_cuda_errors("CUDA Errors in CPU Caching Test");
    for(size_t i = 0; i < num_total_cols; i++) { 
        verify_gpu_column_equality(single_loaded_relationship->columns[i], gpu_relationship->columns[i]);
        verify_gpu_column_equality(multi_loaded_relationship->columns[i], gpu_relationship->columns[i]);
    }
}

TEST_CASE("test_cpu_cache_multiple_cols_with_only_row_ids", "[cpu_cache]") {
    // Initialize the buffer manager
    GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

    // Create a relationship with multiple string and integer columns
    size_t num_records = 1024; size_t chars_per_record = 8;
    size_t num_int_cols = 2; size_t num_string_cols = 2;

    size_t num_total_cols = num_int_cols + num_string_cols;
    duckdb::shared_ptr<GPUIntermediateRelation> gpu_relationship = make_shared_ptr<GPUIntermediateRelation>(num_total_cols);

    size_t total_relationship_bytes = 0;
    for(size_t i = 0; i < num_int_cols; i++) { 
        size_t num_row_ids = rand() % num_records;
        gpu_relationship->columns[i] = create_column_with_random_data(GPUColumnTypeId::INT32, num_records, 1, num_row_ids);
        total_relationship_bytes += gpu_relationship->columns[i]->getTotalColumnSize();
    }
    for(size_t i = 0; i < num_string_cols; i++) { 
        size_t num_row_ids = rand() % num_records;
        gpu_relationship->columns[num_int_cols + i] = create_column_with_random_data(GPUColumnTypeId::VARCHAR, num_records, chars_per_record, num_row_ids);
        total_relationship_bytes += gpu_relationship->columns[num_int_cols + i]->getTotalColumnSize();
    }

    // Create two caches - one with single stream and the other with multiple streams
    size_t cpu_cache_bytes = calculate_test_cpu_cache_size(total_relationship_bytes);
    MallocCPUCache cpu_cache(cpu_cache_bytes, num_total_cols);

    // Cache the column on both of the caches
    uint32_t chunk_id = cpu_cache.moveDataToCPU(gpu_relationship);
    REQUIRE(chunk_id == 0); 

    // Now load the column back from the CPU caches to the GPU
    duckdb::shared_ptr<GPUIntermediateRelation> loaded_relationship = cpu_cache.moveDataToGPU(chunk_id, true);
    REQUIRE(loaded_relationship->columns.size() == num_total_cols);

    // Verify that we got the expected result by comparing column by column
    verify_cuda_errors("CUDA Errors in CPU Caching Test");
    for(size_t i = 0; i < num_total_cols; i++) { 
        verify_gpu_column_equality(loaded_relationship->columns[i], gpu_relationship->columns[i]);
    }
}

TEST_CASE("test_cpu_cache_multiple_cols_with_only_validitiy_mask", "[cpu_cache]") {
    // Initialize the buffer manager
    GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

    // Create a relationship with multiple string and integer columns
    size_t num_records = 1024; size_t chars_per_record = 8;
    size_t num_int_cols = 2; size_t num_string_cols = 2;

    size_t num_total_cols = num_int_cols + num_string_cols;
    duckdb::shared_ptr<GPUIntermediateRelation> gpu_relationship = make_shared_ptr<GPUIntermediateRelation>(num_total_cols);

    size_t total_relationship_bytes = 0;
    for(size_t i = 0; i < num_int_cols; i++) { 
        gpu_relationship->columns[i] = create_column_with_random_data(GPUColumnTypeId::INT32, num_records, 1, 0, true);
        total_relationship_bytes += gpu_relationship->columns[i]->getTotalColumnSize();
    }
    for(size_t i = 0; i < num_string_cols; i++) { 
        gpu_relationship->columns[num_int_cols + i] = create_column_with_random_data(GPUColumnTypeId::VARCHAR, num_records, chars_per_record, 0, true);
        total_relationship_bytes += gpu_relationship->columns[num_int_cols + i]->getTotalColumnSize();
    }

    // Create two caches - one with single stream and the other with multiple streams
    size_t cpu_cache_bytes = calculate_test_cpu_cache_size(total_relationship_bytes);
    MallocCPUCache cpu_cache(cpu_cache_bytes, num_total_cols);

    // Cache the column on both of the caches
    uint32_t chunk_id = cpu_cache.moveDataToCPU(gpu_relationship);
    REQUIRE(chunk_id == 0); 

    // Now load the column back from the CPU caches to the GPU
    duckdb::shared_ptr<GPUIntermediateRelation> loaded_relationship = cpu_cache.moveDataToGPU(chunk_id, true);
    REQUIRE(loaded_relationship->columns.size() == num_total_cols);

    // Verify that we got the expected result by comparing column by column
    verify_cuda_errors("CUDA Errors in CPU Caching Test");
    for(size_t i = 0; i < num_total_cols; i++) { 
        verify_gpu_column_equality(loaded_relationship->columns[i], gpu_relationship->columns[i]);
    }
}

TEST_CASE("test_cpu_cache_multiple_cols_with_row_ids_and_validitiy_mask", "[cpu_cache]") {
    // Initialize the buffer manager
    GPUBufferManager* gpuBufferManager = initialize_test_buffer_manager();

    // Create a relationship with multiple string and integer columns
    size_t num_records = 1024; size_t chars_per_record = 8;
    size_t num_int_cols = 2; size_t num_string_cols = 2;

    size_t num_total_cols = num_int_cols + num_string_cols;
    duckdb::shared_ptr<GPUIntermediateRelation> gpu_relationship = make_shared_ptr<GPUIntermediateRelation>(num_total_cols);

    size_t total_relationship_bytes = 0;
    for(size_t i = 0; i < num_int_cols; i++) { 
        size_t num_row_ids = rand() % num_records;
        gpu_relationship->columns[i] = create_column_with_random_data(GPUColumnTypeId::INT32, num_records, 1, num_row_ids, true);
        total_relationship_bytes += gpu_relationship->columns[i]->getTotalColumnSize();
    }
    for(size_t i = 0; i < num_string_cols; i++) { 
        size_t num_row_ids = rand() % num_records;
        gpu_relationship->columns[num_int_cols + i] = create_column_with_random_data(GPUColumnTypeId::VARCHAR, num_records, chars_per_record, num_row_ids, true);
        total_relationship_bytes += gpu_relationship->columns[num_int_cols + i]->getTotalColumnSize();
    }

    // Create two caches - one with single stream and the other with multiple streams
    size_t cpu_cache_bytes = calculate_test_cpu_cache_size(total_relationship_bytes);
    MallocCPUCache cpu_cache(cpu_cache_bytes, num_total_cols);

    // Cache the column on both of the caches
    uint32_t chunk_id = cpu_cache.moveDataToCPU(gpu_relationship);
    REQUIRE(chunk_id == 0); 

    // Now load the column back from the CPU caches to the GPU
    duckdb::shared_ptr<GPUIntermediateRelation> loaded_relationship = cpu_cache.moveDataToGPU(chunk_id, true);
    REQUIRE(loaded_relationship->columns.size() == num_total_cols);

    // Verify that we got the expected result by comparing column by column
    verify_cuda_errors("CUDA Errors in CPU Caching Test");
    for(size_t i = 0; i < num_total_cols; i++) { 
        verify_gpu_column_equality(loaded_relationship->columns[i], gpu_relationship->columns[i]);
    }
}