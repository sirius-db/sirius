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

#pragma once

#include "catch.hpp"

#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"

#include <random>

namespace duckdb {

// The buffer manager is shared across all threads so we need to allocate the memory needed by all tests
// upfront when initializating the buffer manager
constexpr size_t TEST_BUFFER_MANAGER_MEMORY_BYTES = 2L * 1024L * 1024L * 1024L; // 2 GB for testing

std::mt19937_64& global_rng();

template <typename T>
T rand_int(T low, T high);

std::string rand_str(int len);

GPUBufferManager* initialize_test_buffer_manager();

void fill_gpu_buffer_with_random_data(uint8_t* gpu_buffer, size_t num_bytes);

shared_ptr<GPUIntermediateRelation> create_table(
  GPUBufferManager* gpu_buffer_manager, const vector<GPUColumnType>& types, const int num_rows,
  uint8_t**& host_data, uint64_t**& host_offset);

void verify_table(GPUBufferManager* gpu_buffer_manager, GPUIntermediateRelation& table,
                  uint8_t** expected_host_data, uint64_t** expected_host_offset);

void free_cpu_buffer(const vector<GPUColumnType>& types, uint8_t** host_data, uint64_t** host_offset);

void verify_cuda_errors(const char *msg); 

void verify_gpu_buffer_equality(uint8_t* buffer_1, uint8_t* buffer_2, size_t num_bytes);

void verify_gpu_column_equality(shared_ptr<GPUColumn> col1, shared_ptr<GPUColumn> col2);

shared_ptr<GPUColumn> create_column_with_random_data(GPUColumnTypeId col_type, size_t num_records, 
  size_t chars_per_record = 1, size_t num_materialize_row_ids = 0, bool has_null_mask = false);

}
