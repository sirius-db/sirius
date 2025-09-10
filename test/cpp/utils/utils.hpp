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

#include "gpu_buffer_manager.hpp"

#include <random>

namespace duckdb {

std::mt19937_64& global_rng();

template <typename T>
T rand_int(T low, T high);

std::string rand_str(int len);

shared_ptr<GPUIntermediateRelation> create_table(
  GPUBufferManager* gpuBufferManager, const vector<GPUColumnType>& types, const int num_rows,
  uint8_t**& host_data, uint64_t**& host_offset);

void verify_table(GPUBufferManager* gpuBufferManager, GPUIntermediateRelation& table,
                  uint8_t** expected_host_data, uint64_t** expected_host_offset);

void free_buffer(GPUBufferManager* gpuBufferManager, const vector<GPUColumnType>& types,
                 uint8_t** host_data, uint64_t** host_offset);

}
