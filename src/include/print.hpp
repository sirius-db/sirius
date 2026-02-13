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

#include <cudf/table/table_view.hpp>

namespace cucascade {
class data_batch;
}

namespace duckdb {

template <typename T>
void printGPUColumn(T* a, size_t N, int gpu);

void printGPUTable(GPUIntermediateRelation& table, ClientContext& context);

}  // namespace duckdb

namespace sirius {

/**
 * Print the contents of a cudf::table_view to stdout (printf).
 * Copies device data to host. Supports common numeric types; other types
 * print as "(unprinted)". Limits to the first max_rows rows (default 20).
 */
void print_table_contents(cudf::table_view const& table, cudf::size_type max_rows = 20);

/**
 * Print the contents of a cucascade::data_batch to stdout (printf).
 * Equivalent to printing the underlying cudf table view.
 */
void print_data_batch_contents(cucascade::data_batch const& batch, cudf::size_type max_rows = 20);

}  // namespace sirius
