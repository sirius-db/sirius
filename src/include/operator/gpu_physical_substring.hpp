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

#include "gpu_columns.hpp"
#include <tuple>

namespace duckdb {

std::tuple<char*, uint64_t*, uint64_t> PerformSubstring(char* char_data, uint64_t* str_indices, uint64_t num_chars, uint64_t num_strings, uint64_t start_idx, uint64_t length);
shared_ptr<GPUColumn> HandleSubString(shared_ptr<GPUColumn> string_column, uint64_t start_idx, uint64_t length);

// For CuDF compatibility
std::unique_ptr<cudf::column> DoSubstring(const char* input_data,
                                          cudf::size_type input_count,
                                          const int64_t* input_offsets,
                                          int64_t start_idx,
                                          int64_t length,
                                          rmm::device_async_resource_ref mr,
                                          rmm::cuda_stream_view stream = rmm::cuda_stream_default);

} // namespace duckdb