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

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace duckdb {
namespace sirius {

// Returns an INT32 column where result[i] = length in bytes of the string at position i.
// Computed via offset arithmetic: result[i] = offsets[i+1] - offsets[i].
// Equivalent to LENGTH(col) but avoids materializing the string chars buffer.
std::unique_ptr<cudf::column> StrlenFromOffsets(const uint64_t* offsets,
                                                const uint64_t* row_ids,
                                                size_t num_rows,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace sirius
}  // namespace duckdb
