/*
 * Copyright 2026, Sirius Contributors.
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

// Public entry point of the generator kernels: what the operator and the tests call. The hash
// helpers and the per-column steps behind it are in generator_detail.cuh.

#include "op/scan/generator/generator_spec.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::op::scan::generator {

/// Generates batch `batch_index`: columns in spec order; INTEGER -> INT32, BIGINT -> INT64,
/// DOUBLE -> FLOAT64, VARCHAR -> STRING (see "VARCHAR" in generator_spec.hpp). Defined in
/// generator_batch.cu.
/// @throws std::out_of_range if batch_index >= spec.num_batches.
/// @throws std::invalid_argument if rows_per_batch exceeds a cudf column or a column fails
/// validation_error.
std::unique_ptr<cudf::table> generate_batch(const generator_spec& spec,
                                            std::uint64_t batch_index,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/// Generates only `output_columns` of batch `batch_index`, in the order given. Each entry is an
/// index into `spec.columns` and is also the column index the values are hashed with, so every
/// output column is identical to that column of the full batch; the other columns are never
/// drawn. Defined in generator_batch.cu.
/// @throws std::out_of_range if batch_index >= spec.num_batches or an entry is not a spec column.
/// @throws std::invalid_argument if rows_per_batch exceeds a cudf column or an output column
/// fails validation_error.
std::unique_ptr<cudf::table> generate_batch(const generator_spec& spec,
                                            const std::vector<std::size_t>& output_columns,
                                            std::uint64_t batch_index,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

}  // namespace sirius::op::scan::generator
