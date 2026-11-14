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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>

namespace sirius {
namespace ast {
struct cast;
}

/// Whether the observed carrier of a direct reference proves a checked SQL cast lossless.
/// Used with the planned carrier for costing and with the actual input carrier for execution.
[[nodiscard]] bool checked_integer_cast_fits_carrier(ast::cast const& cast,
                                                     cudf::data_type carrier);

/// Convert native integer carriers, nulling overflowing rows for TRY_CAST or requesting
/// DuckDB replay for CAST. The caller handles HUGEINT's fixed-point representation separately.
[[nodiscard]] std::unique_ptr<cudf::column> cast_checked_integer(cudf::column_view const& input,
                                                                 cudf::data_type target,
                                                                 bool try_cast,
                                                                 ::cuda::stream_ref stream,
                                                                 rmm::device_async_resource_ref mr);

}  // namespace sirius
