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

#include "helper/logical_type.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Arrow C Data Interface structs, forward-declared so this header needs no Arrow header. The .cpp
// uses DuckDB's layout-identical definition, the one this library already links.
struct ArrowSchema;
struct ArrowArray;

namespace sirius {

/**
 * @brief Copy one host Arrow record batch (a struct array) to the device as a `cudf::table` whose
 * column types equal `get_cudf_type(types[i])`. Columns bind by position.
 *
 * Refused before any copy: dictionary encoding, 64-bit offsets, timezone-aware timestamps,
 * `decimal256`, struct-level nulls, a column declared `HUGEINT`, `UHUGEINT` or nested, a scalar
 * format that disagrees with its declared type, and a `decimal128` with another scale or a larger
 * precision. Other formats (date64, time, binary, string view, nested) are type-checked after
 * their copy. A `decimal128` is narrowed to the declared width before the next column is copied.
 * The struct's `offset`/`length` window is honoured, which `cudf::from_arrow_column` ignores.
 *
 * Copies run on `stream`; the caller syncs it before the producer releases the structs. A throw
 * after a copy started syncs `stream` first. The input is never released.
 *
 * @param what Message prefix naming the batch.
 * @param names Declared column names, for messages; as many as `types`.
 * @param mr Allocates the returned columns.
 * @throws sirius::invalid_input_exception on a refused batch, naming the column.
 */
std::unique_ptr<cudf::table> import_arrow_host_table(const ArrowSchema* schema,
                                                     const ArrowArray* array,
                                                     std::string_view what,
                                                     const std::vector<std::string>& names,
                                                     const std::vector<logical_type>& types,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace sirius
