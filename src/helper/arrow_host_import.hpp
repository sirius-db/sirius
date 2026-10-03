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
 * @brief Copy one host-memory Arrow record batch (a struct array) to the device as a
 * `cudf::table` whose column types equal `get_cudf_type(types[i])`.
 *
 * The checks run before any buffer is copied, except that a format outside the scalar set
 * (date64, time, binary, string view, nested) is type-checked after its copy:
 * - Shapes the engine cannot consume are refused by name: dictionary encoding, 64-bit offsets
 *   (`large_utf8`, `large_binary`, `large_list`), timezone-aware timestamps, `decimal256`,
 *   struct-level nulls, and a column declared `HUGEINT`/`UHUGEINT` (no 128-bit integer on the GPU).
 * - A scalar column whose format disagrees with its declared type is refused. A `decimal128`
 *   must carry the declared scale and at most the declared precision; it is narrowed to the
 *   declared width after the copy, because Arrow producers emit `decimal128` at any precision.
 * - The struct's own `offset`/`length` window is honoured: `cudf::from_arrow_column` reads each
 *   child by its own offset, so the window is pushed into each child first.
 *
 * Columns are imported one at a time and narrowed before the next is copied, so the transient
 * device footprint is one column at its arriving width, not the whole batch.
 *
 * The copies run on `stream`; the caller synchronizes before letting the producer release the
 * structs. On an error after the copy started, `stream` is synchronized before the throw. The
 * input is never released.
 *
 * @param what Message prefix naming the batch, e.g. `"Arrow batch for stream 3"`.
 * @throws sirius::invalid_input_exception on null or released structs, a non-struct top level, a
 *         column-count mismatch, a window past a child, a refused shape, or a type mismatch. Each
 *         per-column message names the column by index and declared name.
 */
std::unique_ptr<cudf::table> import_arrow_host_table(const ArrowSchema* schema,
                                                     const ArrowArray* array,
                                                     std::string_view what,
                                                     const std::vector<std::string>& names,
                                                     const std::vector<logical_type>& types,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace sirius
