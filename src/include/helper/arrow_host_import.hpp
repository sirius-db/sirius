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

// Arrow C Data Interface structs. Forward-declared only, so this header needs no Arrow header
// and can sit next to any definition of them (DuckDB's `duckdb/common/arrow/arrow.hpp`, Apache
// Arrow's `arrow/c/abi.h`, a vendored copy — all under the shared ARROW_C_DATA_INTERFACE guard,
// all layout-identical). The .cpp uses DuckDB's, the one definition this library already has.
struct ArrowSchema;
struct ArrowArray;

namespace sirius {

/**
 * @brief Import one host-memory Arrow struct array (a record batch) into a `cudf::table` and
 * reconcile it against the column types a stream was declared with.
 *
 * The engine reads a stream's columns through the schema it was declared with, so a
 * disagreement between the declaration and the arriving batch must be a loud error at the entry
 * point, not reinterpreted bits downstream — the same rule `Fragment::push_packed` applies with
 * `sirius::get_cudf_type` column by column. On top of that guard this helper:
 *
 * - refuses **by name**, before any buffer is touched, the shapes cudf would import into
 *   something the engine cannot consume or would import with a changed meaning: dictionary
 *   encoding, `large_list` / `large_utf8` / `large_binary` (64-bit offsets — the engine's string
 *   and list kernels take 32-bit offsets), timezone-aware timestamps (no timezone carrier on the
 *   GPU), `decimal256`, and a column declared `HUGEINT`/`UHUGEINT` (cudf has no 128-bit integer;
 *   `get_cudf_type` would narrow it to 64 bits). Only top-level columns are inspected;
 * - picks the decimal storage width from the **declared** precision (DECIMAL32/64/128 with cudf's
 *   negated scale) and narrows an arriving decimal128 to it, since Arrow producers emit
 *   decimal128 whatever the precision. Values that overflow the declared precision are the
 *   producer's schema violation and are truncated by the cast;
 * - relies on cudf for the remaining normalizations the proposal lists: the Arrow bool bitmap
 *   becomes BOOL8, `utf8` becomes STRING with 32-bit offsets, `date32` becomes TIMESTAMP_DAYS.
 *
 * The buffers are copied to the device on `stream`; the caller synchronizes before it lets the
 * producer release the Arrow structs. The input is never released by this function.
 *
 * @param schema Arrow schema of the struct array (`+s`), one child per declared column.
 * @param array  Arrow array in host memory whose children are the columns.
 * @param what   Message prefix naming the batch, e.g. `"Arrow batch for stream 3"`.
 * @param names  Declared column names, used only in messages.
 * @param types  Declared column types; the import is reconciled against `get_cudf_type` of each.
 * @param stream CUDA stream for the host-to-device copies and the decimal casts.
 * @param mr     Device memory resource the table is allocated from.
 * @return The imported table, column types equal to `get_cudf_type(types[i])` for every `i`.
 * @throws sirius::invalid_input_exception on null pointers, on structs that were already
 *         released (`release == NULL`), a non-struct top-level array, a column-count mismatch,
 *         a refused shape, or a type mismatch (a fixed-point scale mismatch names both scales);
 *         every per-column message names the column by index and declared name.
 */
std::unique_ptr<cudf::table> import_arrow_host_table(const ArrowSchema* schema,
                                                     const ArrowArray* array,
                                                     std::string_view what,
                                                     const std::vector<std::string>& names,
                                                     const std::vector<logical_type>& types,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace sirius
