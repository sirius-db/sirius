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

#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/main/query_result.hpp>

namespace sirius::exec {

// Exports a materialized result. The stream owns its state until release; exported
// schemas and arrays have independent ownership and must also be released.
void export_arrow_result(duckdb::unique_ptr<duckdb::QueryResult> result,
                         ArrowArrayStream& out,
                         duckdb::idx_t batch_size);

}  // namespace sirius::exec
