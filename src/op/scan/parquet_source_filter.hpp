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

#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/table_filter.hpp>

namespace sirius::op::scan {

// UNKNOWN means unsupported evaluation, distinct from SQL NULL. Only FALSE/SQL_NULL
// may eliminate a source in a WHERE conjunct; the full residual remains authoritative.
enum class source_filter_result { UNKNOWN, MATCH, NO_MATCH, SQL_NULL };

[[nodiscard]] source_filter_result evaluate_parquet_source_filter(duckdb::TableFilter const& filter,
                                                                  duckdb::Value const& value);

}  // namespace sirius::op::scan
