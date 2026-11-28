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

#include "duckdb/common/typedefs.hpp"

#include <vector>

namespace duckdb {
class ClientContext;
class LogicalGet;
}  // namespace duckdb

namespace sirius::planner {

/// The output positions of the scan @p get that are each proven to hold no value twice and no
/// NULL: a column the pinned copy of a Parquet scan's files proved distinct at pin time, which
/// counts only with pin_table_prove_unique_keys, or a native table's single-column PRIMARY KEY.
/// Empty when nothing is proven. A filter or a projection above the scan keeps the fact: a subset
/// of distinct values is distinct.
std::vector<duckdb::idx_t> proven_unique_scan_outputs(duckdb::ClientContext& context,
                                                      duckdb::LogicalGet& get);

}  // namespace sirius::planner
