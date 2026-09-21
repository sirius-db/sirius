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

#include <cstddef>
#include <string>

namespace sirius::op::scan {

/**
 * @brief Hive partition column metadata (not in parquet file)
 */
struct hive_partition_column {
  std::string column_name;          ///< Partition column name (e.g. "year")
  std::size_t duckdb_column_index;  ///< Index into scan_op->names / column_ids
};

}  // namespace sirius::op::scan
