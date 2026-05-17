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

#include "helper/logical_type.hpp"
#include "op/scan/duckdb_native_metadata.hpp"

#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace sirius::op::scan {

struct duckdb_native_scan_info {
  duckdb::DataTable* storage = nullptr;
  duckdb::ClientContext* context = nullptr;

  std::vector<projected_column> projected_cols;
  std::vector<sirius::logical_type> projected_types;

  std::string db_path;

  std::size_t approximate_batch_size = 0;
};

}  // namespace sirius::op::scan
