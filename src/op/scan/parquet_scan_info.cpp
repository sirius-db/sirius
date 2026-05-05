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

#include "op/scan/parquet_scan_info.hpp"

#include "scan_manager/parquet_split_provider.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <memory>
#include <utility>

namespace sirius::op::scan {

std::unique_ptr<scan_manager::split_provider> parquet_scan_info::make_provider(
  scan_manager::sirius_scan_manager& manager)
{
  return std::make_unique<scan_manager::parquet_split_provider>(
    returned_types,
    file_paths,
    column_ids,
    projection_ids,
    names,
    scan_output_arity,
    std::move(table_filters),
    partition_indices,
    approximate_batch_size,
    scan_manager::parquet_split_provider::DEFAULT_MAX_FILE_PROCESSED,
    manager.shared_io_ctx());
}

}  // namespace sirius::op::scan
