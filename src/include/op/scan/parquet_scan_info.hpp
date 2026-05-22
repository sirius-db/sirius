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

#include "op/scan/scan_info.hpp"

#include <memory>

namespace sirius::op::scan {

/**
 * @brief Bind-data for a parquet scan.
 *
 * All scan fields are inherited from @ref scan_info; this subclass exists so
 * @ref make_provider dispatches to @ref scan_manager::parquet_split_provider.
 *
 * Populated once during plan generation and parked on the gpu parquet scan
 * operator. The scan_manager reads it during prepare_for_query.
 */
struct parquet_scan_info : scan_info {
  std::unique_ptr<scan_manager::split_provider> make_provider(
    scan_manager::sirius_scan_manager& manager,
    std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs) override;
};

}  // namespace sirius::op::scan
