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

#include "op/scan/duckdb_native_scan_info.hpp"

#include "scan_manager/duckdb_native_split_provider.hpp"

#include <memory>
#include <utility>

namespace sirius::op::scan {

std::unique_ptr<scan_manager::split_provider> duckdb_native_scan_info::make_provider(
  scan_manager::sirius_scan_manager& manager,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs)
{
  // The .duckdb-native path routes through a single local ioctx, not S3 path
  // routing, so the scan_manager is unused here — unlike parquet/S3, which use
  // it via io_ctx_shared_for(path). It stays in the signature only to satisfy
  // the combined-runtime scan_info::make_provider(manager, gpu_ioctxs) contract.
  (void)manager;
  // duckdb_native_scan_task currently uses a single ioctx (Phase B lift is per
  // single-GPU). Pick the first ioctx in the map; pass nullptr when empty so
  // the split_provider falls through to BufferManager::Pin.
  auto io_ctx = gpu_ioctxs.empty() ? nullptr : gpu_ioctxs.begin()->second;
  return std::make_unique<scan_manager::duckdb_native_split_provider>(std::move(*this),
                                                                      std::move(io_ctx));
}

}  // namespace sirius::op::scan
