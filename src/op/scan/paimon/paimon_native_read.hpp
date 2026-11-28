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
#include "op/scan/paimon/paimon_batch_cursor.hpp"
#include "op/scan/paimon/paimon_read_metadata.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace paimon {
class FileSystem;
class MemoryPool;
}  // namespace paimon
namespace sirius::op::scan {
// Run on the serialized native lane, after provider authentication and resource admission.
// pool bounds only allocations routed through it; this function establishes no all-native quota.
// An explicit filesystem is shared by planning and reading. No Sirius Parquet cache
// identity or background prefetch is shared. Native owners outlive reader/root cleanup.
// Null means validated empty-at-bind or a genuinely empty retained plan, never an SDK error.
std::unique_ptr<paimon_batch_cursor> open_paimon_read(
  paimon_read_metadata const&,
  std::vector<int32_t> const& stable_field_ids,
  std::shared_ptr<paimon::MemoryPool>,
  std::shared_ptr<void> native_modules,
  int64_t max_batch_rows,
  scan_manager::preparation_stop_token,
  std::shared_ptr<paimon::FileSystem> filesystem = {});
}  // namespace sirius::op::scan
