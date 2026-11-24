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
#include <functional>
#include <memory>
#include <vector>

namespace paimon {
class FileSystem;
class MemoryPool;
}  // namespace paimon
namespace sirius::op::scan {
// Opt-in for a bound local, immutable snapshot on the qualified libstdc++ SDK.
// The caller supplies the selected local filesystem and an accounting lease for the
// SDK's ordinary heap buffer. An empty lease means the query cannot afford the input.
struct paimon_snapshot_input_budget {
  std::shared_ptr<paimon::FileSystem> filesystem;
  std::function<std::shared_ptr<void>(uint64_t)> try_reserve;
};

// Run on the serialized native lane, after provider authentication and resource admission.
// pool bounds only allocations routed through it; this function establishes no all-native quota.
// No cache or background prefetch is shared. All native owners outlive reader/root cleanup.
// Null means validated empty-at-bind or a genuinely empty retained plan, never an SDK error.
std::unique_ptr<paimon_batch_cursor> open_paimon_read(
  paimon_read_metadata const&,
  std::vector<int32_t> const& stable_field_ids,
  std::shared_ptr<paimon::MemoryPool>,
  std::shared_ptr<void> native_modules,
  int64_t max_batch_rows,
  scan_manager::preparation_stop_token,
  paimon_snapshot_input_budget const* = nullptr);
}  // namespace sirius::op::scan
