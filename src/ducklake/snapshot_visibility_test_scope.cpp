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

#include "ducklake/snapshot_visibility_test_scope.hpp"

#include <mutex>
#include <unordered_map>
namespace sirius::ducklake::testing {
namespace {
struct scope_registry {
  std::mutex mutex;
  std::unordered_map<duckdb::DatabaseInstance const*, size_t> counts;
};
scope_registry& registry()
{
  static scope_registry value;
  return value;
}
}  // namespace
bool snapshot_visibility_probe_enabled(duckdb::DatabaseInstance const& database)
{
  auto& state = registry();
  std::lock_guard lock(state.mutex);
  return state.counts.contains(&database);
}
scoped_snapshot_visibility_probe::scoped_snapshot_visibility_probe(
  duckdb::DatabaseInstance const& database)
  : database_(&database)
{
  auto& state = registry();
  std::lock_guard lock(state.mutex);
  ++state.counts[database_];
}
scoped_snapshot_visibility_probe::~scoped_snapshot_visibility_probe()
{
  auto& state = registry();
  std::lock_guard lock(state.mutex);
  auto entry = state.counts.find(database_);
  if (--entry->second == 0) state.counts.erase(entry);
}
}  // namespace sirius::ducklake::testing
