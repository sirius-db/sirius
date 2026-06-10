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

#include "io/gpu_ingestible.hpp"

#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace sirius::scan_manager {

class sirius_scan_manager;
struct pinned_entry;

/**
 * @brief Single entry point for producing the right gpu_ingestible for a
 *        scan source during @c sirius_scan_manager::prepare_for_query.
 *
 * Consolidates two concerns that used to be split across
 * @c sirius_scan_manager::create_ingestible_for and
 * @c sirius_scan_manager::try_make_cached_ingestible:
 *   1. Cache short-circuit — match the operator's bind data against a
 *      pinned_entry and, on hit, build a @c pinned_table_gpu_ingestible.
 *   2. Format dispatch — on cache miss, hand off to
 *      @c io::make_gpu_ingestible, which routes through the bind data's
 *      virtual factory.
 *
 * The factory holds a borrowed reference to scan_manager's pinned-entries
 * map. Per-query state (gpu_memory_spaces, scan_manager reference) is
 * passed in each @ref produce call, so the factory itself
 * has a stable lifetime — it's held as a scan_manager member and lives
 * with the manager.
 *
 * The cache-match logic today only fires on @c parquet_ingestible_table_info
 * binds (the only @c pin_table source). Future formats add a second
 * dynamic_cast branch in @c try_cached without touching the
 * @c pinned_table_gpu_ingestible class.
 */
class gpu_ingestible_factory {
 public:
  /// @param pinned_entries  Borrowed reference to the scan_manager's
  ///                        pinned-entries map. The factory must not
  ///                        outlive the map.
  explicit gpu_ingestible_factory(
    std::unordered_map<std::string, pinned_entry> const& pinned_entries) noexcept;

  /**
   * @brief Build the right gpu_ingestible for an operator's bind data.
   *
   * Steals @p table_info on cache hit (the cached ingestible co-owns the
   * bind data via its base @c gpu_ingestible::_table_info). On miss,
   * @p table_info is forwarded to @c io::make_gpu_ingestible.
   *
   * Returns nullptr only when @p table_info itself is null (caller error).
   *
   * @param table_info        Operator's bind data, taken via @c op->take_table_info().
   * @param mgr               Scan manager reference; forwarded to
   *                          @c io::make_gpu_ingestible for ioctx routing.
   * @param gpu_memory_spaces device_id -> GPU memory_space lookup; used by
   *                          the HOST-tier cached path to materialize host
   *                          chunks onto the executing GPU.
   * @param op_id             Operator id (logging only).
   */
  std::shared_ptr<io::gpu_ingestible> produce(
    std::unique_ptr<io::ingestible_table_info> table_info,
    sirius_scan_manager const& mgr,
    std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces,
    std::size_t op_id);

 private:
  /// Try the pinned-cache short-circuit. Peeks @p table_info; on a hit,
  /// steals it into the constructed @c pinned_table_gpu_ingestible.
  /// Returns nullptr on miss (table_info stays intact).
  std::shared_ptr<io::gpu_ingestible> try_cached(
    std::unique_ptr<io::ingestible_table_info>& table_info,
    std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces,
    std::size_t op_id) const;

  std::unordered_map<std::string, pinned_entry> const& _pinned_entries;
};

}  // namespace sirius::scan_manager
