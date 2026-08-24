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

#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace sirius::compression {

/**
 * @brief Device-allocation accounting for the spill-compression path.
 *
 * The question this answers is "what does compressing a spill cost the
 * allocator" — how many device allocations the encode adds, how large they are,
 * and how much is outstanding at once. Without it the only visible symptom of a
 * badly-sized encode is an `rmm::out_of_memory` from an unrelated operator, which
 * says nothing about how many allocations preceded it or how big they were.
 *
 * Enabled by setting `SIRIUS_COMPRESSION_ALLOC_STATS=1` in the environment: it
 * inserts a counting adaptor between the encode and its memory resource (the
 * arena when one is installed, else the query pool), so it is off by default and
 * costs nothing when off. Counting itself is a handful of relaxed atomics per
 * allocation, small against a codec launch but not free.
 */

/// A point-in-time reading of the counters. Sizes are device bytes.
struct alloc_stats_snapshot {
  /// Number of allocations and deallocations served since the process started.
  std::uint64_t allocations   = 0;
  std::uint64_t deallocations = 0;
  /// Sum of all allocation sizes ever served — the encode's total allocator traffic.
  std::uint64_t total_bytes = 0;
  /// Bytes allocated and not yet freed, and the high-water mark of that value.
  std::uint64_t outstanding_bytes = 0;
  std::uint64_t peak_bytes        = 0;
  /// Largest single allocation served.
  std::uint64_t largest_bytes = 0;
  /// Allocation counts bucketed by size: <64 KiB, <1 MiB, <8 MiB, <64 MiB,
  /// <256 MiB, >=256 MiB. The shape matters more than the total: a codec that
  /// asks for many small scratch buffers and one that asks for one buffer the
  /// size of the batch fail for different reasons and are fixed differently.
  std::uint64_t histogram[6] = {0, 0, 0, 0, 0, 0};
};

/// True when `SIRIUS_COMPRESSION_ALLOC_STATS=1` was set at process start.
[[nodiscard]] bool alloc_stats_enabled() noexcept;

/// Wrap @p upstream in the counting adaptor, or return it unchanged when the
/// accounting is disabled. The adaptor is a process-wide singleton, so every
/// caller shares one set of counters.
[[nodiscard]] rmm::device_async_resource_ref alloc_stats_wrap(
  rmm::device_async_resource_ref upstream);

/// Current counter values; all zero when the accounting is disabled.
[[nodiscard]] alloc_stats_snapshot alloc_stats_read() noexcept;

/// One log line's worth of the counters, formatted for the debug log.
[[nodiscard]] std::string alloc_stats_format();

}  // namespace sirius::compression
