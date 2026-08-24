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

namespace sirius::compression {

/**
 * @brief A device arena reserved for spill-compression working memory.
 *
 * The spill encoder allocates from the same RMM pool as the query, and it runs
 * precisely when that pool is exhausted — a downgrade only happens because the
 * GPU is full. Sharing is therefore circular: the allocation that would relieve
 * the pressure is the one certain to fail, and an `rmm::out_of_memory` there
 * latches `spill_compression_suppressed` for the rest of the episode.
 *
 * Measured on q3/SF1000 with no arena: compression latched off and back on 11
 * times in one query while the monitor issued 111,641 downgrade requests, and
 * the query had to be killed. With a 4 GiB arena the same query ran in 44.3 s,
 * level with an untouched baseline.
 *
 * The arena is a fixed allocation taken once at startup and never grown, so it
 * cannot compete with the query later. It is a *partition* of the device, not
 * extra memory: reserving N bytes requires lowering
 * `memory.gpu.usage_limit_fraction` by the same N, and getting that wrong is not
 * a gradient but a cliff — at 1 GiB (too small for the concurrent encodes) the
 * same query failed outright and fell back to DuckDB.
 */

/// Allocate the arena. @p bytes == 0 disables it and the compress path falls back
/// to the current device resource. Idempotent; returns false when the arena could
/// not be reserved, leaving the fallback in effect.
bool init_compression_device_pool(std::size_t bytes);

/// The resource spill compression allocates its transients from: the arena when
/// one is installed, else the current device resource (the query's pool).
rmm::device_async_resource_ref compression_device_mr();

/// True when an arena is installed.
bool compression_device_pool_enabled() noexcept;

/// Configured arena size in bytes; 0 when disabled.
std::size_t compression_device_pool_bytes() noexcept;

}  // namespace sirius::compression
