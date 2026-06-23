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

#include "io/io_context.hpp"

#include "io/cache/prefetching_cache.hpp"
#include "io/sirius_datasource.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

namespace sirius::io {

sirius_ioctx::sirius_ioctx()  = default;
sirius_ioctx::~sirius_ioctx() = default;

void sirius_ioctx::initialize_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  size_t inflight_budget_chunks,
  uint32_t buffer_pool_slabs,
  std::shared_ptr<const sirius::memory::topology_index> topology_index) noexcept
{
  // One-shot.  Repeated calls are silent no-ops so callers can be
  // robust to multiple wiring sites.
  if (_cache) return;
  try {
    _cache = std::make_unique<cache::prefetching_cache>(reservation_manager,
                                                        this,
                                                        inflight_budget_chunks,
                                                        buffer_pool_slabs,
                                                        std::move(topology_index));
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("prefetching_cache construction failed: {}", e.what());
    _cache.reset();
  } catch (...) {
    SIRIUS_LOG_ERROR("prefetching_cache construction failed: unknown error");
    _cache.reset();
  }
}

void sirius_ioctx::shutdown_cache() noexcept { _cache.reset(); }

std::unique_ptr<sirius_datasource> sirius_ioctx::open_datasource(std::string path)
{
  // Create the backend-appropriate io_object (local fds / object-store HEAD /
  // ...) and wrap it in a sirius_datasource bound to this ioctx.  Datasource
  // construction is uniform across backends, so it lives here rather than in a
  // per-backend hook.
  return std::make_unique<sirius_datasource>(shared_from_this(), create_io_object(std::move(path)));
}

}  // namespace sirius::io
