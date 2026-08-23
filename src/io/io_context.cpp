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

#include "io/cache/config.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::io {

ioctx::ioctx()  = default;
ioctx::~ioctx() = default;

void ioctx::initialize_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  io::cache::config const& cache_config,
  std::shared_ptr<const sirius::memory::topology_index> topology_index) noexcept
{
  // One-shot.  Repeated calls are silent no-ops so callers can be
  // robust to multiple wiring sites.
  if (_cache) {
    SIRIUS_LOG_WARN("ioctx::initialize_cache() called but prefetching_cache already present");
    return;
  }
  if (!can_use_prefetching_cache()) {
    SIRIUS_LOG_WARN(
      "ioctx::initialize_cache() called but backend does not support vector host read");
    return;
  }
  try {
    _cache = std::make_unique<cache::prefetching_cache>(
      reservation_manager, this, cache_config, std::move(topology_index));
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("prefetching_cache construction failed: {}", e.what());
    _cache.reset();
  } catch (...) {
    SIRIUS_LOG_ERROR("prefetching_cache construction failed: unknown error");
    _cache.reset();
  }
}

void ioctx::shutdown_cache() noexcept { _cache.reset(); }

std::unique_ptr<sirius_datasource> ioctx::open_datasource(std::string path)
{
  // Create the backend-appropriate io_object (local fds / object-store HEAD /
  // ...) and wrap it in a sirius_datasource bound to this ioctx.  Datasource
  // construction is uniform across backends, so it lives here rather than in a
  // per-backend hook.
  return std::make_unique<sirius_datasource>(shared_from_this(), create_io_object(std::move(path)));
}

std::unique_ptr<sirius_datasource> ioctx::open_datasource(std::string path, open_hint hint)
{
  return std::make_unique<sirius_datasource>(shared_from_this(),
                                             create_io_object(std::move(path), hint));
}

std::unique_ptr<sirius_datasource> ioctx::open_datasource(std::string path,
                                                          std::uint64_t known_size)
{
  return std::make_unique<sirius_datasource>(shared_from_this(),
                                             create_io_object(std::move(path), known_size));
}

std::shared_ptr<io_object> ioctx::create_io_object(std::string path, open_hint /*hint*/)
{
  return create_io_object(std::move(path));
}

std::shared_ptr<io_object> ioctx::create_io_object(std::string path, std::uint64_t /*known_size*/)
{
  return create_io_object(std::move(path));
}

exec::semi_future<size_t> ioctx::host_read_async_io(const io_object& obj,
                                                    size_t offset,
                                                    size_t size,
                                                    uint8_t* dst) noexcept
{
  if (size == 0) return exec::make_semi_future<size_t>(0);
  try {
    if (dst == nullptr) throw std::invalid_argument("host read destination is null");
    std::vector<prepared_io_slice> slices{prepared_io_slice{range{offset, size}, host_buffer{dst}}};
    return host_device_readv_async_io(obj, std::move(slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::device_read_async_io(const io_object& obj,
                                                      size_t offset,
                                                      size_t size,
                                                      uint8_t* dst,
                                                      rmm::cuda_stream_view stream) noexcept
{
  if (size == 0) return exec::make_semi_future<size_t>(0);
  try {
    if (dst == nullptr) throw std::invalid_argument("device read destination is null");
    std::vector<prepared_io_slice> slices{
      prepared_io_slice{range{offset, size}, device_buffer{dst, stream}}};
    return host_device_readv_async_io(obj, std::move(slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::host_readv_async_io(const io_object& obj,
                                                     std::span<const slice> slices) noexcept
{
  if (slices.empty()) return exec::make_semi_future<size_t>(0);
  try {
    std::vector<prepared_io_slice> prepared_slices;
    prepared_slices.reserve(slices.size());
    for (auto const& current : slices) {
      if (current.size() == 0) continue;
      if (current.dst == nullptr) throw std::invalid_argument("host readv destination is null");
      prepared_slices.emplace_back(range{current.offset(), current.size()},
                                   host_buffer{current.dst});
    }
    return host_device_readv_async_io(obj, std::move(prepared_slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::device_readv_async_io(const io_object& obj,
                                                       std::span<const slice> slices,
                                                       rmm::cuda_stream_view stream) noexcept
{
  if (slices.empty()) return exec::make_semi_future<size_t>(0);
  try {
    std::vector<prepared_io_slice> prepared_slices;
    prepared_slices.reserve(slices.size());
    for (auto const& current : slices) {
      if (current.size() == 0) continue;
      if (current.dst == nullptr) throw std::invalid_argument("device readv destination is null");
      prepared_slices.emplace_back(range{current.offset(), current.size()},
                                   device_buffer{current.dst, stream});
    }
    return host_device_readv_async_io(obj, std::move(prepared_slices));
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

}  // namespace sirius::io
