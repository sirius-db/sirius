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

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace sirius::io {

/**
 * @file bulk_read.hpp
 * @brief Sirius's choice of the bulk (vectored) Parquet read route per io context.
 *
 * The bulk route of @c op::scan::materialize_parquet reads every column chunk of a split's
 * file in ONE vectored device request.  cuCascade lets a backend ask for it with the
 * compile-time @c ioctx::prefers_bulk_io() (REST does: every request is a round trip).  The
 * cuFile backend says no, yet with fused lanes on it wants the whole split in one request too,
 * because it only fuses the slices of one request.  Sirius therefore asks for the route on a
 * context it owns by registering it here (see @c scan_manager.cufile_bulk_read); every reader
 * of the preference goes through @ref prefers_bulk_read, so the route and the decisions keyed
 * off it agree.
 */

namespace detail {

struct bulk_read_contexts {
  std::mutex mutex;
  std::vector<cucascade::io::ioctx const*> contexts;
  /// Fast path: nothing is registered (every backend but a configured cuFile one).
  std::atomic<std::size_t> count{0};
};

inline bulk_read_contexts& bulk_read_registry() noexcept
{
  static bulk_read_contexts registry;
  return registry;
}

}  // namespace detail

/// Whether Sirius asked for the bulk route on @p ctx (see @ref bulk_read_registration).
[[nodiscard]] inline bool bulk_read_registered(cucascade::io::ioctx const& ctx) noexcept
{
  auto& registry = detail::bulk_read_registry();
  if (registry.count.load(std::memory_order_acquire) == 0) { return false; }
  std::lock_guard lock(registry.mutex);
  return std::ranges::find(registry.contexts, &ctx) != registry.contexts.end();
}

/// Whether Parquet column chunks served by @p ctx are read on the bulk route: the backend's
/// own preference (REST), or a Sirius registration (a cuFile context with
/// @c cufile_bulk_read on).
[[nodiscard]] inline bool prefers_bulk_read(cucascade::io::ioctx const& ctx) noexcept
{
  return ctx.prefers_bulk_io() || bulk_read_registered(ctx);
}

/// @ref prefers_bulk_read for the context serving @p source.
[[nodiscard]] inline bool prefers_bulk_read(cucascade::io::datasource const& source) noexcept
{
  if (source.prefers_bulk_io()) { return true; }
  auto const ctx = source.io_ctx();
  return ctx && bulk_read_registered(*ctx);
}

/**
 * @brief Asks for the bulk route on one io context for as long as it lives.
 *
 * The owner of the context (the scan manager, for its default cuFile context) holds one and
 * must destroy it before it releases the context, so the address is never confused with a
 * later context's.  Datasources that outlive the registration fall back to the backend's own
 * preference.  Move-only.
 */
class bulk_read_registration {
 public:
  bulk_read_registration() noexcept = default;

  explicit bulk_read_registration(cucascade::io::ioctx const& ctx) : _ctx(&ctx)
  {
    auto& registry = detail::bulk_read_registry();
    std::lock_guard lock(registry.mutex);
    registry.contexts.push_back(_ctx);
    registry.count.fetch_add(1, std::memory_order_release);
  }

  bulk_read_registration(bulk_read_registration&& other) noexcept
    : _ctx(std::exchange(other._ctx, nullptr))
  {
  }

  bulk_read_registration& operator=(bulk_read_registration&& other) noexcept
  {
    if (this != &other) {
      reset();
      _ctx = std::exchange(other._ctx, nullptr);
    }
    return *this;
  }

  bulk_read_registration(bulk_read_registration const&)            = delete;
  bulk_read_registration& operator=(bulk_read_registration const&) = delete;

  ~bulk_read_registration() { reset(); }

  /// Withdraw the registration (no-op when empty).
  void reset() noexcept
  {
    if (_ctx == nullptr) { return; }
    auto& registry = detail::bulk_read_registry();
    std::lock_guard lock(registry.mutex);
    if (auto const it = std::ranges::find(registry.contexts, _ctx); it != registry.contexts.end()) {
      registry.contexts.erase(it);
      registry.count.fetch_sub(1, std::memory_order_release);
    }
    _ctx = nullptr;
  }

  [[nodiscard]] bool active() const noexcept { return _ctx != nullptr; }

 private:
  cucascade::io::ioctx const* _ctx{nullptr};
};

}  // namespace sirius::io
