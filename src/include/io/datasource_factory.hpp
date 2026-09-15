/*
 * Copyright 2025, Sirius Contributors.
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

#include "io/io_context.hpp"
#include "sirius_config.hpp"

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>

namespace sirius {
struct sirius_config;
}

namespace cucascade::memory {
class memory_reservation_manager;
}

namespace sirius::io {

// ---------------------------------------------------------------------------
// datasource_registry
// ---------------------------------------------------------------------------

/**
 * @brief Thread-safe registry of @c ioctx backends, resolved by full path.
 *
 * The engine constructs a registry at startup and registers one entry per backend
 * (kvikio / uring / restful), each carrying a path-capability checker.  At
 * datasource-creation time @c lookup_path runs the checkers against a full path
 * (the checkers parse the URI / stat the filesystem themselves) and picks the
 * backend, preferring an explicit backend over the kvikio catch-all.
 *
 * All operations are safe under concurrent reads; mutations take an exclusive
 * lock but are expected only at engine bootstrap / shutdown.
 */
class io_context_registry {
 public:
  using config_type = scan_manager::scan_manager_config;

  /// @param config              Scan-manager configuration consumed by the
  ///                            per-backend factories at construction time.
  /// @param reservation_manager Source of the tier-specific memory resources the
  ///                            backends need (e.g. the HOST-tier pinned staging
  ///                            resource for the uring / rest reactors).  Must
  ///                            outlive the registry — the factory closures hold
  ///                            it by reference.
  io_context_registry(config_type config,
                      cucascade::memory::memory_reservation_manager& reservation_manager);
  ~io_context_registry() = default;

  io_context_registry(io_context_registry const&)            = delete;
  io_context_registry& operator=(io_context_registry const&) = delete;

  using scheme_checker_type = std::function<bool(std::string_view)>;
  using factory_type        = std::function<std::shared_ptr<io::ioctx>(const config_type&)>;

  /**
   * @brief Register an ioctx backend. Replaces any prior registration for the
   *        same type.
   *
   * @param type    Backend identifier (uring / restful / kvikio).
   * @param checker Decides whether this backend claims a given path.
   * @param factory Constructs the backend's ioctx; invoked by @c make_ioctx.
   */
  void register_ioctx(io_context_type type, scheme_checker_type checker, factory_type factory);

  /// Resolve the backend for a full @p path (not a bare scheme — the checkers
  /// parse the URI / stat the filesystem themselves).  Explicit backends
  /// (uring / restful) take precedence over the kvikio catch-all, so `s3://`
  /// never resolves to kvikio and a local file routes to uring before the
  /// universal fallback.  When the registry was built with
  /// `backend: kvikio`, the uring local backend is suppressed so local
  /// files fall through to kvikio.  std::nullopt when nothing matches.
  std::optional<io_context_type> lookup_path(std::string_view path) const noexcept;

  std::shared_ptr<ioctx> make_ioctx(io_context_type type) const noexcept;

  /**
   * @brief Drop all registered ioctxs. Callers are responsible for shutting
   *        them down before clearing.
   */
  void clear();

 private:
  struct entry {
    io_context_type type;
    scheme_checker_type checker;
    factory_type factory;
  };
  const config_type _config;
  cucascade::memory::memory_reservation_manager& _reservation_manager;
  /// Set when @c backend=kvikio: kvikIO then serves BOTH local files (instead
  /// of uring) and @c s3:// objects (instead of rest) for reads.  LIST / glob
  /// still goes to the REST backend, which the scan manager obtains by type.
  bool _prefer_kvikio{false};
  mutable std::shared_mutex _mtx;
  std::unordered_map<io_context_type, entry> _entries;
};

// ---------------------------------------------------------------------------
// Per-backend factory builders
// ---------------------------------------------------------------------------
//
// Each returns a @c factory_type closure that builds one backend ioctx from a
// @c scan_manager_config.  The closure captures @p reservation_manager by
// reference (it sources the HOST-tier staging resource the reactors need), so
// @p reservation_manager must outlive every ioctx the returned factory creates.
// The closures are @c noexcept-safe: a construction failure (missing resource,
// unconfigured credentials, …) is logged and reported as a null ioctx rather
// than thrown, matching @c io_context_registry::make_ioctx.

/// kvikio fallback backend (drives @c kvikio::FileHandle directly).  Takes no
/// reservation manager — kvikio owns no reactor staging.
io_context_registry::factory_type make_kvikio_ioctx_factory();

/// io_uring local-disk backend.  Builds a @c uring_reactor::reactor_context from
/// @c config.uring and @c config.uring_n_reactors. Pinned staging uses the
/// HOST-tier resource block size; physical grouping is chosen by the worker.
io_context_registry::factory_type make_uring_ioctx_factory(
  cucascade::memory::memory_reservation_manager& reservation_manager);

/// RESTful object-store (s3://) backend.  Builds a SigV4 authorizer from
/// @c config.object_store and a @c rest_reactor::reactor_context from
/// @c config.rest.  Returns a null ioctx when the object store is not
/// configured (empty endpoint / credentials / region).
io_context_registry::factory_type make_rest_ioctx_factory(
  cucascade::memory::memory_reservation_manager& reservation_manager);

}  // namespace sirius::io
