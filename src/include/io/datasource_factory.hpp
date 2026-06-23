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

#include <absl/functional/any_invocable.h>

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>

namespace sirius {
struct sirius_config;
}

namespace sirius::io {

// ---------------------------------------------------------------------------
// datasource_registry
// ---------------------------------------------------------------------------

/**
 * @brief Thread-safe registry mapping URI schemes to @c sirius_ioctx instances.
 *
 * The engine constructs a registry at startup and populates it with one
 * @c sirius_ioctx per backend (uring / gds / s3 / rdma_s3). The factory looks
 * up the correct backend by URI scheme at datasource-creation time.
 *
 * Scheme matching is case-insensitive: @c register_ioctx and @c lookup both
 * lowercase the scheme before storing / searching, matching the
 * normalization done by @c sirius::io::parse (RFC 3986 §3.1). Callers may
 * register / look up with any casing — @c register_ioctx("S3", ...) and
 * @c lookup("s3") refer to the same entry.
 *
 * All operations are safe under concurrent reads; mutations take an exclusive
 * lock but are expected only at engine bootstrap / shutdown.
 */
class io_context_registry {
 public:
  using config_type = scan_manager::scan_manager_config;

  explicit io_context_registry(config_type config);
  ~io_context_registry() = default;

  io_context_registry(io_context_registry const&)            = delete;
  io_context_registry& operator=(io_context_registry const&) = delete;

  using scheme_checker_type = std::function<bool(std::string_view)>;
  using factory_type        = std::function<std::shared_ptr<io::sirius_ioctx>(const config_type&)>;

  /**
   * @brief Register an ioctx for a scheme. Replaces any prior registration
   *        for the same scheme.
   *
   * The scheme is lowercased before storage; subsequent @c lookup calls
   * with any casing of the same scheme resolve to this entry.
   * @param type    Opaque identifier for the ioctx type. Used by the engine to
   *                identify the backend.
   */
  void register_ioctx(io_context_type type, scheme_checker_type checker, factory_type factory);

  std::optional<io_context_type> lookup(std::string_view scheme) const noexcept;

  std::shared_ptr<sirius_ioctx> make_ioctx(io_context_type type) const noexcept;

  /**
   * @brief Drop all registered ioctxs. Callers are responsible for shutting
   *        them down before clearing.
   */
  void clear();

 private:
  struct entry {
    scheme_checker_type checker;
    factory_type factory;
    io_context_type type;
  };
  const config_type _config;
  mutable std::shared_mutex _mtx;
  std::unordered_map<io_context_type, entry> _entries;
  bool prefer_kvikio_for_file_scheme{false};  // single-GPU opt-out of sirius_datasource for file://
};

}  // namespace sirius::io
