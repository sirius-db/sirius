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

#include "io/types.hpp"

#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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
 * All operations are safe under concurrent reads; mutations take an exclusive
 * lock but are expected only at engine bootstrap / shutdown.
 */
class datasource_registry {
 public:
  datasource_registry()                                      = default;
  ~datasource_registry()                                     = default;
  datasource_registry(datasource_registry const&)            = delete;
  datasource_registry& operator=(datasource_registry const&) = delete;

  /**
   * @brief Register an ioctx for a scheme. Replaces any prior registration
   *        for the same scheme.
   */
  void register_ioctx(std::string scheme, std::shared_ptr<sirius_ioctx> ioctx);

  /**
   * @brief Look up the ioctx registered for @p scheme.
   * @return The ioctx, or @c nullptr if no backend is registered for @p scheme.
   */
  [[nodiscard]] std::shared_ptr<sirius_ioctx> lookup(std::string_view scheme) const;

  /**
   * @brief Return all registered schemes (copy, for testing / diagnostics).
   */
  [[nodiscard]] std::vector<std::string> schemes() const;

  /**
   * @brief Drop all registered ioctxs. Callers are responsible for shutting
   *        them down before clearing.
   */
  void clear();

 private:
  mutable std::shared_mutex _mtx;
  std::unordered_map<std::string, std::shared_ptr<sirius_ioctx>> _ioctxs;
};

// ---------------------------------------------------------------------------
// datasource_factory
// ---------------------------------------------------------------------------

/**
 * @brief Factory that constructs an @c io_datasource from a URI.
 *
 * Dispatch is by URI scheme: the factory extracts the scheme, looks up the
 * registered @c sirius_ioctx, and asks it to construct a datasource for the
 * corresponding @c sirius_io_object.
 *
 * URI forms supported in PR1 (full URI parsing lands in PR8):
 *   - <tt>/absolute/path</tt>   — treated as scheme @c "file"
 *   - <tt>file:///abs/path</tt> — scheme @c "file"
 *   - <tt>\<scheme\>://...</tt> — any other scheme dispatches to registry[scheme]
 *
 * The Windows-style drive-letter form <tt>C:/...</tt> is not supported (Sirius
 * builds on Linux only; see CMakeLists.txt requirements).
 */
class datasource_factory {
 public:
  /**
   * @brief Create an @c io_datasource for @p uri.
   *
   * @param uri      The resource URI (e.g. @c "/data/file.parquet",
   *                 @c "file:///data/file.parquet", @c "s3://bucket/key").
   * @param registry Registry to look up the backend ioctx.
   * @param config   Engine config (read by object-store backends in later PRs).
   *
   * @return A new datasource on success.
   *
   * @throw std::invalid_argument if the URI is empty or malformed.
   * @throw std::runtime_error    if no backend is registered for the scheme,
   *                              or if the backend's @c make_datasource does
   *                              not produce an @c io_datasource.
   */
  static std::unique_ptr<io_datasource> create(std::string_view uri,
                                               datasource_registry const& registry,
                                               sirius_config const& config);

  /**
   * @brief Extract the URI scheme. Exposed for testing.
   *
   * Rules:
   *   - An empty input throws.
   *   - A path beginning with @c '/' has scheme @c "file".
   *   - Otherwise the scheme is the substring preceding @c "://".
   *   - If no @c "://" delimiter is present, returns @c "file" (treat input as
   *     a relative or bare path; downstream ioctx resolves it).
   */
  [[nodiscard]] static std::string extract_scheme(std::string_view uri);

  /**
   * @brief Extract the path portion of @p uri (everything after the scheme
   *        delimiter, or the whole URI when no scheme delimiter exists).
   *
   * Examples:
   *   - @c "/data/f.parquet"          -> @c "/data/f.parquet"
   *   - @c "file:///data/f.parquet"   -> @c "/data/f.parquet"
   *   - @c "s3://bucket/key"          -> @c "bucket/key"
   *   - @c "relative/path"            -> @c "relative/path"
   */
  [[nodiscard]] static std::string extract_path(std::string_view uri);
};

}  // namespace sirius::io
