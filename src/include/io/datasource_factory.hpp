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
 * Scheme matching is case-insensitive: @c register_ioctx and @c lookup both
 * lowercase the scheme before storing / searching, matching the
 * normalization done by @c sirius::io::parse (RFC 3986 §3.1). Callers may
 * register / look up with any casing — @c register_ioctx("S3", ...) and
 * @c lookup("s3") refer to the same entry.
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
   *
   * The scheme is lowercased before storage; subsequent @c lookup calls
   * with any casing of the same scheme resolve to this entry.
   */
  void register_ioctx(std::string scheme, std::shared_ptr<sirius_ioctx> ioctx);

  /**
   * @brief Look up the ioctx registered for @p scheme.
   *
   * The scheme is lowercased before lookup; @c lookup("S3") and
   * @c lookup("s3") resolve to the same entry.
   *
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
 * Dispatch is by URI scheme: the factory extracts the scheme and looks up the
 * registered @c sirius_ioctx. PR1 intentionally ships only the backend-neutral
 * skeleton; backend-specific object construction (for example S3 bucket/key
 * handles) lands with the corresponding backend PR.
 *
 * URI forms supported (parsing is delegated to @c sirius::io::parse):
 *   - <tt>/absolute/path</tt>      — treated as scheme @c "file"
 *   - <tt>file:///abs/path</tt>    — scheme @c "file"
 *   - <tt>s3://bucket/key</tt>,
 *     <tt>gs://bucket/key</tt>,
 *     <tt>azure://container/blob</tt> — object-store schemes (host/key split)
 *   - Relative bare paths are rejected; use absolute or a scheme.
 *
 * The Windows-style drive-letter form <tt>C:/...</tt> is not supported (Sirius
 * builds on Linux only; see CMakeLists.txt requirements).
 */
class datasource_factory {
 public:
  /**
   * @brief Create a @c cudf::io::datasource for @p uri.
   *
   * Dispatch:
   *   - Local file paths (@c "/data/foo.parquet", @c "file:///data/foo.parquet")
   *     return cudf's default datasource (@c cudf::io::datasource::create) —
   *     pre-PR3 baseline. The registry is not consulted for file scheme.
   *   - Object-store schemes (@c s3://, etc.) first go through the registry
   *     lookup. In PR1, a registered object-store scheme still throws
   *     "object construction is not yet implemented"; the backend PR that owns
   *     the concrete @c sirius_io_object wires the final construction step.
   *
   * The wider return type is deliberate: it lets the file branch keep using
   * cudf's proven default reader without forcing a sirius-specific adapter,
   * which sidesteps regression risk on the local-parquet hot path until a
   * GDS-aware backend (PR6) makes the switch worthwhile.
   *
   * @param uri      The resource URI (e.g. @c "/data/file.parquet",
   *                 @c "file:///data/file.parquet", @c "s3://bucket/key").
   * @param registry Registry to look up the backend ioctx (object-store schemes only).
   * @param config   Engine config (read by object-store backends in later PRs).
   *
   * @return A new datasource on success.
   *
   * @throw std::invalid_argument if the URI is empty or malformed.
   * @throw std::runtime_error    if an object-store scheme has no backend
   *                              registered, or if backend-specific object
   *                              construction has not landed yet.
   */
  static std::unique_ptr<cudf::io::datasource> create(std::string_view uri,
                                                      datasource_registry const& registry,
                                                      sirius_config const& config);

  /**
   * @brief Lenient entry point for parquet-scan IO whose @p uri originates in
   *        DuckDB's @c MultiFileBindData / @c scan_op.parameters and may be a
   *        bare relative path rather than a normalized URI.
   *
   * Dispatch:
   *   - Relative bare path (no leading @c '/' and no @c "://"): bypass the
   *     factory and return cudf's default datasource directly. This matches
   *     the pre-PR3 baseline; it covers iceberg / hive test fixtures that
   *     hand out paths like @c "test/cpp/integration/data/...parquet".
   *   - Anything else (absolute path, @c file:///..., @c s3://..., future
   *     object-store schemes): delegate to the strict @c create above. The
   *     parser accepts these and routes file scheme to cudf default, object
   *     stores to the registered @c sirius_ioctx.
   *
   * The strict @c create keeps its parser-strict contract — prefer it for
   * callers that should reject unscheme'd input as a real bug.
   *
   * @throw std::invalid_argument if the URI is empty (preserves strict
   *                              @c create's empty-URI rejection).
   * @throw std::runtime_error    on object-store dispatch failure (same
   *                              as strict @c create).
   */
  static std::unique_ptr<cudf::io::datasource> create_for_parquet_scan(
    std::string_view uri, datasource_registry const& registry, sirius_config const& config);

  /**
   * @brief Extract the URI scheme. Thin shim over @c sirius::io::parse.
   *        Prefer calling @c parse directly for new code; retained for
   *        compatibility with PR1 callsites and tests.
   *
   * Throws @c std::invalid_argument on the same inputs as @c parse (empty URI,
   * empty scheme, relative bare path, malformed URI).
   */
  [[nodiscard]] static std::string extract_scheme(std::string_view uri);

  /**
   * @brief Extract the path portion of @p uri. Thin shim over
   *        @c sirius::io::parse; returns the parser's @c path field
   *        (percent-decoded, no host, exactly one bucket/key separator
   *        slash stripped for object-store schemes; leading slash retained
   *        for @c file).
   *
   * Examples:
   *   - @c "/data/f.parquet"          -> @c "/data/f.parquet"
   *   - @c "file:///data/f.parquet"   -> @c "/data/f.parquet"
   *   - @c "s3://bucket/key"          -> @c "key"
   *   - @c "s3://bucket//key"         -> @c "/key"
   *   - @c "s3://bucket///key"        -> @c "//key"
   *
   * Throws on relative paths, empty keys, malformed URIs.
   */
  [[nodiscard]] static std::string extract_path(std::string_view uri);
};

}  // namespace sirius::io
