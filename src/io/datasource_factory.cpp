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

#include "io/datasource_factory.hpp"

#include "io/uri_parser.hpp"
#include "sirius_config.hpp"

#include <cudf/io/datasource.hpp>

#include <cctype>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::io {

namespace {

// RFC 3986 §3.1: schemes are case-insensitive. uri_parser lowercases the
// parsed scheme on the read side; registry must normalize the same way on
// the write side so callers don't need to remember which side does it. Used
// by both register_ioctx and lookup so a register("S3", ...) is found by a
// lookup("s3"), and vice versa.
std::string to_lower_scheme(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// datasource_registry
// ---------------------------------------------------------------------------

void datasource_registry::register_ioctx(std::string scheme, std::shared_ptr<sirius_ioctx> ioctx)
{
  if (scheme.empty()) throw std::invalid_argument("datasource_registry: empty scheme");
  if (!ioctx) throw std::invalid_argument("datasource_registry: null ioctx for '" + scheme + "'");
  scheme = to_lower_scheme(scheme);
  std::unique_lock lk{_mtx};
  _ioctxs[std::move(scheme)] = std::move(ioctx);
}

std::shared_ptr<sirius_ioctx> datasource_registry::lookup(std::string_view scheme) const
{
  std::shared_lock lk{_mtx};
  auto it = _ioctxs.find(to_lower_scheme(scheme));
  return it == _ioctxs.end() ? nullptr : it->second;
}

std::vector<std::string> datasource_registry::schemes() const
{
  std::shared_lock lk{_mtx};
  std::vector<std::string> out;
  out.reserve(_ioctxs.size());
  for (auto const& [scheme, _] : _ioctxs)
    out.push_back(scheme);
  return out;
}

void datasource_registry::clear()
{
  std::unique_lock lk{_mtx};
  _ioctxs.clear();
}

// ---------------------------------------------------------------------------
// datasource_factory
// ---------------------------------------------------------------------------

namespace {

// Documentation constant: the canonical "file" scheme name registered by
// SiriusContext::initialize(). The factory looks up schemes via
// registry.lookup(p.scheme) without branching on this name, so the constant
// is intentionally unused at runtime — kept as a single source of truth that
// the registration site (sirius_context.cpp) and any future scheme-specific
// factory code can refer to.
[[maybe_unused]] constexpr std::string_view kFileScheme = "file";

}  // namespace

// Retained for compatibility with PR1 callsites/tests; prefer sirius::io::parse()
// for new code. Both helpers route through the real URI parser.
std::string datasource_factory::extract_scheme(std::string_view uri) { return parse(uri).scheme; }

std::string datasource_factory::extract_path(std::string_view uri) { return parse(uri).path; }

std::unique_ptr<cudf::io::datasource> datasource_factory::create_for_parquet_scan(
  std::string_view uri, datasource_registry const& registry, sirius_config const& config)
{
  // Relative bare paths normalize to file:///<absolute> and dispatch through
  // create(). We do not emit a cudf-default fallback here: callers that want
  // the bundled kvikio path use cudf::io::datasource::create directly outside
  // this factory (single-GPU use_sirius_datasource=false). In multi-GPU mode
  // kvikio is forbidden — sirius_config::enforce_sirius_datasource_for_multi_gpu()
  // forces use_sirius_datasource=true whenever >1 GPU is configured.
  //
  // DuckDB's iceberg / hive fixtures hand out paths like
  // "test/cpp/integration/data/...parquet"; we resolve those against the
  // process CWD via std::filesystem::absolute (matching cudf-default
  // semantics for relative paths) and prepend "file://" so create()'s parser
  // routes them through the registered kFileScheme ioctx.
  if (!uri.empty() && uri.front() != '/' && uri.find("://") == std::string_view::npos) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto abs = fs::absolute(fs::path{std::string{uri}}, ec);
    if (ec) {
      throw std::runtime_error(
        "datasource_factory::create_for_parquet_scan: failed to resolve relative path '" +
        std::string{uri} + "' to absolute: " + ec.message());
    }
    std::string normalized = "file://" + abs.string();
    return create(normalized, registry, config);
  }
  return create(uri, registry, config);
}

std::unique_ptr<cudf::io::datasource> datasource_factory::create(
  std::string_view uri, datasource_registry const& registry, sirius_config const& /*config*/)
{
  auto p = parse(uri);

  // ALL schemes (including kFileScheme) MUST be resolved via the registry —
  // this factory does not silently fall back to the cudf default datasource
  // (which routes through libkvikio and binds a single CUDA context per
  // FileHandle, breaking multi-GPU residency). SiriusContext::initialize()
  // registers kFileScheme -> sirius_ioctx whenever use_sirius_datasource is
  // true; in a single-GPU configuration with use_sirius_datasource=false the
  // file scheme is intentionally not registered and callers route through
  // cudf::io::datasource::create directly instead of invoking this factory.
  // Multi-GPU mode always uses sirius_datasource (enforced by
  // sirius_config::enforce_sirius_datasource_for_multi_gpu()).
  auto ioctx = registry.lookup(p.scheme);
  if (!ioctx) {
    throw std::runtime_error("datasource_factory: no ioctx registered for scheme '" + p.scheme +
                             "' (uri=" + std::string{uri} + ")");
  }

  return ioctx->open_datasource(std::move(p.path));
}

}  // namespace sirius::io
