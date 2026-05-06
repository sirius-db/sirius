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

constexpr std::string_view kFileScheme = "file";

}  // namespace

// Retained for compatibility with PR1 callsites/tests; prefer sirius::io::parse()
// for new code. Both helpers route through the real URI parser.
std::string datasource_factory::extract_scheme(std::string_view uri) { return parse(uri).scheme; }

std::string datasource_factory::extract_path(std::string_view uri) { return parse(uri).path; }

std::unique_ptr<cudf::io::datasource> datasource_factory::create_for_parquet_scan(
  std::string_view uri, datasource_registry const& registry, sirius_config const& config)
{
  // Relative bare paths (no leading '/' and no scheme://) — DuckDB's iceberg /
  // hive fixtures still hand these out, and the strict parser deliberately
  // rejects them. Bypass to cudf default; semantically identical to the pre-PR3
  // baseline. Anything else (absolute path, file:///..., s3://...) goes through
  // the strict create() so its parser routes file→cudf and object-store→ioctx
  // uniformly.
  if (!uri.empty() && uri.front() != '/' && uri.find("://") == std::string_view::npos) {
    return cudf::io::datasource::create(std::string{uri});
  }
  return create(uri, registry, config);
}

std::unique_ptr<cudf::io::datasource> datasource_factory::create(
  std::string_view uri, datasource_registry const& registry, sirius_config const& /*config*/)
{
  auto p = parse(uri);

  // Local file paths stay on cudf's default pread-based datasource. This
  // matches the pre-PR3 baseline and sidesteps the new IO framework for the
  // hot path that 99% of queries hit. A future PR (e.g. gds_ioctx) can opt
  // local NVMe paths into a sirius-managed backend via a SET knob without
  // touching the call sites.
  if (p.scheme == kFileScheme) { return cudf::io::datasource::create(std::move(p.path)); }

  // Object-store schemes go through the registry → ioctx → sirius_datasource.
  auto ioctx = registry.lookup(p.scheme);
  if (!ioctx) {
    throw std::runtime_error("datasource_factory: no backend registered for scheme '" + p.scheme +
                             "' (uri=" + std::string{uri} + ")");
  }

  // Object-store io_object construction is backend-specific and lives in each
  // backend's PR (s3 lands in PR3). PR1 ships only the dispatch skeleton — no
  // scheme-aware branches in the factory.
  (void)ioctx;
  throw std::runtime_error("datasource_factory: scheme '" + p.scheme +
                           "' is registered but object construction is not yet implemented");
}

}  // namespace sirius::io
