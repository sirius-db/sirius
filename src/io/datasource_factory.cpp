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
#include "io/uring/uring_reactor.hpp"
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
// SiriusContext::initialize() (Plan 22.1-01). Post-22.1 the factory looks
// up schemes via registry.lookup(p.scheme) without branching on this name,
// so the constant is intentionally unused at runtime — kept as a single
// source of truth that the registration site (sirius_context.cpp) and any
// future scheme-specific factory code can refer to.
[[maybe_unused]] constexpr std::string_view kFileScheme = "file";

}  // namespace

// Retained for compatibility with PR1 callsites/tests; prefer sirius::io::parse()
// for new code. Both helpers route through the real URI parser.
std::string datasource_factory::extract_scheme(std::string_view uri) { return parse(uri).scheme; }

std::string datasource_factory::extract_path(std::string_view uri) { return parse(uri).path; }

std::unique_ptr<cudf::io::datasource> datasource_factory::create_for_parquet_scan(
  std::string_view uri, datasource_registry const& registry, sirius_config const& config)
{
  // Phase 22.1 D-07: relative bare paths normalize to file:///<absolute>
  // and dispatch through create(). The pre-22.1 bypass to cudf's default
  // datasource routed through libkvikio internally; that's the kvikio path
  // D-09 forbids.
  // DuckDB's iceberg / hive fixtures still hand out paths like
  // "test/cpp/integration/data/...parquet"; we resolve those against the process
  // CWD via std::filesystem::absolute (matches pre-22.1 cudf-default semantics
  // for relative paths) and prepend "file://" so create()'s parser routes them
  // through the registered kFileScheme ioctx (Plan 22.1-01).
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

  // Phase 22.1 D-07/D-09/D-10: ALL schemes (including kFileScheme) MUST be
  // resolved via the registry. The pre-22.1 kFileScheme bypass returned the
  // cudf default datasource which routed through libkvikio internally
  // (binds a single CUDA context per FileHandle, breaks multi-GPU residency).
  // Plan 22.1-01 registers kFileScheme -> sirius_ioctx at SiriusContext::initialize().
  auto ioctx = registry.lookup(p.scheme);
  if (!ioctx) {
    throw std::runtime_error(
      "datasource_factory: no ioctx registered for scheme '" + p.scheme +
      "' — kvikio path is forbidden (uri=" + std::string{uri} + ")");
  }

  // For all currently-supported schemes (kFileScheme post-Plan 22.1-01),
  // the io_object is a uring_io_object constructed from the parsed path.
  // Object-store schemes (s3, gs, azure) will need scheme-specific
  // io_object construction; that work is out of scope for 22.1.
  auto io_object = std::make_shared<sirius::io::uring_io_object>(std::move(p.path));
  return ioctx->make_datasource(std::move(io_object));
}

}  // namespace sirius::io
