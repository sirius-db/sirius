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

#include "io/s3/s3_io_object.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/uri_parser.hpp"
#include "sirius_config.hpp"

#include <cudf/io/datasource.hpp>

#include <stdexcept>
#include <utility>

namespace sirius::io {

// ---------------------------------------------------------------------------
// datasource_registry
// ---------------------------------------------------------------------------

void datasource_registry::register_ioctx(std::string scheme, std::shared_ptr<sirius_ioctx> ioctx)
{
  if (scheme.empty()) throw std::invalid_argument("datasource_registry: empty scheme");
  if (!ioctx) throw std::invalid_argument("datasource_registry: null ioctx for '" + scheme + "'");
  std::unique_lock lk{_mtx};
  _ioctxs[std::move(scheme)] = std::move(ioctx);
}

std::shared_ptr<sirius_ioctx> datasource_registry::lookup(std::string_view scheme) const
{
  std::shared_lock lk{_mtx};
  auto it = _ioctxs.find(std::string{scheme});
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
constexpr std::string_view kS3Scheme   = "s3";

}  // namespace

// Retained for compatibility with PR1 callsites/tests; prefer sirius::io::parse()
// for new code. PR8 routes both through the real URI parser.
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

  std::shared_ptr<sirius_io_object> io_object;
  if (p.scheme == kS3Scheme) {
    // s3://bucket/key — host carries the bucket, path carries the key.
    if (p.host.empty()) throw std::invalid_argument("datasource_factory: s3 URI missing bucket");
    auto* s3_ctx = dynamic_cast<s3::s3_ioctx*>(ioctx.get());
    if (!s3_ctx)
      throw std::runtime_error("datasource_factory: scheme 's3' registered with non-s3 ioctx");
    auto obj_size = s3_ctx->head_object_size(p.host, p.path);
    io_object = std::make_shared<s3::s3_io_object>(std::move(p.host), std::move(p.path), obj_size);
  } else {
    throw std::runtime_error("datasource_factory: scheme '" + p.scheme +
                             "' is registered but object construction is not yet implemented");
  }

  return ioctx->make_datasource(std::move(io_object));
}

}  // namespace sirius::io
