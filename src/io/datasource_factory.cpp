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
#include "io/uring/uring_ioctx.hpp"
#include "sirius_config.hpp"

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

std::unique_ptr<io_datasource> datasource_factory::create(std::string_view uri,
                                                          datasource_registry const& registry,
                                                          sirius_config const& /*config*/)
{
  auto p     = parse(uri);
  auto ioctx = registry.lookup(p.scheme);
  if (!ioctx) {
    throw std::runtime_error("datasource_factory: no backend registered for scheme '" + p.scheme +
                             "' (uri=" + std::string{uri} + ")");
  }

  // Per-scheme io_object construction. gds (PR6) and rdma_s3 (PR10) land
  // later; until then, their schemes can be registered but not constructed.
  std::unique_ptr<sirius_io_object> io_object;
  if (p.scheme == kFileScheme) {
    io_object = std::make_unique<uring_io_object>(std::move(p.path));
  } else if (p.scheme == kS3Scheme) {
    // s3://bucket/key — host carries the bucket, path carries the key.
    if (p.host.empty()) throw std::invalid_argument("datasource_factory: s3 URI missing bucket");
    auto* s3_ctx = dynamic_cast<s3::s3_ioctx*>(ioctx.get());
    if (!s3_ctx)
      throw std::runtime_error("datasource_factory: scheme 's3' registered with non-s3 ioctx");
    auto obj_size = s3_ctx->head_object_size(p.host, p.path);
    io_object = std::make_unique<s3::s3_io_object>(std::move(p.host), std::move(p.path), obj_size);
  } else {
    throw std::runtime_error("datasource_factory: scheme '" + p.scheme +
                             "' is registered but object construction is not yet implemented");
  }

  auto ds     = ioctx->make_datasource(std::move(io_object));
  auto* io_ds = dynamic_cast<io_datasource*>(ds.get());
  if (!io_ds) {
    throw std::runtime_error("datasource_factory: ioctx for '" + p.scheme +
                             "' returned a non-io_datasource");
  }
  (void)ds.release();
  return std::unique_ptr<io_datasource>{io_ds};
}

}  // namespace sirius::io
