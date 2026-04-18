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
  for (auto const& [scheme, _] : _ioctxs) out.push_back(scheme);
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

constexpr std::string_view kSchemeDelim = "://";
constexpr std::string_view kFileScheme  = "file";

}  // namespace

std::string datasource_factory::extract_scheme(std::string_view uri)
{
  if (uri.empty()) throw std::invalid_argument("datasource_factory: empty URI");
  if (uri.front() == '/') return std::string{kFileScheme};

  auto pos = uri.find(kSchemeDelim);
  if (pos == std::string_view::npos) return std::string{kFileScheme};
  if (pos == 0) throw std::invalid_argument("datasource_factory: URI has empty scheme");
  return std::string{uri.substr(0, pos)};
}

std::string datasource_factory::extract_path(std::string_view uri)
{
  if (uri.empty()) throw std::invalid_argument("datasource_factory: empty URI");
  auto pos = uri.find(kSchemeDelim);
  if (pos == std::string_view::npos) return std::string{uri};
  return std::string{uri.substr(pos + kSchemeDelim.size())};
}

std::unique_ptr<io_datasource> datasource_factory::create(std::string_view uri,
                                                          datasource_registry const& registry,
                                                          sirius_config const& /*config*/)
{
  auto scheme = extract_scheme(uri);
  auto ioctx  = registry.lookup(scheme);
  if (!ioctx) {
    throw std::runtime_error("datasource_factory: no backend registered for scheme '" + scheme +
                             "' (uri=" + std::string{uri} + ")");
  }

  auto path = extract_path(uri);

  // PR1 only knows how to build uring_io_object (scheme == "file"). Other
  // schemes' io_object construction lands in their respective PRs: gds in PR6,
  // s3 in PR9, rdma_s3 in PR10.
  std::unique_ptr<sirius_io_object> io_object;
  if (scheme == kFileScheme) {
    io_object = std::make_unique<uring_io_object>(std::move(path));
  } else {
    throw std::runtime_error("datasource_factory: scheme '" + scheme +
                             "' is registered but object construction is not implemented in PR1");
  }

  auto ds = ioctx->make_datasource(std::move(io_object));
  auto* io_ds = dynamic_cast<io_datasource*>(ds.get());
  if (!io_ds) {
    throw std::runtime_error("datasource_factory: ioctx for '" + scheme +
                             "' returned a non-io_datasource");
  }
  (void)ds.release();
  return std::unique_ptr<io_datasource>{io_ds};
}

}  // namespace sirius::io
