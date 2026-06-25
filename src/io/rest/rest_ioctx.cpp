/*
 * Copyright 2026, Sirius Contributors.
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

#include "io/rest/rest_ioctx.hpp"

#include "io/uri_parser.hpp"

#include <fmt/format.h>

#include <stdexcept>
#include <utility>

namespace sirius::io::rest {

rest_ioctx::rest_ioctx(std::size_t n_reactors, std::shared_ptr<rest_reactor::reactor_context> ctx)
  : templated_ioctx<rest_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<rest_reactor>(ctx, fmt::format("rest-{}", i++));
    })
{
}

std::shared_ptr<sirius_io_object> rest_ioctx::create_io_object(std::string path)
{
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("rest_ioctx::create_io_object: unsupported scheme '" +
                                parsed.scheme + "'");
  }
  if (_reactors.empty()) { throw std::runtime_error("rest_ioctx::create_io_object: no reactors"); }

  // A blocking HEAD on the caller thread (a one-time metadata round-trip) via
  // any reactor's authorizer — head_object_size uses a local easy handle and
  // does not touch worker state, so any reactor is equivalent.
  size_t const size = _reactors.front()->head_object_size(parsed.host, parsed.path);
  return std::make_shared<rest_io_object>(
    std::move(path), std::move(parsed.host), std::move(parsed.path), size);
}

}  // namespace sirius::io::rest
