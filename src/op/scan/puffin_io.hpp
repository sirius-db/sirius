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

#pragma once

#include "io/ioctx_resolver.hpp"
#include <cucascade/io/uri_parser.hpp>
#include "op/scan/puffin_reader.hpp"
#include "scan_manager/preparation_stop_token.hpp"

namespace sirius::op::scan {
// Planning only: lookup must not open objects, resolve credentials or probe the network.
struct puffin_io_capability {
  bool remote            = false;
  bool backend_available = true;
  bool permits_route(bool deferred) const { return !remote || (backend_available && deferred); }
};

template <typename Lookup>
puffin_io_capability inspect_puffin_io(std::string const& path, Lookup&& lookup)
{
  puffin_io_capability result;
  std::string scheme;
  try {
    auto parsed   = cucascade::io::parse(path);
    scheme        = parsed.scheme;
    result.remote = scheme != "file";
  } catch (std::invalid_argument const&) {
    // Preserve legacy bare-local-path handling; malformed explicit URIs are refused.
    result.remote = path.find("://") != std::string::npos;
  }
  if (!result.remote) return result;
  auto backend = lookup(path);
  result.backend_available =
    scheme == "s3" && backend &&
    (*backend == cucascade::io::io_context_type::restful || *backend == cucascade::io::io_context_type::kvikio);
  return result;
}

struct opened_puffin_source {
  std::unique_ptr<puffin_byte_source> source;
  std::string identity;
};

// Requires prior route admission. The resolver supplies captured query access scope.
// Typed exceptions remain available to adapter-specific counters and error policy.
opened_puffin_source open_puffin_source(std::string const& path,
                                        io::ioctx_resolver const& resolve,
                                        scan_manager::preparation_stop_token stop);
}  // namespace sirius::op::scan
