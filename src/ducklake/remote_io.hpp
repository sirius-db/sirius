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
#include "ducklake/dv_visibility.hpp"
#include "op/scan/puffin_io.hpp"

namespace sirius::ducklake {
// Called for a selected unit on the provider owner lane before constructing its
// reader. Only metadata-only lookup is allowed here; no credential/network probe.
template <typename Lookup>
void validate_unit_io(unit_descriptor const& unit, bool deferred, Lookup&& lookup)
{
  validate_parquet_unit_dependencies(unit, dependency_scope::s3_deletes);
  for (auto const* path : {&unit.input.path, &unit.delete_path}) {
    if (path->empty()) continue;
    std::optional<cucascade::io::io_context_type> backend;
    auto capability = op::scan::inspect_puffin_io(*path, [&](auto const& uri) {
      backend = lookup(uri);
      return backend;
    });
    if (!capability.remote) continue;
    if (!capability.backend_available)
      throw std::invalid_argument("DuckLake dependency has no planning-capable remote backend: " +
                                  *path);
    if (!capability.permits_route(deferred))
      throw std::invalid_argument("DuckLake remote dependencies require deferred preparation");
    // Additional backends require their own DuckLake qualification receipts.
    if (backend != cucascade::io::io_context_type::restful)
      throw std::invalid_argument("DuckLake remote dependency backend is not qualified: " + *path);
  }
}
}  // namespace sirius::ducklake
