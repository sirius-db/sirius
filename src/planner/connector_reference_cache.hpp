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

#include <duckdb/function/table_function.hpp>

#include <unordered_map>
#include <utility>

namespace sirius::planner::detail {

// Trusted definitions are isolated by DuckDB host module. The registry serializes access.
// Empty results are final for planning lookups, just like successful resolutions.
// Only an independent factory or an extension-load bootstrap may publish definitions.
class connector_reference_cache {
 public:
  template <class Resolver>
  duckdb::vector<duckdb::TableFunction> const& get_or_resolve(void const* host, Resolver&& resolve)
  {
    auto& entry = hosts[host];
    if (!entry.resolved) publish(host, std::forward<Resolver>(resolve)());
    return entry.functions;
  }

  bool has_verified_functions(void const* host) const
  {
    auto entry = hosts.find(host);
    return entry != hosts.end() && !entry->second.functions.empty();
  }

  void publish(void const* host, duckdb::vector<duckdb::TableFunction> verified)
  {
    auto& entry     = hosts[host];
    entry.functions = std::move(verified);
    entry.resolved  = true;
  }

 private:
  struct host_reference {
    duckdb::vector<duckdb::TableFunction> functions;
    bool resolved = false;
  };
  std::unordered_map<void const*, host_reference> hosts;
};
}  // namespace sirius::planner::detail
