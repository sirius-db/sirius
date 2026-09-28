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

#include <utility>

namespace sirius::planner::detail {

// One cache per connector in the loaded Sirius module. The registry serializes access.
// Empty results are final for planning lookups, just like successful resolutions.
// Only an independent factory or an extension-load bootstrap may publish definitions.
class connector_reference_cache {
 public:
  template <class Resolver>
  duckdb::vector<duckdb::TableFunction> const& get_or_resolve(Resolver&& resolve)
  {
    if (!resolved) publish(std::forward<Resolver>(resolve)());
    return functions;
  }

  bool has_verified_functions() const { return !functions.empty(); }

  void publish(duckdb::vector<duckdb::TableFunction> verified)
  {
    functions = std::move(verified);
    resolved  = true;
  }

 private:
  duckdb::vector<duckdb::TableFunction> functions;
  bool resolved = false;
};
}  // namespace sirius::planner::detail
