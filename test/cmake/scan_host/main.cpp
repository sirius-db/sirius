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

#include "planner/duckdb_host.hpp"

#include <duckdb/catalog/duck_catalog.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/main/database.hpp>

#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, char const* message)
{
  if (!condition) throw std::runtime_error(message);
}

void const* module_of(void const* address)
{
  Dl_info info{};
  void* module{};
  check(::dladdr1(address, &info, &module, RTLD_DL_LINKMAP), "dladdr1 failed");
  return module;
}

duckdb::TableFunction wrong_factory() { return {}; }
}  // namespace

int main()
{
  try {
    duckdb::DBConfig config;
    config.options.maximum_threads = 1;
    config.options.load_extensions = false;
    duckdb::DuckDB db(nullptr, &config);
    auto& catalog      = duckdb::Catalog::GetSystemCatalog(*db.instance);
    auto* factory      = &duckdb::TableScanFunction::GetFunction;
    auto const* symbol = "_ZN6duckdb17TableScanFunction11GetFunctionEv";
    // Referencing the concrete RTTI forces a copy relocation in an x86-64 PIE.
    std::cout << "DuckCatalog RTTI: " << &typeid(duckdb::DuckCatalog) << '\n';
#ifdef EXPECT_COPY_RELOCATION
    check(module_of(&typeid(catalog)) == module_of(reinterpret_cast<void*>(&wrong_factory)),
          "fixture did not copy RTTI into the executable");
    check(module_of(&typeid(catalog)) != module_of(reinterpret_cast<void*>(factory)),
          "fixture must keep scan code in the shared library");
#endif
    check(sirius::planner::detail::host_module(*db.instance) ==
            module_of(reinterpret_cast<void*>(factory)),
          "host detection selected copied metadata instead of DuckDB code");
    auto resolved = sirius::planner::detail::host_factory(*db.instance, factory, symbol);
    check(resolved && resolved().function == factory().function,
          "valid shared DuckDB scan factory rejected");
    resolved = sirius::planner::detail::host_factory(*db.instance, &wrong_factory, symbol);
    check(resolved && resolved().function == factory().function,
          "failed to resolve the host factory independently of the local implementation");
    check(!sirius::planner::detail::host_factory(*db.instance, &wrong_factory, "missing_factory"),
          "missing host export must not trust a foreign factory");
    std::cout << "Shared DuckDB scan verification passed\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
