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

#include <catch.hpp>
#include <dlfcn.h>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb_static_extension.h>
#include <link.h>

// The two names `reference_functions()` looks up in a host image. They must stay in sync with
// `src/planner/connector_registry.cpp`; this test fails when a DuckDB upgrade renames either.
extern "C" int32_t duckdb_extension_parquet_describe(duckdb_extension_descriptor*);

namespace {

constexpr char const* kTableScanSymbol = "_ZN6duckdb17TableScanFunction11GetFunctionEv";
constexpr char const* kParquetSymbol   = "duckdb_extension_parquet_describe";

link_map const* main_program_module()
{
  void* handle = ::dlopen(nullptr, RTLD_NOW | RTLD_NOLOAD);
  if (!handle) return nullptr;
  link_map* module{};
  if (::dlinfo(handle, RTLD_DI_LINKMAP, &module) != 0) module = nullptr;
  ::dlclose(handle);
  return module;
}

}  // namespace

TEST_CASE("the production reference symbols resolve from this DuckDB image's static symbol table",
          "[planner][duckdb_host]")
{
  auto const* module = main_program_module();
  REQUIRE(module != nullptr);

  // The test executable links DuckDB statically and exports its symbols, so this exercises
  // the static-table helper on its own image, not the no-export host path as a whole: the
  // lookup must find the same functions the linker bound.
  void* table_scan = sirius::planner::detail::host_static_symbol(module, kTableScanSymbol);
  REQUIRE(table_scan != nullptr);
  REQUIRE(table_scan == reinterpret_cast<void*>(&duckdb::TableScanFunction::GetFunction));

  void* parquet = sirius::planner::detail::host_static_symbol(module, kParquetSymbol);
  REQUIRE(parquet != nullptr);
  REQUIRE(parquet == reinterpret_cast<void*>(&duckdb_extension_parquet_describe));

  // Calling through the resolved address yields the same definition the direct call does.
  auto get_table_scan = reinterpret_cast<duckdb::TableFunction (*)()>(table_scan);
  auto resolved       = get_table_scan();
  auto direct         = duckdb::TableScanFunction::GetFunction();
  CHECK(resolved.name == direct.name);
  CHECK(resolved.function == direct.function);
  CHECK(resolved.bind == direct.bind);
}
