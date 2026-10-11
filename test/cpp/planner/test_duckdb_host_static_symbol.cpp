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
#include <link.h>

#include <cstdint>

// C linkage: the `.symtab` name needs no demangling.
extern "C" int sirius_test_host_static_symbol_probe(int value) { return value + 1; }

namespace {

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

TEST_CASE("host_static_symbol resolves a function from the executable's static symbol table",
          "[planner][duckdb_host]")
{
  auto const* module = main_program_module();
  REQUIRE(module != nullptr);

  void* resolved =
    sirius::planner::detail::host_static_symbol(module, "sirius_test_host_static_symbol_probe");
  REQUIRE(resolved != nullptr);
  REQUIRE(resolved == reinterpret_cast<void*>(&sirius_test_host_static_symbol_probe));

  auto* probe = reinterpret_cast<int (*)(int)>(resolved);
  REQUIRE(probe(41) == 42);
  Dl_info info{};
  void* owner{};
  REQUIRE(::dladdr1(resolved, &info, &owner, RTLD_DL_LINKMAP) != 0);
  REQUIRE(owner == static_cast<void const*>(module));
}

TEST_CASE("host_static_symbol declines unknown symbols and foreign modules",
          "[planner][duckdb_host]")
{
  auto const* module = main_program_module();
  REQUIRE(module != nullptr);
  REQUIRE(sirius::planner::detail::host_static_symbol(module, "sirius_test_no_such_symbol") ==
          nullptr);
  REQUIRE(sirius::planner::detail::host_static_symbol(
            nullptr, "sirius_test_host_static_symbol_probe") == nullptr);

  Dl_info info{};
  void* owner{};
  REQUIRE(::dladdr1(reinterpret_cast<void*>(&::dlopen), &info, &owner, RTLD_DL_LINKMAP) != 0);
  auto const* libc_module = static_cast<link_map const*>(owner);
  if (libc_module != module) {
    REQUIRE(sirius::planner::detail::host_static_symbol(
              libc_module, "sirius_test_host_static_symbol_probe") == nullptr);
  }
}
