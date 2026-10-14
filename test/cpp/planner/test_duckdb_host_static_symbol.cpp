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
#include <fcntl.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Probes with C names so their `.symtab` entries need no demangling. GLOBAL and LOCAL.
extern "C" int sirius_test_host_static_symbol_probe(int value) { return value + 1; }
extern "C" {
static __attribute__((noinline, used)) int sirius_test_host_static_symbol_local_probe(int value)
{ return value + 2; }
}

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

/// Copies the running executable to a temp file, optionally flipping one byte of the GLOBAL
/// probe's code (program headers unchanged). Returns an open fd, or -1.
int copy_of_self(bool corrupt_probe)
{
  std::vector<unsigned char> image;
  int src = ::open("/proc/self/exe", O_RDONLY);
  if (src < 0) return -1;
  unsigned char buf[1 << 16];
  for (ssize_t n; (n = ::read(src, buf, sizeof(buf))) > 0;)
    image.insert(image.end(), buf, buf + n);
  ::close(src);
  if (corrupt_probe) {
    auto const* code =
      reinterpret_cast<unsigned char const*>(&sirius_test_host_static_symbol_probe);
    auto it = std::search(image.begin(), image.end(), code, code + 32);
    if (it == image.end()) return -1;
    *(it + 8) ^= 0xFF;
  }
  char path[] = "/tmp/sirius-host-image-XXXXXX";
  int fd      = ::mkstemp(path);
  if (fd < 0) return -1;
  ::unlink(path);
  if (::write(fd, image.data(), image.size()) != static_cast<ssize_t>(image.size())) {
    ::close(fd);
    return -1;
  }
  return fd;
}

}  // namespace

TEST_CASE("host_static_symbol resolves a GLOBAL function from the mapped image",
          "[planner][duckdb_host]")
{
  auto const* module = main_program_module();
  REQUIRE(module != nullptr);

  void* resolved =
    sirius::planner::detail::host_static_symbol(module, "sirius_test_host_static_symbol_probe");
  REQUIRE(resolved == reinterpret_cast<void*>(&sirius_test_host_static_symbol_probe));
  REQUIRE(reinterpret_cast<int (*)(int)>(resolved)(41) == 42);
}

TEST_CASE("host_static_symbol resolves a unique LOCAL function", "[planner][duckdb_host]")
{
  auto const* module = main_program_module();
  REQUIRE(module != nullptr);

  void* resolved = sirius::planner::detail::host_static_symbol(
    module, "sirius_test_host_static_symbol_local_probe");
  REQUIRE(resolved == reinterpret_cast<void*>(&sirius_test_host_static_symbol_local_probe));
  REQUIRE(reinterpret_cast<int (*)(int)>(resolved)(40) == 42);
}

TEST_CASE("host_static_symbol declines unknown symbols, null and foreign modules",
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

TEST_CASE("host_static_symbol_in_image rejects a replacement file with the same program headers",
          "[planner][duckdb_host]")
{
  auto const* module = main_program_module();
  REQUIRE(module != nullptr);

  int intact = copy_of_self(false);
  REQUIRE(intact >= 0);
  // An exact copy still names the right function: the byte comparison passes.
  REQUIRE(sirius::planner::detail::host_static_symbol_in_image(
            module, "sirius_test_host_static_symbol_probe", intact) ==
          reinterpret_cast<void*>(&sirius_test_host_static_symbol_probe));
  ::close(intact);

  int corrupt = copy_of_self(true);
  REQUIRE(corrupt >= 0);
  // Same program headers, same symbol table, different code at the symbol: rejected.
  REQUIRE(sirius::planner::detail::host_static_symbol_in_image(
            module, "sirius_test_host_static_symbol_probe", corrupt) == nullptr);
  ::close(corrupt);
}
