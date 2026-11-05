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

#include <dlfcn.h>
#include <execinfo.h>
#include <link.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace {

struct fault_state {
  int scope            = 0;
  std::size_t fail_at  = 0;
  std::size_t observed = 0;
  bool inspecting      = false;
  bool fired           = false;
};

thread_local fault_state fault;

struct function_range {
  std::uintptr_t begin;
  std::size_t bytes;
};

function_range load_range(char const* variable)
{
  auto const* value = std::getenv(variable);
  if (value == nullptr) { std::abort(); }
  char* separator    = nullptr;
  auto const address = std::strtoull(value, &separator, 16);
  if (*separator != ':') { std::abort(); }
  auto const bytes    = std::strtoull(separator + 1, nullptr, 16);
  std::uintptr_t base = 0;
  dl_iterate_phdr(
    [](dl_phdr_info* image, std::size_t, void* result) {
      if (image->dlpi_name[0] != '\0') { return 0; }
      *static_cast<std::uintptr_t*>(result) = image->dlpi_addr;
      return 1;
    },
    &base);
  return {base + address, bytes};
}

bool matches_scope() noexcept
{
  if (fault.scope == 2) { return true; }
  static auto const range = load_range("SIRIUS_HOST_FAULT_SHELL_RANGE");
  void* frames[32];
  auto const count = backtrace(frames, 32);
  for (int index = 0; index < count; ++index) {
    auto const address = reinterpret_cast<std::uintptr_t>(frames[index]);
    if (address >= range.begin && address - range.begin < range.bytes) { return true; }
  }
  return false;
}

}  // namespace

/**
 * @brief Arms one publisher thread: scope 1 selects filter shells and 2 selects every allocation. A
 * zero ordinal only counts matching allocations.
 *
 * This helper is loaded only for the hidden host-allocation tests. Ordinary test runs do not link
 * or load it.
 */
extern "C" void sirius_test_host_fault_arm(int scope, std::size_t ordinal) noexcept
{
  fault = {.scope = scope, .fail_at = ordinal};
}

extern "C" std::size_t sirius_test_host_fault_disarm() noexcept
{
  fault.scope = 0;
  return fault.observed;
}

extern "C" bool sirius_test_host_fault_fired() noexcept { return fault.fired; }

void* operator new(std::size_t bytes)
{
  using allocator            = void* (*)(std::size_t);
  static auto const upstream = reinterpret_cast<allocator>(dlsym(RTLD_NEXT, "_Znwm"));
  if (upstream == nullptr) { std::abort(); }
  if (fault.scope != 0 && !fault.inspecting) {
    fault.inspecting   = true;
    auto const matches = matches_scope();
    fault.inspecting   = false;
    if (matches && ++fault.observed == fault.fail_at) {
      fault.scope = 0;
      fault.fired = true;
      throw std::bad_alloc{};
    }
  }
  return upstream(bytes);
}
