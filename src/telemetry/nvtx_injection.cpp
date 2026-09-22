/*
 * Copyright 2025, Sirius Contributors.
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

#include "telemetry/nvtx_injection.hpp"

#include <dlfcn.h>
#include <link.h>

#include <cstdlib>
#include <filesystem>
#include <string_view>

// Statically embedded by telemetry_bridge. Rust archives are localized by the
// extension link (`--exclude-libs,ALL`), so this regular C++ object supplies the
// public NVTX entry point and forwards it into the same Quent hook state used by
// Sirius's static-injection pointer.
extern "C" int quent_InitializeInjectionNvtx2(void* get_export_table);
extern "C" __attribute__((visibility("default"))) int InitializeInjectionNvtx2(
  void* get_export_table)
{
  return quent_InitializeInjectionNvtx2(get_export_table);
}

// NVTX v3 discovers an injector independently in each ELF image. libcudf's
// injection pointer is local to libcudf.so and its process-global preinjection
// lookup is compiled out, so it can only reach Quent through its dlopen path.
// A statically linked Sirius has no DSO to name. Interpose just our private
// sentinel and turn that request into dlopen(NULL), whose handle exposes the
// initializer exported by the running DuckDB executable. Every other request
// is forwarded unchanged to libc.
extern "C" __attribute__((visibility("default"))) void* dlopen(const char* filename, int flags)
{
  using dlopen_fn   = void* (*)(const char*, int);
  auto* real_dlopen = reinterpret_cast<dlopen_fn>(::dlsym(RTLD_NEXT, "dlopen"));
  if (real_dlopen == nullptr) { return nullptr; }

  if (filename != nullptr &&
      std::string_view{filename} == sirius::telemetry::detail::static_injection_path) {
    return real_dlopen(nullptr, flags);
  }
  return real_dlopen(filename, flags);
}

namespace sirius::telemetry::detail {

void configure_nvtx_injection(bool enabled, const std::string& library) noexcept
{
  if (std::getenv("NVTX_INJECTION64_PATH") != nullptr || !enabled) { return; }

  if (!library.empty()) {
    ::setenv("NVTX_INJECTION64_PATH", library.c_str(), /*overwrite=*/0);
    return;
  }

  try {
    Dl_info self{};
    void* extra = nullptr;
    // The main executable has an empty link-map name, including PIE builds.
    // Use a private anchor so another image cannot interpose its address.
    if (::dladdr1(
          reinterpret_cast<void*>(&configure_nvtx_injection), &self, &extra, RTLD_DL_LINKMAP) ==
          0 ||
        extra == nullptr) {
      return;
    }
    auto const* mapping = static_cast<const link_map*>(extra);
    if (mapping->l_name == nullptr) { return; }
    if (mapping->l_name[0] == '\0') {
      // Probe before publishing: NVTX cannot fall back after dynamic lookup fails.
      auto* handle = ::dlopen(static_injection_path, RTLD_LAZY | RTLD_LOCAL);
      if (handle == nullptr) { return; }
      auto* initializer = ::dlsym(handle, "InitializeInjectionNvtx2");
      bool const usable = initializer == reinterpret_cast<void*>(&InitializeInjectionNvtx2);
      ::dlclose(handle);
      if (usable) { ::setenv("NVTX_INJECTION64_PATH", static_injection_path, /*overwrite=*/0); }
      return;
    }
    if (self.dli_fname == nullptr) { return; }

    std::error_code error;
    auto self_path = std::filesystem::canonical(self.dli_fname, error);
    if (error) { return; }
    ::setenv("NVTX_INJECTION64_PATH", self_path.c_str(), /*overwrite=*/0);
  } catch (...) {
    // NVTX capture is optional; self-path discovery must not prevent Sirius
    // from loading.
  }
}

}  // namespace sirius::telemetry::detail
