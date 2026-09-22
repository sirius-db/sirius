// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "telemetry/nvtx_injection.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

extern "C" void configure_injection(bool enabled, const char* library);

static void require(bool condition, const char* message)
{
  if (!condition) { throw std::runtime_error(message); }
}

int main(int argc, char** argv)
{
  try {
    auto* process = dlopen(nullptr, RTLD_LAZY | RTLD_LOCAL);
    require(process != nullptr, "ordinary dlopen request was not forwarded");
    dlclose(process);

    unsetenv("NVTX_INJECTION64_PATH");
    configure_injection(false, "");
    require(getenv("NVTX_INJECTION64_PATH") == nullptr, "disabled telemetry changed discovery");

    setenv("NVTX_INJECTION64_PATH", "existing-injector", 1);
    configure_injection(true, "configured-injector");
    require(std::string(getenv("NVTX_INJECTION64_PATH")) == "existing-injector",
            "existing environment did not take precedence");

    unsetenv("NVTX_INJECTION64_PATH");
    configure_injection(true, "configured-injector");
    require(std::string(getenv("NVTX_INJECTION64_PATH")) == "configured-injector",
            "explicit configuration was ignored");

    unsetenv("NVTX_INJECTION64_PATH");
    configure_injection(true, "");
    auto* path = getenv("NVTX_INJECTION64_PATH");
    require(path != nullptr, "default discovery path missing");
    if (argc == 2) {
      require(std::filesystem::equivalent(path, argv[1]), "shared discovery chose the wrong image");
    } else {
      require(std::string(path) == sirius::telemetry::detail::static_injection_path,
              "static discovery did not select the executable");
    }
    auto* handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    require(handle != nullptr, "cannot open the selected injector");
    auto initialize = reinterpret_cast<int (*)(void*)>(dlsym(handle, "InitializeInjectionNvtx2"));
    require(initialize != nullptr, "injector initializer was discarded");
    require(initialize(nullptr) == 42, "initializer does not forward to the embedded injector");
    dlclose(handle);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
