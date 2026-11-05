// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "telemetry/nvtx_injection.hpp"

extern "C" int quent_InitializeInjectionNvtx2(void*) { return 42; }

// Match Quent's strong static-injection shim: capture starts armed, and Sirius
// must explicitly clear it when NVTX is disabled.
extern "C" {
int (*InitializeInjectionNvtx2_fnptr)(void*) = &quent_InitializeInjectionNvtx2;
}

extern "C" __attribute__((visibility("default"))) bool injection_armed()
{
  return InitializeInjectionNvtx2_fnptr != nullptr;
}

extern "C" __attribute__((visibility("default"))) bool configure_injection(bool enabled,
                                                                           const char* library)
{
  sirius::telemetry::detail::configure_nvtx_injection(enabled, library);
  return injection_armed();
}
