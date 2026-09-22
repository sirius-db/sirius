// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "telemetry/nvtx_injection.hpp"

extern "C" int quent_InitializeInjectionNvtx2(void*) { return 42; }

extern "C" __attribute__((visibility("default"))) void configure_injection(bool enabled,
                                                                           const char* library)
{
  sirius::telemetry::detail::configure_nvtx_injection(enabled, library);
}
