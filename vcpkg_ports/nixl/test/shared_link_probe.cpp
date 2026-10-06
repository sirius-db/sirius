/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nixl.h>

extern "C" int nixl_static_link_probe()
{
  nixlAgentConfig config;
  nixlAgent agent("static-nixl-shared-library", config);
  nixlBackendH* backend{};
  return agent.createBackend("UCX", {}, backend);
}
