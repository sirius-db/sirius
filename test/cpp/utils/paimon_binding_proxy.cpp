// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <cstdint>

// Force a real dependency on the fixture, but export no Paimon initializer here.
// DuckDB's dlsym can LOAD through the dependency. The consumer must refuse it.
extern "C" uint64_t sirius_paimon_fixture_released();
extern "C" __attribute__((visibility("default"))) uint64_t sirius_paimon_proxy_reference()
{
  return sirius_paimon_fixture_released();
}
