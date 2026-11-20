// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius-c/context/config_builder.h>
#include <sirius-c/version.h>

// Materialize the macro expression so bindgen evaluates UINT32_C.
static const uint32_t SIRIUS_BINDGEN_ABI_VERSION = SIRIUS_ABI_VERSION;
