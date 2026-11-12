// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/context/context.hpp>

// Compile the public context header independently of engine and integration headers.
auto* volatile public_context_factory = &sirius::Context::create;
