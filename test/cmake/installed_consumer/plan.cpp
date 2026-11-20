// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/plan/plan.hpp>

// Compile and link plan construction with the installed public headers and library.
auto* volatile public_plan_factory   = &sirius::Plan::from_root;
auto* volatile public_schema_factory = &sirius::Schema::create;
