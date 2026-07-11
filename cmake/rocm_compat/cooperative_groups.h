/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! @c <cooperative_groups.h> redirect for ROCm.
//! ROCm provides cooperative groups via @c <hip/hip_cooperative_groups.h>.
//! Sirius includes the CUDA-style @c <cooperative_groups.h> which doesn't
//! exist on stock ROCm. This redirect maps it.

#pragma once

#include <hip/hip_cooperative_groups.h>

// HIP puts cooperative_groups in the same namespace as CUDA.
// No namespace alias needed — hip_cooperative_groups.h defines
// namespace cooperative_groups { ... } directly.
