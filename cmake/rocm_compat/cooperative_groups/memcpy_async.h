/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! @c <cooperative_groups/memcpy_async.h> redirect for ROCm.
//! Sirius uses @c cg::memcpy_async with @c ::cuda::aligned_size_t — the
//! aligned_size shim provides the type, and hip_cooperative_groups provides
//! the function. This header just ensures the include resolves.

#pragma once

#include <hip/hip_cooperative_groups.h>
#include "cuda/__memory/aligned_size.h"
