/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! @c <curand.h> redirect for ROCm.
//! Legacy Sirius code (src/legacy/cuda/operator/cuda_helper.cuh:31) includes
//! @c <curand.h>. ROCm's equivalent is @c <hiprand.h>.

#pragma once

#include <hiprand.h>
