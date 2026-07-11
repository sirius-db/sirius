/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! CCCL `<cuda/std/*>` redirect for ROCm/HIP.
//!
//! NVIDIA's CCCL (libcudacxx) provides `<cuda/std/limits>`, `<cuda/std/numeric>`,
//! etc. — device-annotated versions of std types in `namespace cuda::std`.
//! CCCL does NOT detect HIP (it only checks `__CUDACC__`/`__NVCC__`), so on
//! stock ROCm these headers are unavailable or fall to the host path.
//!
//! hipDF bundles its own CCCL patched for HIP. When hipDF is installed and on
//! the include path, these redirects are NOT reached (hipDF's CCCL takes
//! precedence). They exist as a fallback for stock-ROCm-only builds.
//!
//! The redirect maps `<cuda/std/X>` → standard `<X>` and aliases
//! `cuda::std` → `std`. This works because hip-clang treats lambdas and
//! functors in device code natively (no `__device__` annotation needed on
//! std types for the operations Sirius uses: numeric_limits, accumulate,
//! etc.).

#pragma once

// Map cuda::std to std. hip-clang compiles std types in device code natively.
#include_next <limits>
namespace cuda { namespace std { using std::numeric_limits; } }
