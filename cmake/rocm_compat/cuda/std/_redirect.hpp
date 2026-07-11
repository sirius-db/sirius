/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! Umbrella redirect: maps @c namespace cuda::std to @c namespace std.
//!
//! Each @c <cuda/std/X> header includes the standard @c <X> header then
//! includes this file. hip-clang compiles std types in device code natively
//! (no @c __device__ annotation needed on @c std::numeric_limits etc.),
//! so the alias is sufficient for the operations Sirius uses.
//!
//! When hipDF is installed, its bundled CCCL (patched for HIP) takes
//! precedence on the include path — these redirects are only reached on
//! stock-ROCm-only builds without hipDF.

#pragma once

#include <limits>
#include <numeric>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace cuda {
namespace std {
using std::numeric_limits;
using std::accumulate;
using std::min;
using std::max;
using std::equal_to;
using std::plus;
using std::minus;
using std::multiplies;
using std::is_same;
using std::is_integral;
using std::enable_if;
using std::conditional;
using std::integral_constant;
using std::true_type;
using std::false_type;
}  // namespace std
}  // namespace cuda
