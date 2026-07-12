/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! CCCL fallback detection for ROCm.
//!
//! When hipDF is installed, its bundled CCCL provides the real @c <cuda/std/*>
//! headers with full device support. The shim directory is @c BEFORE on the
//! include path (needed for @c cuda_runtime.h), which means the shim's
//! @c <cuda/std/*> files shadow hipDF's CCCL.
//!
//! Each @c <cuda/std/X> redirect uses @c #include_next to delegate to hipDF's
//! CCCL if present. If @c #include_next fails (no hipDF), the redirect falls
//! back to the standard @c <X> header + the @c cuda::std namespace alias.
//!
//! This file is included by each redirect AFTER the @c #include_next attempt.

#pragma once

// If we get here, either #include_next found hipDF's CCCL (which defines
// cuda::std::* natively), or the redirect's standard <X> include was used.
// In the latter case, provide the namespace alias.
#ifndef CUDA_STD_REDIRECT_ALREADY_DEFINED

#include <limits>
#include <numeric>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <atomic>
#include <memory>
#include <utility>

namespace cuda {
namespace std {

using ::std::numeric_limits;
using ::std::accumulate;
using ::std::min;
using ::std::max;
using ::std::clamp;
using ::std::copy;
using ::std::fill;
using ::std::transform;
using ::std::for_each;
using ::std::sort;
using ::std::equal_to;
using ::std::plus;
using ::std::minus;
using ::std::multiplies;
using ::std::less;
using ::std::greater;
using ::std::hash;
using ::std::is_same;
using ::std::is_same_v;
using ::std::is_integral;
using ::std::is_floating_point;
using ::std::is_arithmetic;
using ::std::enable_if;
using ::std::enable_if_t;
using ::std::conditional;
using ::std::conditional_t;
using ::std::integral_constant;
using ::std::true_type;
using ::std::false_type;
using ::std::declval;
using ::std::remove_cv;
using ::std::remove_cv_t;
using ::std::remove_reference;
using ::std::remove_reference_t;
using ::std::decay;
using ::std::decay_t;
using ::std::common_type;
using ::std::common_type_t;
using ::std::byte;
using ::std::size_t;
using ::std::ptrdiff_t;
using ::std::uintptr_t;
using ::std::intptr_t;
using ::std::int8_t;
using ::std::int16_t;
using ::std::int32_t;
using ::std::int64_t;
using ::std::uint8_t;
using ::std::uint16_t;
using ::std::uint32_t;
using ::std::uint64_t;
using ::std::memory_order;
using ::std::memory_order_relaxed;
using ::std::memory_order_acquire;
using ::std::memory_order_release;
using ::std::memory_order_acq_rel;
using ::std::memory_order_seq_cst;
using ::std::atomic;
#if __cpp_lib_atomic_ref
using ::std::atomic_ref;
#endif
using ::std::move;
using ::std::forward;
using ::std::pair;
using ::std::make_pair;

}  // namespace std
}  // namespace cuda

#endif  // CUDA_STD_REDIRECT_ALREADY_DEFINED
