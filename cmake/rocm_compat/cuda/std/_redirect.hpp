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
//! When hipDF is installed AND its CCCL include path precedes the shim,
//! the real CCCL headers provide @c cuda::std::* with full device support.
//! These redirects are only reached on stock-ROCm-only builds.

#pragma once

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

// Limits
using ::std::numeric_limits;

// Algorithms
using ::std::accumulate;
using ::std::min;
using ::std::max;
using ::std::clamp;
using ::std::copy;
using ::std::fill;
using ::std::transform;
using ::std::for_each;
using ::std::sort;

// Functional
using ::std::equal_to;
using ::std::plus;
using ::std::minus;
using ::std::multiplies;
using ::std::less;
using ::std::greater;
using ::std::hash;

// Type traits
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

// cstdint / cstddef types
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

// Atomic
using ::std::memory_order;
using ::std::memory_order_relaxed;
using ::std::memory_order_acquire;
using ::std::memory_order_release;
using ::std::memory_order_acq_rel;
using ::std::memory_order_seq_cst;
using ::std::atomic;
using ::std::atomic_ref;

// Utility
using ::std::move;
using ::std::forward;
using ::std::pair;
using ::std::make_pair;

}  // namespace std
}  // namespace cuda
