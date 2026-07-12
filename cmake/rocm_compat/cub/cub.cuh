/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! CUB → hipCUB compatibility shim.
//!
//! hipCUB provides the same API as CUB but in the @c hipcub:: namespace (on
//! AMD; on NVIDIA it aliases @c cub::). Sirius code uses @c cub:: throughout.
//! This shim pulls in hipCUB and creates @c namespace cub = hipcub; so all
//! @c cub:: symbols resolve to their hipCUB equivalents.
//!
//! @c cub::detail::warp_threads is a CUB internal constant (= 32 on NVIDIA).
//! hipCUB/rocPRIM does not expose it. AMD wavefronts are 64 wide (gfx90a/
//! gfx942/gfx950), so the shim defines it as 64. It must be defined in
//! @c namespace hipcub::detail BEFORE the alias, because you cannot add
//! members to a namespace through its alias in C++.
//!
//! Note: hipCUB's ShuffleUp/ShuffleIndex/ShuffleDown ignore the member_mask
//! parameter (rocPRIM does not support masked shuffles). Sirius's FULL_MASK
//! is passed but silently ignored — functionally correct for unmasked warps.

#ifndef SIRIUS_ROCM_COMPAT_CUB_CUB_CUH
#define SIRIUS_ROCM_COMPAT_CUB_CUB_CUH

#include <hipcub/hipcub.hpp>
// Provide _CCCL_DEVICE/_CCCL_FORCEINLINE macros (used by Sirius detail headers).
#include "cuda/std/__cccl_config.h"

// Define warp_threads INSIDE hipcub::detail (the real namespace), not via
// the alias. You cannot add members through a namespace alias in C++.
namespace hipcub {
namespace detail {
inline constexpr int warp_threads = 64;
}  // namespace detail
}  // namespace hipcub

// Now create the alias. cub::detail::warp_threads resolves to
// hipcub::detail::warp_threads (already defined above).
namespace cub = hipcub;

#endif  // SIRIUS_ROCM_COMPAT_CUB_CUB_CUH
