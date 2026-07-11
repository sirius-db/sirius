/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
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
//! gfx942/gfx950), so the shim defines it as 64. Code that templates on this
//! value (e.g. @c cub::WarpScan<T, cub::detail::warp_threads>) will
//! instantiate with 64 — correct for AMD.
//!
//! Note: hipCUB's ShuffleUp/ShuffleIndex/ShuffleDown ignore the member_mask
//! parameter (rocPRIM does not support masked shuffles). Sirius's @c FULL_MASK
//! is passed but silently ignored — functionally correct for unmasked warps.

#ifndef SIRIUS_ROCM_COMPAT_CUB_CUB_CUH
#define SIRIUS_ROCM_COMPAT_CUB_CUB_CUH

#include <hipcub/hipcub.hpp>

// hipCUB lives in namespace hipcub (versioned inline). Alias it as cub so
// Sirius's cub:: usage resolves.
namespace cub = hipcub;

// CUB exposes cub::detail::warp_threads as a compile-time constant (32 on
// NVIDIA). hipCUB has no equivalent. AMD wavefront = 64 on CDNA (gfx90a/
// gfx942/gfx950 — the architectures hipDF supports).
namespace cub {
namespace detail {
inline constexpr int warp_threads = 64;
}  // namespace detail
}  // namespace cub

#endif  // SIRIUS_ROCM_COMPAT_CUB_CUB_CUH
