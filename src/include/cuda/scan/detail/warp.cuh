/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! @file
//! Warp-level constants shared across the decode kernels.

#pragma once

#include <cub/util_arch.cuh>  // cub::detail::warp_threads
#include <cuda/std/cstdint>

namespace sirius::cuda::scan::detail {

//! All-lanes mask for the warp shuffle / ballot collectives.
//! On AMD (gfx90a/942/950) the wavefront is 64 wide; on NVIDIA the warp is 32.
//! Using 64-bit covers both: the upper 32 bits are ignored on NVIDIA (only
//! 32 lanes exist), and on AMD all 64 lanes are enabled.
constexpr unsigned long long FULL_MASK = 0xFFFFFFFFFFFFFFFFull;

}  // namespace sirius::cuda::scan::detail
