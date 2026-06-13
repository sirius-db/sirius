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

#include <cub/config.cuh>
#include <cuda/std/cstdint>

namespace sirius::cuda::scan::detail {

//! Threads per warp on every supported architecture.
constexpr uint32_t WARP_THREADS = 32;

//! All-lanes mask for the warp `__shfl_*_sync` / ballot collectives.
constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;

}  // namespace sirius::cuda::scan::detail
