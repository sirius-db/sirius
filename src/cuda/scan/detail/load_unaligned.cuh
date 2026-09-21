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
//! Alignment-agnostic scalar load for on-disk segment reads whose byte address
//! is not guaranteed to satisfy `alignof(T)` (e.g. a typed field at a
//! byte-granular segment offset). Reading through a typed pointer there is
//! undefined behaviour; `load_unaligned` reads via `memcpy` instead.

#pragma once

#include <cub/config.cuh>
#include <cuda/std/cstring>

namespace sirius::cuda::scan::detail {

//! @brief Read a trivially-copyable @p T from @p p at any byte alignment.
template <typename T>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T load_unaligned(const void* p)
{
  T v;
  memcpy(&v, p, sizeof(T));
  return v;
}

}  // namespace sirius::cuda::scan::detail
