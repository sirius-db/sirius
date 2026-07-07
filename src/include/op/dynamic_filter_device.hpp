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

#pragma once

#include <cuda_runtime_api.h>

namespace sirius::op::detail {

/// @brief Resolve a dynamic-filter device argument, treating a negative ID as the current device.
///
/// @return The explicit device ID, the current CUDA device, or -1 if CUDA cannot report one.
[[nodiscard]] inline int resolve_dynamic_filter_device_id(int device_id) noexcept
{
  if (device_id >= 0) { return device_id; }
  int current = -1;
  return cudaGetDevice(&current) == cudaSuccess ? current : -1;
}

}  // namespace sirius::op::detail
