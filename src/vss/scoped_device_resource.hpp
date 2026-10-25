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

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <utility>

namespace sirius::vss {

/// RAII: route allocations on the current device through @p mr for the guard's lifetime,
/// restoring the previous resource on exit. Needed around any cuVS call whose internal scratch
/// must come out of a Sirius reservation rather than the ambient default, since cuVS allocates
/// from the current device resource and takes no allocator argument.
///
/// The evicted resource is captured by value (an owning any_resource) per the rmm 26.06
/// behavior documented in sirius_memory_reservation_manager: capturing a non-owning ref would
/// dangle once the per-device map entry is moved out.
struct scoped_current_device_resource {
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> prev;

  explicit scoped_current_device_resource(rmm::device_async_resource_ref mr)
    : prev(cudf::set_current_device_resource(mr))
  {
  }
  scoped_current_device_resource(const scoped_current_device_resource&)            = delete;
  scoped_current_device_resource& operator=(const scoped_current_device_resource&) = delete;
  scoped_current_device_resource(scoped_current_device_resource&&)                 = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource&&)      = delete;
  ~scoped_current_device_resource() { cudf::set_current_device_resource(std::move(prev)); }
};

}  // namespace sirius::vss
