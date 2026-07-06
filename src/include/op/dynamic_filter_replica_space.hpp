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

#include <functional>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::op {

/// @brief Non-owning placement handle for one device-local dynamic-filter replica.
///
/// The GPU memory space supplies the replica allocator and pooled CUDA stream. The paired HOST
/// memory space supplies NUMA-local, pre-pinned fixed blocks when direct peer DMA is unavailable.
class dynamic_filter_replica_space final {
 public:
  dynamic_filter_replica_space(cucascade::memory::memory_space const& gpu_space,
                               cucascade::memory::memory_space const& host_staging_space) noexcept
    : _gpu_space{gpu_space}, _host_staging_space{host_staging_space}
  {
  }

  [[nodiscard]] cucascade::memory::memory_space const& get_gpu_space() const noexcept
  { return _gpu_space.get(); }

  [[nodiscard]] cucascade::memory::memory_space const& get_host_staging_space() const noexcept
  { return _host_staging_space.get(); }

 private:
  std::reference_wrapper<cucascade::memory::memory_space const> _gpu_space;
  std::reference_wrapper<cucascade::memory::memory_space const> _host_staging_space;
};

}  // namespace sirius::op
