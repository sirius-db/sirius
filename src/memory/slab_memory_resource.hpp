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

#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace cucascade::memory {
class memory_space;
}

namespace sirius {
namespace memory {

/// The single cudaMalloc region behind a slab pool, which a transport can register once.
struct slab_region {
  int device;
  std::uintptr_t base;
  std::size_t len;
  /// Keeps the slab allocated after its pool is destroyed, e.g. until a transport deregisters it.
  std::shared_ptr<void const> keepalive;

  [[nodiscard]] bool covers(std::uintptr_t p, std::size_t n) const noexcept
  {
    return p >= base && n <= len && p - base <= len - n;
  }
};

/**
 * @brief GPU memory resource factory for a pool that suballocates one cudaMalloc slab.
 *
 * The slab is the capacity rounded down to 2 MiB, taken whole when the pool is built; the pool
 * never grows.
 */
[[nodiscard]] cucascade::memory::DeviceMemoryResourceFactoryFn make_slab_pool_factory();

/// The slab behind @p gpu's allocator, or nullopt unless make_slab_pool_factory built it.
[[nodiscard]] std::optional<slab_region> find_slab(cucascade::memory::memory_space const& gpu);

}  // namespace memory
}  // namespace sirius
