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

#include "memory/topology_index.hpp"

#include <optional>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::preload {

/// Hardware capabilities forced by tests. Production leaves both empty, so they are detected.
struct hw_caps_override {
  std::optional<bool> c2c;
  std::optional<bool> gds;
};

/**
 * @brief Pure C2C decision from a GPU's CUDA driver attributes.
 *
 * A cache-coherent chip-to-chip host link is present only when the device supports native host
 * atomics and reaches pageable host memory through the host page tables. Testable without
 * hardware.
 *
 * @param host_native_atomics CUDA `cudaDevAttrHostNativeAtomicSupported`; 0 when unavailable.
 * @param host_page_tables     CUDA `cudaDevAttrPageableMemoryAccessUsesHostPageTables`; 0 when
 *                             unavailable.
 */
[[nodiscard]] constexpr bool c2c_from_attributes(int host_native_atomics,
                                                 int host_page_tables) noexcept
{
  return host_native_atomics != 0 && host_page_tables != 0;
}

/**
 * @brief Whether GPU @p device reaches host memory over a cache-coherent chip-to-chip link.
 *
 * Read from the CUDA runtime device attributes (native host atomics plus host-page-table access).
 * False when the device is out of range or the attributes cannot be queried. The @p topology is
 * accepted for symmetry with the rest of preload but is not consulted; @p device is a CUDA device
 * ordinal.
 */
[[nodiscard]] bool gpu_has_c2c(memory::topology_index const& topology, int device) noexcept;

/**
 * @brief Whether DISK space @p disk moves data with GPUDirect Storage.
 *
 * The pinned cuCascade exposes no GDS / io-context query, so preload treats every DISK lane as
 * non-GDS: this always returns false. Kept as a function so the executor's lane sizing and token
 * rules are unchanged from the GDS-capable design.
 */
[[nodiscard]] bool disk_uses_gds(cucascade::memory::memory_space const& disk) noexcept;

}  // namespace sirius::preload
