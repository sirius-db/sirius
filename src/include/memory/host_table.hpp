/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Vendored from cuCascade (b4abc2d:include/memory/host_table.hpp) when cuCascade
// dropped its libcudf dependency (NVIDIA/cuCascade#142). The cuDF-specific host-table allocation
// now lives in Sirius under namespace sirius::memory; the generic column_metadata it references
// stays in cuCascade.

#pragma once

#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace cucascade {
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius {
namespace memory {

// cuCascade-owned generic memory types referenced by the vendored host-table code, surfaced into
// sirius::memory so the bodies below can reference them unqualified (matching the original code).
using cucascade::memory::column_metadata;
using cucascade::memory::fixed_multiple_blocks_allocation;
using cucascade::memory::fixed_size_host_memory_resource;
using cucascade::memory::memory_space;
using cucascade::memory::Tier;

/**
 * @brief Host memory allocation containing directly-copied column buffers and custom metadata.
 *
 * Unlike host_table_packed_allocation (which relies on cudf::pack/unpack and therefore requires an
 * intermediate GPU-side contiguous copy), this class copies each column's GPU buffers
 * directly to pinned host memory. This avoids the extra GPU memory allocation and bandwidth
 * that cudf::pack incurs.
 *
 * Buffers are reference-counted via shared_ptr so a host_table_allocation produced by slice()
 * shares the underlying storage with its parent. Reconstruction uses the per-column
 * column_metadata descriptors rather than cudf's packed-table serialization format.
 */
class host_table_allocation {
 public:
  using buffers_ptr = std::shared_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation>;

  /**
   * @brief Construct a host_table_allocation taking ownership of @p buffers.
   *
   * The buffers are converted from a unique allocation to a shared one so that future slices
   * of this allocation can share the underlying storage.
   */
  static std::unique_ptr<host_table_allocation> create(fixed_multiple_blocks_allocation buffers,
                                                       std::vector<column_metadata> columns,
                                                       std::size_t data_size);

  /**
   * @brief Create a slice that shares buffers but exposes only the requested columns.
   *
   * Column offsets within the returned allocation reference the same shared buffers; reading
   * from the slice reads from the original storage. The reported `data_size` is the sum of the
   * selected columns' buffer bytes (null masks + data, recursive over children).
   *
   * @param col_indices Column indices to retain in the slice. Must be in [0, num_columns()).
   * @return A new host_table_allocation referencing the shared buffers.
   * @throws std::out_of_range if any index is invalid.
   */
  std::unique_ptr<host_table_allocation> slice(std::span<const std::size_t> col_indices) const;

  /**
   * @brief Deep-copy currently-present column buffers into @p target memory_space.
   *
   * Allocates fresh storage in the target host memory space sized to fit only the columns
   * present in this allocation, copies each column's null-mask and data buffers into a compact
   * 8-byte-aligned layout, and returns a new owning host_table_allocation.
   *
   * @param target Host-tier memory_space that owns a fixed_size_host_memory_resource.
   */
  std::unique_ptr<host_table_allocation> clone(memory_space& target) const;

  /// @brief Number of top-level columns described by this allocation.
  [[nodiscard]] std::size_t num_columns() const noexcept { return columns.size(); }

  /**
   * @brief Total bytes (null mask + data, recursive over children) for column @p i.
   *
   * @throws std::out_of_range if @p i >= num_columns().
   */
  [[nodiscard]] std::size_t column_size(std::size_t i) const;

  /// Shared pinned blocks containing all raw column buffer data.
  buffers_ptr allocation;
  /// One metadata entry per top-level column; children are stored recursively.
  std::vector<column_metadata> columns;
  /// Total number of live data bytes referenced by `columns`.
  std::size_t data_size{0};

 private:
  host_table_allocation(buffers_ptr buffers,
                        std::vector<column_metadata> cols,
                        std::size_t data_sz);
};

}  // namespace memory
}  // namespace sirius
