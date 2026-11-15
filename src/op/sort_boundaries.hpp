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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::op {

/**
 * @brief The boundary rows of a range sort, readable from every GPU
 *
 * SORT_SAMPLE computes the boundary rows once, on the GPU of its first input batch. Every
 * SORT_PARTITION task then searches its own batch for them, on the GPU that holds that batch. A
 * kernel cannot read another GPU's memory pool, so the rows are kept on the host and each GPU gets
 * its own copy the first time a task on it asks.
 *
 * A copy is complete before any task can see it: the constructor and the first @ref view_for call
 * for a GPU both synchronize their stream before the copy is stored. Later tasks on the same GPU
 * read it from other streams without further ordering. The copies are never modified.
 *
 * Destroy the object only after every task that holds a view from @ref view_for has finished.
 */
class sort_boundaries {
 public:
  /**
   * @brief Takes a copy of @p boundaries, which lives in @p owner
   *
   * Synchronizes @p stream, so the producers of @p boundaries have finished when this returns.
   *
   * @param boundaries The boundary rows (sort key columns only)
   * @param owner The memory space that holds @p boundaries
   * @param stream A stream on @p owner's GPU, ordered after the writers of @p boundaries
   */
  sort_boundaries(cudf::table_view const& boundaries,
                  cucascade::memory::memory_space const& owner,
                  ::cuda::stream_ref stream);

  sort_boundaries(sort_boundaries const&)            = delete;
  sort_boundaries& operator=(sort_boundaries const&) = delete;

  /**
   * @brief Returns the boundary rows in @p space's memory
   *
   * The first call for a memory space allocates from it, uploads the rows on @p stream and
   * synchronizes @p stream. Later calls return the same copy. The view stays valid until this
   * object is destroyed.
   *
   * @param space The memory space of the batch the caller is about to search
   * @param stream A stream on @p space's GPU
   */
  [[nodiscard]] cudf::table_view view_for(cucascade::memory::memory_space const& space,
                                          ::cuda::stream_ref stream);

  //! Number of boundary rows
  [[nodiscard]] cudf::size_type num_rows() const noexcept { return _num_rows; }

  //! Number of memory spaces that hold a copy, including the owner
  [[nodiscard]] std::size_t num_copies() const;

 private:
  struct device_copy {
    std::unique_ptr<rmm::device_buffer> data;
    cudf::table_view view;
  };

  cudf::size_type _num_rows;
  //! The rows as `cudf::pack` lays them out: host metadata plus one contiguous data buffer
  std::vector<std::uint8_t> _metadata;
  std::vector<std::uint8_t> _host_data;

  mutable std::mutex _mutex;
  std::unordered_map<cucascade::memory::memory_space const*, device_copy> _copies;
};

}  // namespace sirius::op
