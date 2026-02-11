/*
 * Copyright 2025, Sirius Contributors.
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

// sirius
#include <helper/utils.hpp>
#include <memory/host_table_utils.hpp>
#include <memory/multiple_blocks_allocation_accessor.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/vector_size.hpp>
#include <duckdb/main/client_context.hpp>

// cudf
#include <cudf/types.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

// standard library
#include <memory>
#include <vector>

namespace sirius::op::result {

//===----------------------------------------------------------------------===//
// host_table_chunk_reader
//===----------------------------------------------------------------------===//

/**
 * @brief Reads chunks of data from a cucascade::host_table_representation into duckdb data chunkss
 */
class host_table_chunk_reader {
  using multiple_blocks_allocation =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  //===----------------------------------------------------------------------===//
  // column_reader
  //===----------------------------------------------------------------------===//
  /**
   * @brief Reads a single column from the host table representation and produces duckdb vectors and
   * validity masks
   */
  struct column_reader {
    size_t size{0};        ///< The number of rows in the column
    size_t null_count{0};  ///< The number of null values in the column
    cudf::data_type cudf_col_type{
      cudf::type_id::EMPTY};  ///< Source cudf type (with scale for decimals)
    memory::multiple_blocks_allocation_accessor<uint8_t>
      data_accessor;  ///< Accessor to the column data in the multiple blocks allocation
    memory::multiple_blocks_allocation_accessor<uint8_t>
      mask_accessor;  ///< Accessor to the null mask data in the multiple blocks allocation
    memory::multiple_blocks_allocation_accessor<int32_t>
      offset_accessor_32;  ///< Accessor to the STRING offsets (INT32) in the multiple blocks
                           ///< allocation
    memory::multiple_blocks_allocation_accessor<int64_t>
      offset_accessor_64;  ///< Accessor to the STRING offsets (INT64) in the multiple blocks
                           ///< allocation
    bool use_int64_offsets{false};  ///< Whether the offset column uses INT64 (from cudf::pack)

    /**
     * @brief Construct a new column reader object
     * @param[in] node The metadata node for the column (via unpack proxy)
     * @param[in] allocation The multiple blocks allocation containing the column data
     */
    column_reader(metadata_node const& node,
                  std::unique_ptr<multiple_blocks_allocation> const& allocation);

    /**
     * @brief Copy the null mask to the duckdb validity mask for the given row range
     *
     * @param[in,out] validity The duckdb validity mask to copy into
     * @param[in] row_offset The starting row offset to copy from
     * @param[in] count The number of rows to copy
     * @param[in] allocation The multiple blocks allocation containing the column data
     */
    void copy_mask_to_validity(duckdb::ValidityMask& validity,
                               size_t row_offset,
                               size_t count,
                               std::unique_ptr<multiple_blocks_allocation> const& allocation);

    /**
     * @brief Copy fixed-width data into the duckdb vector for the given row range
     *
     * @param[in,out] vector The duckdb vector to copy into
     * @param[in] row_offset The starting row offset to copy from
     * @param[in] count The number of rows to copy
     * @param[in] allocation The multiple blocks allocation containing the column data
     */
    void copy_fixed_width(duckdb::Vector& vector,
                          size_t row_offset,
                          size_t count,
                          std::unique_ptr<multiple_blocks_allocation> const& allocation);

    /**
     * @brief Copy string data into the duckdb vector for the given row range
     *
     * @param[in,out] vector The duckdb vector to copy into
     * @param[in] row_offset The starting row offset to copy from
     * @param[in] count The number of rows to copy
     * @param[in] allocation The multiple blocks allocation containing the column data
     */
    void copy_string(duckdb::Vector& vector,
                     size_t row_offset,
                     size_t count,
                     std::unique_ptr<multiple_blocks_allocation> const& allocation);
  };

 public:
  /**
   * @brief Construct a new host table chunk reader object
   *
   * @param[in] client_ctx The duckdb client context (for allocation)
   * @param[in] host_table The cucascade::host_table_representation to read from
   * @param[in] types The duckdb logical types for the chunk columns
   * @throw std::runtime_error If there is a mismatch in metadata and types, if the row count is
   * negative or inconsistent across metadata_nodes, or if the duckdb output logical type for any
   * column is HUGEINT.
   */
  host_table_chunk_reader(duckdb::ClientContext& client_ctx,
                          cucascade::host_table_representation const& host_table,
                          duckdb::vector<duckdb::LogicalType> const& types);
  ~host_table_chunk_reader() = default;

  /**
   * @brief Get the next data chunk from the host table representation
   *
   * @param[out] chunk The duckdb data chunk to populate
   * @return true If a chunk was read successfully
   *
   * @note This method allocates the memory needed to populate the chunk
   */
  bool get_next_chunk(duckdb::DataChunk& chunk);

  /**
   * @brief Calculate the total number of chunks in the data batch
   *
   * @return size_t The total number of chunks
   */
  size_t calculate_num_chunks()
  {
    return utils::ceil_div(_total_rows, static_cast<size_t>(STANDARD_VECTOR_SIZE));
  }

 private:
  duckdb::ClientContext& _client_ctx;  ///< The duckdb client context (for allocation)
  std::unique_ptr<multiple_blocks_allocation> const&
    _allocation;  ///< The multiple blocks allocation for the data batch
  duckdb::vector<duckdb::LogicalType> _types;  ///< The duckdb logical types for each column
  size_t _total_rows{0};                       ///< The total number of rows in the data batch
  size_t _row_offset{0};                       ///< The current row offset for reading chunks
  std::vector<column_reader> _column_readers;  ///< The column readers for each column
};

}  // namespace sirius::op::result
