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

#include <data/data_repository_manager.hpp>
#include <memory/fixed_size_host_memory_resource.hpp>
#include <memory/memory_reservation.hpp>
#include <operator/gpu_physical_table_scan.hpp>
#include <parallel/task.hpp>
#include <scan/duckdb_scan_executor.hpp>
#include <scan/physical_table_scan_adapter.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/client_context.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sirius::parallel {

//===----------------------------------------------------------------------===//
// DuckDB Scan Task Global State
//===----------------------------------------------------------------------===//

/**
 * @brief TODO
 */
class duckdb_scan_task_global_state : public itask_global_state, public duckdb::GlobalSourceState {
 public:
  duckdb_scan_task_global_state(uint64_t pipeline_id,
                                duckdb_scan_executor& scan_exec,
                                duckdb::ClientContext& client_ctx,
                                duckdb::physical_table_scan_adapter const& ptsa);

  uint64_t MaxThreads() override { return max_threads; }

  bool IsSourceDrained() const { return source_drained.load(std::memory_order_acquire); }

  void SetSourceDrained() { source_drained.store(true, std::memory_order_release); }

  //===----------Fields----------===//
  std::atomic<bool> source_drained{false};  ///< Whether the table scan source is fully drained
  uint64_t pipeline_id;                     ///< The pipeline id to which this table scan belongs
  uint64_t max_threads;                     ///< Maximum number of threads for this scan task

  unique_ptr<duckdb::GlobalTableFunctionState>
    global_tf_state;  ///< Global state for the table function
  duckdb_scan_executor& scan_executor;
  const duckdb::PhysicalTableScan& op;  ///< The physical table scan operator adapter being executed
};

//===----------------------------------------------------------------------===//
// DuckDB Scan Task Local State
//===----------------------------------------------------------------------===//
template <typename T>
struct multiple_blocks_allocation_accessor {
  using underlying_type = T;
  using multiple_blocks_allocation =
    memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  size_t block_index     = 0;
  size_t offset_in_block = 0;
  unique_ptr<multiple_blocks_allocation> allocation;

  void initialize(unique_ptr<multiple_blocks_allocation> alloc)
  {
    allocation = std::move(alloc);

    // We require the underlying type to align with the block size.
    if (allocation->block_size % sizeof(underlying_type) != 0) {
      std::string error_msg =
        "[multiple_blocks_allocation_accessor]: type size and block size must align.";
      throw duckdb::InternalException(error_msg);
    }
  }
  void set_cursor(size_t byte_offset)
  {
    block_index     = byte_offset / allocation->block_size;
    offset_in_block = byte_offset % allocation->block_size;
  };
  void set_current(T value)
  {
    *reinterpret_cast<T*>(static_cast<uint8_t*>(allocation->blocks[block_index]) +
                          offset_in_block) = value;
  }
  T get_current() const
  {
    return *reinterpret_cast<T*>(static_cast<uint8_t*>(allocation->blocks[block_index]) +
                                 offset_in_block);
  }
  void advance()
  {
    offset_in_block += sizeof(underlying_type);
    if (offset_in_block == allocation->block_size) {
      ++block_index;
      offset_in_block = 0;
    }
  }
  void memcpy_from(void const* src, size_t bytes)
  {
    size_t bytes_copied = 0;
    // Loop over blocks into which to copy the src
    while (bytes_copied < bytes) {
      // Do as much of a bulk copy as possible in the current block
      auto const bytes_to_copy =
        std::min(bytes - bytes_copied, allocation->block_size - offset_in_block);
      std::memcpy(static_cast<uint8_t*>(allocation->blocks[block_index]) + offset_in_block,
                  static_cast<uint8_t const*>(src) + bytes_copied,
                  bytes_to_copy);
      bytes_copied += bytes_to_copy;
      offset_in_block += bytes_to_copy;
      // Check if we need to advance to the next block
      if (offset_in_block == allocation->block_size) {
        ++block_index;
        offset_in_block = 0;
      }
    }
  }
};

/**
 * @brief TODO
 *
 */
class duckdb_scan_task_local_state : public itask_local_state {
  static constexpr size_t DEFAULT_APPROXIMATE_BATCH_SIZE = 2ULL * 1024 * 1024 * 1024;  ///< 2 GB

  // The following VARCHAR size is selected to accomodate all CHAR/VARCHAR TPC-H columns.
  static constexpr size_t DEFAULT_VARCHAR_SIZE = 256ULL;  ///< 256 bytes

 public:
  using multiple_blocks_allocation =
    memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  struct column_builder {
    static constexpr size_t HOST_SPACE_INDEX =
      0;  ///< There is currently only one HOST memory space

    //===----------Fields----------===//
    duckdb::LogicalType type;  ///< DuckDB logical type of the column
    size_t type_size;  ///< Size of the column data type in bytes (only used for NON-VARCHAR)
    size_t total_data_bytes =
      0;  ///< Total number of data bytes written for this column (only needed for VARCHAR)

    // The memory reservations
    struct memory::any_memory_space_in_tier res_request =
      memory::any_memory_space_in_tier(memory::Tier::HOST);
    unique_ptr<memory::reservation> data_reservation;
    unique_ptr<memory::reservation> mask_reservation;
    unique_ptr<memory::reservation> offset_reservation = nullptr;

    // The memory allocations
    multiple_blocks_allocation_accessor<uint8_t> data_blocks_accessor;
    multiple_blocks_allocation_accessor<uint8_t> mask_blocks_accessor;
    multiple_blocks_allocation_accessor<int64_t> offset_blocks_accessor;

    // The memory resource
    unique_ptr<memory::fixed_size_host_memory_resource> allocator;

    column_builder() = default;
    column_builder(duckdb::LogicalType t);
    // no copying
    column_builder(const column_builder&)            = delete;
    column_builder& operator=(const column_builder&) = delete;
    // explicit moves (required if you declared dtor or copy ops)
    column_builder(column_builder&&) noexcept            = default;
    column_builder& operator=(column_builder&&) noexcept = default;
    ~column_builder();

    void reserve_memory(size_t estimated_num_rows);
    void allocate_memory();
    bool sufficient_space_for_column(duckdb::Vector& vec,
                                     duckdb::ValidityMask const& validity,
                                     size_t num_rows);
    void process_mask_for_column(duckdb::ValidityMask const& validity,
                                 size_t num_rows,
                                 size_t row_offset);
    void process_column(duckdb::Vector& vec,
                        duckdb::ValidityMask const& validity,
                        size_t num_rows,
                        size_t row_offset);
  };

  //===----------Constructor & Destructor----------===//
  duckdb_scan_task_local_state(duckdb_scan_task_global_state& g_state,
                               duckdb::ExecutionContext& exec_ctx,
                               size_t approximate_batch_size);

  //===----------Fields----------===//
  size_t approximate_batch_size;           ///< Approximate target batch size in bytes
  size_t num_columns;                      ///< Number of columns to be scanned
  size_t estimated_row_size;               ///< Estimated size of each row in bytes
  size_t estimated_rows_per_batch;         ///< Estimated number of rows per batch
  vector<column_builder> column_builders;  ///< Column builders for each column
  vector<size_t> varchar_indices;          ///< Indices of VARCHAR columns

  duckdb::DataChunk chunk;  ///< DataChunk buffer
  size_t row_offset = 0;    ///< Current row offset in buffers

  unique_ptr<duckdb::LocalTableFunctionState>
    local_tf_state;                    ///< Local state for the table function.
  duckdb::ExecutionContext& exec_ctx;  ///< The duckdb execution context, needed for initializings
                                       ///< the local table function state

 private:
  void initialize_local_table_function_state(duckdb::PhysicalTableScan const& op,
                                             duckdb::ExecutionContext& exec_ctx,
                                             duckdb::GlobalTableFunctionState* global_tf_state);
  void initialize_builders(const duckdb::PhysicalTableScan& op);
  void estimate_rows_per_batch();
  void initialize_buffers();
};

//===----------------------------------------------------------------------===//
// DuckDB Scan Task
//===----------------------------------------------------------------------===//
/**
 * @brief TODO
 */
class duckdb_scan_task : public itask {
 public:
  duckdb_scan_task(uint64_t task_id,
                   data_repository_manager& dr_mgr,
                   unique_ptr<duckdb_scan_task_local_state> l_state,
                   shared_ptr<duckdb_scan_task_global_state> g_state)
    : task_id(task_id), dr_mgr(dr_mgr), itask(std::move(l_state), g_state) {};

  void execute() override;

 private:
  //===----------Methods----------===//
  static bool get_next_chunk(duckdb_scan_task_local_state& l_state,
                             duckdb_scan_task_global_state& g_state);
  static bool chunk_fits(duckdb_scan_task_local_state& l_state);
  void process_chunk(duckdb_scan_task_local_state& l_state);

  //===----------Fields----------===//
  data_repository_manager& dr_mgr;  ///< Data repository manager to which to push batches
  uint64_t task_id;                 ///< The unique id of this scan task
};

}  // namespace sirius::parallel