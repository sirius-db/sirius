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
#include "duckdb/main/connection.hpp"

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
 * @brief The global state for a duckdb_scan_task.
 */
class duckdb_scan_task_global_state : public itask_global_state, public duckdb::GlobalSourceState {
 public:
  //===----------Constructor----------===//
  /**
   * @brief Construct a new duckdb_scan_task_global_state object
   *
   * @param[in] pipeline_id The pipeline id to which this scan task belongs
   * @param[in] scan_exec The scan executor with which to schedule new scan tasks
   * @param[in] client_ctx The DuckDB client context
   * @param[in] ptsa The physical table scan adapter being executed
   */
  duckdb_scan_task_global_state(uint64_t pipeline_id,
                                duckdb_scan_executor& scan_exec,
                                duckdb::ClientContext& client_ctx,
                                duckdb::physical_table_scan_adapter const& ptsa);

  //===----------Methods----------===//
  /**
   * @brief Get the maximum number of threads for executing a table scan (the number of threads in
   * the thread pool of the scan_executor)
   *
   * @return The maximum number of threads for the table scan
   */
  uint64_t MaxThreads() override { return max_threads; }

  /**
   * @brief Check if the table scan source is fully drained
   *
   * @return true if the source is fully drained, false otherwise
   */
  bool is_source_drained() const { return source_drained.load(std::memory_order_acquire); }

  /**
   * @brief Set the table scan source as fully drained
   */
  void set_source_drained() { source_drained.store(true, std::memory_order_release); }

  //===----------Fields----------===//
  std::atomic<bool> source_drained{false};  ///< Whether the table scan source is fully drained
  uint64_t pipeline_id;                     ///< The pipeline id to which this table scan belongs
  uint64_t max_threads;                     ///< Maximum number of threads for this scan task

  unique_ptr<duckdb::GlobalTableFunctionState>
    global_tf_state;                    ///< Global state for the table function
  unique_ptr<duckdb::LocalTableFunctionState>
    shared_local_tf_state;              ///< Shared local state for sequential scanning
  duckdb_scan_executor& scan_executor;  ///< The scan executor executing this scan task
  const duckdb::PhysicalTableScan& op;  ///< The physical table scan being executed
  
  std::mutex scan_mutex;  ///< Mutex to protect table function calls
};

//===----------------------------------------------------------------------===//
// Multiple Blocks Allocation Accessor
//===----------------------------------------------------------------------===//

/**
 * @brief Accessor for multiple blocks allocation from fixed_size_host_memory_resource.
 *
 * This accessor facilitates reading and writing data across multiple blocks
 * allocated by the fixed-size host memory resource. It manages the current
 * position within the allocation and provides methods to set/get values,
 * advance the cursor, and perform memcpy operations.
 * NOTE: the caller is responsible for allocating sufficient blocks and ensure the cursor does not
 * go out of bounds. Otherwise, behavior is undefined.
 *
 * @tparam T The underlying data type to be accessed.
 */
template <typename T>
struct multiple_blocks_allocation_accessor {
  using underlying_type = T;
  using multiple_blocks_allocation =
    memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  //===----------Fields----------===//
  size_t block_index     = 0;                         ///< The current block index
  size_t offset_in_block = 0;                         ///< The byte offset in the block
  unique_ptr<multiple_blocks_allocation> allocation;  ///< The multiple blocks allocation

  /**
   * @brief Initialize the accessor with a multiple blocks allocation.
   *
   * @param[in] alloc The multiple blocks allocation to initialize the accessor with.
   * @throws duckdb::InternalException if the underlying type size does not align with the block
   * size.
   */
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

  /**
   * @brief Set the cursor to a specific byte offset within the allocation.
   */
  void set_cursor(size_t byte_offset)
  {
    block_index     = byte_offset / allocation->block_size;
    offset_in_block = byte_offset % allocation->block_size;
  };

  /**
   * @brief Set value at the current position in the allocation.
   */
  void set_current(T value)
  {
    *reinterpret_cast<T*>(static_cast<uint8_t*>(allocation->blocks[block_index]) +
                          offset_in_block) = value;
  }

  /**
   * @brief Get the value at the current position in the allocation as a different type.
   *
   * @tparam S The type to which to cast the value.
   */
  template <typename S>
  S get_current_as() const
  {
    assert(block_index < allocation->blocks.size());
    return *reinterpret_cast<S*>(static_cast<uint8_t*>(allocation->blocks[block_index]) +
                                 offset_in_block);
  }

  /**
   * @brief Get the value at the current position in the allocation using the underlying type.
   */
  T get_current() const { return get_current_as<underlying_type>(); }

  /**
   * @brief Advance the cursor into the allocation to the next position as type S.
   *
   * @tparam S The type size to use for advancing the cursor.
   */
  template <typename S>
  void advance_as()
  {
    offset_in_block += sizeof(S);
    if (offset_in_block == allocation->block_size) {
      ++block_index;
      offset_in_block = 0;
    }
  }

  /**
   * @brief Advance the cursor into the allocation to the next position using the underlying type.
   */
  void advance() { advance_as<underlying_type>(); }

  /**
   * @brief Copy from a given source buffer into the allocation starting at the current position.
   *
   * @param[in] src Pointer to the source buffer.
   * @param[in] bytes Number of bytes to copy from the source buffer.
   */
  void memcpy_from(void const* src, size_t bytes)
  {
    size_t bytes_copied = 0;
    // Loop over blocks into which to copy the src
    while (bytes_copied < bytes) {
      assert(block_index < allocation->blocks.size());
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

  /**
   * @brief Copy the data from the allocation to a destination buffer.
   *
   * @param[in] dest Pointer to the destination buffer.
   * @param[in] bytes Number of bytes to copy to the destination buffer.
   */
  void memcpy_to(void* dest, size_t bytes)
  {
    size_t bytes_copied = 0;
    // Loop over blocks from which to copy the data
    while (bytes_copied < bytes) {
      assert(block_index < allocation->blocks.size());
      // Do as much of a bulk copy as possible in the current block
      auto const bytes_to_copy =
        std::min(bytes - bytes_copied, allocation->block_size - offset_in_block);
      std::memcpy(static_cast<uint8_t*>(dest) + bytes_copied,
                  static_cast<uint8_t*>(allocation->blocks[block_index]) + offset_in_block,
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

//===----------------------------------------------------------------------===//
// Scan Task Local State
//===----------------------------------------------------------------------===//

/**
 * @brief The local state for a duckdb scan task.
 *
 * This class manages the state specific to a single scan task, most importantly the memory buffers
 * into which to accumulate data from a DuckDB table scan and the logic for processing DuckDB data
 * chunks into those buffers.
 *
 */
class duckdb_scan_task_local_state : public itask_local_state {
  static constexpr size_t DEFAULT_APPROXIMATE_BATCH_SIZE = 2ULL * 1024 * 1024 * 1024;  ///< 2 GB

  // The following VARCHAR size is selected to accomodate all CHAR/VARCHAR TPC-H columns.
  static constexpr size_t DEFAULT_VARCHAR_SIZE = 256ULL;  ///< 256 bytes

 public:
  using multiple_blocks_allocation =
    memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  //===----------------------------------------------------------------------===//
  // Column Builder
  //===----------------------------------------------------------------------===//

  /**
   * @brief The column builder maintains the memory and logic for building a column from a duckdb
   * vector.
   */
  struct column_builder {
    static constexpr size_t HOST_SPACE_INDEX =
      0;                                       ///< There is currently only one HOST memory space
    static constexpr size_t FULL_MASK = 0xFF;  ///< Mask with all bits set

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
    memory::fixed_size_host_memory_resource* allocator;

    //===----------Constructors & Destructor----------===//
    column_builder() = default;

    /**
     * @brief Construct a new column_builder object with the given logical type.
     *
     * Uses the type to initialize type_size, which is only used for fixed-width columns.
     *
     * @param[in] t The DuckDB logical type of the column.
     */
    column_builder(duckdb::LogicalType t);

    // no copying
    column_builder(const column_builder&)            = delete;
    column_builder& operator=(const column_builder&) = delete;
    // explicit moves
    column_builder(column_builder&&) noexcept            = default;
    column_builder& operator=(column_builder&&) noexcept = default;

    /**
     * @brief Destructor for the column builder.
     *
     * Releases any memory reservations held by the column builder.
     */
    ~column_builder();

    //===----------Methods----------===//
    /**
     * @brief Reserve memory for the column based on the estimated number of rows.
     *
     * @param[in] estimated_num_rows The estimated number of rows to reserve memory for.
     */
    void reserve_memory(size_t estimated_num_rows);

    /**
     * @brief Allocate memory for the column based on the memory reservation.
     */
    void allocate_memory();

    /**
     * @brief Checks if there is enough space allocated to hold the data for the given vector.
     *
     * @param[in] vec The DuckDB vector containing the data to be processed.
     * @param[in] validity The validity mask indicating NULL values in the vector.
     * @param[in] num_rows The number of rows to be processed from the vector.
     */
    bool sufficient_space_for_column(duckdb::Vector& vec,
                                     duckdb::ValidityMask const& validity,
                                     size_t num_rows);

    /**
     * @brief Process the validity mask for the column and store it in the allocated mask buffer.
     *
     * @param[in] validity The validity mask indicating NULL values in the vector.
     * @param[in] num_rows The number of rows to be processed from the vector.
     * @param[in] row_offset The row offset within the buffers for the current scan task.
     */
    void process_mask_for_column(duckdb::ValidityMask const& validity,
                                 size_t num_rows,
                                 size_t row_offset);

    /**
     * @brief Process the given DuckDB vector and copy its data into the allocated buffers.
     *
     * @param[in] vec The DuckDB vector containing the data to be processed.
     * @param[in] validity The validity mask indicating NULL values in the vector.
     * @param[in] num_rows The number of rows to be processed from the vector.
     * @param[in] row_offset The row offset within the buffers for the current scan task.
     */
    void process_column(duckdb::Vector& vec,
                        duckdb::ValidityMask const& validity,
                        size_t num_rows,
                        size_t row_offset);
  };

  //===----------Constructor & Destructor----------===//
  /**
   * @brief Construct a new duckdb_scan_task_local_state object.
   *
   * @param[in] g_state The global state for the scan task.
   * @param[in] exec_ctx The DuckDB execution context.
   * @param[in] approximate_batch_size The approximate target batch size in bytes.
   * @param[in] default_varchar_size The default size for VARCHAR columns in bytes (used for row
   * size estimation)
   * @param[in] existing_local_tf_state Optional existing local table function state to reuse
   * (for continuing a scan across multiple tasks)
   */
  duckdb_scan_task_local_state(
    duckdb_scan_task_global_state& g_state,
    duckdb::ExecutionContext& exec_ctx,
    size_t approximate_batch_size                        = DEFAULT_APPROXIMATE_BATCH_SIZE,
    size_t default_varchar_size                          = DEFAULT_VARCHAR_SIZE,
    unique_ptr<duckdb::LocalTableFunctionState> existing_local_tf_state = nullptr);

  //===----------Fields----------===//
  size_t approximate_batch_size;           ///< Approximate target batch size in bytes
  size_t default_varchar_size;             ///< Default size for VARCHAR columns in bytes
  size_t num_columns;                      ///< Number of columns to be scanned
  size_t estimated_row_size = 0;           ///< Estimated size of each row in bytes
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
  /**
   * @brief Initializes the duckdb table function local state.
   *
   * @param[in] op The physical table scan operator being executed.
   * @param[in] exec_ctx The duckdb execution context.
   * @param[in] global_tf_state The duckdb table function global state.
   */
  void initialize_local_table_function_state(duckdb::PhysicalTableScan const& op,
                                             duckdb::ExecutionContext& exec_ctx,
                                             duckdb::GlobalTableFunctionState* global_tf_state);

  /**
   * @brief Initilizes the column builders. Does not reserve or allocate memory for the builders.
   *
   * @param[in] op The physical table scan operator being executed.
   */
  void initialize_builders(const duckdb::PhysicalTableScan& op);

  /**
   * @brief Estimate the maximum number of rows to process for a batch given the target batch size.
   *
   * Uses the actual width of fixed-width types, and a default VARCHAR width, for estimation.
   */
  void estimate_rows_per_batch();

  /**
   * @brief Initializes the column buffers in the colum_builders based on the estimated rows per
   * batch.
   */
  void initialize_buffers();
};

//===----------------------------------------------------------------------===//
// DuckDB Scan Task
//===----------------------------------------------------------------------===//

/**
 * @brief A scan task for the scan_executor using the DuckDB table function interface.
 *
 * The duckdb_scan_task represents a unit of work for scanning data from a DuckDB table function.
 * It accumulates approximately the target batch size specified in the local state before pushing
 * the batch to the data repository and notifying the task creater. If the table scan is
 * incomplete upon task completion, the task will push a new scan_task onto the task queue.
 */
class duckdb_scan_task : public itask {
  // Friend declaration for test access
  friend class test_scan_task;

 public:
  //===----------Constructor----------===//
  /**
   * @brief Construct a duckdb_scan_task object.
   *
   * @param[in] task_id The unique id of this scan task.
   * @param[in] dr_mgr The data repository manager to which to push batches.
   * @param[in] l_state The local state for this scan task.
   * @param[in] g_state The shared global state for this scan task.
   */
  duckdb_scan_task(uint64_t task_id,
                   data_repository_manager& dr_mgr,
                   unique_ptr<duckdb_scan_task_local_state> l_state,
                   shared_ptr<duckdb_scan_task_global_state> g_state)
    : task_id(task_id), dr_mgr(dr_mgr), itask(std::move(l_state), g_state) {};

  void execute() override;

  /// TODO: change protected to private when data can be tested against data in the data repository
 protected:
  //===----------Methods----------===//
  /**
   * @brief Fetches the next data chunk from the DuckDB table function into the local state's data
   * chunk.
   *
   * @param[in,out] l_state The local state of the scan task.
   * @param[in,out] g_state The global state of the scan task.
   * @return true if a new chunk was fetched, false if the source is drained.
   */
  static bool get_next_chunk(duckdb_scan_task_local_state& l_state,
                             duckdb_scan_task_global_state& g_state);

  /**
   * @brief Checks if the current data chunk fits in the allocated buffers of the column buiilders.
   *
   * @param[in,out] l_state The local state of the scan task.
   * @return true if the chunk fits, false otherwise.
   */
  static bool chunk_fits(duckdb_scan_task_local_state& l_state);

  /**
   * @brief Processes the current data chunk and copies its data into the column builders' buffers.
   */
  void process_chunk(duckdb_scan_task_local_state& l_state);

 public:
  /**
   * @brief Gets the global state of the scan task.
   *
   * @return A pointer to the global state.
   */
  duckdb_scan_task_global_state const* get_global_state() const
  {
    return &this->_global_state->cast<duckdb_scan_task_global_state>();
  }

  /**
   * @brief Gets the local state of the scan task.
   *
   * @return A pointer to the local state.
   */
  duckdb_scan_task_local_state const* get_local_state() const
  {
    return &this->_local_state->cast<duckdb_scan_task_local_state>();
  }

 protected:
  //===----------Fields----------===//
  data_repository_manager& dr_mgr;  ///< Data repository manager to which to push batches
  uint64_t task_id;                 ///< The unique id of this scan task
};

}  // namespace sirius::parallel