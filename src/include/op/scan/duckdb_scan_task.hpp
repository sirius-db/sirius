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
#include <cudf/utilities/default_stream.hpp>

#include <config.hpp>
#include <memory/host_table_utils.hpp>
#include <memory/multiple_blocks_allocation_accessor.hpp>
#include <op/sirius_physical_duckdb_scan.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <parallel/task.hpp>
#include <pipeline/pipeline_executor.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <pipeline/sirius_pipeline_itask.hpp>
#include <pipeline/sirius_pipeline_itask_local_state.hpp>
#include <sirius_context.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

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

namespace sirius::op::scan {
//===----------------------------------------------------------------------===//
// DuckDB Scan Task Global State
//===----------------------------------------------------------------------===//

/**
 * @brief The global state for a duckdb_scan_task.
 */
class duckdb_scan_task_global_state : public sirius::parallel::itask_global_state,
                                      public duckdb::GlobalSourceState {
  friend class duckdb_scan_task;
  friend class duckdb_scan_task_local_state;

 public:
  //===----------Constructor----------===//
  /**
   * @brief Construct a new duckdb_scan_task_global_state object
   *
   * @param[in] pipeline The GPU pipeline to which this table scan belongs
   * @param[in] pipeline_exec The pipeline executor with which to schedule new scan tasks
   * @param[in] client_ctx The DuckDB client context
   * @param[in] gpu_pts The GPU physical table scan being executed
   */
  duckdb_scan_task_global_state(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                                pipeline::pipeline_executor& pipeline_exec,
                                duckdb::ClientContext& client_ctx,
                                sirius_physical_duckdb_scan* scan_op);

  //===----------Methods----------===//
  /**
   * @brief Get the maximum number of threads for executing a table scan (the number of threads in
   * the thread pool of the scan_executor)
   *
   * @return The maximum number of threads for the table scan
   */
  uint64_t MaxThreads() override { return _max_threads; }

  /**
   * @brief Check if the table scan source is fully drained
   *
   * @return true if the source is fully drained, false otherwise
   */
  bool is_source_drained() const { return _source_drained.load(std::memory_order_acquire); }

  /**
   * @brief Set the table scan source as fully drained
   */
  void set_source_drained()
  {
    _source_drained.store(true, std::memory_order_release);
    if (_pipeline) {
      auto* scan_op =
        dynamic_cast<sirius_physical_duckdb_scan*>(&_pipeline->get_operators().at(0).get());

      if (scan_op) { scan_op->exhausted.store(true, std::memory_order_release); }
    }
  }

  /**
   * @brief Increment the number of active local table function states
   *
   * We keep track of the number of active local states to determine when the table source is
   * fully drained. Only when all local states have exhausted their scan range is the table fully
   * read.
   */
  void increment_local_states() { _active_local_states.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Decrement the number of active local table function states
   *
   * See increment_local_states for more details.
   */
  void decrement_local_states()
  {
    auto const remaining = _active_local_states.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) { set_source_drained(); }
  }

  std::vector<sirius_physical_operator*> get_output_consumers() const noexcept
  {
    std::vector<sirius_physical_operator*> output_consumers;
    auto ports = _op.get_next_port_after_sink();
    for (auto& [child, port_id] : ports) {
      output_consumers.push_back(child);
    }
    return output_consumers;
  }

  [[nodiscard]] size_t get_pipeline_id() const { return _pipeline->get_pipeline_id(); }

 private:
  //===----------Fields----------===//
  duckdb::shared_ptr<pipeline::sirius_pipeline>
    _pipeline;                         ///< The pipeline to which this table scan belongs
  duckdb::SiriusContext* _sirius_ctx;  ///< The Sirius context
  std::unique_ptr<duckdb::GlobalTableFunctionState>
    _global_tf_state;  ///< Global state for the table function
  pipeline::pipeline_executor&
    _pipeline_executor;                      ///< The pipeline executor for scheduling scan tasks
  sirius_physical_duckdb_scan& _op;          ///< The physical table scan being executed
  std::atomic<bool> _source_drained{false};  ///< Whether the table scan source is fully drained
  std::atomic<int64_t> _active_local_states{0};  ///< Number of active local table function states
  uint64_t _max_threads;                         ///< Maximum number of threads for this scan task
};

//===----------------------------------------------------------------------===//
// Scan Task Local State
//===----------------------------------------------------------------------===//

/**
 * @brief The local state for a duckdb scan task.
 *
 * This class manages the state specific to a single scan task, most importantly the memory
 * buffers into which to accumulate data from a DuckDB table scan and the logic for processing
 * DuckDB data chunks into those buffers.
 *
 */
class duckdb_scan_task_local_state : public sirius::pipeline::sirius_pipeline_itask_local_state {
  using data_batch = cucascade::data_batch;

 public:
  friend class duckdb_scan_task;

  using multiple_blocks_allocation =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  //===----------------------------------------------------------------------===//
  // Column Builder
  //===----------------------------------------------------------------------===//

  /**
   * @brief The column builder maintains the memory and logic for building a column from a duckdb
   * vector.
   */
  struct column_builder {
    static constexpr uint8_t FULL_MASK = 0xFF;  ///< Byte mask with all bits set

    //===----------Fields----------===//
    duckdb::LogicalType type;  ///< DuckDB logical type of the column
    size_t type_size;  ///< Size of the column data type in bytes (for VARCHAR, just data size)
    size_t total_data_bytes =
      0;  ///< Total number of data bytes written for this column (only needed for VARCHAR)
    size_t total_data_bytes_allocated =
      0;  ///< Total number of data bytes allocated for this column (only needed for VARCHAR)
    size_t null_count = 0;  ///< Number of NULL values in the column

    // The allocation accessors for the column data, mask, and offsets
    memory::multiple_blocks_allocation_accessor<uint8_t> data_blocks_accessor;
    memory::multiple_blocks_allocation_accessor<uint8_t> mask_blocks_accessor;
    memory::multiple_blocks_allocation_accessor<int32_t> offset_blocks_accessor;

    //===----------Constructors & Destructor----------===//
    column_builder() = default;

    /**
     * @brief Construct a new column_builder object with the given logical type.
     *
     * Uses the type to initialize type_size, which is only used for fixed-width columns.
     *
     * @param[in] t The DuckDB logical type of the column.
     * @param[in] default_varchar_size The default size to use for VARCHAR data.
     */
    column_builder(duckdb::LogicalType t, size_t default_varchar_size);

    // no copying
    column_builder(const column_builder&)            = delete;
    column_builder& operator=(const column_builder&) = delete;
    // explicit moves
    column_builder(column_builder&&) noexcept            = default;
    column_builder& operator=(column_builder&&) noexcept = default;
    ~column_builder()                                    = default;

    //===----------Methods----------===//
    /**
     * @brief Initialize the allocation accessors for the column.
     *
     * @param[in] estimated_num_rows The estimated number of rows for the scan task data batch.
     * @param[in] byte_offset The byte offset within the overall allocation where this column's
     * data starts.
     * @param[in,out] allocation The allocation into which to write data for this column.
     */
    void initialize_accessors(size_t estimated_num_rows,
                              size_t byte_offset,
                              std::unique_ptr<multiple_blocks_allocation>& allocation);

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
     * @brief Process the validity mask for the column.
     *
     * @param[in] validity The validity mask indicating NULL values in the vector.
     * @param[in] num_rows The number of rows to be processed from the vector.
     * @param[in] row_offset The row offset within the allocation for the current scan task.
     * @param[in] allocation The allocation into which to write data for this column.
     */
    void process_mask_for_column(duckdb::ValidityMask const& validity,
                                 size_t num_rows,
                                 size_t row_offset,
                                 std::unique_ptr<multiple_blocks_allocation>& allocation);

    /**
     * @brief Process the given DuckDB vector and copy its data into the allocation.
     *
     * @param[in] vec The DuckDB vector containing the data to be processed.
     * @param[in] validity The validity mask indicating NULL values in the vector.
     * @param[in] num_rows The number of rows to be processed from the vector.
     * @param[in] row_offset The row offset within the allocation for the current scan task.
     * @param[in] allocation The allocation into which to write data for this column.
     */
    void process_column(duckdb::Vector& vec,
                        duckdb::ValidityMask const& validity,
                        size_t num_rows,
                        size_t row_offset,
                        std::unique_ptr<multiple_blocks_allocation>& allocation);

    /**
     * @brief Create a metadata node for this column for building a host_table_allocation.
     *
     * @param[in] num_rows The number of rows in the column.
     * @return metadata_node The constructed metadata node.
     */
    [[nodiscard]] metadata_node make_metadata_node(size_t num_rows) const;
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
    size_t approximate_batch_size = duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE,
    size_t default_varchar_size   = duckdb::Config::DEFAULT_SCAN_TASK_VARCHAR_SIZE,
    std::unique_ptr<duckdb::LocalTableFunctionState> existing_local_tf_state = nullptr);

  [[nodiscard]] std::size_t get_estimated_reservation_size() const noexcept
  {
    return _approximate_batch_size;
  }

  //===----------Methods----------===//
  /**
   * @brief Creates a data batch from the current state of the column builders.
   *
   * @return A shared pointer to the created data batch.
   */
  std::shared_ptr<data_batch> make_data_batch();

 private:
  //===----------Fields----------===//
  size_t _approximate_batch_size;                ///< Approximate target batch size in bytes
  size_t _default_varchar_size;                  ///< Default size for VARCHAR columns in bytes
  size_t _num_columns;                           ///< Number of columns to be scanned
  size_t _estimated_rows_per_batch;              ///< Estimated number of rows per batch
  std::vector<column_builder> _column_builders;  ///< Column builders for each column
  std::vector<size_t> _varchar_indices;          ///< Indices of VARCHAR columns

  cucascade::memory::any_memory_space_in_tier _res_request =
    cucascade::memory::any_memory_space_in_tier(cucascade::memory::Tier::HOST);
  std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation>
    _allocation;  ///< Memory allocation for all column data
  cucascade::memory::memory_space* _host_space = nullptr;

  duckdb::DataChunk _chunk;           ///< DataChunk buffer
  size_t _row_offset        = 0;      ///< Current row offset in buffers
  bool _local_state_drained = false;  ///< Whether this local state has fully drained

  std::unique_ptr<duckdb::LocalTableFunctionState>
    _local_tf_state;                    ///< Local state for the table function.
  duckdb::ExecutionContext& _exec_ctx;  ///< The duckdb execution context, needed for initializing
                                        ///< the local table function state

  /**
   * @brief Get the byte offset within the allocation where the column data ends.
   *
   * @return The byte offset within the allocation where the column data ends.
   */
  [[nodiscard]] size_t get_tail_byte_offset() const;

  /**
   * @brief Estimate the maximum number of rows to process for a batch given the target batch
   * size.
   *
   * Uses the actual width of fixed-width types, and a default VARCHAR width, for estimation.
   *
   * @param[in] op The physical table scan operator being executed.
   */
  void estimate_rows_per_batch(sirius_physical_duckdb_scan const& op);

  /**
   * @brief Initializes the column builders.
   */
  void initialize_builders();

  /**
   * @brief Initializes the duckdb table function local state.
   *
   * @param[in] op The physical table scan operator being executed.
   * @param[in] exec_ctx The duckdb execution context.
   * @param[in] global_tf_state The duckdb table function global state.
   */
  void initialize_local_table_function_state(sirius_physical_duckdb_scan const& op,
                                             duckdb::ExecutionContext& exec_ctx,
                                             duckdb::GlobalTableFunctionState* global_tf_state);
};

//===----------------------------------------------------------------------===//
// DuckDB Scan Task
//===----------------------------------------------------------------------===//

/**
 * @brief A scan task for the scan_executor using the DuckDB table function interface.
 *
 * The duckdb_scan_task represents a unit of work for scanning data from a DuckDB table function.
 * It accumulates approximately the target batch size specified in the local state before pushing
 * the batch to the data repository and notifying the task creator. If the table scan is
 * incomplete upon task completion, the task will push a new scan_task onto the task queue.
 */
class duckdb_scan_task : public sirius::pipeline::sirius_pipeline_itask {
  using shared_data_repository = cucascade::shared_data_repository;
  // Friend declaration for test access
  friend class test_scan_task;

 public:
  //===----------Constructor----------===//
  /**
   * @brief Construct a duckdb_scan_task object.
   *
   * @param[in] task_id The unique id of this scan task.
   * @param[in] data_repo The data repository to which to push batches.
   * @param[in] l_state The local state for this scan task.
   * @param[in] g_state The shared global state for this scan task.
   */
  duckdb_scan_task(uint64_t task_id,
                   shared_data_repository* data_repo,
                   std::unique_ptr<duckdb_scan_task_local_state> l_state,
                   std::shared_ptr<duckdb_scan_task_global_state> g_state)
    : sirius::pipeline::sirius_pipeline_itask(std::move(l_state), g_state),
      _task_id(task_id),
      _data_repo(data_repo) {};

  //===----------Destructor----------===//
  ~duckdb_scan_task();

  void execute(rmm::cuda_stream_view stream) override;

 private:
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
   * @brief Checks if the current data chunk fits in the allocated buffers of the column builders.
   *
   * @param[in,out] l_state The local state of the scan task.
   * @return true if the chunk fits, false otherwise.
   */
  static bool chunk_fits(duckdb_scan_task_local_state& l_state);

  /**
   * @brief Processes the current data chunk and copies its data into the column builders'
   * buffers.
   */
  void process_chunk(duckdb_scan_task_local_state& l_state);

 public:
  /**
   * @brief Gets the global state of the scan task.
   *
   * @return A pointer to the global state.
   */
  [[nodiscard]] duckdb_scan_task_global_state const* get_global_state() const
  {
    return &this->_global_state->cast<duckdb_scan_task_global_state>();
  }

  /**
   * @brief Gets the local state of the scan task.
   *
   * @return A pointer to the local state.
   */
  [[nodiscard]] duckdb_scan_task_local_state const* get_local_state() const
  {
    return &this->_local_state->cast<duckdb_scan_task_local_state>();
  }

  /**
   * @brief Compute and return the output data batches for this task.
   *
   * Scans data from the DuckDB table function and accumulates it into data batches.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return std::vector<std::shared_ptr<cucascade::data_batch>> The computed output batches
   */
  std::unique_ptr<op::operator_data> compute_task(
    [[maybe_unused]] rmm::cuda_stream_view stream) override;

  /**
   * @brief Publish the computed output batches to the data repository.
   *
   * Pushes the output batches to the configured data repository and schedules
   * new scan tasks if the table scan is incomplete.
   *
   * @param output_batches The data batches to publish
   */
  void publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream) override;

  std::size_t get_estimated_reservation_size() const override
  {
    return this->_local_state->cast<duckdb_scan_task_local_state>()
      .get_estimated_reservation_size();
  }

  /// @brief Get the output consumer operators for this task.
  std::vector<op::sirius_physical_operator*> get_output_consumers() override
  {
    return this->_global_state->cast<duckdb_scan_task_global_state>().get_output_consumers();
  }

  [[nodiscard]] size_t get_pipeline_id() const
  {
    return this->_global_state->cast<duckdb_scan_task_global_state>().get_pipeline_id();
  }

 private:
  //===----------Fields----------===//
  shared_data_repository* _data_repo;  ///< Data repository to which to push batches
  uint64_t _task_id;                   ///< The unique id of this scan task
};

}  // namespace sirius::op::scan
