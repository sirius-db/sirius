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
#include <config.hpp>
#include <memory/multiple_blocks_allocation_accessor.hpp>
#include <op/sirius_physical_parquet_scan.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <pipeline/sirius_pipeline_itask.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>
#include <sirius_config.hpp>
#include <sirius_context.hpp>

// cucascade
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>

// duckdb
#include <duckdb/main/client_context.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>

// standard library
#include <atomic>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Parquet Scan Task Global State
//===----------------------------------------------------------------------===//
class parquet_scan_task_global_state : public pipeline::sirius_pipeline_task_global_state {
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

 public:
  /**
   * @brief Struct representing a range of row groups assigned to a scan task.
   */
  struct row_group_range {
    row_group_range(size_t file_idx,
                    size_t start_row_group_p,
                    size_t row_group_count_p,
                    size_t reserved_uncompressed_bytes_p,
                    size_t reserved_compressed_bytes_p)
      : file_idx(file_idx),
        start_row_group(start_row_group_p),
        row_group_count(row_group_count_p),
        reserved_uncompressed_bytes(reserved_uncompressed_bytes_p),
        reserved_compressed_bytes(reserved_compressed_bytes_p)
    {
    }

    size_t file_idx;
    size_t start_row_group;
    size_t row_group_count;
    size_t reserved_uncompressed_bytes;
    size_t reserved_compressed_bytes;
  };

  //===----------Constructor----------===//
  /**
   * @brief Construct the global state for the parquet scan task.
   *
   * @param[in] pipeline The pipeline associated with this task
   * @param[in] scan_op The physical table scan operator
   * @param[in] approximate_batch_size The target approximate batch size for the scan tasks
   */
  parquet_scan_task_global_state(
    duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
    sirius_physical_parquet_scan* scan_op,
    size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE);

  //===----------Methods----------===//
  /**
   * @brief Get the physical parquet scan operator associated with this global state.
   *
   * @return A reference to the physical parquet scan operator.
   */
  [[nodiscard]] sirius_physical_parquet_scan& get_operator() { return *_scan_op; }

  /**
   * @brief Get the file path of the Parquet file to scan.
   *
   * @param[in] file_idx The index of the file path to retrieve.
   * @return A const reference to the file path string.
   */
  [[nodiscard]] std::string const& get_file_path(size_t file_idx) const
  {
    return _file_paths[file_idx];
  }

  /**
   * @brief Get the Parquet reader options, e.g., projections, filters, etc.
   *
   * This is used for fetching byte ranges for the assigned row groups in the scan task local state,
   * and for constructing the hybrid scan reader in the scan task local state.
   *
   * @return A const reference to the Parquet reader options.
   */
  [[nodiscard]] cudf::io::parquet_reader_options const& get_options() const
  {
    return _reader_options;
  }

  /**
   * @brief Get the number of row group partitions required to exhaust the scan.
   *
   * This number equals the number of tasks that need to be scheduled to exhaust the scan.
   *
   * @return The number of row group partitions.
   */
  [[nodiscard]] size_t get_num_row_group_partitions() const { return _row_group_partitions.size(); }

  /**
   * @brief Atomically get the next row group partition index to be processed by a scan task.
   *
   * @return The next row group partition index.
   */
  [[nodiscard]] std::optional<size_t> get_next_rg_partition_idx()
  {
    auto const total = _row_group_partitions.size();
    size_t current   = _next_rg_partition.load(std::memory_order_relaxed);
    while (true) {
      if (current >= total) { return std::nullopt; }
      if (_next_rg_partition.compare_exchange_weak(
            current, current + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return current;
      }
    }
  }

  /**
   * @brief Check if there are remaining row group partitions.
   *
   * @return True if there are more partitions to process.
   */
  [[nodiscard]] bool has_more_partitions() const
  {
    return _next_rg_partition.load(std::memory_order_relaxed) < _row_group_partitions.size();
  }

  /**
   * @brief Get the row group range metadata associated with the given partition.
   *
   * @param[in] idx The row group partition index.
   * @return The row group range metadata associated with the given partition.
   */
  [[nodiscard]] row_group_range const& get_row_group_partition(size_t idx) const
  {
    return _row_group_partitions[idx];
  }

  /**
   * @brief Make a hybrid scan Parquet reader with the underlying reader options.
   *
   * Each task/data batch will need its own reader for concurrency reasons.
   *
   * @param[in] file_idx The file index of the parquet file to read.
   * @return A unique pointer to the hybrid scan Parquet reader.
   */
  [[nodiscard]] std::unique_ptr<hybrid_scan_reader> make_reader(size_t file_idx) const
  {
    return std::make_unique<hybrid_scan_reader>(_file_metadatas[file_idx], _reader_options);
  }

 private:
  /**
   * @brief Fill the vector of column indices for this scan after projection.
   */
  void make_selected_column_indices(sirius_physical_parquet_scan const& scan_op);

  /**
   * @brief Accumulate the compressed and uncompressed byte sizes for each row group in the file
   * metadata, which are needed for partitioning the row groups into scan tasks.
   */
  void accumulate_row_group_byte_sizes();

  /**
   * @brief Partition the row groups into scan tasks based on the accumulated byte sizes and the
   * target approximate batch size.
   */
  void partition_row_groups();

  //===----------Fields----------===//
  size_t _approximate_batch_size;          ///< Target approximate batch size for scan tasks
  sirius_physical_parquet_scan* _scan_op;  ///< The physical parquet scan operator being executed
  bool _is_projected;                      ///< Whether projection is applied

  std::vector<std::string> _file_paths;                          ///< The parquet file paths
  std::vector<cudf::io::parquet::FileMetaData> _file_metadatas;  ///< The parquet file metadata
  cudf::io::parquet_reader_options _reader_options;              ///< Parquet reader options

  std::vector<std::vector<size_t>>
    _row_group_uncompressed_bytes;  ///< Per-(file,row-group) uncompressed bytes
  std::vector<std::vector<size_t>>
    _row_group_compressed_bytes;                       ///< Per-(file,row-group) compressed bytes
  std::vector<row_group_range> _row_group_partitions;  ///< Row-group partitions for tasks
  std::vector<size_t> _selected_column_indices;        ///< Column indices to read (projection)

  std::atomic<size_t> _next_rg_partition{0};  ///< Number of local states created
};

//===----------------------------------------------------------------------===//
// Parquet Scan Task Local State
//===----------------------------------------------------------------------===//
/**
 * @brief Local state for parquet_scan_task, which manages the row group indices assigned to this
 * task and makes the memory allocation for the task.
 */
class parquet_scan_task_local_state : public pipeline::sirius_pipeline_task_local_state {
  using multiple_blocks_allocation =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
  using memory_space = cucascade::memory::memory_space;

 public:
  //===----------Constructor----------===//
  /**
   * @brief Construct the local state for the parquet scan task.
   *
   * @param[in] g_state The global state for the parquet scan task
   * @param[in] partition_idx The assigned row group partition index
   */
  parquet_scan_task_local_state(parquet_scan_task_global_state& g_state, size_t partition_idx);

  //===----------Methods----------===//
  /**
   * @brief Make a memory allocation for this local state corresponding to the compressed bytes
   * assigned to this local state.
   *
   * @return A unique pointer to the multiple blocks memory allocation.
   */
  std::unique_ptr<multiple_blocks_allocation> make_allocation();

  /**
   * @brief Get a pointer to the memory space associated with this local state's reservation.
   *
   * @return A pointer to the memory space.
   */
  memory_space* get_memory_space()
  {
    return const_cast<memory_space*>(&_reservation->get_memory_space());
  }

  /**
   * @brief Get the file index of the parquet file to read for this local state.
   *
   * @return The file index.
   */
  [[nodiscard]] size_t get_file_idx() const { return _file_idx; }

  /**
   * @brief Get the host span corresponding to the row group indices assigned to this local state.
   */
  [[nodiscard]] cudf::host_span<cudf::size_type const> get_rg_span() const
  {
    return cudf::host_span<cudf::size_type const>(_rg_indices.data(), _rg_indices.size());
  };

  /**
   * @brief Get the number of uncompressed bytes reserved by this local state.
   *
   * @return The number of uncompressed bytes reserved.
   */
  [[nodiscard]] size_t get_reserved_uncompressed_bytes() const
  {
    return _reserved_uncompressed_bytes;
  }

  /**
   * @brief Get the number of compressed bytes reserved by this local state.
   *
   * @return The number of compressed bytes reserved.
   */
  [[nodiscard]] size_t get_reserved_compressed_bytes() const { return _reserved_compressed_bytes; }

  /**
   * @brief Get the vector of row group indices assigned to this local state.
   *
   * @return A (const) reference to the vector of row group indices.
   */
  [[nodiscard]] std::vector<cudf::size_type> const& get_rg_indices() const { return _rg_indices; }
  [[nodiscard]] std::vector<cudf::size_type>& get_rg_indices() { return _rg_indices; }

 private:
  size_t _file_idx;  ///< The file index of the parquet file to read
  size_t _reserved_uncompressed_bytes =
    0;  ///< Number of uncompressed bytes reserved by the row group range
  size_t _reserved_compressed_bytes =
    0;  ///< Number of compressed bytes reserved by the row group range
  std::vector<cudf::size_type> _rg_indices;  ///< The row group indices assigned to this local state
};

//===----------------------------------------------------------------------===//
// Parquet Scan Task
//===----------------------------------------------------------------------===//
/**
 * @brief A scan task for reading compressed slices of parquet files into memory, which will then be
 * converted to table representations by representation converters.
 *
 * This scan task is similar to the byte-range preloader, as described in the Theseus paper:
 * https://arxiv.org/html/2508.05029v1#S3.SS4
 *
 */
class parquet_scan_task : public pipeline::sirius_pipeline_itask {
  using shared_data_repository = cucascade::shared_data_repository;
  using multiple_blocks_allocation =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
  using multiple_blocks_allocation_accessor = memory::multiple_blocks_allocation_accessor<uint8_t>;

 public:
  //===----------Constructor----------===//
  /**
   * @brief Construct a new parquet_scan_task object.
   *
   * @param[in] task_id The unique ID of this task.
   * @param[in] data_repo The shared data repository to which the produced data batch will be
   * pushed.
   * @param[in] l_state The local state for this task.
   * @param[in] g_state The global state for this task.
   */
  parquet_scan_task(uint64_t task_id,
                    shared_data_repository* data_repo,
                    std::unique_ptr<parquet_scan_task_local_state> l_state,
                    std::shared_ptr<parquet_scan_task_global_state> g_state)
    : pipeline::sirius_pipeline_itask(std::move(l_state), g_state),
      _task_id(task_id),
      _data_repo(data_repo)
  {
    auto& l_state_cast = this->_local_state->cast<parquet_scan_task_local_state>();
    _datasource = cudf::io::datasource::create(g_state->get_file_path(l_state_cast.get_file_idx()));
  }

  ~parquet_scan_task() override;

  //===----------Methods----------===//
  /**
   * @brief Compute the parquet scan task and produce a host_parquet_representation.
   *
   * This involves reading the assigned row groups and column chunks into a memory allocation, and
   * constructing the host_parquet_representation from the allocated memory.
   *
   * @param[in] stream The CUDA stream on which to perform memory operations.
   * @return A vector of shared pointers to data batches produced by this task.
   */
  std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view stream) override;

  /**
   * @brief Publish the output data batches produced by this task to the shared data repository.
   *
   * @param[in] output_batches The data batches produced by this task to be published.
   * @param[in] stream The CUDA stream on which to perform memory operations (ignored in this task).
   */
  void publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream) override;

  /**
   * @brief Get the estimated reservation size for this task, which is the number of compressed
   * bytes reserved by this task's local state.
   *
   * @return The estimated reservation size in bytes.
   */
  [[nodiscard]] size_t get_estimated_reservation_size() const override
  {
    auto& l_state = this->_local_state->cast<parquet_scan_task_local_state>();
    return l_state.get_reserved_compressed_bytes();
  }

  /**
   * @brief Get the output consumers operators for this task.
   *
   * @return A vector of pointers to the output consumer operators.
   */
  std::vector<op::sirius_physical_operator*> get_output_consumers() override
  {
    auto& g_state = this->_global_state->cast<parquet_scan_task_global_state>();
    std::vector<sirius_physical_operator*> output_consumers;
    auto ports = g_state.get_operator().get_next_port_after_sink();
    for (auto& [child, port_id] : ports) {
      output_consumers.push_back(child);
    }
    return output_consumers;
  }

  /**
   * @brief Get the unique ID of this task.
   *
   * @return The unique ID of this task.
   */
  [[nodiscard]] uint64_t get_task_id() const { return _task_id; }

  /**
   * @brief Set whether this task should operate on materialized (decoded) columns.
   *
   * @param materialized_columns True to use materialized columns, false otherwise.
   * @param gpu_memory_space     Pointer to the GPU memory space used for materialization.
   */
  void set_materialized_columns(bool wrap_in_cache,
                                bool materialized_columns,
                                cucascade::memory::memory_space* gpu_memory_space)
  {
    _wrap_in_cache        = wrap_in_cache;
    _materialized_columns = materialized_columns;
    _gpu_memory_space     = gpu_memory_space;
  }

 private:
  /**
   * @brief Read the given byte range from the parquet file into the memory allocation for this
   * task.
   */
  void read_range_into_allocation(size_t file_offset,
                                  size_t n_bytes,
                                  multiple_blocks_allocation_accessor& data_blocks_accessor,
                                  std::unique_ptr<multiple_blocks_allocation>& allocation,
                                  std::vector<std::future<std::size_t>>& read_futures);

  //===----------Fields----------===//
  uint64_t _task_id;                   ///< The unique ID of this task
  shared_data_repository* _data_repo;  ///< The shared data repository to which to push batches
  std::shared_ptr<cudf::io::datasource> _datasource;  ///< The cudf datasource for the input file
  bool _wrap_in_cache{false};
  bool _materialized_columns{false};  ///< Whether this task operates on materialized columns
  cucascade::memory::memory_space* _gpu_memory_space{
    nullptr};  ///< GPU memory space for materialization
};

}  // namespace sirius::op::scan
