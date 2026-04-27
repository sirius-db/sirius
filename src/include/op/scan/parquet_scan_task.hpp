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
#include <data/host_parquet_representation.hpp>
#include <memory/multiple_blocks_allocation_accessor.hpp>
#include <op/scan/hive_partition.hpp>
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
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/main/client_context.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// sirius scan operator data
#include <op/scan/parquet_scan_operator_data.hpp>

// standard library
#include <atomic>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Utility (shared with iceberg_scan_task)
//===----------------------------------------------------------------------===//
namespace detail {
/**
 * @brief Compute the parquet column indices to read given column and projection id vectors.
 *
 * Applies projection_ids / column_ids to select only the needed columns.
 * Virtual columns, duplicates, and hive partition columns are excluded.
 * Defined in parquet_scan_task.cpp.
 *
 * Note: the returned indices are DuckDB primary indices. Hive partition columns
 * (present in the DuckDB schema but not in the parquet files) are NOT removed
 * here — callers must filter them post-hoc using the bind data's
 * `hive_partitioning_indexes`.
 *
 * @param column_ids     All column ids exposed by the table function.
 * @param projection_ids Subset of column_ids positions selected by the planner (empty = no
 *                       projection).
 */
std::vector<size_t> make_selected_column_indices(
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids);

/**
 * @brief Return true if all projected leaf columns have a flat (depth-1) schema.
 *
 * Looks up each projected column by name in the parquet file's schema and
 * checks that its `path_in_schema` has length 1 (i.e., not nested).
 * Defined in parquet_scan_task.cpp.
 */
bool projected_columns_are_flat(cudf::io::parquet::FileMetaData const& meta,
                                std::vector<std::string> const& projected_column_names);
}  // namespace detail

//===----------------------------------------------------------------------===//
// Post-Convert Hook Type
//===----------------------------------------------------------------------===//
// Defined in host_parquet_representation.hpp (sirius::post_convert_fn_t).
// Re-exported here so that callers in namespace sirius::op::scan can use it
// without a qualification.
using sirius::post_convert_fn_t;

//===----------------------------------------------------------------------===//
// Parquet Scan Task Global State
//===----------------------------------------------------------------------===//
class parquet_scan_task_global_state : public pipeline::sirius_pipeline_task_global_state {
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

 public:
  /// Row-group range type shared with the new metadata/GPU scan operators.
  using row_group_range = ::sirius::op::scan::row_group_range;

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
    std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE);

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
  [[nodiscard]] std::string const& get_file_path(std::size_t file_idx) const
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
  [[nodiscard]] std::size_t get_num_row_group_partitions() const
  {
    return _row_group_partitions.size();
  }

  /**
   * @brief Atomically claim and move out the next row group partition index to be processed by a
   * scan task.
   *
   * @return The next row group partition, moved out of global state; std::nullopt if exhausted.
   */
  [[nodiscard]] std::optional<row_group_range> claim_next_rg_partition()
  {
    auto const total    = _row_group_partitions.size();
    std::size_t current = _next_rg_partition.load(std::memory_order_relaxed);
    while (true) {
      if (current >= total) { return std::nullopt; }
      if (_next_rg_partition.compare_exchange_weak(
            current, current + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return std::move(_row_group_partitions[current]);
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
   * @brief Make a hybrid scan Parquet reader with the underlying reader options.
   *
   * Each task/data batch will need its own reader for concurrency reasons.
   *
   * @param[in] file_idx The file index of the parquet file to read.
   * @return A unique pointer to the hybrid scan Parquet reader.
   */
  [[nodiscard]] std::unique_ptr<hybrid_scan_reader> make_reader(std::size_t file_idx) const
  {
    auto opts = _reader_options;
    // Per-file column projection for schema evolution: each file may have
    // a different subset of the requested columns.
    if (!_per_file_column_names.empty() && file_idx < _per_file_column_names.size()) {
#if CUDF_VERSION_NUM >= 2604
      opts.set_column_names(_per_file_column_names[file_idx]);
#else
      opts.set_columns(_per_file_column_names[file_idx]);
#endif
    }
    return std::make_unique<hybrid_scan_reader>(_file_metadatas[file_idx], opts);
  }

  /**
   * @brief Rebind this global state to a new pipeline and scan operator.
   *
   * Reuses all cached file metadata and row group partitions, avoiding
   * footer reads and metadata parsing.  Only the pipeline/operator pointers
   * and the partition counter are updated.
   */
  void rebind(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
              sirius_physical_parquet_scan* scan_op)
  {
    set_pipeline(std::move(pipeline));
    _scan_op = scan_op;
    _next_rg_partition.store(0, std::memory_order_relaxed);
    scan_op->has_more_partitions.store(true, std::memory_order_relaxed);
    scan_op->exhausted.store(false, std::memory_order_relaxed);
  }

  /**
   * @brief Get the file size for the given file index.
   */
  [[nodiscard]] std::size_t get_file_size(std::size_t file_idx) const
  {
    return _file_sizes[file_idx];
  }

  /**
   * @brief Get the total number of parquet metadata bytes (header + footer + trailer)
   * that must be cached alongside the column-chunk data for file @p file_idx.
   */
  [[nodiscard]] std::size_t get_metadata_byte_size(std::size_t file_idx) const
  {
    return _metadata_byte_sizes[file_idx];
  }

  /**
   * @brief Get the file offset where the parquet footer begins for file @p file_idx.
   * The footer range covers [footer_offset, file_size).
   */
  [[nodiscard]] std::size_t get_footer_offset(std::size_t file_idx) const
  {
    return _footer_offsets[file_idx];
  }

  /** @brief Get a shared_ptr that pins the translated AST filter expression alive.
   *
   * This is passed to host_parquet_representation so the filter expression (which
   * parquet_reader_options stores as a reference) survives until materialization.
   *
   * @return A shared_ptr to the translated filter expression (may be null if no filter). */
  [[nodiscard]] std::shared_ptr<gpu_expression_translator::translated_expression>
  get_filter_expression() const
  {
    return _translated_filter;
  }

  /**
   * @brief Get projection ids of projected columns post-filter.
   *
   * @return A const reference to the vector of projection ids.
   */
  [[nodiscard]] std::vector<std::size_t> const& get_post_filter_projection_ids() const
  {
    return _post_filter_projection_ids;
  }

  // -------------------------------------------------------------------------
  // Post-convert hook (used by iceberg scan for delete application)
  // -------------------------------------------------------------------------

  /**
   * @brief Install a post-convert hook that fires after each row-group batch
   * is decompressed to a GPU table by the host->GPU converter.
   *
   * The hook receives the freshly produced cudf::table and must return a
   * (possibly filtered) replacement table. The iceberg scan path uses this to
   * apply V2 positional and equality deletes without any pipeline operator.
   */
  void set_post_convert_fn(post_convert_fn_t fn) { _post_convert_fn = std::move(fn); }

  [[nodiscard]] bool has_post_convert_fn() const { return _post_convert_fn != nullptr; }

  /**
   * @brief Return a copy of the installed post-convert hook.
   *
   * Called by compute_task to attach the hook to each host_parquet_representation
   * so the converter can invoke it when decompressing that specific batch.
   */
  [[nodiscard]] post_convert_fn_t get_post_convert_fn() const { return _post_convert_fn; }

  /**
   * @brief Return the selected column indices (DuckDB primary indices of data-only
   * columns, in cudf table order — hive partition columns excluded).
   *
   * Used by iceberg_scan_task_global_state to compute data_key_indices that correctly map
   * equality-delete key names to cudf table column positions.
   */
  [[nodiscard]] std::vector<size_t> const& get_selected_column_indices() const
  {
    return _selected_column_indices;
  }

  /**
   * @brief Return the parquet FileMetaData for the given data file index.
   *
   * Used by iceberg_scan_task_global_state to extract Iceberg field IDs
   * from data file schemas for schema-evolution-safe column matching.
   */
  [[nodiscard]] cudf::io::parquet::FileMetaData const& get_file_metadata(size_t file_idx) const
  {
    return _file_metadatas[file_idx];
  }

  /// Return the number of data files.
  [[nodiscard]] size_t num_files() const { return _file_paths.size(); }

  // -------------------------------------------------------------------------
  // Hive partition column support
  // -------------------------------------------------------------------------

  /// True if this scan involves hive-partitioned files.
  [[nodiscard]] bool has_hive_partitions() const { return !_hive_partition_columns.empty(); }

  /// True if a partition_inject_fn was installed (hive partitioning OR schema
  /// reconciliation). Schema evolution can install one even when there are no
  /// hive partition columns, so callers that propagate the fn down to
  /// host_parquet_representation should gate on this rather than
  /// has_hive_partitions().
  [[nodiscard]] bool has_partition_inject_fn() const
  {
    return static_cast<bool>(_partition_inject_fn);
  }

  /// Return the partition injection function (may be null).
  [[nodiscard]] partition_inject_fn_t const& get_partition_inject_fn() const
  {
    return _partition_inject_fn;
  }

  /// Return the hive partition column metadata.
  [[nodiscard]] std::vector<hive_partition_column> const& get_hive_partition_columns() const
  {
    return _hive_partition_columns;
  }

  /// Return the set of DuckDB column indices that are hive partitions.
  [[nodiscard]] std::unordered_set<size_t> const& get_hive_partition_index_set() const
  {
    return _hive_partition_index_set;
  }

  /**
   * @brief Initialize hive partition metadata and build the injection function.
   *
   * Called from both the public constructor (plain parquet) and the iceberg
   * subclass constructor. Safe to call after base construction.
   */
  void init_hive_partitions(duckdb::MultiFileBindData const& bind_data,
                            sirius_physical_parquet_scan* scan_op);

  /**
   * @brief Detect missing/schema-evolution columns and set up per-file projection.
   *
   * Scans all file metadatas to find columns missing from ALL files (→ partition
   * columns) vs missing from SOME files (→ schema evolution, per-file projection).
   * Also applies column-name projection to reader options.
   *
   * Called from both constructors after file metadata is available.
   */
  void detect_columns_and_apply_projection(sirius_physical_parquet_scan* scan_op);

  /**
   * @brief Build a schema reconciliation function for schema evolution.
   *
   * When files have different column sets (e.g., column added in later snapshot),
   * this builds a partition_inject_fn that injects typed NULL columns for columns
   * missing from specific files, producing a consistent output schema.
   *
   * Must be called AFTER init_hive_partitions (chains with any existing inject fn).
   */
  void build_schema_reconciliation(sirius_physical_parquet_scan* scan_op);

 protected:
  /**
   * @brief Protected constructor for subclasses that pre-process the file list.
   *
   * Skips the MultiFileBindData extraction step; the caller supplies already-
   * resolved @p file_paths and @p selected_column_indices directly. Everything
   * else (footer reads, metadata parsing, row-group partitioning) is identical
   * to the public constructor.
   *
   * Used by iceberg_scan_task_global_state, which separates data files from
   * delete files before constructing the base state.
   *
   * @param pipeline                The pipeline for this scan.
   * @param scan_op                 The physical scan operator (provides column
   *                                names for projection, exhausted flag, etc.).
   * @param file_paths              Pre-filtered list of DATA-file paths only.
   * @param selected_column_indices Column indices to read (may be widened for
   *                                equality-delete key columns).
   * @param approximate_batch_size  Target uncompressed batch size for partitioning.
   */
  parquet_scan_task_global_state(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                                 sirius_physical_parquet_scan* scan_op,
                                 std::vector<std::string> file_paths,
                                 std::vector<size_t> selected_column_indices,
                                 std::size_t approximate_batch_size);

 private:
  /**
   * @brief Shared initialization: read footers, apply projections/filters, parse
   * metadata, detect hive partitions from the parquet schema, and partition row
   * groups. Called by both constructors after _file_paths and
   * _selected_column_indices have been populated.
   */
  void initialize_from_files();

  //===----------Fields----------===//
  std::size_t _approximate_batch_size;     ///< Target approximate batch size for scan tasks
  sirius_physical_parquet_scan* _scan_op;  ///< The physical parquet scan operator being executed

  /// DuckDB primary indices of data-only columns to read, in cudf table order.
  /// Hive partition columns are excluded.
  std::vector<size_t> _selected_column_indices;

  std::vector<std::string> _file_paths;                          ///< The parquet file paths
  std::vector<cudf::io::parquet::FileMetaData> _file_metadatas;  ///< The parquet file metadata
  cudf::io::parquet_reader_options _reader_options;              ///< Parquet reader options

  std::vector<std::size_t> _file_sizes;           ///< Per-file total file size in bytes
  std::vector<std::size_t> _metadata_byte_sizes;  ///< Per-file header+footer+trailer bytes
  std::vector<std::size_t> _footer_offsets;       ///< Per-file offset where footer begins

  std::shared_ptr<gpu_expression_translator::translated_expression>
    _translated_filter;  ///< The translated filter expression, if any, to keep alive for
                         ///< materialization
  std::vector<std::size_t>
    _post_filter_projection_ids;  ///< The indices of projected columns in the reader output

  std::vector<row_group_range>
    _row_group_partitions;  ///< The row group partitions for this scan (1 per task)
  std::atomic<std::size_t> _next_rg_partition{0};  ///< Number of local states created

  /// Optional hook called after each batch is decompressed to a GPU table.
  /// Null for plain parquet scans; set by iceberg_scan_task_global_state.
  post_convert_fn_t _post_convert_fn;

  /// Optional partition column injection (null unless hive-partitioned).
  partition_inject_fn_t _partition_inject_fn;

  /// Per-file column names for cuDF projection (schema evolution support).
  /// Empty if all files share the same schema; otherwise one entry per file.
  std::vector<std::vector<std::string>> _per_file_column_names;

  /// Hive partition columns (not present in parquet files).
  std::vector<hive_partition_column> _hive_partition_columns;
  std::unordered_set<size_t> _hive_partition_index_set;
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
   * @param[in] partition The assigned row group partition for this local state
   */
  parquet_scan_task_local_state(parquet_scan_task_global_state const& g_state,
                                parquet_scan_task_global_state::row_group_range partition)
    : _partition(std::move(partition)),
      _metadata_bytes(g_state.get_metadata_byte_size(_partition.file_idx))
  {
  }

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
  [[nodiscard]] std::size_t get_file_idx() const { return _partition.file_idx; }

  /**
   * @brief Get the host span corresponding to the row group indices assigned to this local state.
   */
  [[nodiscard]] cudf::host_span<cudf::size_type const> get_rg_span() const
  {
    return cudf::host_span<cudf::size_type const>(_partition.row_group_indices.data(),
                                                  _partition.row_group_indices.size());
  };

  /**
   * @brief Get the number of uncompressed bytes reserved by this local state.
   *
   * @return The number of uncompressed bytes reserved.
   */
  [[nodiscard]] std::size_t get_reserved_uncompressed_bytes() const
  {
    return _partition.reserved_uncompressed_bytes;
  }

  /**
   * @brief Get the number of compressed bytes reserved by this local state.
   *
   * @return The number of compressed bytes reserved.
   */
  [[nodiscard]] std::size_t get_reserved_compressed_bytes() const
  {
    return _partition.reserved_compressed_bytes + _metadata_bytes;
  }

  [[nodiscard]] std::size_t get_task_consumption_basis() const override
  {
    return get_reserved_compressed_bytes();
  }

  /**
   * @brief Get the vector of row group indices assigned to this local state.
   *
   * @return A reference to the vector of row group indices.
   */
  [[nodiscard]] std::vector<cudf::size_type>& get_rg_indices()
  {
    return _partition.row_group_indices;
  }

 private:
  parquet_scan_task_global_state::row_group_range _partition;  ///< Assigned row-group partition
  std::size_t _metadata_bytes;                                 ///< The number of metadata bytes
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
  }

  ~parquet_scan_task() override;

  //===----------Methods----------===//

  /**
   * @brief Execute the parquet scan task and record memory metrics.
   */
  void execute(rmm::cuda_stream_view stream) override;

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
  [[nodiscard]] std::size_t get_estimated_reservation_size() const override;

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
    for (auto& next_port : ports) {
      output_consumers.push_back(next_port.next_operator);
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
  void read_range_into_allocation(std::size_t file_offset,
                                  std::size_t n_bytes,
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
