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
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <op/scan/hive_partition.hpp>  // For partition_inject_fn_t

// cucascade
#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <functional>
#include <memory>
#include <vector>

namespace sirius {

/**
 * @brief Function called after a parquet batch is decompressed to a GPU cuDF table.
 *
 * Used by the iceberg scan path to apply V2 positional and equality deletes on the
 * GPU without adding any pipeline operator. The hook receives the freshly produced
 * cudf::table, the data-file path that produced this batch, the 0-based absolute
 * row offset of the first row in the batch within the data file, and the CUDA stream.
 * It must return a (possibly filtered) replacement table.
 *
 * - data_file_path  : identifies which Iceberg data file this batch came from
 *                     (used to look up matching positional-delete records).
 * - first_row_offset: absolute row index of the first row in this batch within
 *                     the data file.  Needed to translate Iceberg V2 positional
 *                     delete `pos` values into batch-relative indices.
 */
using post_convert_fn_t =
  std::function<std::unique_ptr<cudf::table>(std::unique_ptr<cudf::table>,
                                             std::string const& data_file_path,
                                             int64_t first_row_offset,
                                             rmm::cuda_stream_view)>;

// partition_inject_fn_t is the legacy typedef declared in <op/scan/hive_partition.hpp>
// (sirius:: namespace). It carries file_path so the schema-reconciliation closure built by
// parquet_scan_task_global_state::build_schema_reconciliation can do per-file column-set lookups.

/**
 * @brief A host representation of Parquet data for use in a hybrid scan.
 *
 * This class encapsulates the necessary components to decompress a slice and/or projection of
 * Parquet data using cudf's hybrid scan capabilities.
 * See:
 * https://docs.rapids.ai/api/libcudf/stable/classcudf_1_1io_1_1parquet_1_1experimental_1_1hybrid__scan__reader
 * The APIs for this reader are still marked experimental and are likely volatile.
 */
class host_parquet_representation : public cucascade::idata_representation {
  using hybrid_scan_reader    = cudf::io::parquet::experimental::hybrid_scan_reader;
  using translated_expression = gpu_expression_translator::translated_expression;

 public:
  /**
   * @brief Constructs a host_parquet_representation.
   *
   * @param[in] memory_space The memory space to which the representation beloongs.
   * @param[in] column_chunks The fixed multiple blocks allocation containing the Parquet column
   * chunks.
   * @param[in] parquet_reader An instance hybrid scan Parquet reader for a given Parquet file.
   * @param[in] reader_options The Parquet reader options used to configure the hybrid scan reader
   * for materializing data.
   * @param[in] row_group_indices The row group indices of the row groups represented in the
   * multiple blocks allocation.
   * @param[in] column_chunk_byte_ranges The byte ranges in the multiple blocks allocation
   * representing the column chunks to be read.
   * @param[in] size_in_bytes The size of the representation in bytes (compressed).
   * @param[in] uncompressed_size_in_bytes The uncompressed size of the data represented by this
   * representation.
   * @param[in] file_size The original parquet file size in bytes.
   * @param[in] fallback_datasource An optional fallback datasource for uncached byte ranges.
   * @param[in]
   */
  host_parquet_representation(cucascade::memory::memory_space* memory_space,
                              cucascade::memory::fixed_multiple_blocks_allocation column_chunks,
                              std::shared_ptr<hybrid_scan_reader> parquet_reader,
                              cudf::io::parquet_reader_options reader_options,
                              std::vector<cudf::size_type> row_group_indices,
                              std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges,
                              std::size_t size_in_bytes,
                              std::size_t uncompressed_size_in_bytes,
                              std::size_t file_size,
                              std::shared_ptr<cudf::io::datasource> fallback_datasource,
                              std::shared_ptr<std::unordered_map<int, translated_expression>>
                                filter_expression_by_device                       = nullptr,
                              std::vector<std::size_t> post_filter_projection_ids = {})
    : idata_representation(*memory_space),
      _column_chunks(std::move(column_chunks)),
      _parquet_reader(std::move(parquet_reader)),
      _reader_options(std::move(reader_options)),
      _row_group_indices(std::move(row_group_indices)),
      _column_chunk_byte_ranges(std::move(column_chunk_byte_ranges)),
      _size_in_bytes(size_in_bytes),
      _uncompressed_size_in_bytes(uncompressed_size_in_bytes),
      _file_size(file_size),
      _fallback_datasource(fallback_datasource),
      _filter_expression_by_device(std::move(filter_expression_by_device)),
      _post_filter_projection_ids(std::move(post_filter_projection_ids))
  {
  }

  /**
   * @brief Deep copies the host_parquet_representation.
   *
   * @param[in] stream CUDA stream for memory operations
   * @return A unique pointer to the cloned host_parquet_representation.
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  /**
   * @brief Shallow copies the host_parquet_representation.
   *
   * @return A unique pointer to the shallow cloned host_parquet_representation.
   */
  std::unique_ptr<cucascade::idata_representation> shallow_clone();

  /**
   * @brief Gets the fixed multiple blocks allocation containing the Parquet column chunks.
   *
   * @return A const reference to the fixed multiple blocks allocation containing the Parquet
   */
  [[nodiscard]] auto const& get_column_chunks() const { return _column_chunks; };

  /**
   * @brief Gets the hybrid scan Parquet reader as needed for materializing column data.
   *
   * @return A const reference to the hybrid scan Parquet reader.
   */
  [[nodiscard]] hybrid_scan_reader const& get_parquet_reader() const { return *_parquet_reader; };

  /**
   * @brief Moves the hybrid scan Parquet reader out of the representation.
   *
   * @return A unique pointer to the hybrid scan Parquet reader.
   */
  [[nodiscard]] std::shared_ptr<hybrid_scan_reader> get_parquet_reader()
  {
    return _parquet_reader;
  };

  /**
   * @brief Gets the Parquet reader options used to configure the hybrid scan reader.
   *
   * @return A const reference to the Parquet reader options.
   */
  [[nodiscard]] cudf::io::parquet_reader_options const& get_reader_options() const
  {
    return _reader_options;
  };

  /**
   * @brief Gets the row group indices of the row groups represented in the multiple blocks
   * allocation.
   *
   * @return A const reference to the vector of row group indices.
   */
  [[nodiscard]] std::vector<cudf::size_type> const& get_row_group_indices() const
  {
    return _row_group_indices;
  };

  /**
   * @brief Gets the row group indices of the row groups represented in the multiple blocks
   * allocation.
   *
   * @return A reference to the vector of row group indices.
   */
  [[nodiscard]] std::vector<cudf::size_type>& get_row_group_indices()
  {
    return _row_group_indices;
  };

  /**
   * @brief Gets a host span of the row group indices of the row groups represented in the multiple
   * blocks allocation.
   *
   * @return A host span of the row group indices.
   */
  [[nodiscard]] cudf::host_span<cudf::size_type const> get_rg_span() const
  {
    return cudf::host_span<cudf::size_type const>(_row_group_indices.data(),
                                                  _row_group_indices.size());
  };

  /**
   * @brief Gets the byte ranges in the multiple blocks allocation representing the column chunks to
   * be read.
   *
   * @return A const reference to the vector of byte ranges.
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> const& get_column_chunk_byte_ranges()
    const
  {
    return _column_chunk_byte_ranges;
  };

  /**
   * @brief Gets the byte ranges in the multiple blocks allocation representing the column chunks to
   * be read.
   *
   * @return A reference to the vector of byte ranges.
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info>& get_column_chunk_byte_ranges()
  {
    return _column_chunk_byte_ranges;
  };

  /**
   * @brief Gets the size of the representation in bytes (compressed in the multiple blocks
   * allocation).
   *
   * @return The size of the representation in bytes.
   */
  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _size_in_bytes; }

  /**
   * @brief Gets the uncompressed (logical) size of the data represented by this representation.
   *
   * @return The uncompressed size of the data.
   */
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override
  {
    return _uncompressed_size_in_bytes;
  }

  /**
   * @brief Gets the optional fallback datasource for uncached byte ranges.
   *
   * @return A shared_ptr to the fallback datasource, or nullptr if not set.
   */
  [[nodiscard]] std::shared_ptr<cudf::io::datasource> const& get_fallback_datasource() const
  {
    return _fallback_datasource;
  }

  /**
   * @brief Gets the original parquet file size in bytes.
   */
  [[nodiscard]] std::size_t get_file_size() const { return _file_size; }

  /**
   * @brief Sets the fallback datasource for uncached byte ranges.
   *
   * @param ds A shared_ptr to the fallback datasource.
   */
  void set_fallback_datasource(std::shared_ptr<cudf::io::datasource> ds)
  {
    _fallback_datasource = std::move(ds);
  }

  /**
   * @brief Gets the per-GPU map of translated filter expressions for filter pushdown.
   *
   * The converter uses this to pick the entry matching its task's target device so the
   * cudf::scalar device buffers evaluated by cudf::io::read_parquet live on the current
   * GPU.
   *
   * @return A shared_ptr to the per-device filter map (may be null or empty).
   */
  [[nodiscard]] std::shared_ptr<std::unordered_map<int, translated_expression>> const&
  get_filter_expression_by_device() const
  {
    return _filter_expression_by_device;
  }

  /**
   * @brief Gets the vector of projection IDs of columns that remain after filter pushdown.
   *
   * @return A const reference to the vector of post-filter projection IDs.
   */
  [[nodiscard]] std::vector<std::size_t> const& get_post_filter_projection_ids() const
  {
    return _post_filter_projection_ids;
  }

  // -------------------------------------------------------------------------
  // Post-convert hook (used by the iceberg scan path for delete application)
  // -------------------------------------------------------------------------

  /**
   * @brief Install a post-convert hook.
   *
   * Copied from parquet_scan_task_global_state::get_post_convert_fn() in
   * parquet_scan_task::compute_task() so that the host->GPU converter can
   * invoke it immediately after cudf::io::read_parquet returns.
   */
  void set_post_convert_fn(post_convert_fn_t fn) { _post_convert_fn = std::move(fn); }

  [[nodiscard]] bool has_post_convert_fn() const { return _post_convert_fn != nullptr; }

  /**
   * @brief Apply the post-convert hook to @p tbl and return the filtered table.
   *
   * Computes the absolute first-row offset from the parquet reader metadata and
   * forwards it together with the data-file path to the hook.  Only call after
   * checking has_post_convert_fn().
   */
  [[nodiscard]] std::unique_ptr<cudf::table> apply_post_convert(std::unique_ptr<cudf::table> tbl,
                                                                rmm::cuda_stream_view stream)
  {
    return _post_convert_fn(std::move(tbl), _data_file_path, compute_first_row_offset(), stream);
  }

  [[nodiscard]] post_convert_fn_t const& get_post_convert_fn() const { return _post_convert_fn; }

  /**
   * @brief Set the path of the Iceberg data file this batch was produced from.
   *
   * Called by parquet_scan_task::compute_task() so the converter can forward
   * the path to the post-convert hook.
   */
  void set_data_file_path(std::string path) { _data_file_path = std::move(path); }

  [[nodiscard]] std::string const& get_data_file_path() const { return _data_file_path; }

  // -------------------------------------------------------------------------
  // Hive partition column injection
  // -------------------------------------------------------------------------

  void set_partition_inject_fn(partition_inject_fn_t fn) { _partition_inject_fn = std::move(fn); }

  [[nodiscard]] bool has_partition_inject_fn() const { return _partition_inject_fn != nullptr; }

  /// Hive partition values for this representation's file, in hive_partition_columns order.
  /// Set once per task by the caller that installs the inject fn.
  void set_partition_values(std::vector<std::string> values)
  {
    _partition_values = std::move(values);
  }

  [[nodiscard]] std::vector<std::string> const& get_partition_values() const
  {
    return _partition_values;
  }

  /// Only call after checking has_partition_inject_fn().
  [[nodiscard]] std::unique_ptr<cudf::table> apply_partition_inject(
    std::unique_ptr<cudf::table> tbl, rmm::cuda_stream_view stream)
  {
    D_ASSERT(_partition_inject_fn);
    return _partition_inject_fn(std::move(tbl), _data_file_path, _partition_values, stream);
  }

  [[nodiscard]] partition_inject_fn_t const& get_partition_inject_fn() const
  {
    return _partition_inject_fn;
  }

  /**
   * @brief Compute the 0-based absolute row offset of the first row in this batch.
   *
   * Sums num_rows for all row groups preceding the first row group in
   * _row_group_indices using the parquet reader's metadata.
   * Returns 0 if there are no row groups.
   */
  [[nodiscard]] int64_t compute_first_row_offset() const
  {
    if (_row_group_indices.empty()) return 0;
    auto const first_rg = static_cast<size_t>(_row_group_indices[0]);
    auto const& meta    = _parquet_reader->parquet_metadata();
    int64_t offset      = 0;
    for (size_t rg = 0; rg < first_rg && rg < meta.row_groups.size(); ++rg) {
      offset += meta.row_groups[rg].num_rows;
    }
    return offset;
  }

 private:
  host_parquet_representation(
    cucascade::memory::memory_space* memory_space,
    std::shared_ptr<hybrid_scan_reader> parquet_reader,
    cudf::io::parquet_reader_options reader_options,
    std::vector<cudf::size_type> row_group_indices,
    std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges,
    std::size_t size_in_bytes,
    std::size_t uncompressed_size_in_bytes,
    std::size_t file_size,
    std::shared_ptr<cudf::io::datasource> fallback_datasource,
    std::shared_ptr<std::unordered_map<int, translated_expression>> filter_expression_by_device,
    std::vector<std::size_t> post_filter_projection_ids)
    : idata_representation(*memory_space),
      _parquet_reader(std::move(parquet_reader)),
      _reader_options(std::move(reader_options)),
      _row_group_indices(std::move(row_group_indices)),
      _column_chunk_byte_ranges(std::move(column_chunk_byte_ranges)),
      _size_in_bytes(size_in_bytes),
      _uncompressed_size_in_bytes(uncompressed_size_in_bytes),
      _file_size(file_size),
      _fallback_datasource(fallback_datasource),
      _filter_expression_by_device(std::move(filter_expression_by_device)),
      _post_filter_projection_ids(std::move(post_filter_projection_ids))
  {
  }

  std::shared_ptr<cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation>
    _column_chunks;
  std::shared_ptr<hybrid_scan_reader> _parquet_reader;
  cudf::io::parquet_reader_options _reader_options;
  std::vector<cudf::size_type> _row_group_indices;
  std::vector<cudf::io::text::byte_range_info> _column_chunk_byte_ranges;
  std::size_t _size_in_bytes;
  std::size_t _uncompressed_size_in_bytes;
  std::size_t _file_size{0};
  std::shared_ptr<cudf::io::datasource> _fallback_datasource;
  std::shared_ptr<std::unordered_map<int, translated_expression>> _filter_expression_by_device;
  std::vector<std::size_t> _post_filter_projection_ids;
  /// Optional post-convert hook (null for plain parquet, set for iceberg V2 deletes).
  post_convert_fn_t _post_convert_fn;
  /// Optional partition column injection hook (null unless hive-partitioned).
  partition_inject_fn_t _partition_inject_fn;
  /// Path of the data file this batch was read from.
  std::string _data_file_path;
  /// Hive partition values for @ref _data_file_path, in hive_partition_columns order.
  /// Empty unless the table has hive partitions; consumed by apply_partition_inject().
  std::vector<std::string> _partition_values;
};
}  // namespace sirius
