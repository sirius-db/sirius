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
#include <expression_executor/gpu_expression_translator.hpp>

// cucascade
#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/types.hpp>

// standard library
#include <memory>
#include <vector>

namespace sirius {

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
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

 public:
  /**
   * @brief Describes how column chunk byte ranges are organized in the allocation.
   *
   * In non-multistage mode, all column chunks are stored contiguously in `all`.
   * In multistage mode, filter and payload column chunks are stored separately.
   */
  struct column_byte_ranges {
    std::vector<cudf::io::text::byte_range_info> all;      ///< All column chunks (non-multistage)
    std::vector<cudf::io::text::byte_range_info> filter;   ///< Filter column chunks (multistage)
    std::vector<cudf::io::text::byte_range_info> payload;  ///< Payload column chunks (multistage)

    /// @brief Returns true if this represents a multistage split.
    [[nodiscard]] bool is_multistage() const { return !filter.empty() || !payload.empty(); }
  };

  /**
   * @brief Constructs a host_parquet_representation.
   *
   * @param[in] memory_space The memory space to which the representation belongs.
   * @param[in] column_chunks The fixed multiple blocks allocation containing the Parquet column
   * chunks.
   * @param[in] parquet_reader An instance hybrid scan Parquet reader for a given Parquet file.
   * @param[in] reader_options The Parquet reader options used to configure the hybrid scan reader
   * for materializing data.
   * @param[in] row_group_indices The row group indices of the row groups represented in the
   * multiple blocks allocation.
   * @param[in] byte_ranges The column chunk byte ranges (all, or split into filter/payload).
   * @param[in] size_in_bytes The size of the representation in bytes (compressed).
   * @param[in] uncompressed_size_in_bytes The uncompressed size of the data represented by this
   * representation.
   * @param[in] translated_filter Shared ownership of the translated filter expression, to keep the
   * cuDF AST alive through this representation's lifetime.
   * @param[in] page_index_buffer Optional buffer containing the page index bytes for multistage
   * decompression.
   * @param[in] column_reorder_map Permutation for reordering concatenated filter and payload
   * columns back to original order (multistage only).
   */
  host_parquet_representation(
    cucascade::memory::memory_space* memory_space,
    cucascade::memory::fixed_multiple_blocks_allocation column_chunks,
    std::unique_ptr<hybrid_scan_reader> parquet_reader,
    cudf::io::parquet_reader_options reader_options,
    std::vector<cudf::size_type> row_group_indices,
    column_byte_ranges byte_ranges,
    std::size_t size_in_bytes,
    std::size_t uncompressed_size_in_bytes,
    std::shared_ptr<gpu_expression_translator::translated_expression> translated_filter = nullptr,
    std::shared_ptr<cudf::io::datasource::buffer> page_index_buffer                    = nullptr,
    std::vector<cudf::size_type> column_reorder_map                                    = {})
    : idata_representation(*memory_space),
      _column_chunks(std::move(column_chunks)),
      _parquet_reader(std::move(parquet_reader)),
      _reader_options(std::move(reader_options)),
      _row_group_indices(std::move(row_group_indices)),
      _byte_ranges(std::move(byte_ranges)),
      _size_in_bytes(size_in_bytes),
      _uncompressed_size_in_bytes(uncompressed_size_in_bytes),
      _translated_filter_pin(std::move(translated_filter)),
      _page_index_buffer(std::move(page_index_buffer)),
      _column_reorder_map(std::move(column_reorder_map))
  {
  }

  /**
   * @brief Deep copies the host_parquet_representation.
   *
   * @param[in] stream CUDA stream for memory operations
   * @return A unique pointer to the cloned host_parquet_representation.
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  //===----------Accessors----------===//

  [[nodiscard]] cucascade::memory::fixed_multiple_blocks_allocation const& get_column_chunks() const
  {
    return _column_chunks;
  }

  [[nodiscard]] hybrid_scan_reader const& get_parquet_reader() const { return *_parquet_reader; }

  [[nodiscard]] cudf::io::parquet_reader_options const& get_reader_options() const
  {
    return _reader_options;
  }

  [[nodiscard]] std::vector<cudf::size_type> const& get_row_group_indices() const
  {
    return _row_group_indices;
  }

  [[nodiscard]] cudf::host_span<cudf::size_type const> get_rg_span() const
  {
    return cudf::host_span<cudf::size_type const>(_row_group_indices.data(),
                                                  _row_group_indices.size());
  }

  /// @brief Returns true if this representation uses multistage decompression.
  [[nodiscard]] bool is_multistage() const { return _byte_ranges.is_multistage(); }

  /// @brief Gets the byte ranges struct.
  [[nodiscard]] column_byte_ranges const& get_byte_ranges() const { return _byte_ranges; }

  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _size_in_bytes; }

  [[nodiscard]] std::size_t get_uncompressed_size_in_bytes() const
  {
    return _uncompressed_size_in_bytes;
  }

  [[nodiscard]] std::shared_ptr<cudf::io::datasource::buffer> get_page_index_buffer() const
  {
    return _page_index_buffer;
  }

  [[nodiscard]] std::vector<cudf::size_type> const& get_column_reorder_map() const
  {
    return _column_reorder_map;
  }

  [[nodiscard]] std::shared_ptr<gpu_expression_translator::translated_expression>
  get_translated_filter_pin() const
  {
    return _translated_filter_pin;
  }

 private:
  cucascade::memory::fixed_multiple_blocks_allocation
    _column_chunks;  ///< Multiple blocks allocation containing contiguous Parquet column chunks
  std::unique_ptr<hybrid_scan_reader>
    _parquet_reader;  ///< Hybrid scan Parquet reader for decompression
  cudf::io::parquet_reader_options
    _reader_options;  ///< Parquet reader options (needed for copies/clones)
  std::vector<cudf::size_type>
    _row_group_indices;          ///< Row group indices represented in the allocation
  column_byte_ranges _byte_ranges;  ///< Column chunk byte ranges (unified or split)

  std::size_t _size_in_bytes;               ///< Compressed size of the data in bytes
  std::size_t _uncompressed_size_in_bytes;  ///< Uncompressed size of the data in bytes

  std::shared_ptr<gpu_expression_translator::translated_expression>
    _translated_filter_pin;  ///< Pins the cuDF AST and its owned scalars alive through this
                             ///< representation's lifetime
  std::shared_ptr<cudf::io::datasource::buffer>
    _page_index_buffer;  ///< Page index bytes for multistage decompression
  std::vector<cudf::size_type>
    _column_reorder_map;  ///< Permutation for reordering concatenated filter/payload columns
};
}  // namespace sirius
