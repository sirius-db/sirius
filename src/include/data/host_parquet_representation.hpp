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
   */
  host_parquet_representation(cucascade::memory::memory_space* memory_space,
                              cucascade::memory::fixed_multiple_blocks_allocation column_chunks,
                              std::unique_ptr<hybrid_scan_reader> parquet_reader,
                              cudf::io::parquet_reader_options reader_options,
                              std::vector<cudf::size_type> row_group_indices,
                              std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges,
                              std::size_t size_in_bytes,
                              std::size_t uncompressed_size_in_bytes)
    : idata_representation(*memory_space),
      _column_chunks(std::move(column_chunks)),
      _parquet_reader(std::move(parquet_reader)),
      _reader_options(std::move(reader_options)),
      _row_group_indices(std::move(row_group_indices)),
      _all_column_chunk_byte_ranges(std::move(column_chunk_byte_ranges)),
      _size_in_bytes(size_in_bytes),
      _uncompressed_size_in_bytes(uncompressed_size_in_bytes)
  {
  }

  /**
   * @brief Constructs a host_parquet_representation for multi-stage decompression, which requires
   * separate byte ranges for filter and payload columns.
   *
   * @param[in] memory_space The memory space to which the representation beloongs.
   * @param[in] column_chunks The fixed multiple blocks allocation containing the Parquet column
   * chunks.
   * @param[in] parquet_reader An instance hybrid scan Parquet reader for a given Parquet file.
   * @param[in] reader_options The Parquet reader options used to configure the hybrid scan reader
   * for materializing data.
   * @param[in] row_group_indices The row group indices of the row groups represented in the
   * multiple blocks allocation.
   * @param[in] filter_column_chunk_byte_ranges The byte ranges in the multiple blocks allocation
   * representing the filter column chunks to be read.
   * @param[in] payload_column_chunk_byte_ranges The byte ranges in the multiple blocks allocation
   * representing the payload column chunks to be read.
   * @param[in] size_in_bytes The size of the representation in bytes (compressed).
   * @param[in] uncompressed_size_in_bytes The uncompressed size of the data represented by this
   * representation.
   * @param[in] page_index_buffer Optional buffer containing the page index bytes for multistage
   * decompression. We need this in each representation so that clones can setup the page index in
   * the reader.
   */
  host_parquet_representation(
    cucascade::memory::memory_space* memory_space,
    cucascade::memory::fixed_multiple_blocks_allocation column_chunks,
    std::unique_ptr<hybrid_scan_reader> parquet_reader,
    cudf::io::parquet_reader_options reader_options,
    std::vector<cudf::size_type> row_group_indices,
    std::vector<cudf::io::text::byte_range_info> filter_column_chunk_byte_ranges,
    std::vector<cudf::io::text::byte_range_info> payload_column_chunk_byte_ranges,
    std::size_t size_in_bytes,
    std::size_t uncompressed_size_in_bytes,
    std::shared_ptr<cudf::io::datasource::buffer> page_index_buffer)
    : idata_representation(*memory_space),
      _column_chunks(std::move(column_chunks)),
      _parquet_reader(std::move(parquet_reader)),
      _reader_options(std::move(reader_options)),
      _row_group_indices(std::move(row_group_indices)),
      _filter_column_chunk_byte_ranges(std::move(filter_column_chunk_byte_ranges)),
      _payload_column_chunk_byte_ranges(std::move(payload_column_chunk_byte_ranges)),
      _size_in_bytes(size_in_bytes),
      _uncompressed_size_in_bytes(uncompressed_size_in_bytes),
      _page_index_buffer(std::move(page_index_buffer))
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
   * @brief Gets the fixed multiple blocks allocation containing the Parquet column chunks.
   *
   * @return A const reference to the fixed multiple blocks allocation containing the Parquet
   */
  [[nodiscard]] cucascade::memory::fixed_multiple_blocks_allocation const& get_column_chunks() const
  {
    return _column_chunks;
  };

  /**
   * @brief Gets the hybrid scan Parquet reader as needed for materializing column data.
   *
   * @return A const reference to the hybrid scan Parquet reader.
   */
  [[nodiscard]] hybrid_scan_reader const& get_parquet_reader() const { return *_parquet_reader; };

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
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> const&
  get_all_column_chunk_byte_ranges() const
  {
    return _all_column_chunk_byte_ranges;
  };

  /**
   * @brief Gets the byte ranges in the multiple blocks allocation representing the filter column
   * chunks to be read for multi-stage decompression.
   *
   * @return A const reference to the vector of byte ranges for the filter columns.
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> const&
  get_filter_column_chunk_byte_ranges() const
  {
    return _filter_column_chunk_byte_ranges;
  };

  /**
   * @brief Gets the byte ranges in the multiple blocks allocation representing the payload column
   * chunks to be read for multi-stage decompression.
   *
   * @return A const reference to the vector of byte ranges for the payload columns.
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> const&
  get_payload_column_chunk_byte_ranges() const
  {
    return _payload_column_chunk_byte_ranges;
  };

  /**
   * @brief Gets the size of the representation in bytes (compressed in the multiple blocks
   * allocation).
   *
   * @return The size of the representation in bytes.
   */
  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _size_in_bytes; }

  /**
   * @brief Gets the uncompressed size of the data represented by this representation.
   *
   * @return The uncompressed size of the data.
   */
  [[nodiscard]] std::size_t get_uncompressed_size_in_bytes() const
  {
    return _uncompressed_size_in_bytes;
  }

  [[nodiscard]] std::shared_ptr<cudf::io::datasource::buffer> get_page_index_buffer() const
  {
    return _page_index_buffer;
  }

 private:
  cucascade::memory::fixed_multiple_blocks_allocation
    _column_chunks;  ///< The multiple blocks allocation containing contiguous Parquet column chunks
  std::unique_ptr<hybrid_scan_reader>
    _parquet_reader;  ///< The hybrid scan Parquet reader for decompression
  cudf::io::parquet_reader_options
    _reader_options;  ///< The Parquet reader options used to configure the hybrid scan reader
                      ///< (needed for copies/clones)
  std::vector<cudf::size_type>
    _row_group_indices;  ///< The row group indices of the row groups represented in the multiple
                         ///< blocks allocation
  std::vector<cudf::io::text::byte_range_info>
    _all_column_chunk_byte_ranges;  ///< The byte range info needed to parse all column chunks
  std::vector<cudf::io::text::byte_range_info>
    _filter_column_chunk_byte_ranges;  ///< The byte range info needed to parse the filter column
                                       ///< chunks (for multi-stage decompression)
  std::vector<cudf::io::text::byte_range_info>
    _payload_column_chunk_byte_ranges;  /// The byte range info needed to parse the payload column
                                        /// chunks (for multi-stage decompression)
  std::size_t _size_in_bytes;           ///< The compressed size of the data in bytes
  std::size_t _uncompressed_size_in_bytes;  ///< The uncompressed size of the data in bytes
  std::shared_ptr<cudf::io::datasource::buffer>
    _page_index_buffer;  ///< Buffer containing the page index bytes for multistage decompression
                         ///< (necessary for copies/clones, otherwise would be set up in scan task)
};
}  // namespace sirius
