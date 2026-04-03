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

#include "op/scan/sirius_parquet_metadata_scan_operator.hpp"

#include "log/logging.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/scan/parquet_scan_task.hpp"  // detail::make_selected_column_indices, detail::projected_columns_are_flat

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>

#if CUDF_VERSION_NUM >= 2604
#include <cudf/io/parquet_io_utils.hpp>
#endif

#include <algorithm>
#include <stdexcept>

namespace sirius::op::scan {

// ---------------------------------------------------------------------------
// Fallback footer reader for cudf < 26.04
// ---------------------------------------------------------------------------
#if CUDF_VERSION_NUM < 2604
namespace {
// NOTE: The buffer returned here must have an identical byte layout to what
// cudf::io::parquet::fetch_footer_to_host (cudf >= 26.04) returns so that the
// footer_len / footer_offset / metadata_bytes calculations in execute() remain
// consistent across both code paths.  Both paths return only the thrift-encoded
// footer body (footer_len bytes starting at file_size - TAIL_SIZE - footer_len),
// NOT including the leading PAR1 magic bytes.  The caller separately accounts for
// the leading magic and the 8-byte trailer when computing metadata_bytes.
std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host_fallback(
  cudf::io::datasource& datasource)
{
  constexpr size_t PARQUET_MAGIC_SIZE = 4;
  constexpr size_t FOOTER_LEN_SIZE    = 4;
  constexpr size_t TAIL_SIZE          = PARQUET_MAGIC_SIZE + FOOTER_LEN_SIZE;

  auto const file_size = datasource.size();
  if (file_size < TAIL_SIZE + PARQUET_MAGIC_SIZE) {
    throw std::runtime_error("File too small to be a valid Parquet file");
  }

  auto tail_buf    = datasource.host_read(file_size - TAIL_SIZE, TAIL_SIZE);
  auto const* tail = tail_buf->data();

  uint32_t footer_len      = tail[0] | (tail[1] << 8) | (tail[2] << 16) | (tail[3] << 24);
  auto const footer_offset = file_size - TAIL_SIZE - footer_len;
  return datasource.host_read(footer_offset, footer_len);
}
}  // namespace
#endif

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
sirius_parquet_metadata_scan_operator::sirius_parquet_metadata_scan_operator(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::idx_t estimated_cardinality,
  std::vector<std::string> file_paths,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids,
  duckdb::vector<std::string> const& names,
  size_t approximate_batch_size,
  size_t max_file_processed)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARQUET_METADATA_SCAN, std::move(types), estimated_cardinality),
    _file_paths(std::move(file_paths)),
    _approximate_batch_size(approximate_batch_size),
    _max_file_processed(max_file_processed)
{
  if (_file_paths.empty()) {
    throw std::runtime_error("[sirius_parquet_metadata_scan_operator] No input files to scan.");
  }
  _total_files = _file_paths.size();

  _selected_column_indices = detail::make_selected_column_indices(column_ids, projection_ids);
  _is_projected            = !projection_ids.empty();

  if (_is_projected && !names.empty()) {
    _projected_column_names.reserve(_selected_column_indices.size());
    for (auto col_idx : _selected_column_indices) {
      _projected_column_names.push_back(names[col_idx]);
    }
  }
}

// ---------------------------------------------------------------------------
// Source interface
// ---------------------------------------------------------------------------
std::optional<task_creation_hint> sirius_parquet_metadata_scan_operator::get_next_task_hint()
{
  if (_next_file_idx.load(std::memory_order_relaxed) < _total_files) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return std::nullopt;
}

bool sirius_parquet_metadata_scan_operator::all_ports_empty()
{
  return _next_file_idx.load(std::memory_order_relaxed) >= _total_files;
}

std::unique_ptr<operator_data> sirius_parquet_metadata_scan_operator::get_next_task_input_data()
{
  auto start = _next_file_idx.fetch_add(_max_file_processed, std::memory_order_relaxed);
  if (start >= _total_files) { return nullptr; }

  auto end = std::min(start + _max_file_processed, _total_files);
  std::vector<std::string> batch_files(_file_paths.begin() + static_cast<ptrdiff_t>(start),
                                       _file_paths.begin() + static_cast<ptrdiff_t>(end));

  return std::make_unique<parquet_metadata_input>(std::move(batch_files), _approximate_batch_size);
}

// ---------------------------------------------------------------------------
// execute() — metadata parsing
// ---------------------------------------------------------------------------
std::unique_ptr<operator_data> sirius_parquet_metadata_scan_operator::execute(
  const operator_data& input_data, rmm::cuda_stream_view /*stream*/)
{
  auto const* input_ptr = dynamic_cast<const parquet_metadata_input*>(&input_data);
  if (!input_ptr) {
    throw std::runtime_error(
      "[sirius_parquet_metadata_scan_operator] execute() called with unexpected operator_data "
      "type; expected parquet_metadata_input.");
  }
  auto const& input = *input_ptr;

  constexpr size_t PARQUET_MAGIC_SIZE = 4;
  constexpr size_t FOOTER_TAIL_SIZE   = 8;

  auto result                     = std::make_unique<partitioned_parquet_metadata>();
  result->file_paths              = input.file_paths;
  result->selected_column_indices = _selected_column_indices;
  result->file_sizes.reserve(input.file_paths.size());
  result->footer_offsets.reserve(input.file_paths.size());
  result->metadata_byte_sizes.reserve(input.file_paths.size());

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  datasources.reserve(input.file_paths.size());
  footer_buffers.reserve(input.file_paths.size());

  for (auto const& file_path : input.file_paths) {
    auto datasource      = cudf::io::datasource::create(file_path);
    auto const file_size = datasource->size();
    datasources.push_back(std::move(datasource));

#if CUDF_VERSION_NUM >= 2604
    footer_buffers.push_back(cudf::io::parquet::fetch_footer_to_host(*datasources.back()));
    auto const footer_len = footer_buffers.back()->size();
#else
    footer_buffers.push_back(fetch_footer_to_host_fallback(*datasources.back()));
    auto const footer_len = footer_buffers.back()->size();
#endif

    auto const footer_offset  = file_size - FOOTER_TAIL_SIZE - footer_len;
    auto const metadata_bytes = PARQUET_MAGIC_SIZE + footer_len + FOOTER_TAIL_SIZE;

    result->file_sizes.push_back(file_size);
    result->footer_offsets.push_back(footer_offset);
    result->metadata_byte_sizes.push_back(metadata_bytes);
  }

  // Build reader options
  result->reader_options = cudf::io::parquet_reader_options::builder().build();

  // Parse metadata by creating a hybrid_scan_reader per footer
  result->file_metadatas.reserve(input.file_paths.size());
  for (auto& footer_buffer : footer_buffers) {
    auto reader = cudf::io::parquet::experimental::hybrid_scan_reader(
      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      result->reader_options);
    result->file_metadatas.push_back(reader.parquet_metadata());
  }

  // Apply column-name projection
  if (_is_projected && !_projected_column_names.empty()) {
    for (auto const& meta : result->file_metadatas) {
      if (!detail::projected_columns_are_flat(meta, _selected_column_indices)) {
        throw std::runtime_error(
          "[sirius_parquet_metadata_scan_operator] Parquet scans with projections currently only "
          "support flat projected columns.");
      }
    }
#if CUDF_VERSION_NUM >= 2604
    result->reader_options.set_column_names(_projected_column_names);
#else
    result->reader_options.set_columns(_projected_column_names);
#endif
  }

  // Accumulate per-(file, row-group) byte sizes
  std::vector<std::vector<size_t>> row_group_uncompressed_bytes(input.file_paths.size());
  std::vector<std::vector<size_t>> row_group_compressed_bytes(input.file_paths.size());

  for (size_t file_idx = 0; file_idx < result->file_metadatas.size(); ++file_idx) {
    auto const& meta    = result->file_metadatas[file_idx];
    auto const total_rg = meta.row_groups.size();
    row_group_uncompressed_bytes[file_idx].reserve(total_rg);
    row_group_compressed_bytes[file_idx].reserve(total_rg);

    for (auto const& rg : meta.row_groups) {
      size_t uncompressed = 0;
      size_t compressed   = 0;
      for (auto col_idx : _selected_column_indices) {
        if (col_idx >= rg.columns.size()) {
          throw std::runtime_error(
            "[sirius_parquet_metadata_scan_operator] Column index out of range for row group — "
            "parquet file may be malformed or have inconsistent schema across row groups.");
        }
        auto const& col_meta = rg.columns[col_idx].meta_data;
        if (col_meta.total_uncompressed_size > 0) {
          uncompressed += col_meta.total_uncompressed_size;
        }
        if (col_meta.total_compressed_size > 0) { compressed += col_meta.total_compressed_size; }
      }
      row_group_uncompressed_bytes[file_idx].push_back(uncompressed);
      row_group_compressed_bytes[file_idx].push_back(compressed);
    }
  }

  // Partition row groups into ranges based on approximate_batch_size
  size_t const approx = input.approximate_batch_size;
  for (size_t file_idx = 0; file_idx < result->file_metadatas.size(); ++file_idx) {
    auto const& meta = result->file_metadatas[file_idx];

    size_t partition_uncompressed = 0;
    size_t partition_compressed   = 0;
    size_t rg_start               = 0;
    size_t rg_count               = 0;

    for (size_t rg_idx = 0; rg_idx < meta.row_groups.size(); ++rg_idx) {
      partition_uncompressed += row_group_uncompressed_bytes[file_idx][rg_idx];
      partition_compressed += row_group_compressed_bytes[file_idx][rg_idx];
      ++rg_count;

      if (partition_uncompressed >= approx) {
        result->row_group_partitions.emplace_back(
          file_idx, rg_start, rg_count, partition_uncompressed, partition_compressed);
        partition_uncompressed = 0;
        partition_compressed   = 0;
        rg_start               = rg_idx + 1;
        rg_count               = 0;
      }
    }
    if (rg_count > 0) {
      result->row_group_partitions.emplace_back(
        file_idx, rg_start, rg_count, partition_uncompressed, partition_compressed);
    }
  }

  SIRIUS_LOG_DEBUG(
    "[sirius_parquet_metadata_scan_operator] Parsed {} files, produced {} row-group partitions",
    input.file_paths.size(),
    result->row_group_partitions.size());

  return result;
}

}  // namespace sirius::op::scan
