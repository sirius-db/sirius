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

#include "op/scan/parquet_metadata_scanner.hpp"

#include "log/logging.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"  // hybrid_scan_reader
#include "op/scan/parquet_schema_mapping.hpp"      // detail::leaf_indices_for_column

#include <cudf/io/parquet_io_utils.hpp>

#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

file_metadata_scan_result scan_parquet_file_metadata(
  std::string const& file_path,
  scan_plan const& plan,
  cudf::io::parquet_reader_options const& reader_options,
  std::vector<std::string> const& data_column_names,
  std::unordered_set<std::size_t> const& pure_filter_positions,
  rmm::cuda_stream_view stream)
{
  file_metadata_scan_result result;

  //===----------Read metadata footer----------===//
  auto datasource    = cudf::io::datasource::create(file_path);
  auto footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

  //===----------Parse metadata----------===//
  hybrid_scan_reader reader(
    cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()), reader_options);
  result.file_metadata =
    std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata());

  //===----------Resolve selected DuckDB columns to parquet column chunk indices----------===//
  // row_group.columns is indexed in parquet schema-leaf order (preorder), which can differ from
  // DuckDB's logical column order. Resolve by name per file (chunk order is consistent across row
  // groups in a single file, but can vary across files).
  if (plan.is_projected()) {
    result.selected_chunk_indices.reserve(data_column_names.size());
    for (std::size_t k = 0; k < data_column_names.size(); ++k) {
      auto leaves = detail::leaf_indices_for_column(*result.file_metadata, data_column_names[k]);
      if (leaves.empty()) {
        throw std::runtime_error("[scan_parquet_file_metadata] Projected column '" +
                                 data_column_names[k] +
                                 "' not found in parquet file: " + file_path);
      }
      bool const is_pure_filter = pure_filter_positions.count(k);
      for (auto const leaf : leaves) {
        result.selected_chunk_indices.push_back(leaf);
        if (is_pure_filter) { result.pure_filter_chunk_indices.insert(leaf); }
      }
    }
  }

  //===----------Row Group Pruning----------===//
  // Single source of truth for "is filter-pushdown active": the AST filter installed on
  // reader_options. Avoids the caller passing a redundant boolean that could drift out of sync.
  result.selected_row_group_indices = reader.all_row_groups(reader_options);
  if (reader_options.get_filter().has_value()) {
    auto const before = result.selected_row_group_indices.size();
    // clang-format off
    SIRIUS_LOG_DEBUG("[scan_parquet_file_metadata] Row group pruning: file: {}\n" \
                     "                                                    before: {}",
                     file_path,
                     before);
    // clang-format on
    result.selected_row_group_indices = reader.filter_row_groups_with_stats(
      result.selected_row_group_indices, reader_options, stream);
    auto const after = result.selected_row_group_indices.size();
    // clang-format off
    SIRIUS_LOG_DEBUG("[scan_parquet_file_metadata]                       after: {} (pruned {})",
                     after,
                     before - after);
    // clang-format on
  }

  result.datasource = std::shared_ptr<cudf::io::datasource>(std::move(datasource));
  return result;
}

row_group_bytes compute_row_group_bytes(
  cudf::io::parquet::RowGroup const& row_group,
  scan_plan const& plan,
  std::vector<std::size_t> const& selected_chunk_indices,
  std::unordered_set<std::size_t> const& pure_filter_chunk_indices)
{
  row_group_bytes bytes{0, 0};

  auto accumulate = [&](cudf::io::parquet::ColumnChunk const& chunk, bool is_pure_filter) {
    auto const& md = chunk.meta_data;
    // Pure filter columns are not part of the scan result, so we omit them from the
    // uncompressed byte count used for sizing partitions.
    if (md.total_uncompressed_size > 0 && !is_pure_filter) {
      bytes.uncompressed_bytes += static_cast<std::size_t>(md.total_uncompressed_size);
    }
    if (md.total_compressed_size > 0) {
      bytes.compressed_bytes += static_cast<std::size_t>(md.total_compressed_size);
    }
  };

  if (plan.is_projected()) {
    for (auto const chunk_idx : selected_chunk_indices) {
      accumulate(row_group.columns[chunk_idx], pure_filter_chunk_indices.contains(chunk_idx));
    }
  } else {
    // Non-projected: all chunks contribute, no pure-filter pruning.
    for (auto const& chunk : row_group.columns) {
      accumulate(chunk, false);
    }
  }

  return bytes;
}

}  // namespace sirius::op::scan
