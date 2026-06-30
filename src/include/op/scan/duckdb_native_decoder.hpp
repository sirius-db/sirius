/*
 * Copyright 2026, Sirius Contributors.
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
#include <io/io_context.hpp>  // sirius_ioctx + sirius_io_object
#include <op/scan/duckdb_native_metadata.hpp>

// duckdb
#include <duckdb/storage/single_file_block_manager.hpp>

// cudf
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <memory>
#include <vector>

namespace sirius::op::scan {

class duckdb_native_ingestible_table_info;

//===----------------------------------------------------------------------===//
// row_group_file_ranges
//===----------------------------------------------------------------------===//
/// @brief On-disk byte ranges @p row_group's projected segments occupy in the .db file. Used both
/// to prefetch (fadvise) and to drive the decode reads, so the two touch identical bytes.
std::vector<cudf::io::text::byte_range_info> row_group_file_ranges(
  duckdb::SingleFileBlockManager const& block_manager, duckdb_row_group_metadata const& row_group);

//===----------------------------------------------------------------------===//
// decode_duckdb_native_split
//===----------------------------------------------------------------------===//
/// @brief Decode a single split as provided by the split_provider into a cudf::table.
/// Supported: fixed-width data (UNCOMPRESSED / RLE / BITPACKING / CONSTANT),
///            varchar data (UNCOMPRESSED / DICTIONARY / FSST / DICT_FSST),
///            validity (UNCOMPRESSED / EMPTY / CONSTANT / ROARING),
///            rowid synthesis.
/// @throws std::runtime_error on any codec the walker accepted but this decoder does not implement.
std::unique_ptr<cudf::table> decode_duckdb_native_split(
  std::vector<duckdb_row_group_metadata> const& row_groups,
  duckdb_native_ingestible_table_info const& table_info,
  sirius::io::sirius_datasource& datasource,
  cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op::scan
