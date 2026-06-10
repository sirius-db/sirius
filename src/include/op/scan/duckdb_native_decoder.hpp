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

// cudf
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
// duckdb_native_split_payload
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split payload consumed by @ref decode_duckdb_native_split.
 *
 * Built by @c duckdb_native_gpu_ingestible::next_split_provider from a
 * contiguous run of the walker's @c duckdb_row_group_metadata. The
 * @ref table_info pointer aliases the ingestible's owned bind data and is
 * valid for the lifetime of every emitted split (the ingestible outlives
 * its splits).
 */
struct duckdb_native_split_payload {
  std::vector<duckdb_row_group_metadata> row_groups;
  /// Stable alias into the ingestible's owned bind data. Carries
  /// @c storage / @c context / @c projected_cols / @c projected_types that
  /// the decoder reads.
  duckdb_native_ingestible_table_info const* table_info = nullptr;
  /// Sirius IO substrate handles used to read .db blocks via
  /// @c sirius_ioctx::host_read. Required by the decoder, which throws if
  /// either is absent.
  std::shared_ptr<sirius::io::sirius_ioctx> io_ctx;
  std::shared_ptr<sirius::io::sirius_io_object> db_io_object;
};

//===----------------------------------------------------------------------===//
// decode_duckdb_native_split
//===----------------------------------------------------------------------===//
/// Decode a single split (contiguous run of row-group metadata from the
/// walker) into a cudf::table. Supported: fixed-width data (UNCOMPRESSED /
/// RLE / BITPACKING / CONSTANT), varchar data (UNCOMPRESSED / DICTIONARY /
/// FSST / DICT_FSST), validity (UNCOMPRESSED / EMPTY / CONSTANT / ROARING),
/// rowid synthesis. Throws std::runtime_error on any codec the walker
/// accepted but this decoder does not implement.
std::unique_ptr<cudf::table> decode_duckdb_native_split(duckdb_native_split_payload const& split,
                                                        cucascade::memory::memory_space& mem_space,
                                                        rmm::cuda_stream_view stream);

}  // namespace sirius::op::scan
