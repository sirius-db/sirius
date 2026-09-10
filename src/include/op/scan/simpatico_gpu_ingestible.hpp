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

// A .hpln file as a scan SOURCE rather than only a pin-time representation.
//
// A .hpln holds a single compressed chunk, so this source is one file, one chunk, one split: no
// pruning, no projection and no row filter. The decode itself is the pinned-chunk path unchanged
// -- the payload stages into pinned host memory byte-for-byte (read_hpln_into_pinned) and is then
// reconstructed and decompressed exactly as a pinned chunk is (compression_converters.cpp,
// decompress_host_to_gpu). What this class supplies is the split-provider shape the scan operator
// drives.

// sirius
#include <op/scan/gpu_ingestible.hpp>

// cudf
#include <cudf/types.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// simpatico_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Bind data for a scan over one .hpln file; factory input for
 *        @c simpatico_gpu_ingestible.
 *
 * Everything here comes from @c read_hpln_schema, which parses the header and the file's
 * `logical_types` segment without staging any payload and without a GPU -- so a bind costs one
 * tail read plus one header read, which is what makes the format bindable at all.
 */
class simpatico_ingestible_table_info : public ingestible_table_info {
 public:
  /// The .hpln path, and the pinned-cache identity for this table. Exactly one entry: a .hpln
  /// carries one chunk, so a table is one file.
  std::vector<std::string> resolved_file_paths;
  std::vector<std::string> names;
  /// The engine's logical types, from the file when it declares them and derived from the cuDF
  /// physical types otherwise -- lossy for DECIMAL precision, nullability and time zones.
  duckdb::vector<duckdb::LogicalType> types;
  /// What each column decodes to in cuDF. Sizing the decoded table needs these rather than
  /// @ref types: a DECIMAL is INT64 to the engine and may be DECIMAL32 in the file.
  std::vector<cudf::data_type> physical_types;
  std::int64_t num_rows = 0;

  /// Where the payload stages before it is fetched to the GPU. Pinned, so the H2D copy is a DMA
  /// rather than a staged bounce, and so a pin can adopt the same blob untouched. Required: an
  /// ingest with nowhere to stage cannot be served, and refusing at construction beats failing on
  /// the first split.
  cucascade::memory::memory_space* host_space = nullptr;

  simpatico_ingestible_table_info() = default;

  [[nodiscard]] std::span<std::string const> column_names() const override { return names; }

  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    return resolved_file_paths;
  }
};

/// Bind @p path: read its schema and build the table info a @c simpatico_gpu_ingestible is
/// constructed from. Throws std::runtime_error if the file is missing or is not a readable .hpln.
[[nodiscard]] std::unique_ptr<simpatico_ingestible_table_info> bind_simpatico_file(
  std::string const& path, cucascade::memory::memory_space& host_space);

//===----------------------------------------------------------------------===//
// simpatico_scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief One unit of .hpln scan work: the whole file.
 *
 * Carries no byte ranges to advise. Where parquet's split names row groups so the sequencer can
 * prefetch them, a .hpln has one chunk and its reader locates the payload from the trailer, so
 * there is nothing to hand the prefetcher that it does not already read in one pass.
 */
class simpatico_scan_info : public scan_info {
 public:
  std::string path;
  /// Decoded size of the columns this split produces; drives the memory reservation.
  std::size_t decoded_bytes = 0;

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override { return decoded_bytes; }
};

//===----------------------------------------------------------------------===//
// simpatico_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Scan source over a .hpln file.
 *
 * The metadata walk emits exactly one split and the coalescer passes it straight through: with one
 * chunk in the file there is nothing to partition and nothing to bundle.
 */
class simpatico_gpu_ingestible : public gpu_ingestible {
 public:
  explicit simpatico_gpu_ingestible(std::unique_ptr<simpatico_ingestible_table_info> info);

  ~simpatico_gpu_ingestible() override;

  [[nodiscard]] std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;

  [[nodiscard]] bool has_processed_all_metadata() const override;

  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override;

  filtered_table materialize_metadata_to_table(
    scan_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
    std::unique_ptr<cudf::column>* survivors,
    std::span<std::size_t const> elided) override;

  [[nodiscard]] ingestible_table_info const& table_info() const noexcept override { return *_info; }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override;

 private:
  std::unique_ptr<simpatico_ingestible_table_info> _info;
  /// Claims the one split. An atomic for the same reason parquet's index is: the driver may call
  /// @ref next_split_provider from several dispatcher threads and only one may win.
  std::atomic<bool> _split_claimed{false};
};

[[nodiscard]] std::shared_ptr<simpatico_gpu_ingestible> make_ingestible(
  std::unique_ptr<simpatico_ingestible_table_info> info);

}  // namespace sirius::op::scan
