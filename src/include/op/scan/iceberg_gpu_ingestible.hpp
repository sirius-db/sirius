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

#include <op/scan/iceberg_delete_filter.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// iceberg_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Parquet bind data plus the table's materialized Iceberg delete data.
 *
 * An Iceberg table's data files ARE parquet, and @c iceberg_scan resolves its manifests at
 * bind time into the same @c MultiFileBindData @c read_parquet produces — so everything the
 * parquet path needs is already carried by the base. The only thing iceberg adds is deletes.
 */
class iceberg_ingestible_table_info : public parquet_ingestible_table_info {
 public:
  /// The table path passed to @c iceberg_scan; identity for logging and delete discovery.
  std::string table_path;

  /// Delete data resolved at plan time. Never null on this path: an unreadable manifest must
  /// fail planning (CPU fallback), never arrive as "no deletes" — silently empty delete data
  /// is how a V2 table returns rows it logically deleted while looking like a success.
  std::shared_ptr<const IcebergDeleteData> delete_data;
};

//===----------------------------------------------------------------------===//
// iceberg_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Parquet ingestible that applies Iceberg deletes to each decoded batch.
 *
 * Extends @c parquet_gpu_ingestible rather than reimplementing it: reading the data files is
 * identical, and only two things change.
 *
 * 1. @ref create_batch_coalescer wraps the parquet coalescer to force reader-side filter
 *    pushdown off for every emitted split. Positional deletes and deletion vectors are keyed
 *    on a row's position within its data file; if the reader drops rows during decode, the
 *    decoded rows no longer line up with file positions and the mapping cannot be recovered.
 *    The scan's predicate is still applied — @c post_filter_and_project applies it after
 *    deletes, which is also the required order.
 *
 * 2. @ref materialize_metadata_to_table decodes through the base, then applies the delete
 *    pipeline to the result. Row-group pruning stays enabled: it only removes rows the
 *    predicate could not have matched, and the surviving row groups' file offsets are known
 *    from the footer.
 *
 * Equality deletes are NOT applied here yet — they need the key columns force-projected into
 * the scan, which the planner does not do. The planner declines tables carrying them.
 */
class iceberg_gpu_ingestible : public parquet_gpu_ingestible {
 public:
  explicit iceberg_gpu_ingestible(std::unique_ptr<iceberg_ingestible_table_info> info);

  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;

  filtered_table materialize_metadata_to_table(scan_info const& info,
                                               const cucascade::memory::memory_space& mem_space,
                                               rmm::cuda_stream_view stream) override;

 private:
  /// Establish which scanned data file each delete-map key refers to. Manifest paths and the
  /// paths DuckDB resolved are usually identical strings, and a mismatch would silently find no
  /// deletes for that file, so the correspondence is resolved once and ambiguity is refused.
  void build_delete_key_map(std::vector<std::string> const& resolved_file_paths);

  /// The delete-map key for a path the scan reads; the path itself when they already agree.
  [[nodiscard]] std::string const& delete_key_for(std::string const& scan_path) const;

  std::shared_ptr<const IcebergDeleteData> _delete_data;
  /// Delete stages to run, in order. Empty when the table has no deletes, in which case this
  /// behaves exactly like the parquet ingestible apart from the pushdown suppression.
  iceberg_delete_pipeline _pipeline;
  std::string _table_path;
  /// Scan path -> delete-map key, only for the paths where the two spellings differ.
  std::unordered_map<std::string, std::string> _delete_key_by_scan_path;
};

std::shared_ptr<iceberg_gpu_ingestible> make_ingestible(
  std::unique_ptr<iceberg_ingestible_table_info> info);

/**
 * @brief Row provenance of the batch a @c parquet_split_info decodes to.
 *
 * The decoded table is the concatenation, in split order, of each slice's selected row groups.
 * Each (slice, row group) pair becomes one @ref batch_row_run whose file offset is the prefix
 * sum of that file's preceding row-group row counts. Exposed for testing.
 */
std::vector<batch_row_run> build_batch_layout(parquet_split_info const& split);

}  // namespace sirius::op::scan
