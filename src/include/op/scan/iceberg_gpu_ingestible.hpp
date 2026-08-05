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

/**
 * @brief Parquet bind data plus the table's materialized Iceberg delete data.
 *
 * Iceberg data files ARE parquet, and @c iceberg_scan binds to the same @c MultiFileBindData
 * @c read_parquet produces, so the base already carries everything but the deletes.
 */
class iceberg_ingestible_table_info : public parquet_ingestible_table_info {
 public:
  /// As passed to @c iceberg_scan.
  std::string table_path;

  /// Resolved at plan time, never null: an unreadable manifest must fail planning rather than
  /// arrive as "no deletes", which is how a V2 table returns rows it deleted while looking fine.
  std::shared_ptr<const IcebergDeleteData> delete_data;
};

/**
 * @brief Parquet ingestible that applies Iceberg deletes to each decoded batch.
 *
 * Only two things differ from the base:
 *
 * 1. @ref create_batch_coalescer forces reader-side pushdown off for every split. Positional
 *    deletes are keyed on a row's position within its file, so rows dropped during decode make
 *    the mapping unrecoverable. The predicate still runs, in @c post_filter_and_project, after
 *    the deletes — the order Iceberg requires. Row-group PRUNING stays on: it only removes rows
 *    the predicate could not match, and the footer still gives the survivors' offsets.
 *
 * 2. @ref materialize_metadata_to_table decodes through the base, then applies the pipeline.
 *
 * Equality deletes are declined by the planner: they need their key columns force-projected.
 */
class iceberg_gpu_ingestible : public parquet_gpu_ingestible {
 public:
  explicit iceberg_gpu_ingestible(std::unique_ptr<iceberg_ingestible_table_info> info);

  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;

  filtered_table materialize_metadata_to_table(scan_info const& info,
                                               const cucascade::memory::memory_space& mem_space,
                                               rmm::cuda_stream_view stream) override;

 private:
  /// Resolves which scanned file each manifest-side key refers to, once. A mismatch would
  /// silently find no deletes for that file, so ambiguity is refused.
  void build_delete_key_map(std::vector<std::string> const& resolved_file_paths);

  /// The delete-map key for a path the scan reads; the path itself when they already agree.
  [[nodiscard]] std::string const& delete_key_for(std::string const& scan_path) const;

  std::shared_ptr<const IcebergDeleteData> _delete_data;
  /// Empty when the table has no deletes, which also leaves pushdown enabled.
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
