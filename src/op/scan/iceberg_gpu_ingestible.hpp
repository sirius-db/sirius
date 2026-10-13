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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {
class iceberg_dv_preparation;

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

  /// Legacy adapter input, used only when delete_sets and deferred are absent.
  /// Resolved at plan time on the legacy path: an unreadable manifest must fail planning rather
  /// than arrive as "no deletes", which is how a V2 table returns rows it deleted while looking
  /// fine.
  std::shared_ptr<const IcebergDeleteData> delete_data;

  /// Complete per-file results keyed by manifest path, including proved-empty files.
  /// When absent on the legacy path, L0 adapts delete_data without changing the producer.
  std::optional<iceberg_delete_sets> delete_sets;
  std::shared_ptr<iceberg_dv_preparation> deferred;
};

/**
 * @brief Parquet ingestible that applies Iceberg deletes to each decoded batch.
 *
 * Only two things differ from the base:
 *
 * 1. @ref next_split_provider forces reader-side pushdown off for files with deletes. Positional
 *    deletes are keyed on a row's position within its file, so rows dropped during decode make
 *    the mapping unrecoverable. The predicate still runs, in @c post_filter_and_project, after
 *    the deletes — the order Iceberg requires. Row-group PRUNING stays on: it only removes rows
 *    the predicate could not match, and the footer still gives the survivors' offsets.
 *
 * 2. @ref materialize_metadata_to_table decodes through the base, then applies per-file deletes.
 *
 * Equality deletes are declined by the planner: they need their key columns force-projected.
 */
class iceberg_gpu_ingestible : public parquet_gpu_ingestible {
 public:
  explicit iceberg_gpu_ingestible(std::unique_ptr<iceberg_ingestible_table_info> info);

  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override;
  bool can_claim_preparation();
  scan_manager::unit_key next_preparation_unit() const;
  void stop_preparation() noexcept;
  void finish_preparation() noexcept;

  filtered_table materialize_metadata_to_table(
    scan_info const& info,
    const cucascade::memory::memory_space& mem_space,
    ::cuda::stream_ref stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache) override;

 private:
  /// Resolves which scanned file each manifest-side key refers to, once. A mismatch would
  /// silently find no deletes for that file, so ambiguity is refused.
  void build_delete_key_map(std::vector<std::string> const& resolved_file_paths,
                            IcebergDeleteData const* legacy,
                            iceberg_delete_sets const& sets);

  /// The delete-map key for a path the scan reads; the path itself when they already agree.
  [[nodiscard]] std::string const& delete_key_for(std::string const& scan_path) const;

  std::shared_ptr<iceberg_delete_sets const> _delete_sets;
  std::shared_ptr<iceberg_dv_preparation> _deferred;
  scan_manager::w_permit _next_permit;
  size_t _next_preparation = 0;
  bool _next_registered    = false;
  std::string _table_path;
  /// Scan path -> delete-map key, only for the paths where the two spellings differ.
  std::unordered_map<std::string, std::string> _delete_key_by_scan_path;
};

std::shared_ptr<iceberg_gpu_ingestible> make_ingestible(
  std::unique_ptr<iceberg_ingestible_table_info> info);

}  // namespace sirius::op::scan
