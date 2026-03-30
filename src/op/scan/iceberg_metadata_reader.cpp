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

#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/iceberg_avro_reader.hpp>
#include <log/logging.hpp>

#include <duckdb/main/connection.hpp>

namespace sirius::op::scan {

IcebergDeleteFiles read_iceberg_delete_metadata(
  duckdb::ClientContext& context,
  std::string const& table_path)
{
  IcebergDeleteFiles files;

  try {
    // Get the manifest-list path from iceberg_snapshots().
    // This is lightweight (one row) and works even when iceberg_metadata() fails.
    duckdb::Connection snap_conn(*context.db);
    auto snap_result =
      snap_conn.Query("SELECT manifest_list FROM iceberg_snapshots('" + table_path +
                      "') ORDER BY sequence_number DESC LIMIT 1");

    if (snap_result->HasError()) {
      SIRIUS_LOG_WARN("[iceberg_metadata_reader] iceberg_snapshots() failed for '{}': {}",
                      table_path, snap_result->GetError());
      return files;
    }

    auto chunk = snap_result->Fetch();
    if (!chunk || chunk->size() == 0) {
      SIRIUS_LOG_DEBUG("[iceberg_metadata_reader] No snapshots for '{}'.", table_path);
      return files;
    }

    std::string manifest_list_path = chunk->GetValue(0, 0).ToString();

    // Parse manifest-list Avro → list of (manifest_path, content_type).
    auto manifests = read_iceberg_manifest_list(manifest_list_path);

    constexpr int kPositionDeletes = 1;
    constexpr int kEqualityDeletes = 2;

    for (auto const& [mpath, mcontent] : manifests) {
      if (mcontent == kPositionDeletes) {
        auto del_files = read_iceberg_manifest_delete_files(mpath, kPositionDeletes);
        for (auto& f : del_files) {
          files.positional_delete_files.push_back(std::move(f));
        }
      } else if (mcontent == kEqualityDeletes) {
        auto del_files = read_iceberg_manifest_delete_files(mpath, kEqualityDeletes);
        for (auto& f : del_files) {
          files.equality_delete_files.push_back(std::move(f));
        }
      }
    }

    SIRIUS_LOG_INFO(
      "[iceberg_metadata_reader] '{}': {} positional-delete, {} equality-delete file(s).",
      table_path,
      files.positional_delete_files.size(),
      files.equality_delete_files.size());

  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN(
      "[iceberg_metadata_reader] Failed for '{}': {}. Treating as V1.",
      table_path, e.what());
  }

  return files;
}

}  // namespace sirius::op::scan
