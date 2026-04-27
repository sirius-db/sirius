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

#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <duckdb/main/connection.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_avro_reader.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/puffin_reader.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

namespace {

// =========================================================================
// Path discovery (from Avro manifests)
// =========================================================================

/// Everything discovered from a single manifest-list scan.
struct IcebergManifestDiscovery {
  std::vector<std::string> positional_delete_files;
  std::vector<IcebergDeleteFileEntry> equality_delete_entries;
  std::vector<IcebergDeleteFileEntry> deletion_vector_entries;
  /// Per-data-file sequence numbers (from data manifests, for eq delete filtering).
  std::unordered_map<std::string, int64_t> data_file_sequence_numbers;
};

/// Escape single quotes for safe SQL interpolation.
std::string escape_sql_string(std::string const& s)
{
  std::string out = s;
  for (std::string::size_type pos = 0; (pos = out.find('\'', pos)) != std::string::npos; pos += 2) {
    out.replace(pos, 1, "''");
  }
  return out;
}

/// Discover all delete files and data file metadata using DuckDB's iceberg_metadata().
/// Delegates ALL Avro/manifest parsing to DuckDB's iceberg extension — handles all
/// manifest versions, codecs, catalog types, and snapshot selection automatically.
/// The only fallback to our custom Avro reader is for V3 deletion vectors (PUFFIN),
/// where iceberg_metadata() doesn't expose content_offset/size/referenced_data_file.
IcebergManifestDiscovery discover_from_manifests(duckdb::ClientContext& context,
                                                 std::string const& table_path,
                                                 std::optional<uint64_t> snapshot_id)
{
  IcebergManifestDiscovery result;

  duckdb::Connection conn(*context.db);
  conn.Query("SET unsafe_enable_version_guessing = true");

  // Build the iceberg_metadata() query, with optional snapshot selection.
  std::string query =
    "SELECT content, file_path, manifest_sequence_number, file_format, manifest_path "
    "FROM iceberg_metadata('" +
    escape_sql_string(table_path) + "'";
  if (snapshot_id.has_value()) {
    query += ", snapshot_from_id = " + std::to_string(snapshot_id.value());
  }
  query += ")";

  auto meta_result = conn.Query(query);
  if (meta_result->HasError()) {
    SIRIUS_LOG_WARN(
      "[iceberg] iceberg_metadata() failed for '{}': {}", table_path, meta_result->GetError());
    return result;
  }

  static constexpr auto kPositionDeletes = "POSITION_DELETES";
  static constexpr auto kEqualityDeletes = "EQUALITY_DELETES";
  static constexpr auto kExisting        = "EXISTING";
  static constexpr auto kFormatPuffin    = "PUFFIN";

  // Cache manifest reads by path to avoid N+1 re-reads for multiple DVs
  // in the same manifest (iceberg_metadata() doesn't expose DV-specific fields).
  std::unordered_map<std::string, std::vector<IcebergDeleteFileEntry>> dv_manifest_cache;

  while (true) {
    auto chunk = meta_result->Fetch();
    if (!chunk || chunk->size() == 0) break;

    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      auto content  = chunk->GetValue(0, i).ToString();
      auto filepath = chunk->GetValue(1, i).ToString();
      auto seq      = chunk->GetValue(2, i).GetValue<int64_t>();

      if (content == kPositionDeletes) {
        auto file_format = chunk->GetValue(3, i).ToString();
        if (file_format == kFormatPuffin) {
          // V3 deletion vector: iceberg_metadata() doesn't expose content_offset,
          // content_size, or referenced_data_file, so we fall back to our custom
          // Avro reader for the containing manifest.
          auto manifest_path = chunk->GetValue(4, i).ToString();
          auto& cached       = dv_manifest_cache[manifest_path];
          if (cached.empty()) { cached = read_iceberg_manifest_entries(manifest_path, 1); }
          for (auto& dv : cached) {
            if (dv.is_deletion_vector()) { result.deletion_vector_entries.push_back(dv); }
          }
        } else {
          result.positional_delete_files.push_back(std::move(filepath));
        }
      } else if (content == kEqualityDeletes) {
        IcebergDeleteFileEntry entry;
        entry.file_path       = std::move(filepath);
        entry.content         = 2;
        entry.sequence_number = seq;
        result.equality_delete_entries.push_back(std::move(entry));
      } else if (content == kExisting) {
        result.data_file_sequence_numbers[filepath] = seq;
      }
    }
  }

  SIRIUS_LOG_INFO(
    "[iceberg] '{}': {} positional-delete, {} equality-delete, "
    "{} deletion-vector, {} data file(s).",
    table_path,
    result.positional_delete_files.size(),
    result.equality_delete_entries.size(),
    result.deletion_vector_entries.size(),
    result.data_file_sequence_numbers.size());

  return result;
}

// =========================================================================
// Content readers (moved from iceberg_scan_task.cpp)
// =========================================================================

/**
 * @brief Read a positional-delete parquet file and append its records to @p out_map.
 *
 * Uses DuckDB's CPU-based read_parquet to avoid a wasteful GPU round-trip
 * (delete files are tiny metadata — no reason to allocate GPU memory).
 * The file must have schema: { file_path VARCHAR, pos BIGINT }.
 */
void read_positional_delete_file(duckdb::DatabaseInstance& db,
                                 std::string const& delete_file_path,
                                 std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  duckdb::Connection conn(db);

  auto result = conn.Query("SELECT file_path, pos FROM read_parquet('" +
                           escape_sql_string(delete_file_path) + "')");

  if (result->HasError()) {
    throw std::runtime_error("[iceberg] Failed to read positional-delete file '" +
                             delete_file_path + "': " + result->GetError());
  }

  while (true) {
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0) break;

    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      auto file_path = chunk->GetValue(0, i).ToString();
      auto pos_val   = chunk->GetValue(1, i).GetValue<int64_t>();
      out_map[file_path].push_back(pos_val);
    }
  }
}

/// Result of reading an equality-delete parquet file.
struct equality_delete_read_result {
  std::unique_ptr<cudf::table> tbl;
  std::vector<std::string> col_names;
  std::vector<std::optional<int32_t>> field_ids;
};

/**
 * @brief Read an equality-delete parquet file and return table, column names, and field IDs.
 *
 * Uses GPU-accelerated cudf::io::read_parquet because equality-delete tables
 * can be large and columnar, and the result feeds directly into a GPU-side
 * cudf::distinct_hash_join — so keeping data on device avoids a round-trip.
 *
 * Also reads the parquet footer to extract Iceberg field IDs for each key
 * column, enabling schema-evolution-safe matching against data files.
 */
equality_delete_read_result read_equality_delete_file(std::string const& delete_file_path)
{
  auto stream = cudf::get_default_stream();
  auto opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{delete_file_path}).build();
  auto result = cudf::io::read_parquet(opts, stream);

  if (!result.tbl) {
    throw std::runtime_error("[iceberg] Failed to read equality-delete file: " + delete_file_path);
  }

  std::vector<std::string> col_names;
  col_names.reserve(result.metadata.schema_info.size());
  for (auto const& si : result.metadata.schema_info) {
    col_names.push_back(si.name);
  }

  // Extract Iceberg field IDs from the parquet footer schema.
  std::vector<std::optional<int32_t>> field_ids;
  try {
    auto ds = cudf::io::datasource::create(delete_file_path);
    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.push_back(std::move(ds));
    auto footers = cudf::io::read_parquet_footers(sources);

    if (!footers.empty()) {
      auto id_map = extract_field_id_map(footers[0]);
      field_ids.reserve(col_names.size());
      for (auto const& name : col_names) {
        auto it = id_map.find(name);
        field_ids.push_back(it != id_map.end() ? std::optional{it->second} : std::nullopt);
      }
    }
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG(
      "[iceberg] Could not extract field IDs from '{}': {}", delete_file_path, e.what());
    field_ids.clear();
  }

  stream.synchronize();
  SIRIUS_LOG_INFO("[iceberg] read equality-delete: path={} rows={} cols={}",
                  delete_file_path,
                  result.tbl->num_rows(),
                  result.tbl->num_columns());
  return {std::move(result.tbl), std::move(col_names), std::move(field_ids)};
}

// =========================================================================
// Materialization helpers
// =========================================================================

/// Read all positional deletes + deletion vectors into a single map.
void materialize_positional_deletes(duckdb::DatabaseInstance& db,
                                    IcebergManifestDiscovery const& files,
                                    std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  // V2 positional-delete parquet files (CPU read via DuckDB).
  if (!files.positional_delete_files.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} positional-delete file(s).",
                    files.positional_delete_files.size());
    for (auto const& del_path : files.positional_delete_files) {
      SIRIUS_LOG_DEBUG("[iceberg] Reading positional-delete file: {}", del_path);
      read_positional_delete_file(db, del_path, out_map);
    }
  }

  // V3 deletion vectors from Puffin files.
  // Per iceberg spec, deletion vectors supersede positional deletes for the same data file.
  if (!files.deletion_vector_entries.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} deletion vector(s).",
                    files.deletion_vector_entries.size());
    for (auto const& dv_entry : files.deletion_vector_entries) {
      SIRIUS_LOG_DEBUG("[iceberg] Reading deletion vector for data file '{}' from '{}'",
                       dv_entry.referenced_data_file,
                       dv_entry.file_path);
      auto positions = read_deletion_vector(
        dv_entry.file_path, dv_entry.content_offset, dv_entry.content_size_in_bytes);
      // DVs supersede V2 positional deletes for the same data file — overwrite.
      out_map[dv_entry.referenced_data_file] = std::move(positions);
    }
  }

  // Sort all position vectors.
  for (auto& [path, positions] : out_map) {
    std::sort(positions.begin(), positions.end());
  }

  if (!out_map.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loaded positional deletes for {} data file(s).", out_map.size());
  }
}

/// Build one EqualityDeleteGroup from a set of tables sharing the same schema.
EqualityDeleteGroup build_equality_group(std::vector<std::string> key_names,
                                         std::vector<std::optional<int32_t>> key_field_ids,
                                         std::vector<cudf::table_view> const& views)
{
  auto stream = cudf::get_default_stream();

  auto all_rows = (views.size() == 1) ? std::make_unique<cudf::table>(views[0], stream)
                                      : cudf::concatenate(views, stream);

  std::vector<cudf::size_type> all_key_indices(static_cast<size_t>(all_rows->num_columns()));
  std::iota(all_key_indices.begin(), all_key_indices.end(), cudf::size_type{0});

  auto deduped = cudf::distinct(all_rows->view(),
                                all_key_indices,
                                cudf::duplicate_keep_option::KEEP_FIRST,
                                cudf::null_equality::EQUAL,
                                cudf::nan_equality::ALL_EQUAL,
                                stream);

  // Avoid fmt::join — it requires <fmt/ranges.h> which the vcpkg fmt build
  // does not pull in transitively. Format the comma-joined list manually.
  std::string key_names_joined;
  for (size_t i = 0; i < key_names.size(); ++i) {
    if (i > 0) key_names_joined += ", ";
    key_names_joined += key_names[i];
  }
  SIRIUS_LOG_INFO(
    "[iceberg] Equality-delete group [{}]: {} row(s).", key_names_joined, deduped->num_rows());

  auto hash_join = std::make_unique<cudf::distinct_hash_join>(
    deduped->view(), cudf::null_equality::EQUAL, 0.5, stream);
  stream.synchronize();

  EqualityDeleteGroup group;
  group.delete_table  = std::move(deduped);
  group.key_names     = std::move(key_names);
  group.key_field_ids = std::move(key_field_ids);
  group.hash_join     = std::move(hash_join);
  return group;
}

/// Read equality deletes, group by (schema + sequence number), build per-group hash joins.
/// This matches DuckDB's approach: each group has exactly one sequence number,
/// so the scan-time check is a simple CPU comparison (no extra GPU work).
void materialize_equality_deletes(std::vector<IcebergDeleteFileEntry> const& eq_entries,
                                  IcebergDeleteData& data)
{
  if (eq_entries.empty()) return;

  SIRIUS_LOG_INFO("[iceberg] Loading {} equality-delete file(s).", eq_entries.size());

  // Group delete files by (column names + sequence number).
  struct FileGroup {
    std::vector<std::string> key_names;
    std::vector<std::optional<int32_t>> key_field_ids;
    std::vector<cudf::table_view> views;
    std::vector<std::unique_ptr<cudf::table>> owned;
    int64_t sequence_number;
  };
  std::vector<FileGroup> groups;

  for (auto const& eq_entry : eq_entries) {
    SIRIUS_LOG_DEBUG("[iceberg] Reading equality-delete file: {} (seq={})",
                     eq_entry.file_path,
                     eq_entry.sequence_number);
    auto read_result = read_equality_delete_file(eq_entry.file_path);

    // Find existing group with same column names AND same sequence number.
    FileGroup* target = nullptr;
    for (auto& g : groups) {
      if (g.key_names == read_result.col_names && g.sequence_number == eq_entry.sequence_number) {
        target = &g;
        break;
      }
    }
    if (!target) {
      groups.push_back(
        {read_result.col_names, read_result.field_ids, {}, {}, eq_entry.sequence_number});
      target = &groups.back();
    }

    target->views.push_back(read_result.tbl->view());
    target->owned.push_back(std::move(read_result.tbl));
  }

  // Build one EqualityDeleteGroup per (schema, sequence_number).
  for (auto& g : groups) {
    if (g.views.empty()) continue;
    auto group = build_equality_group(std::move(g.key_names), std::move(g.key_field_ids), g.views);
    group.sequence_number = g.sequence_number;
    data.equality_delete_groups.push_back(std::move(group));
  }

  SIRIUS_LOG_INFO("[iceberg] Built {} equality-delete group(s).",
                  data.equality_delete_groups.size());
}

}  // anonymous namespace

// =========================================================================
// Public API
// =========================================================================

std::shared_ptr<const IcebergDeleteData> read_iceberg_delete_data(
  duckdb::ClientContext& context,
  std::string const& table_path,
  std::optional<uint64_t> snapshot_id)
{
  auto data = std::make_shared<IcebergDeleteData>();

  try {
    // Single-pass discovery: reads manifest list once, each manifest once.
    auto discovery = discover_from_manifests(context, table_path, snapshot_id);

    bool has_pos_deletes =
      !discovery.positional_delete_files.empty() || !discovery.deletion_vector_entries.empty();
    bool has_eq_deletes = !discovery.equality_delete_entries.empty();

    if (!has_pos_deletes && !has_eq_deletes) {
      SIRIUS_LOG_DEBUG("[iceberg] No delete files for '{}'; treating as V1.", table_path);
      return data;
    }

    if (has_pos_deletes) {
      materialize_positional_deletes(*context.db, discovery, data->positional_deletes);
    }
    if (has_eq_deletes) {
      data->data_file_sequence_numbers = std::move(discovery.data_file_sequence_numbers);
      materialize_equality_deletes(discovery.equality_delete_entries, *data);
    }

  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[iceberg] Failed for '{}': {}. Treating as V1.", table_path, e.what());
    return std::make_shared<IcebergDeleteData>();
  }

  return data;
}

std::unordered_map<std::string, int32_t> extract_field_id_map(
  cudf::io::parquet::FileMetaData const& file_meta)
{
  std::unordered_map<std::string, int32_t> result;

  // The schema vector is a flattened depth-first representation.
  // Leaf columns have num_children == 0.  The root element (index 0)
  // is the message/file-level container and is always skipped.
  for (size_t i = 1; i < file_meta.schema.size(); ++i) {
    auto const& elem = file_meta.schema[i];
    if (elem.num_children == 0 && elem.field_id.has_value()) {
      result[elem.name] = elem.field_id.value();
    }
  }

  return result;
}

}  // namespace sirius::op::scan
