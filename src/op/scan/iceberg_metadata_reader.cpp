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

#include "io/io_context.hpp"
#include "io/types.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <duckdb/main/client_context.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/transaction/meta_transaction.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/puffin_reader.hpp>
#include <sirius_context.hpp>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

namespace {

/// One file entry from an Iceberg manifest: equality-delete files and V3 deletion vectors.
struct IcebergDeleteFileEntry {
  std::string file_path;
  int content{0};                     // 0=DATA, 1=POSITION_DELETES, 2=EQUALITY_DELETES
  std::string file_format;            // "parquet" or "puffin" (always lowercase)
  std::string referenced_data_file;   // data file this DV applies to (V3, empty if absent)
  int64_t content_offset{-1};         // byte offset in Puffin file (V3, -1 if absent)
  int64_t content_size_in_bytes{-1};  // byte length of DV blob (V3, -1 if absent)
  int64_t sequence_number{0};         // manifest entry sequence number (for eq delete filtering)

  /// Requires file_format already lowercased by the reader.
  [[nodiscard]] bool is_deletion_vector() const
  {
    return file_format == "puffin" && content_offset >= 0 && content_size_in_bytes > 0;
  }
};

/// Everything discovered from a single manifest-list scan.
struct IcebergManifestDiscovery {
  std::vector<std::string> positional_delete_files;
  std::vector<IcebergDeleteFileEntry> equality_delete_entries;
  std::vector<IcebergDeleteFileEntry> deletion_vector_entries;
  /// Per-data-file sequence numbers (from data manifests, for eq delete filtering).
  std::unordered_map<std::string, int64_t> data_file_sequence_numbers;
};

std::string escape_sql_string(std::string const& s)
{
  std::string out = s;
  for (std::string::size_type pos = 0; (pos = out.find('\'', pos)) != std::string::npos; pos += 2) {
    out.replace(pos, 1, "''");
  }
  return out;
}

/// Reads one manifest's POSITION_DELETES entries. @p conn must already be bracketed as an
/// internal query. `lower()` is load-bearing: Iceberg writes "PUFFIN" but is_deletion_vector()
/// tests "puffin", and without the fold the table reads as having no deletes.
std::vector<IcebergDeleteFileEntry> read_deletion_vectors_from_manifest(
  duckdb::Connection& conn, std::string const& manifest_path)
{
  auto result = conn.Query(
    "SELECT data_file.file_path, lower(data_file.file_format), "
    "COALESCE(data_file.referenced_data_file, ''), "
    "COALESCE(data_file.content_offset, -1), "
    "COALESCE(data_file.content_size_in_bytes, -1) "
    "FROM read_avro('" +
    escape_sql_string(manifest_path) +
    "') "
    "WHERE data_file.content = 1");

  if (!result || result->HasError()) {
    // Never degrade to an empty list: that means "no deletion vectors" and returns deleted rows.
    throw std::runtime_error("[iceberg] Failed to read manifest '" + manifest_path +
                             "': " + (result ? result->GetError() : "null result"));
  }

  std::vector<IcebergDeleteFileEntry> entries;
  while (true) {
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0) break;

    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      IcebergDeleteFileEntry entry;
      entry.content               = 1;
      entry.file_path             = chunk->GetValue(0, i).ToString();
      entry.file_format           = chunk->GetValue(1, i).ToString();
      entry.referenced_data_file  = chunk->GetValue(2, i).ToString();
      entry.content_offset        = chunk->GetValue(3, i).GetValue<int64_t>();
      entry.content_size_in_bytes = chunk->GetValue(4, i).GetValue<int64_t>();
      entries.push_back(std::move(entry));
    }
  }
  return entries;
}

/// Discovers delete files and data-file metadata via iceberg_metadata(), which handles every
/// manifest version, codec and catalog type. V3 deletion vectors need a second pass through
/// read_avro: iceberg_metadata() does not expose content_offset/size/referenced_data_file.
IcebergManifestDiscovery discover_from_manifests(duckdb::ClientContext& context,
                                                 std::string const& table_path,
                                                 std::optional<uint64_t> snapshot_id)
{
  IcebergManifestDiscovery result;

  duckdb::Connection conn(*context.db);
  // Per-connection bracket: the caller's guard does not cover this one, and unbracketed these
  // queries contend for the instance-wide plan slot the query being planned already holds.
  duckdb::SiriusContext::InternalQueryGuard conn_guard(*conn.context);
  conn.Query("SET unsafe_enable_version_guessing = true");

  std::string query =
    "SELECT content, file_path, manifest_sequence_number, file_format, manifest_path "
    "FROM iceberg_metadata('" +
    escape_sql_string(table_path) + "'";
  if (snapshot_id.has_value()) {
    query += ", snapshot_from_id = " + std::to_string(snapshot_id.value());
  }
  query += ")";

  auto meta_result = conn.Query(query);
  if (!meta_result || meta_result->HasError()) {
    // Empty would read as "this table has no delete files" and return deleted rows.
    throw std::runtime_error("[iceberg] iceberg_metadata() failed for '" + table_path +
                             "': " + (meta_result ? meta_result->GetError() : "null result"));
  }

  static constexpr auto kPositionDeletes = "POSITION_DELETES";
  static constexpr auto kEqualityDeletes = "EQUALITY_DELETES";
  static constexpr auto kExisting        = "EXISTING";
  static constexpr auto kFormatPuffin    = "PUFFIN";

  // Cached by path: several DVs can share one manifest, and each would otherwise re-read it.
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
          // iceberg_metadata() omits the V3 fields, so re-read the manifest via read_avro.
          auto manifest_path = chunk->GetValue(4, i).ToString();
          auto& cached       = dv_manifest_cache[manifest_path];
          if (cached.empty()) { cached = read_deletion_vectors_from_manifest(conn, manifest_path); }
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

/// Appends one positional-delete file's records to @p out_map. Schema must be
/// { file_path VARCHAR, pos BIGINT }. CPU read: these files are tiny metadata.
void read_positional_delete_file(duckdb::DatabaseInstance& db,
                                 std::string const& delete_file_path,
                                 std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  duckdb::Connection conn(db);
  duckdb::SiriusContext::InternalQueryGuard conn_guard(*conn.context);

  auto result = conn.Query("SELECT file_path, pos FROM read_parquet('" +
                           escape_sql_string(delete_file_path) + "')");

  if (!result || result->HasError()) {
    throw std::runtime_error("[iceberg] Failed to read positional-delete file '" +
                             delete_file_path +
                             "': " + (result ? result->GetError() : "null result"));
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

/// Reads an equality-delete file into a GPU table, plus its column names and Iceberg field ids.
/// Stays on device because the result feeds a cudf::distinct_hash_join directly. @p ioctx must
/// be non-null: this entry point has no kvikio bypass.
equality_delete_read_result read_equality_delete_file(std::string const& delete_file_path,
                                                      sirius::io::sirius_ioctx& ioctx)
{
  auto stream = cudf::get_default_stream();

  // One datasource for both passes: its io_object opens 2 fds, so reusing avoids a reopen.
  auto datasource = ioctx.open_datasource(delete_file_path);

  auto opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasource.get()}).build();
  auto result = cudf::io::read_parquet(opts, stream);

  if (!result.tbl) {
    throw std::runtime_error("[iceberg] Failed to read equality-delete file: " + delete_file_path);
  }

  std::vector<std::string> col_names;
  col_names.reserve(result.metadata.schema_info.size());
  for (auto const& si : result.metadata.schema_info) {
    col_names.push_back(si.name);
  }

  std::vector<std::optional<int32_t>> field_ids;
  try {
    // The table read has completed, so moving ownership to the footer pass is safe.
    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.push_back(std::move(datasource));
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

/// Read all positional deletes + deletion vectors into a single map.
void materialize_positional_deletes(duckdb::DatabaseInstance& db,
                                    IcebergManifestDiscovery const& files,
                                    std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  if (!files.positional_delete_files.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} positional-delete file(s).",
                    files.positional_delete_files.size());
    for (auto const& del_path : files.positional_delete_files) {
      SIRIUS_LOG_DEBUG("[iceberg] Reading positional-delete file: {}", del_path);
      read_positional_delete_file(db, del_path, out_map);
    }
  }

  // Per spec, deletion vectors supersede positional deletes for the same data file.
  if (!files.deletion_vector_entries.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} deletion vector(s).",
                    files.deletion_vector_entries.size());
    for (auto const& dv_entry : files.deletion_vector_entries) {
      SIRIUS_LOG_DEBUG("[iceberg] Reading deletion vector for data file '{}' from '{}'",
                       dv_entry.referenced_data_file,
                       dv_entry.file_path);
      auto positions = read_deletion_vector(
        dv_entry.file_path, dv_entry.content_offset, dv_entry.content_size_in_bytes);
      out_map[dv_entry.referenced_data_file] = std::move(positions);
    }
  }

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

/// Groups equality deletes by (schema, sequence number) so the scan-time applicability check is
/// one CPU comparison per group.
void materialize_equality_deletes(std::vector<IcebergDeleteFileEntry> const& eq_entries,
                                  sirius::io::sirius_ioctx& ioctx,
                                  IcebergDeleteData& data)
{
  if (eq_entries.empty()) return;

  SIRIUS_LOG_INFO("[iceberg] Loading {} equality-delete file(s).", eq_entries.size());

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
    auto read_result = read_equality_delete_file(eq_entry.file_path, ioctx);

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

namespace {

/// Per-query memo for read_iceberg_delete_data. See the wrapper below for why it is keyed and
/// scoped the way it is; see clear_iceberg_delete_data_cache() for why it must be cleared.
std::mutex g_delete_data_cache_mtx;
std::unordered_map<std::string, std::shared_ptr<const IcebergDeleteData>> g_delete_data_cache;

/// Incremented on every read that actually walks the manifests. See the header.
std::atomic<uint64_t> g_uncached_read_count{0};

}  // namespace

uint64_t iceberg_delete_data_uncached_read_count()
{
  return g_uncached_read_count.load(std::memory_order_relaxed);
}

void clear_iceberg_delete_data_cache()
{
  std::lock_guard lk{g_delete_data_cache_mtx};
  g_delete_data_cache.clear();
}

namespace {

std::shared_ptr<const IcebergDeleteData> read_iceberg_delete_data_uncached(
  duckdb::ClientContext& context,
  std::string const& table_path,
  sirius::io::sirius_ioctx* metadata_ioctx,
  std::optional<uint64_t> snapshot_id)
{
  g_uncached_read_count.fetch_add(1, std::memory_order_relaxed);

  auto data = std::make_shared<IcebergDeleteData>();

  if (metadata_ioctx == nullptr) {
    throw std::invalid_argument(
      "[iceberg] read_iceberg_delete_data: metadata_ioctx is null; caller must provide a "
      "sirius_ioctx (this entry point does not implement a kvikio fallback).");
  }

  // Errors propagate rather than degrading to empty delete data: empty is indistinguishable
  // from a table that genuinely has no deletes, so swallowing here would turn a failure to READ
  // the deletes into a decision to IGNORE them, and the scan would return deleted rows. The
  // caller decides what an unreadable manifest means; for the planner that is declining the GPU
  // scan and letting DuckDB read the table.
  auto discovery = discover_from_manifests(context, table_path, snapshot_id);

  bool has_pos_deletes =
    !discovery.positional_delete_files.empty() || !discovery.deletion_vector_entries.empty();
  bool has_eq_deletes = !discovery.equality_delete_entries.empty();

  if (!has_pos_deletes && !has_eq_deletes) {
    SIRIUS_LOG_DEBUG("[iceberg] No delete files for '{}'; append-only.", table_path);
    return data;
  }

  if (has_pos_deletes) {
    materialize_positional_deletes(*context.db, discovery, data->positional_deletes);
  }
  if (has_eq_deletes) {
    data->data_file_sequence_numbers = std::move(discovery.data_file_sequence_numbers);
    materialize_equality_deletes(discovery.equality_delete_entries, *metadata_ioctx, *data);
  }

  return data;
}

}  // namespace

std::shared_ptr<const IcebergDeleteData> read_iceberg_delete_data(
  duckdb::ClientContext& context,
  std::string const& table_path,
  sirius::io::sirius_ioctx* metadata_ioctx,
  std::optional<uint64_t> snapshot_id)
{
  // One query reads this three times: iceberg_scan is not serializable, so the plan is generated
  // twice, and the delete gate probes the manifests before either.
  //
  // Per query, deliberately. EqualityDeleteGroups hold a GPU key table and a prebuilt hash join,
  // so a longer-lived entry would pin GPU memory; and within one query the three passes read the
  // same snapshot by construction, so a hit is always right.
  //
  // Do NOT widen this by resolving "latest" and keying on the resolved id: with snapshot_id
  // unset, iceberg_metadata() reads current-snapshot-id from the table metadata, and resolving
  // latest independently disagrees with that after a rollback — filing one snapshot's deletes
  // under another's key.
  std::string key;
  try {
    key = std::to_string(context.ActiveTransaction().global_transaction_id) + "|" + table_path +
          "|" + (snapshot_id.has_value() ? std::to_string(*snapshot_id) : "current");
  } catch (...) {
    // No usable transaction identity: skip the cache rather than key it ambiguously.
    return read_iceberg_delete_data_uncached(context, table_path, metadata_ioctx, snapshot_id);
  }

  {
    std::lock_guard lk{g_delete_data_cache_mtx};
    if (auto it = g_delete_data_cache.find(key); it != g_delete_data_cache.end()) {
      SIRIUS_LOG_DEBUG("[iceberg] delete-data cache hit for '{}'", table_path);
      return it->second;
    }
  }

  // Built outside the lock: this reads manifests and delete files and can throw. Errors must
  // propagate (never cached, never softened into "no deletes"), and a concurrent duplicate
  // build is wasteful but harmless, whereas holding the lock across the read would serialize
  // planning across every iceberg scan in the process.
  auto data = read_iceberg_delete_data_uncached(context, table_path, metadata_ioctx, snapshot_id);

  std::lock_guard lk{g_delete_data_cache_mtx};
  auto [it, inserted] = g_delete_data_cache.emplace(key, std::move(data));
  return it->second;
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
