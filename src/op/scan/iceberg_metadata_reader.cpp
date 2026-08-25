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
#include <io/uri_parser.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_metadata_connection.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/puffin_reader.hpp>
#include <sirius_context.hpp>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
  int64_t record_count{-1};           // deleted positions the manifest claims (V3, -1 if absent)

  /// Requires file_format already lowercased by the reader. Format alone: an entry that IS a
  /// deletion vector but describes itself incompletely must be rejected, not reclassified as
  /// something other than a deletion vector and skipped.
  [[nodiscard]] bool is_deletion_vector() const { return file_format == "puffin"; }

  /// Whether the manifest gave this vector a locatable blob.
  [[nodiscard]] bool has_complete_descriptor() const
  {
    return content_offset >= 0 && content_size_in_bytes > 0;
  }

  /// Whether this vector's decode is bounded. `record_count` is a required manifest field: absent,
  /// both cardinality cross-checks compare against nothing and the Roaring expansion is unbounded.
  [[nodiscard]] bool has_decodable_record_count() const
  {
    return record_count >= 0 && record_count <= kMaxDeletionVectorPositions;
  }
};

struct IcebergManifestDiscovery {
  std::vector<std::string> positional_delete_files;
  std::vector<IcebergDeleteFileEntry> equality_delete_entries;
  std::vector<IcebergDeleteFileEntry> deletion_vector_entries;
  /// From the data manifests; equality deletes need them to test applicability.
  std::unordered_map<std::string, int64_t> data_file_manifest_sequence_numbers;
};

std::string escape_sql_string(std::string const& s)
{
  std::string out = s;
  for (std::string::size_type pos = 0; (pos = out.find('\'', pos)) != std::string::npos; pos += 2) {
    out.replace(pos, 1, "''");
  }
  return out;
}

/// Reads one manifest's live POSITION_DELETES entries. @p conn must already be bracketed as an
/// internal query. `lower()` is load-bearing: Iceberg writes "PUFFIN" but is_deletion_vector()
/// tests "puffin", and without the fold the table reads as having no deletes.
///
/// `status <> 2` drops DELETED entries. A manifest is a log, not a picture of the present: an
/// entry retired by a later commit stays listed, and a compaction that REPLACES a deletion vector
/// leaves the old one at status 2 beside its status 1 replacement. Both describe the same
/// referenced_data_file, so materialize_positional_deletes() would file them under one key and
/// the survivor would be whichever the manifest happens to list last -- an order Iceberg gives no
/// precedence meaning. Reading the retired one restores rows the table deleted.
std::vector<IcebergDeleteFileEntry> read_deletion_vectors_from_manifest(
  duckdb::Connection& conn, std::string const& manifest_path)
{
  auto result = conn.Query(
    "SELECT data_file.file_path, lower(data_file.file_format), "
    "COALESCE(data_file.referenced_data_file, ''), "
    "COALESCE(data_file.content_offset, -1), "
    "COALESCE(data_file.content_size_in_bytes, -1), "
    "COALESCE(data_file.record_count, -1) "
    "FROM read_avro('" +
    escape_sql_string(manifest_path) +
    "') "
    "WHERE data_file.content = 1 AND status <> 2");

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
      entry.record_count          = chunk->GetValue(5, i).GetValue<int64_t>();

      // Every entry here is live. A Puffin one the manifest failed to locate cannot be skipped:
      // skipping drops its deletes, and dropped deletes are returned rows.
      if (entry.is_deletion_vector() && !entry.has_complete_descriptor()) {
        throw std::runtime_error(
          "[iceberg] Manifest '" + manifest_path + "' lists a live deletion vector at '" +
          entry.file_path + "' with no usable blob descriptor (content_offset=" +
          std::to_string(entry.content_offset) +
          ", content_size_in_bytes=" + std::to_string(entry.content_size_in_bytes) +
          "); its deletes cannot be read and ignoring it would return deleted rows");
      }
      if (entry.is_deletion_vector() && entry.referenced_data_file.empty()) {
        throw std::runtime_error("[iceberg] Manifest '" + manifest_path +
                                 "' lists a live deletion vector at '" + entry.file_path +
                                 "' with no referenced_data_file, so the data file its deletes "
                                 "apply to is unknown");
      }
      // record_count is what bounds the Roaring expansion, and it is a required field. Refuse
      // here rather than at decode time so the verdict is reached while the plan can still
      // decline. See kMaxDeletionVectorPositions for why the bound cannot come from the table.
      if (entry.is_deletion_vector() && !entry.has_decodable_record_count()) {
        throw std::runtime_error(
          "[iceberg] Manifest '" + manifest_path + "' lists a live deletion vector at '" +
          entry.file_path + "' with record_count=" + std::to_string(entry.record_count) +
          ", which is either absent (leaving its decoded size unchecked and unbounded) or above "
          "the " +
          std::to_string(kMaxDeletionVectorPositions) +
          " positions this reader will materialize while planning");
      }
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

  // See iceberg_metadata_connection for why this is a connection, why it cannot observe a
  // different snapshot than the bind did, and why it mirrors settings instead of forcing them.
  iceberg_metadata_connection metadata_conn(context);
  auto& conn = metadata_conn.get();

  std::string query =
    "SELECT content, file_path, manifest_sequence_number, file_format, manifest_path "
    "FROM iceberg_metadata('" +
    escape_sql_string(table_path) + "'";
  if (snapshot_id.has_value()) {
    query += ", snapshot_from_id = " + std::to_string(snapshot_id.value());
  }
  // status is the entry's liveness (ADDED / EXISTING / DELETED); a manifest keeps listing entries
  // that later commits retired. Without this filter a delete file that a compaction replaced is
  // read back as live and removes rows the current snapshot keeps.
  //
  // ⚠️ status and content BOTH use the string "EXISTING" for unrelated things: status EXISTING
  // means "carried over from an earlier snapshot", content EXISTING means "this is a DATA file"
  // (which is what kExisting below tests). Do not merge these two tests.
  query += ") WHERE status <> 'DELETED'";

  auto meta_result = conn.Query(query);
  if (!meta_result || meta_result->HasError()) {
    // Empty would read as "this table has no delete files" and return deleted rows.
    throw std::runtime_error("[iceberg] iceberg_metadata() failed for '" + table_path +
                             "': " + (meta_result ? meta_result->GetError() : "null result"));
  }

  // Values of iceberg_metadata()'s `content` column -- what KIND of file an entry describes.
  //
  // ⚠️ `content` and `status` both use the string "EXISTING" for unrelated things. On `status`
  // (which the query above filters on) it means "carried over from an earlier snapshot". Here it
  // means "this is a DATA file" -- DuckDB's name for what the Iceberg spec calls DATA. Reading
  // one as the other silently reclassifies every data file as a delete file, or vice versa.
  static constexpr auto kPositionDeletes = "POSITION_DELETES";
  static constexpr auto kEqualityDeletes = "EQUALITY_DELETES";
  static constexpr auto kContentDataFile = "EXISTING";
  static constexpr auto kFormatPuffin    = "PUFFIN";

  // iceberg_metadata() returns one PUFFIN row per deletion vector, but read_avro returns ALL of
  // a manifest's vectors at once — so a manifest must be expanded exactly once. Keyed on the
  // manifest rather than the row: N vectors in one manifest would otherwise yield N² entries,
  // and materialize_positional_deletes opens the Puffin file once per entry.
  std::unordered_set<std::string> expanded_manifests;

  // Every PUFFIN row iceberg_metadata() reported, as (manifest, vector path). The two queries read
  // the same manifests by different routes; if the second one comes back without a vector the
  // first one saw, marking the manifest expanded discards that entry's deletes in silence -- a
  // fail-OPEN, and deleted rows come back. Checked after the loop because one manifest is expanded
  // once no matter how many of its rows the discovery query returns.
  //
  // Keyed on the MANIFEST too, not the vector path alone: one Puffin file can be referenced from
  // more than one manifest, and a path-only key would let a manifest that expanded correctly cover
  // for one that did not.
  std::set<std::pair<std::string, std::string>> puffin_rows_from_metadata;
  std::set<std::pair<std::string, std::string>> expanded_vectors;

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
          puffin_rows_from_metadata.emplace(manifest_path, sirius::io::strip_file_scheme(filepath));
          if (expanded_manifests.insert(manifest_path).second) {
            for (auto& dv : read_deletion_vectors_from_manifest(conn, manifest_path)) {
              if (dv.is_deletion_vector()) {
                expanded_vectors.emplace(manifest_path,
                                         sirius::io::strip_file_scheme(dv.file_path));
                result.deletion_vector_entries.push_back(std::move(dv));
              }
            }
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
      } else if (content == kContentDataFile) {
        result.data_file_manifest_sequence_numbers[filepath] = seq;
      } else {
        // Refuse rather than skip. "EXISTING" is DuckDB's name for the spec's DATA, so if a
        // future iceberg extension corrects it, this branch is the difference between the scan
        // declining and it quietly collecting NO data-file sequence numbers -- which is what
        // decides whether an equality delete applies to a file. Silently ignoring an unknown
        // content kind would also drop a delete class Iceberg adds later.
        throw std::runtime_error(
          "[iceberg] iceberg_metadata() returned an unrecognized content kind '" + content +
          "' for '" + filepath +
          "'; this scan path knows only POSITION_DELETES, EQUALITY_DELETES and EXISTING (data), "
          "and guessing which one it resembles would risk applying or skipping deletes wrongly");
      }
    }
  }

  // Hold the read_avro pass to what iceberg_metadata() reported. Under-delivery is the dangerous
  // direction and the only one checked: a vector the discovery query saw but the manifest pass did
  // not return (it read a different content kind, the entry was not spelled as PUFFIN, or the
  // manifest came back empty) would otherwise be dropped with no error. Over-delivery is safe --
  // read_avro returns a whole manifest at once, so it legitimately sees vectors whose discovery
  // rows are still ahead in the result.
  for (auto const& [manifest_path, vector_path] : puffin_rows_from_metadata) {
    if (expanded_vectors.count({manifest_path, vector_path}) == 0) {
      throw std::runtime_error(
        "[iceberg] iceberg_metadata() reports a live deletion vector at '" + vector_path +
        "' in manifest '" + manifest_path + "' for table '" + table_path +
        "', but re-reading that manifest did not return the vector; the two metadata passes "
        "disagree about which deletes are live, and proceeding would drop this vector's deleted "
        "rows back into the result");
    }
  }

  SIRIUS_LOG_INFO(
    "[iceberg] '{}': {} positional-delete, {} equality-delete, "
    "{} deletion-vector, {} data file(s).",
    table_path,
    result.positional_delete_files.size(),
    result.equality_delete_entries.size(),
    result.deletion_vector_entries.size(),
    result.data_file_manifest_sequence_numbers.size());

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

/// Merges V2 positional deletes and V3 deletion vectors into one per-data-file map.
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
    // Which data files a DV has already claimed. Superseding a POSITIONAL entry in out_map is the
    // spec'd behaviour above; a second DV claiming the same data file is not, and assigning it
    // would make the result depend on manifest order, which carries no precedence meaning.
    std::unordered_set<std::string> claimed_by_dv;

    for (auto const& dv_entry : files.deletion_vector_entries) {
      SIRIUS_LOG_DEBUG("[iceberg] Reading deletion vector for data file '{}' from '{}'",
                       dv_entry.referenced_data_file,
                       dv_entry.file_path);
      if (!claimed_by_dv.insert(dv_entry.referenced_data_file).second) {
        throw std::runtime_error(
          "[iceberg] Two live deletion vectors both claim data file '" +
          dv_entry.referenced_data_file + "' (the second is in '" + dv_entry.file_path +
          "'); which one applies would depend on manifest order, which Iceberg does not define");
      }

      auto positions =
        read_deletion_vector({.puffin_path           = dv_entry.file_path,
                              .content_offset        = dv_entry.content_offset,
                              .content_size_in_bytes = dv_entry.content_size_in_bytes,
                              .referenced_data_file  = dv_entry.referenced_data_file,
                              .record_count          = dv_entry.record_count});

      // The footer check above ties the blob to this entry; this ties the decoded bitmap to the
      // count both of them declare, which is the one thing the footer cannot vouch for.
      if (dv_entry.record_count >= 0 &&
          dv_entry.record_count != static_cast<int64_t>(positions.size())) {
        throw std::runtime_error(
          "[iceberg] Deletion vector for data file '" + dv_entry.referenced_data_file + "' in '" +
          dv_entry.file_path + "' decoded " + std::to_string(positions.size()) +
          " positions, but its manifest entry records " + std::to_string(dv_entry.record_count) +
          "; the blob at offset " + std::to_string(dv_entry.content_offset) +
          " is not the one this entry describes");
      }

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
    std::vector<std::unique_ptr<cudf::table>> tables;
    int64_t sequence_number;
  };
  std::vector<FileGroup> groups;
  // Indexed rather than scanned: every commit can write a delete file at a NEW sequence number,
  // so the group count grows with the file count and a linear search per file would compare
  // O(files²) name vectors.
  std::unordered_map<std::string, std::size_t> group_index;

  for (auto const& eq_entry : eq_entries) {
    SIRIUS_LOG_DEBUG("[iceberg] Reading equality-delete file: {} (seq={})",
                     eq_entry.file_path,
                     eq_entry.sequence_number);
    auto read_result = read_equality_delete_file(eq_entry.file_path, ioctx);

    std::string key = std::to_string(eq_entry.sequence_number);
    for (auto const& name : read_result.col_names) {
      key += '\0';  // not a legal column-name character, so the join is unambiguous
      key += name;
    }

    auto const [it, inserted] = group_index.emplace(std::move(key), groups.size());
    if (inserted) {
      groups.push_back(
        {read_result.col_names, read_result.field_ids, {}, eq_entry.sequence_number});
    }
    groups[it->second].tables.push_back(std::move(read_result.tbl));
  }

  for (auto& g : groups) {
    if (g.tables.empty()) continue;
    std::vector<cudf::table_view> views;
    views.reserve(g.tables.size());
    for (auto const& table : g.tables) {
      views.push_back(table->view());
    }
    auto group = build_equality_group(std::move(g.key_names), std::move(g.key_field_ids), views);
    group.sequence_number = g.sequence_number;
    data.equality_delete_groups.push_back(std::move(group));
  }

  SIRIUS_LOG_INFO("[iceberg] Built {} equality-delete group(s).",
                  data.equality_delete_groups.size());
}

}  // anonymous namespace

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

namespace {

/// Per-query memo; the wrapper below explains the key and the scope.
std::mutex g_delete_data_cache_mtx;
std::unordered_map<std::string, std::shared_ptr<const IcebergDeleteData>> g_delete_data_cache;

/// See the header.
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

/// The single switch that turns the equality-delete route on. Two defects keep it false; both are
/// named at the refusal site below.
constexpr bool kEqualityDeleteRouteImplementedToSpec = false;

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
    // The plan gate declines these tables, so reaching here means it has a hole. Two pieces of
    // this machinery do not implement the spec yet and both fail SILENTLY:
    //
    //  1. applies_to compares the MANIFEST's sequence number, not the entry's inherited data
    //     sequence number, so after a manifest rewrite it skips deletes that apply and applies
    //     deletes to data that post-dates them.
    //  2. Keys are every column of the delete file, not the entry's `equality_ids`. A Flink
    //     upsert writer persists the whole row with `equality_ids=[pk]`, so matching on all
    //     columns deletes nothing whose non-key columns have since changed.
    //
    // Fix both and add tests that reach the route before flipping the flag.
    if (!kEqualityDeleteRouteImplementedToSpec) {
      throw std::runtime_error(
        "[iceberg] '" + table_path + "' has " +
        std::to_string(discovery.equality_delete_entries.size()) +
        " live equality-delete entr(ies); the GPU path does not implement equality deletes to "
        "spec (manifest-level sequence numbers, and keys taken from every column rather than the "
        "entry's equality_ids), so this table must be read by DuckDB");
    }
    data->data_file_manifest_sequence_numbers =
      std::move(discovery.data_file_manifest_sequence_numbers);
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
  //
  // Leaf names are not unique -- `a.id` and `b.id` both spell `id` here -- so last-writer-wins
  // would hand one field's id to the other. An ambiguous name is dropped; the caller already
  // reads a missing entry as "no field id for this column".
  std::unordered_set<std::string> ambiguous;
  for (size_t i = 1; i < file_meta.schema.size(); ++i) {
    auto const& elem = file_meta.schema[i];
    if (elem.num_children != 0 || !elem.field_id.has_value()) { continue; }
    auto const [it, inserted] = result.emplace(elem.name, elem.field_id.value());
    if (!inserted && it->second != elem.field_id.value()) { ambiguous.insert(elem.name); }
  }
  for (auto const& name : ambiguous) {
    SIRIUS_LOG_DEBUG(
      "[iceberg] leaf column name '{}' occurs at more than one field id in this parquet schema; "
      "dropping it rather than guessing which field the delete file means",
      name);
    result.erase(name);
  }

  return result;
}

}  // namespace sirius::op::scan
