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

#include "io/path_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/query_result_stream.hpp>
#include <duckdb/transaction/meta_transaction.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_metadata_connection.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/parquet_delete_reader.hpp>
#include <op/scan/puffin_reader.hpp>
#include <sirius_context.hpp>

#include <algorithm>
#include <atomic>
#include <deque>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

namespace {

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
  duckdb::SiriusContext::internal_connection& conn, std::string const& manifest_path)
{
  auto result = conn.Query(
    "SELECT data_file.file_path, lower(data_file.file_format), "
    "COALESCE(data_file.referenced_data_file, ''), "
    "COALESCE(data_file.content_offset, -1), "
    "COALESCE(data_file.content_size_in_bytes, -1), "
    "COALESCE(data_file.record_count, -1), "
    "COALESCE(data_file.file_size_in_bytes, -1) "
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
      entry.file_size_in_bytes    = chunk->GetValue(6, i).GetValue<int64_t>();

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

std::shared_ptr<physical_check_counters> iceberg_counters(duckdb::ClientContext& context)
{
  auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  return state ? state->physical_counters() : nullptr;
}

bool iceberg_metadata_accepts_expected_table_uuid(duckdb::ClientContext& context)
{
  duckdb::ExtensionLoader loader(duckdb::DatabaseInstance::GetDatabase(context), "iceberg");
  auto entry = loader.TryGetTableFunction(duckdb::Identifier("iceberg_metadata"));
  if (!entry) return false;
  duckdb::Identifier const option("expected_table_uuid");
  auto const& functions = entry->Cast<duckdb::TableFunctionCatalogEntry>().functions.functions;
  return std::ranges::any_of(functions, [&](auto const& function) {
    auto const& signature = function->GetSignature();
    auto const kwargs     = signature.GetTypedKwargs();
    return signature.GetParameterIndexByName(option).IsValid() || (kwargs && kwargs->Find(option));
  });
}

}  // namespace

void validate_deletion_vector_claims(iceberg_delete_discovery const& files)
{
  int64_t total = 0;
  for (auto const& dv : files.deletion_vector_entries) {
    if (dv.record_count < 0 || dv.record_count > kMaxDeletionVectorPositionsPerStatement - total)
      throw std::runtime_error(
        "[iceberg] The live deletion vectors of this scan declare more than the " +
        std::to_string(kMaxDeletionVectorPositionsPerStatement) +
        " deleted positions this reader will retain while planning (reached at '" + dv.file_path +
        "'); the scan declines rather than sizing a plan-time allocation "
        "from what the table wrote");
    total += dv.record_count;
  }
  std::set<std::string_view> claimed;
  for (auto const& dv : files.deletion_vector_entries) {
    if (!claimed.insert(dv.referenced_data_file).second)
      throw std::runtime_error(
        "[iceberg] Two live deletion vectors both claim data file '" + dv.referenced_data_file +
        "' (the second is in '" + dv.file_path +
        "'); which one applies would depend on manifest order, which Iceberg does not define");
  }
}

bool IcebergDeleteFileEntry::has_decodable_record_count() const
{
  return record_count >= 0 && record_count <= kMaxDeletionVectorPositions;
}

std::unordered_set<std::string> iceberg_files_without_partition_values(
  duckdb::ClientContext& context, iceberg_delete_inventory const& inventory)
{
  iceberg_metadata_connection connection(context);
  std::unordered_map<std::string, bool> manifests;
  std::unordered_set<std::string> files, unproved;
  for (auto const& entry : inventory.entries) {
    if (entry.content != "DATA") continue;
    auto [it, fresh] = manifests.emplace(entry.manifest_path, false);
    if (fresh) {
      auto result = connection.Query("SELECT data_file.partition FROM read_avro('" +
                                     escape_sql_string(entry.manifest_path) + "') LIMIT 0");
      // The pinned Avro reader represents an empty record as NULL (older builds
      // used INTEGER). Nonempty partition records are STRUCTs. Unknown/
      // unavailable schemas grant no proof.
      it->second = result && !result->HasError() && result->GetTypes().size() == 1 &&
                   (result->GetTypes()[0] == duckdb::LogicalType::SQLNULL ||
                    result->GetTypes()[0] == duckdb::LogicalType::INTEGER);
    }
    auto const path = sirius::io::strip_file_scheme(entry.file_path);
    if (it->second)
      files.insert(path);
    else
      unproved.insert(path);
  }
  for (auto const& path : unproved)
    files.erase(path);
  return files;
}

inventory_result read_delete_inventory(duckdb::ClientContext& context,
                                       std::string const& table_path,
                                       std::optional<int64_t> snapshot_id,
                                       std::string_view injection,
                                       std::string_view expected_table_uuid)
{
  auto counters = iceberg_counters(context);
  iceberg_metadata_connection metadata_conn(context);
  auto& conn         = metadata_conn.get();
  std::string source = "iceberg_metadata('" + escape_sql_string(table_path) + "'";
  if (snapshot_id.has_value()) {
    source += ", snapshot_from_id = " + std::to_string(snapshot_id.value());
  }
  if (!expected_table_uuid.empty() && iceberg_metadata_accepts_expected_table_uuid(context)) {
    source +=
      ", expected_table_uuid = '" + escape_sql_string(std::string(expected_table_uuid)) + "'";
  }
  source += ")";
  // status is the entry's liveness (ADDED / EXISTING / DELETED); a manifest keeps listing entries
  // that later commits retired. Without this filter a delete file that a compaction replaced is
  // read back as live and removes rows the current snapshot keeps.
  auto const live = source + " WHERE status <> 'DELETED'";

  // Injection is latched by the planning attempt, after the original scan bound successfully.
  if (injection == "missing") throw inventory_query_error("", true);
  if (injection == "fail") throw inventory_query_error("injected inventory query failure", false);
  // One query supplies the equality count and the inventory the equality plan is checked against.
  if (counters) ++counters->iceberg_manifest_walks;
  auto meta_result = conn.Submit(
    "SELECT content::VARCHAR, file_path, manifest_sequence_number, file_format, manifest_path, "
    "data_sequence_number "
    "FROM " +
    live);
  if (!meta_result) throw inventory_query_error("", true);
  if (meta_result->HasError()) throw inventory_query_error(meta_result->GetError(), false);
  inventory_result result{iceberg_delete_inventory{}, 0};
  std::vector<duckdb::unique_ptr<duckdb::DataChunk>> chunks;
  uint64_t bytes      = sizeof(iceberg_delete_inventory);
  uint64_t peak_bytes = bytes;
  try {
    duckdb::QueryResultStream<> stream(std::move(meta_result));
    while (auto chunk = stream.Fetch()) {
      if (chunk->size() == 0) break;
      duckdb::UnifiedVectorFormat content;
      chunk->data[0].ToUnifiedFormat(chunk->size(), content);
      auto const* values = content.GetData<duckdb::string_t>();
      for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
        auto const row = content.sel->get_index(i);
        if (content.validity.RowIsValid(row) &&
            std::string_view(values[row].GetDataUnsafe(), values[row].GetSize()) ==
              "EQUALITY_DELETES")
          ++result.equality_count;
      }
      peak_bytes = std::max(peak_bytes, bytes + chunk->GetAllocationSize());
      bytes += chunk->GetAllocationSize();
      chunks.push_back(std::move(chunk));
    }
    if (stream.HasError()) throw std::runtime_error(stream.GetError());
  } catch (std::exception const& error) {
    throw inventory_query_error(error.what(), false);
  }
  for (auto& chunk : chunks) {
    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      // A NULL inherited sequence number cannot order a delete against a data file. Keep it
      // distinguishable from every real one; read_equality_plan() refuses it.
      auto const data_seq = chunk->GetValue(5, i);
      iceberg_delete_inventory::entry row{
        chunk->GetValue(0, i).ToString(),
        chunk->GetValue(1, i).ToString(),
        chunk->GetValue(2, i).GetValue<int64_t>(),
        chunk->GetValue(3, i).ToString(),
        chunk->GetValue(4, i).ToString(),
        data_seq.IsNull() ? int64_t{-1} : data_seq.GetValue<int64_t>()};
      bytes += row.content.capacity() + row.file_path.capacity() + row.file_format.capacity() +
               row.manifest_path.capacity() + 4;
      auto old_capacity = result.inventory->entries.capacity();
      result.inventory->entries.push_back(std::move(row));
      auto capacity = result.inventory->entries.capacity();
      peak_bytes    = std::max(peak_bytes,
                            bytes + (capacity + (capacity != old_capacity ? old_capacity : 0)) *
                                      sizeof(iceberg_delete_inventory::entry));
    }
    bytes -= chunk->GetAllocationSize();
    chunk.reset();
  }
  if (counters) {
    auto peak = counters->iceberg_inventory_bytes_peak.load();
    while (peak < peak_bytes &&
           !counters->iceberg_inventory_bytes_peak.compare_exchange_weak(peak, peak_bytes)) {}
  }
  return result;
}

static_assert(kMaxEqualityDeleteRowsPerFile == kMaxDeletionVectorPositions,
              "the per-file equality-delete ceiling is the per-vector deletion-vector ceiling");

bool charge_equality_deletes(iceberg_equality_plan const& plan,
                             equality_delete_charge& total,
                             int64_t statement_row_limit)
{
  for (auto const& file : plan.files) {
    // Compared by subtraction: the terms come from manifests, and a sum could overflow.
    if (file.record_count < 0 || file.file_size_in_bytes < 0 ||
        file.record_count > kMaxEqualityDeleteRowsPerFile ||
        file.record_count > statement_row_limit - total.rows ||
        file.file_size_in_bytes > kMaxEqualityDeleteBytesPerStatement - total.bytes)
      return false;
    total.rows += file.record_count;
    total.bytes += file.file_size_in_bytes;
  }
  return true;
}

iceberg_equality_plan read_equality_plan(duckdb::ClientContext& context,
                                         std::string const& table_path,
                                         int64_t snapshot_id,
                                         iceberg_delete_inventory const& inventory)
{
  // What the inventory -- a separate walk of the same manifests -- says about each file.
  std::unordered_map<std::string_view, int64_t> data_sequence, delete_sequence;
  for (auto const& entry : inventory.entries) {
    if (entry.content == "DATA") {
      data_sequence.emplace(entry.file_path, entry.data_sequence_number);
    } else if (entry.content == "EQUALITY_DELETES") {
      delete_sequence.emplace(entry.file_path, entry.data_sequence_number);
    }
  }

  iceberg_metadata_connection metadata_conn(context);
  // The struct is repacked so its field order is this function's, not the extension's.
  auto result = metadata_conn.Query(
    "SELECT file_path, snapshot_id, "
    "[{'p': d.file_path, 'c': d.\"content\", 'ids': d.equality_ids, 'n': d.record_count, "
    "'b': d.file_size_in_bytes} for d in delete_files] "
    "FROM iceberg_scan_plan('" +
    escape_sql_string(table_path) + "', snapshot_from_id = " + std::to_string(snapshot_id) + ")");
  if (!result || result->HasError()) {
    throw equality_plan_error("[iceberg] iceberg_scan_plan failed for '" + table_path +
                              "': " + (result ? result->GetError() : "null result"));
  }

  auto const fail = [&](std::string const& what) -> equality_plan_error {
    return equality_plan_error("[iceberg] equality plan for '" + table_path + "': " + what);
  };

  iceberg_equality_plan plan;
  std::unordered_map<std::string, uint32_t> delete_index;
  while (true) {
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0) break;
    for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
      auto const path_value     = chunk->GetValue(0, row);
      auto const snapshot_value = chunk->GetValue(1, row);
      auto const deletes_value  = chunk->GetValue(2, row);
      if (path_value.IsNull() || snapshot_value.IsNull())
        throw fail("the scan plan returned a task without a file path or snapshot id");
      auto data_file = path_value.ToString();
      if (snapshot_value.GetValue<int64_t>() != snapshot_id)
        throw fail("the scan plan describes snapshot " +
                   std::to_string(snapshot_value.GetValue<int64_t>()) + ", not the requested " +
                   std::to_string(snapshot_id));
      auto const data_seq = data_sequence.find(data_file);
      if (data_seq == data_sequence.end())
        throw fail("the scan plan lists data file '" + data_file +
                   "', which the inventory does not list as a live DATA entry");
      if (data_seq->second < 0)
        throw fail("data file '" + data_file + "' has no data sequence number");

      std::vector<uint32_t> applicable;
      if (!deletes_value.IsNull()) {
        for (auto const& delete_value : duckdb::ListValue::GetChildren(deletes_value)) {
          if (delete_value.IsNull()) throw fail("the scan plan returned a NULL delete file");
          auto const& fields = duckdb::StructValue::GetChildren(delete_value);
          if (fields.size() != 5 || fields[0].IsNull() || fields[1].IsNull())
            throw fail("the scan plan returned a delete file without a path or content kind");
          auto const content = fields[1].GetValue<int32_t>();
          // Position deletes and deletion vectors are discovered and applied by position.
          if (content == 1) continue;
          if (content != 2)
            throw fail("the scan plan returned delete content kind " + std::to_string(content));
          auto delete_file      = fields[0].ToString();
          auto const delete_seq = delete_sequence.find(delete_file);
          if (delete_seq == delete_sequence.end())
            throw fail("the scan plan applies equality delete '" + delete_file +
                       "', which the inventory does not list as a live EQUALITY_DELETES entry");
          // The one rule that separates an upsert from deleting the row just written.
          if (delete_seq->second < 0 || data_seq->second >= delete_seq->second)
            throw fail("the scan plan applies equality delete '" + delete_file +
                       "' (data sequence number " + std::to_string(delete_seq->second) +
                       ") to data file '" + data_file + "' (data sequence number " +
                       std::to_string(data_seq->second) +
                       "); an equality delete applies only to strictly older data");
          auto [slot, fresh] =
            delete_index.emplace(delete_file, static_cast<uint32_t>(plan.files.size()));
          if (fresh) {
            iceberg_equality_delete_file file;
            file.file_path = std::move(delete_file);
            if (!fields[2].IsNull())
              for (auto const& id : duckdb::ListValue::GetChildren(fields[2])) {
                if (id.IsNull()) throw fail("an equality_ids entry is NULL");
                file.equality_ids.push_back(id.GetValue<int32_t>());
              }
            if (file.equality_ids.empty())
              throw fail("equality delete '" + file.file_path + "' has no equality_ids");
            if (std::set<int32_t>(file.equality_ids.begin(), file.equality_ids.end()).size() !=
                file.equality_ids.size())
              throw fail("equality delete '" + file.file_path + "' repeats an equality id");
            if (fields[3].IsNull() || fields[4].IsNull() || fields[3].GetValue<int64_t>() < 0 ||
                fields[4].GetValue<int64_t>() < 0)
              throw fail("equality delete '" + file.file_path +
                         "' has no usable record count or file size");
            file.record_count         = fields[3].GetValue<int64_t>();
            file.file_size_in_bytes   = fields[4].GetValue<int64_t>();
            file.data_sequence_number = delete_seq->second;
            plan.files.push_back(std::move(file));
          }
          applicable.push_back(slot->second);
        }
      }
      std::sort(applicable.begin(), applicable.end());
      if (std::adjacent_find(applicable.begin(), applicable.end()) != applicable.end())
        throw fail("data file '" + data_file + "' lists one equality delete twice");
      if (!plan.by_data_file.emplace(std::move(data_file), std::move(applicable)).second)
        throw fail("the scan plan lists a data file twice");
    }
  }
  // Under-delivery would leave a scanned file unplanned. Counts suffice: both sides are sets and
  // membership was checked above.
  if (plan.by_data_file.size() != data_sequence.size())
    throw fail("the scan plan lists " + std::to_string(plan.by_data_file.size()) +
               " data file(s) but the inventory lists " + std::to_string(data_sequence.size()));
  return plan;
}

iceberg_delete_discovery discover_from_manifests(duckdb::ClientContext& context,
                                                 std::string const& table_path,
                                                 iceberg_delete_inventory&& inventory)
{
  iceberg_delete_discovery result;
  result.equality_plan   = std::move(inventory.equality);
  std::size_t data_files = 0;

  // See iceberg_metadata_connection for why this is a connection, why it cannot observe a
  // different snapshot than the bind did, and why it mirrors settings instead of forcing them.
  iceberg_metadata_connection metadata_conn(context);
  auto& conn = metadata_conn.get();

  // Values of iceberg_metadata()'s `content` column -- what KIND of file an entry describes.
  static constexpr auto kPositionDeletes = "POSITION_DELETES";
  static constexpr auto kEqualityDeletes = "EQUALITY_DELETES";
  static constexpr auto kContentDataFile = "DATA";
  static constexpr auto kFormatPuffin    = "PUFFIN";

  // iceberg_metadata() returns one PUFFIN row per deletion vector, but read_avro returns ALL of
  // a manifest's vectors at once — so a manifest must be expanded exactly once. Keyed on the
  // manifest rather than the row: N vectors in one manifest would otherwise yield N² entries,
  // and materialize_positional_deletes opens the Puffin file once per entry.
  std::unordered_set<std::string> expanded_manifests;

  // Every PUFFIN row iceberg_metadata() reported, as (manifest, vector path) -> COUNT. The two
  // queries read the same manifests by different routes; if the second one comes back without a
  // vector the first one saw, marking the manifest expanded discards that entry's deletes in
  // silence -- a fail-OPEN, and deleted rows come back. Checked after the loop because one
  // manifest is expanded once no matter how many of its rows the discovery query returns.
  //
  // Keyed on the MANIFEST too, not the vector path alone: one Puffin file can be referenced from
  // more than one manifest, and a path-only key would let a manifest that expanded correctly cover
  // for one that did not.
  //
  // COUNTED, not merely present: one Puffin can hold SEVERAL deletion vectors, distinguished by
  // content_offset and referenced_data_file but sharing a path. Membership alone lets 2-reported
  // vs 1-expanded pass -- both sides hold the same single pair -- and the second vector's deleted
  // rows come back. `iceberg_v3_dv_misbound` has exactly that shape.
  std::map<std::pair<std::string, std::string>, size_t> puffin_rows_from_metadata;
  std::map<std::pair<std::string, std::string>, size_t> expanded_vectors;

  for (auto& row : inventory.entries) {
    auto const& content = row.content;
    auto filepath       = std::move(row.file_path);

    if (content == kPositionDeletes) {
      auto const& file_format = row.file_format;
      if (file_format == kFormatPuffin) {
        // iceberg_metadata() omits the V3 fields, so re-read the manifest via read_avro.
        auto const& manifest_path = row.manifest_path;
        ++puffin_rows_from_metadata[{manifest_path, sirius::io::strip_file_scheme(filepath)}];
        if (expanded_manifests.insert(manifest_path).second) {
          if (auto counters = iceberg_counters(context)) ++counters->iceberg_dv_manifest_reads;
          for (auto& dv : read_deletion_vectors_from_manifest(conn, manifest_path)) {
            if (dv.is_deletion_vector()) {
              ++expanded_vectors[{manifest_path, sirius::io::strip_file_scheme(dv.file_path)}];
              result.deletion_vector_entries.push_back(std::move(dv));
            }
          }
        }
      } else {
        result.positional_delete_files.push_back(std::move(filepath));
      }
    } else if (content == kEqualityDeletes) {
      IcebergDeleteFileEntry entry;
      entry.file_path = std::move(filepath);
      entry.content   = 2;
      result.equality_delete_entries.push_back(std::move(entry));
    } else if (content == kContentDataFile) {
      ++data_files;
    } else {
      // Unknown content could omit data-file sequence numbers or a new delete class.
      // Decline rather than risk applying or skipping deletes incorrectly.
      throw std::runtime_error(
        "[iceberg] iceberg_metadata() returned an unrecognized content kind '" + content +
        "' for '" + filepath +
        "'; this scan path knows only POSITION_DELETES, EQUALITY_DELETES and DATA, "
        "and guessing which one it resembles would risk applying or skipping deletes wrongly");
    }
  }

  // Hold the inventory and expansion to EXACT agreement, per (manifest, vector path). At the end
  // both directions are a reader disagreement: under-delivery drops a live vector's deletes, and
  // over-delivery APPLIES deletes the discovery pass never reported as live.
  auto const reconcile =
    [&](std::pair<std::string, std::string> const& key, size_t reported, size_t expanded) {
      if (reported == expanded) { return; }
      throw std::runtime_error(
        "[iceberg] The two metadata passes disagree about the deletion vector(s) at '" +
        key.second + "' in manifest '" + key.first + "' for table '" + table_path +
        "': iceberg_metadata() "
        "reports " +
        std::to_string(reported) + " live, re-reading the manifest returned " +
        std::to_string(expanded) +
        ". A Puffin may hold several vectors at one path, so this is a count and not a presence "
        "check; proceeding would " +
        (reported > expanded ? std::string("drop deleted rows back into the result")
                             : std::string("apply deletes that were never reported as live")));
    };
  for (auto const& [key, reported] : puffin_rows_from_metadata) {
    auto const it = expanded_vectors.find(key);
    reconcile(key, reported, it == expanded_vectors.end() ? 0 : it->second);
  }
  for (auto const& [key, expanded] : expanded_vectors) {
    if (puffin_rows_from_metadata.count(key) == 0) { reconcile(key, 0, expanded); }
  }

  SIRIUS_LOG_INFO(
    "[iceberg] '{}': {} positional-delete, {} equality-delete, "
    "{} deletion-vector, {} data file(s).",
    table_path,
    result.positional_delete_files.size(),
    result.equality_delete_entries.size(),
    result.deletion_vector_entries.size(),
    data_files);

  // The gate plans equality deletes before lowering. Entries without a plan mean a caller
  // skipped it, and loading them unplanned would apply them to every data file.
  if (!result.equality_delete_entries.empty() && !result.equality_plan) {
    throw std::runtime_error("[iceberg] '" + table_path + "' has " +
                             std::to_string(result.equality_delete_entries.size()) +
                             " live equality-delete entr(ies) but no equality plan reached "
                             "lowering; the plan gate must plan them or decline the scan");
  }

  return result;
}

namespace {

/// Appends one positional-delete file's records to @p out_map. Schema must be
/// { file_path VARCHAR, pos BIGINT }. CPU read: these files are tiny metadata.
void read_positional_delete_file(duckdb::ClientContext& context,
                                 std::string const& delete_file_path,
                                 std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  using position_map = std::unordered_map<std::string, std::vector<int64_t>>;
  std::deque<position_map> partials;
  read_parquet_delete_batches(
    context,
    duckdb::OpenFileInfo(delete_file_path),
    {},
    [](auto const& columns, uint64_t) {
      std::vector<uint64_t> projection;
      for (auto const& required : {std::string("file_path"), std::string("pos")}) {
        std::optional<uint64_t> found;
        for (uint64_t i = 0; i < columns.size(); ++i) {
          if (duckdb::StringUtil::CIEquals(columns[i].name, required)) {
            if (found) throw std::runtime_error("[iceberg] Ambiguous position-delete column");
            found = i;
          }
        }
        auto type = required == "pos" ? duckdb::LogicalType::BIGINT : duckdb::LogicalType::VARCHAR;
        if (!found || columns[*found].type != type)
          throw std::runtime_error(
            "[iceberg] Position deletes require file_path VARCHAR and pos BIGINT");
        projection.push_back(*found);
      }
      return projection;
    },
    [&] {
      auto& positions = partials.emplace_back();
      return [&positions](duckdb::DataChunk& chunk) {
        duckdb::UnifiedVectorFormat paths;
        duckdb::UnifiedVectorFormat offsets;
        chunk.data[0].ToUnifiedFormat(paths);
        chunk.data[1].ToUnifiedFormat(offsets);
        auto const* path_data = duckdb::UnifiedVectorFormat::GetData<duckdb::string_t>(paths);
        auto const* pos_data  = duckdb::UnifiedVectorFormat::GetData<int64_t>(offsets);
        std::vector<int64_t>* target = nullptr;
        duckdb::string_t target_key;
        for (duckdb::idx_t i = 0; i < chunk.size(); ++i) {
          auto const path_idx = paths.sel->get_index(i);
          auto const pos_idx  = offsets.sel->get_index(i);
          if (!paths.validity.RowIsValid(path_idx) || !offsets.validity.RowIsValid(pos_idx))
            throw std::runtime_error("[iceberg] Invalid null position delete");
          auto const& key = path_data[path_idx];
          if (target == nullptr || !(key == target_key)) {
            target     = &positions[key.GetString()];
            target_key = key;
          }
          // Preserve signed legacy positions; visibility filtering excludes out-of-range values.
          target->push_back(pos_data[pos_idx]);
        }
      };
    });
  std::unordered_map<std::string_view, size_t> remaining;
  for (auto& partial : partials) {
    out_map.merge(partial);
    for (auto const& [path, positions] : partial)
      remaining[path] += positions.size();
  }
  for (auto& partial : partials) {
    for (auto& [path, positions] : partial) {
      auto& dest = out_map.at(path);
      auto total = remaining.extract(path);
      // For small appends, retain vector's amortized growth across delete files.
      if (!total.empty() && total.mapped() > dest.size())
        dest.reserve(dest.size() + total.mapped());
      dest.insert(dest.end(), positions.begin(), positions.end());
      std::vector<int64_t>().swap(positions);
    }
  }
}

/// Top-level leaf columns of a parquet file by Iceberg field ID. Nested keys are declined by the
/// gate, so nested leaves are skipped.
std::unordered_map<int32_t, std::string> top_level_leaves_by_field_id(
  cudf::io::parquet::FileMetaData const& file_meta)
{
  std::unordered_map<int32_t, std::string> result;
  std::unordered_set<int32_t> repeated;
  if (file_meta.schema.empty()) return result;
  std::size_t next = 1;
  for (int32_t child = 0; child < file_meta.schema[0].num_children; ++child) {
    if (next >= file_meta.schema.size())
      throw std::runtime_error("[iceberg] equality-delete file has a truncated parquet schema");
    auto const& element = file_meta.schema[next];
    if (element.num_children == 0 && element.field_id.has_value() &&
        !result.emplace(element.field_id.value(), element.name).second)
      repeated.insert(element.field_id.value());
    // Skip this element's whole subtree.
    std::size_t pending = 1;
    while (pending > 0) {
      if (next >= file_meta.schema.size())
        throw std::runtime_error("[iceberg] equality-delete file has a truncated parquet schema");
      pending += static_cast<std::size_t>(std::max(file_meta.schema[next].num_children, 0));
      --pending;
      ++next;
    }
  }
  for (auto id : repeated)
    result.erase(id);
  return result;
}

/// Reads the key columns of one equality-delete file onto the device, in @p key_field_ids order
/// and normalized to the bound key types. No other column is read.
std::unique_ptr<cudf::table> read_equality_delete_file(
  std::string const& delete_file_path,
  std::shared_ptr<cucascade::io::ioctx> const& ioctx,
  std::vector<int32_t> const& key_field_ids,
  iceberg_equality_plan const& plan)
{
  ::cuda::stream_ref stream = cudf::get_default_stream();

  // One datasource for both passes: its io_object opens 2 fds, so reusing avoids a reopen.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(sirius::io::open_datasource(ioctx, delete_file_path));
  auto footers = cudf::io::read_parquet_footers(sources);
  if (footers.size() != 1)
    throw std::runtime_error("[iceberg] Failed to read equality-delete footer: " +
                             delete_file_path);
  auto const leaves = top_level_leaves_by_field_id(footers[0]);

  // Keys are found by field ID, never by name: a renamed key keeps its ID and changes its name.
  std::vector<std::string> names;
  names.reserve(key_field_ids.size());
  for (auto id : key_field_ids) {
    auto const leaf = leaves.find(id);
    if (leaf == leaves.end())
      throw std::runtime_error("[iceberg] equality-delete file '" + delete_file_path +
                               "' has no top-level column with field id " + std::to_string(id) +
                               ", which its manifest entry lists in equality_ids");
    names.push_back(leaf->second);
  }
  if (std::set<std::string>(names.begin(), names.end()).size() != names.size())
    throw std::runtime_error("[iceberg] equality-delete file '" + delete_file_path +
                             "' maps two key field ids to one column name");

  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{sources[0].get()})
                .column_names(names)
                .build();
  auto result = cudf::io::read_parquet(opts, stream);
  if (!result.tbl || result.metadata.schema_info.size() != names.size() ||
      static_cast<std::size_t>(result.tbl->num_columns()) != names.size())
    throw std::runtime_error("[iceberg] Failed to read equality-delete file: " + delete_file_path);

  // Placed by returned name rather than trusting the reader to keep the requested order.
  auto columns = result.tbl->release();
  std::vector<std::unique_ptr<cudf::column>> keys(names.size());
  for (std::size_t k = 0; k < names.size(); ++k) {
    std::size_t found = names.size();
    for (std::size_t c = 0; c < names.size(); ++c)
      if (result.metadata.schema_info[c].name == names[k]) found = c;
    if (found == names.size() || !columns[found])
      throw std::runtime_error("[iceberg] equality-delete file '" + delete_file_path +
                               "' did not return key column '" + names[k] + "'");
    auto const& bound = plan.keys.at(key_field_ids[k]).type;
    auto const want   = equality_key_cudf_type(bound);
    auto const have   = columns[found]->type();
    if (want && have == *want) {
      keys[k] = std::move(columns[found]);
    } else if (want && have.id() == cudf::type_id::INT32 && want->id() == cudf::type_id::INT64) {
      // The one promotion Iceberg allows for an integer key: the delete file predates it.
      keys[k] = cudf::cast(columns[found]->view(), *want, stream);
    } else {
      throw std::runtime_error("[iceberg] equality-delete file '" + delete_file_path +
                               "' stores key field id " + std::to_string(key_field_ids[k]) +
                               " as cuDF type " + std::to_string(static_cast<int>(have.id())) +
                               ", which cannot be compared with the bound " + bound.ToString() +
                               " column");
    }
  }
  stream.sync();
  auto table = std::make_unique<cudf::table>(std::move(keys));
  SIRIUS_LOG_INFO("[iceberg] read equality-delete: path={} rows={} key_cols={}",
                  delete_file_path,
                  table->num_rows(),
                  table->num_columns());
  return table;
}

/// Merges V2 positional deletes and V3 deletion vectors into one per-data-file map.
void materialize_positional_deletes(duckdb::ClientContext& context,
                                    iceberg_delete_discovery const& files,
                                    std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  if (!files.positional_delete_files.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} positional-delete file(s).",
                    files.positional_delete_files.size());
    for (auto const& del_path : files.positional_delete_files) {
      if (auto counters = iceberg_counters(context)) ++counters->iceberg_delete_payload_loads;
      SIRIUS_LOG_DEBUG("[iceberg] Reading positional-delete file: {}", del_path);
      read_positional_delete_file(context, del_path, out_map);
    }
  }

  // Per spec, deletion vectors supersede positional deletes for the same data file.
  if (!files.deletion_vector_entries.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} deletion vector(s).",
                    files.deletion_vector_entries.size());

    // Charge the WHOLE statement's deletion-vector payload before opening the first Puffin. The
    // per-vector ceiling admits each of N vectors on its own, but all N are retained together
    // until execution ends, so N admissible vectors still add up to an inadmissible total. Summed
    // over live ENTRIES, not Puffin paths -- one Puffin may hold several vectors -- and the
    // addition is overflow-checked because every term comes from a manifest.
    int64_t total_positions = 0;
    for (auto const& dv_entry : files.deletion_vector_entries) {
      if (dv_entry.record_count < 0 ||
          dv_entry.record_count > kMaxDeletionVectorPositionsPerStatement - total_positions) {
        throw std::runtime_error(
          "[iceberg] The live deletion vectors of this scan declare more than the " +
          std::to_string(kMaxDeletionVectorPositionsPerStatement) +
          " deleted positions this reader will retain while planning (reached at '" +
          dv_entry.file_path +
          "'); the scan declines rather than sizing a plan-time allocation "
          "from what the table wrote");
      }
      total_positions += dv_entry.record_count;
    }
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

      auto counters = iceberg_counters(context);
      if (counters) ++counters->iceberg_delete_payload_loads;
      if (counters && counters->track_units && counters->iceberg_dv_phase_for_testing)
        counters->iceberg_dv_phase_for_testing(dv_entry.file_path, true);
      auto positions =
        read_deletion_vector({.puffin_path           = dv_entry.file_path,
                              .content_offset        = dv_entry.content_offset,
                              .content_size_in_bytes = dv_entry.content_size_in_bytes,
                              .referenced_data_file  = dv_entry.referenced_data_file,
                              .record_count          = dv_entry.record_count});
      if (counters && counters->track_units && counters->iceberg_dv_phase_for_testing)
        counters->iceberg_dv_phase_for_testing(dv_entry.file_path, false);

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

/// Builds the delete groups the scan probes and records which of them each data file needs.
/// A group is the delete files that share their keys and apply to exactly the same data files,
/// so memory grows with the delete rows, not with the number of distinct applicability sets.
void materialize_equality_deletes(duckdb::ClientContext& context,
                                  iceberg_equality_plan const& plan,
                                  std::shared_ptr<cucascade::io::ioctx> const& ioctx,
                                  IcebergDeleteData& data)
{
  // One class per distinct applicability list.
  std::map<std::vector<uint32_t>, uint32_t> class_of_list;
  std::vector<std::vector<uint32_t> const*> classes;
  for (auto const& [data_file, files] : plan.by_data_file) {
    auto const [slot, fresh] =
      class_of_list.emplace(files, static_cast<uint32_t>(class_of_list.size()));
    if (fresh) classes.push_back(&slot->first);
  }

  // Each delete file's signature: the classes that use it. Unused files are never opened.
  std::vector<std::vector<uint32_t>> signature(plan.files.size());
  for (auto const& [files, id] : class_of_list)
    for (auto file : files)
      signature.at(file).push_back(id);
  for (auto& classes_using : signature)
    std::sort(classes_using.begin(), classes_using.end());

  std::map<std::pair<std::vector<int32_t>, std::vector<uint32_t>>, uint32_t> group_of;
  std::vector<std::vector<uint32_t>> group_files;
  std::vector<uint32_t> group_of_file(plan.files.size(), 0);
  for (uint32_t file = 0; file < plan.files.size(); ++file) {
    if (signature[file].empty()) continue;
    // Canonical key order, so two files listing the same ids in a different order still merge.
    auto ids = plan.files[file].equality_ids;
    std::sort(ids.begin(), ids.end());
    auto const [slot, fresh] = group_of.emplace(std::pair{std::move(ids), signature[file]},
                                                static_cast<uint32_t>(group_files.size()));
    if (fresh) group_files.emplace_back();
    group_files[slot->second].push_back(file);
    group_of_file[file] = slot->second;
  }

  SIRIUS_LOG_INFO("[iceberg] Loading equality deletes: {} file(s) in {} group(s).",
                  plan.files.size(),
                  group_files.size());
  data.equality_delete_groups.resize(group_files.size());
  for (auto const& [key, group_index] : group_of) {
    auto const& ids = key.first;
    std::vector<std::size_t> primary;
    primary.reserve(ids.size());
    for (auto id : ids)
      primary.push_back(plan.keys.at(id).primary_index);
    std::vector<std::unique_ptr<cudf::table>> tables;
    std::vector<cudf::table_view> views;
    for (auto file : group_files[group_index]) {
      if (auto counters = iceberg_counters(context)) ++counters->iceberg_delete_payload_loads;
      tables.push_back(read_equality_delete_file(plan.files[file].file_path, ioctx, ids, plan));
      views.push_back(tables.back()->view());
    }
    data.equality_delete_groups[group_index] = build_equality_group(ids, std::move(primary), views);
  }

  std::vector<std::shared_ptr<equality_group_list const>> list_of_class(classes.size());
  for (auto const& [files, id] : class_of_list) {
    equality_group_list groups;
    for (auto file : files)
      groups.push_back(group_of_file[file]);
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    list_of_class[id] = std::make_shared<equality_group_list const>(std::move(groups));
  }
  for (auto const& [data_file, files] : plan.by_data_file)
    data.equality_groups_by_data_file.emplace(data_file, list_of_class[class_of_list.at(files)]);
}

}  // anonymous namespace

std::optional<cudf::data_type> equality_key_cudf_type(duckdb::LogicalType const& type)
{
  switch (type.id()) {
    case duckdb::LogicalTypeId::BOOLEAN:
    case duckdb::LogicalTypeId::INTEGER:
    case duckdb::LogicalTypeId::BIGINT:
    case duckdb::LogicalTypeId::DATE:
    case duckdb::LogicalTypeId::TIMESTAMP:
    case duckdb::LogicalTypeId::TIMESTAMP_NS:
    case duckdb::LogicalTypeId::VARCHAR: return sirius::get_cudf_type(type);
    default: return std::nullopt;
  }
}

EqualityDeleteGroup build_equality_group(std::vector<int32_t> key_field_ids,
                                         std::vector<std::size_t> key_primary_indices,
                                         std::vector<cudf::table_view> const& views)
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  if (views.empty() || key_field_ids.empty() || key_field_ids.size() != key_primary_indices.size())
    throw std::invalid_argument("[iceberg] an equality-delete group needs files and keys");
  for (auto const& view : views)
    if (static_cast<std::size_t>(view.num_columns()) != key_field_ids.size())
      throw std::invalid_argument("[iceberg] an equality-delete table does not match its keys");

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

  SIRIUS_LOG_INFO("[iceberg] Equality-delete group over {} key column(s): {} row(s).",
                  key_field_ids.size(),
                  deduped->num_rows());

  // A NULL delete key matches a NULL data value -- `col IS NULL`, the opposite of a SQL join.
  auto hash_join = std::make_unique<cudf::distinct_hash_join>(
    deduped->view(), cudf::null_equality::EQUAL, 0.5, stream);
  stream.sync();

  EqualityDeleteGroup group;
  group.delete_table        = std::move(deduped);
  group.key_field_ids       = std::move(key_field_ids);
  group.key_primary_indices = std::move(key_primary_indices);
  group.hash_join           = std::move(hash_join);
  return group;
}

namespace {

/// See the header.
std::atomic<uint64_t> g_uncached_read_count{0};

}  // namespace

uint64_t iceberg_delete_data_uncached_read_count()
{
  return g_uncached_read_count.load(std::memory_order_relaxed);
}

namespace {

std::shared_ptr<const IcebergDeleteData> read_iceberg_delete_data_uncached(
  duckdb::ClientContext& context,
  std::string const& table_path,
  std::shared_ptr<cucascade::io::ioctx> const& metadata_ioctx,
  iceberg_delete_discovery const& discovery)
{
  g_uncached_read_count.fetch_add(1, std::memory_order_relaxed);

  auto data = std::make_shared<IcebergDeleteData>();

  if (metadata_ioctx == nullptr) {
    throw std::invalid_argument(
      "[iceberg] read_iceberg_delete_data: metadata_ioctx is null; caller must provide an "
      "ioctx.");
  }

  // Errors propagate rather than degrading to empty delete data: empty is indistinguishable
  // from a table that genuinely has no deletes, so swallowing here would turn a failure to READ
  // the deletes into a decision to IGNORE them, and the scan would return deleted rows. The
  // caller decides what an unreadable manifest means; for the planner that is declining the GPU
  // scan and letting DuckDB read the table.

  bool has_pos_deletes =
    !discovery.positional_delete_files.empty() || !discovery.deletion_vector_entries.empty();
  bool has_eq_deletes = !discovery.equality_delete_entries.empty();

  if (!has_pos_deletes && !has_eq_deletes) {
    SIRIUS_LOG_DEBUG("[iceberg] No delete files for '{}'; append-only.", table_path);
    return data;
  }

  if (has_pos_deletes) {
    materialize_positional_deletes(context, discovery, data->positional_deletes);
  }
  if (has_eq_deletes) {
    if (!discovery.equality_plan)
      throw std::runtime_error("[iceberg] '" + table_path +
                               "' has live equality deletes but no equality plan; the plan gate "
                               "must plan them or decline the scan");
    materialize_equality_deletes(context, *discovery.equality_plan, metadata_ioctx, *data);
  }

  return data;
}

}  // namespace

std::shared_ptr<const IcebergDeleteData> load_delete_payload(
  duckdb::ClientContext& context,
  std::string const& table_path,
  std::shared_ptr<cucascade::io::ioctx> const& metadata_ioctx,
  iceberg_delete_discovery const& discovery)
{
  return read_iceberg_delete_data_uncached(context, table_path, metadata_ioctx, discovery);
}

}  // namespace sirius::op::scan
