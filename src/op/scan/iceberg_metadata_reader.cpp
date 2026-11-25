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
#include "io/s3/duckdb_secret_config.hpp"
#include "yyjson.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/types/value_map.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/query_result_stream.hpp>
#include <duckdb/transaction/meta_transaction.hpp>
#include <log/logging.hpp>
#include <memory/device_transfer.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
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
#include <span>
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

inventory_result read_delete_inventory(
  duckdb::ClientContext& context,
  std::string const& table_path,
  std::optional<int64_t> snapshot_id,
  std::string_view injection,
  std::string_view expected_table_uuid,
  std::function<std::string(duckdb::ClientContext&)> const& unconfirmed_table)
{
  auto counters = iceberg_counters(context);
  iceberg_metadata_connection metadata_conn(context);
  auto& conn         = metadata_conn.get();
  std::string source = "iceberg_metadata('" + escape_sql_string(table_path) + "'";
  if (snapshot_id.has_value()) {
    source += ", snapshot_from_id = " + std::to_string(snapshot_id.value());
  }
  bool const extension_checks_uuid =
    !expected_table_uuid.empty() && iceberg_metadata_accepts_expected_table_uuid(context);
  if (extension_checks_uuid) {
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
  if (unconfirmed_table && !extension_checks_uuid) {
    // Returned, not thrown: an exception from the callback would invalidate this connection's
    // transaction, which the inventory query below runs in.
    std::string reason;
    conn.read_catalog([&](duckdb::ClientContext& inner) { reason = unconfirmed_table(inner); });
    if (!reason.empty()) throw inventory_query_error(reason, false, /*unconfirmed=*/true);
  }
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
  result.inventory->table = table_path;
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

namespace {

/// Whether a manifest's `partition-spec` partitions: it has a field whose transform is not `void`.
bool spec_partitions(std::string const& spec_json)
{
  using namespace duckdb_yyjson;
  std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
    yyjson_read(spec_json.data(), spec_json.size(), 0), yyjson_doc_free);
  auto* fields = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
  if (!yyjson_is_arr(fields)) throw std::invalid_argument("partition-spec is not a JSON array");
  std::size_t index, count;
  yyjson_val* field;
  yyjson_arr_foreach(fields, index, count, field)
  {
    auto const* transform = yyjson_get_str(yyjson_obj_get(field, "transform"));
    if (!transform) throw std::invalid_argument("partition-spec field has no transform");
    if (std::string_view(transform) != "void") return true;
  }
  return false;
}

/// What a manifest records about one key column of one file.
struct key_bounds {
  std::string lower, upper;  // Serialized bounds; empty when not recorded.
  bool no_nulls = false;     // The null count is recorded as zero.
};

/// A serialized bound, or empty when the manifest records none.
std::string bound_bytes(duckdb::Value const& value)
{
  return value.IsNull() || value.type().id() != duckdb::LogicalTypeId::BLOB
           ? std::string()
           : duckdb::StringValue::Get(value);
}

/// What a manifest entry's `data_file` records about field @p id. The three maps are optional
/// fields, so they are looked up by name: a manifest without them records nothing.
key_bounds key_bounds_of(duckdb::Value const& data_file, int32_t id)
{
  auto const entry = [&](char const* name) {
    if (data_file.IsNull() || data_file.type().id() != duckdb::LogicalTypeId::STRUCT)
      return duckdb::Value();
    auto const& members = duckdb::StructValue::GetChildren(data_file);
    for (duckdb::idx_t member = 0; member < members.size(); ++member) {
      auto const& map = members[member];
      if (duckdb::StructType::GetChildName(data_file.type(), member) != name || map.IsNull() ||
          map.type().id() != duckdb::LogicalTypeId::MAP)
        continue;
      for (auto const& pair : duckdb::MapValue::GetChildren(map)) {
        auto const& fields = duckdb::StructValue::GetChildren(pair);
        if (!fields[0].IsNull() && fields[0].GetValue<int32_t>() == id) return fields[1];
      }
    }
    return duckdb::Value();
  };
  auto const nulls = entry("null_value_counts");
  return {bound_bytes(entry("lower_bounds")),
          bound_bytes(entry("upper_bounds")),
          !nulls.IsNull() && nulls.GetValue<int64_t>() == 0};
}

/// Takes one data file's bounds; false to stop reading.
using key_bounds_sink = std::function<bool(std::string, std::vector<key_bounds>)>;

/// Hands @p sink what the live entries of @p manifests record for @p ids, one entry per ID.
/// False when the manifests cannot be read or @p sink stops: bounds are optional and so is
/// pruning by them, so that is not an error. The query names no optional field, which a
/// manifest may not have, and is streamed, so one chunk of entries is held at a time. It can
/// still fail, and a failed query aborts the connection's transaction: nothing may follow it.
bool read_key_bounds(iceberg_metadata_connection& connection,
                     std::set<std::string_view> const& manifests,
                     std::vector<int32_t> const& ids,
                     key_bounds_sink const& sink)
{
  std::string sql = "SELECT data_file.file_path, data_file FROM read_avro([";
  for (auto const manifest : manifests)
    sql += (sql.back() == '[' ? "'" : ", '") + escape_sql_string(std::string(manifest)) + "'";
  sql += "]) WHERE status <> 2";
  try {
    auto result = connection.get().Submit(sql);
    if (!result || result->HasError()) return false;
    duckdb::QueryResultStream<> stream(std::move(result));
    while (auto chunk = stream.Fetch()) {
      if (chunk->size() == 0) break;
      for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
        auto const data_file = chunk->GetValue(1, row);
        std::vector<key_bounds> keys;
        keys.reserve(ids.size());
        for (auto const id : ids)
          keys.push_back(key_bounds_of(data_file, id));
        if (!sink(chunk->GetValue(0, row).ToString(), std::move(keys))) return false;
      }
    }
    if (stream.HasError()) return false;
  } catch (std::exception const&) {
    return false;
  }
  return true;
}

/// A bound of an integral key as a manifest serializes it: little-endian, in the width of the
/// type that wrote it. Nullopt for any other length.
std::optional<int64_t> integral_bound(duckdb::LogicalTypeId type, std::string_view bytes)
{
  uint64_t raw = 0;
  for (auto i = bytes.size(); i-- > 0;)
    raw = raw << 8 | static_cast<uint8_t>(bytes[i]);
  switch (type) {
    case duckdb::LogicalTypeId::BOOLEAN:
      return bytes.size() == 1 ? std::optional<int64_t>(raw != 0) : std::nullopt;
    case duckdb::LogicalTypeId::BIGINT:
      // A file written before the key was promoted from int records 4-byte bounds.
      if (bytes.size() == 8) return static_cast<int64_t>(raw);
      [[fallthrough]];
    case duckdb::LogicalTypeId::INTEGER:
    case duckdb::LogicalTypeId::DATE:
      return bytes.size() == 4 ? std::optional<int64_t>(static_cast<int32_t>(raw)) : std::nullopt;
    case duckdb::LogicalTypeId::TIMESTAMP:
    case duckdb::LogicalTypeId::TIMESTAMP_NS:
      return bytes.size() == 8 ? std::optional<int64_t>(static_cast<int64_t>(raw)) : std::nullopt;
    default: return std::nullopt;
  }
}

}  // namespace

bool equality_key_ranges_disjoint(duckdb::LogicalType const& type,
                                  std::string_view a_lower,
                                  std::string_view a_upper,
                                  std::string_view b_lower,
                                  std::string_view b_upper)
{
  if (a_lower.empty() || a_upper.empty() || b_lower.empty() || b_upper.empty()) return false;
  if (type.id() == duckdb::LogicalTypeId::VARCHAR) {
    for (auto const bound : {a_lower, a_upper, b_lower, b_upper})
      if (!std::all_of(bound.begin(), bound.end(), [](unsigned char c) { return c < 0x80; }))
        return false;
    return a_upper < b_lower || b_upper < a_lower;
  }
  auto const a_low  = integral_bound(type.id(), a_lower);
  auto const a_high = integral_bound(type.id(), a_upper);
  auto const b_low  = integral_bound(type.id(), b_lower);
  auto const b_high = integral_bound(type.id(), b_upper);
  return a_low && a_high && b_low && b_high && (*a_high < *b_low || *b_high < *a_low);
}

iceberg_equality_plan read_equality_plan(duckdb::ClientContext& context,
                                         iceberg_delete_inventory const& inventory,
                                         std::size_t max_files_per_data_file,
                                         equality_key_types const& key_types,
                                         std::size_t max_key_bounds_bytes)
{
  auto const fail = [](std::string const& what) {
    return equality_plan_error("[iceberg] equality deletes: " + what);
  };

  struct listed {
    int64_t sequence;
    std::string_view manifest;
    bool read = false;
  };
  std::unordered_map<std::string_view, listed> data, deletes;
  for (auto const& entry : inventory.entries) {
    auto* files = entry.content == "DATA"               ? &data
                  : entry.content == "EQUALITY_DELETES" ? &deletes
                                                        : nullptr;
    if (!files) continue;
    // Without its inherited sequence number a file cannot be ordered against the others.
    if (entry.data_sequence_number < 0)
      throw fail("'" + entry.file_path + "' has no data sequence number");
    if (!files->emplace(entry.file_path, listed{entry.data_sequence_number, entry.manifest_path})
           .second)
      throw fail("the inventory lists '" + entry.file_path + "' twice");
  }

  iceberg_metadata_connection metadata_conn(context);
  // One manifest's live entries of one content kind, with the spec its header names. Kind 0 is
  // not tested: a data manifest holds data files only, and has no `content` field when format
  // version 1 wrote it. The inventory supplies the manifest paths, so the table path is not
  // resolved again.
  auto const read_manifest = [&](std::string_view manifest, std::string const& columns, int kind) {
    auto const path = "'" + escape_sql_string(std::string(manifest)) + "'";
    auto result     = metadata_conn.Query(
      "SELECT data_file.file_path, data_file.\"partition\", "
          "(SELECT value FROM avro_metadata(" +
      path +
      ") WHERE key = 'partition-spec-id'), "
          "(SELECT value FROM avro_metadata(" +
      path + ") WHERE key = 'partition-spec')" + columns + " FROM read_avro(" + path + ") WHERE " +
      (kind == 0 ? "" : "data_file.\"content\" = " + std::to_string(kind) + " AND ") +
      "status <> 2");
    if (!result || result->HasError())
      throw fail("manifest '" + std::string(manifest) +
                 "' could not be read: " + (result ? result->GetError() : "null result"));
    return result;
  };
  // Marks @p path as read from @p manifest, or refuses an entry the inventory does not list there.
  auto const claim = [&](std::unordered_map<std::string_view, listed>& files,
                         std::string const& path,
                         std::string_view manifest) -> std::pair<std::string_view, listed const&> {
    auto const found = files.find(path);
    if (found == files.end() || found->second.manifest != manifest || found->second.read)
      throw fail("manifest '" + std::string(manifest) + "' lists '" + path +
                 "' as live, which the inventory does not list there exactly once");
    found->second.read = true;
    return {found->first, found->second};
  };
  auto const require_all_read = [&](std::unordered_map<std::string_view, listed> const& files) {
    for (auto const& [path, entry] : files)
      if (!entry.read)
        throw fail("the inventory lists '" + std::string(path) + "' in manifest '" +
                   std::string(entry.manifest) + "', which does not list it as live");
  };
  auto const spec_id_of = [&](duckdb::Value const& value, std::string_view manifest) {
    try {
      if (!value.IsNull()) return value.GetValue<int32_t>();
    } catch (std::exception const&) {
    }
    throw fail("manifest '" + std::string(manifest) + "' does not name its partition spec");
  };

  struct scoped_delete {
    uint32_t file;
    int64_t sequence;
  };
  std::vector<iceberg_equality_delete_file> files;
  // Deletes stored under a spec that does not partition apply to every data file. The others
  // apply within their partition: the same spec and the same partition tuple.
  std::vector<scoped_delete> everywhere;
  std::map<int32_t, duckdb::value_map_t<std::vector<scoped_delete>>> by_partition;
  std::map<int32_t, duckdb::LogicalType> tuple_type;

  std::set<std::string_view> delete_manifests;
  for (auto const& [path, entry] : deletes)
    delete_manifests.insert(entry.manifest);
  // What each delete file records about its keys, in `equality_ids` order; empty when the
  // caller compares no bounds.
  std::vector<std::vector<key_bounds>> delete_bounds;
  for (auto const manifest : delete_manifests) {
    // The bounds come with the entries, as the whole `data_file`: they are optional fields, and
    // naming a field a manifest does not have fails the query and aborts this connection's
    // transaction.
    auto result = read_manifest(manifest,
                                std::string(", data_file.equality_ids, data_file.record_count, "
                                            "data_file.file_size_in_bytes") +
                                  (key_types ? ", data_file" : ""),
                                2);
    while (true) {
      auto chunk = result->Fetch();
      if (!chunk || chunk->size() == 0) break;
      for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
        auto const [path, entry] = claim(deletes, chunk->GetValue(0, row).ToString(), manifest);
        iceberg_equality_delete_file file;
        file.file_path = std::string(path);
        auto const ids = chunk->GetValue(4, row);
        if (!ids.IsNull())
          for (auto const& id : duckdb::ListValue::GetChildren(ids)) {
            if (id.IsNull())
              throw fail("an equality_ids entry of '" + file.file_path + "' is NULL");
            file.equality_ids.push_back(id.GetValue<int32_t>());
          }
        if (file.equality_ids.empty())
          throw fail("equality delete '" + file.file_path + "' has no equality_ids");
        if (std::set<int32_t>(file.equality_ids.begin(), file.equality_ids.end()).size() !=
            file.equality_ids.size())
          throw fail("equality delete '" + file.file_path + "' repeats an equality id");
        auto const rows  = chunk->GetValue(5, row);
        auto const bytes = chunk->GetValue(6, row);
        if (rows.IsNull() || bytes.IsNull() || rows.GetValue<int64_t>() < 0 ||
            bytes.GetValue<int64_t>() < 0)
          throw fail("equality delete '" + file.file_path +
                     "' has no usable record count or file size");
        file.record_count       = rows.GetValue<int64_t>();
        file.file_size_in_bytes = bytes.GetValue<int64_t>();
        std::vector<key_bounds> keys;
        if (key_types) {
          try {
            auto const data_file = chunk->GetValue(7, row);
            for (auto const id : file.equality_ids)
              keys.push_back(key_bounds_of(data_file, id));
          } catch (std::exception const&) {
            keys.clear();  // Maps of an unexpected shape prove nothing.
          }
        }
        delete_bounds.push_back(std::move(keys));

        auto const spec_id = spec_id_of(chunk->GetValue(2, row), manifest);
        auto const spec    = chunk->GetValue(3, row);
        bool partitions    = false;
        try {
          if (spec.IsNull()) throw std::invalid_argument("no partition-spec");
          partitions = spec_partitions(spec.ToString());
        } catch (std::invalid_argument const& error) {
          throw fail("manifest '" + std::string(manifest) + "': " + error.what());
        }
        scoped_delete const scoped{static_cast<uint32_t>(files.size()), entry.sequence};
        if (!partitions) {
          everywhere.push_back(scoped);
        } else {
          auto tuple = chunk->GetValue(1, row);
          if (tuple.IsNull() || tuple.type().id() != duckdb::LogicalTypeId::STRUCT)
            throw fail("equality delete '" + file.file_path + "' has no partition tuple");
          // DuckDB's Iceberg reader takes -0.0 and +0.0 for one partition, while the spec compares
          // their bit patterns. A delete stored under either zero is left to DuckDB.
          for (auto const& value : duckdb::StructValue::GetChildren(tuple))
            if (!value.IsNull() && value.type().IsFloating() && value.GetValue<double>() == 0)
              throw fail("equality delete '" + file.file_path +
                         "' is stored under a floating-point partition value of zero");
          // Tuples of one spec are compared as values, which needs one type for all of them.
          if (auto const [type, fresh] = tuple_type.emplace(spec_id, tuple.type());
              !fresh && type->second != tuple.type())
            throw fail("partition tuples of spec " + std::to_string(spec_id) + " differ in type");
          by_partition[spec_id][std::move(tuple)].push_back(scoped);
        }
        files.push_back(std::move(file));
      }
    }
  }
  require_all_read(deletes);

  std::set<std::string_view> data_manifests;
  for (auto const& [path, entry] : data)
    data_manifests.insert(entry.manifest);

  // The keys whose type the caller compares, and the bounds the data files record for them in
  // `ranged_ids` order. The data manifests are read for them only when a delete file records a
  // usable range of such a key.
  std::vector<int32_t> ranged_ids;
  std::vector<duckdb::LogicalType> ranged_types;
  std::unordered_map<std::string, std::vector<key_bounds>> data_bounds;
  bool usable = false;
  if (key_types) {
    std::set<int32_t> ids;
    for (auto const& file : files)
      ids.insert(file.equality_ids.begin(), file.equality_ids.end());
    for (auto const id : ids) {
      if (auto type = key_types(id)) {
        ranged_ids.push_back(id);
        ranged_types.push_back(std::move(*type));
      }
    }
    for (std::size_t file = 0; file < files.size(); ++file)
      for (std::size_t key = 0; key < delete_bounds[file].size(); ++key)
        usable |=
          std::binary_search(ranged_ids.begin(), ranged_ids.end(), files[file].equality_ids[key]) &&
          !delete_bounds[file][key].lower.empty() && !delete_bounds[file][key].upper.empty();
  }
  // The bounds kept for the plan are host memory that grows with the data files. Above the
  // ceiling none are kept and nothing is pruned.
  std::size_t bounds_bytes = 0;
  auto const keep_bounds   = [&](std::string path, std::vector<key_bounds> keys) {
    bounds_bytes += path.size() + sizeof(key_bounds) * keys.size();
    for (auto const& key : keys)
      bounds_bytes += key.lower.size() + key.upper.size();
    if (bounds_bytes > max_key_bounds_bytes) return false;
    data_bounds.insert_or_assign(std::move(path), std::move(keys));
    return true;
  };

  // Data manifests are read for their partitions when a delete is partition-scoped, and for
  // their bounds when a delete file records a usable range: one read serves both.
  struct placement {
    int32_t spec_id;
    duckdb::Value tuple;
  };
  std::unordered_map<std::string_view, placement> placed;
  if (!by_partition.empty()) {
    for (auto const manifest : data_manifests) {
      auto result = read_manifest(manifest, usable ? ", data_file" : "", 0);
      while (true) {
        auto chunk = result->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
          auto const path = claim(data, chunk->GetValue(0, row).ToString(), manifest).first;
          placed.emplace(
            path,
            placement{spec_id_of(chunk->GetValue(2, row), manifest), chunk->GetValue(1, row)});
          if (!usable) continue;
          try {
            auto const data_file = chunk->GetValue(4, row);
            std::vector<key_bounds> keys;
            for (auto const id : ranged_ids)
              keys.push_back(key_bounds_of(data_file, id));
            usable = keep_bounds(std::string(path), std::move(keys));
          } catch (std::exception const&) {
            usable = false;  // Maps of an unexpected shape prove nothing.
          }
        }
      }
    }
    require_all_read(data);
  } else if (usable) {
    // The last query on this connection: it may fail.
    usable = read_key_bounds(metadata_conn, data_manifests, ranged_ids, keep_bounds);
  }
  if (!usable) data_bounds.clear();

  // Whether the bounds prove that delete file @p file matches no row of a data file. One key is
  // enough. A NULL key matches a NULL value and bounds leave NULLs out, so a key decides only
  // when one of the two files holds no NULL in it.
  auto const ruled_out = [&](uint32_t file, std::vector<key_bounds> const* data_keys) {
    auto const& delete_keys = delete_bounds[file];
    if (data_keys == nullptr) return false;
    for (std::size_t key = 0; key < delete_keys.size(); ++key) {
      auto const id     = files[file].equality_ids[key];
      auto const ranged = std::lower_bound(ranged_ids.begin(), ranged_ids.end(), id);
      if (ranged == ranged_ids.end() || *ranged != id) continue;
      auto const k         = static_cast<std::size_t>(ranged - ranged_ids.begin());
      auto const& deleted  = delete_keys[key];
      auto const& existing = (*data_keys)[k];
      if ((deleted.no_nulls || existing.no_nulls) &&
          equality_key_ranges_disjoint(
            ranged_types[k], deleted.lower, deleted.upper, existing.lower, existing.upper))
        return true;
    }
    return false;
  };

  auto const by_sequence = [](scoped_delete const& a, scoped_delete const& b) {
    return a.sequence < b.sequence;
  };
  std::sort(everywhere.begin(), everywhere.end(), by_sequence);
  for (auto& [spec_id, tuples] : by_partition)
    for (auto& [tuple, scope] : tuples)
      std::sort(scope.begin(), scope.end(), by_sequence);

  iceberg_equality_plan plan;
  std::vector<bool> applies(files.size(), false);
  // The data files each list of deletes is ordered against: (data sequence number, data file).
  using domain = std::vector<std::pair<int64_t, uint32_t>>;
  std::unordered_map<std::vector<scoped_delete> const*, domain> domains;
  uint32_t data_index = 0;
  for (auto const& [path, entry] : data) {
    std::vector<uint32_t> applicable;
    std::vector<key_bounds> const* data_keys = nullptr;
    if (!data_bounds.empty()) {
      if (auto const found = data_bounds.find(std::string(path)); found != data_bounds.end())
        data_keys = &found->second;
    }
    // Strictly newer deletes only: an upsert must not delete the row it writes.
    auto const add = [&](std::vector<scoped_delete> const& scope) {
      domains[&scope].emplace_back(entry.sequence, data_index);
      auto first = std::upper_bound(
        scope.begin(), scope.end(), entry.sequence, [](int64_t sequence, scoped_delete const& d) {
          return sequence < d.sequence;
        });
      for (; first != scope.end() && applicable.size() <= max_files_per_data_file; ++first)
        if (!ruled_out(first->file, data_keys)) applicable.push_back(first->file);
    };
    add(everywhere);
    if (auto const place = placed.find(path); place != placed.end()) {
      if (auto const spec = by_partition.find(place->second.spec_id); spec != by_partition.end()) {
        if (place->second.tuple.type() != tuple_type.at(spec->first))
          throw fail("partition tuples of spec " + std::to_string(spec->first) + " differ in type");
        if (auto const scope = spec->second.find(place->second.tuple); scope != spec->second.end())
          add(scope->second);
      }
    }
    std::sort(applicable.begin(), applicable.end());
    for (auto file : applicable)
      applies[file] = true;
    plan.by_data_file.emplace(std::string(path), std::move(applicable));
    ++data_index;
  }

  // Deletes share a scope when they apply to the same data files by sequence number and
  // partition, whatever the bounds ruled out. In one list those are the deletes that follow the
  // same number of its data files; across lists the sets are compared, found by a hash of the
  // set and then checked.
  struct scope_set {
    domain const* files;
    std::size_t count;
    uint32_t scope;
  };
  auto const same_files = [](domain const& a, domain const& b, std::size_t count) {
    if (&a == &b) return true;
    std::vector<uint32_t> in_a, in_b;
    for (std::size_t i = 0; i < count; ++i) {
      in_a.push_back(a[i].second);
      in_b.push_back(b[i].second);
    }
    std::sort(in_a.begin(), in_a.end());
    std::sort(in_b.begin(), in_b.end());
    return in_a == in_b;
  };
  std::map<std::pair<std::size_t, uint64_t>, scope_set> scope_of_set;
  uint32_t scopes = 0;
  for (auto& [deletes, data_files] : domains) {
    std::sort(data_files.begin(), data_files.end());
    // hashes[n] identifies the first n data files as a set.
    std::vector<uint64_t> hashes(data_files.size() + 1, 0);
    for (std::size_t i = 0; i < data_files.size(); ++i)
      hashes[i + 1] = hashes[i] + duckdb::Hash<uint64_t>(data_files[i].second);
    std::ptrdiff_t preceding = -1;
    uint32_t scope           = 0;
    for (auto const& scoped : *deletes) {
      auto const below =
        std::lower_bound(
          data_files.begin(), data_files.end(), std::pair{scoped.sequence, uint32_t{0}}) -
        data_files.begin();
      if (below != preceding) {
        auto const count = static_cast<std::size_t>(below);
        auto const known = scope_of_set.find({count, hashes[count]});
        if (known != scope_of_set.end() && same_files(*known->second.files, data_files, count)) {
          scope = known->second.scope;
        } else {
          scope = ++scopes;
          if (known == scope_of_set.end())
            scope_of_set.emplace(std::pair{count, hashes[count]},
                                 scope_set{&data_files, count, scope});
        }
      }
      preceding                = below;
      files[scoped.file].scope = scope;
    }
  }

  // A delete file that applies to nothing is neither charged nor opened.
  std::vector<uint32_t> renumbered(files.size(), 0);
  for (std::size_t file = 0; file < files.size(); ++file) {
    if (!applies[file]) continue;
    renumbered[file] = static_cast<uint32_t>(plan.files.size());
    plan.files.push_back(std::move(files[file]));
  }
  for (auto& [path, applicable] : plan.by_data_file)
    for (auto& file : applicable)
      file = renumbered[file];
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
      // Writers disagree on the case: DuckDB records "puffin".
      if (duckdb::StringUtil::CIEquals(file_format, kFormatPuffin)) {
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
  iceberg_equality_delete_file const& file,
  std::shared_ptr<cucascade::io::ioctx> const& ioctx,
  std::vector<int32_t> const& key_field_ids,
  iceberg_equality_plan const& plan,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const& delete_file_path = file.file_path;
  // Check the file against the manifest values charged during planning before decoding keys.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(sirius::io::open_datasource(ioctx, delete_file_path));
  if (file.file_size_in_bytes < 0 ||
      sources.front()->size() != static_cast<uint64_t>(file.file_size_in_bytes))
    throw std::runtime_error("[iceberg] equality-delete file '" + delete_file_path +
                             "' size differs from its manifest entry");
  auto footers = cudf::io::read_parquet_footers(sources);
  if (footers.size() != 1)
    throw std::runtime_error("[iceberg] Failed to read equality-delete footer: " +
                             delete_file_path);
  auto rows_remaining = file.record_count;
  for (auto const& group : footers.front().row_groups) {
    if (group.num_rows < 0 || group.num_rows > rows_remaining)
      throw std::runtime_error("[iceberg] equality-delete file '" + delete_file_path +
                               "' row count differs from its manifest entry");
    rows_remaining -= group.num_rows;
  }
  if (rows_remaining != 0 || footers.front().num_rows != file.record_count)
    throw std::runtime_error("[iceberg] equality-delete file '" + delete_file_path +
                             "' row count differs from its manifest entry");
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

  auto opts   = cudf::io::parquet_reader_options::builder().column_names(names).build();
  auto result = cudf::io::read_parquet(std::move(sources), std::move(footers), opts, stream, mr);
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
      keys[k] = cudf::cast(columns[found]->view(), *want, stream, mr);
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

/// Indexes a group's distinct @p keys for the probe.
EqualityDeleteGroup index_equality_group(std::vector<int32_t> key_field_ids,
                                         std::vector<std::size_t> key_primary_indices,
                                         std::unique_ptr<cudf::table> keys,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  // A NULL delete key matches a NULL data value -- `col IS NULL`, the opposite of a SQL join.
  auto hash_join = std::make_unique<cudf::distinct_hash_join>(
    keys->view(), cudf::null_equality::EQUAL, 0.5, stream, mr);
  stream.sync();

  EqualityDeleteGroup group;
  group.delete_table        = std::move(keys);
  group.key_field_ids       = std::move(key_field_ids);
  group.key_primary_indices = std::move(key_primary_indices);
  group.hash_join           = std::move(hash_join);
  return group;
}

/// Builds the delete groups the scan probes, once on each device of @p resources, and records
/// which of them each data file needs. The first device reads and decodes the delete files; the
/// others index a copy of its keys.
/// A group is the delete files that share their keys and their scope, so memory grows with the
/// delete rows, not with the number of distinct applicability sets, and key bounds only take
/// groups off a data file's list.
void materialize_equality_deletes(duckdb::ClientContext& context,
                                  iceberg_equality_plan const& plan,
                                  std::shared_ptr<cucascade::io::ioctx> const& ioctx,
                                  std::span<equality_build_resources const> resources,
                                  IcebergDeleteData& data)
{
  auto grouping           = group_equality_deletes(plan);
  auto const& group_files = grouping.files;

  SIRIUS_LOG_INFO("[iceberg] Loading equality deletes: {} file(s) in {} group(s).",
                  plan.files.size(),
                  group_files.size());
  // A scan's batches land on every device of @p resources, and each probes tables in its own
  // device's memory: whether another device's can be reached depends on the hardware.
  // Deletes that apply to no data file leave no group, and no replica to hold one.
  if (group_files.empty()) resources = {};
  data.equality_replicas.reserve(resources.size());
  for (auto const& target : resources) {
    data.equality_replicas.push_back({target.device, {}});
    data.equality_replicas.back().groups.resize(group_files.size());
  }
  for (uint32_t group_index = 0; group_index < group_files.size(); ++group_index) {
    // Canonical key order, as the grouping compared it.
    auto ids = plan.files[group_files[group_index].front()].equality_ids;
    std::sort(ids.begin(), ids.end());
    std::vector<std::size_t> primary;
    primary.reserve(ids.size());
    for (auto id : ids)
      primary.push_back(plan.keys.at(id).primary_index);
    auto const& origin = resources.front();
    rmm::cuda_set_device_raii origin_guard{origin.device};
    {
      std::vector<std::unique_ptr<cudf::table>> tables;
      std::vector<cudf::table_view> views;
      for (auto file : group_files[group_index]) {
        auto const counters = iceberg_counters(context);
        if (counters) ++counters->iceberg_delete_payload_loads;
        auto const& path = plan.files[file].file_path;
        if (counters && counters->track_units && counters->iceberg_equality_open_for_testing)
          counters->iceberg_equality_open_for_testing(path);
        auto file_ioctx = ioctx;
        if (sirius::io::s3::is_s3_path(path)) {
          file_ioctx = origin.remote_ioctx ? origin.remote_ioctx(path) : nullptr;
          // Never read as "no deletes": the scan would return the rows this file removes.
          if (!file_ioctx)
            throw std::runtime_error(
              "[iceberg] no S3 backend is available to read equality-delete file '" + path + "'");
        }
        tables.push_back(read_equality_delete_file(
          plan.files[file], file_ioctx, ids, plan, origin.stream, origin.mr));
        views.push_back(tables.back()->view());
      }
      data.equality_replicas.front().groups[group_index] =
        build_equality_group(ids, primary, views, origin.stream, origin.mr);
    }
    if (resources.size() == 1) continue;
    // The other devices get the group's keys in one buffer copied from the first, so the files
    // are read and decoded once.
    auto const packed =
      cudf::pack(data.equality_replicas.front().groups[group_index].delete_table->view(),
                 origin.stream,
                 origin.mr);
    origin.stream.sync();
    for (std::size_t device = 1; device < resources.size(); ++device) {
      auto const& target = resources[device];
      rmm::cuda_set_device_raii device_guard{target.device};
      std::unique_ptr<cudf::table> keys;
      {
        rmm::device_buffer copy{packed.gpu_data->size(), target.stream, target.mr};
        (void)sirius::memory::enqueue_device_copy(
          {static_cast<std::byte*>(copy.data()), copy.size()},
          target.device,
          target.stream,
          {static_cast<std::byte const*>(packed.gpu_data->data()), copy.size()},
          origin.device,
          origin.space,
          sirius::memory::source_already_complete{},
          target.space
            ? sirius::memory::sirius_memory_reservation_manager::host_staging_space(*target.space)
            : nullptr);
        // The copy reads the first device's buffer: finish it before either buffer can go.
        target.stream.sync();
        keys = std::make_unique<cudf::table>(
          cudf::unpack(packed.metadata->data(), static_cast<uint8_t const*>(copy.data())),
          target.stream,
          target.mr);
        // The keys are read out of the buffer before it goes, so that the hash table is not
        // built beside a second copy of them.
        target.stream.sync();
      }
      data.equality_replicas[device].groups[group_index] =
        index_equality_group(ids, primary, std::move(keys), target.stream, target.mr);
    }
  }

  data.equality_groups_by_data_file = std::move(grouping.by_data_file);
}

}  // anonymous namespace

equality_grouping group_equality_deletes(iceberg_equality_plan const& plan)
{
  equality_grouping grouping;
  std::map<std::pair<std::vector<int32_t>, uint32_t>, uint32_t> group_of;
  std::vector<uint32_t> group_of_file(plan.files.size(), 0);
  for (uint32_t file = 0; file < plan.files.size(); ++file) {
    // Canonical key order, so two files listing the same ids in a different order still merge.
    auto ids = plan.files[file].equality_ids;
    std::sort(ids.begin(), ids.end());
    auto const [slot, fresh] = group_of.emplace(std::pair{std::move(ids), plan.files[file].scope},
                                                static_cast<uint32_t>(grouping.files.size()));
    if (fresh) grouping.files.emplace_back();
    grouping.files[slot->second].push_back(file);
    group_of_file[file] = slot->second;
  }
  // One list object per distinct list of groups, not of files: key bounds leave data files of
  // one scope with different files of the same groups, and those still share a batch.
  std::map<equality_group_list, std::shared_ptr<equality_group_list const>> shared;
  for (auto const& [data_file, files] : plan.by_data_file) {
    equality_group_list groups;
    for (auto file : files)
      groups.push_back(group_of_file[file]);
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    auto& list = shared[groups];
    if (!list) list = std::make_shared<equality_group_list const>(std::move(groups));
    grouping.by_data_file.emplace(data_file, list);
  }
  return grouping;
}

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

IcebergDeleteData::~IcebergDeleteData()
{
  for (auto& replica : equality_replicas) {
    rmm::cuda_set_device_raii device_guard{replica.device};
    replica.groups.clear();
  }
}

void IcebergDeleteData::release_equality_replicas_outside(std::span<int const> devices) const
{
  auto const excluded = [&](equality_replica const& replica) {
    return std::find(devices.begin(), devices.end(), replica.device.value()) == devices.end();
  };
  // Emptied first, each with its device current: the erase moves kept replicas into their slots.
  for (auto& replica : equality_replicas) {
    if (!excluded(replica)) continue;
    rmm::cuda_set_device_raii device_guard{replica.device};
    replica.groups.clear();
  }
  std::erase_if(equality_replicas, excluded);
}

std::vector<EqualityDeleteGroup> const& IcebergDeleteData::equality_groups_on(
  rmm::cuda_device_id device) const
{
  for (auto const& replica : equality_replicas)
    if (replica.device == device) return replica.groups;
  throw std::invalid_argument("[iceberg] no equality-delete tables were built on GPU " +
                              std::to_string(device.value()));
}

EqualityDeleteGroup build_equality_group(std::vector<int32_t> key_field_ids,
                                         std::vector<std::size_t> key_primary_indices,
                                         std::vector<cudf::table_view> const& views,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  if (views.empty() || key_field_ids.empty() || key_field_ids.size() != key_primary_indices.size())
    throw std::invalid_argument("[iceberg] an equality-delete group needs files and keys");
  for (auto const& view : views)
    if (static_cast<std::size_t>(view.num_columns()) != key_field_ids.size())
      throw std::invalid_argument("[iceberg] an equality-delete table does not match its keys");

  auto all_rows = (views.size() == 1) ? std::make_unique<cudf::table>(views[0], stream, mr)
                                      : cudf::concatenate(views, stream, mr);

  std::vector<cudf::size_type> all_key_indices(static_cast<size_t>(all_rows->num_columns()));
  std::iota(all_key_indices.begin(), all_key_indices.end(), cudf::size_type{0});

  auto deduped = cudf::distinct(all_rows->view(),
                                all_key_indices,
                                cudf::duplicate_keep_option::KEEP_FIRST,
                                cudf::null_equality::EQUAL,
                                cudf::nan_equality::ALL_EQUAL,
                                stream,
                                mr);

  SIRIUS_LOG_INFO("[iceberg] Equality-delete group over {} key column(s): {} row(s).",
                  key_field_ids.size(),
                  deduped->num_rows());

  return index_equality_group(
    std::move(key_field_ids), std::move(key_primary_indices), std::move(deduped), stream, mr);
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
  iceberg_delete_discovery const& discovery,
  std::span<equality_build_resources const> equality_resources)
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
    if (equality_resources.empty())
      throw std::invalid_argument("[iceberg] '" + table_path +
                                  "' has live equality deletes but no stream and allocator to "
                                  "build their tables with");
    materialize_equality_deletes(
      context, *discovery.equality_plan, metadata_ioctx, equality_resources, *data);
  }

  return data;
}

}  // namespace

std::shared_ptr<const IcebergDeleteData> load_delete_payload(
  duckdb::ClientContext& context,
  std::string const& table_path,
  std::shared_ptr<cucascade::io::ioctx> const& metadata_ioctx,
  iceberg_delete_discovery const& discovery,
  std::vector<equality_build_resources> equality_resources)
{
  return read_iceberg_delete_data_uncached(
    context, table_path, metadata_ioctx, discovery, equality_resources);
}

}  // namespace sirius::op::scan
