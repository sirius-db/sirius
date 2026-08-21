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

#include "cudf/cudf_utils.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/segment/uncompressed.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/string_stats.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/numeric_narrowing.hpp"
#include "helper/type_conversions.hpp"
#include "io/uri_parser.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace sirius::planner {

namespace {

/// Descends into children so a struct's fields count toward the id space they occupy.
void collect_field_ids(std::vector<duckdb::MultiFileColumnDefinition> const& columns,
                       std::vector<int32_t>& out)
{
  for (auto const& column : columns) {
    if (!column.identifier.IsNull() &&
        column.identifier.type().id() == duckdb::LogicalTypeId::INTEGER) {
      out.push_back(column.identifier.GetValue<int32_t>());
    }
    collect_field_ids(column.children, out);
  }
}

/**
 * @brief Refuse tables whose Iceberg field-id space has a gap, meaning a column was dropped.
 *
 * Iceberg never reuses a field id, so a column dropped and re-added under the same name gets a
 * NEW id and must read NULL in older data files. This path resolves columns by NAME, so it finds
 * the dropped column and returns data the table removed, with no error and no fallback.
 *
 * With no drops N fields occupy ids 1..N, so `max > count` means an id was retired. Reads only
 * bind data, opens no files.
 *
 * A PRE-FILTER, not the whole test: a plain ADDED column keeps the space contiguous and slips
 * through, but that case fails loudly instead of returning wrong rows. Resolving by field id is
 * the complete fix; DuckDB's MultiFileColumnMapper already implements it.
 */
std::optional<std::string> iceberg_retired_field_id_decline_reason(duckdb::LogicalGet& op)
{
  auto const* bind_data = dynamic_cast<duckdb::MultiFileBindData const*>(op.bind_data.get());
  if (bind_data == nullptr) { return std::nullopt; }

  // `reader_bind.schema` is the Iceberg schema and the only one carrying field ids; the generic
  // `columns` list may leave `identifier` null, so it is a fallback only.
  std::vector<int32_t> field_ids;
  collect_field_ids(
    bind_data->reader_bind.schema.empty() ? bind_data->columns : bind_data->reader_bind.schema,
    field_ids);
  if (field_ids.empty()) { return std::nullopt; }

  auto const max_field_id = *std::max_element(field_ids.begin(), field_ids.end());
  if (max_field_id <= static_cast<int32_t>(field_ids.size())) { return std::nullopt; }

  return "iceberg_scan table has a gap in its Iceberg field ids (highest is " +
         std::to_string(max_field_id) + " across " + std::to_string(field_ids.size()) +
         " fields), so a column was dropped; this scan path resolves columns by name and would "
         "read a dropped column's data in place of the re-added one";
}

std::string escape_sql_literal(std::string const& s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\'') { out += '\''; }
    out += c;
  }
  return out;
}

/// Type is empty for a nested container: parquet_schema() reports NULL there.
using iceberg_field_key = std::pair<std::string, int32_t>;
using iceberg_field_map = std::map<iceberg_field_key, std::string>;

/// Descends into children so nested fields count toward the schema each file must match.
void collect_field_id_names(std::vector<duckdb::MultiFileColumnDefinition> const& columns,
                            iceberg_field_map& out)
{
  for (auto const& column : columns) {
    if (!column.identifier.IsNull() &&
        column.identifier.type().id() == duckdb::LogicalTypeId::INTEGER) {
      out.emplace(iceberg_field_key{column.name, column.identifier.GetValue<int32_t>()},
                  column.children.empty() ? column.type.ToString() : std::string{});
    }
    collect_field_id_names(column.children, out);
  }
}

/**
 * @brief Refuse tables whose data files do not all carry the table's current (name, field id,
 *        type) schema.
 *
 * The scan resolves columns by NAME, so every evolution mis-reads: rename and add throw at SCAN
 * time, which takes the runtime fallback and deadlocks the connection, while promotion (int ->
 * long) throws nothing and returns the file's narrower type. Over-declining is the intended bias.
 *
 * Replaced wholesale once columns resolve by field id (DuckDB's MultiFileColumnMapper).
 *
 * @warning Reads every data file's Parquet footer on the planning thread. Fold into the footer
 *          cache the scan needs anyway rather than leaving two passes.
 */
std::optional<std::string> iceberg_schema_evolution_decline_reason(duckdb::LogicalGet& op,
                                                                   duckdb::Connection& conn)
{
  auto const* bind_data = dynamic_cast<duckdb::MultiFileBindData const*>(op.bind_data.get());
  if (bind_data == nullptr) { return std::nullopt; }

  iceberg_field_map table_schema;
  collect_field_id_names(
    bind_data->reader_bind.schema.empty() ? bind_data->columns : bind_data->reader_bind.schema,
    table_schema);
  // No field ids at all means a name-mapped table, which is what this path already assumes.
  if (table_schema.empty()) { return std::nullopt; }

  auto const files =
    resolve_parquet_scan_file_paths(op.function.name, op.bind_data.get(), op.parameters);
  if (files.empty()) { return std::nullopt; }

  // Apache writers record URIs in manifests, so these paths can arrive as `file:///...`. Strip
  // the scheme for the same reason the datasource boundary does: an unstripped path makes this
  // probe fail, and a failing probe declines -- which would quietly send every Apache-written
  // table to the CPU and undo the file:// fix this branch already landed.
  std::vector<std::string> probe_paths;
  probe_paths.reserve(files.size());
  std::string file_list = "[";
  for (std::size_t i = 0; i < files.size(); ++i) {
    probe_paths.push_back(sirius::io::strip_file_scheme(files[i]));
    if (i > 0) { file_list += ','; }
    file_list += "'" + escape_sql_literal(probe_paths.back()) + "'";
  }
  file_list += "]";

  // Do NOT filter `field_id IS NOT NULL` here: that drops every row of a file that has no ids,
  // and a file with no rows is indistinguishable from one that was never probed.
  auto result = conn.Query("SELECT file_name, name, field_id, duckdb_type FROM parquet_schema(" +
                           file_list + ")");
  if (!result || result->HasError()) {
    return "the iceberg schema probe could not read this table's data-file footers (" +
           std::string(result ? result->GetError() : "null result") +
           "), so the files could not be proven to carry the table's current schema";
  }

  std::map<std::string, iceberg_field_map> per_file;
  while (true) {
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0) { break; }
    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      auto& file_schema = per_file[chunk->GetValue(0, i).ToString()];
      auto const id     = chunk->GetValue(2, i);
      if (id.IsNull()) { continue; }
      auto const type = chunk->GetValue(3, i);
      file_schema.emplace(iceberg_field_key{chunk->GetValue(1, i).ToString(),
                                            static_cast<int32_t>(id.GetValue<int64_t>())},
                          type.IsNull() ? std::string{} : type.ToString());
    }
  }

  for (auto const& path : probe_paths) {
    auto const it = per_file.find(path);
    if (it == per_file.end()) {
      return "iceberg_scan data file '" + path +
             "' returned no Parquet schema rows, so it could not be proven to carry the table's "
             "current schema";
    }
    auto const& file_schema = it->second;
    if (file_schema.empty()) {
      return "iceberg_scan data file '" + path +
             "' carries no Parquet field ids while the table's schema declares them, so it is "
             "name-mapped; this scan path resolves columns by name and would read the wrong "
             "column or fail at scan time";
    }

    for (auto const& [key, table_type] : table_schema) {
      auto const found = file_schema.find(key);
      if (found == file_schema.end()) {
        return "iceberg_scan data file '" + path +
               "' does not carry the table's current schema (no match for " + key.first + "#" +
               std::to_string(key.second) +
               "), so the table's schema has evolved; this scan path resolves columns by name and "
               "would read the wrong column or fail at scan time";
      }
      // Empty on either side is a nested container, whose type is implied by its children.
      if (!table_type.empty() && !found->second.empty() && found->second != table_type) {
        return "iceberg_scan data file '" + path + "' stores " + key.first + "#" +
               std::to_string(key.second) + " as " + found->second + " while the table declares " +
               table_type +
               ", so the column's type was promoted; this scan path reads the file's own physical "
               "type and would hand back a column of the wrong type";
      }
    }
    // Extra fields are dropped columns, which a name-based lookup would happily resolve to.
    if (file_schema.size() != table_schema.size()) {
      return "iceberg_scan data file '" + path + "' carries " + std::to_string(file_schema.size()) +
             " field ids where the table declares " + std::to_string(table_schema.size()) +
             ", so the table's schema has evolved; this scan path resolves columns by name and "
             "would read the wrong column or fail at scan time";
    }
  }

  return std::nullopt;
}

// Why an `iceberg_scan` table must decline the GPU path, or nullopt when it may run there.
//
// V2 positional deletes and V3 deletion vectors are applied on GPU. Equality deletes are not:
// they match on key VALUES, so the scan must force-project the key columns even when the query
// does not select them, which is not wired.
//
// Conservative by construction: any failure to PROVE the table free of equality deletes declines
// to CPU. A false positive costs performance; a false negative would drop deletes silently.
//
// Each decline returns the reason it actually hit — a probe that never managed to look is not
// the same as a table that really carries equality deletes.
std::optional<std::string> iceberg_gpu_scan_decline_reason(duckdb::LogicalGet& op,
                                                           duckdb::ClientContext& context)
{
  if (op.parameters.empty() || op.parameters.front().IsNull()) {
    return "iceberg_scan called without a table path, so its delete files cannot be inspected";
  }

  std::string table_path;
  try {
    table_path = op.parameters.front().GetValue<std::string>();
  } catch (...) {
    return "iceberg_scan table path is not a string, so its delete files cannot be inspected";
  }
  if (table_path.empty()) {
    return "iceberg_scan table path is empty, so its delete files cannot be inspected";
  }

  // iceberg_scan has three snapshot selectors; the delete path honours only snapshot_from_id,
  // so the others would resolve deletes against the CURRENT snapshot while the scan reads the
  // time-travelled one — filtering snapshot A's data by snapshot B's deletes.
  //
  // Deliberately coarse: narrowing it to tables that actually have deletes at the SELECTED
  // snapshot needs the very selector resolution that is missing. To lift it, thread the full
  // selector into read_iceberg_delete_data AND into its cache key — the key currently records
  // snapshot_from_id or "current", so two timestamps against one table would collide.
  for (auto const& selector : {"snapshot_from_timestamp", "version"}) {
    auto it = op.named_parameters.find(selector);
    if (it != op.named_parameters.end() && !it->second.IsNull()) {
      return std::string("iceberg_scan was given '") + selector +
             "', but the GPU scan path resolves delete files only by snapshot_from_id, so its "
             "deletes would be read from the wrong snapshot";
    }
  }

  // The flag rewrites bound data-file paths to <table_path>/data/<name>, while delete discovery
  // calls iceberg_metadata() without it and keeps the manifests' originals. The two then match
  // nothing and every delete is silently dropped.
  if (auto it = op.named_parameters.find("allow_moved_paths");
      it != op.named_parameters.end() && !it->second.IsNull()) {
    bool moved = false;
    try {
      moved = it->second.GetValue<bool>();
    } catch (...) {
      moved = true;  // Unparsable: assume the rewrite is on rather than assume it is off.
    }
    if (moved) {
      return "iceberg_scan was given 'allow_moved_paths', which rewrites its data-file paths; the "
             "GPU path discovers delete files under the paths the manifests record, so the two "
             "would not match and the table's deletes would be dropped";
    }
  }

  // "current" is resolved independently by DuckDB's bind, by Sirius's rebind of the serialized
  // plan, and by delete discovery; nothing makes the three agree, so a commit landing between any
  // two pairs one snapshot's data files with another's deletes. The id DuckDB bound is not
  // reachable -- it lives in types the iceberg extension compiles privately.
  //
  // Do NOT resolve it here instead. Highest sequence number is not the current snapshot under
  // rollback, branches or staged WAP commits, and comparing data-file sets cannot separate two
  // snapshots that differ only in deletes. Lifting this needs serializable iceberg bind data
  // upstream, or Sirius not planning twice.
  {
    auto const it = op.named_parameters.find("snapshot_from_id");
    if (it == op.named_parameters.end() || it->second.IsNull()) {
      return "iceberg_scan was called without 'snapshot_from_id', so the snapshot DuckDB bound "
             "cannot be recovered; the GPU path resolves delete files in a separate pass and "
             "would risk pairing one snapshot's data files with another's deletes";
    }
  }

  if (auto reason = iceberg_retired_field_id_decline_reason(op)) { return reason; }

  std::string escaped;
  escaped.reserve(table_path.size());
  for (char c : table_path) {
    if (c == '\'') { escaped += '\''; }
    escaped += c;
  }

  std::string query = "SELECT count(*) FROM iceberg_metadata('" + escaped + "'";
  // Inspect the same snapshot the scan will read.
  auto sid_it = op.named_parameters.find("snapshot_from_id");
  if (sid_it != op.named_parameters.end() && !sid_it->second.IsNull()) {
    try {
      query += ", snapshot_from_id = " + std::to_string(sid_it->second.GetValue<int64_t>());
    } catch (...) {
      return "iceberg_scan snapshot_from_id is not an integer, so the snapshot the scan will read "
             "cannot be inspected for delete files";
    }
  }
  // status <> 'DELETED' matches discover_from_manifests: a manifest keeps listing entries later
  // commits retired, so without it a table whose only equality-delete file has been compacted away
  // is refused the GPU forever over a delete that no longer applies. Both sites must agree on
  // which entries are live, or the gate's verdict describes a different table than the scan reads.
  query += ") WHERE content = 'EQUALITY_DELETES' AND status <> 'DELETED'";

  try {
    // Opening a second Connection to the same database re-registers the SAME SiriusContext, so
    // its query-lifecycle callbacks would fire QueryBegin (resetting next_operator_id and
    // task_creator state) and QueryEnd (clearing all data repositories) underneath the query
    // currently being planned. InternalQueryGuard suppresses both; without it this probe hangs
    // the outer query. Same pattern as sirius_extension.cpp's CPU-fallback replay.
    auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (!sirius_ctx) {
      SIRIUS_LOG_DEBUG(
        "[sirius_plan_get] iceberg delete probe: no SiriusContext — CPU fallback for '{}'",
        table_path);
      return "the iceberg delete probe could not acquire the Sirius context, so the table could "
             "not be proven free of equality-delete files";
    }
    duckdb::SiriusContext::InternalQueryGuard guard(context);

    duckdb::Connection conn(*context.db);
    // Per-connection bracket: guarding `context` above does not cover this fresh connection.
    // Without it the probe contends for the plan-generation slot the outer query holds, fails,
    // and "any failure declines" then refuses every table.
    duckdb::SiriusContext::InternalQueryGuard conn_guard(*conn.context);
    // The probe runs on a FRESH connection, which does not inherit the session's settings. Without
    // this, a table whose metadata carries no version hint fails the probe outright — and because
    // any failure declines, a table with ZERO delete files was refused and told it had equality
    // deletes. Only a user who had typed `SET GLOBAL unsafe_enable_version_guessing` (global scope
    // reaches new connections; plain `SET` does not) ever got past this.
    //
    // Matches discover_from_manifests (iceberg_metadata_reader.cpp), which sets it on its own
    // connection for the same reason. Both read the same manifests through iceberg_metadata(), so
    // they must agree on which tables are legible; if they disagree, the gate's verdict is about a
    // different table than the one the scan goes on to read.
    conn.Query("SET unsafe_enable_version_guessing = true");
    auto result = conn.Query(query);
    if (!result || result->HasError()) {
      SIRIUS_LOG_DEBUG("[sirius_plan_get] iceberg delete probe failed for '{}': {} — CPU fallback",
                       table_path,
                       result ? result->GetError() : "null result");
      return "the iceberg delete probe could not read this table's metadata (" +
             std::string(result ? result->GetError() : "null result") +
             "), so it could not be proven free of equality-delete files";
    }
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0) {
      return "the iceberg delete probe returned no rows, so the table could not be proven free of "
             "equality-delete files";
    }
    auto const n_delete_files = chunk->GetValue(0, 0).GetValue<int64_t>();
    if (n_delete_files > 0) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_get] iceberg table '{}' has {} equality-delete file(s); the GPU scan path "
        "does not apply equality deletes yet — falling back to DuckDB CPU.",
        table_path,
        n_delete_files);
      return "this iceberg table has " + std::to_string(n_delete_files) +
             " equality-delete file(s), which the GPU scan path does not apply yet";
    }

    // Reuses this connection deliberately: it is already bracketed as an internal query, and a
    // third Connection here would re-register the same SiriusContext underneath the query being
    // planned.
    if (auto reason = iceberg_schema_evolution_decline_reason(op, conn)) { return reason; }

    return std::nullopt;
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG("[sirius_plan_get] iceberg plan-time probe threw for '{}': {} — CPU fallback",
                     table_path,
                     e.what());
    return "an iceberg plan-time probe threw (" + std::string(e.what()) +
           "), so the table could not be proven free of equality-delete files or of schema "
           "evolution";
  }
}

// Translate a vector of DuckDB expressions into Sirius AST nodes at the planner
// boundary. The source vector is drained; size and order are preserved, with a
// null slot wherever from_duckdb declines an unsupported shape (a fallback
// signal) — matching the prior bulk-translation null-skip semantics.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(e ? sirius::ast::from_duckdb(*e) : nullptr);
  }
  return out;
}

std::vector<cudf::data_type> scan_physical_schema(duckdb::LogicalGet& op,
                                                  duckdb::SiriusContext* state,
                                                  bool compressed_materialization_on,
                                                  const duckdb::vector<duckdb::ColumnIndex>& ids,
                                                  sirius::scan_manager::pinned_entry const* entry)
{
  if (!state || !compressed_materialization_on || entry == nullptr) { return {}; }

  // Install a narrow sidecar only when the pinned cache can serve this scan. Each target comes from
  // recorded stored-column metadata, so compressed and uncompressed chunks use the same path. A
  // chunk already at the target passes through; one stored narrower widens to the target. An
  // unpinned scan follows the native path, and a pinned-native column is not narrowed while
  // serving. A pin that cannot serve every requested column also takes the native disk-read path,
  // avoiding recurring range verification and downcasts.
  auto const projection = entry->cache_info.column_projection_for(ids);
  if (projection.empty()) { return {}; }

  std::vector<cudf::data_type> result;
  result.reserve(op.types.size());
  bool changed = false;
  for (std::size_t output_idx = 0; output_idx < op.types.size(); output_idx++) {
    auto const logical = sirius::from_duckdb(op.types[output_idx]);
    auto const native  = sirius::try_get_cudf_type(logical);
    if (!native) { return {}; }
    result.push_back(*native);
    if (!sirius::is_narrowable_numeric_type(logical)) { continue; }

    std::size_t ids_position = output_idx;
    if (!op.projection_ids.empty()) {
      if (output_idx >= op.projection_ids.size()) { continue; }
      ids_position = op.projection_ids[output_idx];
    }
    if (ids_position >= ids.size()) { continue; }

    auto const target =
      sirius::scan_manager::pinned_column_narrow_carrier(*entry, projection[ids_position], *native);
    if (!target) { continue; }
    result.back() = *target;
    changed       = true;
  }
  return changed ? result : std::vector<cudf::data_type>{};
}

// An OPTIONAL_FILTER is advisory and an IS_NOT_NULL is applied by the scan itself, so
// neither contributes to the predicate convert_table_filters_to_expression builds
// (scan_utils.cpp). This must stay in step with that skip set: probing a filter the
// scan discharges would reject plans the scan handles correctly.
[[nodiscard]] bool is_discharged_without_translation(duckdb::TableFilterType filter_type)
{
  return filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
         filter_type == duckdb::TableFilterType::IS_NOT_NULL;
}

// Pushed-down filters bypass LogicalFilter, so validate the remaining predicate at plan
// time. The runtime translates again because it resolves references against
// batch-relative positions.
void reject_untranslatable_table_filter(duckdb::TableFilter const& filter,
                                        duckdb::LogicalType const& column_type,
                                        std::string const& column_name)
{
  if (is_discharged_without_translation(filter.filter_type)) { return; }
  auto column_ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(column_type, 0);
  auto expression = filter.ToExpression(*column_ref);
  if (sirius::ast::from_duckdb(*expression) == nullptr) {
    throw duckdb::NotImplementedException("Unsupported filter predicate on column '" + column_name +
                                          "' (falling back to CPU): " + expression->ToString());
  }
}

}  // namespace

duckdb::unique_ptr<duckdb::TableFilterSet> create_table_filter_set(
  duckdb::TableFilterSet& table_filters, const duckdb::vector<duckdb::ColumnIndex>& column_ids)
{
  // create the table filter map
  auto table_filter_set = duckdb::make_uniq<duckdb::TableFilterSet>();
  for (auto& table_filter : table_filters.filters) {
    // find the relative column index from the absolute column index into the table
    duckdb::optional_idx column_index;
    for (std::size_t i = 0; i < column_ids.size(); i++) {
      if (table_filter.first == column_ids[i].GetPrimaryIndex()) {
        column_index = i;
        break;
      }
    }
    if (!column_index.IsValid()) {
      throw duckdb::InternalException("Could not find column index for table filter");
    }
    table_filter_set->filters[column_index.GetIndex()] = std::move(table_filter.second);
  }
  return table_filter_set;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalGet& op)
{
  auto column_ids = op.GetColumnIds();

  // Only GPU-route known table scan functions; all others (pragma, system catalog
  // functions, etc.) must fall back to CPU.
  static const std::unordered_set<std::string> kSupportedScanFunctions = {
    "seq_scan", "parquet_scan", "read_parquet", "sirius_read_parquet", "iceberg_scan"};
  if (kSupportedScanFunctions.find(op.function.name) == kSupportedScanFunctions.end()) {
    throw duckdb::NotImplementedException("Table function '%s' is not supported in Sirius",
                                          op.function.name);
  }

  // An iceberg table's data files are parquet, and `iceberg_scan` binds them into the same
  // MultiFileBindData `read_parquet` uses, so the parquet ingestible reads them as-is; the
  // iceberg ingestible then applies positional deletes and deletion vectors to each decoded
  // batch. Equality deletes are not applied yet, and reading a table that has them as plain
  // parquet would silently return rows the table logically deleted — so refuse here, where
  // NotImplementedException is the established CPU-fallback signal.
  //
  // Ordered ahead of the residency probing below because this path throws: declining first
  // keeps a refused table from paying for pinned-entry lookup and schema resolution.
  if (op.function.name == "iceberg_scan") {
    if (auto reason = iceberg_gpu_scan_decline_reason(op, context)) {
      throw duckdb::NotImplementedException("iceberg_scan declines the GPU scan path: " + *reason);
    }
  }

  auto sirius_state = context.registered_state
                        ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                        : nullptr;

  // Resolved once for this plan against the connection being planned for, so the parquet
  // identity probe and the residency gate below cannot disagree about whether narrowing is on.
  bool const compressed_materialization_on = duckdb::compressed_materialization_enabled(context);

  // One pinned-entry probe per scan: the compressed-materialization residency gate and
  // the seq_scan MVCC cache-or-CPU guard below share the result. The parquet identity
  // feeds only the gate, so its file resolution runs only when the feature is on.
  sirius::scan_manager::pinned_entry const* pinned = nullptr;
  bool serves_insert_deltas                        = false;
  if (sirius_state && op.function.name == "seq_scan") {
    auto* bind = dynamic_cast<duckdb::TableScanBindData*>(op.bind_data.get());
    if (bind != nullptr && bind->table.IsDuckTable()) {
      auto& table = bind->table.Cast<duckdb::DuckTableEntry>();
      pinned      = sirius_state->get_scan_manager().find_pinned_entry_for_duckdb_table(
        table.ParentCatalog().GetName(), table.ParentSchema().name, table.name);
      // Rows beyond the pinned prefix serve as insert-delta splits, decoded fresh at native
      // width. A narrow sidecar over them would pay per-batch exact-range verification and, on
      // an out-of-range inserted value, fail the query over to the CPU fallback — so the
      // residency gate below installs no narrow targets for a delta-serving scan. Entry chunks
      // and their storage metadata are untouched by deltas. See issue ticket #1311.
      if (pinned != nullptr && pinned->mvcc != nullptr &&
          static_cast<std::size_t>(table.GetStorage().GetTotalRows()) > pinned->mvcc->n_cache()) {
        serves_insert_deltas = true;
      }
    }
  } else if (sirius_state && compressed_materialization_on) {
    auto const files =
      resolve_parquet_scan_file_paths(op.function.name, op.bind_data.get(), op.parameters);
    if (!files.empty()) {
      pinned = sirius_state->get_scan_manager().find_pinned_entry_for_parquet_files(files);
    }
  }

  auto physical_types = scan_physical_schema(op,
                                             sirius_state.get(),
                                             compressed_materialization_on,
                                             column_ids,
                                             serves_insert_deltas ? nullptr : pinned);

  if (!op.children.empty()) {
    throw duckdb::NotImplementedException("Table Input Output functions are not supported yet");
  }

  if (!op.projected_input.empty()) {
    throw duckdb::InternalException(
      "LogicalGet::project_input can only be set for table-in-out functions");
  }

  // Plan-time probe for the duckdb-native seq_scan path: strings at/over
  // StringUncompressed::GetStringBlockLimit (a per-value limit) live in overflow
  // blocks the GPU string decoder cannot resolve. Refuse HERE, where the throw still
  // becomes a clean CPU fallback — the walker's refusal at pipeline conversion
  // surfaces as a mid-query error with none. Conservative for DICT_FSST, which
  // inlines strings up to 16 KiB (see prepare_duckdb_native_walk).
  if (op.function.name == "seq_scan" && op.bind_data) {
    auto* table_scan_bind = dynamic_cast<duckdb::TableScanBindData*>(op.bind_data.get());
    if (table_scan_bind != nullptr && table_scan_bind->table.IsDuckTable()) {
      auto& table   = table_scan_bind->table.Cast<duckdb::DuckTableEntry>();
      auto& storage = table.GetStorage();
      auto const block_size =
        storage.GetAttached().GetStorageManager().GetBlockManager().GetBlockSize();
      auto const overflow_limit = duckdb::StringUncompressed::GetStringBlockLimit(block_size);
      for (auto const& col_idx : column_ids) {
        if (!col_idx.HasPrimaryIndex() || col_idx.IsRowIdColumn() || col_idx.IsVirtualColumn() ||
            col_idx.IsEmptyColumn()) {
          continue;
        }
        auto const primary = col_idx.GetPrimaryIndex();
        if (primary >= op.returned_types.size() ||
            op.returned_types[primary].id() != duckdb::LogicalTypeId::VARCHAR) {
          continue;
        }
        auto stats = table.GetStatistics(context, primary);
        if (!stats || !duckdb::StringStats::HasMaxStringLength(*stats) ||
            duckdb::StringStats::MaxStringLength(*stats) >= overflow_limit) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: varchar column %llu may contain strings at/over the "
            "overflow-block limit (%llu bytes); overflow strings are not GPU-decodable",
            static_cast<unsigned long long>(primary),
            static_cast<unsigned long long>(overflow_limit));
        }
      }

      // Sentinel columns (rowid/virtual/empty/field) have no storage backing,
      // which makes a pin unservable for this scan.
      bool has_unservable_column = false;
      for (auto const& col_idx : column_ids) {
        if (!col_idx.HasPrimaryIndex() || col_idx.IsRowIdColumn() || col_idx.IsVirtualColumn() ||
            col_idx.IsEmptyColumn()) {
          has_unservable_column = true;
          break;
        }
      }

      if (pinned != nullptr && pinned->mvcc != nullptr) {
        // Cache-or-CPU guards: while this table is MVCC-pinned, a GPU plan
        // either serves exactly from the pinned cache (DELETE keep-masks) or is
        // refused HERE, where the throw still becomes a clean CPU fallback. The
        // disk-native path is MVCC-blind, and the pin's checkpoint suppression
        // makes its snapshot increasingly stale — so scans the pin cannot serve
        // never fall through to it.
        auto const n_cache = pinned->mvcc->n_cache();
        // (a) snapshot-too-old: this transaction opened before the pin, so
        // the cache's base image is from its future.
        auto const start_time =
          duckdb::DuckTransaction::Get(context, table.ParentCatalog()).start_time;
        if (start_time < pinned->mvcc->v_base) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: transaction snapshot (%llu) predates the pinned cache "
            "snapshot (%llu) for table '%s'",
            static_cast<unsigned long long>(start_time),
            static_cast<unsigned long long>(pinned->mvcc->v_base),
            table.name);
        }
        // (b) transaction-local appends: rows in this transaction's
        // LocalStorage live outside the table's segment trees, so neither the
        // cache nor the insert delta can serve them. Committed rows beyond
        // the pinned prefix are served by the prepare-time insert-delta job,
        // masked to this snapshot's visibility.
        if (duckdb::LocalStorage::Get(context, storage.GetAttached()).GetStorage(storage)) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' has uncommitted appends in this transaction; "
            "transaction-local inserts are not served from the cache",
            table.name);
        }
        // (b') the insert delta does not serve ARRAY columns, so decline
        // while rows beyond the pinned prefix exist. Rows inserted after
        // this check are caught by the capture's own ARRAY check.
        if (static_cast<std::size_t>(storage.GetTotalRows()) > n_cache) {
          for (auto const& col_idx : column_ids) {
            if (!col_idx.HasPrimaryIndex() || col_idx.IsRowIdColumn() ||
                col_idx.IsVirtualColumn() || col_idx.IsEmptyColumn()) {
              continue;
            }
            auto const primary = col_idx.GetPrimaryIndex();
            if (primary < op.returned_types.size() &&
                op.returned_types[primary].id() == duckdb::LogicalTypeId::ARRAY) {
              throw duckdb::NotImplementedException(
                "duckdb-native scan: table '%s' has rows beyond the pinned prefix and an ARRAY "
                "projection; the insert delta does not serve ARRAY columns",
                table.name);
            }
          }
        }
        bool pin_serves = !has_unservable_column;
        if (pin_serves && !column_ids.empty()) {
          pin_serves = !pinned->cache_info.column_projection_for(column_ids).empty();
        }
        if (!pin_serves) {
          // (d) column-mismatch: the scan would fall through to the MVCC-blind
          // disk-native read, so it always declines. The disabled clean-table
          // relaxation below lets a table that provably matches its
          // last-checkpointed image fall through instead (#1160).
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' is MVCC-pinned and the pin cannot serve the "
            "requested columns",
            table.name);
        }
#if 0
        // Disabled — these guards walk every row group of the table at plan
        // time, per query. With this block off, (d) above has no clean-table
        // relaxation. The update-chain branch is redundant because UPDATE
        // statements on pinned tables are rejected before execution; direct
        // UPDATE serving remains out of scope here (#1162).
        std::vector<duckdb::storage_t> projected;
        for (auto const& col_idx : column_ids) {
          if (col_idx.HasPrimaryIndex() && !col_idx.IsRowIdColumn() &&
              !col_idx.IsVirtualColumn() && !col_idx.IsEmptyColumn()) {
            projected.push_back(col_idx.GetPrimaryIndex());
          }
        }
        if (!pin_serves) {
          // (d) column-mismatch: the scan falls through to the disk-native
          // read, so it must pass the same exactness check as an unpinned
          // scan. Guards (a)/(b) already excluded post-pin inserts and
          // transaction-local appends.
          auto& txn = duckdb::DuckTransaction::Get(context, table.ParentCatalog());
          if (sirius::op::scan::check_native_read_mvcc_state(
                storage, projected, duckdb::TransactionData(txn)) !=
              sirius::op::scan::native_read_mvcc_state::exact) {
            throw duckdb::NotImplementedException(
              "duckdb-native scan: table '%s' is MVCC-pinned, the pin cannot serve the "
              "requested columns, and the table has diverged from its last-checkpointed "
              "image",
              table.name);
          }
        } else if (sirius::op::scan::any_update_chains(storage, projected, n_cache)) {
          // (c) update-present on a column the cache would serve: update
          // chains version values in place, invisibly to the DELETE
          // keep-masks — the cached values would be stale.
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' has in-memory update chains on a scanned "
            "column; updated values are not served from the cache",
            table.name);
        }
#endif
      }
#if 0
      // Disabled — plan-time MVCC guards for duckdb-native scans of unpinned
      // tables: the exactness walk touches every row group of the table, per
      // query; #1160 tracks running it from the scan manager at execution
      // time instead. With this block off, the disk-native read of an
      // unpinned table is MVCC-blind (#1143): uncheckpointed deletes and
      // update chains are served silently, and committed-but-uncheckpointed
      // inserts fail loudly at execution.
      if (pinned == nullptr || pinned->mvcc == nullptr) {
        // No MVCC-pinned cache for this table: the plan is the disk-native
        // read, which applies no visibility filtering — refuse any state it
        // would misread HERE, where the throw still becomes a clean CPU
        // fallback. Residual race: a row committing between this check and
        // the scan's metadata capture is read unmasked; the prepare-time
        // keep-masks planned in #1143 close it.
        std::vector<duckdb::storage_t> projected;
        for (auto const& col_idx : column_ids) {
          if (col_idx.HasPrimaryIndex() && !col_idx.IsRowIdColumn() &&
              !col_idx.IsVirtualColumn() && !col_idx.IsEmptyColumn()) {
            projected.push_back(col_idx.GetPrimaryIndex());
          }
        }
        if (duckdb::LocalStorage::Get(context, storage.GetAttached()).GetStorage(storage)) {
          throw duckdb::NotImplementedException(
            "duckdb-native scan: table '%s' has uncommitted appends in this transaction; "
            "the disk-native read cannot see transaction-local rows",
            table.name);
        }
        auto& txn = duckdb::DuckTransaction::Get(context, table.ParentCatalog());
        switch (sirius::op::scan::check_native_read_mvcc_state(
          storage, projected, duckdb::TransactionData(txn))) {
          case sirius::op::scan::native_read_mvcc_state::has_update_chains:
            throw duckdb::NotImplementedException(
              "duckdb-native scan: table '%s' has in-memory update chains on a scanned "
              "column; the disk-native read would return stale values",
              table.name);
          case sirius::op::scan::native_read_mvcc_state::has_invisible_rows:
            throw duckdb::NotImplementedException(
              "duckdb-native scan: table '%s' has rows not visible to this transaction "
              "(uncheckpointed deletes or in-flight inserts); the disk-native read is "
              "MVCC-blind",
              table.name);
          case sirius::op::scan::native_read_mvcc_state::exact: break;
        }
      }
#endif
    }
  }

  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  if (!op.table_filters.filters.empty()) {
    table_filters = create_table_filter_set(op.table_filters, column_ids);
    // Predicates pushed into table_filters bypass the LogicalFilter guard —
    // reject nested columns here too (e.g. `WHERE items IS NULL`).
    for (auto const& entry : table_filters->filters) {
      auto const column_id = column_ids[entry.first].GetPrimaryIndex();
      if (column_id < op.returned_types.size()) {
        auto const column_name =
          column_id < op.names.size() ? op.names[column_id] : std::to_string(column_id);
        reject_nested_column_type(op.returned_types[column_id], column_name, "a filter predicate");
        reject_untranslatable_table_filter(
          *entry.second, op.returned_types[column_id], column_name);
      }
    }
  }

  if (op.function.dependency) { op.function.dependency(dependencies, op.bind_data.get()); }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> filter;
  auto& projection_ids = op.projection_ids;

  // With FILTER_PUSHDOWN enabled, filters from WHERE clauses are pushed into table_filters.
  // Since we don't pass filters to the DuckDB table function (they're applied by Sirius),
  // we need to ensure all filter columns are included in BOTH column_ids and projection_ids.
  // We track the original projection_ids so we can project back after filtering.
  duckdb::vector<std::size_t> original_projection_ids = projection_ids;

  // Save the original types before we modify projection_ids, because modifying projection_ids
  // might affect the types when we call ResolveOperatorTypes()
  duckdb::vector<duckdb::LogicalType> original_types = op.types;

  if (table_filters) {
    for (auto& entry : table_filters->filters) {
      // entry.first is the column index in the table_filters (after remapping by
      // create_table_filter_set) We need to ensure this column is in projection_ids so it gets
      // scanned by DuckDB

      bool found_in_projection = false;
      for (std::size_t j = 0; j < projection_ids.size(); j++) {
        if (projection_ids[j] == entry.first) {
          found_in_projection = true;
          break;
        }
      }

      if (!found_in_projection) { projection_ids.push_back(entry.first); }
    }
  }

  // Handle cases where table function doesn't support pushdown for specific column types
  if (table_filters && op.function.supports_pushdown_type) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    duckdb::unordered_set<std::size_t> to_remove;
    for (auto& entry : table_filters->filters) {
      auto column_id = column_ids[entry.first].GetPrimaryIndex();
      auto& type     = op.returned_types[column_id];

      // If the table function doesn't support pushdown for this column type,
      // create a separate filter operator for it
      if (!op.function.supports_pushdown_type(*op.bind_data, column_id)) {
        std::size_t column_id_filter = entry.first;
        auto column = duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, column_id_filter);
        select_list.push_back(entry.second->ToExpression(*column));
        to_remove.insert(entry.first);
      }
    }
    for (auto& col : to_remove) {
      table_filters->filters.erase(col);
    }

    if (!select_list.empty()) {
      duckdb::vector<duckdb::LogicalType> filter_types;
      for (auto& c : projection_ids) {
        auto column_id = column_ids[c].GetPrimaryIndex();
        filter_types.push_back(op.returned_types[column_id]);
      }
      // sirius_physical_filter owns a single expression; AND-merge predicates when there are many.
      duckdb::unique_ptr<duckdb::Expression> combined;
      if (select_list.size() > 1) {
        auto conjunction = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
          duckdb::ExpressionType::CONJUNCTION_AND);
        for (auto& expr : select_list) {
          conjunction->children.push_back(std::move(expr));
        }
        combined = std::move(conjunction);
      } else {
        combined = std::move(select_list[0]);
      }
      filter =
        duckdb::make_uniq<sirius::op::sirius_physical_filter>(sirius::from_duckdb_vec(filter_types),
                                                              sirius::ast::from_duckdb(*combined),
                                                              op.estimated_cardinality);
    }
  }
  op.ResolveOperatorTypes();
  // create the table scan node
  if (!op.function.projection_pushdown) {
    // function does not support projection pushdown
    auto node = duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
      sirius::from_duckdb_vec(op.returned_types),
      op.function,
      std::move(op.bind_data),
      sirius::from_duckdb_vec(op.returned_types),
      column_ids,
      duckdb::vector<duckdb::column_t>(),
      op.names,
      std::move(table_filters),
      op.estimated_cardinality,
      std::move(op.extra_info),
      std::move(op.parameters),
      std::move(op.virtual_columns));
    node->named_parameters = std::move(op.named_parameters);
    // first check if an additional projection is necessary
    if (column_ids.size() == op.returned_types.size()) {
      bool projection_necessary = false;
      for (std::size_t i = 0; i < column_ids.size(); i++) {
        if (column_ids[i].GetPrimaryIndex() != i) {
          projection_necessary = true;
          break;
        }
      }
      if (!projection_necessary) {
        // a projection is not necessary if all columns have been requested in-order
        // in that case we just return the node
        if (filter) {
          filter->children.push_back(std::move(node));
          return filter;
        }
        return std::move(node);
      }
    }
    // push a projection on top that does the projection
    duckdb::vector<duckdb::LogicalType> types;
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
    for (std::size_t i = 0; i < column_ids.size(); ++i) {
      auto& column_id = column_ids[i];
      if (column_id.IsVirtualColumn()) {
        throw duckdb::NotImplementedException("Virtual columns require projection pushdown");
      } else {
        auto col_id = column_id.GetPrimaryIndex();
        auto type   = op.returned_types[col_id];
        types.push_back(type);
        // The Sirius scan emits exactly the column_ids columns, in order, at
        // positions 0..M-1 (build_scan_plan with empty projection_ids) — unlike
        // DuckDB's native full-width scan this branch was modeled on.  So
        // reference the column by its position i in the scan output, not by its
        // original parquet index col_id, which can exceed the M-column width.
        expressions.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, i));
      }
    }
    duckdb::unique_ptr<sirius::op::sirius_physical_operator> scan_child;
    if (filter) {
      filter->children.push_back(std::move(node));
      scan_child = std::move(filter);
    } else {
      scan_child = std::move(node);
    }
    return push_projection(std::move(scan_child),
                           sirius::from_duckdb_vec(types),
                           translate_expressions(std::move(expressions)),
                           op.estimated_cardinality);
  }

  auto node = duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
    sirius::from_duckdb_vec(original_types),  // Use original types, not modified
    op.function,
    std::move(op.bind_data),
    sirius::from_duckdb_vec(op.returned_types),
    column_ids,
    op.projection_ids,
    op.names,
    std::move(table_filters),
    op.estimated_cardinality,
    std::move(op.extra_info),
    std::move(op.parameters),
    std::move(op.virtual_columns));
  if (!physical_types.empty() && physical_types.size() == node->types.size()) {
    node->set_physical_types(std::move(physical_types));
    node->sidecar_from_gpu_tier_pin =
      pinned != nullptr && pinned->tier == cucascade::memory::Tier::GPU;
    if (sirius_state) { sirius_state->record_compressed_materialization_scan_sidecar_installed(); }
  }
  node->named_parameters = std::move(op.named_parameters);
  node->dynamic_filters  = op.dynamic_filters;
  if (op.dynamic_filters) {
    node->sirius_dynamic_filters = get_or_create_dynamic_filter_channel(op.dynamic_filters.get());
    SIRIUS_LOG_INFO("[sirius_plan_get] LogicalGet has dynamic_filters attached (channel key={}).",
                    static_cast<void const*>(op.dynamic_filters.get()));
  }
  if (filter) {
    filter->children.push_back(std::move(node));
    return filter;
  }
  return std::move(node);
}

}  // namespace sirius::planner
