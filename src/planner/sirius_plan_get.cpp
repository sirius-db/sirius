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
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/segment/uncompressed.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/string_stats.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "exec/stream_bind_catalog.hpp"
#include "exec/stream_plan_bindings.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/numeric_narrowing.hpp"
#include "helper/type_conversions.hpp"
#include "io/uri_parser.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "op/scan/iceberg_metadata_connection.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/kmeans_functions.hpp"
#include "vss/sirius_physical_vector_join_materialize.hpp"
#include "vss/sirius_physical_vector_join_reduce_local.hpp"
#include "vss/sirius_physical_vector_join_select.hpp"
#include "vss/sirius_physical_vector_join_stream.hpp"
#include "vss/vector_join.hpp"
#include "vss/vector_join_binding.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
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
///
/// @p order receives the same ids in recursive PREORDER — the walk order is load-bearing, because
/// it is what a file's physical layout is compared against.
void collect_field_id_names(std::vector<duckdb::MultiFileColumnDefinition> const& columns,
                            iceberg_field_map& out,
                            std::vector<int32_t>& order)
{
  for (auto const& column : columns) {
    if (!column.identifier.IsNull() &&
        column.identifier.type().id() == duckdb::LogicalTypeId::INTEGER) {
      auto const id = column.identifier.GetValue<int32_t>();
      out.emplace(iceberg_field_key{column.name, id},
                  column.children.empty() ? column.type.ToString() : std::string{});
      order.push_back(id);
    }
    collect_field_id_names(column.children, out, order);
  }
}

/**
 * @brief Refuse tables whose data files do not all carry the table's current (name, field id,
 *        type) schema, in the same physical ORDER.
 *
 * The scan resolves columns by NAME and emits them in the FILE's order, so every evolution
 * mis-reads: rename and add throw at SCAN time, which takes the runtime fallback and deadlocks the
 * connection; promotion (int -> long) throws nothing and returns the file's narrower type; and a
 * permutation returns the right columns under the wrong names. Over-declining is the intended
 * bias, and a valid-but-declined Iceberg layout is read correctly by DuckDB.
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
  std::vector<int32_t> table_field_order;
  collect_field_id_names(
    bind_data->reader_bind.schema.empty() ? bind_data->columns : bind_data->reader_bind.schema,
    table_schema,
    table_field_order);
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
  // column_id is the row's index into the Parquet footer's flattened schema, i.e. its position in
  // the file's own recursive preorder. It is how physical ORDER is compared, which membership in a
  // (name, field id) map cannot see.
  auto result =
    conn.Query("SELECT file_name, name, field_id, duckdb_type, column_id FROM parquet_schema(" +
               file_list + ")");
  if (!result || result->HasError()) {
    return "the iceberg schema probe could not read this table's data-file footers (" +
           std::string(result ? result->GetError() : "null result") +
           "), so the files could not be proven to carry the table's current schema";
  }

  struct file_footer {
    iceberg_field_map schema;
    /// (column_id, field_id) for every ID-bearing footer row, sorted into physical order below.
    std::vector<std::pair<int64_t, int32_t>> id_order;
  };
  std::map<std::string, file_footer> per_file;
  while (true) {
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0) { break; }
    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      auto& footer  = per_file[chunk->GetValue(0, i).ToString()];
      auto const id = chunk->GetValue(2, i);
      // A row with no field id is either the schema root or a synthetic nesting level (a 3-level
      // list's `list` group, a map's `key_value`); neither is a field the table declares.
      if (id.IsNull()) { continue; }
      auto const type     = chunk->GetValue(3, i);
      auto const field_id = static_cast<int32_t>(id.GetValue<int64_t>());
      footer.schema.emplace(iceberg_field_key{chunk->GetValue(1, i).ToString(), field_id},
                            type.IsNull() ? std::string{} : type.ToString());
      auto const column_id = chunk->GetValue(4, i);
      footer.id_order.emplace_back(column_id.IsNull() ? 0 : column_id.GetValue<int64_t>(),
                                   field_id);
    }
  }

  for (auto const& path : probe_paths) {
    auto const it = per_file.find(path);
    if (it == per_file.end()) {
      return "iceberg_scan data file '" + path +
             "' returned no Parquet schema rows, so it could not be proven to carry the table's "
             "current schema";
    }
    auto const& file_schema = it->second.schema;
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

    // Everything above is MEMBERSHIP, which a permuted file satisfies. Order matters because the
    // GPU path does not map columns by field id: for a full `SELECT *`, build_scan_plan leaves
    // needs_reader_projection false, so cuDF emits columns in the first footer's order while the
    // rest of the plan expects the bound snapshot's. DuckDB's own reader uses BY_FIELD_ID and is
    // unaffected, so the two disagree silently -- and a castable permutation converts the values
    // rather than erroring, since the runtime schema check only logs. Nested children are included
    // because a reordered struct child fails the same way.
    auto id_order = it->second.id_order;
    std::sort(id_order.begin(), id_order.end());
    std::vector<int32_t> file_field_order;
    file_field_order.reserve(id_order.size());
    for (auto const& [column_id, field_id] : id_order) {
      file_field_order.push_back(field_id);
    }
    if (file_field_order != table_field_order) {
      return "iceberg_scan data file '" + path +
             "' stores the table's fields in a different physical order than the bound snapshot's "
             "schema declares; this scan path emits columns in the file's own order and would hand "
             "back the right columns under the wrong names";
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

    // See iceberg_metadata_connection: one entry point, shared with discover_from_manifests so
    // both agree on which tables are legible, and pinned to the same snapshot the scan was bound
    // to. It mirrors the session's `unsafe_enable_version_guessing` rather than forcing it; a
    // table the outer session cannot read fails here and the decline below sends it to DuckDB.
    sirius::op::scan::iceberg_metadata_connection metadata_conn(context);
    auto result = metadata_conn.Query(query);
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
    if (auto reason = iceberg_schema_evolution_decline_reason(op, metadata_conn.get())) {
      return reason;
    }

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

//! A scan of one join side's table, built the way DuckDB builds a base-table scan
//! (`GetScanFunction` mints the same bind data its binder would) rather than bound from SQL.
//! The vector join's LogicalGet has no children -- it is a table function, and a table-in-out
//! surface is a separate piece of work -- so a fed side's input is constructed here instead.
//! The vector column is projected first, which is why both the fold and materialize can index
//! it as column 0 without a name lookup.
/// @param extra_column  Appended after the emitted columns when non-empty. Used for the
///                      corpus's cluster ids, which the fold reads but the join never emits --
///                      putting it last is what keeps materialize's column positions unchanged.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_side_scan(
  duckdb::ClientContext& context,
  const sirius::vss::vector_join_side& side,
  const std::string& extra_column = {})
{
  auto& entry_base = duckdb::Catalog::GetEntry(
    context, duckdb::CatalogType::TABLE_ENTRY, side.catalog, side.schema, side.table);
  auto& table_entry = entry_base.Cast<duckdb::DuckTableEntry>();

  auto const& columns = table_entry.GetColumns();
  auto const names    = columns.GetColumnNames();
  auto const types    = columns.GetColumnTypes();

  auto const it = std::find(names.begin(), names.end(), side.column);
  if (it == names.end()) {
    throw duckdb::InternalException("sirius_knn_join: vector column '" + side.column +
                                    "' vanished between bind and plan");
  }
  auto const vec_idx = static_cast<duckdb::idx_t>(std::distance(names.begin(), it));

  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  auto scan_function = table_entry.GetScanFunction(context, bind_data);

  // Vector column first, then the columns the join emits. Materialize reads those in this same
  // batch order, so they have to travel in the batches the fold numbered -- not be re-read from
  // the table afterwards.
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.emplace_back(vec_idx);
  duckdb::vector<duckdb::LogicalType> projected_types{types[vec_idx]};
  for (auto const& out_col : side.output_columns) {
    auto const out_it = std::find(names.begin(), names.end(), out_col);
    if (out_it == names.end()) {
      throw duckdb::InternalException("sirius_knn_join: output column '" + out_col +
                                      "' vanished between bind and plan");
    }
    auto const idx = static_cast<duckdb::idx_t>(std::distance(names.begin(), out_it));
    column_ids.emplace_back(idx);
    projected_types.push_back(types[idx]);
  }
  if (!extra_column.empty()) {
    auto const extra_it = std::find(names.begin(), names.end(), extra_column);
    if (extra_it == names.end()) {
      throw duckdb::InternalException("sirius_knn_join: cluster column '" + extra_column +
                                      "' vanished between bind and plan");
    }
    auto const extra_idx = static_cast<duckdb::idx_t>(std::distance(names.begin(), extra_it));
    column_ids.emplace_back(extra_idx);
    projected_types.push_back(types[extra_idx]);
  }

  return duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
    sirius::from_duckdb_vec(projected_types),
    scan_function,
    std::move(bind_data),
    sirius::from_duckdb_vec(types),
    std::move(column_ids),
    duckdb::vector<std::size_t>{},
    names,
    /*table_filters=*/nullptr,
    table_entry.GetStorage().GetTotalRows(),
    duckdb::ExtraOperatorInfo{},
    duckdb::vector<duckdb::Value>{},
    table_entry.GetVirtualColumns());
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
sirius_physical_plan_generator::create_streaming_source_plan(duckdb::LogicalGet& op)
{
  auto const* bind = dynamic_cast<sirius::exec::stream_source_bind_data const*>(op.bind_data.get());
  if (bind == nullptr) {
    throw duckdb::InternalException("sirius_stream_source is missing its stream bind data");
  }

  auto catalog        = sirius::exec::catalog_for(context);
  auto const& binding = catalog->get(bind->stream_id);

  // Projection pushdown is off; a narrowed column list here would disagree with the binder.
  auto column_ids = op.GetColumnIds();
  if (column_ids.size() != binding.types.size()) {
    throw duckdb::NotImplementedException(
      "sirius_stream_source: column projection into a stream read is not supported (stream "
      "declares %llu columns, plan requests %llu)",
      static_cast<unsigned long long>(binding.types.size()),
      static_cast<unsigned long long>(column_ids.size()));
  }

  auto source = duckdb::make_uniq<sirius::op::sirius_physical_streaming_source>(
    binding.types, op.EstimateCardinality(context), binding.repository, binding.expected_senders);

  // Plan owns op; catalog.built back-pointer for session registration.
  catalog->set_built(bind->stream_id, source.get());
  return source;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalGet& op)
{
  // sirius_knn_join produces its join result from two pinned tables, so it gets its own leaf
  // builder rather than being routed through the scan path.
  if (op.function.name == "sirius_knn_join" || op.function.name == "sirius_knn_join_rel") {
    return create_plan_knn_join(op);
  }

  auto column_ids = op.GetColumnIds();

  // Only GPU-route known table scan functions; all others (pragma, system catalog
  // functions, etc.) must fall back to CPU.
  static const std::unordered_set<std::string> kSupportedScanFunctions = {
    "seq_scan",
    "parquet_scan",
    "read_parquet",
    "sirius_read_parquet",
    "iceberg_scan",
    sirius::exec::kStreamSourceFunctionName};
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
  // Both probes are OWNING: the shared_ptr holds the entry alive across the guards below, so a
  // concurrent UNPIN on another connection cannot invalidate it mid-check. `pinned` is the
  // shared raw view over whichever probe matched.
  std::shared_ptr<sirius::scan_manager::pinned_entry const> pinned_owner;
  sirius::scan_manager::pinned_entry const* pinned = nullptr;
  bool serves_insert_deltas                        = false;
  bool mvcc_pin_serves_scan                        = false;
  if (sirius_state && op.function.name == "seq_scan") {
    auto* bind = dynamic_cast<duckdb::TableScanBindData*>(op.bind_data.get());
    if (bind != nullptr && bind->table.IsDuckTable()) {
      auto& table  = bind->table.Cast<duckdb::DuckTableEntry>();
      pinned_owner = sirius_state->get_scan_manager().find_pinned_entry_for_duckdb_table(
        table.ParentCatalog().GetName(),
        table.ParentSchema().name,
        table.name,
        &column_ids,
        &op.returned_types);
      pinned = pinned_owner.get();
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
      pinned_owner = sirius_state->get_scan_manager().find_pinned_entry_for_parquet_files(files);
      pinned       = pinned_owner.get();
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

  // STREAMING_SOURCE leaf: fragment-declared repo + senders, not a file scan.
  if (op.function.name == sirius::exec::kStreamSourceFunctionName) {
    return create_streaming_source_plan(op);
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

      // Plan and serve must judge recorded types against the scan's the same way. Treat a
      // mismatched pin as unpinned so the fresh disk path remains available.
      if (pinned != nullptr && pinned->mvcc != nullptr &&
          sirius::scan_manager::pinned_native_types_match_columns(
            *pinned, column_ids, op.returned_types)) {
        // Cache-or-CPU guards: while this table is MVCC-pinned, a GPU plan
        // either serves exactly from the pinned cache (DELETE keep-masks) or is
        // refused HERE, where the throw still becomes a clean CPU fallback. The
        // disk-native path is MVCC-blind, and the pin's checkpoint suppression
        // makes its snapshot increasingly stale — so scans the pin cannot serve
        // never fall through to it.

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
        // Every guard passed, so the pinned entry serves this scan.
        mvcc_pin_serves_scan = true;
#if 0
        // Disabled — these guards walk every row group of the table at plan
        // time, per query. With this block off, (d) above has no clean-table
        // relaxation. The update-chain branch is redundant because UPDATE
        // statements on pinned tables are rejected before execution; direct
        // UPDATE serving remains out of scope here (#1162).
        auto const n_cache = pinned->mvcc->n_cache();
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
    node->named_parameters     = std::move(op.named_parameters);
    node->mvcc_pin_serves_scan = mvcc_pin_serves_scan;
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
    if (sirius_state) {
      sirius_state->get_event_publisher().publish_compressed_materialization(
        sirius::event::compressed_materialization_activity::scan_sidecar_installed);
    }
  }
  node->named_parameters     = std::move(op.named_parameters);
  node->mvcc_pin_serves_scan = mvcc_pin_serves_scan;
  if (filter) {
    filter->children.push_back(std::move(node));
    return filter;
  }
  return std::move(node);
}

// A view corpus: bind its SELECT, optimize it, plan it on the GPU and reorder its columns into
// the layout the fold expects (vector, emitted columns, cluster column) -- the shape
// make_side_scan produces from a base table. The nested bind runs under the internal-query
// guard so the transparent optimizer hook does not recurse into it.
duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::make_view_side(const sirius::vss::vector_join_side& side,
                                               const std::string& extra_column)
{
  duckdb::SiriusContext::InternalQueryGuard guard(context);
  auto bound  = sirius::vss::bind_view_select(context, side);
  auto binder = duckdb::Binder::CreateBinder(context);
  duckdb::Optimizer optimizer(*binder, context);
  auto plan = optimizer.Optimize(std::move(bound.plan));
  duckdb::ColumnBindingResolver resolver;
  resolver.VisitOperator(*plan);
  plan->ResolveOperatorTypes();
  auto const child_types = plan->types;
  auto const card        = plan->estimated_cardinality;
  auto planned           = create_plan(*plan);

  auto index_of = [&](const std::string& col) -> std::size_t {
    for (std::size_t i = 0; i < bound.names.size(); ++i) {
      if (bound.names[i] == col) { return i; }
    }
    throw duckdb::InternalException("sirius_knn_join: view column '" + col +
                                    "' vanished between bind and plan");
  };
  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
  auto add = [&](std::size_t idx) {
    types.push_back(child_types[idx]);
    exprs.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(child_types[idx], idx));
  };
  add(index_of(side.column));
  for (auto const& col : side.output_columns) {
    add(index_of(col));
  }
  if (!extra_column.empty()) { add(index_of(extra_column)); }
  return push_projection(std::move(planned),
                         sirius::from_duckdb_vec(types),
                         translate_expressions(std::move(exprs)),
                         card);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan_knn_join(duckdb::LogicalGet& op)
{
  auto const& bind_data_probe = op.bind_data->Cast<sirius::vss::SiriusVectorJoinBindData>();
  if (!op.children.empty() && !bind_data_probe.probe_is_relation) {
    throw duckdb::NotImplementedException("sirius_knn_join does not take table inputs");
  }
  auto const expected_children = static_cast<std::size_t>(bind_data_probe.probe_is_relation) +
                                 static_cast<std::size_t>(bind_data_probe.req.right.from_relation);
  if (op.children.size() != expected_children) {
    throw duckdb::NotImplementedException(
      "sirius_knn_join: expected " + std::to_string(expected_children) + " input relations, got " +
      std::to_string(op.children.size()));
  }
  op.ResolveOperatorTypes();

  auto const& bind_data = op.bind_data->Cast<sirius::vss::SiriusVectorJoinBindData>();

  // The search stage reads both pinned tables directly, so it needs the scan
  // manager to resolve them at execution time.
  auto sirius_state = context.registered_state
                        ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                        : nullptr;
  if (!sirius_state) {
    throw duckdb::InternalException(
      "sirius_knn_join requires the Sirius context to be initialized");
  }
  auto& scan_manager = sirius_state->get_scan_manager();

  // If k is larger than the number of rows in the right table, lower it to that row count.
  // The count comes from the bind, which reads it from the pin or from the table itself
  // depending on where this join's corpus is coming from.
  auto req              = bind_data.req;
  auto const right_rows = static_cast<std::int64_t>(bind_data.right_rows);

  // Corpus predicates the table function took off a filter above it (see
  // SiriusVectorJoinPushdownFilter), keyed by output column. Only right-side columns are ever
  // pushed, and only as constant comparisons or conjunctions of them.
  for (auto const& [out, filter] : op.table_filters.filters) {
    auto const n_left_out = req.left.output_columns.size();
    if (out < n_left_out || out >= n_left_out + req.right.output_columns.size()) {
      throw duckdb::InternalException("sirius_knn_join: a table filter on a non-corpus column");
    }
    auto const& column = req.right.output_columns[out - n_left_out];
    auto add           = [&](duckdb::TableFilter const& f) {
      if (f.filter_type != duckdb::TableFilterType::CONSTANT_COMPARISON) {
        throw duckdb::InternalException("sirius_knn_join: unsupported pushed corpus filter");
      }
      auto const& c = f.Cast<duckdb::ConstantFilter>();
      using cmp     = sirius::vss::corpus_predicate::op;
      cmp op_kind;
      switch (c.comparison_type) {
        case duckdb::ExpressionType::COMPARE_EQUAL: op_kind = cmp::eq; break;
        case duckdb::ExpressionType::COMPARE_NOTEQUAL: op_kind = cmp::ne; break;
        case duckdb::ExpressionType::COMPARE_LESSTHAN: op_kind = cmp::lt; break;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO: op_kind = cmp::le; break;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN: op_kind = cmp::gt; break;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: op_kind = cmp::ge; break;
        default: throw duckdb::InternalException("sirius_knn_join: unsupported pushed comparison");
      }
      req.right_predicates.push_back({column, op_kind, c.constant});
    };
    if (filter->filter_type == duckdb::TableFilterType::CONJUNCTION_AND) {
      for (auto const& child : filter->Cast<duckdb::ConjunctionAndFilter>().child_filters) {
        add(*child);
      }
    } else {
      add(*filter);
    }
  }
  // k is per left row except in the global mode, where it counts pairs across all left rows and
  // can exceed the right table (the stream stage clamps each row's search depth on its own).
  if (right_rows > 0 && req.k > right_rows &&
      req.mode != sirius::vss::vector_join_mode::global_top_k) {
    req.k = right_rows;
  }

  // Three-stage pipeline: select (per-pair top-k) → reduce_local (per-left-batch
  // reduction) → materialize (gather output columns + score into the TVF rows).
  // The select/reduce stages carry a [neighbor_id BIGINT, distance FLOAT] schema;
  // only materialize emits the TVF's declared columns.
  // Per-pair partials, before any left row is resolved: [neighbor_id, distance].
  auto partial_types = []() {
    duckdb::vector<sirius::logical_type> t;
    t.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    t.push_back(sirius::logical_type::make(sirius::type_id::FLOAT));
    return t;
  };

  // What the join stage hands to materialize: [left_row, neighbor_id, distance].
  //
  // left_row is carried explicitly rather than implied by position. Under per-row top-k the
  // result is exactly k rows per left row and the left index can be recovered as row/k, which
  // is what materialize used to do. Threshold and global top-k break that: their output is
  // ragged by construction, so position carries no information and the left row has to travel
  // with the pair. Making it explicit for every mode is what keeps one materialize.
  auto joined_types = []() {
    duckdb::vector<sirius::logical_type> t;
    t.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
    t.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    t.push_back(sirius::logical_type::make(sirius::type_id::FLOAT));
    return t;
  };

  // The fused streaming operator folds each right batch into a running top-k instead
  // of emitting a partial per (left, right) pair for a separate reduce stage. Both
  // paths emit the same [neighbor_id, distance] schema partitioned by left batch, so
  // materialize is shared and the two can be compared directly on the same query.
  //
  // Streaming is the default. The split path is kept for A/B comparison only and is
  // known to be wrong whenever the right table pins to more than one chunk: reduce_local
  // merges each partial on its own instead of folding the partition, and materialize's
  // fixed-k slicing then attributes neighbours to the wrong left rows. Select it with
  // SIRIUS_VECTOR_JOIN_STREAMING=0, and only on a single-chunk right table.
  const char* streaming_env = std::getenv("SIRIUS_VECTOR_JOIN_STREAMING");
  bool const use_streaming  = streaming_env == nullptr || std::string_view{streaming_env} != "0";

  // Projection pushdown. The declared output is
  // [left_output_columns..., right_output_columns..., score]; column_ids names the subset the
  // query reads, narrowed by DuckDB because the function sets projection_pushdown. Dropping the
  // rest here is not a cosmetic saving: materialize concatenates the corpus's output columns
  // across the WHOLE corpus to gather by neighbour id, so an unread column costs O(corpus)
  // device memory on the path whose premise is that the corpus need not fit.
  auto const declared_left  = req.left.output_columns.size();
  auto const declared_right = req.right.output_columns.size();
  auto const score_idx_orig = declared_left + declared_right;

  // What materialize emits is grouped by side -- [kept_left..., kept_right..., score] -- while
  // what the query wants is one entry per `projection_ids` (or per `column_ids` when that is
  // empty), in that order. Those two orders coincide only when column_ids happens to be
  // ascending, so the two are tracked separately here and reconciled by a projection at the end.
  auto const& column_ids = op.GetColumnIds();
  std::size_t n_left     = 0;
  std::size_t n_right    = 0;
  bool score_read        = false;
  for (auto const& column_id : column_ids) {
    if (!column_id.HasPrimaryIndex()) {
      throw duckdb::NotImplementedException("sirius_knn_join: virtual/rowid columns unsupported");
    }
    auto const idx = column_id.GetPrimaryIndex();
    if (idx < declared_left) {
      ++n_left;
    } else if (idx < score_idx_orig) {
      ++n_right;
    } else {
      score_read = true;
    }
  }

  std::vector<std::string> kept_left;
  std::vector<std::string> kept_right;
  kept_left.reserve(n_left);
  kept_right.reserve(n_right);
  // The score is always produced -- it is one column of the output rows, not of the corpus --
  // and projected away below when unread.
  duckdb::vector<duckdb::LogicalType> kept_types(n_left + n_right + 1);
  // Physical position, in the emitted layout, of each entry of column_ids.
  std::vector<std::size_t> emitted_pos(column_ids.size());
  for (std::size_t ci = 0; ci < column_ids.size(); ++ci) {
    auto const idx  = column_ids[ci].GetPrimaryIndex();
    std::size_t pos = 0;
    if (idx < declared_left) {
      pos = kept_left.size();
      kept_left.push_back(req.left.output_columns[idx]);
    } else if (idx < score_idx_orig) {
      pos = n_left + kept_right.size();
      kept_right.push_back(req.right.output_columns[idx - declared_left]);
    } else {
      pos = n_left + n_right;
    }
    kept_types[pos] = op.returned_types[idx];
    emitted_pos[ci] = pos;
  }
  kept_types[n_left + n_right] = op.returned_types[score_idx_orig];
  SIRIUS_LOG_DEBUG(
    "[vector_join] output columns: declared left={} right={}, read {} -> keeping left={} "
    "right={} score={}",
    declared_left,
    declared_right,
    op.GetColumnIds().size(),
    kept_left.size(),
    kept_right.size(),
    score_read);
  req.score_read           = score_read;
  req.left.output_columns  = std::move(kept_left);
  req.right.output_columns = std::move(kept_right);

  // Row orders for the fed sides, minted here so the fold and materialize resolve positions
  // against one list per side rather than each deriving one of their own.
  std::shared_ptr<sirius::vss::materialized_side_buffer> build_side =
    req.build_from_scan ? std::make_shared<sirius::vss::materialized_side_buffer>() : nullptr;
  std::shared_ptr<sirius::vss::materialized_side_buffer> probe_side =
    req.probe_from_scan ? std::make_shared<sirius::vss::materialized_side_buffer>() : nullptr;

  if ((build_side || probe_side) && !use_streaming) {
    throw duckdb::NotImplementedException(
      "sirius_knn_join: build_source / probe_source => 'scan' need the streaming operator; the "
      "split path (SIRIUS_VECTOR_JOIN_STREAMING=0) reads both sides from pins only");
  }

  // The relational surface hands us the probe already bound and optimized -- filters pushed
  // into its scan, projections trimmed -- so it is planned as-is and only reordered into the
  // layout the fold expects: vector column first, then the columns the join emits.
  // A side given as a child relation: resolve its bindings (ColumnBindingResolver stops at a
  // LOGICAL_GET and does not descend into its children, which is why DuckDB's own plan_get calls
  // ResolveAndPlan on the child -- and so must this), plan it, and project it to the
  // [vector, outputs...] layout a scanned side has.
  auto plan_relation_side = [&](duckdb::LogicalOperator& child,
                                const sirius::vss::vector_join_side& side,
                                const duckdb::vector<duckdb::string>& fallback_names) {
    duckdb::ColumnBindingResolver child_resolver;
    child_resolver.VisitOperator(child);
    duckdb::vector<duckdb::LogicalType> child_types = child.types;
    auto planned                                    = create_plan(child);

    auto index_of = [&](const std::string& col) -> std::size_t {
      auto const& names = side.relation_columns.empty()
                            ? std::vector<std::string>(fallback_names.begin(), fallback_names.end())
                            : side.relation_columns;
      for (std::size_t i = 0; i < names.size(); ++i) {
        if (names[i] == col) { return i; }
      }
      throw duckdb::InternalException("sirius_knn_join: column '" + col +
                                      "' vanished between bind and plan");
    };

    duckdb::vector<duckdb::LogicalType> types;
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
    auto add = [&](std::size_t idx) {
      types.push_back(child_types[idx]);
      exprs.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(child_types[idx], idx));
    };
    add(index_of(side.column));
    for (auto const& col : side.output_columns) {
      add(index_of(col));
    }
    return push_projection(std::move(planned),
                           sirius::from_duckdb_vec(types),
                           translate_expressions(std::move(exprs)),
                           child.estimated_cardinality);
  };

  std::optional<duckdb::unique_ptr<sirius::op::sirius_physical_operator>> probe_child;
  std::optional<duckdb::unique_ptr<sirius::op::sirius_physical_operator>> corpus_child;
  if (bind_data.probe_is_relation) {
    probe_child = plan_relation_side(*op.children[0], req.left, op.input_table_names);
  }
  if (req.right.from_relation) {
    corpus_child = plan_relation_side(*op.children.back(), req.right, {});
  }

  // With a relational probe the bind could not know the row count, so the cardinality callback
  // declined and DuckDB fell back to the child's estimate -- the join's input, not its output.
  // The child's estimate is in hand here, so the k is applied where it can be.
  if (probe_child.has_value() && !req.right.from_relation) {
    auto const child_rows = static_cast<std::size_t>(op.children[0]->estimated_cardinality);
    auto const k          = static_cast<std::size_t>(std::max<std::int64_t>(req.k, 0));
    op.estimated_cardinality =
      req.mode == sirius::vss::vector_join_mode::global_top_k ? k : child_rows * k;
  }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> join_stage;
  duckdb::SiriusContext* join_ctx = nullptr;  // the materialize stage restores disk-resident pieces through it
  if (use_streaming) {
    // Resolved here rather than inside the operator: the operator holds only a scan manager and
    // has no route to the session's index cache, while the planner does. The cache owns the
    // centroids, so the operator borrows them for the life of the query.
    const cudf::column* centroids = nullptr;
    // Borrowed for the life of the query: the context is the session's, the clustered path
    // reports what it pruned to it, and a streamed corpus asks its downgrade executor for room.
    auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    duckdb::SiriusContext* prune_stats = sirius_ctx.get();
    join_ctx                           = prune_stats;
    if (!req.clustering.empty()) {
      if (!sirius_ctx) {
        throw duckdb::InvalidInputException(
          "sirius_knn_join: clustering requires the Sirius context to be initialized");
      }
      centroids = sirius::vss::find_clustering_centroids(*sirius_ctx, req.clustering);
      if (centroids == nullptr) {
        throw duckdb::InvalidInputException("sirius_knn_join: no clustering named '" +
                                            req.clustering + "'; run sirius_kmeans_fit first");
      }
    }
    auto stream_op =
      duckdb::make_uniq<sirius::op::sirius_physical_vector_join_stream>(joined_types(),
                                                                        op.estimated_cardinality,
                                                                        req,
                                                                        &scan_manager,
                                                                        build_side,
                                                                        probe_side,
                                                                        centroids,
                                                                        prune_stats);
    // Probe first, then corpus: wrap_vector_join walks the children in that order to decide
    // which is the build side, matching wrap_join's probe=0 / build=1 convention.
    if (req.probe_from_scan) {
      stream_op->children.push_back(probe_child.has_value() ? std::move(*probe_child)
                                                            : make_side_scan(context, req.left));
    }
    if (req.build_from_scan) {
      // The cluster ids ride along with the corpus so the fold can read them from the same
      // batches it searches. Anything else -- a second scan, a pin behind the scan -- would be
      // a different row order than the one the build side's snapshot fixed.
      stream_op->children.push_back(
        corpus_child.has_value() ? std::move(*corpus_child)
        : req.right.is_view      ? make_view_side(req.right, req.build_cluster_column)
                                 : make_side_scan(context, req.right, req.build_cluster_column));
    }
    join_stage = std::move(stream_op);
  } else {
    auto selection = duckdb::make_uniq<sirius::op::sirius_physical_vector_join_select>(
      partial_types(), op.estimated_cardinality, req, &scan_manager);

    auto reduce_local = duckdb::make_uniq<sirius::op::sirius_physical_vector_join_reduce_local>(
      joined_types(), op.estimated_cardinality, req.k);

    reduce_local->children.push_back(std::move(selection));
    join_stage = std::move(reduce_local);
  }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> node =
    duckdb::make_uniq<sirius::op::sirius_physical_vector_join_materialize>(
      sirius::from_duckdb_vec(kept_types),
      op.estimated_cardinality,
      req,
      &scan_manager,
      build_side,
      probe_side,
      join_ctx);
  node->children.push_back(std::move(join_stage));

  // Global top-k finishes above materialize rather than inside the join. The join stage
  // searches every left row to depth k, which bounds the candidates correctly (one left row
  // can own at most k of the global winners) but leaves one top-k per left batch. Ranking the
  // materialized rows by score with an ordinary TOP_N collapses those into a single answer,
  // and the plan generator's post-pass wraps any TOP_N with MERGE_TOP_N, which is exactly the
  // cross-partition merge this needs — so left-side partitioning is not a restriction here.
  if (req.mode == sirius::vss::vector_join_mode::global_top_k) {
    // The score is last in what materialize EMITS, which is narrower than the declared
    // output whenever DuckDB reads a subset -- indexing the declared width put this past
    // the end of the table and surfaced as "TopN order index out of range".
    auto const score_idx = n_left + n_right;
    auto const ascending =
      req.output_type == sirius::vss::vector_join_output_type::similarity ? false : true;

    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.emplace_back(
      ascending ? duckdb::OrderType::ASCENDING : duckdb::OrderType::DESCENDING,
      duckdb::OrderByNullType::NULLS_LAST,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(kept_types[score_idx], score_idx));

    auto top_n =
      duckdb::make_uniq<sirius::op::sirius_physical_top_n>(sirius::from_duckdb_vec(kept_types),
                                                           std::move(orders),
                                                           static_cast<std::size_t>(req.k),
                                                           /*offset=*/std::size_t{0},
                                                           /*dynamic_filter=*/nullptr,
                                                           op.estimated_cardinality);
    top_n->children.push_back(std::move(node));
    node = std::move(top_n);
  }

  // Reconcile the emitted layout with the requested one. `projection_ids`, when set, holds
  // indices INTO column_ids and is what LogicalGet::GetColumnBindings/ResolveTypes use, so it
  // -- not column_ids -- defines both the order and the width of this operator's output.
  duckdb::vector<duckdb::idx_t> requested;
  if (op.projection_ids.empty()) {
    for (duckdb::idx_t i = 0; i < column_ids.size(); ++i) {
      requested.push_back(i);
    }
  } else {
    requested = op.projection_ids;
  }

  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  types.reserve(requested.size());
  for (auto const request_idx : requested) {
    if (request_idx >= emitted_pos.size()) {
      throw duckdb::InternalException(
        "sirius_knn_join: projection_ids entry %llu is out of range "
        "for %llu read columns",
        static_cast<std::uint64_t>(request_idx),
        static_cast<std::uint64_t>(emitted_pos.size()));
    }
    auto const pos = emitted_pos[request_idx];
    types.push_back(kept_types[pos]);
    expressions.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(kept_types[pos], pos));
  }

  // The schema this operator hands upwards has to be exactly what the binder resolved, because
  // everything above binds by POSITION: a mismatch does not fail, it silently reads the wrong
  // column. That is how a `WHERE distance <= x` under a `GROUP BY` came to evaluate the
  // predicate against `left_id`. Check it here rather than trusting the layout to line up.
  if (types.size() != op.types.size()) {
    throw duckdb::InternalException(
      "sirius_knn_join: emitting %llu columns but the plan expects %llu",
      static_cast<std::uint64_t>(types.size()),
      static_cast<std::uint64_t>(op.types.size()));
  }
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (types[i] != op.types[i]) {
      throw duckdb::InternalException(
        "sirius_knn_join: output column %llu is %s but the plan expects %s",
        static_cast<std::uint64_t>(i),
        types[i].ToString(),
        op.types[i].ToString());
    }
  }

  // Nothing to do when the emitted layout already is the requested one -- the common case,
  // and the only one the operator used to handle.
  bool is_identity = requested.size() == kept_types.size();
  for (std::size_t i = 0; is_identity && i < requested.size(); ++i) {
    is_identity = emitted_pos[requested[i]] == i;
  }
  if (is_identity) { return node; }

  return push_projection(std::move(node),
                         sirius::from_duckdb_vec(types),
                         translate_expressions(std::move(expressions)),
                         op.estimated_cardinality);
}

}  // namespace sirius::planner
