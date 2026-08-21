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
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/attached_database.hpp"
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
#include "log/logging.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace sirius::planner {

namespace {

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

// Plan-time counterpart of the DuckDB serve-time native-type gate. Every storage-backed projected
// column must retain its pin-time native mapping across all chunks. Column coverage remains the
// cache-serviceability guard's responsibility; an empty matrix passes for zero-chunk compatibility.
[[nodiscard]] bool pinned_native_types_match_columns(
  sirius::scan_manager::pinned_entry const& entry,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::LogicalType> const& returned_types)
{
  if (entry.column_storage.empty()) { return true; }

  std::unordered_map<duckdb::idx_t, std::size_t> entry_pos_by_primary;
  entry_pos_by_primary.reserve(entry.cache_info.column_ids.size());
  for (std::size_t i = 0; i < entry.cache_info.column_ids.size(); ++i) {
    entry_pos_by_primary.emplace(entry.cache_info.column_ids[i].GetPrimaryIndex(), i);
  }

  for (auto const& col_idx : column_ids) {
    if (!col_idx.HasPrimaryIndex() || col_idx.IsRowIdColumn() || col_idx.IsVirtualColumn() ||
        col_idx.IsEmptyColumn()) {
      continue;
    }
    auto const primary = col_idx.GetPrimaryIndex();
    auto const it      = entry_pos_by_primary.find(primary);
    if (it == entry_pos_by_primary.end()) { continue; }
    std::optional<cudf::data_type> native;
    if (primary < returned_types.size()) {
      native = sirius::try_get_cudf_type(sirius::from_duckdb(returned_types[primary]));
    }
    if (!native) { return false; }
    for (auto const& row : entry.column_storage) {
      if (it->second >= row.size() || row[it->second].native != *native) { return false; }
    }
  }
  return true;
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
    "seq_scan", "parquet_scan", "read_parquet", "sirius_read_parquet"};
  if (kSupportedScanFunctions.find(op.function.name) == kSupportedScanFunctions.end()) {
    throw duckdb::NotImplementedException("Table function '%s' is not supported in Sirius",
                                          op.function.name);
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

      // Plan and serve must classify native-type drift identically. Treat a mismatched pin as
      // unpinned so the fresh disk path remains available.
      if (pinned != nullptr && pinned->mvcc != nullptr &&
          pinned_native_types_match_columns(*pinned, column_ids, op.returned_types)) {
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
  if (filter) {
    filter->children.push_back(std::move(node));
    return filter;
  }
  return std::move(node);
}

}  // namespace sirius::planner
