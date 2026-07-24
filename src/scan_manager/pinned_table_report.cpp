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

#include "scan_manager/pinned_table_report.hpp"

#include "log/logging.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>

#include <algorithm>
#include <numeric>

namespace sirius::scan_manager {

namespace {

/// Best-effort live-table probes: delta row counts and per-snapshot dirty
/// chunks. Any failure (table dropped, snapshot older than the pin, capture
/// validation) leaves the fields absent — the report never throws.
void fill_live_table_state(pinned_table_report& report,
                           pinned_entry const& entry,
                           duckdb::ClientContext& context)
{
  try {
    // Non-template lookup + Cast: the templated GetEntry<DuckTableEntry> would
    // ODR-use a static constexpr against libduckdb_static (see PinTableFunction).
    auto& entry_base = duckdb::Catalog::GetEntry(context,
                                                 duckdb::CatalogType::TABLE_ENTRY,
                                                 report.database_name,
                                                 report.schema_name,
                                                 report.table_name);
    auto& storage    = entry_base.Cast<duckdb::DuckTableEntry>().GetStorage();

    // Schema drift: the live DataTable differs from the pinned one (structural
    // ALTER / DROP replaces it) — the same check the plan-time guard makes.
    auto const pinned_storage = entry.mvcc->pin_storage.lock();
    report.stale = !pinned_storage || pinned_storage.get() != &storage || !storage.IsMainTable();

    auto const n_cache       = entry.mvcc->n_cache();
    auto const n_total       = static_cast<std::size_t>(storage.GetTotalRows());
    report.delta_insert_rows = n_total > n_cache ? n_total - n_cache : 0;

    auto const plan     = op::scan::capture_mvcc_visibility_plan(storage, context, *entry.mvcc);
    report.dirty_chunks = static_cast<std::size_t>(
      std::count(plan.chunk_has_version_state.begin(), plan.chunk_has_version_state.end(), true));
    report.delta_delete_rows = op::scan::count_invisible_pinned_rows(plan);
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG(
      "[pinned_table_report] live-table state unavailable for pin '{}': {}", report.name, e.what());
  }
}

}  // namespace

pinned_table_report make_pinned_table_report(std::string_view name,
                                             pinned_entry const& entry,
                                             duckdb::ClientContext& context)
{
  pinned_table_report report;
  report.name          = std::string(name);
  report.format        = entry.cache_info.table_name.empty() ? "parquet" : "duckdb";
  report.database_name = entry.cache_info.catalog_name;
  report.schema_name   = entry.cache_info.schema_name;
  report.table_name    = entry.cache_info.table_name;
  report.tier          = entry.tier == cucascade::memory::Tier::GPU ? "gpu" : "host";
  report.column_count  = entry.cache_info.column_ids.size();
  report.chunk_count =
    entry.tier == cucascade::memory::Tier::GPU
      ? (entry.data_batches_by_column.empty() ? 0
                                              : entry.data_batches_by_column.begin()->second.size())
      : entry.host_chunks.size();
  report.base_rows = entry.mvcc != nullptr ? entry.mvcc->n_cache() : entry.num_rows;

  try {
    std::vector<std::size_t> all_columns(report.column_count);
    std::iota(all_columns.begin(), all_columns.end(), std::size_t{0});
    validate_pinned_entry_for_serving(entry, all_columns);
    report.is_valid = true;
  } catch (std::exception const&) {
    report.is_valid = false;
  }

  if (entry.mvcc != nullptr) {
    report.v_base          = entry.mvcc->v_base;
    report.promoted_chunks = entry.mvcc->promotion.promoted_chunks;
    report.promoted_rows   = entry.mvcc->promotion.promoted_rows;
    if (!entry.mvcc->promotion.disabled_reason.empty()) {
      report.promotion_status = "disabled: " + entry.mvcc->promotion.disabled_reason;
    } else if (!entry.mvcc->promotion.last_skip_reason.empty()) {
      report.promotion_status = "last skip: " + entry.mvcc->promotion.last_skip_reason;
    }
    fill_live_table_state(report, entry, context);
  }
  return report;
}

std::vector<pinned_table_report> collect_pinned_table_reports(sirius_scan_manager const& manager,
                                                              duckdb::ClientContext& context)
{
  std::vector<pinned_table_report> reports;
  manager.visit_pinned_entries([&](std::string_view name, pinned_entry const& entry) {
    reports.push_back(make_pinned_table_report(name, entry, context));
    return true;
  });
  return reports;
}

}  // namespace sirius::scan_manager
