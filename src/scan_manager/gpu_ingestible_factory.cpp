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

#include "scan_manager/gpu_ingestible_factory.hpp"

#include "log/logging.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/pinned_table_gpu_ingestible.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

gpu_ingestible_factory::gpu_ingestible_factory(
  std::unordered_map<std::string, pinned_entry> const& pinned_entries) noexcept
  : _pinned_entries(pinned_entries)
{
}

std::shared_ptr<io::gpu_ingestible> gpu_ingestible_factory::produce(
  std::unique_ptr<io::ingestible_table_info> table_info,
  sirius_scan_manager const& mgr,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces,
  std::size_t op_id)
{
  if (!table_info) { return nullptr; }

  // Cache short-circuit: peeks file_paths(), and on a hit steals table_info
  // into the cached ingestible. On a miss, table_info stays valid for
  // io::make_gpu_ingestible.
  if (auto cached = try_cached(table_info, gpu_memory_spaces, op_id)) { return cached; }
  return io::make_gpu_ingestible(std::move(table_info), mgr, gpu_ioctxs);
}

std::shared_ptr<io::gpu_ingestible> gpu_ingestible_factory::try_cached(
  std::unique_ptr<io::ingestible_table_info>& table_info,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces,
  std::size_t op_id) const
{
  if (!table_info) { return nullptr; }

  // Cache is parquet-only today (no pin_duckdb_table path). Cast probe;
  // non-parquet table_info falls through.
  auto const* parquet_info =
    dynamic_cast<op::scan::parquet_ingestible_table_info const*>(table_info.get());
  if (parquet_info == nullptr) { return nullptr; }
  auto const& info = *parquet_info;  // alias so the cache body reads unchanged

  // If a pinned entry's file paths match this table_info, build the same
  // scan_plan the parquet path would build and serve the scan from cache.
  auto matches_scan_info = [&info](const pinned_entry& entry) {
    if (entry.file_paths.size() != info.resolved_file_paths.size()) { return false; }
    auto sorted_a = entry.file_paths;
    auto sorted_b = info.resolved_file_paths;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    return sorted_a == sorted_b;
  };
  try {
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      if (!matches_scan_info(entry)) { continue; }
      // A partial pin (pin_table(..., n_rows=N) capped below the full file
      // content) MUST NOT serve cached reads — the incoming table_info
      // carries no n_rows budget, so a partial-entry hit would silently
      // mask missing rows. Fall through to the per-format path.
      if (entry.is_partial) {
        SIRIUS_LOG_DEBUG(
          "[gpu_ingestible_factory::try_cached] pinned entry '{}' matches op_id={} but is "
          "partial (row-count budget at pin time); falling through to per-format ingestible",
          pinned_name,
          op_id);
        break;
      }

      // Build the canonical scan_plan once. Everything downstream — cached
      // column layout, filter pushdown indices, post-read assembly — reads
      // from this. Held by shared_ptr<const> so each emitted operator_data
      // can carry it to the gpu scan operator's per-task assembly check
      // without copying.
      auto plan_shared = std::make_shared<op::scan::scan_plan const>(
        op::scan::build_scan_plan(info.column_ids,
                                  info.projection_ids,
                                  info.names,
                                  info.returned_types,
                                  info.scan_output_arity,
                                  info.partition_indices));
      auto const& plan = *plan_shared;

      // The post-decode dynamic-filter operator filters this cached scan too; producers reference
      // probe columns in column_ids space, so install the same column_ids → output-position
      // translation the disk ingestible does. Idempotent with the parquet ctor on a fall-through.
      if (info.sirius_dynamic_filters) {
        info.sirius_dynamic_filters->set_consumer_column_remap(plan.output_position_by_column_id);
      }

      // Hive partitions on a cached scan would require per-chunk file_path
      // metadata that pinned entries don't carry today. Fall through to
      // the per-format path, which extracts partition values per file at
      // read time.
      if (plan.has_partitions()) {
        SIRIUS_LOG_DEBUG(
          "[gpu_ingestible_factory::try_cached] pinned entry '{}' matches op_id={} but scan "
          "has hive partitions; falling through to per-format ingestible",
          pinned_name,
          op_id);
        break;
      }

      // Filter expression: BoundReferences are in D-space, via
      // plan.batch_position_by_column_id. Same recipe parquet's run_batch
      // uses, so the filter evaluates correctly against the cached batch
      // (which is in D-order by construction above). Built before the
      // tier-specific assembly so both branches share the same filter.
      std::shared_ptr<duckdb::Expression> filter_expression;
      if (info.table_filters && !info.table_filters->filters.empty()) {
        auto duckdb_expression =
          op::convert_table_filters_to_expression(*info.table_filters,
                                                  info.column_ids,
                                                  info.returned_types,
                                                  plan.batch_position_by_column_id,
                                                  plan.partition_primary_indices);
        if (duckdb_expression) {
          filter_expression = std::shared_ptr<duckdb::Expression>(std::move(duckdb_expression));
        }
      }

      if (entry.tier == cucascade::memory::Tier::HOST) {
        // HOST-tier entries store one host_data_representation per chunk in
        // entry.host_chunks; chunk_memory_spaces is intentionally empty (see
        // pinned_entry doc comment + insert_pinned_entry_host). Validate the
        // host_chunks vector instead.
        if (entry.host_chunks.empty()) {
          throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned host entry '" +
                                   pinned_name + "' has no host_chunks");
        }
        for (std::size_t i = 0; i < entry.host_chunks.size(); ++i) {
          if (!entry.host_chunks[i]) {
            throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned host entry '" +
                                     pinned_name + "' host_chunks[" + std::to_string(i) +
                                     "] is null");
          }
        }
        // The HOST cached path materializes host chunks onto the executing
        // GPU via converter_registry.convert<gpu_table_representation>(...).
        // Without a GPU memory_space map there is no destination — fall
        // through to the per-format path so the query still succeeds.
        if (gpu_memory_spaces.empty()) {
          SIRIUS_LOG_DEBUG(
            "[gpu_ingestible_factory::try_cached] pinned host entry '{}' matches op_id={} "
            "but no gpu_memory_spaces map was provided; falling through to per-format "
            "ingestible",
            pinned_name,
            op_id);
          break;
        }

        // Map each D-position to its index inside the captured host chunk.
        // column_names is in capture order, so we look up the requested data
        // column by name. A missing column means the user pinned a subset
        // that doesn't cover this scan — fall back to the per-format path
        // so the query still succeeds.
        std::vector<std::size_t> column_indices;
        column_indices.reserve(plan.data_columns.size());
        for (auto const& dc : plan.data_columns) {
          auto it = std::find(entry.column_names.begin(), entry.column_names.end(), dc.name);
          if (it == entry.column_names.end()) {
            throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                     pinned_name + "' missing column '" + dc.name +
                                     "' required by scan op");
          }
          column_indices.push_back(
            static_cast<std::size_t>(std::distance(entry.column_names.begin(), it)));
        }

        SIRIUS_LOG_DEBUG(
          "[gpu_ingestible_factory::try_cached] using host pinned_table_gpu_ingestible for "
          "op_id={} (pinned='{}' data_cols={} chunks={} needs_assembly={})",
          op_id,
          pinned_name,
          column_indices.size(),
          entry.host_chunks.size(),
          op::scan::needs_output_assembly(plan));

        return std::make_shared<op::scan::pinned_table_gpu_ingestible>(std::move(table_info),
                                                                       entry.host_chunks,
                                                                       std::move(column_indices),
                                                                       *entry.memory_space,
                                                                       gpu_memory_spaces,
                                                                       std::move(filter_expression),
                                                                       std::move(plan_shared));
      }

      // GPU-tier validation: every cached chunk has an owning memory_space.
      // chunk_memory_spaces is parallel to the inner vectors of
      // data_batches_by_column; empty vector means no chunks; null entries
      // violate the chunks-at-index-i invariant.
      if (entry.chunk_memory_spaces.empty()) {
        throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                 pinned_name + "' has no chunk_memory_spaces");
      }
      for (std::size_t i = 0; i < entry.chunk_memory_spaces.size(); ++i) {
        if (entry.chunk_memory_spaces[i] == nullptr) {
          throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                   pinned_name + "' chunk_memory_spaces[" + std::to_string(i) +
                                   "] is null");
        }
      }

      // Look up the pinned chunks for each D-position by name. data_columns
      // is in D-order, so columns_per_request[d] is the chunk vector for
      // D-position d.
      std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request;
      columns_per_request.reserve(plan.data_columns.size());
      for (auto const& dc : plan.data_columns) {
        auto it = entry.data_batches_by_column.find(dc.name);
        if (it == entry.data_batches_by_column.end()) {
          throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                   pinned_name + "' missing column '" + dc.name +
                                   "' required by scan op");
        }
        columns_per_request.push_back(it->second);
      }

      SIRIUS_LOG_DEBUG(
        "[gpu_ingestible_factory::try_cached] using pinned_table_gpu_ingestible for op_id={} "
        "(pinned='{}' data_cols={} needs_assembly={})",
        op_id,
        pinned_name,
        columns_per_request.size(),
        op::scan::needs_output_assembly(plan));

      // Each chunk's data_batch is tagged with its actual memory_space so
      // data-locality scheduling fans cached-scan tasks across GPUs.
      return std::make_shared<op::scan::pinned_table_gpu_ingestible>(std::move(table_info),
                                                                     std::move(columns_per_request),
                                                                     entry.chunk_memory_spaces,
                                                                     std::move(filter_expression),
                                                                     std::move(plan_shared));
    }
  } catch (...) {
    SIRIUS_LOG_TRACE("not all the columns are pinned for this query");
  }
  return nullptr;
}

}  // namespace sirius::scan_manager
