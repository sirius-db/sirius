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

#include "scan_manager/sirius_scan_manager.hpp"

#include "log/logging.hpp"
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/cached_split_provider.hpp"
#include "scan_manager/parquet_split_provider.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <algorithm>
#include <exception>
#include <utility>

namespace sirius::scan_manager {

sirius_scan_manager::sirius_scan_manager(exec::thread_pool_config config)
  : _config(std::move(config))
{
}

sirius_scan_manager::~sirius_scan_manager() { stop(); }

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] pipelines={}",
                   query.get_pipelines().size());

  for (auto const& pipeline : query.get_pipelines()) {
    if (!pipeline) { continue; }
    auto source = pipeline->get_source();
    if (!source) { continue; }
    if (source->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_PARQUET_SCAN) { continue; }

    auto* op = &source->Cast<op::scan::sirius_gpu_parquet_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }

    auto provider = create_provider_for(op);
    if (!provider) {
      // No scan_info parked on the operator (e.g. tests construct the operator
      // directly). Skip — caller is responsible for the connector.
      continue;
    }
    op->set_split_connector(std::make_unique<split_connector>());
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);

    SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] registered op_id={}",
                     op->get_operator_id());
  }

  if (_scan_op_order.empty()) { return; }

  if (!_thread_pool) {
    throw std::runtime_error("[sirius_scan_manager::prepare_for_query] thread pool not started");
  }

  _driver_thread = std::thread(&sirius_scan_manager::run_driver_loop, this);
}

std::unique_ptr<split_provider> sirius_scan_manager::create_provider_for(
  op::scan::sirius_gpu_parquet_scan_operator* op)
{
  auto info = op->take_scan_info();
  if (!info) { return nullptr; }

  // If a pinned entry's file paths match this operator's scan_info, build the same
  // scan_plan the parquet path would build and serve the scan from cache.
  auto matches_scan_info = [&info](const pinned_entry& entry) {
    if (entry.file_paths.size() != info->file_paths.size()) { return false; }
    auto sorted_a = entry.file_paths;
    auto sorted_b = info->file_paths;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    return sorted_a == sorted_b;
  };
  try {
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      if (!matches_scan_info(entry)) { continue; }
      if (entry.memory_space == nullptr) {
        throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
                                 pinned_name + "' has no memory_space");
      }

      // Build the canonical scan_plan once. Everything downstream — cached column
      // layout, filter pushdown indices, post-read assembly — reads from this.
      // Held by shared_ptr<const> so each emitted scan_cached_operator_data can
      // carry it to the GPU scan operator's per-task assembly check without copying.
      auto plan_shared = std::make_shared<op::scan::scan_plan const>(
        op::scan::build_scan_plan(info->column_ids,
                                  info->projection_ids,
                                  info->names,
                                  info->returned_types,
                                  op->get_types().size(),
                                  info->partition_indices));
      auto const& plan = *plan_shared;

      // Hive partitions on a cached scan would require per-chunk file_path metadata
      // that pinned entries don't carry today. Fall through to the parquet path,
      // which extracts partition values per file at read time.
      if (plan.has_partitions()) {
        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager::create_provider_for] pinned entry '{}' matches op_id={} but "
          "scan has hive partitions; falling through to parquet_split_provider",
          pinned_name,
          op->get_operator_id());
        break;
      }

      // Filter expression: BoundReferences are in D-space, via plan.batch_position_by_column_id.
      // Same recipe parquet_split_provider uses, so the filter evaluates correctly against
      // the cached batch (which is in D-order by construction above). Built before the
      // tier-specific assembly so both branches share the same filter.
      std::shared_ptr<duckdb::Expression> filter_expression;
      if (info->table_filters && !info->table_filters->filters.empty()) {
        auto duckdb_expression =
          op::convert_table_filters_to_expression(*info->table_filters,
                                                  info->column_ids,
                                                  info->returned_types,
                                                  plan.batch_position_by_column_id,
                                                  plan.partition_primary_indices);
        if (duckdb_expression) {
          filter_expression = std::shared_ptr<duckdb::Expression>(std::move(duckdb_expression));
        }
      }

      if (entry.tier == cucascade::memory::Tier::HOST) {
        // Map each D-position to its index inside the captured host chunk. column_names
        // is in capture order, so we look up the requested data column by name. A missing
        // column means the user pinned a subset that doesn't cover this scan — fall back
        // to the parquet path so the query still succeeds.
        std::vector<std::size_t> column_indices;
        column_indices.reserve(plan.data_columns.size());
        for (auto const& dc : plan.data_columns) {
          auto it = std::find(entry.column_names.begin(), entry.column_names.end(), dc.name);
          if (it == entry.column_names.end()) {
            throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
                                     pinned_name + "' missing column '" + dc.name +
                                     "' required by scan op");
          }
          column_indices.push_back(
            static_cast<std::size_t>(std::distance(entry.column_names.begin(), it)));
        }

        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager::create_provider_for] using host cached_split_provider for "
          "op_id={} (pinned='{}' data_cols={} chunks={} needs_assembly={})",
          op->get_operator_id(),
          pinned_name,
          column_indices.size(),
          entry.host_chunks.size(),
          op::scan::needs_output_assembly(plan));

        return std::make_unique<cached_split_provider>(entry.host_chunks,
                                                       std::move(column_indices),
                                                       *entry.memory_space,
                                                       std::move(filter_expression),
                                                       std::move(plan_shared));
      }

      // Look up the pinned chunks for each D-position by name. data_columns is in
      // D-order, so columns_per_request[d] is the chunk vector for D-position d.
      std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request;
      columns_per_request.reserve(plan.data_columns.size());
      for (auto const& dc : plan.data_columns) {
        auto it = entry.data_batches_by_column.find(dc.name);
        if (it == entry.data_batches_by_column.end()) {
          throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
                                   pinned_name + "' missing column '" + dc.name +
                                   "' required by scan op");
        }
        columns_per_request.push_back(it->second);
      }

      SIRIUS_LOG_DEBUG(
        "[sirius_scan_manager::create_provider_for] using cached_split_provider for op_id={} "
        "(pinned='{}' data_cols={} needs_assembly={})",
        op->get_operator_id(),
        pinned_name,
        columns_per_request.size(),
        op::scan::needs_output_assembly(plan));

      return std::make_unique<cached_split_provider>(std::move(columns_per_request),
                                                     *entry.memory_space,
                                                     std::move(filter_expression),
                                                     std::move(plan_shared));
    }
  } catch (...) {
    SIRIUS_LOG_TRACE("not all the columns are pinned for this query");
  }
  return std::make_unique<parquet_split_provider>(info->returned_types,
                                                  info->file_paths,
                                                  info->column_ids,
                                                  info->projection_ids,
                                                  info->names,
                                                  op->get_types().size(),
                                                  std::move(info->table_filters),
                                                  info->partition_indices,
                                                  info->approximate_batch_size);
}

void sirius_scan_manager::run_driver_loop()
{
  for (auto* op : _scan_op_order) {
    auto it = _providers_by_op.find(op);
    if (it == _providers_by_op.end()) { continue; }
    auto* connector = op->get_split_connector();
    if (connector == nullptr) { continue; }

    try {
      auto future = it->second->start(*_thread_pool, *connector);
      future.get();
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("[sirius_scan_manager] driver: provider failed: {}", e.what());
      // Make sure the consumer is unblocked even on failure.
      connector->close();
    }
  }
}

void sirius_scan_manager::reset()
{
  if (_driver_thread.joinable()) { _driver_thread.join(); }
  _scan_op_order.clear();
  _providers_by_op.clear();
}

void sirius_scan_manager::start()
{
  if (_thread_pool) { return; }
  _thread_pool = std::make_unique<exec::thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
}

void sirius_scan_manager::stop()
{
  if (_driver_thread.joinable()) { _driver_thread.join(); }
  if (!_thread_pool) { return; }
  _thread_pool->stop();
  _thread_pool.reset();
}

void sirius_scan_manager::insert_pinned_entry(const std::string& name,
                                              std::vector<std::string> column_names,
                                              std::vector<std::string> file_paths,
                                              std::vector<std::unique_ptr<cudf::table>> data_tables,
                                              cucascade::memory::memory_space& memory_space)
{
  // Compute the total row count of the incoming tables before releasing them
  // (release() empties the table; num_rows() would then return 0).
  std::size_t new_num_rows = 0;
  for (auto const& table : data_tables) {
    if (table) { new_num_rows += static_cast<std::size_t>(table->num_rows()); }
  }

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    if (existing_it->second.num_rows == new_num_rows) {
      // Same row count → merge unique columns into the existing entry.
      auto& entry = existing_it->second;
      for (auto& table : data_tables) {
        if (!table) { continue; }
        auto cols = table->release();
        if (cols.size() != column_names.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] table column count " +
            std::to_string(cols.size()) + " does not match column_names size " +
            std::to_string(column_names.size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
          auto const& col_name = column_names[i];
          if (entry.data_batches_by_column.contains(col_name)) {
            // Already cached — drop the duplicate column.
            continue;
          }
          entry.data_batches_by_column[col_name].emplace_back(std::move(cols[i]));
        }
      }
      // Append any new column names to the entry's column_names list so its
      // metadata reflects the union of pinned columns.
      for (auto& cn : column_names) {
        if (std::find(entry.column_names.begin(), entry.column_names.end(), cn) ==
            entry.column_names.end()) {
          entry.column_names.push_back(std::move(cn));
        }
      }
      return;
    }
    // Row count differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.column_names = std::move(column_names);
  entry.file_paths   = std::move(file_paths);
  entry.tier         = cucascade::memory::Tier::GPU;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != entry.column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(entry.column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      entry.data_batches_by_column[entry.column_names[i]].emplace_back(std::move(cols[i]));
    }
  }

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::insert_pinned_entry_host(
  const std::string& name,
  std::vector<std::string> column_names,
  std::vector<std::string> file_paths,
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space)
{
  // The host-tier path captures one chunk per emitted batch; each chunk holds every
  // pinned column. Re-insert always replaces — there is no per-column merge analog
  // to the GPU path because the chunk-vs-column dimensions are flipped.
  std::size_t new_num_rows = 0;
  for (auto const& chunk : host_chunks) {
    if (!chunk) { continue; }
    auto const& host_table = chunk->get_host_table();
    if (host_table && !host_table->columns.empty()) {
      new_num_rows += static_cast<std::size_t>(host_table->columns.front().num_rows);
    }
  }

  pinned_entry entry;
  entry.column_names = std::move(column_names);
  entry.file_paths   = std::move(file_paths);
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;
  entry.host_chunks  = std::move(host_chunks);

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

}  // namespace sirius::scan_manager
