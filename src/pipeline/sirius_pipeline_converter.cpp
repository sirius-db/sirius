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

#include "pipeline/sirius_pipeline_converter.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/shared_ptr_ipp.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_cpu_source.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_iceberg_scan.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "sirius_config.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::pipeline {

duckdb::unique_ptr<op::sirius_physical_operator> construct_sirius_specific_operator(
  op::sirius_physical_operator& physical_op,
  const std::unordered_map<std::string, std::shared_ptr<const op::scan::IcebergDeleteData>>*
    iceberg_cache)
{
  if (physical_op.type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& scan_physical_op = physical_op.Cast<op::sirius_physical_table_scan>();
    if (scan_physical_op.function.name == "parquet_scan" ||
        scan_physical_op.function.name == "read_parquet" ||
        scan_physical_op.function.name == "sirius_read_parquet") {
      return duckdb::make_uniq<op::sirius_physical_parquet_scan>(&scan_physical_op);
    } else if (scan_physical_op.function.name == "iceberg_scan") {
      if (!iceberg_cache) {
        throw duckdb::InternalException(
          "iceberg_cache must be provided when constructing iceberg scan operators");
      }
      auto iceberg_scan = duckdb::make_uniq<op::sirius_physical_iceberg_scan>(&scan_physical_op);
      if (!scan_physical_op.parameters.empty()) {
        std::string const table_path = scan_physical_op.parameters[0].ToString();
        auto it                      = iceberg_cache->find(table_path);
        if (it != iceberg_cache->end()) { iceberg_scan->delete_data = it->second; }
      }
      return iceberg_scan;
    } else if (scan_physical_op.function.name == "seq_scan") {
      return duckdb::make_uniq<op::sirius_physical_duckdb_scan>(&scan_physical_op);
    } else {
      throw duckdb::NotImplementedException("Unsupported scan function: " +
                                            scan_physical_op.function.name);
    }
  } else if (physical_op.type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    auto& group_by_physical_op = physical_op.Cast<op::sirius_physical_grouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_grouped_aggregate_merge>(&group_by_physical_op);
  } else if (physical_op.type == op::SiriusPhysicalOperatorType::ORDER_BY) {
    auto& order_by_physical_op = physical_op.Cast<op::sirius_physical_order>();
    return duckdb::make_uniq<op::sirius_physical_merge_sort>(&order_by_physical_op);
  } else if (physical_op.type == op::SiriusPhysicalOperatorType::TOP_N) {
    auto& topn_physical_op = physical_op.Cast<op::sirius_physical_top_n>();
    return duckdb::make_uniq<op::sirius_physical_top_n_merge>(&topn_physical_op);
  } else if (physical_op.type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE) {
    auto& ungrouped_agg_physical_op = physical_op.Cast<op::sirius_physical_ungrouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_ungrouped_aggregate_merge>(
      &ungrouped_agg_physical_op);
  } else {
    throw duckdb::InternalException(
      "Unsupported operator type: " + SiriusPhysicalOperatorToString(physical_op.type) +
      " for constructing sirius specific operator.");
  }
}

sirius_pipeline_converter::sirius_pipeline_converter(
  const pipeline_build_context& ctx,
  const sirius::operator_params& op_params,
  const std::unordered_map<std::string, std::shared_ptr<const op::scan::IcebergDeleteData>>*
    iceberg_cache,
  duckdb::ClientContext* client_context)
  : build_ctx_(ctx),
    op_params_(op_params),
    iceberg_cache_(iceberg_cache),
    client_context_(client_context)
{
}

pipeline_conversion_result sirius_pipeline_converter::convert(sirius_meta_pipeline& root_pipeline)
{
  scheduled_.clear();
  inserted_operators_.clear();
  repository_wirings_.clear();

  auto copied_scheduled = schedule_and_copy_pipelines(root_pipeline);
  if (!duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    // Phase 3.3 (#604): under the tree-based path, the plan generator + build_pipelines
    // virtuals already place every operator. There is nothing to split or insert at
    // convert time.
    split_pipelines(copied_scheduled);
  } else {
    // Under flag ON, split_pipelines never runs to populate scheduled_, so the post-build
    // pipelines need to be transferred directly. compute_repository_wiring_tree_based and
    // the dump helper both read scheduled_ — without this transfer the conversion result
    // is empty even though build_pipelines produced a full pipeline tree.
    scheduled_ = std::move(copied_scheduled);
  }
  compute_repository_wiring(root_pipeline.get_state());
  setup_pipeline_parents();
  finalize_pipeline_structure();
  link_join_partition_siblings();
  configure_partition_min_partitions();

  return {std::move(scheduled_),
          std::move(inserted_operators_),
          std::move(repository_wirings_),
          meta_pipeline_count_};
}

duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>
sirius_pipeline_converter::schedule_and_copy_pipelines(sirius_meta_pipeline& root_pipeline)
{
  duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>> to_schedule;
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> sirius_scheduled;
  scheduled_.clear();
  root_pipeline.get_meta_pipelines(to_schedule, true, true);

  // number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to
  // set it here
  meta_pipeline_count_ = to_schedule.size();

  SIRIUS_LOG_DEBUG("Total meta pipelines {}", to_schedule.size());
  int schedule_count = 0;
  int meta           = 0;
  while (schedule_count < to_schedule.size()) {
    duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>> children;
    to_schedule[to_schedule.size() - 1 - meta]->get_meta_pipelines(children, false, true);
    auto base_pipeline   = to_schedule[to_schedule.size() - 1 - meta]->get_base_pipeline();
    bool should_schedule = true;

    // already scheduled
    if (std::ranges::find(sirius_scheduled, base_pipeline) != sirius_scheduled.end()) {
      should_schedule = false;
    } else {
      // check if all children are scheduled
      for (auto& child : children) {
        if (std::ranges::find(sirius_scheduled, child->get_base_pipeline()) ==
            sirius_scheduled.end()) {
          should_schedule = false;
          break;
        }
      }
      // check if all dependencies are scheduled
      for (const auto& dependency : base_pipeline->dependencies) {
        if (std::ranges::find(sirius_scheduled, dependency) == sirius_scheduled.end()) {
          should_schedule = false;
          break;
        }
      }
    }
    if (should_schedule) {
      duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> pipeline_inside;
      to_schedule[to_schedule.size() - 1 - meta]->get_pipelines(pipeline_inside, false);
      for (auto& pipeline : pipeline_inside) {
        // Legacy-only filter: under flag OFF, `set_pipeline_source` is overwritten as
        // the recursion descends, so any HJ/NLJ-source pipeline at this point is a stale
        // build-meta whose true source is its sink. The post-split `[HJ, ..., sink]`
        // shape only emerges later in `split_pipelines`. DuckDB adds a build-side scan
        // pipeline (join as source) for right/outer joins; sirius joins emit unmatched
        // build rows inline, so keeping it would wire the join's downstream ports twice.
        // Under flag ON, join-source pipelines that reach this point are legitimate
        // (inner-join outputs feeding PARTITION); dropping them would corrupt the
        // schedule.
        if (!duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD &&
            (pipeline->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
             pipeline->source->type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN)) {
          continue;
        }
        sirius_scheduled.push_back(pipeline);
      }
      schedule_count++;
    }
    meta = (meta + 1) % to_schedule.size();
  }

  // perform deep copy on scheduled pipelines. The copies isolate legacy's
  // `split_pipelines` mutations from the meta-pipeline state. Under
  // `USE_TREE_BASED_PIPELINE_BUILD` no split runs, and `build_pipelines`
  // already produced final-shape pipelines — so return the originals
  // directly so downstream code (wiring lookups via `state.cte_scan_consumers`
  // populated in `build_pipelines`) can resolve consumer pipelines by
  // pointer identity.
  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) { return sirius_scheduled; }

  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> copied_scheduled;
  for (const auto& pipeline : sirius_scheduled) {
    auto copied_pipeline = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    // copy source
    copied_pipeline->source = pipeline->source;
    // copy operators
    for (size_t j = 0; j < pipeline->operators.size(); j++) {
      copied_pipeline->operators.push_back(pipeline->operators[j]);
    }
    // copy sink
    copied_pipeline->sink = pipeline->sink;
    copied_scheduled.push_back(copied_pipeline);
  }

  return copied_scheduled;
}

// TODO: batch_lock_utils RAII migration may affect this converter — review.
// TODO: if writer_event recording happens here, ensure the
// cudaStreamWaitEvent chain remains intact post-Scan-Manager.
//===----------------------------------------------------------------------===//
// insert_parquet_scan_operator()
//
// Rewrites a DuckDB parquet table scan into a Sirius gpu_scan_op as the
// pipeline source. The companion metadata scan operator is constructed here
// (so we can extract bind_data while we still have it) but is not placed in
// any pipeline — it is parked on the gpu_scan_op via attach_metadata_scan_op()
// so the scan_manager can take ownership and drive its execute() on its own
// thread pool during prepare_for_query.
//===----------------------------------------------------------------------===//
void sirius_pipeline_converter::insert_parquet_scan_operator(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline)
{
  auto& scan_op = current_pipeline->get_source()->Cast<op::sirius_physical_table_scan>();

  auto table_info            = std::make_unique<op::scan::parquet_ingestible_table_info>();
  table_info->returned_types = scan_op.returned_types;
  table_info->column_ids     = scan_op.column_ids;
  table_info->projection_ids = scan_op.projection_ids;
  table_info->names          = scan_op.names;
  table_info->table_filters  = std::move(scan_op.table_filters);

  if (scan_op.function.name == "sirius_read_parquet") {
    if (scan_op.parameters.empty() || scan_op.parameters.front().IsNull()) {
      throw std::runtime_error(
        "[sirius_pipeline_converter::insert_parquet_scan_operator] sirius_read_parquet scan "
        "has no URI parameter");
    }
    table_info->resolved_file_paths = {scan_op.parameters.front().GetValue<std::string>()};
  } else {
    auto const& bind_data = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
    if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
      throw std::runtime_error(
        "[sirius_pipeline_converter::insert_parquet_scan_operator] No input files to scan");
    }
    std::vector<std::string> file_paths;
    for (auto const& file : bind_data.file_list->GetAllFiles()) {
      file_paths.push_back(file.path);
    }
    table_info->resolved_file_paths = std::move(file_paths);
    table_info->partition_indices   = bind_data.reader_bind.hive_partitioning_indexes;
  }
  table_info->scan_output_arity      = scan_op.types.size();
  table_info->approximate_batch_size = op_params_.scan_task_batch_size;

  // The ingestible uses dynamic filters for read-time row-group pruning; the dynamic-filter
  // operator inserted below applies them post-decode. If no producer was wired after planning,
  // elide both paths for this scan.
  auto dynamic_filters = scan_op.sirius_dynamic_filters;
  if (dynamic_filters && !dynamic_filters->has_producers()) { dynamic_filters.reset(); }
  table_info->sirius_dynamic_filters = dynamic_filters;

  auto parquet_ingestible = op::scan::make_ingestible(std::move(table_info));
  auto gpu_scan_op        = duckdb::make_uniq<op::scan::sirius_gpu_scan_operator>(
    scan_op.types, scan_op.estimated_cardinality, std::move(parquet_ingestible));

  auto* gpu_scan_ptr = gpu_scan_op.get();

  // finalize_pipeline_structure() will set current_pipeline->source = &operators[0] = gpu_scan_op.
  current_pipeline->operators.insert(current_pipeline->operators.begin(), *gpu_scan_ptr);

  inserted_operators_.push_back(std::move(gpu_scan_op));

  // Insert the dynamic-filter operator directly above the scan at operators[1]. It filters both the
  // disk and cached resolutions of this parquet scan.
  if (dynamic_filters) {
    auto dynamic_filter_op = duckdb::make_uniq<op::scan::sirius_physical_dynamic_filter>(
      scan_op.types,
      scan_op.estimated_cardinality,
      std::move(dynamic_filters),
      op_params_.dynamic_filter_keep_threshold);
    auto* dynamic_filter_ptr = dynamic_filter_op.get();
    current_pipeline->operators.insert(current_pipeline->operators.begin() + 1,
                                       *dynamic_filter_ptr);
    inserted_operators_.push_back(std::move(dynamic_filter_op));
  }
}

void sirius_pipeline_converter::insert_duckdb_native_scan_operator(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline)
{
  auto& scan_op = current_pipeline->get_source()->Cast<op::sirius_physical_table_scan>();
  if (!scan_op.bind_data) {
    throw std::runtime_error(
      "[sirius_pipeline_converter::insert_duckdb_native_scan_operator] seq_scan has no bind_data");
  }
  auto* table_scan_bind = dynamic_cast<duckdb::TableScanBindData*>(scan_op.bind_data.get());
  if (table_scan_bind == nullptr) {
    throw std::runtime_error(
      "[sirius_pipeline_converter::insert_duckdb_native_scan_operator] seq_scan bind_data is not "
      "TableScanBindData; the GPU-native duckdb scan path supports only seq_scan over base "
      "tables.");
  }
  auto& bind_data = *table_scan_bind;
  auto& table     = bind_data.table.Cast<duckdb::DuckTableEntry>();

  if (client_context_ == nullptr) {
    throw std::runtime_error(
      "[sirius_pipeline_converter::insert_duckdb_native_scan_operator] no client_context passed "
      "to converter; seq_scan GPU-native path requires it");
  }

  auto table_info     = std::make_unique<op::scan::duckdb_native_ingestible_table_info>();
  table_info->storage = &table.GetStorage();
  table_info->context = client_context_;
  table_info->db_path = table.GetStorage().GetAttached().GetStorageManager().GetDBPath();
  // Qualified-name identity for the pin cache — derived from the resolved
  // DuckTableEntry so it matches the pin-side derivation (build_duckdb_pin_info) exactly.
  table_info->catalog_name           = table.ParentCatalog().GetName();
  table_info->schema_name            = table.ParentSchema().name;
  table_info->table_name             = table.name;
  table_info->approximate_batch_size = op_params_.scan_task_batch_size;

  std::vector<std::size_t> source_ids_fallback;
  if (scan_op.projection_ids.empty()) {
    source_ids_fallback.resize(scan_op.column_ids.size());
    std::iota(source_ids_fallback.begin(), source_ids_fallback.end(), 0);
  }
  auto const& source_ids =
    scan_op.projection_ids.empty() ? source_ids_fallback : scan_op.projection_ids;

  table_info->projected_cols.reserve(source_ids.size());
  table_info->projected_types.reserve(source_ids.size());
  for (std::size_t k = 0; k < source_ids.size(); ++k) {
    auto pid            = source_ids[k];
    auto const& col_idx = scan_op.column_ids[pid];
    op::scan::projected_column pc;
    pc.is_rowid = col_idx.IsRowIdColumn();
    if (!pc.is_rowid) { pc.storage_idx = duckdb::StorageIndex(col_idx.GetPrimaryIndex()); }
    table_info->projected_cols.push_back(pc);

    sirius::logical_type t;
    if (k < scan_op.types.size()) {
      t = scan_op.types[k];
    } else {
      t = scan_op.returned_types.at(col_idx.GetPrimaryIndex());
    }
    table_info->projected_types.push_back(t);
  }

  // Filters drive row-group pruning in the metadata walk and post-decode filtering.
  if (scan_op.table_filters) {
    table_info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    for (auto& [col_idx, filt] : scan_op.table_filters->filters) {
      table_info->table_filters->filters[col_idx] = filt->Copy();
    }
  }
  table_info->column_ids     = scan_op.column_ids;
  table_info->projection_ids = scan_op.projection_ids;
  table_info->returned_types = scan_op.returned_types;
  table_info->output_types   = scan_op.types;

  auto duckdb_native_ingestible = op::scan::make_ingestible(std::move(table_info));
  auto gpu_scan_op              = duckdb::make_uniq<op::scan::sirius_gpu_scan_operator>(
    scan_op.types, scan_op.estimated_cardinality, std::move(duckdb_native_ingestible));

  auto* gpu_scan_ptr = gpu_scan_op.get();
  current_pipeline->operators.insert(current_pipeline->operators.begin(), *gpu_scan_ptr);
  inserted_operators_.push_back(std::move(gpu_scan_op));
}

void sirius_pipeline_converter::split_table_scan_source(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline)
{
  if (current_pipeline->source->type != op::SiriusPhysicalOperatorType::TABLE_SCAN) { return; }

  auto& scan_op = current_pipeline->get_source()->Cast<op::sirius_physical_table_scan>();
  // If parquet scan, route to metadata scan + gpu scan operator pipeline
  if (scan_op.function.name == "parquet_scan" || scan_op.function.name == "read_parquet" ||
      scan_op.function.name == "sirius_read_parquet") {
    insert_parquet_scan_operator(current_pipeline);
    return;
  }

  if (scan_op.function.name == "seq_scan") {
    insert_duckdb_native_scan_operator(current_pipeline);
    return;
  }

  // The legacy seq_scan / iceberg_scan path built duckdb_scan / iceberg_scan
  // operators (executed by the now-removed scan tasks).  Parquet and GPU-native
  // seq_scan are handled above via the GPU scan operators; anything else is
  // unsupported.
  throw std::runtime_error("Unsupported scan function: " + scan_op.function.name);
}

void sirius_pipeline_converter::split_cpu_source(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline)
{
  auto src_type = current_pipeline->source->type;
  // COLUMN_DATA_SCAN with a null collection is LEFT_DELIM_JOIN's cached chunk
  // scan — populated at runtime by the delim-join sink, not by a
  // cpu_source_task. Splitting it would create a second pipeline referencing
  // the same operator and trip "Repository already exists" on complex queries.
  bool is_column_data_scan =
    src_type == op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN &&
    current_pipeline->get_source()->Cast<op::sirius_physical_column_data_scan>().collection !=
      nullptr;
  if (src_type != op::SiriusPhysicalOperatorType::EMPTY_RESULT &&
      src_type != op::SiriusPhysicalOperatorType::DUMMY_SCAN && !is_column_data_scan) {
    return;
  }

  auto* source_op = current_pipeline->get_source().get();

  duckdb::unique_ptr<op::sirius_physical_cpu_source> cpu_source_op;
  if (src_type == op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN) {
    auto& col_scan = source_op->Cast<op::sirius_physical_column_data_scan>();
    cpu_source_op  = duckdb::make_uniq<op::sirius_physical_cpu_source>(
      source_op->types, source_op->estimated_cardinality, std::move(col_scan.collection));
  } else if (src_type == op::SiriusPhysicalOperatorType::DUMMY_SCAN) {
    cpu_source_op = duckdb::make_uniq<op::sirius_physical_cpu_source>(
      source_op->types, source_op->estimated_cardinality, true);
  } else {
    // EMPTY_RESULT: no data
    cpu_source_op = duckdb::make_uniq<op::sirius_physical_cpu_source>(
      source_op->types, source_op->estimated_cardinality, false);
  }

  auto new_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  new_pipeline->source = nullptr;
  new_pipeline->sink   = cpu_source_op.get();

  current_pipeline->source = cpu_source_op.get();
  current_pipeline->operators.insert(current_pipeline->operators.begin(), *source_op);

  scheduled_.push_back(new_pipeline);
  inserted_operators_.push_back(std::move(cpu_source_op));
}

void sirius_pipeline_converter::split_intermediate_joins(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline)
{
  duckdb::vector<std::size_t> join_positions;
  for (std::size_t op_idx = 0; op_idx < current_pipeline->operators.size(); op_idx++) {
    if (current_pipeline->operators[op_idx].get().type ==
          op::SiriusPhysicalOperatorType::HASH_JOIN ||
        current_pipeline->operators[op_idx].get().type ==
          op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
      join_positions.push_back(op_idx);
    }
  }

  if (join_positions.empty()) { return; }

  duckdb::shared_ptr<sirius_pipeline> previous_pipeline = nullptr;
  op::sirius_physical_concat* prev_concat_ptr           = nullptr;

  for (size_t hj_idx = 0; hj_idx < join_positions.size(); hj_idx++) {
    std::size_t join_pos = join_positions[hj_idx];
    duckdb::unique_ptr<op::sirius_physical_concat> concat_op;

    // Create a PARTITION and CONCAT operator
    if (join_pos == 0) {
      concat_op =
        make_uniq<op::sirius_physical_concat>(current_pipeline->get_source()->types,
                                              current_pipeline->get_source()->estimated_cardinality,
                                              &current_pipeline->operators[join_pos].get(),
                                              false,
                                              op_params_.concat_batch_bytes);
      auto partition_op = make_uniq<op::sirius_physical_partition>(
        current_pipeline->get_source()->types,
        current_pipeline->get_source()->estimated_cardinality,
        &current_pipeline->operators[join_pos].get(),
        false,
        op_params_.hash_partition_bytes);
      inserted_operators_.push_back(std::move(partition_op));
    } else {
      concat_op = make_uniq<op::sirius_physical_concat>(
        current_pipeline->operators[join_pos - 1].get().types,
        current_pipeline->operators[join_pos - 1].get().estimated_cardinality,
        &current_pipeline->operators[join_pos].get(),
        false,
        op_params_.concat_batch_bytes);
      auto partition_op = make_uniq<op::sirius_physical_partition>(
        current_pipeline->operators[join_pos - 1].get().types,
        current_pipeline->operators[join_pos - 1].get().estimated_cardinality,
        &current_pipeline->operators[join_pos].get(),
        false,
        op_params_.hash_partition_bytes);
      inserted_operators_.push_back(std::move(partition_op));
    }

    auto* partition_ptr =
      static_cast<op::sirius_physical_partition*>(inserted_operators_.back().get());

    if (join_pos > 0) {
      auto new_pipeline = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);

      if (hj_idx == 0) {
        // Move operators from current pipeline to new pipeline except for the last operator
        // before the join
        for (std::size_t j = 0; j < join_pos - 1; j++) {
          new_pipeline->operators.push_back(current_pipeline->operators[j]);
        }
        // set the sink to the operator before the join
        new_pipeline->sink   = current_pipeline->operators[join_pos - 1].get();
        new_pipeline->source = current_pipeline->source;
      } else {
        // Move operators from current pipeline to new pipeline except for the last operator
        // before the join
        for (std::size_t j = join_positions[hj_idx - 1]; j < join_pos - 1; j++) {
          new_pipeline->operators.push_back(current_pipeline->operators[j]);
        }
        // set the sink to the operator before the join
        new_pipeline->sink   = current_pipeline->operators[join_pos - 1].get();
        new_pipeline->source = prev_concat_ptr;
      }

      scheduled_.push_back(new_pipeline);

      // new pipeline for partition_op
      auto partition_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
      partition_pipeline->source = new_pipeline->sink.get();
      partition_pipeline->sink   = partition_ptr;
      scheduled_.push_back(partition_pipeline);
    } else {
      // new pipeline for partition_op
      auto partition_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
      partition_pipeline->source = current_pipeline->source;
      partition_pipeline->sink   = partition_ptr;
      scheduled_.push_back(partition_pipeline);
    }

    // new pipeline for concat_op
    auto concat_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    concat_pipeline->source = partition_ptr;
    concat_pipeline->sink   = concat_op.get();

    inserted_operators_.push_back(std::move(concat_op));
    auto* concat_ptr = static_cast<op::sirius_physical_concat*>(inserted_operators_.back().get());

    scheduled_.push_back(concat_pipeline);

    // update current pipeline at the last join position
    if (hj_idx == join_positions.size() - 1) {
      // remove operators from current pipeline
      current_pipeline->operators.erase(current_pipeline->operators.begin(),
                                        current_pipeline->operators.begin() + join_pos);
      current_pipeline->source = concat_ptr;
    }

    // create a shared ptr from new pipeline
    previous_pipeline = concat_pipeline;
    prev_concat_ptr   = concat_ptr;
  }
}

void sirius_pipeline_converter::split_join_sink(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline)
{
  // replace hash join sink with partition
  duckdb::unique_ptr<op::sirius_physical_partition> partition_op;
  duckdb::unique_ptr<op::sirius_physical_concat> concat_op;
  auto hash_join_op = current_pipeline->get_sink();
  if (current_pipeline->operators.size() == 0) {
    // source -> partition -> hash join
    concat_op =
      make_uniq<op::sirius_physical_concat>(current_pipeline->get_source()->types,
                                            current_pipeline->get_source()->estimated_cardinality,
                                            hash_join_op.get(),
                                            true,
                                            op_params_.concat_batch_bytes);
    partition_op = make_uniq<op::sirius_physical_partition>(
      current_pipeline->get_source()->types,
      current_pipeline->get_source()->estimated_cardinality,
      hash_join_op.get(),
      true,
      op_params_.hash_partition_bytes);
  } else {
    concat_op = make_uniq<op::sirius_physical_concat>(
      current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
      current_pipeline->operators[current_pipeline->operators.size() - 1]
        .get()
        .estimated_cardinality,
      hash_join_op.get(),
      true,
      op_params_.concat_batch_bytes);
    partition_op = make_uniq<op::sirius_physical_partition>(
      current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
      current_pipeline->operators[current_pipeline->operators.size() - 1]
        .get()
        .estimated_cardinality,
      hash_join_op.get(),
      true,
      op_params_.hash_partition_bytes);
  }

  auto* partition_ptr = static_cast<op::sirius_physical_partition*>(partition_op.get());

  if (current_pipeline->operators.size() > 0) {
    // Last op before HASH_JOIN becomes the sink
    op::sirius_physical_operator* last_op_ptr = &current_pipeline->operators.back().get();
    current_pipeline->sink                    = last_op_ptr;
    current_pipeline->operators.erase(current_pipeline->operators.end() - 1);
    scheduled_.push_back(current_pipeline);

    // Partition pipeline: last_op (source) -> PARTITION (sink)
    auto partition_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    partition_pipeline->source = last_op_ptr;
    partition_pipeline->sink   = partition_ptr;
    scheduled_.push_back(partition_pipeline);

    // CONCAT pipeline: PARTITION (source) -> CONCAT (sink)
    auto concat_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    concat_pipeline->source = partition_ptr;
    concat_pipeline->sink   = concat_op.get();
    scheduled_.push_back(concat_pipeline);
  } else {
    // No ops before HASH_JOIN (or the sole op is the source itself) — PARTITION is the sink
    // of current_pipeline.
    current_pipeline->sink = partition_ptr;
    scheduled_.push_back(current_pipeline);

    // CONCAT pipeline: PARTITION (source) -> CONCAT (sink)
    auto concat_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    concat_pipeline->source = partition_ptr;
    concat_pipeline->sink   = concat_op.get();
    scheduled_.push_back(concat_pipeline);
  }

  inserted_operators_.push_back(std::move(partition_op));
  inserted_operators_.push_back(std::move(concat_op));
}

void sirius_pipeline_converter::split_group_aggregate_sink(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
  size_t pipeline_idx)
{
  auto group_agg_op = current_pipeline->sink;
  if (group_agg_op->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    // Create a PARTITION operator
    auto partition_op =
      make_uniq<op::sirius_physical_partition>(current_pipeline->get_sink()->types,
                                               current_pipeline->get_sink()->estimated_cardinality,
                                               current_pipeline->get_sink().get(),
                                               false,
                                               op_params_.hash_partition_bytes);
    inserted_operators_.push_back(std::move(partition_op));

    auto* partition_ptr =
      static_cast<op::sirius_physical_partition*>(inserted_operators_.back().get());

    // Keep GROUP_BY as the sink (don't move it to operators)
    scheduled_.push_back(current_pipeline);

    // Create partition pipeline: GROUP_BY (source) -> PARTITION (sink)
    auto partition_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    partition_pipeline->source = group_agg_op.get();
    partition_pipeline->sink   = partition_ptr;
    scheduled_.push_back(partition_pipeline);

    // Create merge pipeline: PARTITION (source) -> MERGE_OP (sink)
    auto merge_op          = construct_sirius_specific_operator(*group_agg_op, iceberg_cache_);
    auto merge_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    merge_pipeline->source = partition_ptr;
    merge_pipeline->sink   = merge_op.get();

    // Update downstream pipelines to use MERGE_OP as source
    for (size_t j = pipeline_idx + 1; j < copied_scheduled.size(); j++) {
      if (copied_scheduled[j]->source.get() == group_agg_op.get()) {
        copied_scheduled[j]->source = merge_op.get();
      }
    }
    scheduled_.push_back(merge_pipeline);
    inserted_operators_.push_back(std::move(merge_op));
  } else {
    // UNGROUPED_AGGREGATE — no PARTITION needed
    scheduled_.push_back(current_pipeline);

    auto merge_op        = construct_sirius_specific_operator(*group_agg_op, iceberg_cache_);
    auto new_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    new_pipeline->source = group_agg_op;
    new_pipeline->sink   = merge_op.get();

    // Update downstream pipelines to use MERGE_OP as source
    for (size_t j = pipeline_idx + 1; j < copied_scheduled.size(); j++) {
      if (copied_scheduled[j]->source.get() == group_agg_op.get()) {
        copied_scheduled[j]->source = merge_op.get();
      }
    }
    scheduled_.push_back(new_pipeline);
    inserted_operators_.push_back(std::move(merge_op));
  }
}

void sirius_pipeline_converter::split_order_by_sink(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
  size_t pipeline_idx)
{
  auto order_op   = current_pipeline->sink;
  auto* order_ptr = static_cast<op::sirius_physical_order*>(order_op.get());

  // Save the original projection and replace with identity so ORDER outputs all columns.
  // Sort keys must remain in the output for SORT_SAMPLE and SORT_PARTITION to reference.
  // MERGE_SORT will apply the final projection.
  auto original_projections = order_ptr->projections;
  {
    auto& child_types = current_pipeline->operators.size() > 0
                          ? current_pipeline->operators.back().get().types
                          : current_pipeline->source->types;
    duckdb::vector<std::size_t> identity_proj;
    for (std::size_t col_idx = 0; col_idx < child_types.size(); col_idx++) {
      identity_proj.push_back(col_idx);
    }
    order_ptr->projections = std::move(identity_proj);
    order_ptr->types       = child_types;
  }

  // Pipeline A: current pipeline keeps ORDER as sink (local sort per batch)
  scheduled_.push_back(current_pipeline);

  // Create SORT_SAMPLE operator
  auto sample_op = duckdb::make_uniq<op::sirius_physical_sort_sample>(
    order_ptr,
    op_params_.sort_sample_bytes,
    op_params_.max_sort_partition_bytes,
    op_params_.max_sort_partition_memory_fraction);
  auto* sample_ptr = sample_op.get();

  // Create SORT_PARTITION operator
  auto partition_op   = duckdb::make_uniq<op::sirius_physical_sort_partition>(order_ptr);
  auto* partition_ptr = partition_op.get();

  // Wire sort_partition to read boundaries from sort_sample
  partition_ptr->set_sample_op(sample_ptr);

  // Pipeline B: ORDER (source) -> SORT_SAMPLE -> SORT_PARTITION (sink)
  // Sample and partition run in one gpu_pipeline_task so partition sees sample
  // boundaries immediately after sample completes on the same batch.
  auto sample_partition_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  sample_partition_pipeline->source = order_op.get();
  sample_partition_pipeline->operators.push_back(*sample_ptr);
  sample_partition_pipeline->sink = partition_ptr;
  scheduled_.push_back(sample_partition_pipeline);

  // Create MERGE_SORT operator
  auto merge_op   = duckdb::make_uniq<op::sirius_physical_merge_sort>(order_ptr);
  auto* merge_ptr = merge_op.get();

  // If ORDER had a non-identity projection, set it as MERGE_SORT's final projection
  {
    bool is_identity = (original_projections.size() == order_ptr->types.size());
    if (is_identity) {
      for (std::size_t proj_idx = 0; proj_idx < original_projections.size(); proj_idx++) {
        if (original_projections[proj_idx] != proj_idx) {
          is_identity = false;
          break;
        }
      }
    }
    if (!is_identity) {
      duckdb::vector<sirius::logical_type> output_types;
      for (auto idx : original_projections) {
        output_types.push_back(order_ptr->types[idx]);
      }
      merge_ptr->set_final_projections(std::move(original_projections), std::move(output_types));
    }
  }

  // Pipeline C: SORT_PARTITION (source) -> MERGE_SORT (sink)
  auto merge_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  merge_pipeline->source = partition_ptr;
  merge_pipeline->sink   = merge_ptr;
  scheduled_.push_back(merge_pipeline);

  // Update downstream pipelines to use MERGE_SORT as source
  for (size_t j = pipeline_idx + 1; j < copied_scheduled.size(); j++) {
    if (copied_scheduled[j]->source.get() == order_op.get()) {
      copied_scheduled[j]->source = merge_ptr;
    }
  }

  // Store ownership
  inserted_operators_.push_back(std::move(sample_op));
  inserted_operators_.push_back(std::move(partition_op));
  inserted_operators_.push_back(std::move(merge_op));
}

void sirius_pipeline_converter::split_top_n_sink(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
  size_t pipeline_idx)
{
  auto top_n_op  = current_pipeline->sink;
  auto* topn_ptr = static_cast<op::sirius_physical_top_n*>(top_n_op.get());

  // Pipeline A: current pipeline keeps TOP_N as sink
  scheduled_.push_back(current_pipeline);

  // Create MERGE_TOP_N operator
  auto merge_op = duckdb::unique_ptr<op::sirius_physical_top_n_merge>(
    new op::sirius_physical_top_n_merge(topn_ptr));
  auto* merge_ptr = merge_op.get();

  // Pipeline B: TOP_N (source) -> MERGE_TOP_N (sink)
  auto merge_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  merge_pipeline->source = top_n_op.get();
  merge_pipeline->sink   = merge_ptr;
  scheduled_.push_back(merge_pipeline);

  // Update downstream pipelines to use MERGE_TOP_N as source
  for (size_t j = pipeline_idx + 1; j < copied_scheduled.size(); j++) {
    if (copied_scheduled[j]->source.get() == top_n_op.get()) {
      copied_scheduled[j]->source = merge_ptr;
    }
  }

  // Store ownership
  inserted_operators_.push_back(std::move(merge_op));
}

void sirius_pipeline_converter::split_delim_join_sink(
  duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
  size_t pipeline_idx)
{
  auto delim_join   = current_pipeline->get_sink();
  auto& join_op     = delim_join->Cast<op::sirius_physical_delim_join>().join;
  auto* distinct_op = delim_join->Cast<op::sirius_physical_delim_join>().distinct;

  duckdb::unique_ptr<op::sirius_physical_partition> partition_join;
  if (delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    if (current_pipeline->operators.size() == 0) {
      partition_join = make_uniq<op::sirius_physical_partition>(
        current_pipeline->get_source()->types,
        current_pipeline->get_source()->estimated_cardinality,
        join_op.get(),
        true,
        op_params_.hash_partition_bytes);
    } else {
      partition_join = make_uniq<op::sirius_physical_partition>(
        current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
        current_pipeline->operators[current_pipeline->operators.size() - 1]
          .get()
          .estimated_cardinality,
        join_op.get(),
        true,
        op_params_.hash_partition_bytes);
    }
    delim_join->Cast<op::sirius_physical_right_delim_join>().partition_join =
      static_cast<op::sirius_physical_partition*>(partition_join.get());
  } else if (delim_join->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
    delim_join->Cast<op::sirius_physical_left_delim_join>().column_data_scan =
      static_cast<op::sirius_physical_column_data_scan*>(join_op->children[0].get());
  }

  // Create partition_distinct — external to delim join, in its own pipeline
  auto partition_distinct =
    make_uniq<op::sirius_physical_partition>(distinct_op->types,
                                             distinct_op->estimated_cardinality,
                                             distinct_op,
                                             false,
                                             op_params_.hash_partition_bytes);
  auto* partition_distinct_ptr =
    static_cast<op::sirius_physical_partition*>(partition_distinct.get());

  // The pipeline that contains the delim join as sink
  duckdb::shared_ptr<sirius_pipeline> delim_join_pipeline;

  if (delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN &&
      current_pipeline->operators.size() > 0) {
    // Pipeline breaker before RIGHT_DELIM_JOIN:
    // Pipeline Pre: [ops except last] -> last_op (sink)
    op::sirius_physical_operator* last_op_ptr = &current_pipeline->operators.back().get();
    current_pipeline->sink                    = last_op_ptr;
    current_pipeline->operators.erase(current_pipeline->operators.end() - 1);
    scheduled_.push_back(current_pipeline);

    // Pipeline A: last_op (source) -> RIGHT_DELIM_JOIN (sink)
    auto delim_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    delim_pipeline->source = last_op_ptr;
    delim_pipeline->sink   = delim_join.get();
    scheduled_.push_back(delim_pipeline);
    delim_join_pipeline = delim_pipeline;
  } else {
    // No pipeline breaker needed (no ops before delim join, or LEFT_DELIM_JOIN)
    scheduled_.push_back(current_pipeline);
    delim_join_pipeline = current_pipeline;
  }

  if (delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    // CONCAT pipeline: partition_join (source) -> CONCAT (sink)
    auto concat_pipeline = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    duckdb::unique_ptr<op::sirius_physical_concat> concat_op =
      make_uniq<op::sirius_physical_concat>(partition_join.get()->types,
                                            partition_join.get()->estimated_cardinality,
                                            join_op.get(),
                                            true,
                                            op_params_.concat_batch_bytes);
    concat_pipeline->source = partition_join.get();
    concat_pipeline->sink   = concat_op.get();

    inserted_operators_.push_back(std::move(partition_join));
    inserted_operators_.push_back(std::move(concat_op));
    scheduled_.push_back(concat_pipeline);
  }

  // PARTITION_DISTINCT pipeline (single-op): reads distinct output, partitions it
  auto partition_distinct_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  partition_distinct_pipeline->source = distinct_op;
  partition_distinct_pipeline->sink   = partition_distinct_ptr;
  scheduled_.push_back(partition_distinct_pipeline);

  // Merge distinct pipeline: PARTITION_DISTINCT (source) -> merge_distinct (sink)
  auto merge_distinct_op = construct_sirius_specific_operator(*distinct_op, iceberg_cache_);
  auto merge_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  merge_pipeline->source = partition_distinct_ptr;
  merge_pipeline->sink   = merge_distinct_op.get();

  // Update downstream pipelines to use MERGE_DISTINCT as source
  for (size_t j = pipeline_idx + 1; j < copied_scheduled.size(); j++) {
    if (copied_scheduled[j]->source.get() == distinct_op) {
      copied_scheduled[j]->source = merge_distinct_op.get();
    }
  }

  inserted_operators_.push_back(std::move(partition_distinct));
  inserted_operators_.push_back(std::move(merge_distinct_op));
  scheduled_.push_back(merge_pipeline);
}

void sirius_pipeline_converter::split_pipelines(
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled)
{
  for (size_t i = 0; i < copied_scheduled.size(); i++) {
    auto current_pipeline = copied_scheduled[i];  // Copy duckdb::shared_ptr to avoid invalidation

    // Preprocessing: replace TABLE_SCAN source with concrete scan operator
    split_table_scan_source(current_pipeline);

    // Preprocessing: split COLUMN_DATA_SCAN/EMPTY_RESULT/DUMMY_SCAN sources
    // into a CPU_SOURCE scan pipeline (analogous to TABLE_SCAN → PARQUET_SCAN).
    split_cpu_source(current_pipeline);

    // Preprocessing: split intermediate joins (modifies current_pipeline in place)
    split_intermediate_joins(current_pipeline);

    // Dispatch on sink type (mutually exclusive)
    auto sink_type = current_pipeline->sink->type;
    if (sink_type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
        sink_type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
      split_join_sink(current_pipeline);
    } else if (sink_type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY ||
               sink_type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE) {
      split_group_aggregate_sink(current_pipeline, copied_scheduled, i);
    } else if (sink_type == op::SiriusPhysicalOperatorType::ORDER_BY) {
      split_order_by_sink(current_pipeline, copied_scheduled, i);
    } else if (sink_type == op::SiriusPhysicalOperatorType::TOP_N) {
      split_top_n_sink(current_pipeline, copied_scheduled, i);
    } else if (sink_type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
               sink_type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
      split_delim_join_sink(current_pipeline, copied_scheduled, i);
    } else {
      scheduled_.push_back(current_pipeline);
    }
  }
}

void sirius_pipeline_converter::compute_repository_wiring(sirius_pipeline_build_state& state)
{
  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    compute_repository_wiring_tree_based(state);
    return;
  }
  (void)state;  // legacy path doesn't need state — it uses pipeline->source.
  // build source to pipelines map
  std::unordered_map<const op::sirius_physical_operator*,
                     duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>>
    source_to_pipelines;
  for (const auto& pipeline : scheduled_) {
    source_to_pipelines[pipeline->source.get()].push_back(pipeline);
  }

  // Assign pipeline IDs before emitting wiring descriptors. Runtime materialization
  // uses these to sort `_ports_list` deterministically.
  for (size_t i = 0; i < scheduled_.size(); i++) {
    scheduled_[i]->set_pipeline_id(i);
  }

  auto emit = [&](std::string_view port_id,
                  op::MemoryBarrierType barrier,
                  op::sirius_physical_operator* source_op,
                  const duckdb::shared_ptr<sirius_pipeline>& src,
                  const duckdb::shared_ptr<sirius_pipeline>& dst) {
    repository_wirings_.push_back({port_id, barrier, source_op, src, dst});
  };

  for (auto& pipeline : scheduled_) {
    auto* sink_op = pipeline->get_sink().get();

    if (pipeline->sink->type == op::SiriusPhysicalOperatorType::MERGE_GROUP_BY ||
        pipeline->sink->type == op::SiriusPhysicalOperatorType::MERGE_SORT ||
        pipeline->sink->type == op::SiriusPhysicalOperatorType::MERGE_TOP_N ||
        pipeline->sink->type == op::SiriusPhysicalOperatorType::MERGE_AGGREGATE) {
      for (auto const& dependent_pipeline : source_to_pipelines[sink_op]) {
        emit("default", op::MemoryBarrierType::FULL, sink_op, pipeline, dependent_pipeline);
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::CTE) {
      auto& cte_op = pipeline->get_sink()->Cast<op::sirius_physical_cte>();
      for (auto cte_scan : cte_op.cte_scans) {
        for (auto const& dependent_pipeline : source_to_pipelines[&cte_scan.get()]) {
          emit("default", op::MemoryBarrierType::FULL, sink_op, pipeline, dependent_pipeline);
        }
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
      auto& right_delim    = pipeline->get_sink()->Cast<op::sirius_physical_right_delim_join>();
      auto* partition_join = right_delim.partition_join;
      auto* distinct_op    = right_delim.distinct;

      // Wire partition_join -> CONCAT (partition_join pushes via its own
      // sink/next_port_after_sink)
      for (auto const& dependent_pipeline : source_to_pipelines[partition_join]) {
        emit("default", op::MemoryBarrierType::FULL, partition_join, pipeline, dependent_pipeline);
      }

      // Wire distinct_op -> partition_distinct (distinct output pushed via distinct's
      // next_port_after_sink)
      for (auto const& dependent_pipeline : source_to_pipelines[distinct_op]) {
        emit("default", op::MemoryBarrierType::FULL, distinct_op, pipeline, dependent_pipeline);
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
      auto& left_delim      = pipeline->get_sink()->Cast<op::sirius_physical_left_delim_join>();
      auto* distinct_op     = left_delim.distinct;
      auto column_data_scan = left_delim.column_data_scan;

      // Wire column_data_scan -> downstream (column_data_scan pushes via its own sink)
      for (auto const& dependent_pipeline : source_to_pipelines[column_data_scan]) {
        emit(
          "default", op::MemoryBarrierType::FULL, column_data_scan, pipeline, dependent_pipeline);
      }

      // Wire distinct_op -> partition_distinct
      for (auto const& dependent_pipeline : source_to_pipelines[distinct_op]) {
        emit("default", op::MemoryBarrierType::FULL, distinct_op, pipeline, dependent_pipeline);
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::CONCAT) {
      auto& concat             = pipeline->get_sink()->Cast<op::sirius_physical_concat>();
      std::string_view port_id = concat.is_build_concat() ? "build" : "default";

      if (concat.is_build_concat()) {
        // For build concats, no pipeline uses the concat as source. Resolve the
        // destination pipeline by finding the one whose first operator (or sink) is the
        // HASH_JOIN that this CONCAT feeds into.
        op::sirius_physical_operator* hash_join_op = concat.get_downstream_join();
        duckdb::shared_ptr<sirius_pipeline> dest_pipeline;
        for (const auto& candidate : scheduled_) {
          if ((candidate->operators.size() > 0 && &candidate->operators[0].get() == hash_join_op) ||
              candidate->sink == hash_join_op) {
            dest_pipeline = candidate;
            break;
          }
        }
        if (!dest_pipeline) {
          throw std::runtime_error(
            "Build concat: could not find pipeline with HASH_JOIN as first operator");
        }
        emit(port_id, op::MemoryBarrierType::FULL, sink_op, pipeline, dest_pipeline);
      } else {
        // Probe concats have dependent pipelines in source_to_pipelines
        for (auto const& dependent_pipeline : source_to_pipelines[sink_op]) {
          emit(port_id, op::MemoryBarrierType::FULL, sink_op, pipeline, dependent_pipeline);
        }
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::PARTITION ||
               pipeline->sink->type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE ||
               pipeline->sink->type == op::SiriusPhysicalOperatorType::TOP_N ||
               pipeline->sink->type == op::SiriusPhysicalOperatorType::MERGE_SORT ||
               pipeline->sink->type == op::SiriusPhysicalOperatorType::SORT_PARTITION) {
      for (auto const& dependent_pipeline : source_to_pipelines[sink_op]) {
        // PARTIAL barrier when the downstream is a CONCAT (it can drain incrementally);
        // otherwise FULL — wait for upstream to finish before processing.
        const bool downstream_is_concat =
          (dependent_pipeline->get_sink()->type == op::SiriusPhysicalOperatorType::CONCAT &&
           dependent_pipeline->get_operators().size() == 0) ||
          (dependent_pipeline->get_operators().size() > 0 &&
           dependent_pipeline->get_operators()[0].get().type ==
             op::SiriusPhysicalOperatorType::CONCAT);
        emit("default",
             downstream_is_concat ? op::MemoryBarrierType::PARTIAL : op::MemoryBarrierType::FULL,
             sink_op,
             pipeline,
             dependent_pipeline);
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::ORDER_BY) {
      // Pipeline barrier — downstream sample+partition pipeline processes batches as produced
      // (sort_sample overrides get_next_task_hint to wait for N batches)
      for (auto const& dependent_pipeline : source_to_pipelines[sink_op]) {
        emit("default", op::MemoryBarrierType::PIPELINE, sink_op, pipeline, dependent_pipeline);
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::CPU_SOURCE) {
      for (auto const& dependent_pipeline : source_to_pipelines[sink_op]) {
        emit("scan", op::MemoryBarrierType::PIPELINE, sink_op, pipeline, dependent_pipeline);
      }
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
      // No wiring needed for RESULT_COLLECTOR sinks
    } else {
      // Intermediate operators acting as pipeline sinks (e.g., filter, projection, join
      // placed as sink before a PARTITION pipeline). The sink pushes data to
      // next_port_after_sink via the data repo.
      for (auto const& dependent_pipeline : source_to_pipelines[sink_op]) {
        emit("default", op::MemoryBarrierType::FULL, sink_op, pipeline, dependent_pipeline);
      }
    }
  }
}

std::string_view sirius_pipeline_converter::resolve_port_id(
  const op::sirius_physical_operator& sink, const op::sirius_physical_operator& /*parent*/)
{
  using T = op::SiriusPhysicalOperatorType;
  // Build-side CONCATs feed the join's "build" port; everything else feeds "default".
  if (sink.type == T::CONCAT) {
    return sink.Cast<op::sirius_physical_concat>().is_build_concat() ? "build" : "default";
  }
  // Leaf scans push splits onto the "scan" port of the next operator.
  // GPU_PARQUET_SCAN is intentionally excluded — legacy treats it as a regular
  // intermediate operator with the "default" port (see compute_repository_wiring's
  // catch-all branch). Adding it here would diverge from the legacy wiring shape.
  if (sink.type == T::DUCKDB_SCAN || sink.type == T::ICEBERG_SCAN || sink.type == T::CPU_SOURCE) {
    return "scan";
  }
  return "default";
}

op::MemoryBarrierType sirius_pipeline_converter::resolve_barrier(
  const op::sirius_physical_operator& sink, const sirius_pipeline& dest)
{
  using T = op::SiriusPhysicalOperatorType;
  // Sort/scan/sample sinks process batches as they arrive — no barrier required.
  // GPU_PARQUET_SCAN is intentionally excluded — legacy emits FULL/PARTIAL for it via
  // the catch-all branch in compute_repository_wiring, not PIPELINE.
  if (sink.type == T::ORDER_BY || sink.type == T::SORT_SAMPLE || sink.type == T::DUCKDB_SCAN ||
      sink.type == T::ICEBERG_SCAN || sink.type == T::CPU_SOURCE) {
    return op::MemoryBarrierType::PIPELINE;
  }
  // Producers that feed CONCAT can drain incrementally (PARTIAL); otherwise wait
  // for the upstream pipeline to finish (FULL).
  if (sink.type == T::PARTITION || sink.type == T::UNGROUPED_AGGREGATE || sink.type == T::TOP_N ||
      sink.type == T::SORT_PARTITION) {
    const auto ops                  = dest.get_operators();
    const bool downstream_is_concat = (!ops.empty() && ops[0].get().type == T::CONCAT) ||
                                      (dest.get_sink() && dest.get_sink()->type == T::CONCAT);
    return downstream_is_concat ? op::MemoryBarrierType::PARTIAL : op::MemoryBarrierType::FULL;
  }
  // Intermediate sink feeding a probe-side PARTITION: the probe pipeline can
  // stream batches, so the upstream→PARTITION_probe edge uses PARTIAL. Build-
  // side PARTITION stays FULL — build must accumulate all partitions before
  // the probe can join them. Aggregate-fanout PARTITIONs (between an HGB and
  // its MERGE_GROUP_BY, or between a DISTINCT and its MERGE_DISTINCT) also
  // stay FULL: the merge operator needs every per-thread bucket before it can
  // emit output. We distinguish join-feeders from aggregate-fanouts by tree-
  // parent type — join-feeder PARTITIONs always sit under a CONCAT (the
  // CONCAT/PARTITION wrap chain emitted by wrap_join_child); aggregate-fanout
  // PARTITIONs sit under MERGE_GROUP_BY / GROUPED_AGGREGATE_MERGE. Exception
  // (#1088): a RIGHT-family join must size from the complete probe input
  // because CONCAT retains the whole probe partition, so its probe PARTITION
  // also keeps FULL; RIGHT_DELIM_JOIN's internal join is exempt — it
  // bootstraps its probe subtree from build-side distinct data. Under the
  // legacy path these same distinctions are enforced via
  // `link_join_partition_siblings`, which only patches barriers on joins'
  // probe partitions.
  if (dest.get_sink() && dest.get_sink()->type == T::PARTITION) {
    auto& partition        = dest.get_sink()->Cast<op::sirius_physical_partition>();
    auto* partition_parent = partition.get_parent_op();
    const bool join_feeder = partition_parent != nullptr && partition_parent->type == T::CONCAT;
    // Tree-parent walk: PARTITION -> CONCAT -> owning join (stamped at plan-gen).
    auto* join = join_feeder ? partition_parent->get_parent_op() : nullptr;
    const bool right_family_full =
      join && join->type == T::HASH_JOIN &&
      join->Cast<op::sirius_physical_hash_join>().is_right_family() &&
      !(join->get_parent_op() && join->get_parent_op()->type == T::RIGHT_DELIM_JOIN);
    if (!partition.is_build_partition() && join_feeder && !right_family_full) {
      return op::MemoryBarrierType::PARTIAL;
    }
  }
  return op::MemoryBarrierType::FULL;
}

void sirius_pipeline_converter::compute_repository_wiring_tree_based(
  sirius_pipeline_build_state& state)
{
  // Build a fast lookup: operator pointer -> the pipeline that "starts at" it.
  // A pipeline P starts at op X iff X is operators[0] (entry-point post-reverse)
  // OR P.sink == X (sink-only pipelines where source == sink).
  std::unordered_map<const op::sirius_physical_operator*, duckdb::shared_ptr<sirius_pipeline>>
    dest_for_op;
  for (const auto& pipeline : scheduled_) {
    const auto ops = pipeline->get_operators();
    if (!ops.empty()) { dest_for_op[&ops[0].get()] = pipeline; }
    if (pipeline->get_sink()) { dest_for_op[pipeline->get_sink().get()] = pipeline; }
  }

  // Assign pipeline IDs before emitting wiring descriptors. Runtime materialization
  // uses these to sort `_ports_list` deterministically.
  for (size_t i = 0; i < scheduled_.size(); i++) {
    scheduled_[i]->set_pipeline_id(i);
  }

  auto emit = [&](std::string_view port_id,
                  op::MemoryBarrierType barrier,
                  op::sirius_physical_operator* source_op,
                  const duckdb::shared_ptr<sirius_pipeline>& src,
                  const duckdb::shared_ptr<sirius_pipeline>& dst) {
    repository_wirings_.push_back({port_id, barrier, source_op, src, dst});
  };

  for (auto& pipeline : scheduled_) {
    auto* sink_op = pipeline->get_sink().get();
    if (!sink_op) { continue; }

    using T = op::SiriusPhysicalOperatorType;

    // RESULT_COLLECTOR is a terminal sink — nothing to emit.
    if (sink_op->type == T::RESULT_COLLECTOR) { continue; }

    // CTE iterates its sibling `cte_scans` (parent_op alone doesn't encode them).
    // Lookup goes through `state.cte_scan_consumers` rather than `dest_for_op`
    // because CTE_SCAN is a routing-only marker and never lands in any
    // pipeline's operators[] under flag ON (matches legacy's contract). The map
    // is populated by `sirius_physical_column_data_scan::build_pipelines`.
    if (sink_op->type == T::CTE) {
      auto& cte_op = sink_op->Cast<op::sirius_physical_cte>();
      for (auto cte_scan : cte_op.cte_scans) {
        auto it = state.cte_scan_consumers.find(cte_scan);
        if (it == state.cte_scan_consumers.end()) { continue; }
        auto dest_pipeline = it->second.get().shared_from_this();
        // Per-consumer barrier: probe-side CTE_SCAN consumers (e.g. q15's first
        // CTE_SCAN feeding the main HJ's probe PARTITION) resolve to PARTIAL via
        // the join-feeder rule; build-side consumers (e.g. q15's second CTE_SCAN
        // feeding the scalar-subquery aggregate chain) resolve to FULL. Hardcoding
        // FULL matches legacy on the build side but disagrees on the probe side.
        emit(
          "default", resolve_barrier(*sink_op, *dest_pipeline), sink_op, pipeline, dest_pipeline);
      }
      continue;
    }

    // RIGHT_DELIM_JOIN: emit partition_join + distinct sibling references.
    if (sink_op->type == T::RIGHT_DELIM_JOIN) {
      auto& right_delim    = sink_op->Cast<op::sirius_physical_right_delim_join>();
      auto* partition_join = right_delim.partition_join;
      auto* distinct_op    = right_delim.distinct;
      if (partition_join) {
        auto it = dest_for_op.find(partition_join);
        // B.3+B.4+B.6 (#604): partition_join is owned by the DELIM_JOIN and executed
        // inline (RIGHT_DELIM_JOIN::sink). Under flag ON it has no pipeline of its own —
        // build_join_pipelines skips the recursion that would have created one — so the
        // direct lookup misses. Fall back to its tree parent (CONCAT_build), which is
        // the build_meta sink and resolves to the externalized [CONCAT_build] single-op
        // pipeline. Matches legacy `PARTITION → CONCAT` build wiring.
        if (it == dest_for_op.end()) {
          if (auto* parent = partition_join->get_parent_op()) { it = dest_for_op.find(parent); }
        }
        if (it != dest_for_op.end()) {
          emit("default", op::MemoryBarrierType::FULL, partition_join, pipeline, it->second);
        }
      }
      if (distinct_op) {
        auto it = dest_for_op.find(distinct_op);
        // B.5 (#604): the bare DISTINCT is owned by DELIM_JOIN and executed inline
        // (RIGHT_DELIM_JOIN::sink). Under flag ON it has no pipeline of its own —
        // its build_pipelines override short-circuits when `_owned_by_delim_join`
        // is set — so the direct lookup misses. Fall back to its tree parent
        // (PARTITION_distinct), which is the partition pipeline that consumes the
        // bare DISTINCT's per-thread output. Matches legacy
        // `HASH_GROUP_BY (src=DELIM_JOIN) → PARTITION (dst=PARTITION_distinct)`.
        if (it == dest_for_op.end()) {
          if (auto* parent = distinct_op->get_parent_op()) { it = dest_for_op.find(parent); }
        }
        if (it != dest_for_op.end()) {
          emit("default", op::MemoryBarrierType::FULL, distinct_op, pipeline, it->second);
        }
      }
      continue;
    }

    // LEFT_DELIM_JOIN: emit column_data_scan + distinct sibling references.
    if (sink_op->type == T::LEFT_DELIM_JOIN) {
      auto& left_delim       = sink_op->Cast<op::sirius_physical_left_delim_join>();
      auto* distinct_op      = left_delim.distinct;
      auto* column_data_scan = left_delim.column_data_scan;
      if (column_data_scan) {
        auto it = dest_for_op.find(column_data_scan);
        // LEFT_DELIM_JOIN ownership (#604): column_data_scan is owned by the
        // delim join and executed inline (LEFT_DELIM_JOIN::sink). Under flag ON
        // its build_pipelines is a no-op, so the direct lookup misses. Fall
        // back to its tree parent (PARTITION_probe), which carries the
        // externalized [PARTITION] pipeline that consumes the cached chunk
        // scan's output. Matches legacy's
        // `COLUMN_DATA_SCAN (src=DELIM_JOIN) → PARTITION (dst=PARTITION_probe)`.
        // Use resolve_barrier so the dest type (PARTITION_probe) dictates the
        // barrier (PARTIAL for probe-side partition, per join-feeder rule).
        if (it == dest_for_op.end()) {
          if (auto* parent = column_data_scan->get_parent_op()) { it = dest_for_op.find(parent); }
        }
        if (it != dest_for_op.end()) {
          emit("default",
               resolve_barrier(*column_data_scan, *it->second),
               column_data_scan,
               pipeline,
               it->second);
        }
      }
      if (distinct_op) {
        auto it = dest_for_op.find(distinct_op);
        // Same fallback as B.5 for RIGHT: bare DISTINCT has no pipeline of its
        // own under flag ON, so resolve to its tree parent (PARTITION_distinct).
        if (it == dest_for_op.end()) {
          if (auto* parent = distinct_op->get_parent_op()) { it = dest_for_op.find(parent); }
        }
        if (it != dest_for_op.end()) {
          emit("default", op::MemoryBarrierType::FULL, distinct_op, pipeline, it->second);
        }
      }
      continue;
    }

    // B.1' (#604): the distinct chain top of a DELIM_JOIN (MERGE_GROUP_BY) sits
    // under DELIM_JOIN in the tree, so the uniform tree-parent walk below would
    // emit `merge_top -> DELIM_JOIN`. Legacy split_delim_join_sink instead
    // retargets the merged output to each delim_scan's downstream consumer
    // (the inner-HJ probe partition). Mirror that here. Detection uses the
    // explicit `_owning_delim_join` back-pointer set in wrap_delim_distinct —
    // only the distinct_root carries it.
    if (auto* owning_delim = sink_op->owning_delim_join()) {
      for (auto& delim_scan_ref : owning_delim->delim_scans) {
        auto& delim_scan  = delim_scan_ref.get();
        auto* scan_parent = delim_scan.get_parent_op();
        if (!scan_parent) { continue; }
        auto cit = dest_for_op.find(scan_parent);
        if (cit == dest_for_op.end()) { continue; }
        emit(resolve_port_id(*sink_op, *scan_parent),
             resolve_barrier(*sink_op, *cit->second),
             sink_op,
             pipeline,
             cit->second);
      }
      continue;
    }

    // Uniform tree-parent lookup for everything else.
    auto* parent_op = sink_op->get_parent_op();
    if (!parent_op) { continue; }

    // B.7 (#604): when sink_op is the `delim.join` of a RIGHT_DELIM_JOIN, the
    // legacy split_delim_join_sink wires the inner join out to the next HJ
    // above the RDJ (via the legacy-constructed external partition_join), not
    // to the RDJ itself. Mirror that: redirect to the RDJ's tree parent so
    // the inner HJ skips over the RDJ. Without this, both `RDJ.children[0]`'s
    // root HJ AND `RDJ.delim.join` resolve to the same RDJ pipeline, and the
    // RDJ-sink emission's CONCAT fallback (line 1300) closes a cycle back
    // through the inner HJ's own build CONCAT.
    if ((parent_op->type == T::RIGHT_DELIM_JOIN) &&
        parent_op->Cast<op::sirius_physical_delim_join>().join.get() == sink_op) {
      auto* grand = parent_op->get_parent_op();
      if (grand) {
        auto it_gp = dest_for_op.find(grand);
        if (it_gp != dest_for_op.end()) {
          emit(resolve_port_id(*sink_op, *grand),
               resolve_barrier(*sink_op, *it_gp->second),
               sink_op,
               pipeline,
               it_gp->second);
          continue;
        }
      }
    }

    auto it = dest_for_op.find(parent_op);
    if (it == dest_for_op.end()) { continue; }

    const auto& dest = it->second;
    emit(resolve_port_id(*sink_op, *parent_op),
         resolve_barrier(*sink_op, *dest),
         sink_op,
         pipeline,
         dest);
  }
}

void sirius_pipeline_converter::setup_pipeline_parents()
{
  // Derive parents off the wiring descriptors instead of reading materialised ports —
  // ports aren't attached until `materialize_repository_wiring()` runs after `convert()`
  // returns. Each descriptor encodes a `source_pipeline -> dest_pipeline` edge that the
  // old code derived from `add_next_port_after_sink({next_op, port_id})`
  for (const auto& pipeline : scheduled_) {
    pipeline->parents.clear();
    pipeline->dependencies.clear();
  }
  for (const auto& wiring : repository_wirings_) {
    wiring.source_pipeline->parents.push_back(
      duckdb::weak_ptr<sirius_pipeline>(wiring.dest_pipeline));
  }
}

void sirius_pipeline_converter::finalize_pipeline_structure()
{
  // Finalize pipeline structure: push sink into operators, set source
  // AFTER THIS POINT: operators[] contains ALL operators (source through sink).
  // source = &operators[0], sink = operators.back().
  for (const auto& pipeline : scheduled_) {
    if (!duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
      // Phase 3.2 (#604): under USE_TREE_BASED_PIPELINE_BUILD, is_ready (C.1)
      // already pushed sink into operators[] and set source = &operators[0].
      // Skip the redoing here; the parent->dependency reverse map still needs
      // to be populated (next loop), so the function as a whole is still
      // called under both flag states until Sub-phase E deletes it.
      pipeline->operators.push_back(*pipeline->sink);
      pipeline->source = &pipeline->operators[0].get();
    }
    // for each parent pipeline, add the current pipeline to the dependencies
    for (auto& parent : pipeline->parents) {
      if (auto locked_parent = parent.lock()) { locked_parent->dependencies.push_back(pipeline); }
    }
  }
}

void sirius_pipeline_converter::link_join_partition_siblings()
{
  for (const auto& pipeline : scheduled_) {
    // for each hash/nested-loop join as a source, get the dependencies (concat) and get the
    // dependencies of concat (partition). Both join types receive the same CONCAT/PARTITION
    // build+probe wrap from `wrap_join` at plan-gen, and the probe pipeline can stream batches
    // for both, so the upstream→probe-partition edge should use PARTIAL in both cases.
    if (pipeline->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
        pipeline->source->type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
      auto build_concat_pipeline    = pipeline->dependencies[0];
      auto build_partition_pipeline = build_concat_pipeline->dependencies[0];
      auto probe_concat_pipeline    = pipeline->dependencies[1];
      auto probe_partition_pipeline = probe_concat_pipeline->dependencies[0];
      bool const is_right_delim     = build_partition_pipeline->get_sink()->type ==
                                  op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN;
      // RIGHT_DELIM_JOIN must bootstrap its probe subtree from build-side distinct data.
      // Right-family sizing applies to hash joins only — NLJ probe partitions always stream.
      bool const probe_drives_partition_count =
        pipeline->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN &&
        pipeline->source->Cast<op::sirius_physical_hash_join>().is_right_family() &&
        !is_right_delim;

      // Probe partitions normally stream through a partial barrier. A RIGHT-family join must
      // size from the complete probe input because CONCAT retains the whole probe partition.
      // The corresponding port doesn't exist yet (materialisation happens after `convert()`
      // returns); mutate the descriptor so the materialiser creates the port with the correct
      // barrier type.
      //
      // Under USE_TREE_BASED_PIPELINE_BUILD, resolve_barrier already emits the
      // upstream→PARTITION_probe edge with the right FULL/PARTIAL barrier directly. Skip
      // the mutation under flag ON to keep barrier decisions consolidated in one
      // place; the sibling-pointer setup below still runs in both flag states.
      if (!duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
        auto wiring_it = std::find_if(
          repository_wirings_.begin(), repository_wirings_.end(), [&](const repository_wiring& w) {
            return w.dest_pipeline == probe_partition_pipeline && w.port_id == "default";
          });
        D_ASSERT(wiring_it != repository_wirings_.end());
        wiring_it->barrier_type = probe_drives_partition_count ? op::MemoryBarrierType::FULL
                                                               : op::MemoryBarrierType::PARTIAL;
      }
      if (is_right_delim) {
        // partition pipeline only has one operator
        auto& right_delim_join_op =
          build_partition_pipeline->get_sink()->Cast<op::sirius_physical_right_delim_join>();
        auto build_partition_op = right_delim_join_op.partition_join;
        auto& probe_partition_op =
          probe_partition_pipeline->get_sink()->Cast<op::sirius_physical_partition>();
        build_partition_op->set_sibling_partition_op(&probe_partition_op);
        probe_partition_op.set_sibling_partition_op(build_partition_op);
      } else {
        // partition pipeline only has one operator, so sink and source are the same
        auto& build_partition_op =
          build_partition_pipeline->get_sink()->Cast<op::sirius_physical_partition>();
        auto& probe_partition_op =
          probe_partition_pipeline->get_sink()->Cast<op::sirius_physical_partition>();
        build_partition_op.set_sibling_partition_op(&probe_partition_op);
        probe_partition_op.set_sibling_partition_op(&build_partition_op);
        if (probe_drives_partition_count) {
          build_partition_op.set_drives_partition_count(false);
          probe_partition_op.set_drives_partition_count(true);
        }
      }
    }
  }
}

void sirius_pipeline_converter::configure_partition_min_partitions()
{
  // Pull num_gpus from the build context (populated from sirius_engine's
  // hardware topology at convert time). Single-GPU runs keep the default
  // min of 1 (no-op). For multi-GPU we force a floor equal to num_gpus on
  // big-enough inputs; small_table_bytes keeps tiny aggregations on a
  // single GPU to avoid cross-device overhead.
  const int num_gpus = build_ctx_.num_gpus();
  if (num_gpus <= 1) return;
  // Heuristic threshold: below ~16 MiB per GPU the partition overhead
  // dominates. Configurable later if we find a workload where this matters.
  const uint64_t small_table_bytes = static_cast<uint64_t>(num_gpus) * uint64_t{16} * 1024 * 1024;

  auto apply_to_op = [&](op::sirius_physical_operator* op) {
    if (op && op->type == op::SiriusPhysicalOperatorType::PARTITION) {
      static_cast<op::sirius_physical_partition*>(op)->set_min_num_partitions(num_gpus,
                                                                              small_table_bytes);
    }
  };
  for (auto& breaker : inserted_operators_) {
    apply_to_op(breaker.get());
  }
  for (auto& pipe : scheduled_) {
    if (!pipe) continue;
    auto sink   = pipe->get_sink();
    auto source = pipe->get_source();
    if (sink) apply_to_op(sink.get());
    if (source) apply_to_op(source.get());
  }
}

void sirius_pipeline_converter::log_pipeline_debug_info() const
{
  // Detailed pipeline debugging information
  SIRIUS_LOG_INFO("\n=== DETAILED PIPELINE DEBUG INFO ===");
  for (size_t i = 0; i < scheduled_.size(); i++) {
    auto pipeline = scheduled_[i];
    SIRIUS_LOG_INFO("Pipeline #{}", i);
    SIRIUS_LOG_INFO(
      "  Source: {} (id={})", pipeline->source->get_name(), pipeline->source->get_operator_id());

    // Print operators
    for (size_t j = 0; j < pipeline->operators.size(); j++) {
      auto& op = pipeline->operators[j].get();
      SIRIUS_LOG_INFO("    Operator[{}]: {} (id={})", j, op.get_name(), op.get_operator_id());
    }

    SIRIUS_LOG_INFO(
      "  Sink: {} (id={})", pipeline->sink->get_name(), pipeline->sink->get_operator_id());

    // Print ports at operator[0] (beginning of pipeline)
    if (pipeline->operators.size() > 0) {
      auto& first_op = pipeline->operators[0].get();
      SIRIUS_LOG_INFO(
        "  Ports at Operator[0] ({}, id={}):", first_op.get_name(), first_op.get_operator_id());

      // Check for different port types based on operator type
      if (first_op.type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
          first_op.type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
        // Joins have "default" and "build" ports
        auto* default_port = first_op.get_port("default");
        if (default_port) {
          SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                          static_cast<int>(default_port->type),
                          static_cast<void*>(default_port->repo));
        }
        auto* build_port = first_op.get_port("build");
        if (build_port) {
          SIRIUS_LOG_INFO("    Port 'build': barrier_type={}, repo={}",
                          static_cast<int>(build_port->type),
                          static_cast<void*>(build_port->repo));
        }
      } else if (first_op.type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
        const auto& scan_name = first_op.Cast<op::sirius_physical_table_scan>().function.name;
        if (scan_name != "seq_scan" && scan_name != "parquet_scan" && scan_name != "read_parquet" &&
            scan_name != "sirius_read_parquet" && scan_name != "iceberg_scan") {
          throw std::runtime_error("Unsupported scan function: " + scan_name);
        }
        // Scans have "scan" port
        auto* scan_port = first_op.get_port("scan");
        if (scan_port) {
          SIRIUS_LOG_INFO("    Port 'scan': barrier_type={}, repo={}",
                          static_cast<int>(scan_port->type),
                          static_cast<void*>(scan_port->repo));
        }
      } else if (first_op.type == op::SiriusPhysicalOperatorType::GPU_SCAN ||
                 first_op.type == op::SiriusPhysicalOperatorType::CPU_SOURCE ||
                 first_op.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR ||
                 first_op.type == op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN ||
                 first_op.type == op::SiriusPhysicalOperatorType::EMPTY_RESULT ||
                 first_op.type == op::SiriusPhysicalOperatorType::DUMMY_SCAN) {
        // scan-like operators don't have a "default" port. GPU_SCAN gets
        // its splits via the scan_manager's connector, not via a port.
      } else {
        // Most operators have "default" port
        auto* default_port = first_op.get_port("default");
        if (default_port) {
          SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                          static_cast<int>(default_port->type),
                          static_cast<void*>(default_port->repo));
        }
      }
    } else {
      SIRIUS_LOG_INFO("  No operators in pipeline - checking sink ports");
      auto* sink = pipeline->sink.get();

      if (sink->type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
          sink->type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
        auto* default_port = sink->get_port("default");
        if (default_port) {
          SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                          static_cast<int>(default_port->type),
                          static_cast<void*>(default_port->repo));
        }
        auto* build_port = sink->get_port("build");
        if (build_port) {
          SIRIUS_LOG_INFO("    Port 'build': barrier_type={}, repo={}",
                          static_cast<int>(build_port->type),
                          static_cast<void*>(build_port->repo));
        }
      } else if (sink->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
        auto* scan_port = sink->get_port("scan");
        if (scan_port) {
          SIRIUS_LOG_INFO("    Port 'scan': barrier_type={}, repo={}",
                          static_cast<int>(scan_port->type),
                          static_cast<void*>(scan_port->repo));
        }
      } else if (sink->type == op::SiriusPhysicalOperatorType::GPU_SCAN ||
                 sink->type == op::SiriusPhysicalOperatorType::CPU_SOURCE ||
                 sink->type == op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN ||
                 sink->type == op::SiriusPhysicalOperatorType::EMPTY_RESULT ||
                 sink->type == op::SiriusPhysicalOperatorType::DUMMY_SCAN) {
        // scan-like operators don't have ports
      } else if (sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
        // ignore RESULT_COLLECTOR since it doesn't have ports
      } else {
        auto* default_port = sink->get_port("default");
        if (default_port) {
          SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                          static_cast<int>(default_port->type),
                          static_cast<void*>(default_port->repo));
        }
      }
    }

    // Print ports and next operators after sink
    SIRIUS_LOG_INFO("  Sink's next operators and ports:");
    for (auto& next_port : pipeline->get_next_ports_after_sink()) {
      auto next_op = next_port.next_operator;
      auto port_id = next_port.next_operator_port_name;
      SIRIUS_LOG_INFO("    Next Op: {} (id={}), Port: '{}'",
                      next_op->get_name(),
                      next_op->get_operator_id(),
                      port_id.data());

      // Print the port details if it exists
      auto* port = next_op->get_port(port_id);
      SIRIUS_LOG_INFO("      Port barrier_type={}, repo={}",
                      static_cast<int>(port->type),
                      static_cast<void*>(port->repo));
    }

    SIRIUS_LOG_INFO("");  // Blank line between pipelines
  }
  SIRIUS_LOG_INFO("=== END DETAILED PIPELINE DEBUG INFO ===\n");
}

std::string dump_pipeline_conversion_result(const pipeline_conversion_result& result)
{
  auto op_name = [](const op::sirius_physical_operator* op) -> std::string {
    return op == nullptr ? std::string{"(null)"} : op::SiriusPhysicalOperatorToString(op->type);
  };

  auto barrier_name = [](op::MemoryBarrierType b) -> std::string {
    switch (b) {
      case op::MemoryBarrierType::PIPELINE: return "PIPELINE";
      case op::MemoryBarrierType::PARTIAL: return "PARTIAL";
      case op::MemoryBarrierType::FULL: return "FULL";
    }
    return "?";
  };

  // Per-pipeline local signature: sink|source|operators... Used as the base layer of the
  // recursive signature below.
  auto local_sig = [&](const sirius_pipeline* p) -> std::string {
    std::string s = op_name(p->get_sink().get());
    s += "|" + op_name(p->get_source().get());
    for (const auto& op_ref : p->get_operators()) {
      s += "|" + op_name(&op_ref.get());
    }
    return s;
  };

  // Downstream-aware signature: a pipeline's signature includes its sorted list of
  // (port_id, downstream_signature). Two pipelines with the same local shape AND the
  // same set of consumers reach the same signature. Memoized; assumes acyclic graph
  // (TPC-H plans are DAGs).
  std::unordered_map<const sirius_pipeline*, std::string> sig_cache;
  // Defensive cycle guard: if the wiring graph ever contains a cycle, compute_sig
  // would otherwise infinite-recurse. Returning "CYCLE" on re-entry keeps the dump
  // function safe; the underlying bug surfaces as a dump mismatch which is easier
  // to triage than a hang. TPC-H plans are DAGs in correct converter output, so
  // this guard should never fire in green CI.
  std::unordered_set<const sirius_pipeline*> sig_visiting;
  std::function<std::string(const sirius_pipeline*)> compute_sig =
    [&](const sirius_pipeline* p) -> std::string {
    auto it = sig_cache.find(p);
    if (it != sig_cache.end()) { return it->second; }
    if (sig_visiting.count(p)) { return "CYCLE"; }
    sig_visiting.insert(p);
    std::vector<std::string> down;
    for (const auto& w : result.repository_wirings) {
      if (w.source_pipeline.get() == p) {
        down.push_back(std::string{w.port_id} + ":" + compute_sig(w.dest_pipeline.get()));
      }
    }
    sig_visiting.erase(p);
    std::sort(down.begin(), down.end());
    std::string s = local_sig(p);
    for (const auto& d : down) {
      s += "|>" + d;
    }
    sig_cache[p] = s;
    return s;
  };

  // Canonical pipeline order: sort by signature. Order in `scheduled_pipelines` reflects
  // the scheduling pass that produced it (legacy iterative split vs tree-based DFS);
  // sorting here makes the dump comparison independent of that order so both flag states
  // produce byte-identical output when the resulting graph is the same.
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> ordered = result.scheduled_pipelines;
  std::sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) -> bool {
    return compute_sig(a.get()) < compute_sig(b.get());
  });

  std::unordered_map<const sirius_pipeline*, std::size_t> pipeline_to_index;
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    pipeline_to_index[ordered[i].get()] = i;
  }
  auto idx_of = [&](const duckdb::shared_ptr<sirius_pipeline>& p) -> std::size_t {
    auto it = pipeline_to_index.find(p.get());
    return it == pipeline_to_index.end() ? std::numeric_limits<std::size_t>::max() : it->second;
  };
  auto pipeline_index = [&](const duckdb::shared_ptr<sirius_pipeline>& p) -> std::string {
    auto it = pipeline_to_index.find(p.get());
    return it == pipeline_to_index.end() ? std::string{"?"} : std::to_string(it->second);
  };

  std::ostringstream out;
  out << "=== pipelines (" << ordered.size() << ") ===\n";
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    auto& p = *ordered[i];
    out << "[pipeline " << i << "]\n";
    out << "  source: " << op_name(p.get_source().get()) << "\n";
    out << "  sink: " << op_name(p.get_sink().get()) << "\n";
    const auto ops = p.get_operators();
    out << "  operators (" << ops.size() << "):\n";
    std::size_t op_idx = 0;
    for (const auto& op_ref : ops) {
      out << "    [" << op_idx++ << "] " << op_name(&op_ref.get()) << "\n";
    }
  }

  // Sort wirings by the canonical pipeline indices so wiring order is also
  // path-independent.
  std::vector<repository_wiring> sorted_wirings(result.repository_wirings.begin(),
                                                result.repository_wirings.end());
  std::sort(sorted_wirings.begin(),
            sorted_wirings.end(),
            [&](const repository_wiring& a, const repository_wiring& b) -> bool {
              auto a_src = idx_of(a.source_pipeline);
              auto b_src = idx_of(b.source_pipeline);
              if (a_src != b_src) { return a_src < b_src; }
              auto a_dst = idx_of(a.dest_pipeline);
              auto b_dst = idx_of(b.dest_pipeline);
              if (a_dst != b_dst) { return a_dst < b_dst; }
              return a.port_id < b.port_id;
            });

  out << "\n=== repository_wirings (" << sorted_wirings.size() << ") ===\n";
  for (std::size_t i = 0; i < sorted_wirings.size(); ++i) {
    const auto& w = sorted_wirings[i];
    out << "[wiring " << i << "]\n";
    out << "  port_id: " << w.port_id << "\n";
    out << "  barrier: " << barrier_name(w.barrier_type) << "\n";
    out << "  source_op: " << op_name(w.source_op) << "\n";
    out << "  src_pipeline: " << pipeline_index(w.source_pipeline) << "\n";
    out << "  dest_pipeline: " << pipeline_index(w.dest_pipeline) << "\n";
  }
  return out.str();
}

}  // namespace sirius::pipeline
