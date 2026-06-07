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

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/shared_ptr_ipp.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_native_scan_info.hpp"
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/sirius_gpu_duckdb_native_scan_operator.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
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
#include <numeric>
#include <stdexcept>

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
  absorbed_pipelines_.clear();

  auto copied_scheduled = schedule_and_copy_pipelines(root_pipeline);
  split_pipelines(copied_scheduled);
  compute_repository_wiring();
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
  // collect all meta-pipelines from the root pipeline
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
        if (pipeline->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN) {
          auto& temp = pipeline->source.get()->Cast<op::sirius_physical_hash_join>();
          if (temp.join_type == duckdb::JoinType::RIGHT ||
              temp.join_type == duckdb::JoinType::RIGHT_SEMI ||
              temp.join_type == duckdb::JoinType::RIGHT_ANTI) {
            // if (!duckdb::Config::MODIFIED_PIPELINE) sirius_scheduled.push_back(pipeline);
          }
          continue;
        } else {
          sirius_scheduled.push_back(pipeline);
        }
      }
      schedule_count++;
    }
    meta = (meta + 1) % to_schedule.size();
  }

  // perform deep copy on scheduled pipelines
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

  auto scan_info            = std::make_unique<op::scan::parquet_scan_info>();
  scan_info->returned_types = scan_op.returned_types;
  scan_info->column_ids     = scan_op.column_ids;
  scan_info->projection_ids = scan_op.projection_ids;
  scan_info->names          = scan_op.names;
  scan_info->table_filters  = std::move(scan_op.table_filters);

  if (scan_op.function.name == "sirius_read_parquet") {
    // Sirius-owned S3 table function: the single URI lives in parameters[0]
    // (its bind returns no MultiFileBindData). No hive partitioning on the S3
    // single-file path, so partition_indices is left empty.
    if (scan_op.parameters.empty() || scan_op.parameters.front().IsNull()) {
      throw std::runtime_error(
        "[sirius_pipeline_converter::insert_parquet_scan_operator] sirius_read_parquet scan "
        "has no URI parameter");
    }
    scan_info->file_paths = {scan_op.parameters.front().GetValue<std::string>()};
  } else {
    // Native read_parquet / parquet_scan: file paths come from MultiFileBindData.
    auto const& bind_data = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
    if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
      throw std::runtime_error(
        "[sirius_pipeline_converter::insert_parquet_scan_operator] No input files to scan");
    }
    std::vector<std::string> file_paths;
    for (auto const& file : bind_data.file_list->GetAllFiles()) {
      file_paths.push_back(file.path);
    }
    scan_info->file_paths        = std::move(file_paths);
    scan_info->partition_indices = bind_data.reader_bind.hive_partitioning_indexes;
  }
  // upstream #749's polymorphic scan_info requires scan_output_arity (consumed by
  // scan_info::make_provider). sql-surface's branch above didn't set it.
  scan_info->scan_output_arity = scan_op.types.size();
  // upstream #792 added approximate_batch_size, driven by operator_params.
  scan_info->approximate_batch_size = op_params_.scan_task_batch_size;

  auto gpu_scan_op = duckdb::make_uniq<op::scan::sirius_gpu_parquet_scan_operator>(
    scan_op.types, scan_op.estimated_cardinality, std::move(scan_info));

  auto* gpu_scan_ptr = gpu_scan_op.get();

  // finalize_pipeline_structure() will set current_pipeline->source = &operators[0] = gpu_scan_op.
  current_pipeline->operators.insert(current_pipeline->operators.begin(), *gpu_scan_ptr);

  inserted_operators_.push_back(std::move(gpu_scan_op));
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
      "tables. Disable enable_gpu_duckdb_native_scan for this query.");
  }
  auto& bind_data = *table_scan_bind;
  auto& table     = bind_data.table.Cast<duckdb::DuckTableEntry>();

  if (client_context_ == nullptr) {
    throw std::runtime_error(
      "[sirius_pipeline_converter::insert_duckdb_native_scan_operator] no client_context passed "
      "to converter; seq_scan GPU-native path requires it");
  }

  auto scan_info     = std::make_unique<op::scan::duckdb_native_scan_info>();
  scan_info->storage = &table.GetStorage();
  scan_info->context = client_context_;
  scan_info->db_path = table.GetStorage().GetAttached().GetStorageManager().GetDBPath();
  scan_info->approximate_batch_size = op_params_.scan_task_batch_size;

  std::vector<std::size_t> source_ids_fallback;
  if (scan_op.projection_ids.empty()) {
    source_ids_fallback.resize(scan_op.column_ids.size());
    std::iota(source_ids_fallback.begin(), source_ids_fallback.end(), 0);
  }
  auto const& source_ids =
    scan_op.projection_ids.empty() ? source_ids_fallback : scan_op.projection_ids;

  // source_ids may exceed scan_op.types when sirius_plan_get.cpp appends filter-only columns
  // (merge_pushdown_filters). decode_duckdb_native_split applies table_filters and projects
  // down to output_types — same pattern as sirius_physical_table_scan::execute.
  scan_info->projected_cols.reserve(source_ids.size());
  scan_info->projected_types.reserve(source_ids.size());
  for (std::size_t k = 0; k < source_ids.size(); ++k) {
    auto pid            = source_ids[k];
    auto const& col_idx = scan_op.column_ids[pid];
    op::scan::projected_column pc;
    pc.is_rowid = col_idx.IsRowIdColumn();
    if (!pc.is_rowid) { pc.storage_idx = duckdb::StorageIndex(col_idx.GetPrimaryIndex()); }
    scan_info->projected_cols.push_back(pc);

    sirius::logical_type t;
    if (k < scan_op.types.size()) {
      t = scan_op.types[k];
    } else {
      t = scan_op.returned_types.at(col_idx.GetPrimaryIndex());
    }
    scan_info->projected_types.push_back(t);
  }

  // Pass filter context to decode_duckdb_native_split (table_filters apply post-decode).
  if (scan_op.table_filters) {
    scan_info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    for (auto& [col_idx, filt] : scan_op.table_filters->filters) {
      scan_info->table_filters->filters[col_idx] = filt->Copy();
    }
  }
  scan_info->column_ids     = scan_op.column_ids;
  scan_info->projection_ids = scan_op.projection_ids;
  scan_info->returned_types = scan_op.returned_types;
  scan_info->output_types   = scan_op.types;

  auto gpu_scan_op = duckdb::make_uniq<op::scan::sirius_gpu_duckdb_native_scan_operator>(
    scan_op.types, scan_op.estimated_cardinality, std::move(scan_info));

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

  if (scan_op.function.name == "seq_scan" && op_params_.enable_gpu_duckdb_native_scan) {
    insert_duckdb_native_scan_operator(current_pipeline);
    return;
  }

  if (scan_op.function.name == "seq_scan" || scan_op.function.name == "iceberg_scan") {
    auto new_pipeline = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);

    auto new_scan_op = construct_sirius_specific_operator(scan_op, iceberg_cache_);
    // todo(bobbi) currently this can be set to any operator since it's never used, and now we
    // set it to scan_op
    new_pipeline->source = nullptr;
    new_pipeline->sink   = new_scan_op.get();

    current_pipeline->source = new_scan_op.get();
    // move scan_op to current_pipeline.operator[0], current_pipeline.operator[0] to
    // current_pipeline.operator[1], ...
    current_pipeline->operators.insert(current_pipeline->operators.begin(), scan_op);

    scheduled_.push_back(new_pipeline);
    inserted_operators_.push_back(std::move(new_scan_op));
  } else {
    throw std::runtime_error("Unsupported scan function: " + scan_op.function.name);
  }
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
        concat_op.get(),
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
        concat_op.get(),
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
      concat_op.get(),
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
      concat_op.get(),
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

namespace {

bool is_fusable_downstream_operator(const op::sirius_physical_operator& op)
{
  return !op.is_sink() && !op.is_source();
}

// Must stay in sync with split_pipelines() sink dispatch: any sink handled by a
// dedicated split_* function must not be absorbed via downstream fusion.
bool sink_needs_pipeline_split(op::SiriusPhysicalOperatorType type)
{
  return type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
         type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN ||
         type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY ||
         type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE ||
         type == op::SiriusPhysicalOperatorType::ORDER_BY ||
         type == op::SiriusPhysicalOperatorType::TOP_N ||
         type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
         type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN;
}

}  // namespace

bool sirius_pipeline_converter::try_fuse_downstream_pipelineable(
  duckdb::shared_ptr<sirius_pipeline>& merge_pipeline,
  op::sirius_physical_operator* merge_op,
  op::sirius_physical_operator* original_agg_source,
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
  size_t from_pipeline_idx)
{
  // If any downstream pipeline from this aggregate ends in a structural sink, do not fuse.
  for (size_t j = from_pipeline_idx + 1; j < copied_scheduled.size(); j++) {
    auto& downstream = copied_scheduled[j];
    if (downstream->source.get() != original_agg_source) { continue; }
    if (sink_needs_pipeline_split(downstream->get_sink()->type)) { return false; }
  }

  for (size_t j = from_pipeline_idx + 1; j < copied_scheduled.size(); j++) {
    auto& downstream = copied_scheduled[j];
    if (downstream->source.get() != original_agg_source) { continue; }

    // Do not fuse a pipelineable downstream when a later structural pipeline reads
    // from its output (e.g. GROUP_BY -> PROJECTION then PROJECTION -> HASH_JOIN).
    for (size_t k = j + 1; k < copied_scheduled.size(); k++) {
      auto& later = copied_scheduled[k];
      if (!sink_needs_pipeline_split(later->get_sink()->type)) { continue; }
      const auto* later_source = later->source.get();
      if (later_source == downstream->get_sink().get()) { return false; }
      for (auto& op_ref : downstream->operators) {
        if (later_source == &op_ref.get()) { return false; }
      }
    }

    for (auto& op_ref : downstream->operators) {
      if (!is_fusable_downstream_operator(op_ref.get())) { return false; }
    }

    merge_pipeline->operators.push_back(*merge_op);
    for (auto& op_ref : downstream->operators) {
      merge_pipeline->operators.push_back(op_ref);
    }
    merge_pipeline->sink = downstream->sink;

    absorbed_pipelines_.insert(downstream.get());
    return true;
  }
  return false;
}

void sirius_pipeline_converter::attach_merge_and_fuse_downstream(
  duckdb::shared_ptr<sirius_pipeline>& merge_pipeline,
  op::sirius_physical_operator* merge_op,
  op::sirius_physical_operator* original_agg_source,
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
  size_t from_pipeline_idx)
{
  // UNGROUPED_AGGREGATE uses the same physical op as both the partial-aggregate sink and the
  // merge pipeline source. Folding the downstream pipeline (typically RESULT_COLLECTOR) into the
  // merge pipeline breaks partial→merge repository wiring and produces wrong merged results
  // (e.g. avg off by orders of magnitude). Keep merge and downstream as separate pipelines.
  const bool allow_fusion =
    original_agg_source->type != op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE;
  const bool fused =
    allow_fusion &&
    try_fuse_downstream_pipelineable(
      merge_pipeline, merge_op, original_agg_source, copied_scheduled, from_pipeline_idx);

  if (!fused) { merge_pipeline->sink = merge_op; }

  // Repoint not-yet-split downstream pipelines to read from the merge op.
  for (size_t j = from_pipeline_idx + 1; j < copied_scheduled.size(); j++) {
    auto& downstream = copied_scheduled[j];
    if (absorbed_pipelines_.count(downstream.get()) > 0) { continue; }
    if (downstream->source.get() == original_agg_source) { downstream->source = merge_op; }
  }
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

    // Create merge pipeline: PARTITION (source) -> MERGE_OP -> pipelineable downstream ops
    auto merge_op          = construct_sirius_specific_operator(*group_agg_op, iceberg_cache_);
    auto* merge_op_ptr     = merge_op.get();
    auto merge_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    merge_pipeline->source = partition_ptr;

    attach_merge_and_fuse_downstream(
      merge_pipeline, merge_op_ptr, group_agg_op.get(), copied_scheduled, pipeline_idx);

    scheduled_.push_back(merge_pipeline);
    inserted_operators_.push_back(std::move(merge_op));
  } else {
    // UNGROUPED_AGGREGATE — no PARTITION needed
    scheduled_.push_back(current_pipeline);

    auto merge_op        = construct_sirius_specific_operator(*group_agg_op, iceberg_cache_);
    auto* merge_op_ptr   = merge_op.get();
    auto new_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
    new_pipeline->source = group_agg_op;

    attach_merge_and_fuse_downstream(
      new_pipeline, merge_op_ptr, group_agg_op.get(), copied_scheduled, pipeline_idx);

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

  // Pipeline B: TOP_N (source) -> MERGE_TOP_N -> pipelineable downstream ops
  auto merge_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  merge_pipeline->source = top_n_op.get();

  attach_merge_and_fuse_downstream(
    merge_pipeline, merge_ptr, top_n_op.get(), copied_scheduled, pipeline_idx);

  scheduled_.push_back(merge_pipeline);

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
  auto& distinct_op = delim_join->Cast<op::sirius_physical_delim_join>().distinct;

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
                                             distinct_op.get(),
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
  partition_distinct_pipeline->source = distinct_op.get();
  partition_distinct_pipeline->sink   = partition_distinct_ptr;
  scheduled_.push_back(partition_distinct_pipeline);

  // Merge distinct pipeline: PARTITION_DISTINCT (source) -> merge_distinct (sink)
  auto merge_distinct_op = construct_sirius_specific_operator(*distinct_op, iceberg_cache_);
  auto merge_pipeline    = duckdb::make_shared_ptr<sirius_pipeline>(build_ctx_);
  merge_pipeline->source = partition_distinct_ptr;
  merge_pipeline->sink   = merge_distinct_op.get();

  // Update downstream pipelines to use MERGE_DISTINCT as source
  for (size_t j = pipeline_idx + 1; j < copied_scheduled.size(); j++) {
    if (copied_scheduled[j]->source.get() == distinct_op.get()) {
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
    if (absorbed_pipelines_.count(current_pipeline.get()) > 0) { continue; }

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

void sirius_pipeline_converter::compute_repository_wiring()
{
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
      auto* distinct_op    = right_delim.distinct.get();

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
      auto* distinct_op     = left_delim.distinct.get();
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
        // HASH_JOIN stored in parent_op.
        op::sirius_physical_operator* hash_join_op = concat.get_parent_op();
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
    } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN ||
               pipeline->sink->type == op::SiriusPhysicalOperatorType::ICEBERG_SCAN ||
               pipeline->sink->type == op::SiriusPhysicalOperatorType::CPU_SOURCE) {
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
    pipeline->operators.push_back(*pipeline->sink);
    pipeline->source = &pipeline->operators[0].get();
    // for each parent pipeline, add the current pipeline to the dependencies
    for (auto& parent : pipeline->parents) {
      if (auto locked_parent = parent.lock()) { locked_parent->dependencies.push_back(pipeline); }
    }
  }
}

void sirius_pipeline_converter::link_join_partition_siblings()
{
  for (const auto& pipeline : scheduled_) {
    // for each hash join as a source, get the dependencies (concat) and get the dependencies of
    // concat (partition)
    if (pipeline->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN) {
      auto build_concat_pipeline    = pipeline->dependencies[0];
      auto build_partition_pipeline = build_concat_pipeline->dependencies[0];
      auto probe_concat_pipeline    = pipeline->dependencies[1];
      auto probe_partition_pipeline = probe_concat_pipeline->dependencies[0];
      // Change probe partition barrier to partial. The corresponding port doesn't exist
      // yet (materialisation happens after `convert()` returns); mutate the descriptor
      // so the materialiser creates the port with the correct barrier type.
      auto wiring_it = std::find_if(
        repository_wirings_.begin(), repository_wirings_.end(), [&](const repository_wiring& w) {
          return w.dest_pipeline == probe_partition_pipeline && w.port_id == "default";
        });
      D_ASSERT(wiring_it != repository_wirings_.end());
      wiring_it->barrier_type = op::MemoryBarrierType::PARTIAL;
      if (build_partition_pipeline->get_sink()->type ==
          op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
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
  const int num_gpus = build_ctx_.num_gpus;
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
      } else if (first_op.type == op::SiriusPhysicalOperatorType::GPU_PARQUET_SCAN ||
                 first_op.type == op::SiriusPhysicalOperatorType::GPU_DUCKDB_NATIVE_SCAN ||
                 first_op.type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN ||
                 first_op.type == op::SiriusPhysicalOperatorType::ICEBERG_SCAN ||
                 first_op.type == op::SiriusPhysicalOperatorType::CPU_SOURCE ||
                 first_op.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR ||
                 first_op.type == op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN ||
                 first_op.type == op::SiriusPhysicalOperatorType::EMPTY_RESULT ||
                 first_op.type == op::SiriusPhysicalOperatorType::DUMMY_SCAN) {
        // scan-like operators don't have a "default" port. GPU_PARQUET_SCAN gets
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
      } else if (sink->type == op::SiriusPhysicalOperatorType::GPU_PARQUET_SCAN ||
                 sink->type == op::SiriusPhysicalOperatorType::GPU_DUCKDB_NATIVE_SCAN ||
                 sink->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN ||
                 sink->type == op::SiriusPhysicalOperatorType::ICEBERG_SCAN ||
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

}  // namespace sirius::pipeline
