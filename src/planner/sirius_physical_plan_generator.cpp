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

#include "planner/sirius_physical_plan_generator.hpp"

#include "config.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_gpu_values.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "planner/sirius_plan_compressed_schema.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <cudf/cudf_utils.hpp>

#include <numeric>
#include <utility>

namespace sirius::planner {

std::vector<std::string> resolve_parquet_scan_file_paths(
  std::string_view function_name,
  duckdb::FunctionData const* bind_data,
  duckdb::vector<duckdb::Value> const& parameters)
{
  if (function_name == "sirius_read_parquet") {
    // Internal S3 rewrite target: its bind_data is SiriusReadParquetBindData, not
    // MultiFileBindData — the resolved URI travels in parameters[0].
    if (parameters.empty() || parameters.front().IsNull()) { return {}; }
    return {parameters.front().GetValue<std::string>()};
  }
  if (function_name == "parquet_scan" || function_name == "read_parquet") {
    // dynamic_cast (never Cast<>, which asserts/throws): an unresolvable
    // identity must degrade to empty, not fail the caller.
    auto const* multi_file_bind = dynamic_cast<duckdb::MultiFileBindData const*>(bind_data);
    if (multi_file_bind == nullptr || !multi_file_bind->file_list ||
        multi_file_bind->file_list->IsEmpty()) {
      return {};
    }
    std::vector<std::string> file_paths;
    for (auto const& file : multi_file_bind->file_list->GetAllFiles()) {
      file_paths.push_back(file.path);
    }
    return file_paths;
  }
  return {};
}

namespace {
bool is_nested_logical_type(duckdb::LogicalType const& type)
{
  auto const id = type.id();
  return id == duckdb::LogicalTypeId::STRUCT || id == duckdb::LogicalTypeId::LIST ||
         id == duckdb::LogicalTypeId::MAP;
}

//! Insert `factory(std::move(parent.children[i]))` between `parent` and its i-th child. The
//! factory takes ownership of the original child and must return a wrapper subtree that
//! already holds the original as a descendant.
template <typename WrapperFactory>
void wrap_child(sirius::op::sirius_physical_operator& parent,
                std::size_t i,
                WrapperFactory&& factory)
{
  auto original      = std::move(parent.children[i]);
  auto wrapper       = std::forward<WrapperFactory>(factory)(std::move(original));
  parent.children[i] = std::move(wrapper);
}

//! Replace the operator at `slot` with `factory(std::move(slot))`: used for sink-wrap rewrites
//! where the original operator becomes a descendant of the wrapper that now occupies its slot.
template <typename WrapperFactory>
void wrap_above(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                WrapperFactory&& factory)
{
  auto original = std::move(slot);
  slot          = std::forward<WrapperFactory>(factory)(std::move(original));
}

//! Build a `parquet_ingestible_table_info` from a TABLE_SCAN. Destructive: `table_filters`
//! is moved out of the scan.
std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> build_parquet_table_info(
  sirius::op::sirius_physical_table_scan& scan_op, const sirius::operator_params& op_params)
{
  auto info                = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  info->returned_types     = scan_op.returned_types;
  info->column_ids         = scan_op.column_ids;
  info->projection_ids     = scan_op.projection_ids;
  info->names              = scan_op.names;
  info->table_filters      = std::move(scan_op.table_filters);
  auto resolved_file_paths = resolve_parquet_scan_file_paths(
    scan_op.function.name, scan_op.bind_data.get(), scan_op.parameters);
  if (scan_op.function.name == "sirius_read_parquet") {
    if (resolved_file_paths.empty()) {
      throw std::runtime_error(
        "[sirius_physical_plan_generator::build_parquet_table_info] sirius_read_parquet scan "
        "has no URI parameter");
    }
    info->resolved_file_paths = std::move(resolved_file_paths);
  } else {
    if (resolved_file_paths.empty()) {
      throw std::runtime_error(
        "[sirius_physical_plan_generator::build_parquet_table_info] No input files to scan");
    }
    info->resolved_file_paths = std::move(resolved_file_paths);
    auto const& bind_data     = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
    info->partition_indices   = bind_data.reader_bind.hive_partitioning_indexes;
  }
  // `scan_output_arity` drives the provider's expected column count — without it the runtime
  // task skips the hive-partition columns it should inject post-read, mis-sizing the output.
  info->scan_output_arity      = scan_op.types.size();
  info->approximate_batch_size = op_params.scan_task_batch_size;
  return info;
}

//! Build a `duckdb_native_ingestible_table_info` from a `seq_scan` TABLE_SCAN. Requires a
//! live `ClientContext` (the ingestible reads table storage during `prepare_for_query`);
//! throws on non-base-table scans.
std::unique_ptr<sirius::op::scan::duckdb_native_ingestible_table_info>
build_duckdb_native_table_info(sirius::op::sirius_physical_table_scan& scan_op,
                               const sirius::operator_params& op_params,
                               duckdb::ClientContext& context)
{
  if (!scan_op.bind_data) {
    throw std::runtime_error(
      "[sirius_physical_plan_generator::build_duckdb_native_table_info] seq_scan has no bind_data");
  }
  auto* table_scan_bind = dynamic_cast<duckdb::TableScanBindData*>(scan_op.bind_data.get());
  if (table_scan_bind == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_plan_generator::build_duckdb_native_table_info] seq_scan bind_data is not "
      "TableScanBindData; the GPU-native duckdb scan path supports only seq_scan over base "
      "tables.");
  }
  auto& bind_data = *table_scan_bind;
  auto& table     = bind_data.table.Cast<duckdb::DuckTableEntry>();

  auto info     = std::make_unique<sirius::op::scan::duckdb_native_ingestible_table_info>();
  info->storage = &table.GetStorage();
  info->context = &context;
  info->db_path = table.GetStorage().GetAttached().GetStorageManager().GetDBPath();
  // Qualified-name identity for the pin cache — derived from the resolved
  // DuckTableEntry so it matches the pin-side derivation (build_duckdb_pin_info)
  // exactly. Without these a pin_table(format='duckdb', ...) query silently misses
  // the pinned cache and falls through to disk.
  info->catalog_name           = table.ParentCatalog().GetName();
  info->schema_name            = table.ParentSchema().name;
  info->table_name             = table.name;
  info->approximate_batch_size = op_params.scan_task_batch_size;

  std::vector<std::size_t> source_ids_fallback;
  if (scan_op.projection_ids.empty()) {
    source_ids_fallback.resize(scan_op.column_ids.size());
    std::iota(source_ids_fallback.begin(), source_ids_fallback.end(), 0);
  }
  auto const& source_ids =
    scan_op.projection_ids.empty() ? source_ids_fallback : scan_op.projection_ids;

  info->projected_cols.reserve(source_ids.size());
  info->projected_types.reserve(source_ids.size());
  for (std::size_t k = 0; k < source_ids.size(); ++k) {
    auto pid            = source_ids[k];
    auto const& col_idx = scan_op.column_ids[pid];
    sirius::op::scan::projected_column pc;
    pc.is_rowid = col_idx.IsRowIdColumn();
    if (!pc.is_rowid) { pc.storage_idx = duckdb::StorageIndex(col_idx.GetPrimaryIndex()); }
    info->projected_cols.push_back(pc);

    sirius::logical_type t;
    if (k < scan_op.types.size()) {
      t = scan_op.types[k];
    } else {
      t = scan_op.returned_types.at(col_idx.GetPrimaryIndex());
    }
    info->projected_types.push_back(t);
  }

  // Filters drive row-group pruning in the metadata walk and post-decode filtering.
  if (scan_op.table_filters) {
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    for (auto& [col_idx, filt] : scan_op.table_filters->filters) {
      info->table_filters->filters[col_idx] = filt->Copy();
    }
  }
  info->column_ids     = scan_op.column_ids;
  info->projection_ids = scan_op.projection_ids;
  info->returned_types = scan_op.returned_types;
  info->output_types   = scan_op.types;
  return info;
}

/**
 * @brief Builds a GPU scan, wrapping it when registered dynamic-filter producers exist
 *
 * @p mode selects membership-only or AST-plus-membership post-decode filtering.
 */
template <typename InfoT>
duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_gpu_scan_leaf(
  std::unique_ptr<InfoT> info,
  const sirius::op::sirius_physical_table_scan& scan,
  const sirius::operator_params& op_params,
  sirius::op::scan::dynamic_filter_apply_mode mode,
  duckdb::SiriusContext* compressed_materialization_observer)
{
  auto dynamic_filters = scan.sirius_dynamic_filters;
  if (dynamic_filters && !dynamic_filters->has_producers()) { dynamic_filters.reset(); }
  info->sirius_dynamic_filters = dynamic_filters;

  auto ingestible = sirius::op::scan::make_ingestible(std::move(info));
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> leaf =
    duckdb::make_uniq<sirius::op::scan::sirius_gpu_scan_operator>(
      scan.types,
      scan.estimated_cardinality,
      std::move(ingestible),
      compressed_materialization_observer);
  // Preserve propagated carriers; dynamic-filter targets are already native.
  if (scan.has_physical_overrides()) { leaf->set_physical_types(scan.get_physical_types()); }

  if (dynamic_filters) {
    auto dynamic_filter_op = duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
      scan.types,
      scan.estimated_cardinality,
      std::move(dynamic_filters),
      op_params.dynamic_filter_keep_threshold,
      mode);
    if (scan.has_physical_overrides()) {
      dynamic_filter_op->set_physical_types(scan.get_physical_types());
    }
    dynamic_filter_op->children.push_back(std::move(leaf));
    leaf = std::move(dynamic_filter_op);
  }
  return leaf;
}

void require_complete_native_scan_schema(const sirius::op::sirius_physical_table_scan& scan)
{
  for (std::size_t column_idx = 0; column_idx < scan.types.size(); ++column_idx) {
    auto const& type = scan.types[column_idx];
    if (sirius::try_get_cudf_type(type)) { continue; }
    throw duckdb::NotImplementedException(
      "GPU scan output column %llu (%s) has no native cuDF "
      "carrier",
      static_cast<unsigned long long>(column_idx),
      type.to_string());
  }
}

//! Rewrite a TABLE_SCAN for `seq_scan` / `parquet_scan` / `read_parquet` /
//! `sirius_read_parquet` (the internal S3 rewrite target): REPLACE the slot with the GPU
//! leaf so it inherits the TABLE_SCAN's tree position and stays the source-leaf of the
//! existing pipeline. Rejects unsupported scan functions and output types without a native cuDF
//! carrier while plan construction can still trigger transparent CPU fallback.
void wrap_table_scan_source(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& table_scan_slot,
  const sirius::operator_params& op_params,
  duckdb::ClientContext& context)
{
  // Table-in-out functions wear a TABLE_SCAN with children — skip them; wrapping would change
  // a child layout the converter and downstream operators don't expect.
  if (!table_scan_slot->children.empty()) { return; }

  auto& scan     = table_scan_slot->Cast<sirius::op::sirius_physical_table_scan>();
  const auto& fn = scan.function.name;
  // GPU_SCAN normalization requires one target per output column. Reject an incomplete schema
  // while transparent execution can still fall back to DuckDB.
  require_complete_native_scan_schema(scan);
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> leaf;
  bool replace_slot = false;
  if (fn == "seq_scan") {
    // Native scans apply AST-capable filters post-decode.
    leaf         = make_gpu_scan_leaf(build_duckdb_native_table_info(scan, op_params, context),
                              scan,
                              op_params,
                              sirius::op::scan::dynamic_filter_apply_mode::include_ast_row_masks,
                              sirius_ctx.get());
    replace_slot = true;
  } else if (fn == "parquet_scan" || fn == "read_parquet" || fn == "sirius_read_parquet") {
    // Parquet applies AST filters in the reader; post-decode uses membership only.
    leaf         = make_gpu_scan_leaf(build_parquet_table_info(scan, op_params),
                              scan,
                              op_params,
                              sirius::op::scan::dynamic_filter_apply_mode::membership_masks_only,
                              sirius_ctx.get());
    replace_slot = true;
  } else {
    throw std::runtime_error(
      "[sirius_physical_plan_generator::wrap_table_scan_source] Unsupported scan function: " + fn);
  }
  if (replace_slot) {
    table_scan_slot = std::move(leaf);
  } else {
    table_scan_slot->children.push_back(std::move(leaf));
  }
}

//! Replace a COLUMN_DATA_SCAN, EMPTY_RESULT, or DUMMY_SCAN slot in place with a GPU_VALUES
//! source. Unlike the scan-companion wraps, no leaf child and no extra pipeline is created:
//! GPU_VALUES is a first-class in-pipeline source driven by the normal task creation
//! loop, exactly like GPU_SCAN (same shape as wrap_table_scan_source's replace path).
//! A null-collection COLUMN_DATA_SCAN is the LEFT_DELIM_JOIN cached chunk scan (filled
//! at runtime) — left as-is. The viability gates run BEFORE the collection is moved out so an
//! unsupported type or oversized single-task source falls back to DuckDB CPU with its data intact.
void replace_with_gpu_values(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                             std::size_t max_source_bytes)
{
  if (!slot->children.empty()) { return; }

  duckdb::unique_ptr<sirius::op::sirius_physical_gpu_values> gpu_values;
  switch (slot->type) {
    case sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN: {
      auto& col_scan = slot->Cast<sirius::op::sirius_physical_column_data_scan>();
      if (!col_scan.collection) { return; }  // LEFT_DELIM_JOIN cached chunk scan — skip
      sirius::op::sirius_physical_gpu_values::throw_if_unsupported_types(slot->types);
      sirius::op::sirius_physical_gpu_values::throw_if_collection_too_large(*col_scan.collection,
                                                                            max_source_bytes);
      gpu_values = duckdb::make_uniq<sirius::op::sirius_physical_gpu_values>(
        slot->types, slot->estimated_cardinality, std::move(col_scan.collection));
      break;
    }
    case sirius::op::SiriusPhysicalOperatorType::DUMMY_SCAN:
      sirius::op::sirius_physical_gpu_values::throw_if_unsupported_types(slot->types);
      gpu_values = duckdb::make_uniq<sirius::op::sirius_physical_gpu_values>(
        slot->types, slot->estimated_cardinality, /*produce_single_row=*/true);
      break;
    case sirius::op::SiriusPhysicalOperatorType::EMPTY_RESULT:
      sirius::op::sirius_physical_gpu_values::throw_if_unsupported_types(slot->types);
      gpu_values = duckdb::make_uniq<sirius::op::sirius_physical_gpu_values>(
        slot->types, slot->estimated_cardinality, /*produce_single_row=*/false);
      break;
    default: return;
  }
  slot = std::move(gpu_values);
}

//! Replace a HASH_GROUP_BY slot with `GROUPED_AGGREGATE_MERGE → PARTITION → HASH_GROUP_BY →
//! original_input`: the original stays the per-thread sink, PARTITION buckets for the merge.
//! The aggregate's physical sidecar (narrow group keys, native aggregate outputs) is copied onto
//! both wrappers: PARTITION forwards the partial-aggregate batches unchanged, and the merge
//! re-groups with raw key views so its output carriers equal the aggregate's.
void wrap_hash_group_by(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                        const sirius::operator_params& op_params,
                        duckdb::SiriusContext* compressed_materialization_observer)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> hgb_op) {
    auto* hgb_ptr = hgb_op.get();

    // Construct the merge while the child still carries the final SQL schema. COUNT(DISTINCT)
    // emits LIST sets locally and only becomes BIGINT after merge post-processing.
    auto merge = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate_merge>(
      &hgb_ptr->Cast<sirius::op::sirius_physical_grouped_aggregate>(),
      op_params.hash_partition_bytes);
    if (hgb_ptr->has_physical_overrides()) {
      merge->set_physical_types(hgb_ptr->get_physical_types());
    }

    auto& grouped = hgb_ptr->Cast<sirius::op::sirius_physical_grouped_aggregate>();
    bool const has_supported_count_distinct_layout =
      grouped.has_count_distinct && !grouped.has_avg && !hgb_ptr->has_physical_overrides() &&
      hgb_ptr->types.size() == grouped.group_idx.size() + grouped.aggregate_slots.size();
    if (has_supported_count_distinct_layout) {
      hgb_ptr->types = grouped.get_count_distinct_local_output_types();
    }

    // The aggregate-fanout partition is the one consumer of runtime size estimation today: with
    // a projected total it fixes its partition count from the first batch, which is what lets
    // its ingress run as a PARTIAL rather than a FULL barrier (see resolve_barrier).
    auto partition = duckdb::make_uniq<sirius::op::sirius_physical_partition>(
      hgb_ptr->types,
      hgb_ptr->estimated_cardinality,
      /*key_source=*/hgb_ptr,
      /*is_build=*/false,
      compressed_materialization_observer,
      op_params.enable_runtime_size_estimation,
      op_params.size_estimate_safety_factor);
    auto* partition_ptr = partition.get();
    if (hgb_ptr->has_physical_overrides()) {
      partition->set_physical_types(hgb_ptr->get_physical_types());
    }
    partition->children.push_back(std::move(hgb_op));

    // The partition's downstream sizing consumer is the merge (key_source hgb only supplies keys).
    partition_ptr->set_downstream_consumer_op(merge.get());
    merge->children.push_back(std::move(partition));
    return merge;
  });
}

//! Replace an UNGROUPED_AGGREGATE slot with `UNGROUPED_AGGREGATE_MERGE → UNGROUPED_AGGREGATE →
//! original_input`. No PARTITION: the merge consumes the single per-thread accumulator directly.
void wrap_ungrouped_aggregate(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> ungrouped_op) {
    auto* ungrouped_ptr = ungrouped_op.get();
    auto merge          = duckdb::make_uniq<sirius::op::sirius_physical_ungrouped_aggregate_merge>(
      &ungrouped_ptr->Cast<sirius::op::sirius_physical_ungrouped_aggregate>());
    // MERGE_AGGREGATE keeps the final SQL schema copied above. The child emits per-task
    // accumulator carriers instead: AVG is SUM + COUNT, so its runtime width is larger than the
    // final width. Recording that schema on the child keeps task-level output validation exact.
    ungrouped_ptr->types = ungrouped_ptr->Cast<sirius::op::sirius_physical_ungrouped_aggregate>()
                             .get_local_output_types();
    merge->children.push_back(std::move(ungrouped_op));
    return merge;
  });
}

//! Replace a TOP_N slot with `TOP_N_MERGE → TOP_N → original_input`.
void wrap_top_n(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> topn_op) {
    auto* topn_ptr = &topn_op->Cast<sirius::op::sirius_physical_top_n>();
    auto merge     = duckdb::make_uniq<sirius::op::sirius_physical_top_n_merge>(topn_ptr);
    merge->children.push_back(std::move(topn_op));
    return merge;
  });
}

//! Replace an ORDER_BY slot with `MERGE_SORT → SORT_PARTITION → SORT_SAMPLE → ORDER_BY →
//! original_input`. Destructive: ORDER_BY's
//! `projections`/`types` are overwritten with the identity over the input's types so every
//! column stays visible to SORT_SAMPLE / SORT_PARTITION; a non-identity original projection is
//! restored on MERGE_SORT via `set_final_projections`. SORT_SAMPLE is a non-sink, so it lands
//! in `operators[]` of the SORT_PARTITION pipeline (3-pipeline shape, matching legacy).
void wrap_order_by(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                   const sirius::operator_params& op_params)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> order_op) {
    auto* order_ptr = &order_op->Cast<sirius::op::sirius_physical_order>();
    if (order_ptr->children.empty()) {
      throw std::runtime_error(
        "[sirius_physical_plan_generator::wrap_order_by] ORDER_BY has no child input");
    }

    auto original_projections = order_ptr->projections;
    auto const& child_types   = order_ptr->children[0]->types;

    duckdb::vector<std::size_t> identity_proj;
    identity_proj.reserve(child_types.size());
    for (std::size_t col_idx = 0; col_idx < child_types.size(); col_idx++) {
      identity_proj.push_back(col_idx);
    }
    order_ptr->projections = std::move(identity_proj);
    order_ptr->types       = child_types;

    auto sample = duckdb::make_uniq<sirius::op::sirius_physical_sort_sample>(
      order_ptr,
      op_params.sort_sample_bytes,
      op_params.max_sort_partition_bytes,
      op_params.max_sort_partition_memory_fraction);
    auto* sample_ptr = sample.get();
    sample->children.push_back(std::move(order_op));

    auto partition = duckdb::make_uniq<sirius::op::sirius_physical_sort_partition>(order_ptr);
    partition->set_sample_op(sample_ptr);
    partition->children.push_back(std::move(sample));

    auto merge = duckdb::make_uniq<sirius::op::sirius_physical_merge_sort>(order_ptr);

    bool is_identity = (original_projections.size() == order_ptr->types.size());
    if (is_identity) {
      for (std::size_t i = 0; i < original_projections.size(); i++) {
        if (original_projections[i] != i) {
          is_identity = false;
          break;
        }
      }
    }
    if (!is_identity) {
      duckdb::vector<sirius::logical_type> output_types;
      output_types.reserve(original_projections.size());
      for (auto idx : original_projections) {
        output_types.push_back(order_ptr->types[idx]);
      }
      merge->set_final_projections(std::move(original_projections), std::move(output_types));
    }

    merge->children.push_back(std::move(partition));
    return merge;
  });
}

//! Wrap `join_op.children[child_idx]` with `CONCAT → PARTITION → original_child`. `is_build`
//! flips build/probe semantics in both wrappers; `join_op` is PARTITION's `key_source`
//! (determines partition keys) and CONCAT's `downstream_join` (picks the coalescing mode).
void wrap_join_child(sirius::op::sirius_physical_operator& join_op,
                     std::size_t child_idx,
                     bool is_build,
                     const sirius::operator_params& op_params,
                     duckdb::SiriusContext* compressed_materialization_observer)
{
  D_ASSERT(join_op.type == sirius::op::SiriusPhysicalOperatorType::HASH_JOIN ||
           join_op.type == sirius::op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN);
  auto* join_op_ptr = &join_op;
  wrap_child(
    join_op, child_idx, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> child_orig) {
      // Capture types/cardinality before the child is moved into PARTITION.
      auto child_types    = child_orig->types;
      auto est_card       = child_orig->estimated_cardinality;
      auto child_physical = child_orig->has_physical_overrides() ? child_orig->get_physical_types()
                                                                 : std::vector<cudf::data_type>{};

      auto concat =
        duckdb::make_uniq<sirius::op::sirius_physical_concat>(child_types,
                                                              est_card,
                                                              /*downstream_join=*/join_op_ptr,
                                                              is_build,
                                                              op_params.concat_batch_bytes);
      auto partition = duckdb::make_uniq<sirius::op::sirius_physical_partition>(
        std::move(child_types),
        est_card,
        /*key_source=*/join_op_ptr,
        is_build,
        compressed_materialization_observer);
      if (!child_physical.empty()) {
        partition->set_physical_types(child_physical);
        concat->set_physical_types(std::move(child_physical));
      }
      partition->children.push_back(std::move(child_orig));
      concat->children.push_back(std::move(partition));
      return concat;
    });
}

//! Wrap each child of a DENSE_COUNT_JOIN with a bare `PARTITION → original_child`, so the fused
//! operator consumes one partition of each side per task instead of both inputs whole.
//!
//! No CONCAT, unlike the join wrap: CONCAT exists to fold a join's build side into one batch for
//! its hash table, and this operator builds no hash table — it drains whole partitions.
//!
//! The preserved side is the build side, which makes it the sizing driver
//! (`_drives_partition_count = _is_build`). Because sizing reads both sides' bytes, both
//! partitions are told to wait for their sibling's input before negotiating a count.
void wrap_dense_count_join(sirius::op::sirius_physical_operator& dense_count_op,
                           duckdb::SiriusContext* compressed_materialization_observer)
{
  D_ASSERT(dense_count_op.type == sirius::op::SiriusPhysicalOperatorType::DENSE_COUNT_JOIN);
  D_ASSERT(dense_count_op.children.size() == 2);
  auto* dense_count_ptr = &dense_count_op;

  // children[0] is the preserved side, children[1] the counted side (stamped by
  // try_plan_dense_count_join).
  for (std::size_t child_idx = 0; child_idx < 2; ++child_idx) {
    bool const is_build = (child_idx == 0);
    wrap_child(dense_count_op,
               child_idx,
               [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> child_orig) {
                 auto child_types    = child_orig->types;
                 auto est_card       = child_orig->estimated_cardinality;
                 auto child_physical = child_orig->has_physical_overrides()
                                         ? child_orig->get_physical_types()
                                         : std::vector<cudf::data_type>{};

                 auto partition = duckdb::make_uniq<sirius::op::sirius_physical_partition>(
                   std::move(child_types),
                   est_card,
                   /*key_source=*/dense_count_ptr,
                   is_build,
                   compressed_materialization_observer);
                 if (!child_physical.empty()) {
                   partition->set_physical_types(std::move(child_physical));
                 }
                 partition->set_downstream_consumer_op(dense_count_ptr);
                 partition->set_sizing_requires_sibling_input(true);
                 partition->children.push_back(std::move(child_orig));
                 return partition;
               });
  }
}

//! Wrap both children of a HASH_JOIN / NESTED_LOOP_JOIN with the CONCAT/PARTITION feeder
//! chain: probe = children[0], build = children[1]. A missing side is skipped.
void wrap_join(sirius::op::sirius_physical_operator& join_op,
               const sirius::operator_params& op_params,
               duckdb::SiriusContext* compressed_materialization_observer)
{
  if (join_op.children.size() >= 1) {
    wrap_join_child(
      join_op, /*child_idx=*/0, /*is_build=*/false, op_params, compressed_materialization_observer);
  }
  if (join_op.children.size() >= 2) {
    wrap_join_child(
      join_op, /*child_idx=*/1, /*is_build=*/true, op_params, compressed_materialization_observer);
  }
}

// Forward declaration: wrap_delim_join recurses into a DELIM JOIN's internal `join`/`distinct`
// subtrees, which live outside `children[]`.
void insert_gpu_pipeline_operators_recursive(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
  const sirius::operator_params& op_params,
  duckdb::ClientContext& context,
  duckdb::SiriusContext* compressed_materialization_observer);

//! Replace a DELIM JOIN's `distinct_root` (the bare DISTINCT) with `DISTINCT_MERGE ->
//! PARTITION_DISTINCT -> original DISTINCT`. The non-owning `delim_base.distinct` borrow stays
//! valid — moving a unique_ptr never relocates the object — and the fan-out wiring uses it.
//! The propagation pass restores `distinct_root` in place, so this chain never carries a
//! physical sidecar and its PARTITION needs no compressed-materialization observer.
void wrap_delim_distinct(sirius::op::sirius_physical_delim_join& delim_base,
                         const sirius::operator_params& op_params)
{
  if (!delim_base.distinct_root) { return; }

  auto original          = std::move(delim_base.distinct_root);
  auto* original_agg_ptr = &original->Cast<sirius::op::sirius_physical_grouped_aggregate>();

  auto partition =
    duckdb::make_uniq<sirius::op::sirius_physical_partition>(original->types,
                                                             original->estimated_cardinality,
                                                             /*key_source=*/original.get(),
                                                             /*is_build=*/false);
  auto* partition_ptr = partition.get();
  partition->children.push_back(std::move(original));

  auto merge = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate_merge>(
    original_agg_ptr, op_params.hash_partition_bytes);
  // The partition's downstream sizing consumer is the merge (key_source only supplies keys).
  partition_ptr->set_downstream_consumer_op(merge.get());
  merge->children.push_back(std::move(partition));

  // Tag the chain top with the owning DELIM_JOIN so the tree-parent wiring redirects its
  // output to each delim_scan's consumer pipeline instead of the DELIM_JOIN itself.
  merge->set_owning_delim_join(&delim_base);

  delim_base.distinct_root = std::move(merge);
}

//! Rewrite a DELIM JOIN's internal subtrees and wire the sibling pointers the operator needs
//! at runtime (both the internal `join` and the distinct chain live in the tree):
//!   - recurse into `delim->join` so its source/sink/join wraps fire — this plants the
//!     CONCAT/PARTITION chain on the internal join's build side;
//!   - recurse into the bare DISTINCT's children first, then wrap_delim_distinct above it,
//!     so the chain wrap never re-visits the freshly-inserted wrappers;
//!   - RIGHT_DELIM_JOIN: point `partition_join` at the freshly-inserted build-side PARTITION.
void wrap_delim_join(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                     const sirius::operator_params& op_params,
                     duckdb::ClientContext& context,
                     duckdb::SiriusContext* compressed_materialization_observer)
{
  auto& delim_base = slot->Cast<sirius::op::sirius_physical_delim_join>();

  if (delim_base.join) {
    insert_gpu_pipeline_operators_recursive(
      delim_base.join, op_params, context, compressed_materialization_observer);
  }
  if (delim_base.distinct_root) {
    // `distinct_root` still holds the bare DISTINCT: wrap below it first, then above.
    for (auto& child_slot : delim_base.distinct_root->children) {
      insert_gpu_pipeline_operators_recursive(
        child_slot, op_params, context, compressed_materialization_observer);
    }
    wrap_delim_distinct(delim_base, op_params);
  }

  if (slot->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& right_delim = slot->Cast<sirius::op::sirius_physical_right_delim_join>();

    auto* internal_join = delim_base.join.get();
    if (internal_join && internal_join->children.size() >= 2) {
      auto* build_child = internal_join->children[1].get();
      if (build_child && build_child->type == sirius::op::SiriusPhysicalOperatorType::CONCAT &&
          !build_child->children.empty()) {
        auto* partition_build = build_child->children[0].get();
        if (partition_build &&
            partition_build->type == sirius::op::SiriusPhysicalOperatorType::PARTITION) {
          right_delim.partition_join =
            &partition_build->Cast<sirius::op::sirius_physical_partition>();
        }
      }
    }
  }
}

//! Post-order recursive walk over the plan tree: children are rewritten before the dispatch on
//! `slot->type`, so freshly-inserted wrapper subtrees are never re-visited and no node can be
//! double-wrapped. DELIM JOIN recurses into its internal `join`/`distinct` fields, which live
//! outside `children[]`.
void insert_gpu_pipeline_operators_recursive(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
  const sirius::operator_params& op_params,
  duckdb::ClientContext& context,
  duckdb::SiriusContext* compressed_materialization_observer)
{
  if (!slot) { return; }

  for (auto& child_slot : slot->children) {
    insert_gpu_pipeline_operators_recursive(
      child_slot, op_params, context, compressed_materialization_observer);
  }

  switch (slot->type) {
    case sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN:
      wrap_table_scan_source(slot, op_params, context);
      break;
    case sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN:
    case sirius::op::SiriusPhysicalOperatorType::EMPTY_RESULT:
      replace_with_gpu_values(slot,
                              op_params.scan_task_batch_size == 0
                                ? sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE
                                : op_params.scan_task_batch_size);
      break;
    case sirius::op::SiriusPhysicalOperatorType::DUMMY_SCAN: {
      // A RIGHT_DELIM_JOIN build-placeholder DUMMY_SCAN carries no runtime data (the delim join
      // fans its input directly to PARTITION_build), so a GPU_VALUES replacement would only
      // materialize a phantom source; real DUMMY_SCANs are replaced.
      auto& dummy = slot->Cast<sirius::op::sirius_physical_dummy_scan>();
      if (!dummy.is_delim_join_placeholder()) {
        replace_with_gpu_values(slot,
                                op_params.scan_task_batch_size == 0
                                  ? sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE
                                  : op_params.scan_task_batch_size);
      }
      break;
    }
    case sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY:
      wrap_hash_group_by(slot, op_params, compressed_materialization_observer);
      break;
    case sirius::op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
      wrap_ungrouped_aggregate(slot);
      break;
    case sirius::op::SiriusPhysicalOperatorType::ORDER_BY: wrap_order_by(slot, op_params); break;
    case sirius::op::SiriusPhysicalOperatorType::TOP_N: wrap_top_n(slot); break;
    case sirius::op::SiriusPhysicalOperatorType::HASH_JOIN:
    case sirius::op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN:
      wrap_join(*slot, op_params, compressed_materialization_observer);
      break;
    case sirius::op::SiriusPhysicalOperatorType::DENSE_COUNT_JOIN:
      wrap_dense_count_join(*slot, compressed_materialization_observer);
      break;
    case sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN:
    case sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN:
      wrap_delim_join(slot, op_params, context, compressed_materialization_observer);
      break;
    default: break;
  }
}

/// Whether this plan should carry narrow carriers: the connection's setting says so and the
/// Sirius runtime that holds the plan sidecar is registered on the context.
bool compressed_materialization_active(duckdb::ClientContext& context)
{
  if (!context.registered_state) { return false; }
  auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  return state != nullptr && duckdb::compressed_materialization_enabled(context);
}

}  // namespace

void sirius_physical_plan_generator::reject_nested_column_operation(duckdb::Expression const& expr,
                                                                    std::string_view operation)
{
  // A nested-typed BOUND_REF names the offending column directly.
  if (is_nested_logical_type(expr.return_type) &&
      expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    auto name = expr.GetName();
    if (name.empty()) { name = expr.ToString(); }
    throw std::runtime_error("nested column operation on column '" + name + "' (" +
                             expr.return_type.ToString() + ") is unsupported in " +
                             std::string(operation) +
                             ": Sirius reads and projects nested columns but cannot operate on "
                             "them yet");
  }

  // Recurse first so a column reference wins over a constructed nested value.
  duckdb::ExpressionIterator::EnumerateChildren(expr,
                                                [&operation](duckdb::Expression const& child) {
                                                  reject_nested_column_operation(child, operation);
                                                });

  // Constructed nested values (struct_pack, list constructors, ...) cannot run
  // on the GPU path either.
  if (is_nested_logical_type(expr.return_type)) {
    throw std::runtime_error("nested column operation on '" + expr.ToString() + "' (" +
                             expr.return_type.ToString() + ") is unsupported in " +
                             std::string(operation));
  }
}

void sirius_physical_plan_generator::reject_nested_column_type(duckdb::LogicalType const& type,
                                                               std::string_view column_name,
                                                               std::string_view operation)
{
  if (is_nested_logical_type(type)) {
    throw std::runtime_error("nested column operation on column '" + std::string(column_name) +
                             "' (" + type.ToString() + ") is unsupported in " +
                             std::string(operation) +
                             ": Sirius reads and projects nested columns but cannot operate on "
                             "them yet");
  }
}

sirius_physical_plan_generator::sirius_physical_plan_generator(duckdb::ClientContext& context)
  : context(context)
{
}

sirius_physical_plan_generator::~sirius_physical_plan_generator() {}

void sirius_physical_plan_generator::set_parent_ops(sirius::op::sirius_physical_operator& op,
                                                    sirius::op::sirius_physical_operator* parent)
{
  op.set_parent_op(parent);

  // CTE is transparent on its consumer side: children[0] materializes into the CTE while
  // children[1]'s result IS the CTE's output (CTE::execute just forwards it). Assign the CTE's
  // own parent to children[1] so the tree-parent wiring never emits
  // consumer_sink -> CTE_pipeline edges, which would close a cycle with the CTE sink's own
  // CTE_pipeline -> consumer emissions.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::CTE) {
    D_ASSERT(op.children.size() == 2);
    set_parent_ops(*op.children[0], &op);
    set_parent_ops(*op.children[1], parent);
    return;
  }

  for (auto& child : op.children) {
    if (child) { set_parent_ops(*child, &op); }
  }
  // DELIM JOIN keeps its internal `join`/`distinct_root` subtrees outside `children[]`;
  // descend so the wrapped operators inside get tree parents. PARTITION's ctor `key_source`
  // is captured for key/type derivation only and never stored, so without this descent its
  // `_parent_op` stays null and the wiring can't resolve its destination.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      op.type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = op.Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join) { set_parent_ops(*delim.join, &op); }
    if (delim.distinct_root) { set_parent_ops(*delim.distinct_root, &op); }
  }
  // RESULT_COLLECTOR keeps its tree child in `plan` (a reference, outside `children[]`) —
  // it's the engine-injected root wrapper. Without descending, the wrapped sink's `_parent_op`
  // stays null and the wiring silently leaves the RESULT_COLLECTOR pipeline with no input
  // source — runtime hang.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    auto& rc = op.Cast<sirius::op::sirius_physical_result_collector>();
    set_parent_ops(rc.plan, &op);
  }
}

namespace {

//! Whether a sink's input semantics permit merge fusion.
//!
//! Only sinks with special multi-input wiring are excluded. Everything else -- including
//! total-input structural sinks such as ORDER_BY, TOP_N, and an outer GROUP BY -- is fusable:
//! those sinks already buffer their full input, so folding the merge into their pipeline does
//! not change when they observe complete data, it only removes a task launch and a repository
//! round-trip.
bool terminal_sink_supports_fusion(const sirius::op::sirius_physical_operator& sink)
{
  using T = sirius::op::SiriusPhysicalOperatorType;
  switch (sink.type) {
    // Joins and CTE/delim terminals have multiple inputs or bespoke sink wiring.
    case T::CTE:
    case T::LEFT_DELIM_JOIN:
    case T::RIGHT_DELIM_JOIN:
    case T::DENSE_COUNT_JOIN:
    case T::HASH_JOIN:
    case T::NESTED_LOOP_JOIN: return false;
    // Partition sinks require complete upstream input in a single task.
    case T::PARTITION:
    case T::SORT_PARTITION: return false;
    default: return true;
  }
}

//! Whether the path to the first downstream sink is unary, streaming, and safe to fuse.
bool merge_downstream_is_streaming_dead_end(const sirius::op::sirius_physical_operator& merge)
{
  // Delim-owned merges require sink wiring.
  if (merge.owning_delim_join() != nullptr) { return false; }
  for (auto* cur = merge.get_parent_op(); cur != nullptr; cur = cur->get_parent_op()) {
    if (cur->is_sink()) { return terminal_sink_supports_fusion(*cur); }
    if (cur->is_source() || cur->children.size() != 1) { return false; }
  }
  return false;
}

}  // namespace

void sirius_physical_plan_generator::mark_fusable_merge_pipelines(
  duckdb::ClientContext& context, sirius::op::sirius_physical_operator& op)
{
  // Merge fusion is the engine-owned production policy. Unit tests can expose a guarded setting
  // to exercise the unfused reference path without making that implementation detail a user knob.
  duckdb::Value setting;
  bool fusion_enabled = true;
  if (context.TryGetCurrentSetting("fuse_merge_pipelines", setting) && !setting.IsNull()) {
    fusion_enabled = setting.GetValue<bool>();
  }
  mark_fusable_merge_pipelines(op, fusion_enabled);
}

void sirius_physical_plan_generator::mark_fusable_merge_pipelines(
  sirius::op::sirius_physical_operator& op, bool fusion_enabled)
{
  // Ungrouped aggregate merges use a different partial-result handoff.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::MERGE_GROUP_BY) {
    op.Cast<sirius::op::sirius_physical_grouped_aggregate_merge>().set_fuse_into_parent(
      fusion_enabled && merge_downstream_is_streaming_dead_end(op));
  } else if (op.type == sirius::op::SiriusPhysicalOperatorType::MERGE_TOP_N) {
    op.Cast<sirius::op::sirius_physical_top_n_merge>().set_fuse_into_parent(
      fusion_enabled && merge_downstream_is_streaming_dead_end(op));
  }

  for (auto& child : op.children) {
    if (child) { mark_fusable_merge_pipelines(*child, fusion_enabled); }
  }
  // Visit subtrees stored outside children[] as well.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      op.type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = op.Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join) { mark_fusable_merge_pipelines(*delim.join, fusion_enabled); }
    if (delim.distinct_root) { mark_fusable_merge_pipelines(*delim.distinct_root, fusion_enabled); }
  }
  if (op.type == sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    mark_fusable_merge_pipelines(op.Cast<sirius::op::sirius_physical_result_collector>().plan,
                                 fusion_enabled);
  }
}

void sirius_physical_plan_generator::insert_gpu_pipeline_operators(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan)
{
  // Sink wraps need the sizing params from SiriusContext. If it's missing, default-constructed
  // op_params make the wraps fall back to the operators' own constructor defaults. The same
  // context doubles as the compressed-materialization counter observer for the PARTITION wraps.
  sirius::operator_params op_params;
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  if (sirius_ctx) { op_params = sirius_ctx->get_config().get_operator_params(); }
  insert_gpu_pipeline_operators_recursive(plan, op_params, context, sirius_ctx.get());
}

sirius::OrderPreservationType sirius_physical_plan_generator::order_preservation_recursive(
  sirius::op::sirius_physical_operator& op)
{
  if (op.is_source()) { return op.source_order(); }

  std::size_t child_idx = 0;
  for (auto& child : op.children) {
    // Do not take the materialization phase of physical CTEs into account
    if (op.type == sirius::op::SiriusPhysicalOperatorType::CTE && child_idx == 0) {
      child_idx++;
      continue;
    }
    auto child_preservation = order_preservation_recursive(*child);
    if (child_preservation != sirius::OrderPreservationType::INSERTION_ORDER) {
      return child_preservation;
    }
    child_idx++;
  }
  return sirius::OrderPreservationType::INSERTION_ORDER;
}

bool sirius_physical_plan_generator::preserve_insertion_order(
  duckdb::ClientContext& context, sirius::op::sirius_physical_operator& plan)
{
  auto preservation_type = order_preservation_recursive(plan);
  if (preservation_type == sirius::OrderPreservationType::FIXED_ORDER) {
    // always need to maintain preservation order
    return true;
  }
  if (preservation_type == sirius::OrderPreservationType::NO_ORDER) {
    // never need to preserve order
    return false;
  }
  // preserve insertion order - check flags
  if (!duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(context)) {
    // preserving insertion order is disabled by config
    return false;
  }
  return true;
}

bool sirius_physical_plan_generator::preserve_insertion_order(
  sirius::op::sirius_physical_operator& plan)
{
  return preserve_insertion_order(context, plan);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::unique_ptr<duckdb::LogicalOperator> op)
{
  auto& profiler = duckdb::QueryProfiler::Get(context);

  // Resolve the types of each operator.
  profiler.StartPhase(duckdb::MetricType::PHYSICAL_PLANNER_RESOLVE_TYPES);
  op->ResolveOperatorTypes();
  profiler.EndPhase();

  // Resolve the column references.
  profiler.StartPhase(duckdb::MetricType::PHYSICAL_PLANNER_COLUMN_BINDING);
  duckdb::ColumnBindingResolver resolver;
  resolver.VisitOperator(*op);
  profiler.EndPhase();

  // then create the main physical plan
  profiler.StartPhase(duckdb::MetricType::PHYSICAL_PLANNER_CREATE_PLAN);
  auto plan = create_plan(*op);
  profiler.EndPhase();

  plan = fold_adjacent_projections(std::move(plan));
  if (compressed_materialization_active(context)) {
    auto const retracted = apply_compressed_schema_passes(plan);
    if (retracted > 0) {
      auto sirius_ctx = context.registered_state
                          ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                          : nullptr;
      if (sirius_ctx) {
        sirius_ctx->record_compressed_materialization_scan_narrow_targets_retracted(retracted);
      }
    }
  }
  plan->verify();

  // Rewrite the plan tree to contain the GPU pipeline operators so the converter becomes a
  // pure topology pass over `build_pipelines` virtuals; `set_parent_ops` then derives every
  // `_parent_op` from the final tree for the tree-parent-lookup wiring.
  insert_gpu_pipeline_operators(plan);
  set_parent_ops(*plan, /*parent=*/nullptr);

  return plan;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalOperator& op)
{
  SIRIUS_LOG_DEBUG("Creating sirius physical plan for logical operator type: {}",
                   duckdb::LogicalOperatorToString(op.type));
  op.estimated_cardinality                                      = op.EstimateCardinality(context);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan = nullptr;

  // SQLNULL-typed columns (e.g. an uncast NULL in VALUES) have no cuDF
  // representation — get_cudf_type() / fixed_width_byte_size() reject them at
  // execution time, after the GPU plan is already running. Reject the plan
  // here instead so transparent execution falls back to DuckDB CPU.
  for (const auto& type : op.types) {
    if (duckdb::TypeVisitor::Contains(type, duckdb::LogicalTypeId::SQLNULL)) {
      throw duckdb::NotImplementedException("SQLNULL-typed column not supported");
    }
  }

  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
      plan = create_plan(op.Cast<duckdb::LogicalGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      plan = create_plan(op.Cast<duckdb::LogicalProjection>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT:
      plan = create_plan(op.Cast<duckdb::LogicalEmptyResult>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
      plan = create_plan(op.Cast<duckdb::LogicalFilter>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      plan = create_plan(op.Cast<duckdb::LogicalAggregate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_WINDOW:
      throw duckdb::NotImplementedException("Window not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalWindow>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UNNEST:
      throw duckdb::NotImplementedException("Unnest not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalUnnest>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
      plan = create_plan(op.Cast<duckdb::LogicalLimit>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_SAMPLE:
      throw duckdb::NotImplementedException("Sample not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSample>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
      plan = create_plan(op.Cast<duckdb::LogicalOrder>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
      plan = create_plan(op.Cast<duckdb::LogicalTopN>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE:
      throw duckdb::NotImplementedException("Copy to file not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCopyToFile>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN:
      plan = create_plan(op.Cast<duckdb::LogicalDummyScan>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN:
      throw duckdb::NotImplementedException("Any join not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalAnyJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ASOF_JOIN:
      throw duckdb::NotImplementedException("Asof join not supported");
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      plan = create_plan(op.Cast<duckdb::LogicalComparisonJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
      throw duckdb::NotImplementedException("Cross product not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCrossProduct>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
      throw duckdb::NotImplementedException("Positional join not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPositionalJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UNION:
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      throw duckdb::NotImplementedException("Set operation not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSetOperation>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_INSERT:
      throw duckdb::NotImplementedException("Insert not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalInsert>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELETE:
      throw duckdb::NotImplementedException("Delete not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalDelete>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CHUNK_GET:
      plan = create_plan(op.Cast<duckdb::LogicalColumnDataGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET:
      plan = create_plan(op.Cast<duckdb::LogicalDelimGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPRESSION_GET:
      plan = create_plan(op.Cast<duckdb::LogicalExpressionGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UPDATE:
      throw duckdb::NotImplementedException("Update not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalUpdate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_TABLE:
      throw duckdb::NotImplementedException("Create table not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateTable>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_INDEX:
      throw duckdb::NotImplementedException("Create index not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateIndex>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SECRET:
      throw duckdb::NotImplementedException("Create secret not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateSecret>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPLAIN:
      throw duckdb::NotImplementedException("Explain not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExplain>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
      throw duckdb::NotImplementedException("Distinct not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalDistinct>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PREPARE:
      throw duckdb::NotImplementedException("Prepare not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPrepare>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXECUTE:
      throw duckdb::NotImplementedException("Execute not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExecute>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_VIEW:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_MACRO:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_TYPE:
      throw duckdb::NotImplementedException("Create not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PRAGMA:
      throw duckdb::NotImplementedException("Pragma not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPragma>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_VACUUM:
      throw duckdb::NotImplementedException("Vacuum not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalVacuum>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_TRANSACTION:
    case duckdb::LogicalOperatorType::LOGICAL_ALTER:
    case duckdb::LogicalOperatorType::LOGICAL_DROP:
    case duckdb::LogicalOperatorType::LOGICAL_LOAD:
    case duckdb::LogicalOperatorType::LOGICAL_ATTACH:
    case duckdb::LogicalOperatorType::LOGICAL_DETACH:
      throw duckdb::NotImplementedException("Simple not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSimple>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
      throw duckdb::NotImplementedException("Recursive CTE not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalRecursiveCTE>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
      plan = create_plan(op.Cast<duckdb::LogicalMaterializedCTE>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CTE_REF:
      plan = create_plan(op.Cast<duckdb::LogicalCTERef>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPORT:
      throw duckdb::NotImplementedException("Export not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExport>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_SET:
      throw duckdb::NotImplementedException("Set not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_RESET:
      throw duckdb::NotImplementedException("Reset not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalReset>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PIVOT:
      throw duckdb::NotImplementedException("Pivot not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPivot>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_COPY_DATABASE:
      throw duckdb::NotImplementedException("Copy database not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCopyDatabase>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
      throw duckdb::NotImplementedException("Update extensions not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSimple>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
      throw duckdb::NotImplementedException("Extension operator not supported");
      // plan = op.Cast<duckdb::LogicalExtensionOperator>().create_plan(context, *this);

      // if (!plan) {
      // 	throw duckdb::InternalException("Missing sirius_physical_operator for Extension
      // Operator");
      // }
      break;
    case duckdb::LogicalOperatorType::LOGICAL_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_INVALID: {
      throw duckdb::NotImplementedException("Unimplemented logical operator type!");
    }
    default: throw duckdb::NotImplementedException("Unimplemented logical operator type");
  }
  if (!plan) { throw duckdb::InternalException("Physical plan generator - no plan generated"); }

  plan->estimated_cardinality = op.estimated_cardinality;
#ifdef DUCKDB_VERIFY_VECTOR_OPERATOR
  auto verify = duckdb::make_uniq<duckdb::PhysicalVerifyVector>(std::move(plan));
  plan        = std::move(verify);
#endif

  return plan;
}

}  // namespace sirius::planner
