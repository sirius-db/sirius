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

#include "duckdb/common/type_visitor.hpp"
#include "config.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "log/logging.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius_context.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_cpu_source.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_iceberg_scan.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "sirius_context.hpp"

#include <optional>
#include <utility>

namespace sirius::planner {

namespace {
/// Read the dynamic-filter-pushdown enable flag from the active SiriusContext config. Defaults to
/// disabled when the state is unavailable (no config to consult outside a configured query).
bool dynamic_filter_pushdown_enabled(duckdb::ClientContext& context)
{
  auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!state) { return false; }
  return state->get_config().get_operator_params().enable_dynamic_filter_pushdown;
}

//! Insert `factory(std::move(parent.children[i]))` between `parent` and its i-th child. The
//! factory takes ownership of the original child and returns the wrapper subtree (which
//! must already hold the original as one of its own descendants).
//!
//! Move-semantics on `parent.children[i]` guarantees no raw pointer is held across the
//! mutation: the slot is null between the `std::move` and the assignment, so the compiler
//! enforces tree integrity. See the master plan (`Tree-mutation pattern`) for the rationale.
template <typename WrapperFactory>
void wrap_child(sirius::op::sirius_physical_operator& parent,
                std::size_t i,
                WrapperFactory&& factory)
{
  auto original      = std::move(parent.children[i]);
  auto wrapper       = std::forward<WrapperFactory>(factory)(std::move(original));
  parent.children[i] = std::move(wrapper);
}

//! Replace the operator at `slot` with `factory(std::move(slot))`. Used for sink-wrap rewrites
//! (HASH_GROUP_BY, ORDER_BY, TOP_N, etc.) where the original operator becomes a child of a
//! newly-inserted wrapper that sits in the slot it used to occupy. The factory receives
//! ownership of the original and must return a wrapper subtree containing the original.
template <typename WrapperFactory>
void wrap_above(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                WrapperFactory&& factory)
{
  auto original = std::move(slot);
  slot          = std::forward<WrapperFactory>(factory)(std::move(original));
}

//! Build a `parquet_scan_info` describing a parquet read by lifting fields out of a
//! `sirius_physical_table_scan`. **Destructive**: `scan_op.table_filters` is moved into the
//! info and `scan_op.table_filters` is left null. Mirrors the field plumbing in today's
//! `sirius_pipeline_converter::insert_parquet_scan_operator` byte-for-byte.
std::unique_ptr<sirius::op::scan::parquet_scan_info> build_parquet_scan_info(
  sirius::op::sirius_physical_table_scan& scan_op)
{
  auto const& bind_data = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
  if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
    throw std::runtime_error(
      "[sirius_physical_plan_generator::build_parquet_scan_info] No input files to scan");
  }
  std::vector<std::string> file_paths;
  for (auto const& file : bind_data.file_list->GetAllFiles()) {
    file_paths.push_back(file.path);
  }
  auto const& partition_indices = bind_data.reader_bind.hive_partitioning_indexes;

  auto info               = std::make_unique<sirius::op::scan::parquet_scan_info>();
  info->returned_types    = scan_op.returned_types;
  info->file_paths        = std::move(file_paths);
  info->column_ids        = scan_op.column_ids;
  info->projection_ids    = scan_op.projection_ids;
  info->names             = scan_op.names;
  info->table_filters     = std::move(scan_op.table_filters);
  info->partition_indices = partition_indices;
  return info;
}

//! Attach a leaf GPU scan operator as the only child of a TABLE_SCAN, mirroring today's
//! `sirius_pipeline_converter::split_table_scan_source` and `insert_parquet_scan_operator`
//! (which construct the same operators as separate pipelines at runtime instead of nesting
//! them in the tree). In the tree-based path the leaf lives in the plan tree from plan time
//! so `build_pipelines` can derive the scan pipeline structurally rather than via runtime
//! mutation.
//!
//! For `iceberg_scan`, attaches the pre-fetched `IcebergDeleteData` from
//! `iceberg_delete_data_cache_` so the GPU operator sees the delete-merge set on every read.
//! Path resolution mirrors `resolve_iceberg_table_path` exactly; cache misses leave
//! `delete_data` null, which matches today's converter behavior at
//! `construct_sirius_specific_operator:65-76`.
//!
//! Throws on truly unsupported scan functions to match `construct_sirius_specific_operator`'s
//! behavior at converter line 80.
void wrap_table_scan_source(
  sirius::op::sirius_physical_operator& table_scan_op,
  const std::unordered_map<std::string, std::shared_ptr<const sirius::op::scan::IcebergDeleteData>>&
    iceberg_cache)
{
  // Table-in-out functions wear a TABLE_SCAN with children — skip per the master plan's
  // exclusion rule. Wrapping them would change their child layout in a way the converter
  // and downstream operators don't expect.
  if (!table_scan_op.children.empty()) { return; }

  auto& scan     = table_scan_op.Cast<sirius::op::sirius_physical_table_scan>();
  const auto& fn = scan.function.name;

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> leaf;
  if (fn == "seq_scan") {
    leaf = duckdb::make_uniq<sirius::op::sirius_physical_duckdb_scan>(&scan);
  } else if (fn == "parquet_scan" || fn == "read_parquet") {
    auto info = build_parquet_scan_info(scan);
    leaf      = duckdb::make_uniq<sirius::op::scan::sirius_gpu_parquet_scan_operator>(
      scan.types, scan.estimated_cardinality, std::move(info));
  } else if (fn == "iceberg_scan") {
    auto iceberg_scan = duckdb::make_uniq<sirius::op::sirius_physical_iceberg_scan>(&scan);
    if (!scan.parameters.empty()) {
      auto const table_path = scan.parameters[0].ToString();
      auto it               = iceberg_cache.find(table_path);
      if (it != iceberg_cache.end()) { iceberg_scan->delete_data = it->second; }
    }
    leaf = std::move(iceberg_scan);
  } else {
    throw std::runtime_error(
      "[sirius_physical_plan_generator::wrap_table_scan_source] Unsupported scan function: " + fn);
  }
  table_scan_op.children.push_back(std::move(leaf));
}

//! Attach a leaf CPU_SOURCE operator as the only child of a COLUMN_DATA_SCAN (with non-null
//! collection), EMPTY_RESULT, or DUMMY_SCAN node. Mirrors the legacy converter's
//! `split_cpu_source`. COLUMN_DATA_SCAN with a null collection is the LEFT_DELIM_JOIN cached
//! chunk scan — populated at runtime by the delim-join sink, not by `cpu_source_task` — so
//! we deliberately leave it as-is.
void wrap_cpu_source(sirius::op::sirius_physical_operator& source_op)
{
  if (!source_op.children.empty()) { return; }

  duckdb::unique_ptr<sirius::op::sirius_physical_cpu_source> leaf;
  switch (source_op.type) {
    case sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN: {
      auto& col_scan = source_op.Cast<sirius::op::sirius_physical_column_data_scan>();
      if (!col_scan.collection) { return; }  // LEFT_DELIM_JOIN cached chunk scan — skip
      leaf = duckdb::make_uniq<sirius::op::sirius_physical_cpu_source>(
        source_op.types, source_op.estimated_cardinality, std::move(col_scan.collection));
      break;
    }
    case sirius::op::SiriusPhysicalOperatorType::DUMMY_SCAN:
      leaf = duckdb::make_uniq<sirius::op::sirius_physical_cpu_source>(
        source_op.types, source_op.estimated_cardinality, /*produce_single_row=*/true);
      break;
    case sirius::op::SiriusPhysicalOperatorType::EMPTY_RESULT:
      leaf = duckdb::make_uniq<sirius::op::sirius_physical_cpu_source>(
        source_op.types, source_op.estimated_cardinality, /*produce_single_row=*/false);
      break;
    default: return;
  }
  source_op.children.push_back(std::move(leaf));
}

//! Post-order recursive walk over the physical plan tree. Children are visited (and rewritten)
//! before the dispatch on `slot->type`, so a later `wrap_above` cannot re-enter the freshly-
//! inserted wrapper subtree and double-wrap the original node. Source-side wraps (this sub-
//! phase) append a leaf to an existing TABLE_SCAN/COLUMN_DATA_SCAN/EMPTY_RESULT/DUMMY_SCAN
//! node, growing it from a leaf into an intermediate; the new leaf has no children of its own,
//! so post-order is equivalent to pre-order in those cases.
void insert_gpu_pipeline_operators_recursive(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
  const std::unordered_map<std::string, std::shared_ptr<const sirius::op::scan::IcebergDeleteData>>&
    iceberg_cache)
{
  if (!slot) { return; }

  for (auto& child_slot : slot->children) {
    insert_gpu_pipeline_operators_recursive(child_slot, iceberg_cache);
  }

  switch (slot->type) {
    case sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN:
      wrap_table_scan_source(*slot, iceberg_cache);
      break;
    case sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN:
    case sirius::op::SiriusPhysicalOperatorType::EMPTY_RESULT:
    case sirius::op::SiriusPhysicalOperatorType::DUMMY_SCAN: wrap_cpu_source(*slot); break;
    default:
      // Sink wraps (HASH_GROUP_BY, ORDER_BY, TOP_N, UNGROUPED_AGGREGATE), joins
      // (HASH_JOIN, NESTED_LOOP_JOIN), and DELIM JOIN internals land in subsequent
      // commits within Sub-phase B.
      break;
  }
}

}  // namespace

sirius_physical_plan_generator::sirius_physical_plan_generator(duckdb::ClientContext& context)
  : context(context)
{
}

sirius_physical_plan_generator::~sirius_physical_plan_generator() {}

std::shared_ptr<sirius::op::sirius_dynamic_filter_set>
sirius_physical_plan_generator::get_or_create_dynamic_filter_channel(
  duckdb::DynamicTableFilterSet const* key)
{
  if (!key) { return nullptr; }
  // Central gate: when dynamic-filter pushdown is disabled, return no channel so neither the
  // producer (join) nor the consumer (scan) wires anything.
  if (!dynamic_filter_pushdown_enabled(context)) { return nullptr; }
  auto [it, inserted] = dynamic_filter_channels.try_emplace(key, nullptr);
  if (inserted) { it->second = std::make_shared<sirius::op::sirius_dynamic_filter_set>(); }
  return it->second;
}

void sirius_physical_plan_generator::set_parent_ops(sirius::op::sirius_physical_operator& op,
                                                    sirius::op::sirius_physical_operator* parent)
{
  op.set_parent_op(parent);
  for (auto& child : op.children) {
    if (child) { set_parent_ops(*child, &op); }
  }
}

void sirius_physical_plan_generator::insert_gpu_pipeline_operators(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan)
{
  insert_gpu_pipeline_operators_recursive(plan, iceberg_delete_data_cache_);
}

std::string sirius_physical_plan_generator::resolve_iceberg_table_path(
  sirius::op::sirius_physical_table_scan& scan_op)
{
  if (!scan_op.parameters.empty()) { return scan_op.parameters[0].ToString(); }

  // REST catalog: derive from bind_data file list.
  if (scan_op.bind_data) {
    auto& bind_data = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
    if (bind_data.file_list && !bind_data.file_list->IsEmpty()) {
      auto files = bind_data.file_list->GetAllFiles();
      if (!files.empty()) {
        auto const& first_path = files[0].path;
        // Strip "/data/<filename>" to get table root.
        auto data_pos = first_path.rfind("/data/");
        if (data_pos != std::string::npos) { return first_path.substr(0, data_pos); }
      }
    }
  }
  return {};
}

void sirius_physical_plan_generator::prefetch_iceberg_delete_data(
  sirius::op::sirius_physical_operator& plan)
{
  // Walk the plan tree and fully materialize delete data for every iceberg scan, mirroring
  // `sirius_engine::prefetch_iceberg_delete_data`. The engine's variant still runs for the
  // flag-off (legacy converter) path until Sub-phase E removes it; this variant feeds the
  // tree-based wrap performed by `wrap_table_scan_source` later in `create_plan`.
  if (plan.type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    if (plan.type == sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
      auto& collector = plan.Cast<sirius::op::sirius_physical_result_collector>();
      prefetch_iceberg_delete_data(collector.plan);
    } else {
      for (auto& child : plan.children) {
        prefetch_iceberg_delete_data(*child);
      }
    }
    return;
  }

  auto& scan_op = plan.Cast<sirius::op::sirius_physical_table_scan>();
  if (scan_op.function.name != "iceberg_scan") { return; }

  std::string const table_path = resolve_iceberg_table_path(scan_op);
  if (table_path.empty()) { return; }
  if (iceberg_delete_data_cache_.count(table_path)) { return; }  // already fetched

  // Extract snapshot parameters if present (for snapshot-aware delete discovery).
  std::optional<uint64_t> snapshot_id;
  auto sid_it = scan_op.named_parameters.find("snapshot_from_id");
  if (sid_it != scan_op.named_parameters.end() && !sid_it->second.IsNull()) {
    snapshot_id = sid_it->second.GetValue<uint64_t>();
  }

  // Opening secondary connections triggers QueryBegin/QueryEnd on the shared SiriusContext;
  // InternalQueryGuard suppresses those side-effects.
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    SIRIUS_LOG_WARN(
      "[sirius_physical_plan_generator] SiriusContext not available; treating iceberg '{}' as V1.",
      table_path);
    iceberg_delete_data_cache_.emplace(table_path,
                                       std::make_shared<sirius::op::scan::IcebergDeleteData>());
    return;
  }

  duckdb::SiriusContext::InternalQueryGuard guard(*sirius_ctx);
  auto data = sirius::op::scan::read_iceberg_delete_data(context, table_path, snapshot_id);
  iceberg_delete_data_cache_.emplace(table_path, std::move(data));
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
  plan->verify();

  // Phase 3 (#601) tree-based pipeline build. When the flag is on, rewrite the plan tree to
  // contain GPU pipeline operators (PARTITION/CONCAT/MERGE_*/SORT_*/scan companions/etc.)
  // so the converter becomes a pure topology pass driven by `build_pipelines` virtuals.
  // Default off; the legacy `sirius_pipeline_converter` is authoritative when the flag is
  // off. Iceberg delete data is pre-fetched before the tree rewrite so
  // `wrap_table_scan_source` can attach `delete_data` to each `sirius_physical_iceberg_scan`
  // it constructs. `set_parent_ops` then derives every operator's `_parent_op` from the
  // final tree, enabling Sub-phase C/D's tree-parent-lookup wiring.
  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    prefetch_iceberg_delete_data(*plan);
    insert_gpu_pipeline_operators(plan);
    set_parent_ops(*plan, /*parent=*/nullptr);
  }

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
