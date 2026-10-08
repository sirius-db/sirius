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
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "expression/ast/utils.hpp"
#include "helper/type_conversions.hpp"
#include "io/s3/duckdb_secret_config.hpp"
#include "io/uri_parser.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/iceberg_dv_preparation.hpp"
#include "op/scan/iceberg_gpu_ingestible.hpp"
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
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_passthrough_sink.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "op/sirius_physical_union.hpp"
#include "planner/connector_registry.hpp"
#include "planner/sirius_plan_compressed_schema.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "scan_manager/preparation_ledger.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "transparent/read_view_registry.hpp"

#include <cudf/cudf_utils.hpp>

#include <duckdb/common/serializer/binary_serializer.hpp>
#include <duckdb/common/serializer/memory_stream.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <thread>
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
  if (function_name == "parquet_scan" || function_name == "read_parquet" ||
      function_name == "iceberg_scan") {
    // iceberg_scan belongs here: the iceberg extension resolves its manifests into the same
    // MultiFileBindData file list read_parquet produces, which is what lets the parquet
    // ingestible read an iceberg table's data files unchanged.
    //
    // dynamic_cast (never Cast<>, which asserts/throws): an unresolvable
    // identity must degrade to empty, not fail the caller.
    auto const* multi_file_bind = dynamic_cast<duckdb::MultiFileBindData const*>(bind_data);
    if (multi_file_bind == nullptr || !multi_file_bind->file_list ||
        multi_file_bind->file_list->IsEmpty()) {
      return {};
    }
    auto files = multi_file_bind->file_list->GetAllFiles();
    std::vector<std::string> file_paths;
    file_paths.reserve(files.size());
    // The bulk snapshot owns these strings; transfer them without copying each path again.
    // The bound file list remains untouched for independent original/candidate captures.
    for (auto& file : files) {
      file_paths.push_back(std::move(file.path));
    }
    return file_paths;
  }
  return {};
}

namespace {
std::atomic<uint64_t> next_standalone_plan_generation{1};

uint64_t reserve_standalone_plan_generation()
{
  auto const generation = next_standalone_plan_generation.fetch_add(1, std::memory_order_relaxed);
  if (generation == 0) { throw std::overflow_error("standalone plan generation space exhausted"); }
  return generation;
}

bool is_nested_logical_type(duckdb::LogicalType const& type)
{
  auto const id = type.id();
  return id == duckdb::LogicalTypeId::STRUCT || id == duckdb::LogicalTypeId::LIST ||
         id == duckdb::LogicalTypeId::MAP;
}

void install_parquet_s3_configs(sirius::op::scan::parquet_ingestible_table_info const& table_info,
                                duckdb::ClientContext& context,
                                duckdb::SiriusContext* sirius_ctx)
{
  if (sirius_ctx == nullptr) { return; }
  auto const& defaults = sirius_ctx->get_config().get_scan_manager_config().object_store;
  for (auto const& path : table_info.resolved_file_paths) {
    if (!sirius::io::s3::is_s3_path(path)) { continue; }
    auto config = sirius::io::s3::resolve_duckdb_s3_secret(context, path, defaults);
    sirius_ctx->get_scan_manager().install_s3_config(path, std::move(config));
  }
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
//! Fill the parquet bind data from a TABLE_SCAN. Split out from
//! `build_parquet_table_info` so the iceberg path can populate the same fields into its own
//! subclass — an iceberg table's data files are parquet, so every field here applies unchanged.
void populate_parquet_schema(sirius::op::scan::parquet_ingestible_table_info& out,
                             sirius::op::sirius_physical_table_scan const& scan_op)
{
  auto* info           = &out;
  info->returned_types = scan_op.returned_types;
  info->column_ids     = scan_op.column_ids;
  info->projection_ids = scan_op.projection_ids;
  info->names          = scan_op.names;
  info->virtual_columns.reserve(scan_op.virtual_columns.size());
  for (auto const& [column_id, column] : scan_op.virtual_columns) {
    std::optional<sirius::op::scan::scan_plan::parquet_virtual_column_kind> kind;
    if (column_id == duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME) {
      kind = sirius::op::scan::scan_plan::parquet_virtual_column_kind::FILENAME;
    } else if (column_id == duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX) {
      kind = sirius::op::scan::scan_plan::parquet_virtual_column_kind::FILE_INDEX;
    } else if (column_id == duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER) {
      kind = sirius::op::scan::scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER;
    }
    info->virtual_columns.push_back(sirius::op::scan::bound_virtual_column{
      column_id, column.name, sirius::from_duckdb(column.type), kind});
  }
  if (scan_op.function.name != "sirius_read_parquet" &&
      dynamic_cast<duckdb::MultiFileBindData const*>(scan_op.bind_data.get())) {
    auto const& bind_data   = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
    info->partition_indices = bind_data.reader_bind.hive_partitioning_indexes;
    // Legacy options use ordinary schema positions; normalize them here.
    auto add_legacy_virtual = [&](duckdb::idx_t primary_idx,
                                  sirius::op::scan::scan_plan::parquet_virtual_column_kind kind) {
      if (primary_idx >= scan_op.names.size() || primary_idx >= scan_op.returned_types.size()) {
        return;
      }
      auto const id     = static_cast<duckdb::column_t>(primary_idx);
      auto const exists = std::any_of(info->virtual_columns.begin(),
                                      info->virtual_columns.end(),
                                      [id](auto const& column) { return column.column_id == id; });
      if (!exists) {
        info->virtual_columns.push_back(sirius::op::scan::bound_virtual_column{
          id, scan_op.names[primary_idx], scan_op.returned_types[primary_idx], kind});
      }
    };
    if (bind_data.reader_bind.filename_idx.IsValid()) {
      add_legacy_virtual(bind_data.reader_bind.filename_idx.GetIndex(),
                         sirius::op::scan::scan_plan::parquet_virtual_column_kind::FILENAME);
    }
    // Read the option from the bind callback. Inferring it from initial_reader is ambiguous with a
    // physical column carrying Iceberg's ordinal field id, and initial_reader is absent for
    // explicit schema= binds.
    bool legacy_file_row_number = false;
    if (scan_op.function.get_bind_info) {
      auto bind_info         = scan_op.function.get_bind_info(scan_op.bind_data.get());
      auto const option      = bind_info.options.find("file_row_number");
      legacy_file_row_number = option != bind_info.options.end() && option->second.GetValue<bool>();
    }
    if (legacy_file_row_number) {
      auto const output = std::find(scan_op.names.begin(), scan_op.names.end(), "file_row_number");
      if (output != scan_op.names.end()) {
        add_legacy_virtual(
          static_cast<duckdb::idx_t>(output - scan_op.names.begin()),
          sirius::op::scan::scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER);
      }
    }
  }
}

void populate_parquet_table_info(sirius::op::scan::parquet_ingestible_table_info& out,
                                 sirius::op::sirius_physical_table_scan& scan_op,
                                 const sirius::operator_params& op_params)
{
  populate_parquet_schema(out, scan_op);
  auto* info          = &out;
  info->table_filters = std::move(scan_op.table_filters);
  auto resolved_file_paths =
    scan_op.contract_file_paths.empty()
      ? resolve_parquet_scan_file_paths(
          scan_op.function.name, scan_op.bind_data.get(), scan_op.parameters)
      : std::move(scan_op.contract_file_paths);
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
  }

  // `scan_output_arity` drives the provider's expected column count — without it the runtime
  // task skips the hive-partition columns it should inject post-read, mis-sizing the output.
  info->scan_output_arity      = scan_op.types.size();
  info->approximate_batch_size = op_params.scan_task_batch_size;
  info->read_views             = scan_op.read_views;
}

std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> build_parquet_table_info(
  sirius::op::sirius_physical_table_scan& scan_op, const sirius::operator_params& op_params)
{
  auto info = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  populate_parquet_table_info(*info, scan_op, op_params);
  return info;
}

void collect_iceberg_schema(std::vector<duckdb::MultiFileColumnDefinition> const& columns,
                            sirius::op::scan::iceberg_table_schema& schema)
{
  for (auto const& column : columns) {
    if (!column.identifier.IsNull() &&
        column.identifier.type().id() == duckdb::LogicalTypeId::INTEGER) {
      schema.fields.push_back({column.name,
                               column.identifier.GetValue<int32_t>(),
                               column.children.empty() ? column.type.ToString() : std::string{}});
    }
    collect_iceberg_schema(column.children, schema);
  }
}

//! Build an `iceberg_ingestible_table_info`: the parquet bind data plus the table's delete
//! data, resolved here at plan time.
//!
//! Reading the deletes can fail — an unreadable manifest, a delete file that is not where the
//! metadata says. Every such failure throws out of here, which the caller turns into a CPU
//! fallback. It must never be softened into "no deletes": that is indistinguishable from an
//! append-only table, and would return rows the table logically deleted.
std::unique_ptr<sirius::op::scan::iceberg_ingestible_table_info> build_iceberg_table_info(
  sirius::op::sirius_physical_table_scan& scan_op,
  const sirius::operator_params& op_params,
  duckdb::ClientContext& context,
  sirius::op::scan::iceberg_delete_discovery* supplied_discovery = nullptr)
{
  if (scan_op.prepared_iceberg_info) return std::move(scan_op.prepared_iceberg_info);
  auto info = std::make_unique<sirius::op::scan::iceberg_ingestible_table_info>();
  populate_parquet_table_info(*info, scan_op, op_params);
  if (iceberg_table_schema_has_field_ids(scan_op.bind_data.get())) {
    auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(scan_op.bind_data.get());
    D_ASSERT(bind != nullptr);
    sirius::op::scan::iceberg_table_schema schema;
    collect_iceberg_schema(
      bind->reader_bind.schema.empty() ? bind->columns : bind->reader_bind.schema, schema);
    if (!schema.fields.empty() && !info->file_paths().empty())
      info->physical_schema = std::move(schema);
  }

  if (scan_op.parameters.empty() || scan_op.parameters.front().IsNull()) {
    throw duckdb::NotImplementedException("iceberg_scan has no table path parameter");
  }
  info->table_path = scan_op.parameters.front().GetValue<std::string>();

  // Second line of the gate in sirius_plan_get.cpp, which already declines an unpinned scan: a
  // caller reaching here without an id must fail loudly rather than read "current" a third time.
  // Deriving one here is not a fallback -- see that gate for why every derivation is unsound.
  auto const sid_it = scan_op.named_parameters.find("snapshot_from_id");
  if (sid_it == scan_op.named_parameters.end() || sid_it->second.IsNull()) {
    throw duckdb::NotImplementedException(
      "iceberg_scan reached GPU physical planning without 'snapshot_from_id': the snapshot its "
      "delete files must be read from is unknown, and reading them from whatever is current now "
      "risks pairing one snapshot's data files with another's deletes");
  }
  std::optional<uint64_t> const snapshot_id =
    static_cast<uint64_t>(sid_it->second.GetValue<int64_t>());

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw duckdb::NotImplementedException(
      "iceberg delete data cannot be read without a registered SiriusContext");
  }

  // Delete discovery opens its own Connection to run iceberg_metadata() and to read the
  // positional-delete parquet files, which re-registers this same SiriusContext. Bracket the
  // planning connection as an internal query so its lifecycle callbacks stay out of the way of
  // the query being planned. Same guard the delete gate uses.
  duckdb::SiriusContext::InternalQueryGuard guard(context);
  auto const delete_started = std::chrono::steady_clock::now();
  if (!scan_op.delete_inventory) {
    throw duckdb::InternalException("iceberg_scan reached lowering without its delete inventory");
  }
  auto inventory = std::move(*scan_op.delete_inventory);
  scan_op.delete_inventory.reset();
  auto discovery =
    sirius::op::scan::discover_from_manifests(context, info->table_path, std::move(inventory));
  if (supplied_discovery)
    *supplied_discovery = std::move(discovery);
  else
    info->delete_data = sirius::op::scan::load_delete_payload(
      context, info->table_path, sirius_ctx->get_scan_manager().io_ctx(), snapshot_id, discovery);
  auto const delete_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - delete_started)
                                .count();
  scan_op.read_views->record_delete_preparation(scan_op.contract_id, delete_elapsed);
  sirius_ctx->record_delete_preparation(delete_elapsed);

  return info;
}

void prepare_iceberg_statement(sirius::op::sirius_physical_operator& root,
                               sirius::operator_params const& params,
                               duckdb::ClientContext& context)
{
  using namespace sirius::op::scan;
  using namespace sirius::scan_manager;
  struct candidate {
    sirius::op::sirius_physical_table_scan* scan;
    std::unique_ptr<iceberg_ingestible_table_info> info;
    std::shared_ptr<iceberg_delete_discovery> discovery =
      std::make_shared<iceberg_delete_discovery>();
  };
  std::vector<candidate> scans;
  std::function<void(sirius::op::sirius_physical_operator&)> visit = [&](auto& node) {
    using type = sirius::op::SiriusPhysicalOperatorType;
    for (auto& child : node.children)
      if (child) visit(*child);
    if (node.type == type::LEFT_DELIM_JOIN || node.type == type::RIGHT_DELIM_JOIN) {
      auto& d = node.template Cast<sirius::op::sirius_physical_delim_join>();
      if (d.join) visit(*d.join);
      if (d.distinct_root) visit(*d.distinct_root);
    }
    if (node.type != type::TABLE_SCAN) return;
    auto& scan = node.template Cast<sirius::op::sirius_physical_table_scan>();
    if (scan.function.name != "iceberg_scan") return;
    candidate c{&scan, {}};
    c.info = build_iceberg_table_info(scan, params, context, c.discovery.get());
    scans.push_back(std::move(c));
  };
  visit(root);
  if (scans.empty()) return;
  std::vector<scan_dv_count> counts;
  std::vector<scan_envelope> envelopes;
  bool qualified = true;
  for (auto const& c : scans) {
    if (c.discovery->positional_delete_files.empty()) validate_deletion_vector_claims(*c.discovery);
    uint64_t total = 0;
    for (auto const& dv : c.discovery->deletion_vector_entries) {
      if (dv.record_count < 0 ||
          static_cast<uint64_t>(dv.record_count) > std::numeric_limits<uint64_t>::max() - total) {
        qualified = false;
        break;
      }
      total += dv.record_count;
    }
    counts.push_back({c.scan->contract_id, total});
    if (!c.discovery->positional_delete_files.empty() ||
        !c.discovery->equality_delete_entries.empty())
      continue;
    if (c.discovery->deletion_vector_entries.empty())
      continue;  // append-only requires no DV capacity
    try {
      auto e = iceberg_dv_preparation::envelope(*c.discovery, c.info->resolved_file_paths);
      qualified &= e.qualified;
      envelopes.push_back(std::move(e));
    } catch (std::overflow_error const&) {
      qualified = false;
    }
  }
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  host_reservation_provider provider(sirius_ctx->get_memory_manager(),
                                     std::make_shared<decltype(sirius_ctx)>(sirius_ctx));
  auto counters  = sirius_ctx->physical_counters();
  auto* selected = counters->track_units && counters->preparation_provider_for_testing
                     ? counters->preparation_provider_for_testing.get()
                     : &provider;
  auto ledger    = std::make_unique<preparation_ledger>(*selected);
  admission_decision decision;
  bool route_allowed =
    counters->track_units && counters->statement_dv_limit_for_testing
      ? statement_dv_route_allowed(counts, *counters->statement_dv_limit_for_testing)
      : statement_dv_route_allowed(counts);
  if (qualified && route_allowed && !envelopes.empty()) {
    auto spaces =
      sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (!spaces.empty()) decision = ledger->admit(envelopes, spaces.front()->get_id());
  }
  std::string_view route_reason = !qualified           ? "unqualified_envelope"
                                  : !route_allowed     ? "statement_dv_limit"
                                  : envelopes.empty()  ? "no_deferred_dv_work"
                                  : !decision.deferred ? "host_admission"
                                                       : "deferred";
  std::shared_ptr<preparation_admission> admission;
  if (decision.deferred) {
    // Construct all arenas before consuming the plan's one owning admission.
    try {
      for (auto& c : scans) {
        if (!c.discovery->positional_delete_files.empty() ||
            !c.discovery->equality_delete_entries.empty() ||
            c.discovery->deletion_vector_entries.empty())
          continue;
        c.info->deferred = std::make_shared<iceberg_dv_preparation>(c.scan->contract_id,
                                                                    c.info->table_path,
                                                                    c.info->resolved_file_paths,
                                                                    c.discovery,
                                                                    *ledger);
      }
      admission = std::make_shared<preparation_admission>(std::move(ledger), decision);
    } catch (preparation_resource_error const&) {
      route_reason = "arena_allocation";
      for (auto& c : scans)
        c.info->deferred.reset();
      ledger.reset();
    } catch (std::bad_alloc const&) {
      route_reason = "arena_allocation";
      for (auto& c : scans)
        c.info->deferred.reset();
      ledger.reset();
    }
  }
  if (counters->track_units && counters->iceberg_statement_route_for_testing)
    counters->iceberg_statement_route_for_testing(route_reason);
  for (auto const& c : scans) {
    if (!c.info->deferred)
      counters->preparation_legacy_route.fetch_add(1, std::memory_order_relaxed);
    if (counters->track_units && counters->iceberg_preparation_route_for_testing)
      counters->iceberg_preparation_route_for_testing(c.scan->contract_id, bool(c.info->deferred));
  }
  for (auto& c : scans) {
    if (c.info->deferred) {
      c.info->deferred->admission = admission;
    } else {
      duckdb::SiriusContext::InternalQueryGuard guard(context);
      auto started = std::chrono::steady_clock::now();
      auto sid =
        static_cast<uint64_t>(c.scan->named_parameters.at("snapshot_from_id").GetValue<int64_t>());
      c.info->delete_data = load_delete_payload(
        context, c.info->table_path, sirius_ctx->get_scan_manager().io_ctx(), sid, *c.discovery);
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - started)
                  .count();
      c.scan->read_views->record_delete_preparation(c.scan->contract_id, us);
      sirius_ctx->record_delete_preparation(us);
    }
    c.scan->prepared_iceberg_info = std::move(c.info);
  }
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
  // Derive the cache identity from the resolved entry on both pin and scan paths.
  info->catalog_name           = table.ParentCatalog().GetName();
  info->schema_name            = table.ParentSchema().name;
  info->table_name             = table.name;
  info->table_identity         = {table.oid, table.GetStorage().GetRowGroupCollection()};
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
  info->contract_id            = scan.contract_id;
  info->profiles               = scan.read_views->profiles;
  info->injections             = scan.read_views->injections;
  if constexpr (std::is_base_of_v<op::scan::parquet_ingestible_table_info, InfoT>) {
    info->physical_contract          = scan.read_views->entry(scan.contract_id).contract;
    info->physical_contract.profiles = info->profiles;
    info->physical_contract.host_export_available = scan.host_export_available;
    info->bound_types      = info->physical_contract.view->identity->bound_types;
    info->semantic_columns = scan.read_views->entry(scan.contract_id).eligibility.semantic_columns;
  }

  auto ingestible = sirius::op::scan::make_ingestible(std::move(info));
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> leaf =
    duckdb::make_uniq<sirius::op::scan::sirius_gpu_scan_operator>(
      scan.types,
      scan.estimated_cardinality,
      std::move(ingestible),
      scan.contract_id,
      compressed_materialization_observer,
      scan.read_views);
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

// A leaf reference uses reader input identity, never the post-projection output index.
struct semantic_leaf {
  op::sirius_physical_table_scan* scan;
  std::size_t input;
};
using column_lineage = std::vector<semantic_leaf>;
struct plan_lineage {
  std::vector<column_lineage> columns;
  column_lineage inputs;
};
void mark_semantic(column_lineage const& leaves)
{
  for (auto const& leaf : leaves)
    leaf.scan->semantic_columns.at(leaf.input) = true;
}
void mark_semantic_columns(std::vector<column_lineage> const& columns)
{
  for (auto const& column : columns)
    mark_semantic(column);
}
column_lineage lineage_at(plan_lineage const& input, std::size_t index)
{
  if (index < input.columns.size()) return input.columns[index];
  mark_semantic(input.inputs);
  return input.inputs;
}
void mark_expression(ast::node const* expression, plan_lineage const& input)
{
  if (!expression) {
    mark_semantic(input.inputs);
    return;
  }
  ast::visit_references(*expression, [&](ast::reference const& ref) {
    mark_semantic(lineage_at(input, ref.column_index));
  });
}
void mark_expression(duckdb::Expression const& expression, plan_lineage const& input)
{
  if (expression.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    mark_semantic(lineage_at(input, expression.Cast<duckdb::BoundReferenceExpression>().index));
  } else {
    duckdb::ExpressionIterator::EnumerateChildren(
      expression, [&](duckdb::Expression const& child) { mark_expression(child, input); });
  }
}

template <typename Indices>
void append_lineage(plan_lineage& output, plan_lineage const& input, Indices const& indices)
{
  for (auto index : indices)
    output.columns.push_back(lineage_at(input, index));
}

// One bottom-up walk for the whole plan. Computed values have already marked their
// inputs; only pass-through values need to carry leaf identities to their consumers.
plan_lineage collect_semantic_lineage(
  op::sirius_physical_operator& node,
  bool force_all,
  std::unordered_map<op::sirius_physical_operator*, plan_lineage>& aliases)
{
  if (auto it = aliases.find(&node); it != aliases.end()) return it->second;
  using type = op::SiriusPhysicalOperatorType;
  if (node.type == type::TABLE_SCAN && node.children.empty()) {
    auto& scan = node.Cast<op::sirius_physical_table_scan>();
    plan_lineage result;
    result.columns.resize(scan.types.size());
    if (scan.function.name != "seq_scan" && scan.children.empty() && !scan.pre_declined) {
      op::scan::parquet_ingestible_table_info info;
      populate_parquet_schema(info, scan);
      auto layout = op::scan::build_scan_plan(info.column_ids,
                                              info.projection_ids,
                                              info.names,
                                              info.returned_types,
                                              scan.types.size(),
                                              info.partition_indices,
                                              info.virtual_columns);
      scan.semantic_columns.assign(layout.data_columns.size(), false);
      for (std::size_t d = 0; d < layout.data_columns.size(); ++d)
        result.inputs.push_back({&scan, d});
      for (std::size_t i = 0; i < layout.output_layout.size() && i < result.columns.size(); ++i) {
        auto const& output = layout.output_layout[i];
        if (output.source == op::scan::scan_plan::output_entry::DATA &&
            output.idx < result.inputs.size())
          result.columns[i].push_back(result.inputs[output.idx]);
      }
      if (scan.table_filters) {
        for (auto const& [c, filter] : scan.table_filters->filters) {
          if (c >= layout.batch_position_by_column_id.size()) {
            mark_semantic(result.inputs);
            continue;
          }
          auto d = layout.batch_position_by_column_id[c];
          if (d && *d < result.inputs.size()) mark_semantic({result.inputs[*d]});
        }
      }
      for (auto d : layout.pure_filter_batch_positions())
        if (!layout.carrier_batch_index || d != *layout.carrier_batch_index)
          mark_semantic({result.inputs.at(d)});
    } else {
      scan.semantic_columns.assign(scan.column_ids.size(), false);
      for (std::size_t c = 0; c < scan.column_ids.size(); ++c)
        result.inputs.push_back({&scan, c});
      for (std::size_t i = 0; i < result.columns.size(); ++i) {
        auto c = scan.projection_ids.empty() ? i : scan.projection_ids.at(i);
        if (c < result.inputs.size())
          result.columns[i].push_back(result.inputs[c]);
        else
          mark_semantic(result.inputs);
      }
      if (scan.table_filters)
        for (auto const& [c, filter] : scan.table_filters->filters) {
          if (c < result.inputs.size())
            mark_semantic({result.inputs[c]});
          else
            mark_semantic(result.inputs);
        }
    }
    if (force_all) mark_semantic(result.inputs);
    return result;
  }
  std::vector<plan_lineage> children;
  plan_lineage result;
  for (auto& child : node.children) {
    children.push_back(collect_semantic_lineage(*child, force_all, aliases));
    auto const& inputs = children.back().inputs;
    result.inputs.insert(result.inputs.end(), inputs.begin(), inputs.end());
  }
  auto all = [&] {
    mark_semantic(result.inputs);
    // All inputs are already semantic; no need to duplicate that set for every output.
    result.columns.assign(node.types.size(), {});
  };
  if (node.type == type::LEFT_DELIM_JOIN || node.type == type::RIGHT_DELIM_JOIN) {
    auto& delim = node.Cast<op::sirius_physical_delim_join>();
    if (!children.empty() && delim.join && delim.join->children.size() == 2) {
      auto side                                 = node.type == type::LEFT_DELIM_JOIN ? 0 : 1;
      aliases[delim.join->children[side].get()] = children[0];
      if (delim.distinct)
        for (auto index : delim.distinct->group_idx)
          if (index >= 0) mark_semantic(lineage_at(children[0], index));
      auto joined = collect_semantic_lineage(*delim.join, force_all, aliases);
      result      = std::move(joined);
    } else
      all();
  } else if (node.type == type::PROJECTION && children.size() == 1) {
    for (auto const& expression : node.Cast<op::sirius_physical_projection>().select_list) {
      if (expression && expression->holds<ast::reference>())
        result.columns.push_back(
          lineage_at(children[0], expression->get<ast::reference>().column_index));
      else {
        mark_expression(expression.get(), children[0]);
        result.columns.emplace_back();
      }
    }
  } else if (node.type == type::FILTER && children.size() == 1) {
    auto& filter = node.Cast<op::sirius_physical_filter>();
    mark_expression(filter.expression.get(), children[0]);
    if (auto* indices = std::get_if<std::vector<cudf::size_type>>(&filter.output_columns))
      append_lineage(result, children[0], *indices);
    else
      result.columns = children[0].columns;
  } else if (node.type == type::HASH_GROUP_BY && children.size() == 1) {
    auto& aggregate = node.Cast<op::sirius_physical_grouped_aggregate>();
    for (auto index : aggregate.group_idx)
      if (index >= 0) mark_semantic(lineage_at(children[0], index));
    for (auto index : aggregate.cudf_aggregate_idx)
      if (index >= 0) mark_semantic(lineage_at(children[0], index));
    result.columns.resize(node.types.size());
  } else if (node.type == type::UNGROUPED_AGGREGATE && children.size() == 1) {
    for (auto const& expression : node.Cast<op::sirius_physical_ungrouped_aggregate>().aggregates)
      mark_expression(expression.get(), children[0]);
    result.columns.resize(node.types.size());
  } else if ((node.type == type::HASH_JOIN || node.type == type::NESTED_LOOP_JOIN) &&
             children.size() == 2) {
    duckdb::JoinType kind;
    auto join = [&](auto const& physical, auto const& left_indices, auto const& right_indices) {
      kind = physical.join_type;
      for (auto const& condition : physical.conditions) {
        mark_expression(condition.left.get(), children[0]);
        mark_expression(condition.right.get(), children[1]);
      }
      if (kind != duckdb::JoinType::RIGHT_SEMI && kind != duckdb::JoinType::RIGHT_ANTI)
        append_lineage(result, children[0], left_indices);
      if (kind == duckdb::JoinType::MARK)
        result.columns.emplace_back();
      else if (kind != duckdb::JoinType::SEMI && kind != duckdb::JoinType::ANTI)
        append_lineage(result, children[1], right_indices);
    };
    if (node.type == type::HASH_JOIN) {
      auto& physical = node.Cast<op::sirius_physical_hash_join>();
      join(physical, physical.lhs_output_columns.col_idxs, physical.rhs_output_columns.col_idxs);
    } else {
      auto& physical = node.Cast<op::sirius_physical_nested_loop_join>();
      auto left = physical.left_output_col_idxs, right = physical.right_output_col_idxs;
      if (left.empty()) {
        left.resize(children[0].columns.size());
        std::iota(left.begin(), left.end(), 0);
      }
      if (right.empty()) {
        right.resize(children[1].columns.size());
        std::iota(right.begin(), right.end(), 0);
      }
      join(physical, left, right);
    }
    if (result.columns.size() != node.types.size()) all();
    // Join payloads are concatenated in cuDF before the host exporter can cast them.
    mark_semantic_columns(result.columns);
  } else if ((node.type == type::ORDER_BY || node.type == type::TOP_N) && children.size() == 1) {
    if (node.type == type::ORDER_BY) {
      auto& order = node.Cast<op::sirius_physical_order>();
      for (auto const& key : order.orders)
        mark_expression(*key.expression, children[0]);
      if (order.projections.empty())
        result.columns = children[0].columns;
      else
        append_lineage(result, children[0], order.projections);
    } else {
      for (auto const& key : node.Cast<op::sirius_physical_top_n>().orders)
        mark_expression(*key.expression, children[0]);
      result.columns = children[0].columns;
    }
    // Sort and TOP_N carry their payload through cuDF before host export.
    mark_semantic_columns(result.columns);
  } else if ((node.type == type::LIMIT || node.type == type::STREAMING_LIMIT ||
              node.type == type::RESULT_COLLECTOR) &&
             children.size() == 1) {
    result.columns = children[0].columns;
  } else if (node.type == type::UNION && children.size() >= 2) {
    result.columns.resize(node.types.size());
    for (auto const& child : children) {
      if (child.columns.size() != result.columns.size()) {
        all();
        break;
      }
      for (std::size_t i = 0; i < result.columns.size(); ++i)
        result.columns[i].insert(
          result.columns[i].end(), child.columns[i].begin(), child.columns[i].end());
    }
  } else
    all();
  if (force_all) mark_semantic(result.inputs);
  return result;
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

void finish_scan_certification(op::sirius_physical_table_scan& scan,
                               op::scan::certification_result result,
                               scan_contract_provenance& provenance,
                               uint64_t elapsed_us)
{
  using namespace op::scan;
  result.semantic_columns                = std::move(scan.semantic_columns);
  result.cost.delete_preparation_time_us = scan.delete_preparation_time_us;
  result.cost.added_time_us += elapsed_us + std::exchange(provenance.lineage_time_us, 0);
  result.cost.added_bytes = sizeof(eligibility_certificate) + result.reason_text.capacity() +
                            (result.semantic_columns.capacity() + 7) / 8;
  if (scan.bound_view) {
    auto const& metrics                 = scan.bound_view->metrics;
    result.cost.inherited_capture_bytes = metrics.canonical_capacity + metrics.evidence_capacity;
    result.cost.borrowed_files          = metrics.file_count;
  }
  // Injection is a charged amount, not a sleep; wall time outside certify is never charged.
  if (!scan.pre_declined && !provenance.budget.declines()) {
    result.cost.added_time_us += provenance.injections.certification_delay_ms * 1000;
    result.cost.added_bytes += provenance.injections.certification_bytes;
  }
  provenance.budget.charge(std::chrono::microseconds{result.cost.added_time_us},
                           result.cost.added_bytes);
  if (!scan.pre_declined && result.verdict != eligibility_verdict::unsupported &&
      provenance.budget.declines()) {
    result.verdict     = eligibility_verdict::incomplete;
    result.reason      = provenance.budget.time_exceeded() ? verdict_reason::budget_time
                                                           : verdict_reason::budget_bytes;
    result.reason_text = "GPU scan certification test budget exceeded";
  }
  scan.read_views->record_verdict(scan.contract_id, result);
}

//! Rewrite a TABLE_SCAN for `seq_scan` / `parquet_scan` / `read_parquet` /
//! `sirius_read_parquet` (the internal S3 rewrite target): REPLACE the slot with the GPU
//! leaf so it inherits the TABLE_SCAN's tree position and stays the source-leaf of the
//! existing pipeline. Rejects unsupported scan functions and output types without a native cuDF
//! carrier while plan construction can still trigger transparent CPU fallback.
void certify_table_scan(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& table_scan_slot,
                        const sirius::operator_params& op_params,
                        duckdb::ClientContext& context,
                        scan_contract_provenance& contract_provenance)
{
  // Table-in-out functions wear a TABLE_SCAN with children — skip them; wrapping would change
  // a child layout the converter and downstream operators don't expect.
  if (!table_scan_slot->children.empty()) { return; }

  auto& scan        = table_scan_slot->Cast<sirius::op::sirius_physical_table_scan>();
  const auto& fn    = scan.function.name;
  auto const* entry = lookup_connector(scan.function, scan.bind_data.get(), context);
  if (!entry || !entry->lower) {
    throw duckdb::NotImplementedException(
      "Table function '%s' is not supported in Sirius (unverified scan source)", fn);
  }
  if (!scan.read_views) {
    throw std::runtime_error("GPU scan has no query-local read-view registry");
  }

  using namespace sirius::op::scan;
  ++contract_provenance.certification_scan_index;
  if (scan.pre_declined) {
    scan.contract_id = scan.read_views->allocate_declined_scan(scan, *entry);
    certification_result result;
    result.verdict     = scan.pre_declined->verdict;
    result.reason      = scan.pre_declined->reason;
    result.reason_text = scan.pre_declined->text;
    finish_scan_certification(scan, std::move(result), contract_provenance, 0);
    return;
  }

  scan.contract_file_paths =
    resolve_parquet_scan_file_paths(scan.function.name, scan.bind_data.get(), scan.parameters);
  scan.bound_view = std::make_shared<sirius::op::scan::bound_read_view const>(
    sirius::op::scan::capture_bound_read_view(scan, context, scan.contract_file_paths));

  sirius::op::scan::column_requirements columns;
  columns.column_ids     = scan.column_ids;
  columns.projection_ids = scan.projection_ids;
  for (auto const& column : scan.column_ids) {
    if (column.IsRowIdColumn()) columns.requires_row_id = true;
    if (column.IsVirtualColumn() && column.HasPrimaryIndex()) {
      columns.virtual_columns.push_back(column.GetPrimaryIndex());
    }
  }
  sirius::op::scan::predicate_contract predicates;
  predicates.pushdown_mode = std::to_string(static_cast<uint8_t>(entry->filter_mode));
  if (scan.table_filters) {
    duckdb::MemoryStream buffer;
    duckdb::BinarySerializer::Serialize(*scan.table_filters, buffer);
    predicates.static_filter_fingerprint.assign(reinterpret_cast<char const*>(buffer.GetData()),
                                                buffer.GetPosition());
  }

  auto materializer_kind = sirius::op::scan::materializer_kind::parquet;
  if (entry->kind == sirius::op::scan::source_kind::duckdb_native) {
    materializer_kind = sirius::op::scan::materializer_kind::duckdb_native;
  } else if (entry->function_name == "iceberg_scan") {
    materializer_kind = sirius::op::scan::materializer_kind::iceberg;
  }
  scan.contract_id =
    sirius::op::scan::allocate_scan_contract(*scan.read_views,
                                             contract_provenance.window_id,
                                             contract_provenance.finalize_generation,
                                             scan.scan_node_id,
                                             scan.bound_view,
                                             std::move(columns),
                                             std::move(predicates),
                                             {materializer_kind, entry->registry_profile},
                                             scan.duckdb_types,
                                             scan.table_index);
  auto started = std::chrono::steady_clock::now();
  certification_result result;
  try {
    require_complete_native_scan_schema(scan);
  } catch (duckdb::NotImplementedException const& error) {
    result.verdict     = eligibility_verdict::unsupported;
    result.reason      = verdict_reason::carrier_missing;
    result.reason_text = duckdb::ErrorData(error).Message();
  }
  if (result.verdict == eligibility_verdict::not_evaluated) {
    if (contract_provenance.budget.declines()) {
      result.verdict     = eligibility_verdict::incomplete;
      result.reason      = contract_provenance.budget.time_exceeded() ? verdict_reason::budget_time
                                                                      : verdict_reason::budget_bytes;
      result.reason_text = "GPU scan certification test budget exceeded";
    } else {
      result = entry->certify(
        scan.read_views->entry(scan.contract_id).contract, scan, context, contract_provenance);
    }
  }
  auto const& injected = contract_provenance.injections.scan_verdict;
  auto separator       = injected.find('@');
  auto value           = injected.substr(0, separator);
  uint64_t target =
    separator == std::string::npos ? 1 : std::stoull(injected.substr(separator + 1));
  if (!value.empty() && target == contract_provenance.certification_scan_index &&
      result.verdict == eligibility_verdict::supported) {
    result.verdict =
      value == "unsupported" ? eligibility_verdict::unsupported : eligibility_verdict::incomplete;
    result.reason =
      value == "unsupported" ? verdict_reason::carrier_missing : verdict_reason::evidence_missing;
    result.reason_text = "Injected GPU scan certification " + value;
  }
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - started)
                   .count();
  finish_scan_certification(scan, std::move(result), contract_provenance, elapsed);
}

void lower_table_scan(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                      sirius::operator_params const& params,
                      duckdb::ClientContext& context)
{
  if (!slot->children.empty()) return;
  auto& scan = slot->Cast<sirius::op::sirius_physical_table_scan>();
  require_complete_native_scan_schema(scan);
  auto const* entry = lookup_connector(scan.function, scan.bind_data.get(), context);
  if (!entry || !entry->lower || !scan.contract_id ||
      scan.read_views->entry(scan.contract_id).eligibility.verdict !=
        sirius::op::scan::eligibility_verdict::supported) {
    throw std::logic_error("scan lowering requires a supported verdict");
  }
  auto state = context.registered_state
                 ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                 : nullptr;
  if (state) state->record_scan_lowering(scan.contract_id);
  slot = entry->lower(scan, params, context, entry->filter_mode);
}

// Visit all owned subtrees, including the join and distinct trees outside children[].
void certify_scans_recursive(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                             sirius::operator_params const& params,
                             duckdb::ClientContext& context,
                             scan_contract_provenance& provenance)
{
  if (!slot) return;
  for (auto& child : slot->children)
    certify_scans_recursive(child, params, context, provenance);
  using type = sirius::op::SiriusPhysicalOperatorType;
  if (slot->type == type::LEFT_DELIM_JOIN || slot->type == type::RIGHT_DELIM_JOIN) {
    auto& delim = slot->Cast<sirius::op::sirius_physical_delim_join>();
    certify_scans_recursive(delim.join, params, context, provenance);
    certify_scans_recursive(delim.distinct_root, params, context, provenance);
  }
  if (slot->type == type::TABLE_SCAN) certify_table_scan(slot, params, context, provenance);
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
      hgb_ptr->types.size() == grouped.num_output_group_columns() + grouped.aggregate_slots.size();
    if (has_supported_count_distinct_layout) {
      hgb_ptr->types = grouped.get_count_distinct_local_output_types();
    }

    // Only aggregate-fanout partitions use runtime size estimation.
    auto partition = duckdb::make_uniq<sirius::op::sirius_physical_partition>(
      hgb_ptr->types,
      hgb_ptr->estimated_cardinality,
      /*key_source=*/hgb_ptr,
      /*is_build=*/false,
      compressed_materialization_observer,
      op_params.enable_runtime_size_estimation);
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

//! Terminate one UNION arm with a `PASSTHROUGH_SINK`. Where a join arm needs `PARTITION -> CONCAT`
//! (a shuffle plus a coalesce), a bag union needs neither, so one operator does the whole job: it
//! forwards each batch unchanged and unpartitioned into UNION's `"union_{child_idx}"` port, which
//! keeps every batch on the GPU its scan produced it on. The sink owns that port name — the wiring
//! descriptor and the upstream `next_port_info` both retain a `string_view` into it.
void wrap_union_child(sirius::op::sirius_physical_operator& union_op, std::size_t child_idx)
{
  D_ASSERT(union_op.type == sirius::op::SiriusPhysicalOperatorType::UNION);
  wrap_child(
    union_op, child_idx, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> child_orig) {
      // Capture cardinality before the child is moved into the sink.
      auto est_card       = child_orig->estimated_cardinality;
      auto child_physical = child_orig->has_physical_overrides() ? child_orig->get_physical_types()
                                                                 : std::vector<cudf::data_type>{};

      // The union's schema, not the arm's: the binder has already cast every arm to it, and a
      // materialized CTE's own `types` are its materialization side rather than what it emits.
      auto sink = duckdb::make_uniq<sirius::op::sirius_physical_passthrough_sink>(
        union_op.types, est_card, sirius::op::sirius_physical_union::port_label(child_idx));
      // Expected to be empty: the compressed-schema pass treats UNION as a native boundary and
      // restores every arm before this runs. Carried anyway so the sink never silently declares a
      // different carrier than the batches flowing through it, mirroring wrap_join_child.
      if (!child_physical.empty()) { sink->set_physical_types(std::move(child_physical)); }
      sink->children.push_back(std::move(child_orig));
      return sink;
    });
}

//! Wrap every UNION arm. `wrap_child` replaces the slot in place rather than resizing `children`,
//! so indexing over the original size is safe.
void wrap_union(sirius::op::sirius_physical_operator& union_op)
{
  D_ASSERT(union_op.children.size() >= 2);
  const auto num_children = union_op.children.size();
  for (std::size_t i = 0; i < num_children; i++) {
    wrap_union_child(union_op, i);
  }
}

// Forward declaration: wrap_delim_join recurses into a DELIM JOIN's internal `join`/`distinct`
// subtrees, which live outside `children[]`.
void insert_gpu_pipeline_operators_recursive(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
  const sirius::operator_params& op_params,
  duckdb::ClientContext& context,
  duckdb::SiriusContext* compressed_materialization_observer,
  scan_contract_provenance const& contract_provenance);

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
                     duckdb::SiriusContext* compressed_materialization_observer,
                     scan_contract_provenance const& contract_provenance)
{
  auto& delim_base = slot->Cast<sirius::op::sirius_physical_delim_join>();

  if (delim_base.join) {
    insert_gpu_pipeline_operators_recursive(delim_base.join,
                                            op_params,
                                            context,
                                            compressed_materialization_observer,
                                            contract_provenance);
  }
  if (delim_base.distinct_root) {
    // `distinct_root` still holds the bare DISTINCT: wrap below it first, then above.
    for (auto& child_slot : delim_base.distinct_root->children) {
      insert_gpu_pipeline_operators_recursive(
        child_slot, op_params, context, compressed_materialization_observer, contract_provenance);
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
  duckdb::SiriusContext* compressed_materialization_observer,
  scan_contract_provenance const& contract_provenance)
{
  if (!slot) { return; }

  for (auto& child_slot : slot->children) {
    insert_gpu_pipeline_operators_recursive(
      child_slot, op_params, context, compressed_materialization_observer, contract_provenance);
  }

  switch (slot->type) {
    case sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN:
      lower_table_scan(slot, op_params, context);
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
    case sirius::op::SiriusPhysicalOperatorType::UNION: wrap_union(*slot); break;
    case sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN:
    case sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN:
      wrap_delim_join(
        slot, op_params, context, compressed_materialization_observer, contract_provenance);
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

duckdb::unique_ptr<sirius::op::sirius_physical_operator> lower_native_scan(
  sirius::op::sirius_physical_table_scan& scan,
  sirius::operator_params const& op_params,
  duckdb::ClientContext& context,
  sirius::op::scan::dynamic_filter_apply_mode mode)
{
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  auto leaf       = make_gpu_scan_leaf(build_duckdb_native_table_info(scan, op_params, context),
                                 scan,
                                 op_params,
                                 mode,
                                 sirius_ctx.get());
  if (sirius_ctx) { sirius_ctx->observe_native_checkpoint_for_testing(context, "native_leaf"); }
  return leaf;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> lower_parquet_scan(
  sirius::op::sirius_physical_table_scan& scan,
  sirius::operator_params const& op_params,
  duckdb::ClientContext& context,
  sirius::op::scan::dynamic_filter_apply_mode mode)
{
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  auto table_info = build_parquet_table_info(scan, op_params);
  // DuckDB may bind an explicit file list after opening only its first file. Sirius opens every
  // resolved file later, so install the matching secret snapshot for each path before the bind
  // data moves into the GPU ingestible.
  install_parquet_s3_configs(*table_info, context, sirius_ctx.get());
  return make_gpu_scan_leaf(std::move(table_info), scan, op_params, mode, sirius_ctx.get());
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> lower_iceberg_scan(
  sirius::op::sirius_physical_table_scan& scan,
  sirius::operator_params const& op_params,
  duckdb::ClientContext& context,
  sirius::op::scan::dynamic_filter_apply_mode mode)
{
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  auto table_info = build_iceberg_table_info(scan, op_params, context);
  // Iceberg resolves manifests into parquet data files. Some files may not be opened during
  // binding, so install each path's credential snapshot before the ingestible opens them later.
  install_parquet_s3_configs(*table_info, context, sirius_ctx.get());
  return make_gpu_scan_leaf(std::move(table_info), scan, op_params, mode, sirius_ctx.get());
}

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
  : sirius_physical_plan_generator(
      context, scan_contract_provenance{std::nullopt, reserve_standalone_plan_generation()})
{
}

sirius_physical_plan_generator::sirius_physical_plan_generator(duckdb::ClientContext& context,
                                                               scan_contract_provenance provenance)
  : read_views(std::make_shared<sirius::transparent::read_view_registry>()),
    contract_provenance(std::move(provenance)),
    context(context)
{
  if (context.registered_state) {
    auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (state) read_views->profiles->counters = state->physical_counters();
  }
  // Match option registration: production performs no setting lookups.
  auto const* enabled = std::getenv("SIRIUS_ENABLE_TEST_OPTIONS");
  if (enabled && std::string_view(enabled) == "1") {
    if (read_views->profiles->counters) read_views->profiles->counters->track_units = true;
    duckdb::Value value;
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_inject_certification_delay_ms", value))
      contract_provenance.injections.certification_delay_ms = value.GetValue<uint64_t>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_inject_certification_bytes", value))
      contract_provenance.injections.certification_bytes = value.GetValue<uint64_t>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_budget_declines", value))
      contract_provenance.injections.budget_declines = value.GetValue<bool>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_inject_scan_verdict", value))
      contract_provenance.injections.scan_verdict = value.GetValue<std::string>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_inject_iceberg_discovery", value))
      contract_provenance.injections.iceberg_discovery = value.GetValue<std::string>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_pause_after_certify_ms", value))
      contract_provenance.injections.pause_after_certify_ms = value.GetValue<uint64_t>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_lineage_unmodelled", value))
      contract_provenance.injections.lineage_unmodelled = value.GetValue<bool>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_synthetic_parquet_codec", value))
      contract_provenance.injections.synthetic_parquet_codec = value.GetValue<std::string>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_synthetic_native_segment", value))
      contract_provenance.injections.synthetic_native_segment = value.GetValue<std::string>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_strip_encryption_evidence", value))
      contract_provenance.injections.strip_encryption_evidence = value.GetValue<bool>();
    ++contract_provenance.setting_lookups;
    if (context.TryGetCurrentSetting("sirius_test_invalidate_pin_witness", value))
      contract_provenance.injections.invalidate_pin_witness = value.GetValue<bool>();
    contract_provenance.budget = op::scan::certification_budget(
      std::chrono::milliseconds{50}, 8u << 20, contract_provenance.injections.budget_declines);
  }
  read_views->injections    = contract_provenance.injections;
  auto const has_window     = contract_provenance.window_id.has_value();
  auto const has_generation = contract_provenance.finalize_generation != 0;
  if (has_window == has_generation) {
    throw std::invalid_argument(
      "scan contract provenance requires exactly one of window or plan generation");
  }
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
  // Sink wraps use resolved sizing params, or hardware defaults without a context.
  // The context also observes compressed-materialization counters for PARTITION wraps.
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  auto op_params =
    sirius_ctx
      ? sirius_ctx->get_config().get_operator_params()
      : sirius::operator_params::with_batch_size(sirius::config::derived_default_batch_size());

  std::unordered_map<op::sirius_physical_operator*, plan_lineage> lineage_aliases;
  auto lineage_started = std::chrono::steady_clock::now();
  collect_semantic_lineage(
    *plan, contract_provenance.injections.lineage_unmodelled, lineage_aliases);
  contract_provenance.lineage_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - lineage_started)
                                          .count();
  certify_scans_recursive(plan, op_params, context, contract_provenance);
  std::vector<std::pair<op::scan::scan_contract_id, op::scan::verdict_reason>> declined;
  std::string first_message;
  if (sirius_ctx)
    sirius_ctx->record_certification_budget(contract_provenance.budget.time_exceeded(),
                                            contract_provenance.budget.bytes_exceeded(),
                                            contract_provenance.setting_lookups);
  for (auto const& entry : read_views->entries()) {
    if (sirius_ctx) sirius_ctx->record_scan_certification(entry.eligibility);
    if (entry.eligibility.verdict == op::scan::eligibility_verdict::supported) continue;
    if (declined.empty()) first_message = entry.eligibility.reason_text;
    declined.emplace_back(entry.eligibility.contract_id, entry.eligibility.reason);
    if (sirius_ctx) sirius_ctx->record_semantic_decline(entry.eligibility.reason);
    SIRIUS_LOG_INFO("Scan verdict declined: contract={} reason={} text={}",
                    entry.eligibility.contract_id,
                    static_cast<unsigned>(entry.eligibility.reason),
                    entry.eligibility.reason_text.substr(0, 512));
  }
  if (!declined.empty()) {
    if (contract_provenance.first_pre_decline)
      first_message = contract_provenance.first_pre_decline->second;
    throw op::scan::scan_verdict_declined(std::move(first_message), std::move(declined));
  }
  if (sirius_ctx && sirius_ctx->physical_counters()->track_units &&
      sirius_ctx->physical_counters()->after_certify_for_testing)
    sirius_ctx->physical_counters()->after_certify_for_testing();
  if (contract_provenance.injections.pause_after_certify_ms)
    std::this_thread::sleep_for(
      std::chrono::milliseconds{contract_provenance.injections.pause_after_certify_ms});
  prepare_iceberg_statement(*plan, op_params, context);
  insert_gpu_pipeline_operators_recursive(
    plan, op_params, context, sirius_ctx.get(), contract_provenance);
  if (auto counters = read_views->profiles->counters;
      counters && counters->track_units && counters->scan_plan_complete_for_testing)
    counters->scan_plan_complete_for_testing();
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
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan;
  auto preserve_first_pre_decline = [&] {
    if (!contract_provenance.first_pre_decline) return;
    auto const& [reason, message] = *contract_provenance.first_pre_decline;
    auto state                    = context.registered_state
                                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                                      : nullptr;
    if (state) state->record_semantic_decline(reason);
    SIRIUS_LOG_INFO("Scan verdict declined before complete tree: reason={} text={}",
                    static_cast<unsigned>(reason),
                    message.substr(0, 512));
    throw duckdb::NotImplementedException(message);
  };
  try {
    plan = create_plan(*op);
  } catch (duckdb::NotImplementedException const&) {
    // A pre-capture refusal used to throw before parent planning. Preserve
    // that first error if a later ordinary planner decline aborts tree build.
    preserve_first_pre_decline();
    throw;
  } catch (duckdb::InvalidInputException const&) {
    preserve_first_pre_decline();
    throw;
  } catch (duckdb::Exception const&) {
    // Interrupt, source-policy and other classified engine errors keep priority.
    throw;
  } catch (std::runtime_error const& error) {
    // Existing nested-column planner declines and read-view capture failures use
    // an unclassified runtime_error. Never replace typed contract/cleanup errors.
    if (typeid(error) == typeid(std::runtime_error)) preserve_first_pre_decline();
    throw;
  } catch (std::invalid_argument const&) {
    preserve_first_pre_decline();
    throw;
  }
  profiler.EndPhase();

  plan = fold_adjacent_projections(std::move(plan));
  if (compressed_materialization_active(context)) {
    auto const retracted = apply_compressed_schema_passes(plan);
    if (retracted > 0) {
      auto sirius_ctx = context.registered_state
                          ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                          : nullptr;
      if (sirius_ctx) {
        sirius_ctx->get_event_publisher().publish_compressed_materialization(
          sirius::event::compressed_materialization_activity::scan_narrow_targets_retracted,
          retracted);
      }
    }
  }
  plan->verify();

  // Rewrite the plan tree to contain the GPU pipeline operators so the converter becomes a
  // pure topology pass over `build_pipelines` virtuals; `set_parent_ops` then derives every
  // `_parent_op` from the final tree for the tree-parent-lookup wiring.
  try {
    insert_gpu_pipeline_operators(plan);
  } catch (duckdb::NotImplementedException const&) {
    preserve_first_pre_decline();
    throw;
  } catch (duckdb::InvalidInputException const&) {
    preserve_first_pre_decline();
    throw;
  } catch (duckdb::Exception const&) {
    // Interrupt, source-policy and other classified engine errors keep priority.
    throw;
  } catch (std::runtime_error const& error) {
    // Existing nested-column planner declines and read-view capture failures use
    // an unclassified runtime_error. Never replace typed contract/cleanup errors.
    if (typeid(error) == typeid(std::runtime_error)) preserve_first_pre_decline();
    throw;
  } catch (std::invalid_argument const&) {
    preserve_first_pre_decline();
    throw;
  }
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
      plan = create_plan(op.Cast<duckdb::LogicalCrossProduct>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
      throw duckdb::NotImplementedException("Positional join not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPositionalJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UNION:
      // UNION ALL only; the builder rejects distinct UNION and ordered arms.
      plan = create_plan(op.Cast<duckdb::LogicalSetOperation>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      throw duckdb::NotImplementedException("Set operation (EXCEPT/INTERSECT) not supported");
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
