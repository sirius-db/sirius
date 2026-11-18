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

#include "planner/dynamic_filter/build_key_domain.hpp"

#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/multi_file/table_function_multi_file.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "helper/type_conversions.hpp"
#include "parquet_multi_file_info.hpp"
#include "planner/connector_registry.hpp"
#include "planner/scan_column_origin.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_registration.hpp"

#include <algorithm>
#include <optional>
#include <string>

namespace sirius::planner {

namespace detail {

std::vector<std::optional<scan_column_origin>> resolve_build_key_scan_columns(
  duckdb::LogicalComparisonJoin const& join)
{
  std::vector<std::optional<scan_column_origin>> scans(join.conditions.size());
  if (join.children.size() != 2) { return scans; }
  for (std::size_t condition_index = 0; condition_index < join.conditions.size();
       ++condition_index) {
    auto const& build_side = join.conditions[condition_index].GetRHS();
    if (build_side.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { continue; }
    auto const ordinal =
      static_cast<std::size_t>(build_side.Cast<duckdb::BoundReferenceExpression>().Index());
    scans[condition_index] =
      resolve_scan_column_origin(*join.children[1], ordinal, origin_policy::row_subset);
  }
  return scans;
}

duckdb::LogicalGet const* resolve_pass_through_scan(duckdb::LogicalOperator const& subtree,
                                                    std::size_t output_ordinal,
                                                    origin_policy policy) noexcept
{
  auto const origin = resolve_scan_column_origin(subtree, output_ordinal, policy);
  return origin ? origin->get : nullptr;
}

std::vector<duckdb::LogicalGet const*> resolve_build_key_scans(
  duckdb::LogicalComparisonJoin const& join, origin_policy policy)
{
  std::vector<duckdb::LogicalGet const*> scans(join.conditions.size(), nullptr);
  if (join.children.size() != 2) { return scans; }
  for (std::size_t condition_index = 0; condition_index < join.conditions.size();
       ++condition_index) {
    auto const& build_side = join.conditions[condition_index].GetRHS();
    if (build_side.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { continue; }
    auto const ordinal =
      static_cast<std::size_t>(build_side.Cast<duckdb::BoundReferenceExpression>().Index());
    scans[condition_index] = resolve_pass_through_scan(*join.children[1], ordinal, policy);
  }
  return scans;
}

}  // namespace detail

namespace {

bool has_default_parquet_reader_options(duckdb::FunctionData const& bind_data)
{
  auto const& options = duckdb::ParquetScanFunction::GetFileOptions(bind_data);
  duckdb::ParquetOptions const defaults;
  return options.binary_as_string == defaults.binary_as_string &&
         options.variant_legacy_encoding == defaults.variant_legacy_encoding &&
         !options.encryption_config &&
         options.explicit_cardinality == defaults.explicit_cardinality &&
         options.can_have_nan == defaults.can_have_nan &&
         options.prefetch_strategy == defaults.prefetch_strategy &&
         options.utf8_validation_option == defaults.utf8_validation_option &&
         options.int96_as == defaults.int96_as;
}

// Pins decode the physical schema with default reader options; transformed readers have no matching
// evidence.
bool is_untransformed_parquet_scan(duckdb::LogicalGet const& get)
{
  if (!get.named_parameters.empty() || get.has_pushed_projection) { return false; }
  if (get.function.GetName() == "sirius_read_parquet") {
    auto const* bind = dynamic_cast<duckdb::SiriusReadParquetBindData const*>(get.bind_data.get());
    return bind != nullptr && get.parameters.size() == 1 && !get.parameters[0].IsNull() &&
           get.parameters[0].GetValue<std::string>() == bind->uri;
  }
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get());
  if (bind == nullptr || !bind->union_readers.empty()) { return false; }
  auto const& options = bind->file_options;
  if (options.filename || options.hive_partitioning || options.union_by_name ||
      !options.schema.empty() || options.file_row_number || !options.custom_options.empty() ||
      options.mapping != duckdb::MultiFileColumnMappingMode::BY_NAME ||
      options.maximum_sample_files != 1) {
    return false;
  }
  auto const* wrapped =
    dynamic_cast<duckdb::TableFunctionMultiFileData const*>(bind->bind_data.get());
  if (wrapped == nullptr || !wrapped->options.named_parameters.empty()) { return false; }
  bool checked = false;
  if (wrapped->options.schema_bind_data) {
    if (!has_default_parquet_reader_options(*wrapped->options.schema_bind_data)) { return false; }
    checked = true;
  }
  if (bind->initial_reader) {
    auto const* reader =
      dynamic_cast<duckdb::TableFunctionFileReader const*>(bind->initial_reader.get());
    if (reader == nullptr || !reader->bind_data ||
        !has_default_parquet_reader_options(*reader->bind_data)) {
      return false;
    }
    checked = true;
  }
  return checked;
}

std::optional<std::size_t> catalog_cardinality(duckdb::LogicalGet const& get,
                                               duckdb::ClientContext& context) noexcept
{
  // Other table functions may report estimates below the true domain.
  if (get.function.GetName() != "seq_scan" || !get.function.cardinality || !get.bind_data) {
    return std::nullopt;
  }
  try {
    auto const stats = get.function.cardinality(context, get.bind_data.get());
    if (!stats || !stats->has_max_cardinality) { return std::nullopt; }
    return static_cast<std::size_t>(stats->max_cardinality);
  } catch (...) {
    // Optional evidence must not fail query planning.
    return std::nullopt;
  }
}

// Exact evidence from one pin of the scanned files, or nullopt when the pin cannot describe the
// scanned column with its native type.
std::optional<op::dynamic_filter_domain_evidence> pinned_column_evidence(
  scan_manager::pinned_entry const& entry,
  duckdb::ColumnIndex const& column,
  std::string const& scan_name,
  duckdb::LogicalType const& scan_type)
{
  if (entry.num_rows == 0 || !entry.identity_evidence.passed || !entry.layout_evidence.passed ||
      !entry.cache_info.table_name.empty() ||
      (entry.tier != cucascade::memory::Tier::GPU && entry.tier != cucascade::memory::Tier::HOST)) {
    return std::nullopt;
  }
  auto const projection = entry.cache_info.column_projection_for({column});
  if (projection.size() != 1 || projection[0] >= entry.cache_info.names.size()) {
    return std::nullopt;
  }
  auto const pinned_position = projection[0];
  auto const& name           = entry.cache_info.names[pinned_position];
  if (name != scan_name) { return std::nullopt; }
  auto const native = sirius::get_cudf_type(sirius::from_duckdb(scan_type));
  if (!native || entry.column_storage.empty()) { return std::nullopt; }
  if (entry.zone_maps.has_stats() && (pinned_position >= entry.zone_maps.column_count() ||
                                      entry.zone_maps.column_type(pinned_position) != scan_type)) {
    return std::nullopt;
  }
  for (auto const& chunk : entry.column_storage) {
    if (pinned_position >= chunk.size() || chunk[pinned_position].native != *native) {
      return std::nullopt;
    }
  }
  return op::dynamic_filter_domain_evidence{
    entry.num_rows,
    op::dynamic_filter_domain_source::pinned_parquet_exact,
    std::binary_search(
      entry.declared_unique_columns.begin(), entry.declared_unique_columns.end(), name)};
}

op::dynamic_filter_domain_evidence evidence_for_column(
  duckdb::LogicalGet const& get,
  std::size_t scan_ordinal,
  duckdb::ClientContext& context,
  scan_manager::sirius_scan_manager const& registry) noexcept
{
  try {
    if (get.function.GetName() == "seq_scan") {
      auto const cardinality = catalog_cardinality(get, context).value_or(0);
      return cardinality == 0
               ? op::dynamic_filter_domain_evidence{}
               : op::dynamic_filter_domain_evidence{
                   cardinality, op::dynamic_filter_domain_source::catalog_bound, false};
    }
    auto const& function_name = get.function.GetName();
    if (function_name != "read_parquet" && function_name != "parquet_scan" &&
        function_name != "sirius_read_parquet") {
      return {};
    }
    if (lookup_connector(get, context) == nullptr || !is_untransformed_parquet_scan(get)) {
      return {};
    }

    std::size_t position = scan_ordinal;
    if (!get.projection_ids.empty()) {
      if (position >= get.projection_ids.size()) { return {}; }
      position = static_cast<std::size_t>(get.projection_ids[position]);
    }
    auto const& columns = get.GetColumnIds();
    if (position >= columns.size()) { return {}; }
    auto const& column = columns[position];
    if (!column.HasPrimaryIndex() || column.IsRowIdColumn() || column.IsVirtualColumn() ||
        column.IsEmptyColumn() || column.HasChildren()) {
      return {};
    }
    auto const primary = column.GetPrimaryIndex();
    if (primary >= get.names.size() || primary >= get.returned_types.size()) { return {}; }

    auto const files = resolve_parquet_scan_file_paths(
      function_name.GetIdentifierName(), get.bind_data.get(), get.parameters);
    // Every pin of one file set describes the same rows, so a declaration on any pin that holds
    // the column applies; pins that disagree on the row count supply no evidence.
    std::optional<op::dynamic_filter_domain_evidence> chosen;
    for (auto const& entry : registry.find_pinned_entries_for_parquet_files(files)) {
      auto const evidence = pinned_column_evidence(
        *entry, column, get.names[primary].GetIdentifierName(), get.returned_types[primary]);
      if (!evidence) { continue; }
      if (chosen && chosen->cardinality != evidence->cardinality) { return {}; }
      if (!chosen || evidence->declared_unique) { chosen = evidence; }
    }
    return chosen.value_or(op::dynamic_filter_domain_evidence{});
  } catch (...) {
    return {};
  }
}

}  // namespace

std::vector<op::dynamic_filter_domain_evidence> build_key_domain_evidence_for(
  duckdb::LogicalComparisonJoin const& join,
  duckdb::ClientContext& context,
  scan_manager::sirius_scan_manager const& registry)
{
  return build_key_domain_evidence_for(
    join, [&](duckdb::LogicalGet const& get, std::size_t ordinal) {
      return evidence_for_column(get, ordinal, context, registry);
    });
}

duckdb_base_table_cardinality::duckdb_base_table_cardinality(
  duckdb::ClientContext& context) noexcept
  : _context{&context}
{
}

std::optional<std::size_t> duckdb_base_table_cardinality::operator()(
  duckdb::LogicalGet const& get) const noexcept
{
  return catalog_cardinality(get, *_context);
}

}  // namespace sirius::planner
