/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/table_scan/field_mapper.hpp"

#include "io/parquet_helpers.hpp"
#include "io/uri_parser.hpp"
#include "op/scan/parquet_schema_mapping.hpp"

#include <algorithm>
#include <unordered_map>

namespace sirius::op::scan {
mapping_outcome map_file(mapping_request const& request, physical_check_counters* counters)
{
  auto refuse = [&](verdict_reason reason,
                    std::string text,
                    std::optional<refusal_rule> rule = {},
                    physical_profile_result evidence = {}) -> mapping_outcome {
    if (rule && counters) counters->record_mapper_refusal(request.rules, *rule);
    return mapping_refusal{
      request.file, request.rules, reason, std::move(text), rule, std::move(evidence)};
  };
  if (!request.footer || !request.footer->file_metadata() || !request.program.contract ||
      !request.program.generation)
    return refuse(verdict_reason::interface_unavailable,
                  "field mapping has no bound footer or program",
                  refusal_rule::unbound_generation);
  auto const& metadata = *request.footer->file_metadata();
  mapping_result result;
  result.footer = request.footer;
  result.bound_types.assign(request.bound_types.begin(), request.bound_types.end());
  result.contract   = request.program.contract;
  result.generation = request.program.generation;
  result.selection  = request.layout.is_projected() ? mapping_selection::reader_projection
                                                    : mapping_selection::full_correspondence;
  if (request.rules == rule_set_id::iceberg_field_id_v1) {
    if (!request.iceberg_schema)
      return refuse(verdict_reason::interface_unavailable,
                    "Iceberg mapping requires bound schema",
                    refusal_rule::unbound_generation);
    result.evidence = check_iceberg_file_schema(metadata,
                                                *request.iceberg_schema,
                                                request.file,
                                                request.footer->original_logical_annotations);
    if (!result.evidence.approved)
      return refuse(result.evidence.reason, result.evidence.text, {}, result.evidence);
  } else if (request.rules == rule_set_id::iceberg_flat_v1) {
    try {
      for (std::size_t i = 1; i < metadata.schema.size(); ++i)
        if (!metadata.schema[i].field_id)
          throw mapping_error(refusal_rule::untagged_field,
                              "iceberg_flat_v1 requires IDs on every physical field");
      bool defaults_proved =
        request.files_without_partition_values && request.files_without_partition_values->contains(
                                                    sirius::io::strip_file_scheme(request.file));
      std::vector<field_requirement> fields;
      auto const ids = index_parquet_field_ids(metadata);
      for (auto const& column : request.layout.data_columns) {
        if (column.primary_idx >= request.requirements.size())
          throw mapping_error(refusal_rule::program_target_mismatch,
                              "field requirement missing from bound schema");
        fields.push_back(request.requirements[column.primary_idx]);
        if (!defaults_proved && fields.back().missing_value && !ids.contains(fields.back().id))
          throw mapping_error(refusal_rule::partition_constant_unproved,
                              "missing field requires an explicitly typed default");
        if (!defaults_proved) fields.back().missing_value.reset();
      }
      result.compiled = request.prepare
                          ? request.prepare(request.footer,
                                            request.footer->built_for_identity,
                                            request.program,
                                            std::move(fields))
                          : std::make_shared<field_id_mapping const>(
                              field_id_mapping::compile(request.footer,
                                                        request.footer->built_for_identity,
                                                        request.program,
                                                        std::move(fields)));
      if (!result.compiled || result.compiled->identity() != request.footer->built_for_identity ||
          result.compiled->program().contract != request.program.contract ||
          result.compiled->program().generation != request.program.generation)
        throw mapping_error(refusal_rule::unbound_generation,
                            "adapter-owned field mapping has mismatched identity");
      result.evidence.approved   = true;
      result.evidence.validation = check_bit(later_check::schema_per_file);
    } catch (mapping_error const& error) {
      return refuse(verdict_reason::interface_unavailable, error.what(), error.rule());
    } catch (std::invalid_argument const& error) {
      // Adapter diagnostics retain their original text without guessing a rule.
      return refuse(verdict_reason::interface_unavailable, error.what());
    }
  } else {
    // Name mapping preserves grouped physical leaves. Empty/missing names are
    // diagnosed by the existing projection stage, at its original phase.
    result.evidence.approved = true;
  }
  std::unordered_map<std::string, std::vector<std::size_t>> by_name;
  if (!metadata.row_groups.empty()) {
    auto const& columns = metadata.row_groups.front().columns;
    for (std::size_t leaf = 0; leaf < columns.size(); ++leaf) {
      auto const& path = columns[leaf].meta_data.path_in_schema;
      if (!path.empty()) by_name[path.front()].push_back(leaf);
    }
  }
  if (result.compiled) {
    for (auto const& name : result.compiled->read_names())
      result.leaves.push_back(by_name[name]);
  } else {
    for (auto const& column : request.layout.data_columns)
      result.leaves.push_back(by_name[column.name]);
  }
  result.physical_leaves = std::move(by_name);
  if (!result.compiled && request.allow_file_carrier && !request.metadata_only &&
      request.layout.has_user_virtual_columns() && request.layout.carrier_batch_index &&
      !metadata.row_groups.empty()) {
    for (auto const& column : metadata.schema) {
      bool partition = std::any_of(request.layout.partition_primary_indices.begin(),
                                   request.layout.partition_primary_indices.end(),
                                   [&](auto p) {
                                     return p < request.bound_names.size() &&
                                            request.bound_names[p] == column.name;
                                   });
      if (partition ||
          std::any_of(request.layout.partition_columns.begin(),
                      request.layout.partition_columns.end(),
                      [&](auto const& partition) { return partition.name == column.name; }))
        continue;
      auto width = detail::carrier_decoded_width(column);
      if (width && (!result.file_carrier || width < result.file_carrier_width)) {
        result.file_carrier       = column.name;
        result.file_carrier_width = width;
      }
    }
  }
  if (!result.compiled && !result.file_carrier && request.layout.carrier_batch_index) {
    auto const& name = request.layout.data_columns.at(*request.layout.carrier_batch_index).name;
    auto leaves      = result.physical_leaves.find(name);
    result.carrier_unavailable = leaves == result.physical_leaves.end() || leaves->second.empty();
  }
  auto& projection = result.reader_projection;
  if (request.metadata_only) {
    result.selection = mapping_selection::full_correspondence;
  } else if (result.compiled) {
    projection.names        = result.compiled->read_names();
    projection.bound_types  = result.compiled->read_types();
    projection.natural_read = projection.names.empty();
  } else if (result.file_carrier) {
    auto schema = sirius::io::parquet_helpers::extract_schema(
      metadata, true, request.footer->original_logical_annotations);
    auto found = std::find(schema.names.begin(), schema.names.end(), *result.file_carrier);
    if (found == schema.names.end())
      throw std::invalid_argument("Parquet carrier missing from physical schema");
    projection.names       = {*result.file_carrier};
    projection.bound_types = {schema.types[std::distance(schema.names.begin(), found)]};
  } else {
    projection.natural_read = !request.layout.is_projected() || result.carrier_unavailable;
    for (auto const& column : request.layout.data_columns) {
      projection.names.push_back(column.name);
      if (!request.bound_types.empty()) {
        if (column.primary_idx >= request.bound_types.size())
          return refuse(verdict_reason::interface_unavailable,
                        "field mapping bound type index is out of range",
                        refusal_rule::program_target_mismatch);
        projection.bound_types.push_back(request.bound_types[column.primary_idx]);
      }
    }
  }
  if (!request.metadata_only)
    for (std::size_t i = 0; i < projection.names.size(); ++i)
      projection.projected.push_back(i);
  if (projection.natural_read) result.selection = mapping_selection::full_correspondence;
  return std::make_shared<mapping_result const>(std::move(result));
}
}  // namespace sirius::op::scan
