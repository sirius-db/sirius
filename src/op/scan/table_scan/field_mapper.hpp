/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "op/scan/field_id_mapping.hpp"
#include "op/scan/scan_plan.hpp"

#include <variant>

namespace sirius::op::scan {
enum class mapping_selection { full_correspondence, reader_projection };
struct mapping_request {
  rule_set_id rules;
  std::shared_ptr<parquet_metadata const> footer;
  std::string file;
  scan_plan const& layout;
  materialization_program program;
  iceberg_table_schema const* iceberg_schema = nullptr;
  // Full bound primary-index order, including filter-only/carrier requirements.
  std::span<field_requirement const> requirements;
  std::unordered_set<std::string> const* files_without_partition_values = nullptr;
  std::span<duckdb::LogicalType const> bound_types;
  std::function<std::shared_ptr<field_id_mapping const>(std::shared_ptr<parquet_metadata const>,
                                                        std::string const&,
                                                        materialization_program,
                                                        std::vector<field_requirement>)>
    prepare;
  bool metadata_only      = false;
  bool allow_file_carrier = false;
  std::span<std::string const> bound_names;
};
struct mapping_result {
  physical_profile_result evidence;
  mapping_selection selection = mapping_selection::full_correspondence;
  std::vector<std::vector<std::size_t>> leaves;
  std::unordered_map<std::string, std::vector<std::size_t>> physical_leaves;
  std::shared_ptr<field_id_mapping const> compiled;
  std::shared_ptr<parquet_metadata const> footer;
  std::vector<duckdb::LogicalType> bound_types;
  // The actual reader choice, separately from logical-to-physical leaf bindings.
  // natural_read means no physical projection, even when a bound carrier is absent.
  effective_reader_projection reader_projection;
  std::optional<std::string> file_carrier;
  std::size_t file_carrier_width = 0;
  bool carrier_unavailable       = false;
  uint64_t contract              = 0;
  uint64_t generation            = 0;
};
struct mapping_refusal {
  std::string file;
  rule_set_id rules;
  verdict_reason reason;
  std::string text;
  std::optional<refusal_rule> rule;
  physical_profile_result evidence;
};
using mapping_outcome = std::variant<std::shared_ptr<mapping_result const>, mapping_refusal>;
mapping_outcome map_file(mapping_request const&, physical_check_counters* counters = nullptr);
}  // namespace sirius::op::scan
