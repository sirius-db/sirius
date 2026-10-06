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

#pragma once

#include "op/scan/table_scan/bound_read_view.hpp"
#include "transparent/plan_source_policy.hpp"

#include <duckdb/function/table_function.hpp>

#include <optional>
#include <span>

namespace duckdb {
class DatabaseInstance;
class LogicalGet;
class PhysicalTableScan;
}  // namespace duckdb
namespace sirius {
struct operator_params;
namespace op {
class sirius_physical_operator;
class sirius_physical_table_scan;
namespace scan {
enum class dynamic_filter_apply_mode;
}
}  // namespace op
}  // namespace sirius
namespace sirius::planner {
using scan_lowering =
  duckdb::unique_ptr<op::sirius_physical_operator> (*)(op::sirius_physical_table_scan&,
                                                       operator_params const&,
                                                       duckdb::ClientContext&,
                                                       op::scan::dynamic_filter_apply_mode);

struct provider_profile {
  std::string name;
  std::string pin;
  std::string isolation_rule_id;
};

struct connector {
  std::string function_name;
  op::scan::source_kind kind;
  std::string registry_profile;
  bool (*bind_data_matches)(duckdb::FunctionData const*);
  scan_lowering lower;
  op::scan::dynamic_filter_apply_mode filter_mode;
  transparent::byte_source_class byte_source;
  bool permits_cpu_replay;
  bool selector_outside_bind_data;
  std::optional<std::string> (*decline_reason)(duckdb::LogicalGet&, duckdb::ClientContext&);
  std::optional<provider_profile> provider;
};

connector const* lookup_connector(duckdb::LogicalGet const&, duckdb::ClientContext&);
connector const* lookup_connector(duckdb::PhysicalTableScan const&, duckdb::ClientContext&);
connector const* lookup_connector(duckdb::TableFunction const&,
                                  duckdb::FunctionData const*,
                                  duckdb::ClientContext&);
std::span<connector const> registered_connectors();

// Register at Sirius load; bootstrap Iceberg trust during extension loading, never in lookup.
void register_scan_source_callbacks(duckdb::DatabaseInstance&);

// These wrappers keep the existing format builders and admission gates at their original sites.
duckdb::unique_ptr<op::sirius_physical_operator> lower_native_scan(
  op::sirius_physical_table_scan&,
  operator_params const&,
  duckdb::ClientContext&,
  op::scan::dynamic_filter_apply_mode);
duckdb::unique_ptr<op::sirius_physical_operator> lower_parquet_scan(
  op::sirius_physical_table_scan&,
  operator_params const&,
  duckdb::ClientContext&,
  op::scan::dynamic_filter_apply_mode);
duckdb::unique_ptr<op::sirius_physical_operator> lower_iceberg_scan(
  op::sirius_physical_table_scan&,
  operator_params const&,
  duckdb::ClientContext&,
  op::scan::dynamic_filter_apply_mode);
std::optional<std::string> registered_iceberg_decline_reason(duckdb::LogicalGet&,
                                                             duckdb::ClientContext&);
}  // namespace sirius::planner
