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
#include "op/scan/table_scan/scan_contract.hpp"
#include "scan_manager/preparation_ledger.hpp"
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
class gpu_ingestible;
}  // namespace scan
}  // namespace op
}  // namespace sirius
namespace sirius::planner {
struct scan_contract_provenance;
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
  op::scan::pre_capture_result (*decline_reason)(duckdb::LogicalGet&,
                                                 duckdb::ClientContext&,
                                                 scan_contract_provenance&);
  std::optional<provider_profile> provider;
  op::scan::certification_result (*certify)(op::scan::bound_table_scan const&,
                                            op::sirius_physical_table_scan const&,
                                            duckdb::ClientContext&,
                                            scan_contract_provenance&) = nullptr;
  // On success, lookup delegates the bound candidate's function_info validation to
  // this hook. The bridge must verify its provider-owned function_info and binding
  // identity together; verifying only the nested Parquet wrapper is insufficient.
  bool (*verify_binding)(duckdb::BaseTableFunction const&,
                         duckdb::FunctionData const*,
                         duckdb::ClientContext&) = nullptr;
  // Original semantic binding only: no SQL rebind, inventory discovery or reader
  // state consumption. May also run for a pre-declined scan to retain correspondence.
  op::scan::bound_read_view (*capture)(duckdb::BoundTableFunction const&,
                                       duckdb::FunctionData const*,
                                       duckdb::ClientContext&) = nullptr;
  // Contribute to the single statement admission before lowering any scan.
  std::optional<scan_manager::preparation_request> (*prepare)(op::sirius_physical_table_scan&,
                                                              operator_params const&,
                                                              duckdb::ClientContext&) = nullptr;
};

enum class lookup_decline : uint8_t {
  none,
  unknown_function,
  bind_data_mismatch,
  catalog_entry_missing,
  no_trusted_reference,
  callback_mismatch
};
struct lookup_outcome {
  connector const* entry = nullptr;
  lookup_decline decline = lookup_decline::none;
};
lookup_outcome lookup_connector_classified(duckdb::LogicalGet const&, duckdb::ClientContext&);
connector const* lookup_connector(duckdb::LogicalGet const&, duckdb::ClientContext&);
connector const* lookup_connector(duckdb::PhysicalTableScan const&, duckdb::ClientContext&);
connector const* lookup_connector(duckdb::TableFunction const&,
                                  duckdb::FunctionData const*,
                                  duckdb::ClientContext&);
connector const* lookup_connector(duckdb::BoundTableFunction const&,
                                  duckdb::FunctionData const*,
                                  duckdb::ClientContext&);
//! Whether an Iceberg bind exposes a complete table schema with integer field IDs.
bool iceberg_flat_mapping_supported(duckdb::FunctionData const* bind_data);
bool iceberg_table_schema_has_field_ids(duckdb::FunctionData const* bind_data);
std::span<connector const> registered_connectors();

// Register at Sirius load; bootstrap Iceberg trust during extension loading, never in lookup.
void register_scan_source_callbacks(duckdb::DatabaseInstance&);

// Assemble an already prepared materializer using the shared dynamic-filter path.
duckdb::unique_ptr<op::sirius_physical_operator> lower_prepared_scan(
  std::shared_ptr<op::scan::gpu_ingestible>,
  op::sirius_physical_table_scan const&,
  operator_params const&,
  duckdb::ClientContext&,
  op::scan::dynamic_filter_apply_mode);

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
op::scan::pre_capture_result registered_iceberg_decline_reason(duckdb::LogicalGet&,
                                                               duckdb::ClientContext&,
                                                               scan_contract_provenance&);
/// Verify the nested multi-file wrapper against this host's Parquet implementation.
/// Does not verify the outer provider's function_info or binding identity.
bool verified_parquet_interface(duckdb::FunctionData const*, duckdb::ClientContext&);

}  // namespace sirius::planner
