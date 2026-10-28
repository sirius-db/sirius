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

#include "ducklake/source_registration.hpp"

#include "ducklake/binding_bridge.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_batch_producer.hpp"
#include "ducklake/inline_varchar.hpp"
#include "ducklake/scan_ingestible.hpp"
#include "op/scan/dynamic_filter_merge.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "transparent/read_view_registry.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>

#include <numeric>
namespace sirius::ducklake {
namespace {
struct scan_layout {
  std::vector<size_t> selected;
  unit_scan_options options;
  bool count_only = false;
};
scan_layout layout(op::sirius_physical_table_scan const& scan)
{
  scan_layout result;
  result.options.profiles = scan.read_views->profiles;
  result.count_only       = scan.column_ids.size() == 1 &&
                      scan.column_ids.front().GetPrimaryIndex() == duckdb::COLUMN_IDENTIFIER_EMPTY;
  if (result.count_only) {
    if (scan.table_filters && scan.table_filters->HasFilters())
      throw std::invalid_argument("DuckLake empty-column scan has a data predicate");
    return result;
  }
  auto const& types = scan.bound_view->identity->bound_types;
  for (auto const& column : scan.column_ids) {
    auto index = column.GetPrimaryIndex();
    if (column.IsVirtualColumn() || column.HasChildren() || index >= types.size() ||
        (!fixed_scalar_arrow_format(types[index]) && types[index] != duckdb::LogicalType::VARCHAR))
      throw std::invalid_argument(
        "DuckLake selected column is outside the scalar materialization profile");
    result.selected.push_back(index);
  }
  if (result.selected.empty() || result.selected.size() > inline_batch_producer::max_columns)
    throw std::invalid_argument(
      "DuckLake selected column count exceeds the materialization profile");
  auto outputs = std::vector<size_t>(scan.projection_ids.begin(), scan.projection_ids.end());
  if (outputs.empty()) {
    outputs.resize(result.selected.size());
    std::iota(outputs.begin(), outputs.end(), 0);
  }
  if (scan.projected_output_count > outputs.size())
    throw std::invalid_argument("DuckLake output projection exceeds selected columns");
  outputs.resize(scan.projected_output_count);
  result.options.output_columns = std::move(outputs);
  if (scan.table_filters && scan.table_filters->HasFilters())
    result.options.filters = scan.table_filters->Copy();
  validate_scan_options(result.options, result.selected.size());
  return result;
}
std::optional<scan_manager::preparation_request> prepare_scan(op::sirius_physical_table_scan& scan,
                                                              operator_params const& params,
                                                              duckdb::ClientContext& context)
{
  if (!bridge_available())
    throw duckdb::NotImplementedException("DuckLake private binding bridge is disabled");
  auto binding = binding_handle::capture(scan.function, scan.bind_data.get(), context);
  if (binding->view().identity->fingerprint != scan.bound_view->identity->fingerprint)
    throw std::invalid_argument("DuckLake prepared binding differs from the certified read view");
  auto selected                          = std::make_shared<scan_layout>(layout(scan));
  selected->options.scan_task_batch_size = params.scan_task_batch_size;
  return scan_manager::preparation_request{
    {{scan_ingestible::envelope()}, {}, true, 1},
    [](scan_manager::preparation_ledger&) {},
    [&scan, binding, selected](std::shared_ptr<scan_manager::preparation_admission> admission) {
      auto const& entry = scan.read_views->entry(scan.contract_id);
      auto source =
        std::make_shared<scan_ingestible>(binding,
                                          scan.contract_id,
                                          entry.window_id.value_or(entry.finalize_generation),
                                          std::move(admission),
                                          selected->selected,
                                          selected->count_only,
                                          std::move(selected->options));
      scan.read_views->bind_split_validator(scan.contract_id, source->materializer().validator);
      scan.prepared_ingestible = std::move(source);
    },
    [&scan]() noexcept { scan.prepared_ingestible.reset(); }};
}
duckdb::unique_ptr<op::sirius_physical_operator> lower_scan(
  op::sirius_physical_table_scan& scan,
  operator_params const& params,
  duckdb::ClientContext& context,
  op::scan::dynamic_filter_apply_mode mode)
{
  if (!bridge_available())
    throw duckdb::NotImplementedException("DuckLake private binding bridge is disabled");
  if (!scan.prepared_ingestible)
    throw scan_manager::preparation_resource_error(
      "DuckLake statement preparation was not admitted", false);
  return planner::lower_prepared_scan(
    std::move(scan.prepared_ingestible), scan, params, context, mode);
}
op::scan::certification_result bridge_unavailable()
{
  op::scan::certification_result result;
  result.verdict     = op::scan::eligibility_verdict::incomplete;
  result.reason      = op::scan::verdict_reason::interface_unavailable;
  result.reason_text = "DuckLake private binding bridge is disabled";
  return result;
}
}  // namespace
planner::connector scan_connector()
{
  return {.function_name    = "ducklake_scan",
          .kind             = op::scan::source_kind::parquet_local,
          .registry_profile = "duckdb.ducklake.local.v1",
          .bind_data_matches =
            [](duckdb::FunctionData const* p) {
              return dynamic_cast<duckdb::MultiFileBindData const*>(p) != nullptr;
            },
          .lower                      = lower_scan,
          .filter_mode                = op::scan::dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
          .byte_source                = transparent::byte_source_class::unclassified,
          .permits_cpu_replay         = true,
          .selector_outside_bind_data = false,
          .decline_reason =
            [](duckdb::LogicalGet&, duckdb::ClientContext&, planner::scan_contract_provenance&) {
              if (bridge_available()) return op::scan::pre_capture_result{};
              auto r = bridge_unavailable();
              op::scan::pre_capture_result result;
              result.decline = op::scan::pre_decline{r.verdict, r.reason, r.reason_text};
              return result;
            },
          .provider =
            planner::provider_profile{"ducklake", "417a9117f5", "ducklake.original-statement.v1"},
          .certify =
            [](op::scan::bound_table_scan const&,
               op::sirius_physical_table_scan const& scan,
               duckdb::ClientContext&,
               planner::scan_contract_provenance&) {
              if (!bridge_available()) return bridge_unavailable();
              op::scan::certification_result result;
              try {
                (void)layout(scan);
              } catch (std::invalid_argument const& error) {
                result.verdict     = op::scan::eligibility_verdict::unsupported;
                result.reason      = op::scan::verdict_reason::carrier_missing;
                result.reason_text = error.what();
                return result;
              }
              result.verdict      = op::scan::eligibility_verdict::supported;
              result.later_checks = op::scan::check_bit(op::scan::later_check::footer_per_file) |
                                    op::scan::check_bit(op::scan::later_check::profile_per_file) |
                                    op::scan::check_bit(op::scan::later_check::schema_per_file) |
                                    op::scan::check_bit(op::scan::later_check::host_staged);
              result.semantic_columns = scan.semantic_columns;
              return result;
            },
          .verify_binding = binding_matches,
          .capture        = capture_read_view,
          .prepare        = prepare_scan};
}
}  // namespace sirius::ducklake
