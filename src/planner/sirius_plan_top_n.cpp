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

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_vss_ann_ivf_flat.hpp"
#include "op/sirius_physical_vss_enn.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/vss_pattern.hpp"

#include <cucascade/memory/common.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

/// Map a LogicalGet output-column position (a BoundReferenceExpression index into
/// the gets output, which the projection's expressions reference) to its base
/// table column name. projection_ids reorders/filters the gets output when set.
/// Returns nullopt if the position is out of range.
std::optional<std::string> get_column_name_at_output(duckdb::LogicalGet const& get,
                                                     duckdb::TableCatalogEntry const& table,
                                                     duckdb::idx_t output_index)
{
  auto const& column_ids = get.GetColumnIds();
  auto const& proj_ids   = get.projection_ids;

  duckdb::idx_t table_col = 0;
  if (proj_ids.empty()) {
    if (output_index >= column_ids.size()) { return std::nullopt; }
    table_col = column_ids[output_index].GetPrimaryIndex();
  } else {
    if (output_index >= proj_ids.size()) { return std::nullopt; }
    table_col = column_ids[proj_ids[output_index]].GetPrimaryIndex();
  }

  auto const& names = table.GetColumns().GetColumnNames();
  if (table_col >= names.size()) { return std::nullopt; }
  return names[table_col];
}

/// ANN routing decision for a matched VSS top-k.
struct ann_route {
  duckdb::SiriusContext* sirius_context;
  std::string table_name;
  std::string vector_column_name;
  /// Base-table column name for each pattern.output_columns entry of kind
  /// gather_input (aligned by index; distance entries hold "").
  std::vector<std::string> output_column_names;
};

/// Decide whether this VSS top-k can be served by a pinned ANN index. Succeeds only
/// when all hold: the projection's input is a direct LogicalGet, the vector column
/// and every gather_input output column resolve to base table columns, a pinned
/// cuVS index covers (table, vector column, metric), and the table is pinned on the
/// GPU tier (the ANN operator gathers output columns from it). Otherwise, nullopt
/// and the caller runs the exact (ENN) path.
std::optional<ann_route> resolve_ann_route(duckdb::ClientContext& context,
                                           duckdb::LogicalProjection const& proj,
                                           sirius::vss::vss_top_k_pattern const& pattern)
{
  if (proj.children.empty() || proj.children[0]->type != duckdb::LogicalOperatorType::LOGICAL_GET) {
    return std::nullopt;
  }
  auto& get  = proj.children[0]->Cast<duckdb::LogicalGet>();
  auto table = get.GetTable();
  if (!table) { return std::nullopt; }

  auto vector_name =
    get_column_name_at_output(get, *table, static_cast<duckdb::idx_t>(pattern.vector_column_index));
  if (!vector_name) { return std::nullopt; }

  std::vector<std::string> output_names(pattern.output_columns.size());
  for (std::size_t i = 0; i < pattern.output_columns.size(); ++i) {
    auto const& oc = pattern.output_columns[i];
    if (oc.which != sirius::vss::vss_output_column::kind::gather_input) { continue; }
    auto name = get_column_name_at_output(get, *table, static_cast<duckdb::idx_t>(oc.input_index));
    if (!name) { return std::nullopt; }
    output_names[i] = std::move(*name);
  }

  if (!context.registered_state) { return std::nullopt; }
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return std::nullopt; }

  // A pinned index must cover (table, vector column, metric)
  if (sirius_ctx->get_cuvs_index_cache().find_by_column(
        table->name, *vector_name, pattern.metric) == nullptr) {
    return std::nullopt;
  }
  // ...AND the table must be GPU-pinned so the operator can gather output columns.
  const auto* pin = sirius_ctx->get_scan_manager().find_pinned_entry(table->name);
  if (pin == nullptr || pin->tier != cucascade::memory::Tier::GPU) { return std::nullopt; }

  return ann_route{sirius_ctx.get(), table->name, std::move(*vector_name), std::move(output_names)};
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalTopN& op)
{
  D_ASSERT(op.children.size() == 1);

  // bypass the projection and plan its child as the VSS source
  if (auto pattern = sirius::vss::match_vss_top_n(op)) {
    auto& proj = op.children[0]->Cast<duckdb::LogicalProjection>();

    // Auto-route: when a pinned cuVS ANN index covers this query's vector column
    // (same metric) and the table is GPU-pinned, serve it with an approximate
    // search over the resident index instead of the exact brute-force path. The
    // ANN operator is a pure source (no children) which reads the pinned table for
    // output columns. Otherwise, fall through to the exact enn path.
    if (auto route = resolve_ann_route(context, proj, *pattern)) {
      SIRIUS_LOG_INFO("VSS: routing to pinned ANN (IVF-Flat) for {}.{}",
                      route->table_name,
                      route->vector_column_name);
      auto ann = duckdb::make_uniq<sirius::op::sirius_physical_vss_ann_ivf_flat>(
        sirius::from_duckdb_vec(op.types),
        std::move(*pattern),
        duckdb::NumericCast<std::size_t>(op.limit),
        duckdb::NumericCast<std::size_t>(op.offset),
        op.estimated_cardinality,
        route->sirius_context,
        std::move(route->table_name),
        std::move(route->vector_column_name),
        std::move(route->output_column_names));
      return std::move(ann);
    }

    auto vss_child = create_plan(*proj.children[0]);
    auto enn       = duckdb::make_uniq<sirius::op::sirius_physical_vss_enn>(
      sirius::from_duckdb_vec(op.types),
      std::move(*pattern),
      duckdb::NumericCast<std::size_t>(op.limit),
      duckdb::NumericCast<std::size_t>(op.offset),
      op.estimated_cardinality);
    enn->children.push_back(std::move(vss_child));
    return std::move(enn);
  }

  auto plan = create_plan(*op.children[0]);

  auto top_n = duckdb::make_uniq<sirius::op::sirius_physical_top_n>(
    sirius::from_duckdb_vec(op.types),
    std::move(op.orders),
    duckdb::NumericCast<std::size_t>(op.limit),
    duckdb::NumericCast<std::size_t>(op.offset),
    std::move(op.dynamic_filter),
    op.estimated_cardinality);

  top_n->children.push_back(std::move(plan));
  return std::move(top_n);
}

}  // namespace sirius::planner
