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

#include "meld/meld_units_pass.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/main/config.hpp"
#include "log/logging.hpp"
#include "meld/meld_plan_writer.hpp"
#include "meld/meld_ungrouped_aggregate.hpp"
#include "meld/meld_unit.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace sirius::meld {

namespace {

using op::SiriusPhysicalOperatorType;

/// Whether @p op's batches are its declared columns, in their native carriers and all there.
bool plain(op::sirius_physical_operator const& op)
{
  return !op.has_physical_overrides() && op.get_alternate_physical_types().empty() &&
         op.deferred_output().empty();
}

/// The Meld unit computing @p aggregate's per-batch partial from @p scan's batches through
/// @p chain, which runs from the operator above the scan up to the one below the aggregate.
std::unique_ptr<meld_unit> unit_for(op::sirius_physical_ungrouped_aggregate const& aggregate,
                                    std::vector<op::sirius_physical_operator*> const& chain,
                                    op::sirius_physical_operator const& scan,
                                    std::string& why)
{
  for (auto const* step : chain) {
    if (!plain(*step)) {
      why = "an operator below it holds its columns in other carriers";
      return nullptr;
    }
  }
  if (!plain(scan) || !plain(aggregate)) {
    why = "its scan holds its columns in other carriers, or defers some";
    return nullptr;
  }

  meld_chain links;
  links.input_types.assign(scan.types.begin(), scan.types.end());
  // The columns a filter keeps, which Meld is told as a projection of them.
  std::vector<std::vector<std::unique_ptr<ast::node>>> kept_columns;
  auto const* input_types = &scan.types;
  for (auto const* step : chain) {
    if (step->type == SiriusPhysicalOperatorType::FILTER) {
      auto const& filter = static_cast<op::sirius_physical_filter const&>(*step);
      links.steps.emplace_back(meld_chain::filter{filter.expression.get()});
      if (auto const* kept = std::get_if<std::vector<cudf::size_type>>(&filter.output_columns)) {
        auto& references = kept_columns.emplace_back();
        meld_chain::projection projection;
        for (auto column : *kept) {
          references.push_back(std::make_unique<ast::node>(
            ast::reference(static_cast<std::uint32_t>(column), (*input_types)[column])));
          projection.expressions.push_back(references.back().get());
        }
        links.steps.emplace_back(std::move(projection));
      }
    } else {
      auto const& projection = static_cast<op::sirius_physical_projection const&>(*step);
      meld_chain::projection projected;
      for (auto const& expression : projection.select_list) {
        projected.expressions.push_back(expression.get());
      }
      links.steps.emplace_back(std::move(projected));
    }
    input_types = &step->types;
  }
  for (auto const& expression : aggregate.aggregates) {
    if (!expression->holds<ast::aggregate>()) {
      why = "it computes something other than aggregates";
      return nullptr;
    }
    links.aggregates.push_back(&expression->get<ast::aggregate>());
  }

  auto plan = write_meld_plan(links, why);
  if (!plan) { return nullptr; }
  auto unit = meld_unit::check(plan->text, plan->column_names, why);
  if (!unit) { return nullptr; }

  std::vector<cudf::data_type> partial_types;
  for (auto const& type : aggregate.get_local_output_types()) {
    auto cudf_type = get_cudf_type(type);
    if (!cudf_type) {
      why = "Sirius has no cuDF type for its partial " + type.to_string();
      return nullptr;
    }
    partial_types.push_back(*cudf_type);
  }
  if (unit->answer_types() != partial_types) {
    why = "Meld answers at other types than Sirius's partials";
    return nullptr;
  }
  return unit;
}

/// Hands the ungrouped aggregate in @p slot to Meld, where Meld takes it, putting a
/// meld_ungrouped_aggregate in its place. @p parent holds @p slot.
void place_aggregate(op::sirius_physical_operator& parent,
                     duckdb::unique_ptr<op::sirius_physical_operator>& slot,
                     std::size_t& placed)
{
  if (slot->type != SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE || slot->children.size() != 1) {
    return;
  }
  auto& aggregate = static_cast<op::sirius_physical_ungrouped_aggregate&>(*slot);

  std::vector<op::sirius_physical_operator*> chain;
  auto* below = aggregate.children.front().get();
  while ((below->type == SiriusPhysicalOperatorType::FILTER ||
          below->type == SiriusPhysicalOperatorType::PROJECTION) &&
         below->children.size() == 1) {
    chain.push_back(below);
    below = below->children.front().get();
  }
  if (below->type != SiriusPhysicalOperatorType::TABLE_SCAN &&
      below->type != SiriusPhysicalOperatorType::GPU_SCAN) {
    return;
  }
  // Bottom up, the order Meld runs them in.
  std::reverse(chain.begin(), chain.end());

  std::string why;
  auto unit = unit_for(aggregate, chain, *below, why);
  if (!unit) {
    SIRIUS_LOG_DEBUG("[meld] an ungrouped aggregate stays with Sirius: {}", why);
    return;
  }
  SIRIUS_LOG_INFO("[meld] an ungrouped aggregate and the {} operator(s) below it run in Meld",
                  chain.size());
  if (!chain.empty()) {
    auto scan                  = std::move(chain.front()->children.front());
    aggregate.children.front() = std::move(scan);
  }
  slot = duckdb::make_uniq<meld_ungrouped_aggregate>(std::move(aggregate), std::move(unit));
  if (parent.type == SiriusPhysicalOperatorType::MERGE_AGGREGATE) {
    static_cast<op::sirius_physical_ungrouped_aggregate_merge&>(parent).child_op = slot.get();
  }
  ++placed;
}

void place(op::sirius_physical_operator& op, std::size_t& placed)
{
  if (op.type == SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    place(static_cast<op::sirius_physical_result_collector&>(op).plan, placed);
  }
  for (auto& child : op.children) {
    place(*child, placed);
    place_aggregate(op, child, placed);
  }
}

}  // namespace

void add_meld_units_option(duckdb::DBConfig& config)
{
  config.AddExtensionOption(
    "meld_units",
    "Hand an ungrouped aggregate and the filters and projections between it and its scan to Meld, "
    "which runs them as one unit over each batch on the GPU, where Meld takes them. Off by default",
    duckdb::LogicalType::BOOLEAN,
    duckdb::Value::BOOLEAN(false));
}

std::size_t place_meld_units(op::sirius_physical_operator& plan)
{
  std::size_t placed = 0;
  place(plan, placed);
  return placed;
}

}  // namespace sirius::meld
