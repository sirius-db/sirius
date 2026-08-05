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

#include "planner/late_mat_plan_pass.hpp"

#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/utils.hpp"
#include "late_mat/column_origin.hpp"
#include "late_mat/plan_deferral.hpp"
#include "log/logging.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_top_n.hpp"

#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::planner {

namespace {

using optype = op::SiriusPhysicalOperatorType;

std::unordered_set<std::size_t> expression_references(sirius::ast::node const& root)
{
  std::unordered_set<std::size_t> refs;
  sirius::ast::visit_references(root, [&](sirius::ast::reference const& r) {
    refs.insert(static_cast<std::size_t>(r.column_index));
  });
  return refs;
}

/// True when @p expr is a bare bound reference (pure positional pass-through).
bool is_bare_reference(sirius::ast::node const& expr, std::size_t& out_index);

/// v3 raw material collected during a scan's march (paired into the shared
/// FD graph by the driver): INNER-join bare-reference equality endpoints and
/// aggregate group-key provenances.
struct march_side_data {
  struct endpoint {
    op::sirius_physical_operator* join{nullptr};
    std::size_t condition{0};
    bool left_side{false};
    std::size_t scan_pos{0};
  };
  std::vector<endpoint> endpoints;
  struct key_prov {
    op::sirius_physical_operator* aggregate{nullptr};
    std::size_t input_pos{0};
    std::size_t scan_pos{0};
  };
  std::vector<key_prov> key_provs;
};

/// One tracked scan column during the upward march.
struct tracked_column {
  std::size_t scan_pos;
  std::size_t cur_pos;
  /// An outer join on the ride may have NULL-extended this position (see
  /// planned_column_deferral::nullified_on_ride for the soundness note).
  bool nullified{false};
  /// Set once the column's CONTENT consumer is recorded (index into the
  /// result's fact vector); tracking continues afterwards only to enrich
  /// group_key_at (a consumed join key can still be a planned group key).
  std::optional<std::size_t> fact_idx;
  /// Group-key reads seen BEFORE content consumption (deferred riders):
  /// moved into the fact when the final consumer is found.
  std::vector<op::sirius_physical_operator*> pending_group_keys;
};

/// March one scan's output positions up the parent chain. Records each
/// column's first CONTENT reader (group-key reads ride through per the
/// group-by-rowid bijection; join-key reads consume but keep tracking so the
/// uniqueness admission can see downstream group-key roles). Any unmodeled
/// shape consumes everything it still tracks (fail-closed: lifetimes only
/// shorten, rides only end early).
std::shared_ptr<const late_mat::planned_deferral> analyze_scan(
  op::scan::sirius_gpu_scan_operator& scan, march_side_data& side)
{
  auto result = std::make_shared<late_mat::planned_deferral>();
  std::vector<tracked_column> live;
  live.reserve(scan.get_types().size());
  for (std::size_t j = 0; j < scan.get_types().size(); ++j) {
    live.push_back({j, j});
  }

  std::size_t crossings = 0;

  // Record the content consumer for a still-unconsumed column.
  auto consume = [&](tracked_column& col,
                     op::sirius_physical_operator* consumer,
                     bool count_only = false) {
    if (col.fact_idx.has_value()) { return; }
    late_mat::planned_column_deferral fact;
    fact.scan_output_position   = col.scan_pos;
    fact.consumer               = consumer;
    fact.final_position         = col.cur_pos;
    fact.consumed_as_count_only = count_only;
    fact.nullified_on_ride      = col.nullified;
    fact.group_key_at           = std::move(col.pending_group_keys);
    fact.crossings              = crossings;
    col.fact_idx                = result->columns.size();
    result->columns.push_back(std::move(fact));
  };
  auto mark_group_key = [&](tracked_column& col, op::sirius_physical_operator* agg) {
    if (col.fact_idx.has_value()) {
      result->columns[*col.fact_idx].group_key_at.push_back(agg);
    } else {
      col.pending_group_keys.push_back(agg);
    }
  };
  // A column leaving tracking unconsumed simply has no fact (dropped).

  op::sirius_physical_operator const* cur = &scan;
  op::sirius_physical_operator* parent    = scan.get_parent_op();
  for (int hops = 0; hops < 256 && parent != nullptr && !live.empty(); ++hops) {
    switch (parent->type) {
      case optype::DYNAMIC_FILTER:
      case optype::PARTITION:
      case optype::CONCAT: break;  // positionally transparent plumbing

      case optype::FILTER: {
        auto const* filter = dynamic_cast<op::sirius_physical_filter const*>(parent);
        if (filter == nullptr || !filter->expression) {
          for (auto& col : live) { consume(col, parent); }
          live.clear();
          break;
        }
        auto const refs = expression_references(*filter->expression);
        for (auto& col : live) {
          if (refs.contains(col.cur_pos)) { consume(col, parent); }
        }
        break;  // filter output layout == input layout; tracking continues
      }

      case optype::PROJECTION: {
        auto const* proj = dynamic_cast<op::sirius_physical_projection const*>(parent);
        if (proj == nullptr) {
          for (auto& col : live) { consume(col, parent); }
          live.clear();
          break;
        }
        std::unordered_map<std::size_t, std::size_t> passthrough;  // input -> output
        std::unordered_set<std::size_t> consumed_inputs;
        for (std::size_t i = 0; i < proj->select_list.size(); ++i) {
          if (!proj->select_list[i]) { continue; }
          std::size_t in_pos = 0;
          if (is_bare_reference(*proj->select_list[i], in_pos)) {
            passthrough.emplace(in_pos, i);
          } else {
            for (auto const r : expression_references(*proj->select_list[i])) {
              consumed_inputs.insert(r);
            }
          }
        }
        std::vector<tracked_column> next;
        for (auto& col : live) {
          if (consumed_inputs.contains(col.cur_pos)) { consume(col, parent); }
          auto const it = passthrough.find(col.cur_pos);
          if (it != passthrough.end()) {
            col.cur_pos = it->second;
            next.push_back(std::move(col));
          }
          // else: not forwarded — tracking ends (fact, if any, is final).
        }
        live = std::move(next);
        break;
      }

      case optype::HASH_JOIN: {
        auto const* join = dynamic_cast<op::sirius_physical_hash_join const*>(parent);
        bool const from_left =
          !parent->children.empty() && parent->children[0].get() == cur;
        bool const passthrough_type = join != nullptr &&
                                      (join->join_type == duckdb::JoinType::INNER ||
                                       join->join_type == duckdb::JoinType::LEFT ||
                                       join->join_type == duckdb::JoinType::RIGHT ||
                                       join->join_type == duckdb::JoinType::OUTER);
        if (!passthrough_type) {
          for (auto& col : live) { consume(col, parent); }
          live.clear();
          break;
        }
        bool const side_nullified =
          join->join_type == duckdb::JoinType::OUTER ||
          (from_left && join->join_type == duckdb::JoinType::RIGHT) ||
          (!from_left && join->join_type == duckdb::JoinType::LEFT);
        std::unordered_set<std::size_t> key_refs;
        for (auto const& cond : join->conditions) {
          auto const& side = from_left ? cond.left : cond.right;
          if (!side) { continue; }
          for (auto const r : expression_references(*side)) { key_refs.insert(r); }
        }
        // v3 equality endpoints: INNER-join bare-reference equality sides only
        // (anything else contributes no edge — fail-closed, the affected key
        // simply rides real).
        if (late_mat::late_mat_v3_enabled() && join->join_type == duckdb::JoinType::INNER) {
          for (std::size_t ci = 0; ci < join->conditions.size(); ++ci) {
            auto const& cond = join->conditions[ci];
            if (cond.comparison != sirius::comparison_type::equal) { continue; }
            auto const& side_expr = from_left ? cond.left : cond.right;
            std::size_t ref       = 0;
            if (!side_expr || !is_bare_reference(*side_expr, ref)) { continue; }
            for (auto const& col : live) {
              if (col.cur_pos == ref && !col.nullified) {
                side.endpoints.push_back({parent, ci, from_left, col.scan_pos});
              }
            }
          }
        }
        auto const& own  = from_left ? join->lhs_output_columns.col_idxs
                                     : join->rhs_output_columns.col_idxs;
        std::size_t const base = from_left ? 0 : join->lhs_output_columns.col_idxs.size();
        std::vector<tracked_column> next;
        for (auto& col : live) {
          // A key read consumes the content but the column usually still
          // rides the projection — keep tracking for group-key enrichment.
          if (key_refs.contains(col.cur_pos)) { consume(col, parent); }
          auto const it =
            std::find(own.begin(), own.end(), static_cast<cudf::size_type>(col.cur_pos));
          if (it == own.end()) { continue; }  // not projected — tracking ends
          col.cur_pos = base + static_cast<std::size_t>(std::distance(own.begin(), it));
          col.nullified = col.nullified || side_nullified;
          next.push_back(std::move(col));
        }
        live = std::move(next);
        ++crossings;
        break;
      }

      case optype::HASH_GROUP_BY:
      case optype::MERGE_GROUP_BY: {
        // Both group-by shapes: keys ride through (group output position i for
        // group_idx[i], groups-then-aggregates output layout); a group-key read
        // is NOT content consumption (the §4-addendum bijection), only a
        // marker. Non-key aggregate inputs consume; nothing else survives.
        std::vector<int> const* group_idx = nullptr;
        std::unordered_set<std::size_t> agg_inputs;
        std::unordered_set<std::size_t> non_count_inputs;
        if (auto const* agg =
              dynamic_cast<op::sirius_physical_grouped_aggregate const*>(parent)) {
          group_idx = &agg->group_idx;
          for (std::size_t a = 0; a < agg->cudf_aggregate_idx.size(); ++a) {
            auto const idx = agg->cudf_aggregate_idx[a];
            if (idx < 0) { continue; }
            agg_inputs.insert(static_cast<std::size_t>(idx));
            bool const is_count =
              a < agg->cudf_aggregates.size() &&
              agg->cudf_aggregates[a] == cudf::aggregation::Kind::COUNT_VALID;
            if (!is_count) { non_count_inputs.insert(static_cast<std::size_t>(idx)); }
          }
          for (auto const& cols : agg->cudf_aggregate_struct_col_indices) {
            for (auto const a : cols) {
              if (a >= 0) {
                agg_inputs.insert(static_cast<std::size_t>(a));
                non_count_inputs.insert(static_cast<std::size_t>(a));
              }
            }
          }
        } else if (auto const* merge =
                     dynamic_cast<op::sirius_physical_grouped_aggregate_merge const*>(parent)) {
          group_idx = &merge->group_idx;
          // Merge-side aggregate inputs: consume-all for tracked non-keys
          // below (conservative; the merge re-reads only aggregate partials,
          // which are never scan pass-throughs anyway).
        }
        if (group_idx == nullptr) {
          for (auto& col : live) { consume(col, parent); }
          live.clear();
          break;
        }
        std::vector<tracked_column> next;
        for (auto& col : live) {
          auto const key_it = std::find(group_idx->begin(), group_idx->end(),
                                        static_cast<int>(col.cur_pos));
          if (key_it != group_idx->end()) {
            mark_group_key(col, parent);
            if (late_mat::late_mat_v3_enabled() && !col.nullified) {
              side.key_provs.push_back({parent, col.cur_pos, col.scan_pos});
            }
            col.cur_pos =
              static_cast<std::size_t>(std::distance(group_idx->begin(), key_it));
            next.push_back(std::move(col));
            continue;
          }
          if (agg_inputs.contains(col.cur_pos)) {
            consume(col, parent, /*count_only=*/!non_count_inputs.contains(col.cur_pos));
          } else if (parent->type == optype::MERGE_GROUP_BY) {
            consume(col, parent);  // conservative for unmodeled merge reads
          }
          // tracking ends: aggregate output carries no non-key input column
        }
        live = std::move(next);
        ++crossings;
        break;
      }

      case optype::TOP_N: {
        auto const* topn = dynamic_cast<op::sirius_physical_top_n const*>(parent);
        bool modeled     = topn != nullptr;
        std::unordered_set<std::size_t> sort_refs;
        if (modeled) {
          for (auto const& ord : topn->orders) {
            if (!ord.expression ||
                ord.expression->GetExpressionClass() !=
                  duckdb::ExpressionClass::BOUND_REF) {
              modeled = false;
              break;
            }
            sort_refs.insert(static_cast<std::size_t>(
              ord.expression->Cast<duckdb::BoundReferenceExpression>().index));
          }
        }
        if (!modeled) {
          for (auto& col : live) { consume(col, parent); }
          live.clear();
          break;
        }
        for (auto& col : live) {
          if (sort_refs.contains(col.cur_pos)) { consume(col, parent); }
        }
        break;  // TOP_N gathers whole rows: positions unchanged, payload rides
      }

      default: {
        for (auto& col : live) { consume(col, parent); }
        live.clear();
        break;
      }
    }
    cur    = parent;
    parent = parent->get_parent_op();
  }
  return result->columns.empty() ? nullptr : result;
}

void collect_scans(op::sirius_physical_operator& node,
                   std::vector<op::scan::sirius_gpu_scan_operator*>& out)
{
  if (auto* scan = dynamic_cast<op::scan::sirius_gpu_scan_operator*>(&node)) {
    out.push_back(scan);
  }
  for (auto& child : node.children) {
    if (child) { collect_scans(*child, out); }
  }
}

bool is_bare_reference(sirius::ast::node const& expr, std::size_t& out_index)
{
  bool bare        = false;
  std::size_t refs = 0;
  sirius::ast::visit_references(expr, [&](sirius::ast::reference const& r) {
    ++refs;
    out_index = static_cast<std::size_t>(r.column_index);
  });
  if (refs != 1) { return false; }
  // Exactly one reference AND the node itself is that reference (not e.g. a
  // cast around it): check the variant alternative directly.
  bare = std::holds_alternative<sirius::ast::reference>(expr.v);
  return bare;
}

}  // namespace

void run_late_mat_plan_pass(op::sirius_physical_operator& root)
{
  if (!late_mat::late_mat_v2_enabled()) { return; }
  std::vector<op::scan::sirius_gpu_scan_operator*> scans;
  collect_scans(root, scans);
  std::vector<std::pair<op::scan::sirius_gpu_scan_operator*, march_side_data>> side_data;
  side_data.reserve(scans.size());
  for (auto* scan : scans) {
    march_side_data side;
    scan->late_mat_plan = analyze_scan(*scan, side);
    side_data.emplace_back(scan, std::move(side));
  }
  // v3: pair equality endpoints into the query-wide FD graph and collect the
  // aggregate key provenances; the lowering runs the determination closure
  // against the pinned entries' uniqueness facts.
  if (late_mat::late_mat_v3_enabled() && !scans.empty()) {
    auto graph = std::make_shared<late_mat::planned_fd_graph>();
    struct half {
      op::scan::sirius_gpu_scan_operator* scan;
      std::size_t scan_pos;
    };
    std::map<std::pair<op::sirius_physical_operator*, std::size_t>, std::pair<std::optional<half>, std::optional<half>>>
      by_condition;
    for (auto& [scan, side] : side_data) {
      for (auto const& ep : side.endpoints) {
        auto& slot = by_condition[{ep.join, ep.condition}];
        auto& mine = ep.left_side ? slot.first : slot.second;
        if (!mine) { mine = half{scan, ep.scan_pos}; }
      }
      for (auto const& kp : side.key_provs) {
        graph->key_provenances.push_back({kp.aggregate, kp.input_pos, scan, kp.scan_pos});
      }
    }
    for (auto const& [key, halves] : by_condition) {
      if (halves.first && halves.second) {
        graph->edges.push_back({halves.first->scan,
                                halves.first->scan_pos,
                                halves.second->scan,
                                halves.second->scan_pos,
                                key.first});
      }
    }
    SIRIUS_LOG_DEBUG("[late_mat v3] fd graph: {} edge(s), {} key provenance(s)",
                     graph->edges.size(),
                     graph->key_provenances.size());
    for (auto* scan : scans) {
      scan->late_mat_fd_graph = graph;
    }
  }
  for (auto* scan : scans) {
    if (scan->late_mat_plan) {
      // NOTE: operator ids are not assigned yet at plan time — identify by
      // address; the lowering logs the id-bearing install/reject lines later.
      SIRIUS_LOG_DEBUG("[late_mat v2] plan pass: scan {} -> {} lifetime fact(s)",
                       static_cast<void const*>(scan),
                       scan->late_mat_plan->columns.size());
    }
  }
}

}  // namespace sirius::planner
