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

#include "cudf/cudf_utils.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "helper/type_conversions.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/top_n_dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/top_n_group_key_producer.hpp"
#include "op/dynamic_filter/top_n_threshold_coordinator.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "planner/dynamic_filter/dynamic_filter_target_discovery.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/top_n_checked_k.hpp"
#include "planner/top_n_key_types.hpp"
#include "sirius_context.hpp"

#include <cudf/types.hpp>

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

/// The allowlist itself lives in `planner/top_n_key_types.hpp` so its bands can be asserted
/// directly -- several of them are unreachable through a built plan (see that header).
using sirius::planner::admitted_top_n_key_storage_type;

/// Checked K and the frozen per-key semantics of an admitted Top-N producer, shared by both
/// admission paths (the row producer and the Stage-5 group-key producer) so they can never
/// disagree about which orders are admissible or about K.
struct admitted_top_n_keys {
  std::size_t k = 0;
  std::vector<sirius::op::top_n_key_semantics> keys;
  bool lex_admitted = true;
};

/// Validate the Top-N producer admission list (main doc, "Immutable Top-N publication plan"), or
/// return empty when this Top-N can carry no producer at all. Key zero's type must be admitted; an
/// unadmitted tail type only degrades the producer to the first-key layer.
std::optional<admitted_top_n_keys> admit_top_n_keys(duckdb::LogicalTopN const& op)
{
  // Checked K = limit + offset: never wrap or truncate, and K must stay representable in the
  // cuDF row index space (main doc, "Limit and offset").
  auto const k = checked_top_n_k(duckdb::NumericCast<std::size_t>(op.limit),
                                 duckdb::NumericCast<std::size_t>(op.offset));
  if (!k || op.orders.empty()) { return std::nullopt; }

  admitted_top_n_keys admitted{.k = *k, .keys = {}, .lex_admitted = true};
  admitted.keys.reserve(op.orders.size());
  for (auto const& order : op.orders) {
    // Mirrors the operator's own bound-reference requirement; non-reference keys fall back to
    // DuckDB before execution, but the rejection counter stays honest if that support widens.
    if (order.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
      return std::nullopt;
    }
    auto const storage = admitted_top_n_key_storage_type(order.expression->return_type);
    if (!storage && admitted.keys.empty()) { return std::nullopt; }
    admitted.lex_admitted = admitted.lex_admitted && storage.has_value();
    // A non-admitted tail's storage type is never consulted: witness extraction records such
    // components as disengaged, and uniformly disengaged components compare equal across offers.
    // Admitted tails after such a gap still participate in tightness, so degraded-mode tightness
    // is pseudo-lexicographic (the true order would consult the gap first); pruning stays sound
    // because the degraded prefilter reads only component 0, which is compared first and can
    // therefore never loosen.
    admitted.keys.push_back(
      {.storage_type = storage.value_or(cudf::data_type{cudf::type_id::EMPTY}),
       .order        = sirius::op::to_cudf_order(order.type),
       .null_order   = sirius::op::to_cudf_null_order(order.type, order.null_order)});
  }
  return admitted;
}

/// Construct the row producer's execution coordinator for sink self-consumption, or return null
/// counting the rejection.
std::shared_ptr<sirius::op::top_n_threshold_coordinator> make_threshold_coordinator(
  duckdb::LogicalTopN const& op, sirius::op::dynamic_filter_stats* stats)
{
  auto admitted = admit_top_n_keys(op);
  if (!admitted) {
    if (stats) { stats->top_n_producers_rejected.fetch_add(1, std::memory_order_relaxed); }
    return nullptr;
  }

  if (stats) {
    stats->top_n_producers_eligible.fetch_add(1, std::memory_order_relaxed);
    if (!admitted->lex_admitted) {
      stats->top_n_producers_first_key_only.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return std::make_shared<sirius::op::top_n_threshold_coordinator>(
    admitted->k, std::move(admitted->keys), admitted->lex_admitted, stats);
}

/// Per-key allowlist verdict, in key order -- the field's documented meaning, as distinct from
/// the global AND that gates the LEX layer.
std::vector<bool> top_n_key_admissions(duckdb::LogicalTopN const& op)
{
  std::vector<bool> admitted;
  admitted.reserve(op.orders.size());
  for (auto const& order : op.orders) {
    admitted.push_back(admitted_top_n_key_storage_type(order.expression->return_type).has_value());
  }
  return admitted;
}

/// The ORDER BY keys' ordinals in the Top-N child's output space, key zero first.
/// @pre Every key is a bound reference -- eligibility enforced it before any caller runs, which
/// is what makes the unchecked Cast below safe.
std::vector<std::size_t> top_n_key_ordinals(duckdb::LogicalTopN const& op)
{
  std::vector<std::size_t> ordinals;
  ordinals.reserve(op.orders.size());
  for (auto const& order : op.orders) {
    ordinals.push_back(
      static_cast<std::size_t>(order.expression->Cast<duckdb::BoundReferenceExpression>().index));
  }
  return ordinals;
}

/// Every active GPU space paired with its local host staging space, covering each device a
/// published filter must be replicated to (scan consumers and sited endpoints alike).
std::vector<sirius::op::dynamic_filter_replica_space> collect_replica_spaces(
  duckdb::SiriusContext& sirius_context)
{
  auto& memory_manager   = sirius_context.get_memory_manager();
  auto const gpu_spaces  = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto const host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  std::vector<sirius::op::dynamic_filter_replica_space> spaces;
  if (host_spaces.empty()) { return spaces; }
  spaces.reserve(gpu_spaces.size());
  for (auto const* gpu_view : gpu_spaces) {
    auto* gpu_space =
      memory_manager.get_memory_space(cucascade::memory::Tier::GPU, gpu_view->get_device_id());
    if (gpu_space == nullptr) { continue; }
    auto const local_host =
      std::find_if(host_spaces.begin(), host_spaces.end(), [gpu_space](auto const* host_space) {
        return host_space->get_device_id() == gpu_space->get_device_id();
      });
    auto const* host_space = local_host == host_spaces.end() ? host_spaces.front() : *local_host;
    spaces.emplace_back(*gpu_space, *host_space);
  }
  return spaces;
}

/// Attach a channel to @p scan if it has none, then register one refinement slot for this
/// producer. Bottom-up construction means an inner producer's scan may already carry a channel.
sirius::op::dynamic_filter_refinement_publisher register_scan_slot(
  sirius::op::sirius_physical_table_scan& scan, std::vector<std::size_t> const& ordinals)
{
  if (!scan.sirius_dynamic_filters) {
    scan.sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  }
  std::vector<std::size_t> referenced{ordinals.begin() + 1, ordinals.end()};
  return scan.sirius_dynamic_filters->register_refinement_slot(ordinals.front(),
                                                               std::move(referenced));
}

/**
 * @brief Whether every key of @p ordinals may be published at @p site
 *
 * Applies @ref sirius::planner::boundary_key_matches_site_type per component, mapping the site's
 * own declared column type into cuDF's space. A fixed-point key whose scale does not match the
 * column it would bind refuses that site: the boundary would be compared as a raw integer against
 * a differently scaled one, which prunes wrong rows and reports nothing. The producer keeps
 * self-consuming; only this target is dropped.
 *
 * The site's declared type is the DuckDB catalog type Sirius carries on every operator. That the
 * decoded GPU column matches it is an engine-wide invariant this check relies on rather than
 * re-establishes -- every operator would misread the column otherwise.
 */
bool site_admits_keys(sirius::op::sirius_physical_operator const& site,
                      std::vector<std::size_t> const& ordinals,
                      std::span<sirius::op::top_n_key_semantics const> keys)
{
  for (std::size_t i = 0; i < ordinals.size() && i < keys.size(); ++i) {
    if (ordinals[i] >= site.types.size()) { return false; }
    cudf::data_type site_type{cudf::type_id::EMPTY};
    try {
      site_type = sirius::get_cudf_type(site.types[ordinals[i]]);
    } catch (std::exception const&) {
      return false;  // a type Sirius cannot map cannot be proven to match
    }
    if (!boundary_key_matches_site_type(keys[i].storage_type, site_type)) { return false; }
  }
  return true;
}

/**
 * @brief Run both Top-N traces over the built child, bind targets, and freeze the publish plan
 *
 * The all-keys trace carries the strict full-tuple predicate to sites where every key coexists;
 * the key-zero trace carries the inclusive first-key bound further upstream (a single-key
 * producer's key-zero target carries its whole strict predicate instead, since `LEX_RANGE`
 * requires two components). A material non-scan LEX terminal is spliced as a LEX endpoint via
 * `place_endpoint_all_keys`, carrying every component ordinal to one arrive-together site. A
 * first-key site coinciding with a LEX site is dropped as subsumed. Returns the child, rewrapped
 * when an endpoint was spliced.
 */
duckdb::unique_ptr<sirius::op::sirius_physical_operator> discover_top_n_targets(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  std::vector<std::size_t> const& key_ordinals,
  std::vector<bool> const& key_admissions,
  sirius::op::top_n_threshold_coordinator& coordinator,
  duckdb::SiriusContext& sirius_context,
  double gate_keep_threshold,
  sirius::op::dynamic_filter_stats& stats)
{
  using sirius::op::top_n_filter_layer;

  sirius::op::top_n_dynamic_filter_publish_plan publish_plan;
  publish_plan.k = coordinator.k();
  // The coordinator is the authority on the witness discipline; the plan only records it, and a
  // `GROUP_KEY` plan differs from a `ROW` one solely in publishing inclusive predicates and in
  // being rooted below an aggregate -- targets, replica spaces, and slots are identical.
  publish_plan.kind = coordinator.kind();
  publish_plan.keys.reserve(coordinator.keys().size());
  for (std::size_t i = 0; i < coordinator.keys().size(); ++i) {
    publish_plan.keys.push_back({.child_ordinal = key_ordinals[i],
                                 .semantics     = coordinator.keys()[i],
                                 .type_admitted = key_admissions[i]});
  }
  publish_plan.replica_spaces = collect_replica_spaces(sirius_context);

  // Under `top_n_self_trace` every accepted hop descends into child 0 -- set operations are
  // refused, and no other accepted operator fans out -- so each trace yields exactly one terminal.
  // The single-branch reasoning below (one scan-bind decision, one endpoint placement) depends on
  // that; revisit it if a fan-out hop is ever admitted.
  descent_policy const policy{.descend_build_blocks = false, .top_n_self_trace = true};
  auto const multi_key = key_ordinals.size() >= 2 && coordinator.lex_admitted();

  // --- All-keys trace: the LEX layer ---
  std::unordered_set<sirius::op::sirius_physical_operator const*> lex_sites;
  std::optional<multi_key_route_terminal> lex_endpoint_terminal;
  if (multi_key) {
    for (auto const& terminal : trace_top_n_all_keys(*child, key_ordinals, policy)) {
      auto const kind = classify_top_n_terminal({.node = terminal.node, .ordinal = 0},
                                                top_n_filter_layer::LEX,
                                                terminal.material_hops,
                                                target_skips_reads(*terminal.node));
      if ((kind == top_n_target_kind::SCAN_BIND || kind == top_n_target_kind::ENDPOINT_SITE) &&
          !site_admits_keys(*terminal.node, terminal.ordinals, coordinator.keys())) {
        // The site's columns do not carry the keys' exact types; publishing there could compare
        // differently scaled fixed-point values. Skip the target, keep the producer. This shares
        // the siting rule's counter: no target is created either way.
        stats.top_n_sites_skipped_no_work_saved.fetch_add(1, std::memory_order_relaxed);
      } else if (kind == top_n_target_kind::SCAN_BIND) {
        auto& scan = terminal.node->Cast<sirius::op::sirius_physical_table_scan>();
        publish_plan.targets.push_back({.publisher = register_scan_slot(scan, terminal.ordinals),
                                        .layer     = top_n_filter_layer::LEX,
                                        .component_ordinals = terminal.ordinals});
        lex_sites.insert(terminal.node);
        stats.top_n_lex_scan_targets.fetch_add(1, std::memory_order_relaxed);
      } else if (kind == top_n_target_kind::ENDPOINT_SITE) {
        // The splice runs after the key-zero classification below, so the site enters `lex_sites`
        // now: the coinciding first-key terminal must classify as subsumed.
        lex_endpoint_terminal = terminal;
        lex_sites.insert(terminal.node);
      } else {
        stats.top_n_sites_skipped_no_work_saved.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  // --- Key-zero trace: the first-key layer ---
  std::span<std::size_t const> const key_zero{key_ordinals.data(), 1};
  auto const key_zero_terminals = trace_top_n_all_keys(*child, key_zero, policy);
  assert(key_zero_terminals.size() <= 1);  // single-terminal invariant; see the policy comment

  // Classify each terminal once: the verdict depends only on the terminal, the layer, and the LEX
  // sites, all of which are already settled.
  std::vector<top_n_target_kind> key_zero_kinds;
  key_zero_kinds.reserve(key_zero_terminals.size());
  for (auto const& terminal : key_zero_terminals) {
    auto kind =
      classify_top_n_terminal({.node = terminal.node, .ordinal = terminal.ordinals.front()},
                              top_n_filter_layer::FIRST_KEY,
                              terminal.material_hops,
                              target_skips_reads(*terminal.node),
                              lex_sites.count(terminal.node) != 0);
    // The first-key layer binds key zero alone, so only key zero's type must match this site. A
    // refusal here shares the siting rule's counter: no target is created either way.
    if ((kind == top_n_target_kind::SCAN_BIND || kind == top_n_target_kind::ENDPOINT_SITE) &&
        !site_admits_keys(
          *terminal.node, {terminal.ordinals.front()}, coordinator.keys().first(1))) {
      kind = top_n_target_kind::SKIPPED_NO_WORK_SAVED;
    }
    key_zero_kinds.push_back(kind);
  }

  // `place_endpoint` must not run for a key that scan-bound any branch. Under the Top-N policy a
  // trace yields exactly one terminal (see the single-terminal invariant above), so this is a
  // single yes/no rather than a per-branch question.
  bool key_zero_bound_at_scan = false;
  for (std::size_t i = 0; i < key_zero_terminals.size(); ++i) {
    auto const& terminal = key_zero_terminals[i];
    switch (key_zero_kinds[i]) {
      case top_n_target_kind::SUBSUMED_BY_LEX:
        stats.top_n_first_key_subsumed_by_lex.fetch_add(1, std::memory_order_relaxed);
        break;
      case top_n_target_kind::SKIPPED_NO_WORK_SAVED:
        stats.top_n_sites_skipped_no_work_saved.fetch_add(1, std::memory_order_relaxed);
        break;
      case top_n_target_kind::SCAN_BIND: {
        auto& scan = terminal.node->Cast<sirius::op::sirius_physical_table_scan>();
        publish_plan.targets.push_back({.publisher = register_scan_slot(scan, terminal.ordinals),
                                        .layer     = top_n_filter_layer::FIRST_KEY,
                                        .component_ordinals = terminal.ordinals});
        key_zero_bound_at_scan = true;
        stats.top_n_first_key_scan_targets.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      default: break;
    }
  }

  // --- LEX endpoint splice ---
  // Runs before the first-key splice: a deeper first-key endpoint descends through the fresh LEX
  // endpoint transparently (DYNAMIC_FILTER is a pass-through hop) and still lands below it. In
  // the coinciding case the first-key splice never runs at all -- SUBSUMED_BY_LEX above.
  if (lex_endpoint_terminal.has_value()) {
    std::vector<std::shared_ptr<sirius::op::sirius_dynamic_filter_set>> site_channels;
    auto placed = place_endpoint_all_keys(
      std::move(child),
      key_ordinals,
      policy,
      [&site_channels, &stats, gate_keep_threshold](
        sirius::op::sirius_physical_operator const& site)
        -> duckdb::unique_ptr<sirius::op::sirius_physical_operator> {
        auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
        auto endpoint = duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
          site.types,
          site.estimated_cardinality,
          channel,
          gate_keep_threshold,
          sirius::op::scan::dynamic_filter_apply_mode::include_ast_row_masks,
          sirius::op::scan::dynamic_filter_endpoint_provenance::top_n_endpoint,
          &stats);
        site_channels.push_back(std::move(channel));
        return endpoint;
      });
    child = std::move(placed.subtree);
    // Trace and splice are two runs of one pure hop function over a subtree nothing mutated in
    // between, so they cannot land apart; assert that loudly rather than assuming it.
    assert(placed.site_ordinals.size() == 1);
    assert(placed.site_ordinals.front() == lex_endpoint_terminal->ordinals);
    for (std::size_t site = 0; site < site_channels.size(); ++site) {
      auto const& ordinals = placed.site_ordinals[site];
      std::vector<std::size_t> referenced{ordinals.begin() + 1, ordinals.end()};
      publish_plan.targets.push_back({.publisher = site_channels[site]->register_refinement_slot(
                                        ordinals.front(), std::move(referenced)),
                                      .layer              = top_n_filter_layer::LEX,
                                      .component_ordinals = ordinals});
      stats.top_n_lex_endpoint_sites_placed.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Endpoint siting for the first-key layer. `place_endpoint` splices every reached branch, so it
  // runs only when no branch bound key zero at a scan.
  auto const wants_endpoint =
    !key_zero_bound_at_scan &&
    std::any_of(key_zero_kinds.begin(), key_zero_kinds.end(), [](top_n_target_kind kind) {
      return kind == top_n_target_kind::ENDPOINT_SITE;
    });
  if (wants_endpoint) {
    std::vector<std::shared_ptr<sirius::op::sirius_dynamic_filter_set>> site_channels;
    auto placed = place_endpoint(
      std::move(child),
      key_ordinals.front(),
      policy,
      [&site_channels, &stats, gate_keep_threshold](
        sirius::op::sirius_physical_operator const& site)
        -> duckdb::unique_ptr<sirius::op::sirius_physical_operator> {
        auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
        auto endpoint = duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
          site.types,
          site.estimated_cardinality,
          channel,
          gate_keep_threshold,
          sirius::op::scan::dynamic_filter_apply_mode::include_ast_row_masks,
          sirius::op::scan::dynamic_filter_endpoint_provenance::top_n_endpoint,
          &stats);
        site_channels.push_back(std::move(channel));
        return endpoint;
      });
    child = std::move(placed.subtree);
    for (std::size_t site = 0; site < site_channels.size(); ++site) {
      std::vector<std::size_t> const ordinals{placed.site_ordinals[site]};
      publish_plan.targets.push_back(
        {.publisher          = site_channels[site]->register_refinement_slot(ordinals.front()),
         .layer              = top_n_filter_layer::FIRST_KEY,
         .component_ordinals = ordinals});
      stats.top_n_first_key_endpoint_sites_placed.fetch_add(1, std::memory_order_relaxed);
    }
  }

  coordinator.set_publish_plan(std::move(publish_plan));
  return child;
}

//===----------------------------------------------------------------------===//
// Group-key producer admission (Stage 5)
//===----------------------------------------------------------------------===//
// A separate admission path, never a relaxation of `descent_policy::top_n_self_trace`. That trace
// refuses aggregates on the hop-set bit itself, independent of producer kind, and must keep doing
// so: a row boundary is strict and row-level, so publishing it against grouped input would drop
// rows of groups that are in the answer and silently corrupt their aggregate values. The group-key
// producer earns its crossing by rooting *below* the aggregate and publishing inclusive-only
// predicates over distinct grouping keys.

/// A grouped aggregate reachable below a Top-N, and where the ORDER BY keys land in its input.
struct group_key_admission {
  sirius::op::sirius_physical_grouped_aggregate* aggregate = nullptr;
  std::vector<std::size_t> input_ordinals;  ///< In the aggregate's input space, key zero first
  bool refused = false;  ///< The shape is a candidate, but something disqualifies it
};

/**
 * @brief Find the grouped aggregate a Top-N's ORDER BY keys all resolve into, if there is one
 *
 * Walks the built child chain through pass-through hops, remapping every key ordinal with the same
 * `descent_steps` rules the traces use, and resolves each ordinal at the aggregate through its
 * `group_idx` -- an aggregate-*output* key can never qualify, because no input row's value
 * determines it. A `FILTER` between the Top-N and the aggregate disqualifies the shape: DuckDB
 * lowers `HAVING` to exactly that, and a predicate there can eliminate whole groups, so a witnessed
 * distinct key no longer proves a surviving group. The walk continues past such a filter only to
 * learn whether an aggregate is there at all, so the rejection can be counted rather than silently
 * looking like an unrelated query shape.
 *
 * @return The aggregate and its remapped ordinals, or empty when no aggregate is reachable -- the
 * ordinary case of a Top-N that is simply not this producer's shape
 */
std::optional<group_key_admission> find_group_key_aggregate(
  sirius::op::sirius_physical_operator& child, std::vector<std::size_t> ordinals)
{
  using sirius::op::SiriusPhysicalOperatorType;

  auto* node   = &child;
  bool refused = false;
  while (node != nullptr) {
    if (node->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
      auto& aggregate = node->Cast<sirius::op::sirius_physical_grouped_aggregate>();
      group_key_admission admission{.aggregate = &aggregate, .input_ordinals = {}, .refused = true};
      if (aggregate.children.size() != 1 || aggregate.children[0] == nullptr) { return admission; }
      std::vector<std::size_t> inputs;
      inputs.reserve(ordinals.size());
      for (auto const ordinal : ordinals) {
        auto const input =
          group_by_key_input(aggregate.group_idx, aggregate.grouping_sets.size(), ordinal);
        if (!input) { return admission; }
        inputs.push_back(*input);
      }
      admission.input_ordinals = std::move(inputs);
      admission.refused        = refused;
      return admission;
    }

    switch (node->type) {
      case SiriusPhysicalOperatorType::FILTER: refused = true; [[fallthrough]];
      case SiriusPhysicalOperatorType::PROJECTION:
      case SiriusPhysicalOperatorType::DYNAMIC_FILTER: break;
      default: return std::nullopt;
    }

    // The hop is accepted; remap every ordinal with the shared rules, under the same restricted
    // hop set the Top-N traces use. The three node types reached here treat both policies alike,
    // so the choice is about not depending on that.
    descent_policy const policy{.descend_build_blocks = false, .top_n_self_trace = true};
    std::vector<std::size_t> next;
    next.reserve(ordinals.size());
    for (auto const ordinal : ordinals) {
      auto const steps = descent_steps(*node, ordinal, policy);
      if (steps.size() != 1 || steps.front().child_index != 0) { return std::nullopt; }
      next.push_back(steps.front().child_ordinal);
    }
    if (node->children.size() != 1 || node->children[0] == nullptr) { return std::nullopt; }
    ordinals = std::move(next);
    node     = node->children[0].get();
  }
  return std::nullopt;
}

/**
 * @brief Install the group-key producer on the aggregate below @p child, when the shape admits one
 *
 * Runs after the row producer's discovery, over the same built child, and mutates only the
 * aggregate's own subtree -- so it never changes where the row producer bound or sited anything.
 */
void install_group_key_producer(duckdb::LogicalTopN const& op,
                                sirius::op::sirius_physical_operator& child,
                                duckdb::SiriusContext& sirius_context,
                                double gate_keep_threshold)
{
  auto& stats = sirius_context.get_dynamic_filter_stats();
  for (auto const& order : op.orders) {
    // Without bound references there are no ordinals to walk. Such a Top-N carries no producer of
    // either kind, and the row path already counted that rejection.
    if (order.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) { return; }
  }

  auto const admission = find_group_key_aggregate(child, top_n_key_ordinals(op));
  if (!admission) { return; }  // not a group-key shape; neither counter moves
  // Defensive: an intervening `TOP_N` ends the walk, so no two Top-N operators can reach the same
  // aggregate today. Refusing rather than replacing keeps that from becoming a silent loss of the
  // first producer if the hop set ever widens.
  if (admission->aggregate->top_n_producer != nullptr) { return; }

  // The K cap belongs to admission, not to the seam: see
  // `top_n_group_key_producer::k_max_admitted_k` for why a large K is refused rather than made
  // cheap. It shares the rejection counter with the structural refusals above.
  auto admitted = admit_top_n_keys(op);
  if (admission->refused || !admitted ||
      admitted->k > sirius::op::top_n_group_key_producer::k_max_admitted_k) {
    stats.top_n_group_producers_rejected.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  stats.top_n_group_producers_eligible.fetch_add(1, std::memory_order_relaxed);

  auto coordinator = std::make_shared<sirius::op::top_n_threshold_coordinator>(
    admitted->k,
    std::move(admitted->keys),
    admitted->lex_admitted,
    &stats,
    sirius::op::top_n_producer_kind::GROUP_KEY);

  // The traces root at the aggregate's input, where the keys are still per-row values, and follow
  // the same hop rules and terminal classification as the row producer's.
  auto& aggregate       = *admission->aggregate;
  aggregate.children[0] = discover_top_n_targets(std::move(aggregate.children[0]),
                                                 admission->input_ordinals,
                                                 top_n_key_admissions(op),
                                                 *coordinator,
                                                 sirius_context,
                                                 gate_keep_threshold,
                                                 stats);

  std::vector<cudf::size_type> key_columns;
  key_columns.reserve(admission->input_ordinals.size());
  for (auto const ordinal : admission->input_ordinals) {
    key_columns.push_back(static_cast<cudf::size_type>(ordinal));
  }
  aggregate.top_n_producer = std::make_unique<sirius::op::top_n_group_key_producer>(
    std::move(coordinator), std::move(key_columns), gate_keep_threshold);
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalTopN& op)
{
  D_ASSERT(op.children.size() == 1);

  // Apply the same sort-key restrictions as LogicalOrder.
  for (auto const& order : op.orders) {
    reject_nested_column_operation(*order.expression, "ORDER BY");
  }

  auto plan = create_plan(*op.children[0]);

  // Stage-1 Top-N threshold refinement: eligibility and the execution coordinator for sink
  // self-consumption. Gated on the experimental flag -- with it off nothing runs and no counter
  // moves.
  std::shared_ptr<sirius::op::top_n_threshold_coordinator> coordinator;
  bool top_n_dynamic_filter_enabled = false;
  double gate_keep_threshold = sirius::op::scan::dynamic_filter_gate::k_default_keep_threshold;
  auto sirius_context        = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_context) {
    auto const& op_params = sirius_context->get_config().get_operator_params();
    if (op_params.enable_top_n_dynamic_filter) {
      top_n_dynamic_filter_enabled = true;
      coordinator = make_threshold_coordinator(op, &sirius_context->get_dynamic_filter_stats());
      gate_keep_threshold = op_params.dynamic_filter_keep_threshold;
    }
  }

  // Stage-4 target discovery: run both traces over the built physical child, apply the siting rule
  // to every terminal it reaches, and freeze the publication plan. Registration happens here,
  // during tree construction, so it precedes scan wrapping's has_producers() elision.
  if (coordinator) {
    auto& stats = sirius_context->get_dynamic_filter_stats();
    plan        = discover_top_n_targets(std::move(plan),
                                  top_n_key_ordinals(op),
                                  top_n_key_admissions(op),
                                  *coordinator,
                                  *sirius_context,
                                  gate_keep_threshold,
                                  stats);
  }

  // Stage-5 group-key producer: its own admission path, over the same built child. It is
  // independent of the row producer -- an aggregate shape that admits one usually rejects the
  // other -- and it installs itself inside the aggregate's subtree, so it can only add targets
  // below the aggregate and never moves the ones discovered above it.
  if (top_n_dynamic_filter_enabled) {
    install_group_key_producer(op, *plan, *sirius_context, gate_keep_threshold);
  }

  auto top_n = duckdb::make_uniq<sirius::op::sirius_physical_top_n>(
    sirius::from_duckdb_vec(op.types),
    std::move(op.orders),
    duckdb::NumericCast<std::size_t>(op.limit),
    duckdb::NumericCast<std::size_t>(op.offset),
    std::move(op.dynamic_filter),
    op.estimated_cardinality,
    gate_keep_threshold);
  top_n->threshold_coordinator = std::move(coordinator);

  top_n->children.push_back(std::move(plan));
  return std::move(top_n);
}

}  // namespace sirius::planner
