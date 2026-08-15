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

/**
 * @file test_top_n_publication.cpp
 * @brief Stage-4 publication: the coordinator's publisher loop against real channels and slots,
 *        plus the pure terminal-classification rules.
 *
 * These exercise the producer side only -- no scan consumes the published filters here -- so
 * every assertion is deterministic: publication does not race scan starts (R2 applies to the
 * *effect* on a consumer, which is asserted only in the integration suite).
 */

#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>

#include <catch.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/dynamic_filter/top_n_dynamic_filter_publish_plan.hpp>
#include <op/dynamic_filter/top_n_threshold_coordinator.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <planner/dynamic_filter/dynamic_filter_target_discovery.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

using sirius::op::dynamic_filter_stats;
using sirius::op::exact_host_key_tuple;
using sirius::op::exact_host_scalar;
using sirius::op::sirius_dynamic_filter_kind;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::threshold_offer_result;
using sirius::op::top_n_distinct_key_witness;
using sirius::op::top_n_dynamic_filter_publish_plan;
using sirius::op::top_n_filter_layer;
using sirius::op::top_n_key_semantics;
using sirius::op::top_n_threshold_coordinator;
using sirius::op::top_n_threshold_witness;
using sirius::planner::classify_top_n_terminal;
using sirius::planner::route_terminal;
using sirius::planner::target_skips_reads;
using sirius::planner::top_n_target_kind;

namespace {

constexpr auto k_int32 = cudf::data_type{cudf::type_id::INT32};

std::vector<top_n_key_semantics> ascending_keys(std::size_t count)
{
  return std::vector<top_n_key_semantics>(count,
                                          {.storage_type = k_int32,
                                           .order        = cudf::order::ASCENDING,
                                           .null_order   = cudf::null_order::AFTER});
}

top_n_threshold_witness witness(std::vector<std::int32_t> const& values)
{
  std::vector<std::optional<exact_host_scalar>> components;
  components.reserve(values.size());
  for (auto const v : values) {
    components.emplace_back(exact_host_scalar{v, k_int32});
  }
  return {exact_host_key_tuple{std::move(components)}, {}};
}

/// The filter currently visible on @p channel's single column, or null.
std::shared_ptr<sirius::op::sirius_dynamic_filter const> visible_filter(
  sirius_dynamic_filter_set const& channel, std::size_t column)
{
  auto const snapshot = channel.snapshot();
  for (auto const& entry : snapshot.columns) {
    if (entry.column == column && !entry.filters.empty()) { return entry.filters.back(); }
  }
  return nullptr;
}

/// A distinct-key witness over INT32 grouping keys, one inner vector per key tuple.
top_n_distinct_key_witness distinct_witness(std::vector<std::vector<std::int32_t>> const& keys)
{
  top_n_distinct_key_witness witness;
  witness.best_keys.reserve(keys.size());
  for (auto const& key : keys) {
    std::vector<std::optional<exact_host_scalar>> components;
    components.reserve(key.size());
    for (auto const v : key) {
      components.emplace_back(exact_host_scalar{v, k_int32});
    }
    witness.best_keys.emplace_back(std::move(components));
  }
  return witness;
}

/// A table scan carrying @p table_function, which is all `target_skips_reads` consults.
sirius::op::sirius_physical_table_scan scan_over(std::string table_function)
{
  duckdb::TableFunction function;
  function.name = std::move(table_function);
  return sirius::op::sirius_physical_table_scan{{},
                                                std::move(function),
                                                nullptr,
                                                {},
                                                {},
                                                {},
                                                {},
                                                nullptr,
                                                0,
                                                duckdb::ExtraOperatorInfo{},
                                                {},
                                                {}};
}

}  // namespace

TEST_CASE("terminal classification follows the documented table", "[dynamic_filter][top_n]")
{
  // Any operator that is not a TABLE_SCAN serves as the non-scan terminal; classification
  // consults only the node's type. (A trace terminal always carries a real node -- the walk
  // bottoms out at worst at the root -- and classify_top_n_terminal asserts that.)
  sirius::op::sirius_physical_operator non_scan_node{
    sirius::op::SiriusPhysicalOperatorType::PROJECTION, {}, /*estimated_cardinality=*/0};
  route_terminal const non_scan{.node = &non_scan_node, .ordinal = 0};
  auto scan_node = scan_over("seq_scan");
  route_terminal const scan{.node = &scan_node, .ordinal = 0};

  // Nothing material between site and sink and no reads to skip: the siting rule declines, because
  // the sink prefilter already applies the same predicate over the same rows.
  REQUIRE(classify_top_n_terminal(non_scan, top_n_filter_layer::LEX, 0, false) ==
          top_n_target_kind::SKIPPED_NO_WORK_SAVED);
  REQUIRE(classify_top_n_terminal(non_scan, top_n_filter_layer::FIRST_KEY, 0, false) ==
          top_n_target_kind::SKIPPED_NO_WORK_SAVED);
  // The same test governs a scan: one that cannot skip reads and shields no work would buy an
  // O(rows) compaction pass to save the identical pass the sink already makes.
  REQUIRE(classify_top_n_terminal(scan, top_n_filter_layer::FIRST_KEY, 0, false) ==
          top_n_target_kind::SKIPPED_NO_WORK_SAVED);
  REQUIRE(classify_top_n_terminal(scan, top_n_filter_layer::LEX, 0, false) ==
          top_n_target_kind::SKIPPED_NO_WORK_SAVED);
  // Either condition alone admits the scan: reads it never makes, or per-row work it shields.
  REQUIRE(classify_top_n_terminal(scan, top_n_filter_layer::FIRST_KEY, 0, true) ==
          top_n_target_kind::SCAN_BIND);
  REQUIRE(classify_top_n_terminal(scan, top_n_filter_layer::LEX, 0, true) ==
          top_n_target_kind::SCAN_BIND);
  REQUIRE(classify_top_n_terminal(scan, top_n_filter_layer::FIRST_KEY, 1, false) ==
          top_n_target_kind::SCAN_BIND);
  // Material work in between makes an endpoint worth its cost even where no read can be skipped --
  // the rule refuses sites that duplicate the sink's pass, not a backend. The verdict is
  // layer-independent: a LEX endpoint splices through place_endpoint_all_keys, carrying every
  // component ordinal.
  REQUIRE(classify_top_n_terminal(non_scan, top_n_filter_layer::FIRST_KEY, 1, false) ==
          top_n_target_kind::ENDPOINT_SITE);
  REQUIRE(classify_top_n_terminal(non_scan, top_n_filter_layer::LEX, 1, false) ==
          top_n_target_kind::ENDPOINT_SITE);
  // A LEX site already carries the stronger predicate, so the first-key bound is redundant there.
  REQUIRE(classify_top_n_terminal(non_scan, top_n_filter_layer::FIRST_KEY, 3, false, true) ==
          top_n_target_kind::SUBSUMED_BY_LEX);
  // Subsumption applies only to the first-key layer.
  REQUIRE(classify_top_n_terminal(non_scan, top_n_filter_layer::LEX, 3, false, true) ==
          top_n_target_kind::ENDPOINT_SITE);
}

TEST_CASE("only a reader with a read-time filter path skips reads", "[dynamic_filter][top_n]")
{
  // The consumer half of the siting rule. Every table function the plan generator routes to the
  // Parquet reader answers yes -- its `set_filter` prunes row groups before they are fetched --
  // and the DuckDB-native reader, whose only filter path runs after decode, answers no.
  for (auto const* parquet : {"parquet_scan", "read_parquet", "sirius_read_parquet"}) {
    auto const scan = scan_over(parquet);
    CHECK(target_skips_reads(scan));
  }
  auto const native = scan_over("seq_scan");
  CHECK_FALSE(target_skips_reads(native));

  // A sited endpoint reads nothing itself: its rows arrive already decoded, so it can only shorten
  // what follows it.
  sirius::op::sirius_physical_operator endpoint;
  endpoint.type = sirius::op::SiriusPhysicalOperatorType::DYNAMIC_FILTER;
  CHECK_FALSE(target_skips_reads(endpoint));
}

TEST_CASE("coordinator publishes a revision per accepted offer and tightens monotonically",
          "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{4, ascending_keys(2), true, &stats};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 4;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true},
               {.child_ordinal = 1, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0, {1}),
                          .layer              = top_n_filter_layer::LEX,
                          .component_ordinals = {0, 1}});
  coordinator.set_publish_plan(std::move(plan));

  REQUIRE(coordinator.offer(witness({50, 5})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
  REQUIRE(stats.top_n_revisions_published.load() == 1);
  REQUIRE(stats.top_n_lex_filters_pushed.load() == 1);
  REQUIRE(stats.top_n_revisions_failed.load() == 0);
  REQUIRE(channel->generation() == 1);
  REQUIRE(channel->filter_count() == 1);

  auto const first = visible_filter(*channel, 0);
  REQUIRE(first != nullptr);
  REQUIRE(first->kind() == sirius_dynamic_filter_kind::LEX_RANGE);

  SECTION("a tighter offer replaces the slot value without growing the logical filter count")
  {
    REQUIRE(coordinator.offer(witness({30, 1})) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    REQUIRE(stats.top_n_revisions_published.load() == 2);
    REQUIRE(channel->filter_count() == 1);  // replacement is not channel growth
    REQUIRE(channel->generation() == 2);
    REQUIRE(visible_filter(*channel, 0) != first);
  }
  SECTION("a looser offer publishes nothing")
  {
    REQUIRE(coordinator.offer(witness({90, 9})) == threshold_offer_result::NOT_TIGHTER);
    REQUIRE(stats.top_n_revisions_published.load() == 1);
    REQUIRE(channel->generation() == 1);
    REQUIRE(visible_filter(*channel, 0) == first);
  }
  SECTION("a closed channel narrows the fan-out without failing the revision")
  {
    channel->close_for_new_filters();
    REQUIRE(coordinator.offer(witness({10, 0})) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    // Closure is not a replica failure: nothing is installed on the drained target, the revision
    // is simply not counted as published, and no failure is recorded.
    REQUIRE(stats.top_n_revisions_failed.load() == 0);
    REQUIRE(stats.top_n_revisions_published.load() == 1);
    REQUIRE(visible_filter(*channel, 0) == first);
    // The boundary still tightens for the sink prefilter.
    REQUIRE(coordinator.boundary_update_count() == 2);
  }
}

TEST_CASE("a single-key ROW producer publishes its whole predicate as a strict RANGE",
          "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{
    3, ascending_keys(1), true, &stats, sirius::op::top_n_producer_kind::ROW};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 3;
  plan.kind = sirius::op::top_n_producer_kind::ROW;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(7),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {7}});
  coordinator.set_publish_plan(std::move(plan));

  REQUIRE(coordinator.offer(witness({42})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
  auto const published = visible_filter(*channel, 7);
  REQUIRE(published != nullptr);
  REQUIRE(published->kind() == sirius_dynamic_filter_kind::RANGE);
  REQUIRE(stats.top_n_first_key_filters_pushed.load() == 1);

  auto const& range = static_cast<sirius::op::sirius_dynamic_range_filter const&>(*published);
  // ROW-specific: a lone *row* key carries the strict form, because full-tuple peers are
  // interchangeable once K witnesses exist. A single-key GROUP_KEY producer is inclusive instead
  // -- see the GROUP_KEY cases below. Do not "fix" this assertion by relaxing it.
  REQUIRE(range.side() == sirius::op::range_bound_side::UPPER);
  REQUIRE_FALSE(range.inclusive());
}

TEST_CASE("a multi-key ROW producer's first-key bound admits boundary-tied key-zero rows",
          "[dynamic_filter][top_n]")
{
  // The first-key layer sees key zero alone, so a row tying the boundary there may still carry a
  // strictly better tail and belong in the top-K. Publishing a strict bound would drop it --
  // wrong results. This kills the `inclusive = false` (always-strict) mutation, which the
  // single-key case below cannot: that one asserts the opposite polarity.
  dynamic_filter_stats stats;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{
    4, ascending_keys(2), true, &stats, sirius::op::top_n_producer_kind::ROW};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 4;
  plan.kind = sirius::op::top_n_producer_kind::ROW;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true},
               {.child_ordinal = 1, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {0}});
  coordinator.set_publish_plan(std::move(plan));

  REQUIRE(coordinator.offer(witness({20, 4})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
  auto const published = visible_filter(*channel, 0);
  REQUIRE(published != nullptr);
  REQUIRE(published->kind() == sirius_dynamic_filter_kind::RANGE);

  auto const& range = static_cast<sirius::op::sirius_dynamic_range_filter const&>(*published);
  REQUIRE(range.inclusive());
  REQUIRE(range.side() == sirius::op::range_bound_side::UPPER);
  REQUIRE(std::get<std::int32_t>(range.bound().value()) == 20);
}

TEST_CASE("the published first-key bound keeps every row the true predicate keeps",
          "[dynamic_filter][top_n]")
{
  // End-to-end over the published filter's own compaction path: a boundary-tied key-zero row must
  // survive for a multi-key producer and must not for a single-key one. Together these kill both
  // mutations -- always-strict fails the multi-key case, always-inclusive fails the single-key
  // case -- without depending on which side of the bound the test data happens to fall.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  auto const publish_first_key = [&](std::size_t key_count) {
    auto channel = std::make_shared<sirius_dynamic_filter_set>();
    auto coordinator =
      std::make_shared<top_n_threshold_coordinator>(4, ascending_keys(key_count), true, nullptr);
    top_n_dynamic_filter_publish_plan plan;
    plan.k = 4;
    for (std::size_t i = 0; i < key_count; ++i) {
      plan.keys.push_back(
        {.child_ordinal = i, .semantics = ascending_keys(1).front(), .type_admitted = true});
    }
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                            .layer              = top_n_filter_layer::FIRST_KEY,
                            .component_ordinals = {0}});
    coordinator->set_publish_plan(std::move(plan));
    std::vector<std::int32_t> boundary(key_count, 4);
    boundary.front() = 20;
    REQUIRE(coordinator->offer(witness(boundary)) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    return std::pair{channel, coordinator};
  };

  // One column holding a row strictly better than the bound, one tying it, one worse.
  auto const rows = std::vector<std::int32_t>{19, 20, 21};
  auto column     = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(rows.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  cudaMemcpy(column->mutable_view().data<std::int32_t>(),
             rows.data(),
             rows.size() * sizeof(std::int32_t),
             cudaMemcpyHostToDevice);
  auto const batch = cudf::table_view{{column->view()}};
  std::vector<cudf::size_type> const key_columns{0};

  auto const survivors = [&](std::shared_ptr<sirius::op::sirius_dynamic_filter const> const& f) {
    auto const* compactable =
      dynamic_cast<sirius::op::sirius_compaction_applicable const*>(f.get());
    REQUIRE(compactable != nullptr);
    auto const result = compactable->apply_compact(batch, key_columns, -1, stream, mr);
    return result.filtered == nullptr ? static_cast<cudf::size_type>(rows.size())
                                      : result.rows_kept;
  };

  auto const [multi_channel, multi_coordinator] = publish_first_key(2);
  // Inclusive: 19 and the tie at 20 survive; 21 is strictly worse and is dropped.
  REQUIRE(survivors(visible_filter(*multi_channel, 0)) == 2);

  auto const [single_channel, single_coordinator] = publish_first_key(1);
  // Strict: the tie is interchangeable once K witnesses exist, so only 19 survives.
  REQUIRE(survivors(visible_filter(*single_channel, 0)) == 1);
}

TEST_CASE("one RANGE fans to every first-key target while each LEX target gets its own filter",
          "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  auto lex_a   = std::make_shared<sirius_dynamic_filter_set>();
  auto lex_b   = std::make_shared<sirius_dynamic_filter_set>();
  auto range_a = std::make_shared<sirius_dynamic_filter_set>();
  auto range_b = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{4, ascending_keys(2), true, &stats};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 4;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true},
               {.child_ordinal = 1, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  // Two LEX targets with *different* per-site mappings -- the case a single shared LEX filter
  // could not express (R7), e.g. a UNION fan-out whose branches order their columns differently.
  plan.targets.push_back({.publisher          = lex_a->register_refinement_slot(0, {1}),
                          .layer              = top_n_filter_layer::LEX,
                          .component_ordinals = {0, 1}});
  plan.targets.push_back({.publisher          = lex_b->register_refinement_slot(5, {2}),
                          .layer              = top_n_filter_layer::LEX,
                          .component_ordinals = {5, 2}});
  plan.targets.push_back({.publisher          = range_a->register_refinement_slot(3),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {3}});
  plan.targets.push_back({.publisher          = range_b->register_refinement_slot(9),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {9}});
  coordinator.set_publish_plan(std::move(plan));

  REQUIRE(coordinator.offer(witness({20, 4})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
  REQUIRE(stats.top_n_lex_filters_pushed.load() == 2);
  REQUIRE(stats.top_n_first_key_filters_pushed.load() == 2);
  REQUIRE(stats.top_n_revisions_published.load() == 1);  // one revision, four installs

  auto const a = visible_filter(*lex_a, 0);
  auto const b = visible_filter(*lex_b, 5);
  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);
  // Distinct objects, each carrying its own site's ordinals.
  REQUIRE(a != b);
  auto const& lex_filter_a = static_cast<sirius::op::sirius_dynamic_lex_range_filter const&>(*a);
  auto const& lex_filter_b = static_cast<sirius::op::sirius_dynamic_lex_range_filter const&>(*b);
  REQUIRE(lex_filter_a.referenced_ordinals()[0] == 0);
  REQUIRE(lex_filter_a.referenced_ordinals()[1] == 1);
  REQUIRE(lex_filter_b.referenced_ordinals()[0] == 5);
  REQUIRE(lex_filter_b.referenced_ordinals()[1] == 2);

  // RANGE carries no ordinal, so one object genuinely fans out to both first-key targets.
  REQUIRE(visible_filter(*range_a, 3) == visible_filter(*range_b, 9));
}

TEST_CASE("a GROUP_KEY producer publishes inclusive predicates at every key count",
          "[dynamic_filter][top_n]")
{
  // R7's flag test. A group-key boundary is the Kth-best *grouping key*, so a row tying it
  // belongs to a group that is in the answer; dropping it lowers that group's aggregates. Both
  // layers must therefore be inclusive regardless of key count. Dropping the kind term from
  // either strictness site fails one of these two sections.
  SECTION("single key: the first-key RANGE is inclusive where a ROW producer is strict")
  {
    dynamic_filter_stats stats;
    auto channel = std::make_shared<sirius_dynamic_filter_set>();
    top_n_threshold_coordinator coordinator{
      3, ascending_keys(1), true, &stats, sirius::op::top_n_producer_kind::GROUP_KEY};
    top_n_dynamic_filter_publish_plan plan;
    plan.k    = 3;
    plan.kind = sirius::op::top_n_producer_kind::GROUP_KEY;
    plan.keys = {
      {.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                            .layer              = top_n_filter_layer::FIRST_KEY,
                            .component_ordinals = {0}});
    coordinator.set_publish_plan(std::move(plan));

    // Three distinct keys fill the set; the boundary is the third-best.
    REQUIRE(coordinator.offer(distinct_witness({{10}, {20}, {30}})) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    auto const published = visible_filter(*channel, 0);
    REQUIRE(published != nullptr);
    REQUIRE(published->kind() == sirius_dynamic_filter_kind::RANGE);
    auto const& range = static_cast<sirius::op::sirius_dynamic_range_filter const&>(*published);
    REQUIRE(range.inclusive());
    REQUIRE(std::get<std::int32_t>(range.bound().value()) == 30);
  }

  SECTION("multi key: the LEX predicate is inclusive where a ROW producer is strict")
  {
    dynamic_filter_stats stats;
    auto channel = std::make_shared<sirius_dynamic_filter_set>();
    top_n_threshold_coordinator coordinator{
      2, ascending_keys(2), true, &stats, sirius::op::top_n_producer_kind::GROUP_KEY};
    top_n_dynamic_filter_publish_plan plan;
    plan.k    = 2;
    plan.kind = sirius::op::top_n_producer_kind::GROUP_KEY;
    plan.keys = {
      {.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true},
      {.child_ordinal = 1, .semantics = ascending_keys(1).front(), .type_admitted = true}};
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(0, {1}),
                            .layer              = top_n_filter_layer::LEX,
                            .component_ordinals = {0, 1}});
    coordinator.set_publish_plan(std::move(plan));

    REQUIRE(coordinator.offer(distinct_witness({{1, 5}, {2, 6}})) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    auto const published = visible_filter(*channel, 0);
    REQUIRE(published != nullptr);
    REQUIRE(published->kind() == sirius_dynamic_filter_kind::LEX_RANGE);
    auto const& lex = static_cast<sirius::op::sirius_dynamic_lex_range_filter const&>(*published);
    REQUIRE(lex.inclusive());
  }
}

TEST_CASE("the distinct-key witness set unions by value and defines the boundary only at K",
          "[dynamic_filter][top_n]")
{
  // R1. Union by *value* is the foundation of the K-distinct proof: two tasks witnessing the same
  // grouping key must count once, or the set claims K proven groups from fewer and publishes a
  // boundary that is too tight -- dropping rows of a group that is in the answer.
  dynamic_filter_stats stats;
  top_n_threshold_coordinator coordinator{
    3, ascending_keys(1), true, &stats, sirius::op::top_n_producer_kind::GROUP_KEY};

  // Two tasks witness overlapping keys; only three *distinct* values exist across both.
  REQUIRE(coordinator.offer(distinct_witness({{10}, {20}})) ==
          threshold_offer_result::NO_ACCEPTING_TARGET);
  REQUIRE_FALSE(coordinator.tightest_boundary().has_value());  // undefined below K
  REQUIRE(stats.top_n_group_witness_set_full.load() == 0);

  // Re-witnessing the same values adds nothing: a multiset would reach K here and publish early.
  REQUIRE(coordinator.offer(distinct_witness({{10}, {20}})) ==
          threshold_offer_result::NO_ACCEPTING_TARGET);
  REQUIRE_FALSE(coordinator.tightest_boundary().has_value());
  REQUIRE(stats.top_n_group_witness_set_full.load() == 0);

  // A genuinely new third value completes the set; the boundary is the Kth (worst) of the three.
  REQUIRE(coordinator.offer(distinct_witness({{30}})) ==
          threshold_offer_result::NO_ACCEPTING_TARGET);  // no targets wired in this test
  REQUIRE(stats.top_n_group_witness_set_full.load() == 1);
  auto const boundary = coordinator.tightest_boundary();
  REQUIRE(boundary.has_value());
  REQUIRE(std::get<std::int32_t>(boundary->component(0)->value()) == 30);

  // Better keys tighten the Kth element monotonically; worse ones never loosen it.
  REQUIRE(coordinator.offer(distinct_witness({{5}})) ==
          threshold_offer_result::NO_ACCEPTING_TARGET);
  REQUIRE(std::get<std::int32_t>(coordinator.tightest_boundary()->component(0)->value()) == 20);
  REQUIRE(coordinator.offer(distinct_witness({{99}})) == threshold_offer_result::NOT_TIGHTER);
  REQUIRE(std::get<std::int32_t>(coordinator.tightest_boundary()->component(0)->value()) == 20);
  REQUIRE(stats.top_n_group_witness_set_full.load() == 1);  // latched, counted once
}

TEST_CASE("a GROUP_KEY producer's published LEX predicate keeps rows tying the whole boundary",
          "[dynamic_filter][top_n]")
{
  // The behavioural counterpart of the multi-key flag assertion above, through the published
  // filter's own compaction path. R7 calls this the site that silently corrupts multi-key group
  // aggregates, and a flag reads the same whether or not the filter actually admits the tie.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  auto const publish_lex = [&](sirius::op::top_n_producer_kind kind) {
    auto channel = std::make_shared<sirius_dynamic_filter_set>();
    auto coordinator =
      std::make_shared<top_n_threshold_coordinator>(2, ascending_keys(2), true, nullptr, kind);
    top_n_dynamic_filter_publish_plan plan;
    plan.k    = 2;
    plan.kind = kind;
    plan.keys = {
      {.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true},
      {.child_ordinal = 1, .semantics = ascending_keys(1).front(), .type_admitted = true}};
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(0, {1}),
                            .layer              = top_n_filter_layer::LEX,
                            .component_ordinals = {0, 1}});
    coordinator->set_publish_plan(std::move(plan));
    if (kind == sirius::op::top_n_producer_kind::GROUP_KEY) {
      REQUIRE(coordinator->offer(distinct_witness({{5, 5}, {7, 7}})) ==
              threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    } else {
      REQUIRE(coordinator->offer(witness({7, 7})) ==
              threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    }
    return std::pair{channel, coordinator};
  };

  // Two key columns: one row strictly better than the boundary (7,7), one tying it exactly, one
  // strictly worse.
  auto const key0        = std::vector<std::int32_t>{7, 7, 7};
  auto const key1        = std::vector<std::int32_t>{6, 7, 8};
  auto const make_column = [&](std::vector<std::int32_t> const& values) {
    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            static_cast<cudf::size_type>(values.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            mr);
    cudaMemcpy(column->mutable_view().data<std::int32_t>(),
               values.data(),
               values.size() * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);
    return column;
  };
  auto const column0 = make_column(key0);
  auto const column1 = make_column(key1);
  auto const batch   = cudf::table_view{{column0->view(), column1->view()}};
  std::vector<cudf::size_type> const key_columns{0, 1};

  auto const survivors = [&](std::shared_ptr<sirius::op::sirius_dynamic_filter const> const& f) {
    auto const* compactable =
      dynamic_cast<sirius::op::sirius_compaction_applicable const*>(f.get());
    REQUIRE(compactable != nullptr);
    auto const result = compactable->apply_compact(batch, key_columns, -1, stream, mr);
    return result.filtered == nullptr ? static_cast<cudf::size_type>(key0.size())
                                      : result.rows_kept;
  };

  auto const [group_channel, group_coordinator] =
    publish_lex(sirius::op::top_n_producer_kind::GROUP_KEY);
  // Inclusive: (7,6) and the full-tuple tie (7,7) survive. Dropping the tie would remove every
  // row of the boundary group from this batch and lower its aggregates.
  REQUIRE(survivors(visible_filter(*group_channel, 0)) == 2);

  auto const [row_channel, row_coordinator] = publish_lex(sirius::op::top_n_producer_kind::ROW);
  // Strict, unchanged: a row producer's full-tuple peers are interchangeable once K witnesses
  // exist, so only the strictly better row survives.
  REQUIRE(survivors(visible_filter(*row_channel, 0)) == 1);
}

TEST_CASE("a LEX endpoint slot applies the published predicate at its declared ordinals",
          "[dynamic_filter][top_n]")
{
  // The endpoint's consumption plumbing -- snapshot, the compaction block's multi-column
  // resolution through referenced_ordinals(), apply_compact -- at deliberately non-identity
  // ordinals: component 0 lives in column 1 and component 1 in column 0, with column 2 as
  // payload. A resolver that read components at their component index instead of their declared
  // ordinal would compare the wrong columns and pass or prune the wrong rows here. Strictness at
  // the endpoint rides the same triple: a strict ROW boundary drops the full-tuple tie that an
  // inclusive GROUP_KEY boundary must keep.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  auto const publish_lex_endpoint = [&](sirius::op::top_n_producer_kind kind) {
    auto channel = std::make_shared<sirius_dynamic_filter_set>();
    auto coordinator =
      std::make_shared<top_n_threshold_coordinator>(2, ascending_keys(2), true, nullptr, kind);
    top_n_dynamic_filter_publish_plan plan;
    plan.k    = 2;
    plan.kind = kind;
    plan.keys = {
      {.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true},
      {.child_ordinal = 1, .semantics = ascending_keys(1).front(), .type_admitted = true}};
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(1, {0}),
                            .layer              = top_n_filter_layer::LEX,
                            .component_ordinals = {1, 0}});
    coordinator->set_publish_plan(std::move(plan));
    if (kind == sirius::op::top_n_producer_kind::GROUP_KEY) {
      REQUIRE(coordinator->offer(distinct_witness({{5, 5}, {7, 7}})) ==
              threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    } else {
      REQUIRE(coordinator->offer(witness({7, 7})) ==
              threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    }
    return std::pair{channel, coordinator};
  };

  // The boundary tuple is (7, 7). Column 1 carries component 0 and column 0 carries component 1,
  // so by (component 0, component 1) the rows read: (7,6) strictly better, (7,7) the exact
  // full-tuple tie, (7,8) strictly worse. Column 2 is payload no component references.
  auto const column0_values = std::vector<std::int32_t>{6, 7, 8};     // component 1
  auto const column1_values = std::vector<std::int32_t>{7, 7, 7};     // component 0
  auto const column2_values = std::vector<std::int32_t>{10, 20, 30};  // payload
  auto const make_column    = [&](std::vector<std::int32_t> const& values) {
    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            static_cast<cudf::size_type>(values.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            mr);
    cudaMemcpy(column->mutable_view().data<std::int32_t>(),
               values.data(),
               values.size() * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);
    return column;
  };
  auto const column0 = make_column(column0_values);
  auto const column1 = make_column(column1_values);
  auto const column2 = make_column(column2_values);
  auto const batch   = cudf::table_view{{column0->view(), column1->view(), column2->view()}};

  auto const survivors = [&](sirius_dynamic_filter_set const& channel) {
    auto const filtered = sirius::op::scan::apply_dynamic_filters_to_view(
      batch,
      channel.snapshot(),
      stream,
      sirius::op::scan::dynamic_filter_apply_mode::include_ast_row_masks,
      nullptr,
      -1);
    return filtered == nullptr ? static_cast<cudf::size_type>(column0_values.size())
                               : filtered->num_rows();
  };

  auto const [row_channel, row_coordinator] =
    publish_lex_endpoint(sirius::op::top_n_producer_kind::ROW);
  auto const installed = visible_filter(*row_channel, 1);
  REQUIRE(installed != nullptr);
  REQUIRE(installed->kind() == sirius_dynamic_filter_kind::LEX_RANGE);
  // The installed filter carries the slot's declared ordinals, primary first.
  auto const& lex     = static_cast<sirius::op::sirius_dynamic_lex_range_filter const&>(*installed);
  auto const declared = lex.referenced_ordinals();
  REQUIRE(declared.size() == 2);
  REQUIRE(declared[0] == 1);
  REQUIRE(declared[1] == 0);
  // Strict: only the strictly better row survives.
  REQUIRE(survivors(*row_channel) == 1);

  auto const [group_channel, group_coordinator] =
    publish_lex_endpoint(sirius::op::top_n_producer_kind::GROUP_KEY);
  // Inclusive: the strictly better row and the exact full-tuple tie survive.
  REQUIRE(survivors(*group_channel) == 2);
}

TEST_CASE("a decimal boundary publishes a fixed-point literal at its own scale",
          "[dynamic_filter][top_n]")
{
  // Covers `make_boundary_scalar`'s fixed-point branches, which nothing else reaches: the
  // prefilter carries its bound in kernel launch parameters and never builds a device scalar, so
  // only publication exercises the literal. The scale must survive onto the scalar -- a literal
  // built at the wrong scale compares against a differently scaled column and prunes wrong rows.
  auto const scaled_key = [](std::int32_t scale, cudf::type_id id) {
    return std::vector<top_n_key_semantics>{{.storage_type = cudf::data_type{id, scale},
                                             .order        = cudf::order::ASCENDING,
                                             .null_order   = cudf::null_order::AFTER}};
  };

  SECTION("DECIMAL32 keeps its scale on the published bound")
  {
    auto channel    = std::make_shared<sirius_dynamic_filter_set>();
    auto const keys = scaled_key(-2, cudf::type_id::DECIMAL32);
    top_n_threshold_coordinator coordinator{3, keys, true, nullptr};
    top_n_dynamic_filter_publish_plan plan;
    plan.k    = 3;
    plan.keys = {{.child_ordinal = 0, .semantics = keys.front(), .type_admitted = true}};
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                            .layer              = top_n_filter_layer::FIRST_KEY,
                            .component_ordinals = {0}});
    coordinator.set_publish_plan(std::move(plan));

    // 1234 at scale -2 is 12.34; the scaled integer is what travels.
    top_n_threshold_witness witness{
      exact_host_key_tuple{
        {exact_host_scalar{std::int32_t{1234}, cudf::data_type{cudf::type_id::DECIMAL32, -2}}}},
      {}};
    REQUIRE(coordinator.offer(std::move(witness)) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    auto const published = visible_filter(*channel, 0);
    REQUIRE(published != nullptr);
    REQUIRE(published->kind() == sirius_dynamic_filter_kind::RANGE);
    auto const& range = static_cast<sirius::op::sirius_dynamic_range_filter const&>(*published);
    REQUIRE(std::get<std::int32_t>(range.bound().value()) == 1234);
    REQUIRE(range.bound().storage_type() == cudf::data_type{cudf::type_id::DECIMAL32, -2});
  }

  SECTION("DECIMAL64 keeps its scale on the published bound")
  {
    auto channel    = std::make_shared<sirius_dynamic_filter_set>();
    auto const keys = scaled_key(-4, cudf::type_id::DECIMAL64);
    top_n_threshold_coordinator coordinator{3, keys, true, nullptr};
    top_n_dynamic_filter_publish_plan plan;
    plan.k    = 3;
    plan.keys = {{.child_ordinal = 0, .semantics = keys.front(), .type_admitted = true}};
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                            .layer              = top_n_filter_layer::FIRST_KEY,
                            .component_ordinals = {0}});
    coordinator.set_publish_plan(std::move(plan));

    top_n_threshold_witness witness{
      exact_host_key_tuple{
        {exact_host_scalar{std::int64_t{987654}, cudf::data_type{cudf::type_id::DECIMAL64, -4}}}},
      {}};
    REQUIRE(coordinator.offer(std::move(witness)) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    auto const published = visible_filter(*channel, 0);
    REQUIRE(published != nullptr);
    auto const& range = static_cast<sirius::op::sirius_dynamic_range_filter const&>(*published);
    REQUIRE(std::get<std::int64_t>(range.bound().value()) == 987654);
    REQUIRE(range.bound().storage_type() == cudf::data_type{cudf::type_id::DECIMAL64, -4});
  }

  SECTION("DECIMAL128 keeps its scale on the published bound")
  {
    // The rep sits one past int64's range, so the round trip through make_boundary_scalar and
    // the compaction marshal proves the __int128_t widening carried the value exactly -- a
    // truncation anywhere would wrap it negative and flip the compaction verdicts below.
    auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
    auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    REQUIRE(space);
    auto const stream = cudf::get_default_stream();
    auto const mr     = space->get_default_allocator();

    auto const k_dec128        = cudf::data_type{cudf::type_id::DECIMAL128, -4};
    constexpr __int128_t k_rep = __int128_t{10} * 1'000'000'000'000'000'000LL + 7;  // 10^19 + 7

    auto channel    = std::make_shared<sirius_dynamic_filter_set>();
    auto const keys = scaled_key(-4, cudf::type_id::DECIMAL128);
    top_n_threshold_coordinator coordinator{3, keys, true, nullptr};
    top_n_dynamic_filter_publish_plan plan;
    plan.k    = 3;
    plan.keys = {{.child_ordinal = 0, .semantics = keys.front(), .type_admitted = true}};
    plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                            .layer              = top_n_filter_layer::FIRST_KEY,
                            .component_ordinals = {0}});
    coordinator.set_publish_plan(std::move(plan));

    top_n_threshold_witness witness{exact_host_key_tuple{{exact_host_scalar{k_rep, k_dec128}}}, {}};
    REQUIRE(coordinator.offer(std::move(witness)) ==
            threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
    auto const published = visible_filter(*channel, 0);
    REQUIRE(published != nullptr);
    REQUIRE(published->kind() == sirius_dynamic_filter_kind::RANGE);
    auto const& range = static_cast<sirius::op::sirius_dynamic_range_filter const&>(*published);
    REQUIRE(std::get<__int128_t>(range.bound().value()) == k_rep);
    REQUIRE(range.bound().storage_type() == k_dec128);

    // Apply the published filter to reps straddling the boundary. The verdict pattern is
    // computed from the filter's own published side/inclusive semantics, so this holds whichever
    // strictness the layer chose -- what it cannot survive is a truncated bound.
    std::vector<__int128_t> const reps{k_rep - 1, k_rep, k_rep + 1};
    auto column = cudf::make_fixed_point_column(k_dec128,
                                                static_cast<cudf::size_type>(reps.size()),
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);
    cudaMemcpy(column->mutable_view().data<__int128_t>(),
               reps.data(),
               reps.size() * sizeof(__int128_t),
               cudaMemcpyHostToDevice);
    auto const batch = cudf::table_view{{column->view()}};
    std::vector<cudf::size_type> const key_columns{0};
    auto const* compactable =
      dynamic_cast<sirius::op::sirius_compaction_applicable const*>(published.get());
    REQUIRE(compactable != nullptr);
    auto const result = compactable->apply_compact(batch, key_columns, -1, stream, mr);

    auto const keeps = [&](__int128_t rep) {
      bool const lower = range.side() == sirius::op::range_bound_side::LOWER;
      if (rep == k_rep) { return range.inclusive(); }
      return lower == (rep > k_rep);
    };
    cudf::size_type expected_kept = 0;
    for (auto const rep : reps) {
      if (keeps(rep)) { ++expected_kept; }
    }
    REQUIRE(result.rows_kept == expected_kept);
  }
}

TEST_CASE("a null-headed distinct-key boundary publishes nothing", "[dynamic_filter][top_n]")
{
  // R6. Group-by makes null a group, so a null grouping key is a legitimate witnessed value and
  // must count toward K. It just cannot be published: neither layer expresses a null first
  // component, and every published RANGE reads component 0. Without the suppression the
  // publisher would dereference a disengaged component.
  dynamic_filter_stats stats;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();
  top_n_threshold_coordinator coordinator{
    2, ascending_keys(1), true, &stats, sirius::op::top_n_producer_kind::GROUP_KEY};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 2;
  plan.kind = sirius::op::top_n_producer_kind::GROUP_KEY;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {0}});
  coordinator.set_publish_plan(std::move(plan));

  // NULLS AFTER makes the null group the worst, so with only one non-null key witnessed the Kth
  // element is the null one.
  top_n_distinct_key_witness null_headed;
  null_headed.best_keys.push_back(exact_host_key_tuple{{exact_host_scalar{7, k_int32}}});
  null_headed.best_keys.push_back(exact_host_key_tuple{{std::nullopt}});
  REQUIRE(coordinator.offer(std::move(null_headed)) ==
          threshold_offer_result::UNSUPPORTED_BOUNDARY);
  REQUIRE(stats.top_n_group_witness_set_full.load() == 1);  // the null group did count toward K
  REQUIRE_FALSE(coordinator.tightest_boundary().has_value());
  REQUIRE(visible_filter(*channel, 0) == nullptr);
  REQUIRE(stats.top_n_revisions_published.load() == 0);
  REQUIRE(stats.top_n_offers_unsupported.load() == 1);

  // A better non-null key displaces the null one from the K best and publication resumes.
  REQUIRE(coordinator.offer(distinct_witness({{3}})) ==
          threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
  auto const published = visible_filter(*channel, 0);
  REQUIRE(published != nullptr);
  REQUIRE(
    std::get<std::int32_t>(
      static_cast<sirius::op::sirius_dynamic_range_filter const&>(*published).bound().value()) ==
    7);
}

TEST_CASE("distinct-key offers are rejected by a ROW coordinator", "[dynamic_filter][top_n]")
{
  top_n_threshold_coordinator coordinator{2, ascending_keys(1), true, nullptr};
  REQUIRE_THROWS_AS(coordinator.offer(distinct_witness({{1}})), std::logic_error);
}

TEST_CASE("a closed target is skipped while an open target still receives the same revision",
          "[dynamic_filter][top_n]")
{
  // The R3 property the single-target closure case cannot show: closure narrows the fan-out, it
  // does not abort the revision. Collapsing closure into the replica-failure path (returning
  // before the install loop) would leave the open target unpublished and fail here.
  dynamic_filter_stats stats;
  auto closed_channel = std::make_shared<sirius_dynamic_filter_set>();
  auto open_channel   = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{3, ascending_keys(1), true, &stats};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 3;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = closed_channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {0}});
  plan.targets.push_back({.publisher          = open_channel->register_refinement_slot(1),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {1}});
  coordinator.set_publish_plan(std::move(plan));

  closed_channel->close_for_new_filters();
  REQUIRE(coordinator.offer(witness({25})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);

  REQUIRE(visible_filter(*closed_channel, 0) == nullptr);  // drained: nothing installed
  REQUIRE(visible_filter(*open_channel, 1) != nullptr);    // still open: the revision landed
  REQUIRE(stats.top_n_revisions_published.load() == 1);
  REQUIRE(stats.top_n_revisions_failed.load() == 0);  // closure is not a replica failure
  REQUIRE(stats.top_n_first_key_filters_pushed.load() == 1);

  // Once every target has drained, a later offer constructs nothing at all -- no revision, no
  // device scalars -- while the boundary still tightens for the sink prefilter.
  open_channel->close_for_new_filters();
  REQUIRE(coordinator.offer(witness({10})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
  REQUIRE(stats.top_n_revisions_published.load() == 1);
  REQUIRE(stats.top_n_revisions_failed.load() == 0);
  REQUIRE(coordinator.boundary_update_count() == 2);
}

TEST_CASE("the publication plan is frozen once offers can occur", "[dynamic_filter][top_n]")
{
  // publish_revision reads _plan without the lock, which is sound only because the plan cannot
  // change after any offer. That is enforced, not merely documented.
  auto channel = std::make_shared<sirius_dynamic_filter_set>();
  top_n_threshold_coordinator coordinator{2, ascending_keys(1), true, nullptr};

  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 2;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {0}});
  coordinator.set_publish_plan(std::move(plan));

  REQUIRE(coordinator.offer(witness({5})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);

  top_n_dynamic_filter_publish_plan late;
  late.k = 2;
  REQUIRE_THROWS_AS(coordinator.set_publish_plan(std::move(late)), std::logic_error);
}

TEST_CASE("a LEX target must declare one ordinal per ORDER BY key", "[dynamic_filter][top_n]")
{
  auto channel = std::make_shared<sirius_dynamic_filter_set>();
  top_n_threshold_coordinator coordinator{4, ascending_keys(2), true, nullptr};

  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 4;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true},
               {.child_ordinal = 1, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  // One ordinal for a two-key boundary: publish_revision would index past the end.
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::LEX,
                          .component_ordinals = {0}});
  REQUIRE_THROWS_AS(coordinator.set_publish_plan(std::move(plan)), std::logic_error);
}

TEST_CASE("finish drains publication and rejects later offers", "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{2, ascending_keys(1), true, &stats};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 2;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {0}});
  coordinator.set_publish_plan(std::move(plan));

  REQUIRE(coordinator.offer(witness({8})) == threshold_offer_result::ACCEPTED_FOR_PUBLICATION);
  coordinator.finish();
  coordinator.finish();  // idempotent
  REQUIRE(coordinator.offer(witness({1})) == threshold_offer_result::REJECTED_STATE);
  REQUIRE(stats.top_n_revisions_published.load() == 1);
  REQUIRE(stats.top_n_revisions_failed.load() == 0);
}

TEST_CASE("cancellation stops publication and finish stays safe afterwards",
          "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{2, ascending_keys(1), true, &stats};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 2;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {0}});
  coordinator.set_publish_plan(std::move(plan));

  coordinator.cancel();
  REQUIRE(coordinator.offer(witness({8})) == threshold_offer_result::REJECTED_STATE);
  REQUIRE(stats.top_n_revisions_published.load() == 0);
  coordinator.finish();
  REQUIRE(coordinator.offer(witness({1})) == threshold_offer_result::REJECTED_STATE);
}

TEST_CASE("concurrent offers publish a strictly increasing revision sequence",
          "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  auto channel = std::make_shared<sirius_dynamic_filter_set>();

  top_n_threshold_coordinator coordinator{2, ascending_keys(1), true, &stats};
  top_n_dynamic_filter_publish_plan plan;
  plan.k    = 2;
  plan.keys = {{.child_ordinal = 0, .semantics = ascending_keys(1).front(), .type_admitted = true}};
  plan.targets.push_back({.publisher          = channel->register_refinement_slot(0),
                          .layer              = top_n_filter_layer::FIRST_KEY,
                          .component_ordinals = {0}});
  coordinator.set_publish_plan(std::move(plan));

  // Descending values from several threads: every accepted offer tightens, and the slot's
  // stale-write rejection plus the single-publisher discipline must leave the channel coherent.
  constexpr int k_threads = 4;
  std::vector<std::thread> threads;
  threads.reserve(k_threads);
  for (int t = 0; t < k_threads; ++t) {
    threads.emplace_back([&coordinator, t] {
      for (std::int32_t v = 500 - t; v > 0; v -= k_threads) {
        (void)coordinator.offer(witness({v}));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  coordinator.finish();

  REQUIRE(stats.top_n_revisions_failed.load() == 0);
  REQUIRE(stats.top_n_revisions_stale.load() == 0);
  REQUIRE(channel->filter_count() == 1);
  // Every published revision bumped the generation exactly once.
  REQUIRE(channel->generation() == stats.top_n_revisions_published.load());
  REQUIRE(visible_filter(*channel, 0) != nullptr);
}
