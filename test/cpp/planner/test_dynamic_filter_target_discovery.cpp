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

/*
 * Unit tests for the dynamic-filter target-discovery walk
 * (`planner/dynamic_filter/dynamic_filter_target_discovery.hpp`).
 *
 * Pure descent rules are tested directly. Lightweight projection and filter trees cover dispatch,
 * trace classification, and endpoint splicing; plan-shape tests cover concrete join and aggregate
 * operators.
 */

#include "expression/ast/cast.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "helper/logical_type.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_replicate.hpp"
#include "op/sirius_physical_union.hpp"
#include "planner/dynamic_filter/dynamic_filter_target_discovery.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb/common/enums/join_type.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using sirius::planner::descent_policy;
using sirius::planner::descent_steps;
using sirius::planner::group_by_key_input;
using sirius::planner::join_block_descent;
using sirius::planner::place_endpoint;
using sirius::planner::projection_reference_input;
using sirius::planner::scan_route_join_type_admissible;
using sirius::planner::trace_probe_key;

constexpr descent_policy kSipOn{.descend_build_blocks = true};
constexpr descent_policy kSipOff{.descend_build_blocks = false};

// The placement code never inspects a column's type, but the operators require a concrete one.
sirius::logical_type int_type() { return sirius::logical_type::make(sirius::type_id::INTEGER); }

// A pass-through projection output: a plain reference to input column `col`.
std::unique_ptr<sirius::ast::node> make_reference(std::uint32_t col)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{col, int_type()});
}

// A cast is not a pass-through projection reference, so tracing terminates here.
std::unique_ptr<sirius::ast::node> make_cast(std::uint32_t col)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::cast{make_reference(col), int_type(), /*try_cast=*/false});
}

// Collect move-only expression nodes into a projection's select_list.
template <class... Nodes>
duckdb::vector<std::unique_ptr<sirius::ast::node>> make_select_list(Nodes&&... nodes)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
  list.reserve(sizeof...(nodes));
  (list.push_back(std::forward<Nodes>(nodes)), ...);
  return list;
}

// A PROJECTION whose i-th output is described by select_list[i]. `types` is sized to match;
// the placement code reads only select_list.
duckdb::unique_ptr<sirius::op::sirius_physical_projection> make_projection(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list)
{
  duckdb::vector<sirius::logical_type> types(select_list.size(), int_type());
  return duckdb::make_uniq<sirius::op::sirius_physical_projection>(
    std::move(types), std::move(select_list), /*estimated_cardinality=*/0);
}

// A leaf scan. descent_steps refuses DUMMY_SCAN, so it terminates any descent.
duckdb::unique_ptr<sirius::op::sirius_physical_dummy_scan> make_scan(std::size_t width)
{
  duckdb::vector<sirius::logical_type> types(width, int_type());
  return duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(std::move(types),
                                                                   /*estimated_cardinality=*/0);
}

// A passthrough FILTER: every input column survives at its input position.
duckdb::unique_ptr<sirius::op::sirius_physical_filter> make_passthrough_filter(std::size_t width)
{
  duckdb::vector<sirius::logical_type> types(width, int_type());
  return duckdb::make_uniq<sirius::op::sirius_physical_filter>(
    std::move(types), make_reference(0), /*estimated_cardinality=*/0);
}

// A gather FILTER: output i is input column output_indices[i].
duckdb::unique_ptr<sirius::op::sirius_physical_filter> make_gather_filter(
  std::vector<cudf::size_type> output_indices)
{
  duckdb::vector<sirius::logical_type> types(output_indices.size(), int_type());
  return duckdb::make_uniq<sirius::op::sirius_physical_filter>(
    std::move(types), make_reference(0), /*estimated_cardinality=*/0, std::move(output_indices));
}

// Use a tagged base operator where descent reads only base-class children and types.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> make_typed_double(
  sirius::op::SiriusPhysicalOperatorType type, std::size_t width)
{
  duckdb::vector<sirius::logical_type> types(width, int_type());
  return duckdb::make_uniq<sirius::op::sirius_physical_operator>(type,
                                                                 std::move(types),
                                                                 /*estimated_cardinality=*/0);
}

// A place_endpoint factory that records the sited operator and the endpoint it splices in, then
// returns a PROJECTION as the (childless) endpoint.
auto capturing_endpoint_factory(sirius::op::sirius_physical_operator const*& sited,
                                sirius::op::sirius_physical_operator*& endpoint)
{
  return [&sited, &endpoint](sirius::op::sirius_physical_operator const& site)
           -> duckdb::unique_ptr<sirius::op::sirius_physical_operator> {
    sited        = &site;
    auto wrapper = make_projection(make_select_list(make_reference(0)));
    endpoint     = wrapper.get();
    return wrapper;
  };
}

// Construct a real endpoint over an empty channel without device work.
duckdb::unique_ptr<sirius::op::scan::sirius_physical_dynamic_filter> make_endpoint_operator(
  std::size_t width)
{
  duckdb::vector<sirius::logical_type> types(width, int_type());
  return duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
    std::move(types),
    /*estimated_cardinality=*/0,
    std::make_shared<sirius::op::sirius_dynamic_filter_set>());
}

// A place_endpoint factory that splices in a real endpoint operator and records every splice.
auto recording_endpoint_factory(std::vector<sirius::op::sirius_physical_operator*>& endpoints)
{
  return [&endpoints](sirius::op::sirius_physical_operator const& site)
           -> duckdb::unique_ptr<sirius::op::sirius_physical_operator> {
    auto wrapper = make_endpoint_operator(site.types.size());
    endpoints.push_back(wrapper.get());
    return wrapper;
  };
}

// Arm `a`'s key `k` is its scan's column kSetOpAllArmKeys[a][k].
constexpr std::size_t kSetOpAllKeyCount = 2;
constexpr std::array<std::array<std::size_t, kSetOpAllKeyCount>, 2> kSetOpAllArmKeys{
  {{2, 0}, {1, 3}}};

struct set_operation_all_plan {
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> root;
  std::array<sirius::op::sirius_physical_operator*, 2> arm_scans{};
};

// The shape `plan_set_operation_all` lowers a two-key EXCEPT ALL (one tag) or INTERSECT ALL (two
// tags) to: REPLICATE -> copy-count PROJECTION -> HASH_GROUP_BY -> UNION of tagged arm projections.
// A cast stands in for the copy count's CASE: both are computed, not plain references.
set_operation_all_plan make_set_operation_all(std::size_t tag_count)
{
  auto const tinyint = sirius::logical_type::make(sirius::type_id::TINYINT);
  auto const bigint  = sirius::logical_type::make(sirius::type_id::BIGINT);
  set_operation_all_plan plan;

  auto union_node = duckdb::make_uniq<sirius::op::sirius_physical_union>(
    duckdb::vector<sirius::logical_type>(kSetOpAllKeyCount + tag_count, int_type()),
    /*estimated_cardinality=*/0);
  for (std::size_t arm = 0; arm < 2; ++arm) {
    auto scan           = make_scan(/*width=*/4);
    plan.arm_scans[arm] = scan.get();
    auto select_list =
      make_select_list(make_reference(static_cast<std::uint32_t>(kSetOpAllArmKeys[arm][0])),
                       make_reference(static_cast<std::uint32_t>(kSetOpAllArmKeys[arm][1])));
    for (std::size_t tag = 0; tag < tag_count; ++tag) {
      select_list.push_back(std::make_unique<sirius::ast::node>(
        sirius::ast::constant{sirius::value{std::int8_t{1}}, tinyint}));
    }
    auto tagged = make_projection(std::move(select_list));
    tagged->children.push_back(std::move(scan));
    union_node->children.push_back(std::move(tagged));
  }

  duckdb::vector<std::unique_ptr<sirius::ast::node>> sums;
  for (std::size_t tag = 0; tag < tag_count; ++tag) {
    std::vector<std::unique_ptr<sirius::ast::node>> arguments;
    arguments.push_back(make_reference(static_cast<std::uint32_t>(kSetOpAllKeyCount + tag)));
    sums.push_back(std::make_unique<sirius::ast::node>(sirius::ast::aggregate{
      sirius::aggregate_id::sum, std::move(arguments), bigint, /*distinct=*/false}));
  }
  duckdb::vector<sirius::logical_type> aggregate_types(kSetOpAllKeyCount, int_type());
  aggregate_types.insert(aggregate_types.end(), tag_count, bigint);
  auto aggregate = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate>(
    std::move(aggregate_types),
    std::move(sums),
    make_select_list(make_reference(0), make_reference(1)),
    /*estimated_cardinality=*/0);
  aggregate->children.push_back(std::move(union_node));

  auto count_projection =
    make_projection(make_select_list(make_reference(0),
                                     make_reference(1),
                                     make_cast(static_cast<std::uint32_t>(kSetOpAllKeyCount))));
  count_projection->children.push_back(std::move(aggregate));

  plan.root = duckdb::make_uniq<sirius::op::sirius_physical_replicate>(
    duckdb::vector<sirius::logical_type>(kSetOpAllKeyCount, int_type()),
    sirius::op::gpu_replicate_impl::limits{.max_rows = 1024, .max_bytes = 1 << 20},
    /*estimated_cardinality=*/0);
  plan.root->children.push_back(std::move(count_projection));
  return plan;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// projection_reference_input
// ---------------------------------------------------------------------------------------------

TEST_CASE("projection_reference_input forwards a plain reference and refuses a computed expression",
          "[dynamic_filter][placement]")
{
  auto const reference = make_reference(7);
  auto const forwarded = projection_reference_input(*reference);
  REQUIRE(forwarded.has_value());
  REQUIRE(*forwarded == 7);

  auto const computed = make_cast(7);
  REQUIRE(projection_reference_input(*computed) == std::nullopt);
}

// ---------------------------------------------------------------------------------------------
// group_by_key_input
// ---------------------------------------------------------------------------------------------

TEST_CASE("group_by_key_input maps a grouping-key output to its input column",
          "[dynamic_filter][placement]")
{
  std::vector<int> const group_idx{4, 2, 9};  // three grouping keys

  SECTION("a grouping-key output maps to its input column")
  {
    auto const first =
      group_by_key_input(group_idx, /*grouping_set_count=*/1, /*output_ordinal=*/0);
    REQUIRE(first.has_value());
    REQUIRE(*first == 4);

    auto const third =
      group_by_key_input(group_idx, /*grouping_set_count=*/1, /*output_ordinal=*/2);
    REQUIRE(third.has_value());
    REQUIRE(*third == 9);
  }

  SECTION("zero grouping sets is treated as a plain single grouping")
  {
    auto const key = group_by_key_input(group_idx, /*grouping_set_count=*/0, /*output_ordinal=*/1);
    REQUIRE(key.has_value());
    REQUIRE(*key == 2);
  }

  SECTION("an aggregate-result output ordinal (at or past the keys) refuses")
  {
    REQUIRE(group_by_key_input(group_idx, 1, /*output_ordinal=*/3) == std::nullopt);
    REQUIRE(group_by_key_input(group_idx, 1, /*output_ordinal=*/4) == std::nullopt);
  }

  SECTION("more than one grouping set (ROLLUP/CUBE/GROUPING SETS) refuses")
  {
    REQUIRE(group_by_key_input(group_idx, /*grouping_set_count=*/2, /*output_ordinal=*/0) ==
            std::nullopt);
  }
}

// ---------------------------------------------------------------------------------------------
// join_block_descent
// ---------------------------------------------------------------------------------------------

TEST_CASE("join_block_descent maps a probe-block output into the probe child",
          "[dynamic_filter][placement]")
{
  std::vector<cudf::size_type> const probe_cols{5, 6, 7};  // probe block size 3
  std::vector<cudf::size_type> const build_cols{1, 2};

  SECTION("admissible probe blocks descend at either policy")
  {
    for (auto const join_type : {duckdb::JoinType::INNER,
                                 duckdb::JoinType::LEFT,
                                 duckdb::JoinType::SEMI,
                                 duckdb::JoinType::ANTI,
                                 duckdb::JoinType::RIGHT,
                                 duckdb::JoinType::OUTER,
                                 duckdb::JoinType::MARK}) {
      for (auto const policy : {kSipOn, kSipOff}) {
        auto const step =
          join_block_descent(join_type, probe_cols, build_cols, /*output_ordinal=*/1, policy);
        REQUIRE(step.has_value());
        REQUIRE(step->child_index == 0);
        REQUIRE(step->child_ordinal == 6);  // probe_cols[1]
      }
    }
  }

  SECTION("the remaining join types refuse the probe block")
  {
    for (auto const join_type : {duckdb::JoinType::SINGLE,
                                 duckdb::JoinType::RIGHT_SEMI,
                                 duckdb::JoinType::RIGHT_ANTI,
                                 duckdb::JoinType::INVALID}) {
      REQUIRE(join_block_descent(join_type, probe_cols, build_cols, 1, kSipOn) == std::nullopt);
    }
  }
}

TEST_CASE("join_block_descent maps an admissible build-block output into the build child",
          "[dynamic_filter][placement]")
{
  std::vector<cudf::size_type> const probe_cols{5, 6, 7};  // probe block size 3
  std::vector<cudf::size_type> const build_cols{1, 2};

  SECTION("INNER/RIGHT descend directly; LEFT/OUTER rely on equality admission")
  {
    for (auto const join_type : {duckdb::JoinType::INNER,
                                 duckdb::JoinType::LEFT,
                                 duckdb::JoinType::RIGHT,
                                 duckdb::JoinType::OUTER}) {
      // output ordinal 4 -> build ordinal 4 - 3 = 1 -> build_cols[1] = 2
      auto const step =
        join_block_descent(join_type, probe_cols, build_cols, /*output_ordinal=*/4, kSipOn);
      REQUIRE(step.has_value());
      REQUIRE(step->child_index == 1);
      REQUIRE(step->child_ordinal == 2);
    }
  }

  SECTION("the policy bit gates every build-block hop")
  {
    // The build hop is the SIP capability: the same INNER ordinal that descends under the SIP
    // policy refuses without it, while probe hops are policy-independent (previous test).
    REQUIRE(join_block_descent(duckdb::JoinType::INNER, probe_cols, build_cols, 4, kSipOff) ==
            std::nullopt);
    REQUIRE(
      join_block_descent(duckdb::JoinType::INNER, probe_cols, build_cols, 4, kSipOn).has_value());
  }

  SECTION("the remaining join types refuse the build block")
  {
    // SEMI/ANTI emit no RHS block, MARK emits a synthetic mark, and right-semi/anti require their
    // RHS-only output layout to be mapped explicitly.
    for (auto const join_type : {duckdb::JoinType::SEMI,
                                 duckdb::JoinType::ANTI,
                                 duckdb::JoinType::MARK,
                                 duckdb::JoinType::SINGLE,
                                 duckdb::JoinType::RIGHT_SEMI,
                                 duckdb::JoinType::RIGHT_ANTI,
                                 duckdb::JoinType::INVALID}) {
      REQUIRE(join_block_descent(join_type, probe_cols, build_cols, 4, kSipOn) == std::nullopt);
    }
  }
}

TEST_CASE("join_block_descent refuses a build ordinal past the build block",
          "[dynamic_filter][placement]")
{
  std::vector<cudf::size_type> const probe_cols{5, 6, 7};  // probe block size 3
  std::vector<cudf::size_type> const build_cols{1, 2};     // build block size 2
  // output ordinal 5 -> build ordinal 2, but the build block has only columns {0, 1}
  REQUIRE(join_block_descent(duckdb::JoinType::INNER, probe_cols, build_cols, 5, kSipOn) ==
          std::nullopt);
}

// ---------------------------------------------------------------------------------------------
// scan_route_join_type_admissible
// ---------------------------------------------------------------------------------------------

TEST_CASE("scan_route_join_type_admissible mirrors DuckDB's producer join-type gate",
          "[dynamic_filter][placement][discovery]")
{
  for (auto const join_type :
       {duckdb::JoinType::INNER, duckdb::JoinType::RIGHT, duckdb::JoinType::SEMI}) {
    REQUIRE(scan_route_join_type_admissible(join_type));
  }
  for (auto const join_type : {duckdb::JoinType::LEFT,
                               duckdb::JoinType::OUTER,
                               duckdb::JoinType::ANTI,
                               duckdb::JoinType::MARK,
                               duckdb::JoinType::SINGLE,
                               duckdb::JoinType::RIGHT_SEMI,
                               duckdb::JoinType::RIGHT_ANTI,
                               duckdb::JoinType::INVALID}) {
    REQUIRE_FALSE(scan_route_join_type_admissible(join_type));
  }
}

// ---------------------------------------------------------------------------------------------
// descent_steps
// ---------------------------------------------------------------------------------------------

TEST_CASE("descent_steps descends a pass-through projection output", "[dynamic_filter][placement]")
{
  auto const projection = make_projection(make_select_list(make_reference(2), make_reference(8)));
  auto const steps      = descent_steps(*projection, /*output_ordinal=*/1, kSipOff);
  REQUIRE(steps.size() == 1);
  REQUIRE(steps[0].child_index == 0);
  REQUIRE(steps[0].child_ordinal == 8);
}

TEST_CASE("descent_steps refuses a projection output that is not a plain reference",
          "[dynamic_filter][placement]")
{
  auto const projection = make_projection(make_select_list(make_cast(2)));
  REQUIRE(descent_steps(*projection, /*output_ordinal=*/0, kSipOff).empty());
}

TEST_CASE("descent_steps refuses an output ordinal past the projection width",
          "[dynamic_filter][placement]")
{
  auto const projection = make_projection(make_select_list(make_reference(2)));
  REQUIRE(descent_steps(*projection, /*output_ordinal=*/1, kSipOff).empty());
}

TEST_CASE("descent_steps refuses a non-descent operator", "[dynamic_filter][placement]")
{
  auto const scan = make_scan(/*width=*/4);
  REQUIRE(descent_steps(*scan, /*output_ordinal=*/0, kSipOff).empty());
}

TEST_CASE("descent_steps passes through a dynamic-filter endpoint", "[dynamic_filter][placement]")
{
  // The endpoint is a row mask: its output space is its input space, so the traced ordinal
  // survives the hop unchanged.
  auto const endpoint = make_endpoint_operator(/*width=*/6);
  auto const steps    = descent_steps(*endpoint, /*output_ordinal=*/4, kSipOff);
  REQUIRE(steps.size() == 1);
  REQUIRE(steps[0].child_index == 0);
  REQUIRE(steps[0].child_ordinal == 4);
}

TEST_CASE("descent_steps passes through a FILTER at the same ordinal (passthrough output)",
          "[dynamic_filter][placement][discovery]")
{
  auto const filter = make_passthrough_filter(/*width=*/5);
  auto const steps  = descent_steps(*filter, /*output_ordinal=*/3, kSipOff);
  REQUIRE(steps.size() == 1);
  REQUIRE(steps[0].child_index == 0);
  REQUIRE(steps[0].child_ordinal == 3);
}

TEST_CASE("descent_steps remaps a FILTER gather output to its input column",
          "[dynamic_filter][placement][discovery]")
{
  // output 0 -> input 4, output 1 -> input 2
  auto const filter = make_gather_filter({4, 2});

  auto const first = descent_steps(*filter, /*output_ordinal=*/0, kSipOff);
  REQUIRE(first.size() == 1);
  REQUIRE(first[0].child_ordinal == 4);

  auto const second = descent_steps(*filter, /*output_ordinal=*/1, kSipOff);
  REQUIRE(second.size() == 1);
  REQUIRE(second[0].child_ordinal == 2);
}

TEST_CASE("descent_steps refuses a FILTER gather ordinal past the gather width",
          "[dynamic_filter][placement][discovery]")
{
  auto const filter = make_gather_filter({4, 2});
  REQUIRE(descent_steps(*filter, /*output_ordinal=*/2, kSipOff).empty());
}

TEST_CASE("descent_steps fans out through a UNION into every child at the same ordinal",
          "[dynamic_filter][placement][discovery]")
{
  auto union_node = make_typed_double(sirius::op::SiriusPhysicalOperatorType::UNION, /*width=*/3);
  union_node->children.push_back(make_scan(3));
  union_node->children.push_back(make_scan(3));
  union_node->children.push_back(make_scan(3));

  auto const steps = descent_steps(*union_node, /*output_ordinal=*/2, kSipOff);
  REQUIRE(steps.size() == 3);
  for (std::size_t child_index = 0; child_index < steps.size(); ++child_index) {
    REQUIRE(steps[child_index].child_index == child_index);
    REQUIRE(steps[child_index].child_ordinal == 2);  // positional alignment, no remap
  }
}

// ---------------------------------------------------------------------------------------------
// trace_probe_key
// ---------------------------------------------------------------------------------------------

TEST_CASE("trace_probe_key bottoms out through a projection + FILTER chain with the exit ordinal",
          "[dynamic_filter][placement][discovery]")
{
  // proj_top: output 0 -> input 3
  // filter:   gather output 3 -> input 7
  // scan(8):  TABLE_SCAN terminal, exit ordinal 7 (the scan's own output space)
  auto scan = make_typed_double(sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN, /*width=*/8);
  auto* scan_node = scan.get();

  auto filter = make_gather_filter({9, 9, 9, 7});
  filter->children.push_back(std::move(scan));

  auto proj_top = make_projection(make_select_list(make_reference(3)));
  proj_top->children.push_back(std::move(filter));

  auto const terminals = trace_probe_key(*proj_top, /*a0=*/0, kSipOff);
  REQUIRE(terminals.size() == 1);
  REQUIRE(terminals[0].node == scan_node);
  REQUIRE(terminals[0].ordinal == 7);
}

TEST_CASE("trace_probe_key reports the refusing node as the terminal",
          "[dynamic_filter][placement][discovery]")
{
  // proj_top: output 0 -> input 1; proj_mid: output 1 is computed -> refuses at proj_mid.
  auto scan      = make_scan(/*width=*/4);
  auto proj_mid  = make_projection(make_select_list(make_reference(0), make_cast(0)));
  auto* mid_node = proj_mid.get();
  proj_mid->children.push_back(std::move(scan));

  auto proj_top = make_projection(make_select_list(make_reference(1)));
  proj_top->children.push_back(std::move(proj_mid));

  auto const terminals = trace_probe_key(*proj_top, /*a0=*/0, kSipOff);
  REQUIRE(terminals.size() == 1);
  REQUIRE(terminals[0].node == mid_node);
  REQUIRE(terminals[0].ordinal == 1);
}

TEST_CASE("trace_probe_key yields one terminal per UNION branch, mixed kinds included",
          "[dynamic_filter][placement][discovery]")
{
  // Branch 0 reaches a TABLE_SCAN through a projection remap; branch 1 stops at a DUMMY_SCAN. One
  // admitted key therefore produces a scan-binding site and an endpoint site.
  auto scan = make_typed_double(sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN, /*width=*/6);
  auto* scan_node = scan.get();
  auto branch0 =
    make_projection(make_select_list(make_reference(5), make_reference(0), make_reference(1)));
  branch0->children.push_back(std::move(scan));

  auto branch1     = make_scan(/*width=*/3);
  auto* dummy_node = branch1.get();

  auto union_node = make_typed_double(sirius::op::SiriusPhysicalOperatorType::UNION, /*width=*/3);
  union_node->children.push_back(std::move(branch0));
  union_node->children.push_back(std::move(branch1));

  auto const terminals = trace_probe_key(*union_node, /*a0=*/0, kSipOff);
  REQUIRE(terminals.size() == 2);
  REQUIRE(terminals[0].node == scan_node);
  REQUIRE(terminals[0].ordinal == 5);  // remapped by branch 0's projection
  REQUIRE(terminals[1].node == dummy_node);
  REQUIRE(terminals[1].ordinal == 0);  // positional identity into branch 1
}

TEST_CASE("trace_probe_key and place_endpoint agree on the site for a spliced key",
          "[dynamic_filter][placement][discovery]")
{
  auto make_subtree = [](sirius::op::sirius_physical_operator** scan_out) {
    auto scan = make_scan(/*width=*/8);
    *scan_out = scan.get();
    auto proj = make_projection(make_select_list(make_reference(2), make_reference(5)));
    proj->children.push_back(std::move(scan));
    return proj;
  };

  sirius::op::sirius_physical_operator* traced_scan = nullptr;
  auto traced_tree                                  = make_subtree(&traced_scan);
  auto const terminals = trace_probe_key(*traced_tree, /*a0=*/1, kSipOff);
  REQUIRE(terminals.size() == 1);
  REQUIRE(terminals[0].node == traced_scan);
  REQUIRE(terminals[0].ordinal == 5);

  sirius::op::sirius_physical_operator* placed_scan = nullptr;
  auto placed_tree                                  = make_subtree(&placed_scan);
  std::vector<sirius::op::sirius_physical_operator*> endpoints;
  auto const placed = place_endpoint(
    std::move(placed_tree), /*a0=*/1, kSipOff, recording_endpoint_factory(endpoints));
  REQUIRE(placed.site_ordinals == std::vector<std::size_t>{5});
  REQUIRE(endpoints.size() == 1);
  REQUIRE(endpoints[0]->children[0].get() == placed_scan);
}

// ---------------------------------------------------------------------------------------------
// place_endpoint
// ---------------------------------------------------------------------------------------------

TEST_CASE("place_endpoint wraps the root when the root refuses (the floor)",
          "[dynamic_filter][placement]")
{
  // Output ordinal 1 is a computed cast, so the root refuses and the endpoint wraps the root.
  auto projection = make_projection(make_select_list(make_reference(0), make_cast(1)));
  auto* root      = projection.get();

  sirius::op::sirius_physical_operator const* sited = nullptr;
  sirius::op::sirius_physical_operator* endpoint    = nullptr;
  auto const placed                                 = place_endpoint(
    std::move(projection), /*a0=*/1, kSipOff, capturing_endpoint_factory(sited, endpoint));

  REQUIRE(sited == root);
  REQUIRE(placed.site_ordinals == std::vector<std::size_t>{1});
  REQUIRE(placed.subtree.get() == endpoint);  // the endpoint is the new root
  REQUIRE(endpoint->children.size() == 1);
  REQUIRE(endpoint->children[0].get() == root);
}

TEST_CASE("place_endpoint descends a projection chain and wraps the deepest site",
          "[dynamic_filter][placement]")
{
  // proj_top: output 0 -> input 3
  // proj_mid: output 3 -> input 7
  // scan:     refuses -> the endpoint wraps the scan, ordinal remapped 0 -> 3 -> 7
  auto scan       = make_scan(/*width=*/8);
  auto* scan_node = scan.get();

  auto proj_mid = make_projection(
    make_select_list(make_reference(0), make_reference(0), make_reference(0), make_reference(7)));
  proj_mid->children.push_back(std::move(scan));

  auto proj_top  = make_projection(make_select_list(make_reference(3)));
  auto* top_node = proj_top.get();
  proj_top->children.push_back(std::move(proj_mid));

  sirius::op::sirius_physical_operator const* sited = nullptr;
  sirius::op::sirius_physical_operator* endpoint    = nullptr;
  auto const placed                                 = place_endpoint(
    std::move(proj_top), /*a0=*/0, kSipOff, capturing_endpoint_factory(sited, endpoint));

  REQUIRE(sited == scan_node);
  REQUIRE(placed.site_ordinals == std::vector<std::size_t>{7});
  REQUIRE(placed.subtree.get() == top_node);  // root unchanged; the endpoint is spliced below
  REQUIRE(endpoint->children.size() == 1);
  REQUIRE(endpoint->children[0].get() == scan_node);
}

TEST_CASE("place_endpoint stops at the deepest pass-through when a mid-tree hop refuses",
          "[dynamic_filter][placement]")
{
  // proj_top: output 0 -> input 1 (descends)
  // proj_mid: output 1 is a computed cast -> refuses; the endpoint wraps proj_mid at ordinal 1
  auto scan      = make_scan(/*width=*/4);
  auto proj_mid  = make_projection(make_select_list(make_reference(0), make_cast(0)));
  auto* mid_node = proj_mid.get();
  proj_mid->children.push_back(std::move(scan));

  auto proj_top = make_projection(make_select_list(make_reference(1)));
  proj_top->children.push_back(std::move(proj_mid));

  sirius::op::sirius_physical_operator const* sited = nullptr;
  sirius::op::sirius_physical_operator* endpoint    = nullptr;
  auto const placed                                 = place_endpoint(
    std::move(proj_top), /*a0=*/0, kSipOff, capturing_endpoint_factory(sited, endpoint));

  REQUIRE(sited == mid_node);
  REQUIRE(placed.site_ordinals == std::vector<std::size_t>{1});
  REQUIRE(endpoint->children[0].get() == mid_node);
}

TEST_CASE("place_endpoint stops when the accepted child is missing", "[dynamic_filter][placement]")
{
  // The projection accepts (descend into child 0) but has no children, so the endpoint wraps it.
  auto projection = make_projection(make_select_list(make_reference(0)));
  auto* root      = projection.get();

  sirius::op::sirius_physical_operator const* sited = nullptr;
  sirius::op::sirius_physical_operator* endpoint    = nullptr;
  auto const placed                                 = place_endpoint(
    std::move(projection), /*a0=*/0, kSipOff, capturing_endpoint_factory(sited, endpoint));

  REQUIRE(sited == root);
  REQUIRE(placed.site_ordinals == std::vector<std::size_t>{0});
  REQUIRE(endpoint->children[0].get() == root);
}

TEST_CASE("place_endpoint descends a FILTER and sites below it",
          "[dynamic_filter][placement][discovery]")
{
  // Q19-class shape: a residual-predicate FILTER stands between the producing join and the probe
  // scan.
  auto scan         = make_scan(/*width=*/6);
  auto* scan_node   = scan.get();
  auto filter       = make_gather_filter({5, 3});
  auto* filter_node = filter.get();
  filter->children.push_back(std::move(scan));

  sirius::op::sirius_physical_operator const* sited = nullptr;
  sirius::op::sirius_physical_operator* endpoint    = nullptr;
  auto const placed                                 = place_endpoint(
    std::move(filter), /*a0=*/1, kSipOff, capturing_endpoint_factory(sited, endpoint));

  REQUIRE(sited == scan_node);
  REQUIRE(placed.site_ordinals == std::vector<std::size_t>{3});
  REQUIRE(placed.subtree.get() == filter_node);
  REQUIRE(filter_node->children[0].get() == endpoint);
  REQUIRE(endpoint->children[0].get() == scan_node);
}

TEST_CASE("place_endpoint splices one endpoint per UNION branch in traversal order",
          "[dynamic_filter][placement][discovery]")
{
  // Branch 0 remaps 1 -> 4 through a projection before its scan; branch 1 refuses at its scan
  // directly, keeping the positional ordinal 1.
  auto scan0 = make_scan(/*width=*/5);
  auto* s0   = scan0.get();
  auto proj0 = make_projection(make_select_list(make_reference(0), make_reference(4)));
  proj0->children.push_back(std::move(scan0));

  auto scan1 = make_scan(/*width=*/2);
  auto* s1   = scan1.get();

  auto union_node = make_typed_double(sirius::op::SiriusPhysicalOperatorType::UNION, /*width=*/2);
  auto* root      = union_node.get();
  union_node->children.push_back(std::move(proj0));
  union_node->children.push_back(std::move(scan1));

  std::vector<sirius::op::sirius_physical_operator*> endpoints;
  auto const placed =
    place_endpoint(std::move(union_node), /*a0=*/1, kSipOff, recording_endpoint_factory(endpoints));

  REQUIRE(placed.subtree.get() == root);
  REQUIRE(placed.site_ordinals == std::vector<std::size_t>{4, 1});
  REQUIRE(endpoints.size() == 2);
  // Factory invocations zip with site_ordinals.
  REQUIRE(endpoints[0]->children[0].get() == s0);
  REQUIRE(endpoints[1]->children[0].get() == s1);
}

TEST_CASE("a second placement descends past the endpoint the first placed",
          "[dynamic_filter][placement]")
{
  // A producer places one endpoint per direct-routed key into the same probe subtree, so placement
  // depth must not depend on key order.
  //
  // proj_top: output 0 -> input 2, output 1 -> input 5
  // scan:     refuses, so both keys site at the scan
  auto scan       = make_scan(/*width=*/8);
  auto* scan_node = scan.get();
  auto proj_top   = make_projection(make_select_list(make_reference(2), make_reference(5)));
  proj_top->children.push_back(std::move(scan));

  std::vector<sirius::op::sirius_physical_operator*> first_endpoints;
  auto first = place_endpoint(
    std::move(proj_top), /*a0=*/0, kSipOff, recording_endpoint_factory(first_endpoints));
  REQUIRE(first.site_ordinals == std::vector<std::size_t>{2});
  REQUIRE(first_endpoints.size() == 1);

  std::vector<sirius::op::sirius_physical_operator*> second_endpoints;
  auto const second = place_endpoint(
    std::move(first.subtree), /*a0=*/1, kSipOff, recording_endpoint_factory(second_endpoints));
  REQUIRE(second.site_ordinals == std::vector<std::size_t>{5});
  REQUIRE(second_endpoints.size() == 1);

  // Both endpoints stack directly above the scan, in placement order.
  REQUIRE(first_endpoints[0]->children.size() == 1);
  REQUIRE(first_endpoints[0]->children[0].get() == second_endpoints[0]);
  REQUIRE(second_endpoints[0]->children.size() == 1);
  REQUIRE(second_endpoints[0]->children[0].get() == scan_node);
}

// ---------------------------------------------------------------------------------------------
// EXCEPT ALL / INTERSECT ALL (REPLICATE over a grouped tag sum)
// ---------------------------------------------------------------------------------------------

TEST_CASE("descent_steps maps a REPLICATE output to the same input column",
          "[dynamic_filter][placement][discovery]")
{
  auto const plan = make_set_operation_all(/*tag_count=*/1);

  auto const steps = descent_steps(*plan.root, /*output_ordinal=*/1, kSipOff);
  REQUIRE(steps.size() == 1);
  REQUIRE(steps[0].child_index == 0);
  REQUIRE(steps[0].child_ordinal == 1);

  // Ordinal kSetOpAllKeyCount is the copy count in REPLICATE's input, not an output.
  REQUIRE(descent_steps(*plan.root, kSetOpAllKeyCount, kSipOff).empty());
}

TEST_CASE("trace_probe_key reaches both arms of an INTERSECT ALL or EXCEPT ALL probe side",
          "[dynamic_filter][placement][discovery]")
{
  for (std::size_t const tag_count : {std::size_t{2}, std::size_t{1}}) {
    INFO("tag_count = " << tag_count);
    auto const plan = make_set_operation_all(tag_count);
    for (std::size_t key = 0; key < kSetOpAllKeyCount; ++key) {
      INFO("key = " << key);
      auto const terminals = trace_probe_key(*plan.root, key, kSipOn);
      REQUIRE(terminals.size() == 2);
      for (std::size_t arm = 0; arm < 2; ++arm) {
        REQUIRE(terminals[arm].node == plan.arm_scans[arm]);
        REQUIRE(terminals[arm].ordinal == kSetOpAllArmKeys[arm][key]);
      }
    }
  }
}

TEST_CASE("place_endpoint splices one endpoint below each arm of an EXCEPT ALL",
          "[dynamic_filter][placement][discovery]")
{
  auto plan        = make_set_operation_all(/*tag_count=*/1);
  auto* const root = plan.root.get();

  std::vector<sirius::op::sirius_physical_operator*> endpoints;
  auto const placed =
    place_endpoint(std::move(plan.root), /*a0=*/0, kSipOff, recording_endpoint_factory(endpoints));

  REQUIRE(placed.subtree.get() == root);
  REQUIRE(placed.site_ordinals ==
          std::vector<std::size_t>{kSetOpAllArmKeys[0][0], kSetOpAllArmKeys[1][0]});
  REQUIRE(endpoints.size() == 2);
  REQUIRE(endpoints[0]->children[0].get() == plan.arm_scans[0]);
  REQUIRE(endpoints[1]->children[0].get() == plan.arm_scans[1]);
}

TEST_CASE("descent below REPLICATE refuses the copy count and the tag columns",
          "[dynamic_filter][placement][discovery]")
{
  constexpr std::size_t tag_count = 2;
  auto const plan                 = make_set_operation_all(tag_count);
  auto const& count_projection    = *plan.root->children[0];
  auto const& aggregate           = *count_projection.children[0];
  auto const& union_node          = *aggregate.children[0];

  REQUIRE(descent_steps(count_projection, kSetOpAllKeyCount, kSipOn).empty());
  for (std::size_t tag = 0; tag < tag_count; ++tag) {
    auto const ordinal = kSetOpAllKeyCount + tag;
    INFO("ordinal = " << ordinal);
    REQUIRE(descent_steps(aggregate, ordinal, kSipOn).empty());
    for (auto const& tagged : union_node.children) {
      REQUIRE(descent_steps(*tagged, ordinal, kSipOn).empty());
    }
  }
}
