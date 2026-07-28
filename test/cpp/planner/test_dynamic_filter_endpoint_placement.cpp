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
 * Unit tests for the join-edge dynamic-filter placement trace
 * (`planner/dynamic_filter/dynamic_filter_endpoint_placement.hpp`).
 *
 * The three descent rules -- projection_reference_input, group_by_key_input, and
 * join_block_descent -- are pure functions of plain arguments, so they are exercised
 * directly and exhaustively. The dispatch (pass_through_step) and the driver
 * (resolve_endpoint_site) are exercised over PROJECTION chains, which are cheap to
 * construct and give full control over the traced ordinal at each hop.
 *
 * HASH_JOIN and HASH_GROUP_BY operators are deliberately NOT constructed here: their
 * constructors build cuDF aggregate definitions / output blocks and are heavy and
 * fragile to stand up in a unit test. Their pass_through_step cases are thin
 * delegations to join_block_descent / group_by_key_input (both tested below), and the
 * operator-reading paths are covered end to end by the pipeline-shape tests
 * (test/cpp/pipeline/test_pipeline_dynamic_filter_native_shape.cpp, which pin the Q5
 * build-side and Q3 group-by SIP plans). The scans below serve only as descent
 * terminators, not as a statement about where a real endpoint would land.
 */

#include "expression/ast/cast.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "helper/logical_type.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_projection.hpp"
#include "planner/dynamic_filter/dynamic_filter_endpoint_placement.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb/common/enums/join_type.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using sirius::planner::group_by_key_input;
using sirius::planner::join_block_descent;
using sirius::planner::pass_through_step;
using sirius::planner::projection_reference_input;
using sirius::planner::resolve_endpoint_site;

// The placement code never inspects a column's type, but the operators require a concrete one.
sirius::logical_type int_type() { return sirius::logical_type::make(sirius::type_id::INTEGER); }

// A pass-through projection output: a plain reference to input column `col`.
std::unique_ptr<sirius::ast::node> make_reference(std::uint32_t col)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{col, int_type()});
}

// A computed projection output: a cast of a reference. Not a plain reference, so the projection
// rule must refuse it (crossing a cast is a deferred follow-up in the design).
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

// A leaf scan. pass_through_step refuses DUMMY_SCAN, so it terminates any descent.
duckdb::unique_ptr<sirius::op::sirius_physical_dummy_scan> make_scan(std::size_t width)
{
  duckdb::vector<sirius::logical_type> types(width, int_type());
  return duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(std::move(types),
                                                                   /*estimated_cardinality=*/0);
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

  SECTION("the probe block is value-preserving for INNER/LEFT/SEMI/ANTI/MARK")
  {
    for (auto const join_type : {duckdb::JoinType::INNER,
                                 duckdb::JoinType::LEFT,
                                 duckdb::JoinType::SEMI,
                                 duckdb::JoinType::ANTI,
                                 duckdb::JoinType::MARK}) {
      auto const step = join_block_descent(join_type, probe_cols, build_cols, /*output_ordinal=*/1);
      REQUIRE(step.has_value());
      REQUIRE(step->child_index == 0);
      REQUIRE(step->child_ordinal == 6);  // probe_cols[1]
    }
  }

  SECTION("the probe block refuses for types that null-pad the left block")
  {
    for (auto const join_type : {duckdb::JoinType::RIGHT,
                                 duckdb::JoinType::OUTER,
                                 duckdb::JoinType::SINGLE,
                                 duckdb::JoinType::RIGHT_SEMI,
                                 duckdb::JoinType::RIGHT_ANTI}) {
      REQUIRE(join_block_descent(join_type, probe_cols, build_cols, 1) == std::nullopt);
    }
  }
}

TEST_CASE(
  "join_block_descent maps a build-block output into the build child for INNER and LEFT only",
  "[dynamic_filter][placement]")
{
  std::vector<cudf::size_type> const probe_cols{5, 6, 7};  // probe block size 3
  std::vector<cudf::size_type> const build_cols{1, 2};

  SECTION("INNER and LEFT descend into the build child")
  {
    for (auto const join_type : {duckdb::JoinType::INNER, duckdb::JoinType::LEFT}) {
      // output ordinal 4 -> build ordinal 4 - 3 = 1 -> build_cols[1] = 2
      auto const step = join_block_descent(join_type, probe_cols, build_cols, /*output_ordinal=*/4);
      REQUIRE(step.has_value());
      REQUIRE(step->child_index == 1);
      REQUIRE(step->child_ordinal == 2);
    }
  }

  SECTION("every other join type refuses the build block")
  {
    // SEMI/ANTI emit no build block; MARK's build-block slot is the synthetic mark; the
    // right-preserving and SINGLE types are deferred/unsupported.
    for (auto const join_type : {duckdb::JoinType::SEMI,
                                 duckdb::JoinType::ANTI,
                                 duckdb::JoinType::MARK,
                                 duckdb::JoinType::RIGHT,
                                 duckdb::JoinType::OUTER,
                                 duckdb::JoinType::SINGLE,
                                 duckdb::JoinType::RIGHT_SEMI,
                                 duckdb::JoinType::RIGHT_ANTI}) {
      REQUIRE(join_block_descent(join_type, probe_cols, build_cols, 4) == std::nullopt);
    }
  }
}

TEST_CASE("join_block_descent refuses a build ordinal past the build block",
          "[dynamic_filter][placement]")
{
  std::vector<cudf::size_type> const probe_cols{5, 6, 7};  // probe block size 3
  std::vector<cudf::size_type> const build_cols{1, 2};     // build block size 2
  // output ordinal 5 -> build ordinal 2, but the build block has only columns {0, 1}
  REQUIRE(join_block_descent(duckdb::JoinType::INNER, probe_cols, build_cols, 5) == std::nullopt);
}

// ---------------------------------------------------------------------------------------------
// pass_through_step
// ---------------------------------------------------------------------------------------------

TEST_CASE("pass_through_step descends a pass-through projection output",
          "[dynamic_filter][placement]")
{
  auto const projection = make_projection(make_select_list(make_reference(2), make_reference(8)));
  auto const step       = pass_through_step(*projection, /*output_ordinal=*/1);
  REQUIRE(step.has_value());
  REQUIRE(step->child_index == 0);
  REQUIRE(step->child_ordinal == 8);
}

TEST_CASE("pass_through_step refuses a projection output that is not a plain reference",
          "[dynamic_filter][placement]")
{
  auto const projection = make_projection(make_select_list(make_cast(2)));
  REQUIRE(pass_through_step(*projection, /*output_ordinal=*/0) == std::nullopt);
}

TEST_CASE("pass_through_step refuses an output ordinal past the projection width",
          "[dynamic_filter][placement]")
{
  auto const projection = make_projection(make_select_list(make_reference(2)));
  REQUIRE(pass_through_step(*projection, /*output_ordinal=*/1) == std::nullopt);
}

TEST_CASE("pass_through_step refuses a non-descent operator", "[dynamic_filter][placement]")
{
  auto const scan = make_scan(/*width=*/4);
  REQUIRE(pass_through_step(*scan, /*output_ordinal=*/0) == std::nullopt);
}

// ---------------------------------------------------------------------------------------------
// resolve_endpoint_site
// ---------------------------------------------------------------------------------------------

TEST_CASE("resolve_endpoint_site returns the root as the floor when the root refuses",
          "[dynamic_filter][placement]")
{
  // Output ordinal 1 is a computed cast, so the root refuses and the site is the root at a0.
  auto projection = make_projection(make_select_list(make_reference(0), make_cast(1)));
  auto* root      = projection.get();

  auto const site = resolve_endpoint_site(root, /*a0=*/1);
  REQUIRE(site.node == root);
  REQUIRE(site.ordinal == 1);
}

TEST_CASE("resolve_endpoint_site descends a projection chain, remapping the ordinal at each hop",
          "[dynamic_filter][placement]")
{
  // proj_top: output 0 -> input 3
  // proj_mid: output 3 -> input 7
  // scan:     refuses -> descent stops, deepest site is the scan at ordinal 7
  auto scan       = make_scan(/*width=*/8);
  auto* scan_node = scan.get();

  auto proj_mid = make_projection(
    make_select_list(make_reference(0), make_reference(0), make_reference(0), make_reference(7)));
  proj_mid->children.push_back(std::move(scan));

  auto proj_top = make_projection(make_select_list(make_reference(3)));
  proj_top->children.push_back(std::move(proj_mid));

  auto const site = resolve_endpoint_site(proj_top.get(), /*a0=*/0);
  REQUIRE(site.node == scan_node);
  REQUIRE(site.ordinal == 7);
}

TEST_CASE("resolve_endpoint_site stops at the deepest pass-through when a mid-tree hop refuses",
          "[dynamic_filter][placement]")
{
  // proj_top: output 0 -> input 1 (descends)
  // proj_mid: output 1 is a computed cast -> refuses; the site is proj_mid at ordinal 1
  auto scan     = make_scan(/*width=*/4);
  auto proj_mid = make_projection(make_select_list(make_reference(0), make_cast(0)));
  proj_mid->children.push_back(std::move(scan));
  auto* mid_node = proj_mid.get();

  auto proj_top = make_projection(make_select_list(make_reference(1)));
  proj_top->children.push_back(std::move(proj_mid));

  auto const site = resolve_endpoint_site(proj_top.get(), /*a0=*/0);
  REQUIRE(site.node == mid_node);
  REQUIRE(site.ordinal == 1);
}

TEST_CASE("resolve_endpoint_site stops when the accepted child is missing",
          "[dynamic_filter][placement]")
{
  // The projection accepts (descend into child 0) but has no children, so the trace stops at it.
  auto projection = make_projection(make_select_list(make_reference(0)));
  auto* root      = projection.get();

  auto const site = resolve_endpoint_site(root, /*a0=*/0);
  REQUIRE(site.node == root);
  REQUIRE(site.ordinal == 0);
}
