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

// [late_mat][lifetime] — how far a scanned column travels before it is read.
// No GPU.
//
// Two properties matter, and they pull in opposite directions. A column that is
// merely moved must keep travelling, or nothing is ever deferred. A column that
// is read must stop, or a deferral produces a rowid where a value was needed.
// The second is the dangerous one, so the analysis fails closed: a shape it
// does not model reads everything, which can only end a ride early.

#include <catch.hpp>
#include <expression/ast/comparison.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_projection.hpp>
#include <planner/late_mat_plan_pass.hpp>

#include <memory>
#include <vector>

using sirius::planner::analyze_column_lifetimes;

namespace {

sirius::logical_type int32_type() { return sirius::logical_type::make(sirius::type_id::INTEGER); }

duckdb::vector<sirius::logical_type> make_types(std::size_t n)
{
  duckdb::vector<sirius::logical_type> out;
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(int32_type());
  }
  return out;
}

std::unique_ptr<sirius::ast::node> ref(std::uint32_t column)
{
  return std::make_unique<sirius::ast::node>(sirius::ast::reference{column, int32_type()});
}

// The plan generator links parents in production; these expose the same field
// so a test can build a chain without one.
struct fake_scan : sirius::op::sirius_physical_operator {
  explicit fake_scan(std::size_t columns)
    : sirius_physical_operator(
        sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN, make_types(columns), 0)
  {
  }
  void link(sirius_physical_operator* parent) { _parent_op = parent; }
};

/// An operator shape the analysis does not model, so it must consume.
struct opaque_op : sirius::op::sirius_physical_operator {
  explicit opaque_op(std::size_t columns)
    : sirius_physical_operator(
        sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY, make_types(columns), 0)
  {
  }
  void link(sirius_physical_operator* parent) { _parent_op = parent; }
};

struct test_filter : sirius::op::sirius_physical_filter {
  using sirius_physical_filter::sirius_physical_filter;
  void link(sirius_physical_operator* parent) { _parent_op = parent; }
};

struct test_projection : sirius::op::sirius_physical_projection {
  using sirius_physical_projection::sirius_physical_projection;
  void link(sirius_physical_operator* parent) { _parent_op = parent; }
};

}  // namespace

TEST_CASE("a column nothing reads travels to the top of the plan", "[late_mat][lifetime]")
{
  fake_scan scan(3);
  // Two projections that only re-order: the classic payload ride.
  duckdb::vector<std::unique_ptr<sirius::ast::node>> first;
  first.push_back(ref(2));
  first.push_back(ref(0));
  first.push_back(ref(1));
  test_projection p1(make_types(3), std::move(first), 0);

  duckdb::vector<std::unique_ptr<sirius::ast::node>> second;
  second.push_back(ref(1));
  second.push_back(ref(2));
  second.push_back(ref(0));
  test_projection p2(make_types(3), std::move(second), 0);

  scan.link(&p1);
  p1.link(&p2);

  auto const lives = analyze_column_lifetimes(scan);
  REQUIRE(lives.size() == 3);
  for (auto const& life : lives) {
    REQUIRE(life.first_reader == nullptr);  // never read
    REQUIRE(life.boundaries == 2);
  }
  // Position is tracked, not assumed: scan col 0 sits at p1's output 1
  // (p1 slot 1 holds ref(0)), and p2's slot 0 holds ref(1), so it ends at 0.
  REQUIRE(lives[0].position_at_reader == 0);
}

TEST_CASE("a filter reads its predicate's columns and moves the rest", "[late_mat][lifetime]")
{
  fake_scan scan(3);
  // WHERE col0 = col0 — reads column 0 only.
  auto predicate = std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::equal, ref(0), ref(0)});
  test_filter filter(make_types(3), std::move(predicate), 0);
  scan.link(&filter);

  auto const lives = analyze_column_lifetimes(scan);
  REQUIRE(lives[0].first_reader == &filter);  // in the predicate: read
  REQUIRE(lives[0].boundaries == 1);
  REQUIRE(lives[1].first_reader == nullptr);  // merely carried past
  REQUIRE(lives[2].first_reader == nullptr);
}

TEST_CASE("a projection that computes with a column reads it", "[late_mat][lifetime]")
{
  fake_scan scan(2);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
  list.push_back(ref(0));  // moved
  list.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::equal, ref(1), ref(1)}));  // computed
  test_projection projection(make_types(2), std::move(list), 0);
  scan.link(&projection);

  auto const lives = analyze_column_lifetimes(scan);
  REQUIRE(lives[0].first_reader == nullptr);
  REQUIRE(lives[1].first_reader == &projection);
}

TEST_CASE("an unmodelled operator consumes everything", "[late_mat][lifetime]")
{
  // The property that makes this extensible: adding a shape can only lengthen
  // lifetimes, so forgetting one is never a correctness bug.
  fake_scan scan(2);
  opaque_op group_by(2);
  scan.link(&group_by);

  auto const lives = analyze_column_lifetimes(scan);
  REQUIRE(lives[0].first_reader == &group_by);
  REQUIRE(lives[1].first_reader == &group_by);
  REQUIRE(lives[0].boundaries == 1);
}

TEST_CASE("a column a projection drops stops there", "[late_mat][lifetime]")
{
  fake_scan scan(2);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
  list.push_back(ref(0));  // column 1 is simply not projected
  test_projection projection(make_types(1), std::move(list), 0);
  scan.link(&projection);

  auto const lives = analyze_column_lifetimes(scan);
  REQUIRE(lives[0].first_reader == nullptr);
  // Dropped rather than read, but for a deferral the answer is the same: it
  // travels no further, so there is nothing downstream to defer it to.
  REQUIRE(lives[1].first_reader == &projection);
}

TEST_CASE("an rhs column is offset by the lhs's EMITTED width", "[late_mat][lifetime]")
{
  using sirius::planner::join_output_position;

  // The lhs has four input columns but emits only two. An rhs column must be
  // offset by the two it emits, not the four it received — getting this wrong
  // does not refuse a deferral, it materializes into a position holding
  // something else.
  std::vector<int> const lhs{0, 3};
  std::vector<int> const rhs{1, 2, 5};

  REQUIRE(join_output_position(true, lhs, rhs, 0) == 0);
  REQUIRE(join_output_position(true, lhs, rhs, 3) == 1);
  REQUIRE(join_output_position(false, lhs, rhs, 1) == 2);  // 2 emitted lhs + slot 0
  REQUIRE(join_output_position(false, lhs, rhs, 2) == 3);
  REQUIRE(join_output_position(false, lhs, rhs, 5) == 4);
}

TEST_CASE("a column a join does not project out stops there", "[late_mat][lifetime]")
{
  using sirius::planner::join_output_position;
  std::vector<int> const lhs{0, 3};
  std::vector<int> const rhs{1};

  // Present on the other side's list, but not on its own: a column is only
  // carried by the side it actually arrived through.
  REQUIRE_FALSE(join_output_position(true, lhs, rhs, 1).has_value());
  REQUIRE_FALSE(join_output_position(false, lhs, rhs, 3).has_value());
  REQUIRE_FALSE(join_output_position(true, lhs, rhs, 7).has_value());
}

TEST_CASE("a join that emits no lhs columns still places its rhs", "[late_mat][lifetime]")
{
  using sirius::planner::join_output_position;
  std::vector<int> const none{};
  std::vector<int> const rhs{4, 9};
  REQUIRE(join_output_position(false, none, rhs, 4) == 0);
  REQUIRE(join_output_position(false, none, rhs, 9) == 1);
}

TEST_CASE("columns that stop at the same operator bundle together", "[late_mat][lifetime]")
{
  using sirius::planner::build_defer_candidates;

  // Two columns read by a group-by, one carried past it — the q10 shape in
  // miniature, where a bundle is what rides and what materializes.
  fake_scan scan(3);
  opaque_op group_by(3);
  scan.link(&group_by);

  auto const candidates = build_defer_candidates(scan, analyze_column_lifetimes(scan));
  REQUIRE(candidates.size() == 1);  // one reader, one slot
  REQUIRE(candidates[0].columns.size() == 3);
  REQUIRE(candidates[0].boundaries == 1);
  // Four-byte integers: three of them, less the eight-byte rowid.
  REQUIRE(candidates[0].net_value_bytes(8) == 3 * 4 - 8);
}

TEST_CASE("a column nothing reads is not a candidate", "[late_mat][lifetime]")
{
  using sirius::planner::build_defer_candidates;

  // No reader means nowhere to install the materializing half, and half a
  // deferral loses the column outright.
  fake_scan scan(2);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
  list.push_back(ref(0));
  list.push_back(ref(1));
  test_projection projection(make_types(2), std::move(list), 0);
  scan.link(&projection);

  REQUIRE(build_defer_candidates(scan, analyze_column_lifetimes(scan)).empty());
}

TEST_CASE("a variable-width column is valued below its real width", "[late_mat][lifetime]")
{
  using sirius::planner::estimated_value_bytes;

  // Understating can only refuse a bundle the policy would have taken;
  // overstating would install one that never repays the rowid.
  REQUIRE(estimated_value_bytes(sirius::logical_type::make(sirius::type_id::INTEGER)) == 4);
  REQUIRE(estimated_value_bytes(sirius::logical_type::make(sirius::type_id::BIGINT)) == 8);
  auto const varchar = estimated_value_bytes(sirius::logical_type::make(sirius::type_id::VARCHAR));
  REQUIRE(varchar > 0);
  REQUIRE(varchar < 72);  // below c_comment, the widest TPC-H string deferred
}
