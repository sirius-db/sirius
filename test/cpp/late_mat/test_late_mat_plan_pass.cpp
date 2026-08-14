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

#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>
#include <expression/ast/comparison.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_projection.hpp>
#include <planner/late_mat_plan_pass.hpp>

#include <memory>
#include <vector>

using sirius::planner::analyze_column_lifetimes;

namespace {

sirius::logical_type int32_type() { return sirius::logical_type::make(sirius::type_id::INTEGER); }
sirius::logical_type string_type() { return sirius::logical_type::make(sirius::type_id::VARCHAR); }

duckdb::vector<sirius::logical_type> make_types(std::size_t n)
{
  duckdb::vector<sirius::logical_type> out;
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(int32_type());
  }
  return out;
}

/// Wide enough that a bundle of two clears the policy's value floor — the
/// dimension columns a deferral is actually for.
duckdb::vector<sirius::logical_type> make_string_types(std::size_t n)
{
  duckdb::vector<sirius::logical_type> out;
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(string_type());
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

/// The same, with columns wide enough for a bundle to clear the value floor.
struct wide_scan : sirius::op::sirius_physical_operator {
  explicit wide_scan(std::size_t columns)
    : sirius_physical_operator(
        sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN, make_string_types(columns), 0)
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

struct test_join : sirius::op::sirius_physical_hash_join {
  using sirius_physical_hash_join::sirius_physical_hash_join;
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

TEST_CASE("a payload riding an outer join is deferrable but marked nullified",
          "[late_mat][lifetime]")
{
  using sirius::planner::join_output_position;

  // Refusing outer joins outright is simpler and costs every outer-shaped
  // query. A row that never matched gets a null rowid, so the column
  // materializes null — sound, provided the consumer is told.
  std::vector<int> const lhs{0, 1};
  std::vector<int> const rhs{0};
  REQUIRE(join_output_position(true, lhs, rhs, 1) == 1);
  REQUIRE(join_output_position(false, lhs, rhs, 0) == 2);
}

TEST_CASE("nothing is nullified on a ride with no outer join", "[late_mat][lifetime]")
{
  fake_scan scan(2);
  opaque_op reader(2);
  scan.link(&reader);

  auto const lives = analyze_column_lifetimes(scan);
  REQUIRE_FALSE(lives[0].nullified_on_ride);
  REQUIRE_FALSE(lives[1].nullified_on_ride);
}

namespace {

/// A chain of pass-through projections, so a ride can be made long enough to
/// clear the policy's boundary floor. Returned by value; the caller links them.
std::vector<std::unique_ptr<test_projection>> pass_through_chain(std::size_t columns,
                                                                 std::size_t links)
{
  std::vector<std::unique_ptr<test_projection>> chain;
  for (std::size_t i = 0; i < links; ++i) {
    duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
    for (std::size_t c = 0; c < columns; ++c) {
      list.push_back(ref(static_cast<std::uint32_t>(c)));
    }
    chain.push_back(
      std::make_unique<test_projection>(make_string_types(columns), std::move(list), 0));
  }
  return chain;
}

}  // namespace

TEST_CASE("a wide bundle over a long ride plans a deferral at its reader", "[late_mat][lifetime]")
{
  // Five string columns carried past four projections into an aggregate that
  // reads them: the q10 shape, and the one the measurements were taken on.
  wide_scan scan(5);
  opaque_op aggregate(5);
  auto chain = pass_through_chain(5, 4);
  scan.link(chain.front().get());
  for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
    chain[i]->link(chain[i + 1].get());
  }
  chain.back()->link(&aggregate);

  auto const planned = sirius::planner::plan_deferral(scan);
  REQUIRE(planned.installable());
  REQUIRE(planned.port == &aggregate);
  REQUIRE(planned.positions == std::vector<std::size_t>{0, 1, 2, 3, 4});
  REQUIRE(planned.restored_types.size() == 5);
  REQUIRE(planned.boundaries == 5);
  REQUIRE(planned.net_value_bytes == 5 * 24 - 8);
  REQUIRE(planned.census.size() == 1);
  REQUIRE(planned.census.front().installed());
}

TEST_CASE("a short ride plans nothing, and says so", "[late_mat][lifetime]")
{
  // Same wide bundle, read one boundary up. The ride does not repay the
  // materialization, and the refusal is recorded rather than dropped — a
  // deferral that silently did not happen looks like one that did nothing.
  wide_scan scan(5);
  opaque_op aggregate(5);
  scan.link(&aggregate);

  auto const planned = sirius::planner::plan_deferral(scan);
  REQUIRE_FALSE(planned.installable());
  REQUIRE(planned.port == nullptr);
  REQUIRE(planned.census.size() == 1);
  REQUIRE(planned.census.front().refusal == sirius::late_mat::defer_refusal::too_short_a_ride);
}

TEST_CASE("one rowid rides, so the narrower of two bundles is refused", "[late_mat][lifetime]")
{
  // Two readers, both far enough away and both wide enough. The substituted
  // output carries ONE rowid, so only one bundle is representable; the loser is
  // refused rather than dropped.
  wide_scan scan(5);
  // A filter over five columns whose predicate reads columns 0 and 1: they stop
  // there, the other three ride on to the aggregate.
  auto predicate = std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{sirius::comparison_type::equal, ref(0), ref(1)});
  test_filter filter(make_string_types(5), std::move(predicate), 0);
  opaque_op aggregate(5);

  auto lower = pass_through_chain(5, 4);
  auto upper = pass_through_chain(5, 4);
  scan.link(lower.front().get());
  for (std::size_t i = 0; i + 1 < lower.size(); ++i) {
    lower[i]->link(lower[i + 1].get());
  }
  lower.back()->link(&filter);
  filter.link(upper.front().get());
  for (std::size_t i = 0; i + 1 < upper.size(); ++i) {
    upper[i]->link(upper[i + 1].get());
  }
  upper.back()->link(&aggregate);

  auto const planned = sirius::planner::plan_deferral(scan);
  REQUIRE(planned.census.size() == 2);
  REQUIRE(planned.installable());
  // Three columns beat two, whichever slot was found first.
  REQUIRE(planned.port == &aggregate);
  REQUIRE(planned.positions == std::vector<std::size_t>{2, 3, 4});
  auto const refused = planned.census[0].installed() ? planned.census[1] : planned.census[0];
  REQUIRE(refused.refusal == sirius::late_mat::defer_refusal::second_bundle);
}

TEST_CASE("both halves install together, and only once", "[late_mat][lifetime]")
{
  using sirius::late_mat::make_defer_pair;

  wide_scan scan(3);
  opaque_op port(3);
  std::vector<cudf::data_type> const schema(3, cudf::data_type{cudf::type_id::STRING});
  std::vector<sirius::late_mat::column_origin> origins;
  for (int i = 0; i < 2; ++i) {
    sirius::late_mat::column_origin origin;
    origin.handle     = std::make_shared<sirius::late_mat::pin_entry_handle>("t", 1);
    origin.generation = 1;
    origins.push_back(std::move(origin));
  }

  REQUIRE(sirius::planner::install_deferral(scan, port, make_defer_pair(schema, {1, 2}, origins)));
  REQUIRE(scan.deferred_output().output_positions == std::vector<std::size_t>{1, 2});
  REQUIRE(port.port_directive().output_positions == std::vector<std::size_t>{1, 2});
  REQUIRE(port.port_directive().valid());

  // A second install would substitute against a schema the first one already
  // rewrote, and an invalid pair must stamp neither half.
  opaque_op other_port(3);
  REQUIRE_FALSE(
    sirius::planner::install_deferral(scan, other_port, make_defer_pair(schema, {1, 2}, origins)));
  REQUIRE(other_port.port_directive().empty());
  wide_scan fresh(3);
  REQUIRE_FALSE(sirius::planner::install_deferral(
    fresh, other_port, make_defer_pair(schema, {2, 1}, origins)));  // unordered
  REQUIRE(fresh.deferred_output().empty());
  REQUIRE(other_port.port_directive().empty());
}

TEST_CASE("a column an outer join could null is withheld from the weighing", "[late_mat][lifetime]")
{
  // Deferring it is sound — a null rowid must materialize a null — but the
  // materializer produces no nulls yet, so it is refused here, where it costs a
  // deferral, rather than at the far end where it would already be an answer.
  auto scan      = duckdb::make_uniq<wide_scan>(3);
  auto* scan_ptr = scan.get();
  auto build     = duckdb::make_uniq<wide_scan>(3);

  duckdb::LogicalDummyScan stub(0);
  stub.types = {duckdb::LogicalType::VARCHAR,
                duckdb::LogicalType::VARCHAR,
                duckdb::LogicalType::VARCHAR,
                duckdb::LogicalType::VARCHAR,
                duckdb::LogicalType::VARCHAR,
                duckdb::LogicalType::VARCHAR};
  duckdb::vector<sirius::join_condition> conditions;
  sirius::join_condition condition;
  condition.left  = std::make_unique<sirius::ast::node>(sirius::ast::reference{0, string_type()});
  condition.right = std::make_unique<sirius::ast::node>(sirius::ast::reference{0, string_type()});
  conditions.push_back(std::move(condition));
  // The scan is the RIGHT side of a LEFT join: an unmatched left row emits its
  // columns as null, so the payload riding from the right is nullified.
  test_join join(stub,
                 std::move(build),
                 std::move(scan),
                 std::move(conditions),
                 duckdb::JoinType::LEFT,
                 /*estimated_cardinality=*/1);
  opaque_op reader(6);
  scan_ptr->link(&join);
  join.link(&reader);

  auto const lives = analyze_column_lifetimes(*scan_ptr);
  REQUIRE(lives[1].nullified_on_ride);

  auto const planned = sirius::planner::plan_deferral(*scan_ptr);
  REQUIRE(planned.nullable_columns_skipped == 2);  // columns 1 and 2; 0 is the key
  REQUIRE_FALSE(planned.installable());
}
