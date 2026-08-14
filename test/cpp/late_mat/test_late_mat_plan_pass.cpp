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
#include <op/sirius_physical_concat.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_partition.hpp>
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

/// A key_source a partition can be built against that hashes on nothing: an
/// NLJ reports a single partition, so the constructor records no keys. That is
/// the payload-carrying partition — the one a ride must pass through.
struct keyless_key_source : sirius::op::sirius_physical_operator {
  keyless_key_source()
    : sirius_physical_operator(
        sirius::op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, make_types(1), 0)
  {
  }
};

/// The plumbing a real plan puts around every join and aggregate. Both are
/// pipeline sinks, so leaving one is the port crossing the policy weighs — and
/// both are positionally transparent, which is what lets a payload reach a
/// consumer several crossings up at all.
struct test_partition : sirius::op::sirius_physical_partition {
  using sirius_physical_partition::sirius_physical_partition;
  void link(sirius_physical_operator* parent) { _parent_op = parent; }
};

struct test_concat : sirius::op::sirius_physical_concat {
  using sirius_physical_concat::sirius_physical_concat;
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
    // Two projections, and neither materializes: the payload rode past both
    // inside one pipeline, which costs nothing and so counts as nothing.
    REQUIRE(life.port_crossings == 0);
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
  REQUIRE(lives[0].port_crossings == 0);      // same pipeline as the scan
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

/// A chain of partitions, which is what a ride long enough to clear the
/// policy's floor actually looks like: each is a pipeline sink, so each costs
/// one port crossing, and each is positionally transparent.
std::vector<std::unique_ptr<test_partition>> partition_chain(std::size_t columns,
                                                             std::size_t links,
                                                             keyless_key_source& keys)
{
  std::vector<std::unique_ptr<test_partition>> chain;
  for (std::size_t i = 0; i < links; ++i) {
    chain.push_back(
      std::make_unique<test_partition>(make_string_types(columns), 0, &keys, /*is_build=*/false));
  }
  return chain;
}

/// Link `scan -> chain[0] -> ... -> chain.back() -> top`.
template <typename Scan, typename Chain, typename Top>
void link_chain(Scan& scan, Chain& chain, Top& top)
{
  scan.link(chain.front().get());
  for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
    chain[i]->link(chain[i + 1].get());
  }
  chain.back()->link(&top);
}

}  // namespace

TEST_CASE("only a pipeline sink costs the ride anything", "[late_mat][lifetime]")
{
  // A projection and a filter hand their columns on inside one pipeline; a
  // partition writes them to a repository for the next one to read. Only the
  // second is what a deferral saves, so only the second is counted — otherwise
  // a chain of projections would look like a ride worth paying to defer.
  keyless_key_source keys;
  opaque_op reader(2);

  // scan -> partition -> reader
  wide_scan bare(2);
  test_partition bare_partition(make_string_types(2), 0, &keys, false);
  bare.link(&bare_partition);
  bare_partition.link(&reader);

  // scan -> projection -> partition -> reader: one more operator, the same
  // number of materializations.
  wide_scan projected(2);
  duckdb::vector<std::unique_ptr<sirius::ast::node>> list;
  list.push_back(ref(0));
  list.push_back(ref(1));
  test_projection projection(make_string_types(2), std::move(list), 0);
  test_partition projected_partition(make_string_types(2), 0, &keys, false);
  projected.link(&projection);
  projection.link(&projected_partition);
  projected_partition.link(&reader);

  auto const plain   = analyze_column_lifetimes(bare);
  auto const through = analyze_column_lifetimes(projected);
  REQUIRE(plain[0].first_reader == &reader);
  REQUIRE(through[0].first_reader == &reader);
  // The projection cost the ride nothing, so it is worth nothing to defer past.
  REQUIRE(through[0].port_crossings == plain[0].port_crossings);
  // And the partition passed the column through rather than ending its life.
  REQUIRE(through[0].position_at_reader == 0);
}

TEST_CASE("a join-child wrap concat carries the payload past", "[late_mat][lifetime]")
{
  // The concat gathers the partitions of ONE flow — the generator builds it at
  // exactly one site, wrapping one child of one join — so a payload crossing it
  // still comes from where the scan said it did, and the ride continues.
  duckdb::LogicalDummyScan stub(0);
  stub.types = duckdb::vector<duckdb::LogicalType>(2, duckdb::LogicalType::VARCHAR);
  duckdb::vector<sirius::join_condition> conditions;
  test_join downstream(stub,
                       duckdb::make_uniq<wide_scan>(2),
                       duckdb::make_uniq<wide_scan>(2),
                       std::move(conditions),
                       duckdb::JoinType::INNER,
                       /*estimated_cardinality=*/1);
  opaque_op reader(2);

  wide_scan scan(2);
  test_concat wrap(make_string_types(2), 0, &downstream, /*is_build=*/false);
  wrap.children.push_back(duckdb::make_uniq<wide_scan>(2));
  scan.link(&wrap);
  wrap.link(&reader);

  auto const lives = analyze_column_lifetimes(scan);
  REQUIRE(lives[0].first_reader == &reader);
  REQUIRE(lives[0].position_at_reader == 0);
  // A concat is a pipeline sink, so crossing it is what the ride is paying for.
  REQUIRE(lives[0].port_crossings == 1);
}

TEST_CASE("a join key may not be deferred, however far it rides", "[late_mat][lifetime]")
{
  // The danger is the PARTITION, not the join: it hashes the key to place a
  // row, and a rowid hashes differently from the value it stands for, so equal
  // keys would land in different partitions and the join would miss matches.
  // Stopping the ride at the join would not help — the port materializes at the
  // join's input, after the partition has already hashed. So the partition
  // itself reports the key read, from the positions it resolved at plan time.
  duckdb::LogicalDummyScan stub(0);
  stub.types = duckdb::vector<duckdb::LogicalType>(6, duckdb::LogicalType::VARCHAR);
  duckdb::vector<sirius::join_condition> conditions;
  sirius::join_condition condition;
  condition.left  = std::make_unique<sirius::ast::node>(sirius::ast::reference{0, string_type()});
  condition.right = std::make_unique<sirius::ast::node>(sirius::ast::reference{0, string_type()});
  conditions.push_back(std::move(condition));
  test_join join(stub,
                 duckdb::make_uniq<wide_scan>(3),
                 duckdb::make_uniq<wide_scan>(3),
                 std::move(conditions),
                 duckdb::JoinType::INNER,
                 /*estimated_cardinality=*/1);

  // Insert the wrap a real plan builds between the join and its probe child:
  // PARTITION(keys from the join) -> the scan.
  auto wrap = duckdb::make_uniq<test_partition>(
    make_string_types(3), 0, /*key_source=*/&join, /*is_build=*/false);
  auto* partition = wrap.get();
  REQUIRE(partition->partition_keys() == std::vector<int>{0});
  wrap->children.push_back(std::move(join.children[0]));
  auto* scan       = static_cast<wide_scan*>(wrap->children[0].get());
  join.children[0] = std::move(wrap);
  scan->link(partition);
  partition->link(&join);

  auto const lives = analyze_column_lifetimes(*scan);
  // Column 0 is the key, and the partition is where that is recognised — one
  // operator before the join could have told us.
  REQUIRE(lives[0].first_reader == partition);
  REQUIRE(lives[0].read_as_join_key);
  REQUIRE_FALSE(lives[1].read_as_join_key);  // merely carried beside it

  auto const planned = sirius::planner::plan_deferral(*scan);
  REQUIRE(planned.join_keys_skipped == 1);
}

TEST_CASE("a wide bundle over a long ride plans a deferral at its reader", "[late_mat][lifetime]")
{
  // Five string columns carried across four port crossings into an aggregate
  // that reads them: the q10 shape, and the one the measurements were taken on.
  wide_scan scan(5);
  opaque_op aggregate(5);
  keyless_key_source keys;
  auto chain = partition_chain(5, 3, keys);
  link_chain(scan, chain, aggregate);

  auto const planned = sirius::planner::plan_deferral(scan);
  REQUIRE(planned.installable());
  REQUIRE(planned.port == &aggregate);
  REQUIRE(planned.positions == std::vector<std::size_t>{0, 1, 2, 3, 4});
  REQUIRE(planned.restored_types.size() == 5);
  // Leaving the scan and leaving each of the three partitions.
  REQUIRE(planned.boundaries == 4);
  REQUIRE(planned.net_value_bytes == 5 * 24 - 8);
  REQUIRE(planned.census.size() == 1);
  REQUIRE(planned.census.front().installed());
}

TEST_CASE("a short ride plans nothing, and says so", "[late_mat][lifetime]")
{
  // Same wide bundle, read one crossing up. The ride does not repay the
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

  keyless_key_source keys;
  auto lower = partition_chain(5, 3, keys);
  auto upper = partition_chain(5, 3, keys);
  link_chain(scan, lower, filter);
  link_chain(filter, upper, aggregate);

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

  REQUIRE(sirius::planner::install_deferral(
    scan, port, make_defer_pair(schema, {1, 2}, schema, {1, 2}, origins)));
  REQUIRE(scan.deferred_output().output_positions == std::vector<std::size_t>{1, 2});
  REQUIRE(port.port_directive().output_positions == std::vector<std::size_t>{1, 2});
  REQUIRE(port.port_directive().valid());

  // A second install would substitute against a schema the first one already
  // rewrote, and an invalid pair must stamp neither half.
  opaque_op other_port(3);
  REQUIRE_FALSE(sirius::planner::install_deferral(
    scan, other_port, make_defer_pair(schema, {1, 2}, schema, {1, 2}, origins)));
  REQUIRE(other_port.port_directive().empty());
  wide_scan fresh(3);
  REQUIRE_FALSE(sirius::planner::install_deferral(
    fresh, other_port, make_defer_pair(schema, {2, 1}, schema, {2, 1}, origins)));  // unordered
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
