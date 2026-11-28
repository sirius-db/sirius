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
 * Unit tests for `planner/dynamic_filter/build_filter_evidence.hpp`. The parity suite compares
 * the DuckDB-mirroring rules against DuckDB's own implementation on optimized plans.
 */

#include "planner/dynamic_filter/build_filter_evidence.hpp"
#include "sirius_config.hpp"

#include <catch.hpp>
#include <duckdb/common/enum_util.hpp>
#include <duckdb/function/aggregate/distributive_functions.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_any_join.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_cteref.hpp>
#include <duckdb/planner/operator/logical_distinct.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_limit.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_set_operation.hpp>
#include <duckdb/planner/operator/logical_top_n.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <utility>

namespace {

using sirius::planner::build_relation_is_opaque;
using sirius::planner::build_subtree_has_filtering_join;
using sirius::planner::build_subtree_is_filtering;

// Minimal constructible scan; the evidence walk never invokes its table function.
duckdb::unique_ptr<duckdb::LogicalGet> make_get()
{
  return duckdb::make_uniq<duckdb::LogicalGet>(
    duckdb::TableIndex(/*table_index=*/0),
    duckdb::BoundTableFunction(duckdb::TableFunction()),
    /*bind_data=*/nullptr,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    duckdb::vector<duckdb::Identifier>{"a"});
}

duckdb::unique_ptr<duckdb::LogicalGet> make_filtered_get()
{
  auto get = make_get();
  get->table_filters.PushFilter(
    duckdb::ProjectionIndex{0},
    sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                  duckdb::Value::INTEGER(5)));
  return get;
}

duckdb::unique_ptr<duckdb::LogicalProjection> make_projection_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(
    duckdb::TableIndex(/*table_index=*/7),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
  projection->expressions.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0));
  projection->children.push_back(std::move(child));
  return projection;
}

duckdb::unique_ptr<duckdb::LogicalCTERef> make_cte_ref()
{
  return duckdb::make_uniq<duckdb::LogicalCTERef>(
    duckdb::TableIndex(/*table_index=*/3),
    duckdb::TableIndex(/*cte_index=*/0),
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    duckdb::vector<duckdb::Identifier>{"a"});
}

duckdb::unique_ptr<duckdb::LogicalAggregate> make_aggregate_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto aggregate = duckdb::make_uniq<duckdb::LogicalAggregate>(
    duckdb::TableIndex(/*group_index=*/4),
    duckdb::TableIndex(/*aggregate_index=*/5),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
  aggregate->children.push_back(std::move(child));
  return aggregate;
}

duckdb::unique_ptr<duckdb::LogicalAggregate> make_dedup_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto aggregate = make_aggregate_over(std::move(child));
  aggregate->groups.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0));
  return aggregate;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_join_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> left,
  duckdb::unique_ptr<duckdb::LogicalOperator> right)
{
  auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  join->children.push_back(std::move(left));
  join->children.push_back(std::move(right));
  return join;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_typed_join(duckdb::JoinType join_type)
{
  auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  join->children.push_back(make_get());
  join->children.push_back(make_get());
  return join;
}

duckdb::unique_ptr<duckdb::LogicalLimit> make_limit_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto limit = duckdb::make_uniq<duckdb::LogicalLimit>(duckdb::BoundLimitNode::ConstantValue(10),
                                                       duckdb::BoundLimitNode());
  limit->children.push_back(std::move(child));
  return limit;
}

duckdb::unique_ptr<duckdb::LogicalOrder> make_order_over(
  duckdb::unique_ptr<duckdb::LogicalOperator> child)
{
  auto order = duckdb::make_uniq<duckdb::LogicalOrder>(duckdb::vector<duckdb::BoundOrderByNode>{});
  order->children.push_back(std::move(child));
  return order;
}

duckdb::unique_ptr<duckdb::LogicalSetOperation> make_set_operation(duckdb::LogicalOperatorType type)
{
  return duckdb::make_uniq<duckdb::LogicalSetOperation>(duckdb::TableIndex(/*table_index=*/8),
                                                        /*column_count=*/1,
                                                        make_get(),
                                                        make_get(),
                                                        type,
                                                        /*setop_all=*/false);
}

}  // namespace

TEST_CASE("a GET without table filters carries no evidence", "[dynamic_filter][evidence]")
{
  auto const get = make_get();
  REQUIRE_FALSE(build_subtree_is_filtering(*get));
}

TEST_CASE("a GET with a table filter carries evidence", "[dynamic_filter][evidence]")
{
  auto const get = make_filtered_get();
  REQUIRE(build_subtree_is_filtering(*get));
}

TEST_CASE("a bare FILTER carries evidence", "[dynamic_filter][evidence]")
{
  duckdb::LogicalFilter filter;
  REQUIRE(build_subtree_is_filtering(filter));
}

TEST_CASE("a TOP_N carries evidence", "[dynamic_filter][evidence]")
{
  duckdb::LogicalTopN top_n({}, /*limit=*/10, /*offset=*/0);
  REQUIRE(build_subtree_is_filtering(top_n));
}

TEST_CASE("evidence propagates up through non-filtering operators", "[dynamic_filter][evidence]")
{
  SECTION("a projection over a filtered GET fires")
  {
    auto const projection = make_projection_over(make_filtered_get());
    REQUIRE(build_subtree_is_filtering(*projection));
  }

  SECTION("a projection over an unfiltered GET stays quiet")
  {
    auto const projection = make_projection_over(make_get());
    REQUIRE_FALSE(build_subtree_is_filtering(*projection));
  }

  SECTION("a join fires when ANY child subtree fires")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
    join->children.push_back(make_get());
    join->children.push_back(make_projection_over(make_filtered_get()));
    REQUIRE(build_subtree_is_filtering(*join));
  }

  SECTION("a join over only unfiltered children stays quiet")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
    join->children.push_back(make_get());
    join->children.push_back(make_get());
    REQUIRE_FALSE(build_subtree_is_filtering(*join));
  }
}

TEST_CASE("a childless non-filtering operator carries no evidence", "[dynamic_filter][evidence]")
{
  auto const projection = duckdb::make_uniq<duckdb::LogicalProjection>(
    duckdb::TableIndex(/*table_index=*/1),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
  REQUIRE_FALSE(build_subtree_is_filtering(*projection));
}

TEST_CASE("a semi or anti join carries join evidence", "[dynamic_filter][evidence]")
{
  auto const join_type = GENERATE(duckdb::JoinType::SEMI,
                                  duckdb::JoinType::ANTI,
                                  duckdb::JoinType::RIGHT_SEMI,
                                  duckdb::JoinType::RIGHT_ANTI);
  CAPTURE(duckdb::EnumUtil::ToString(join_type));

  SECTION("as the build itself")
  {
    auto const join = make_typed_join(join_type);
    REQUIRE(build_subtree_has_filtering_join(*join));
    // DuckDB's mirrored test still sees no filter in the same subtree.
    REQUIRE_FALSE(build_subtree_is_filtering(*join));
  }

  SECTION("below a projection and an inner join")
  {
    auto const outer = make_join_over(make_get(), make_projection_over(make_typed_join(join_type)));
    REQUIRE(build_subtree_has_filtering_join(*outer));
  }
}

TEST_CASE("a right-family join publishes only with filters, accumulation and a Bloom budget",
          "[dynamic_filter][evidence]")
{
  sirius::operator_params params;
  params.enable_dynamic_filter                 = true;
  params.enable_dynamic_filter_multi_partition = true;
  REQUIRE(params.max_dynamic_filter_bloom_bytes_per_gpu() > 0);
  CHECK(sirius::planner::right_family_publication_enabled(params));

  SECTION("dynamic filters off") { params.enable_dynamic_filter = false; }
  SECTION("Bloom accumulation off") { params.enable_dynamic_filter_multi_partition = false; }
  SECTION("no Bloom budget") { params.scan_task_batch_size = 0; }
  CHECK_FALSE(sirius::planner::right_family_publication_enabled(params));
}

namespace {

// A scan with one projected column, so a join key can be traced to it.
duckdb::unique_ptr<duckdb::LogicalGet> make_scannable_get()
{
  auto get = make_get();
  get->AddColumnId(0);
  return get;
}

void add_key(duckdb::LogicalComparisonJoin& join,
             duckdb::idx_t probe_ordinal,
             duckdb::ExpressionType comparison   = duckdb::ExpressionType::COMPARE_EQUAL,
             duckdb::LogicalType const& key_type = duckdb::LogicalType::INTEGER)
{
  join.conditions.emplace_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(key_type, probe_ordinal),
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(key_type, 0),
    comparison);
}

sirius::operator_params filters_on()
{
  sirius::operator_params params;
  params.enable_dynamic_filter                 = true;
  params.enable_dynamic_filter_multi_partition = true;
  return params;
}

}  // namespace

TEST_CASE("a join expects to filter the scan its probe key comes from",
          "[dynamic_filter][evidence]")
{
  using sirius::planner::expected_probe_targets;
  auto const params = filters_on();

  SECTION("an equality on a scan column names that scan")
  {
    auto join         = make_join_over(make_scannable_get(), make_scannable_get());
    auto const* probe = join->children[0].get();
    add_key(*join, 0);
    add_key(*join, 0);  // a second key on the same scan adds nothing
    join->ResolveOperatorTypes();
    auto const targets = expected_probe_targets(*join, params);
    REQUIRE(targets.size() == 1);
    CHECK(targets[0] == probe);
  }
  SECTION("the key may come from the build side of a join below")
  {
    auto inner         = make_join_over(make_scannable_get(), make_scannable_get());
    auto const* target = inner->children[1].get();
    auto join          = make_join_over(std::move(inner), make_scannable_get());
    add_key(*join, 1);
    join->ResolveOperatorTypes();
    auto const targets = expected_probe_targets(*join, params);
    REQUIRE(targets.size() == 1);
    CHECK(targets[0] == target);
  }
  SECTION("a join type that binds no scan names none")
  {
    auto const join_type =
      GENERATE(duckdb::JoinType::LEFT, duckdb::JoinType::OUTER, duckdb::JoinType::ANTI);
    auto join       = make_join_over(make_scannable_get(), make_scannable_get());
    join->join_type = join_type;
    add_key(*join, 0);
    join->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*join, params).empty());
  }
  SECTION("a null-safe equality names none")
  {
    auto join = make_join_over(make_scannable_get(), make_scannable_get());
    add_key(*join, 0, duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM);
    join->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*join, params).empty());
  }
  SECTION("a key type no membership filter supports names none")
  {
    auto join = make_join_over(make_scannable_get(), make_scannable_get());
    add_key(*join, 0, duckdb::ExpressionType::COMPARE_EQUAL, duckdb::LogicalType::DOUBLE);
    join->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*join, params).empty());
  }
  SECTION("a cast on the probe key names none; a computed build key is fine")
  {
    auto join = make_join_over(make_scannable_get(), make_scannable_get());
    join->conditions.emplace_back(
      duckdb::BoundCastExpression::AddDefaultCastToType(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
        duckdb::LogicalType::BIGINT),
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0),
      duckdb::ExpressionType::COMPARE_EQUAL);
    join->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*join, params).empty());

    auto computed = make_join_over(make_scannable_get(), make_scannable_get());
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> operands;
    operands.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0));
    auto build_key = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
      duckdb::ExpressionType::OPERATOR_COALESCE, duckdb::LogicalType::INTEGER);
    build_key->GetChildrenMutable() = std::move(operands);
    computed->conditions.emplace_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
      std::move(build_key),
      duckdb::ExpressionType::COMPARE_EQUAL);
    computed->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*computed, params).size() == 1);
  }
  SECTION("a right semi join names none for a key its accumulator cannot build")
  {
    auto join       = make_join_over(make_scannable_get(), make_scannable_get());
    join->join_type = duckdb::JoinType::RIGHT_SEMI;
    add_key(*join, 0, duckdb::ExpressionType::COMPARE_EQUAL, duckdb::LogicalType::VARCHAR);
    join->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*join, params).empty());
    // The same key on an inner join publishes from a whole build.
    join->join_type = duckdb::JoinType::INNER;
    CHECK(expected_probe_targets(*join, params).size() == 1);
  }
  SECTION("a distributed plan names none")
  {
    auto join = make_join_over(make_scannable_get(), make_scannable_get());
    add_key(*join, 0);
    join->ResolveOperatorTypes();
    auto distributed           = params;
    distributed.exchange_peers = {"rank0", "rank1"};
    REQUIRE(distributed.distributed());
    CHECK(expected_probe_targets(*join, distributed).empty());
  }
  SECTION("a key behind a sort names none")
  {
    auto order = make_order_over(make_scannable_get());
    auto join  = make_join_over(std::move(order), make_scannable_get());
    add_key(*join, 0);
    join->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*join, params).empty());
  }
  SECTION("a right semi join names its scan only when it can publish early")
  {
    auto join       = make_join_over(make_scannable_get(), make_scannable_get());
    join->join_type = duckdb::JoinType::RIGHT_SEMI;
    add_key(*join, 0);
    join->ResolveOperatorTypes();
    CHECK(expected_probe_targets(*join, params).size() == 1);
    auto without_accumulation                                  = params;
    without_accumulation.enable_dynamic_filter_multi_partition = false;
    CHECK(expected_probe_targets(*join, without_accumulation).empty());
  }
}

TEST_CASE("an expected filter target counts when the build side is that scan",
          "[dynamic_filter][evidence]")
{
  using sirius::planner::expected_target_behind_build;
  auto scan          = make_get();
  auto const* target = scan.get();
  auto const other   = make_get();
  std::vector<duckdb::LogicalGet const*> const expected{target};
  std::vector<duckdb::LogicalGet const*> const unrelated{other.get()};

  SECTION("the scan itself")
  {
    CHECK(expected_target_behind_build(*scan, expected) == target);
    CHECK(expected_target_behind_build(*scan, {}) == nullptr);
    CHECK(expected_target_behind_build(*scan, unrelated) == nullptr);
  }
  SECTION("behind a projection and a filter")
  {
    auto filter = duckdb::make_uniq<duckdb::LogicalFilter>();
    filter->children.push_back(make_projection_over(std::move(scan)));
    CHECK(expected_target_behind_build(*filter, expected) == target);
  }
  SECTION("not behind an operator that makes the build's size say nothing")
  {
    // After most of these the build can be smaller than the scan, or larger, for a reason other
    // than the expected filter, so comparing the two sizes would prove nothing. A sort keeps the
    // size, but no filter is ever predicted for a scan behind one.
    duckdb::unique_ptr<duckdb::LogicalOperator> build;
    SECTION("a sort") { build = make_order_over(std::move(scan)); }
    SECTION("a deduplication") { build = make_dedup_over(std::move(scan)); }
    SECTION("a limit") { build = make_limit_over(std::move(scan)); }
    SECTION("an inner join") { build = make_join_over(std::move(scan), make_get()); }
    SECTION("a semi join")
    {
      auto semi       = make_join_over(std::move(scan), make_get());
      semi->join_type = duckdb::JoinType::SEMI;
      build           = std::move(semi);
    }
    CHECK(expected_target_behind_build(*build, expected) == nullptr);
  }
}

TEST_CASE("joins that can keep every row carry no join evidence", "[dynamic_filter][evidence]")
{
  SECTION("a bare GET") { REQUIRE_FALSE(build_subtree_has_filtering_join(*make_get())); }

  SECTION("a filtered GET is filter evidence, not join evidence")
  {
    REQUIRE_FALSE(build_subtree_has_filtering_join(*make_filtered_get()));
  }

  SECTION("other join types")
  {
    auto const join_type = GENERATE(duckdb::JoinType::INNER,
                                    duckdb::JoinType::LEFT,
                                    duckdb::JoinType::RIGHT,
                                    duckdb::JoinType::OUTER,
                                    duckdb::JoinType::MARK,
                                    duckdb::JoinType::SINGLE);
    CAPTURE(duckdb::EnumUtil::ToString(join_type));
    REQUIRE_FALSE(build_subtree_has_filtering_join(*make_typed_join(join_type)));
  }

  SECTION("a semi-typed ANY_JOIN")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalAnyJoin>(duckdb::JoinType::SEMI);
    join->children.push_back(make_get());
    join->children.push_back(make_get());
    REQUIRE_FALSE(build_subtree_has_filtering_join(*join));
  }
}

TEST_CASE("opaque build leaves carry fallback evidence", "[dynamic_filter][evidence]")
{
  SECTION("a bare CTE_REF") { REQUIRE(build_relation_is_opaque(*make_cte_ref())); }

  SECTION("a projected CTE_REF")
  {
    REQUIRE(build_relation_is_opaque(*make_projection_over(make_cte_ref())));
  }

  SECTION("a CTE_REF below stacked projections")
  {
    REQUIRE(build_relation_is_opaque(*make_projection_over(make_projection_over(make_cte_ref()))));
  }
}

TEST_CASE("base-table builds carry no opaque fallback evidence", "[dynamic_filter][evidence]")
{
  REQUIRE_FALSE(build_relation_is_opaque(*make_get()));
  REQUIRE_FALSE(build_relation_is_opaque(*make_filtered_get()));
  REQUIRE_FALSE(build_relation_is_opaque(*make_projection_over(make_get())));
  REQUIRE_FALSE(build_relation_is_opaque(*make_dedup_over(make_get())));
  REQUIRE_FALSE(build_relation_is_opaque(*make_dedup_over(make_filtered_get())));
}

TEST_CASE("ordinary deduplication preserves opaque build evidence", "[dynamic_filter][evidence]")
{
  auto aggregate = make_dedup_over(make_cte_ref());

  SECTION("implicit ordinary grouping") { REQUIRE(build_relation_is_opaque(*aggregate)); }

  SECTION("one complete grouping set")
  {
    aggregate->groups.push_back(aggregate->groups.front()->Copy());
    aggregate->grouping_sets.push_back(
      duckdb::GroupingSet{duckdb::ProjectionIndex(0), duckdb::ProjectionIndex(1)});
    REQUIRE(build_relation_is_opaque(*aggregate));
  }

  SECTION("projections above and below deduplication")
  {
    aggregate->children.front() = make_projection_over(std::move(aggregate->children.front()));
    REQUIRE(build_relation_is_opaque(*make_projection_over(std::move(aggregate))));
  }

  SECTION("nested deduplication")
  {
    REQUIRE(build_relation_is_opaque(*make_dedup_over(std::move(aggregate))));
  }

  SECTION("computed grouping keys")
  {
    auto expression = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
      duckdb::ExpressionType::OPERATOR_IS_NULL, duckdb::LogicalType::BOOLEAN);
    expression->GetChildrenMutable().push_back(std::move(aggregate->groups.front()));
    aggregate->groups.front() = std::move(expression);
    REQUIRE(build_relation_is_opaque(*aggregate));
  }
}

TEST_CASE("aggregate outputs and grouping sets hide opaque build evidence",
          "[dynamic_filter][evidence]")
{
  auto aggregate = make_dedup_over(make_cte_ref());

  SECTION("global aggregation") { aggregate->groups.clear(); }

  SECTION("aggregate output")
  {
    aggregate->expressions.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
      duckdb::BoundAggregateFunction(duckdb::CountStarFun::GetFunction()),
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
      nullptr,
      nullptr,
      duckdb::AggregateType::NON_DISTINCT));
  }

  SECTION("grouping function")
  {
    aggregate->grouping_functions.push_back({duckdb::ProjectionIndex(0)});
  }

  SECTION("empty grouping set") { aggregate->grouping_sets.push_back(duckdb::GroupingSet{}); }

  SECTION("partial grouping set")
  {
    aggregate->groups.push_back(aggregate->groups.front()->Copy());
    aggregate->grouping_sets.push_back(duckdb::GroupingSet{duckdb::ProjectionIndex(0)});
  }

  SECTION("out-of-range grouping index")
  {
    aggregate->grouping_sets.push_back(duckdb::GroupingSet{duckdb::ProjectionIndex(1)});
  }

  SECTION("multiple complete grouping sets")
  {
    aggregate->grouping_sets.push_back(duckdb::GroupingSet{duckdb::ProjectionIndex(0)});
    aggregate->grouping_sets.push_back(duckdb::GroupingSet{duckdb::ProjectionIndex(0)});
  }

  REQUIRE_FALSE(build_relation_is_opaque(*aggregate));
}

TEST_CASE("malformed deduplication carries no opaque build evidence", "[dynamic_filter][evidence]")
{
  auto aggregate = make_dedup_over(make_cte_ref());

  SECTION("no child") { aggregate->children.clear(); }
  SECTION("null child") { aggregate->children.front().reset(); }
  SECTION("multiple children") { aggregate->children.push_back(make_get()); }
  SECTION("null grouping expression") { aggregate->groups.front().reset(); }

  REQUIRE_FALSE(build_relation_is_opaque(*aggregate));
}

TEST_CASE("malformed projections carry no opaque fallback evidence", "[dynamic_filter][evidence]")
{
  SECTION("a childless projection")
  {
    auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(
      duckdb::TableIndex(/*table_index=*/6),
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
    REQUIRE_FALSE(build_relation_is_opaque(*projection));
  }

  SECTION("a projection with a null child")
  {
    auto projection = duckdb::make_uniq<duckdb::LogicalProjection>(
      duckdb::TableIndex(/*table_index=*/6),
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{});
    projection->children.push_back(nullptr);
    REQUIRE_FALSE(build_relation_is_opaque(*projection));
  }

  SECTION("a projection with multiple children")
  {
    auto projection = make_projection_over(make_cte_ref());
    projection->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_opaque(*projection));
  }
}

TEST_CASE("reducing operators are not opaque build roots", "[dynamic_filter][evidence]")
{
  SECTION("a grouped aggregate")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_aggregate_over(make_get())));
  }

  SECTION("a DISTINCT")
  {
    auto distinct = duckdb::make_uniq<duckdb::LogicalDistinct>(
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{}, duckdb::DistinctType::DISTINCT);
    distinct->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_opaque(*distinct));
  }

  SECTION("a comparison join")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_join_over(make_get(), make_get())));
  }

  SECTION("an ANY_JOIN")
  {
    auto join = duckdb::make_uniq<duckdb::LogicalAnyJoin>(duckdb::JoinType::INNER);
    join->children.push_back(make_get());
    join->children.push_back(make_get());
    REQUIRE_FALSE(build_relation_is_opaque(*join));
  }

  SECTION("an INTERSECT")
  {
    REQUIRE_FALSE(build_relation_is_opaque(
      *make_set_operation(duckdb::LogicalOperatorType::LOGICAL_INTERSECT)));
  }

  SECTION("an EXCEPT")
  {
    REQUIRE_FALSE(
      build_relation_is_opaque(*make_set_operation(duckdb::LogicalOperatorType::LOGICAL_EXCEPT)));
  }
}

TEST_CASE("other wrappers hide opaque leaves from the fallback classifier",
          "[dynamic_filter][evidence]")
{
  SECTION("a global aggregate over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_aggregate_over(make_cte_ref())));
  }

  SECTION("a join over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_join_over(make_cte_ref(), make_get())));
  }

  SECTION("a LIMIT over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_limit_over(make_cte_ref())));
  }

  SECTION("an ORDER_BY over a CTE_REF")
  {
    REQUIRE_FALSE(build_relation_is_opaque(*make_order_over(make_cte_ref())));
  }
}
