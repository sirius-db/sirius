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
 * @file test_fold_limits.cpp
 * @brief The INV-FOLD arithmetic (`op/fold_limits.hpp`) and the rule that says which side of a
 *        join folds (`sirius_physical_hash_join::join_folds_side`), including the drift pin that
 *        ties `sirius_physical_concat`'s actual `_concat_all` back to that rule.
 *
 * All GPU-free: these are the pure predicates the partition-sizing floor and the CONCAT fold
 * guard both build on, so they are pinned directly rather than through an end-to-end plan.
 */

#include "helper/type_conversions.hpp"

#include <catch.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <data/data_batch_utils.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/fold_limits.hpp>
#include <op/sirius_physical_concat.hpp>
#include <op/sirius_physical_hash_join.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::op::check_fold_row_limit;
using sirius::op::fold_partition_count;
using sirius::op::fold_partition_floor;
using sirius::op::fold_row_target;
using sirius::op::k_fold_row_limit;
using sirius::op::sirius_physical_concat;
using sirius::op::sirius_physical_hash_join;

namespace {

/// Every join type the hash join implements, i.e. every type `join_folds_side` answers for.
constexpr duckdb::JoinType kImplementedJoinTypes[] = {duckdb::JoinType::INNER,
                                                      duckdb::JoinType::LEFT,
                                                      duckdb::JoinType::RIGHT,
                                                      duckdb::JoinType::OUTER,
                                                      duckdb::JoinType::SEMI,
                                                      duckdb::JoinType::ANTI,
                                                      duckdb::JoinType::MARK,
                                                      duckdb::JoinType::RIGHT_SEMI,
                                                      duckdb::JoinType::RIGHT_ANTI};

/// Owns the LogicalComparisonJoin alongside the physical join: the join stores `op.types` by
/// reference, so the logical operator has to outlive it.
struct join_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;
};

/// A minimal single-condition hash join of @p join_type, enough to construct a CONCAT against.
join_fixture make_hash_join(duckdb::JoinType join_type)
{
  join_fixture fixture;
  fixture.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  fixture.logical_join->types = {duckdb::LogicalType::INTEGER};

  auto make_child = [] {
    return duckdb::make_uniq<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::PROJECTION,
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      0);
  };

  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition condition;
  condition.left =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  condition.right =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(condition));

  fixture.hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
    *fixture.logical_join,
    make_child(),
    make_child(),
    sirius::wrap_join_conditions(std::move(conditions)),
    join_type,
    duckdb::vector<duckdb::idx_t>{},
    duckdb::vector<duckdb::idx_t>{},
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
    /*estimated_cardinality=*/1000);
  return fixture;
}

}  // namespace

TEST_CASE("fold_row_target halves the addressable row limit", "[fold_limit][unit]")
{
  STATIC_REQUIRE(fold_row_target(k_fold_row_limit) == k_fold_row_limit / 2);
  STATIC_REQUIRE(fold_row_target(k_fold_row_limit) == 1073741823ULL);
  // The limit is what cudf::concatenate accepts for a table containing a STRING or LIST column,
  // which needs one offset more than it has rows -- one row below size_type's maximum.
  STATIC_REQUIRE(k_fold_row_limit == 2147483646ULL);
  // A limit too small to halve still leaves a usable target rather than zero.
  STATIC_REQUIRE(fold_row_target(1) == 1);
  STATIC_REQUIRE(fold_row_target(0) == 1);
}

TEST_CASE("fold_partition_floor is the smallest count keeping a fold under the target",
          "[fold_limit][unit]")
{
  constexpr uint64_t target = fold_row_target(k_fold_row_limit);

  STATIC_REQUIRE(fold_partition_floor(0, target) == 1);
  STATIC_REQUIRE(fold_partition_floor(target - 1, target) == 1);
  STATIC_REQUIRE(fold_partition_floor(target, target) == 1);
  STATIC_REQUIRE(fold_partition_floor(target + 1, target) == 2);
  // Exact multiples must not round up: 2 * target splits into exactly 2 folds, not 3.
  STATIC_REQUIRE(fold_partition_floor(2 * target, target) == 2);
  STATIC_REQUIRE(fold_partition_floor(2 * target + 1, target) == 3);

  // The saturation path: no overflow, no wraparound, and never a negative count.
  STATIC_REQUIRE(fold_partition_floor(std::numeric_limits<uint64_t>::max(), 1) ==
                 std::numeric_limits<int>::max());
  STATIC_REQUIRE(fold_partition_floor(std::numeric_limits<uint64_t>::max(), target) > 0);

  // A degenerate target divides by one instead of dividing by zero.
  STATIC_REQUIRE(fold_partition_floor(5, 0) == 5);
}

TEST_CASE("fold_partition_count charges the skew margin only to a genuine split",
          "[fold_limit][unit]")
{
  constexpr uint64_t limit  = k_fold_row_limit;
  constexpr uint64_t target = fold_row_target(limit);

  // The whole band between the target and the hard limit fits ONE cuDF table, so it must not be
  // split: the fold is the measurement there, and there is no distribution to be skewed.
  STATIC_REQUIRE(fold_partition_count(0, limit) == 1);
  STATIC_REQUIRE(fold_partition_count(target, limit) == 1);
  STATIC_REQUIRE(fold_partition_count(target + 1, limit) == 1);
  STATIC_REQUIRE(fold_partition_count(limit, limit) == 1);

  // Past the limit a split is unavoidable, and only then is it costed against the halved target so
  // key skew has headroom. One row past the limit needs three folds rather than two: two folds at
  // the target hold exactly `limit` rows between them, so the extra row opens a third.
  STATIC_REQUIRE(fold_partition_count(limit + 1, limit) == 3);
  STATIC_REQUIRE(fold_partition_count(4 * target, limit) == 4);

  // The same shape at a lowered limit, which is what the max_concat_fold_rows setting drives.
  STATIC_REQUIRE(fold_partition_count(200, 200) == 1);
  STATIC_REQUIRE(fold_partition_count(201, 200) == 3);  // ceil(201 / 100)
}

TEST_CASE("check_fold_row_limit admits exactly the limit and names the overflow",
          "[fold_limit][unit]")
{
  REQUIRE_NOTHROW(check_fold_row_limit(k_fold_row_limit, 4, k_fold_row_limit));
  REQUIRE_NOTHROW(check_fold_row_limit(0, 0, k_fold_row_limit));
  REQUIRE_THROWS_AS(check_fold_row_limit(k_fold_row_limit + 1, 4, k_fold_row_limit),
                    sirius::op::fold_row_limit_exceeded);
  // A distinct type, so a handler can tell an INV-FOLD violation from a device OOM or a
  // non-resident batch; still a runtime_error, so existing handlers keep working.
  REQUIRE_THROWS_AS(check_fold_row_limit(k_fold_row_limit + 1, 4, k_fold_row_limit),
                    std::runtime_error);

  // The marker is what log analysis greps for; the counts are what makes the entry actionable.
  REQUIRE_THROWS_WITH(check_fold_row_limit(2500000000ULL, 7, k_fold_row_limit),
                      Catch::Contains("[fold_limit]") && Catch::Contains("2500000000") &&
                        Catch::Contains(std::to_string(k_fold_row_limit)) && Catch::Contains("7"));
}

TEST_CASE("join_folds_side names the side each join type must see whole", "[fold_limit][unit]")
{
  // LEFT / SEMI / ANTI fold the build; RIGHT-family folds the probe; OUTER folds both;
  // INNER / MARK fold neither.
  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::LEFT, true));
  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::LEFT, false));
  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::SEMI, true));
  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::SEMI, false));
  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::ANTI, true));
  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::ANTI, false));

  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::RIGHT, true));
  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::RIGHT, false));
  STATIC_REQUIRE_FALSE(
    sirius_physical_hash_join::join_folds_side(duckdb::JoinType::RIGHT_SEMI, true));
  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::RIGHT_SEMI, false));
  STATIC_REQUIRE_FALSE(
    sirius_physical_hash_join::join_folds_side(duckdb::JoinType::RIGHT_ANTI, true));
  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::RIGHT_ANTI, false));

  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::OUTER, true));
  STATIC_REQUIRE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::OUTER, false));

  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::INNER, true));
  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::INNER, false));
  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::MARK, true));
  STATIC_REQUIRE_FALSE(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::MARK, false));

  // Types the hash join does not implement have no fold answer at all.
  REQUIRE_THROWS_AS(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::SINGLE, true),
                    std::runtime_error);
  REQUIRE_THROWS_AS(sirius_physical_hash_join::join_folds_side(duckdb::JoinType::INVALID, false),
                    std::runtime_error);
}

TEST_CASE("rows_from_column_metadata reads the row count off the first column",
          "[fold_limit][unit]")
{
  // The spilled-batch half of the row measurement. Every column of one table shares a row count,
  // so the first answers for the table; a zero-column table has no column for cuDF to index and so
  // cannot violate INV-FOLD at any row count.
  using cucascade::memory::column_metadata;

  CHECK(sirius::rows_from_column_metadata({}) == 0);

  std::vector<column_metadata> columns(3);
  for (auto& column : columns) {
    column.num_rows = 4096;
  }
  CHECK(sirius::rows_from_column_metadata(columns) == 4096);

  // A count at the addressable limit is returned as-is; deciding what to do with it belongs to
  // check_fold_row_limit, not here.
  columns[0].num_rows = std::numeric_limits<cudf::size_type>::max();
  CHECK(sirius::rows_from_column_metadata(columns) == 2147483647);
}

TEST_CASE("build_folds_for_filter_publication requires every one of its five obligations",
          "[fold_limit][unit]")
{
  // The gate that decides whether a build is folded whole so a single-batch dynamic filter can
  // publish. Its failure mode is a lost filter -- the q4 mechanism -- so every term is pinned.
  constexpr uint64_t max_build_bytes = 1024;
  constexpr uint64_t max_fold_rows   = 100;

  auto gate = [](sirius::op::partition_sizing_input const& in, int num_partitions, bool publishes) {
    return sirius_physical_hash_join::build_folds_for_filter_publication(
      in, num_partitions, publishes, max_build_bytes, max_fold_rows);
  };
  sirius::op::partition_sizing_input const eligible{.total_bytes      = 512,
                                                    .total_rows       = 50,
                                                    .total_rows_exact = true,
                                                    .is_build_side    = true,
                                                    .build_foldable   = true};
  CHECK(gate(eligible, 1, true));

  // The measurement must come from the build: a right-family join sizes from the probe, where one
  // partition says nothing about the build's size.
  auto probe_sized          = eligible;
  probe_sized.is_build_side = false;
  CHECK_FALSE(gate(probe_sized, 1, true));

  // More than one partition leaves no single batch to publish from, and a join that publishes
  // nothing has nothing to fold for.
  CHECK_FALSE(gate(eligible, 2, true));
  CHECK_FALSE(gate(eligible, 1, false));

  // A build refused BUILD_PROBE for exceeding the hash-table budget must not be folded whole for a
  // filter either.
  auto too_many_bytes        = eligible;
  too_many_bytes.total_bytes = max_build_bytes;
  CHECK_FALSE(gate(too_many_bytes, 1, true));

  // INV-FOLD: an exactly-measured build that cannot become one cuDF table gives up the filter.
  auto too_many_rows       = eligible;
  too_many_rows.total_rows = max_fold_rows + 1;
  CHECK_FALSE(gate(too_many_rows, 1, true));

  // ... but only when the measurement is exact. A pessimistic bound must not surrender a
  // publication for a fold that was never too big.
  auto inexact             = too_many_rows;
  inexact.total_rows_exact = false;
  CHECK(gate(inexact, 1, true));
}

TEST_CASE("a CONCAT's concat_all matches join_folds_side for every join type", "[fold_limit][unit]")
{
  // The drift pin behind the partition-sizing floor: the floor is derived from
  // join_folds_side, and it is only sound because the CONCAT actually folds that side.
  for (auto const join_type : kImplementedJoinTypes) {
    auto fixture = make_hash_join(join_type);
    for (bool const is_build : {true, false}) {
      INFO("join type " << duckdb::JoinTypeToString(join_type)
                        << (is_build ? " build side" : " probe side"));
      sirius_physical_concat concat(
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        /*estimated_cardinality=*/1000,
        fixture.hash_join.get(),
        is_build);
      CHECK(concat.folds_all() == sirius_physical_hash_join::join_folds_side(join_type, is_build));
    }
  }
}
