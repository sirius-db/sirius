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

/**
 * @file test_physical_single_join.cpp
 * @brief SINGLE join on sirius_physical_hash_join: a LEFT join that keeps at most one build row
 * per probe row. Calling execute() with a probe and a build batch runs the STANDARD arm, which is
 * a production path for SINGLE (the build is concat-folded per partition, so the per-task dedupe
 * is complete).
 *
 * Duplicate build keys can only be exercised here: DuckDB wraps a scalar subquery in a first()
 * aggregate, so a SINGLE join planned from SQL always sees unique build keys.
 */

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_hash_join.hpp>

#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

/// The logical_join must outlive the hash_join because hash_join stores op.types by reference.
struct single_join_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;
};

void number_operator_tree(sirius::op::sirius_physical_operator& op, size_t& next_id)
{
  op.operator_id = next_id++;
  for (auto& child : op.children) {
    if (child) { number_operator_tree(*child, next_id); }
  }
}

/// Output: left key, left payload, right key, right payload (empty projection maps keep every
/// column of both sides).
single_join_fixture create_single_join(
  duckdb::ExpressionType comparison = duckdb::ExpressionType::COMPARE_EQUAL)
{
  single_join_fixture f;

  f.logical_join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::SINGLE);
  f.logical_join->types = {duckdb::LogicalType::INTEGER,
                           duckdb::LogicalType::INTEGER,
                           duckdb::LogicalType::INTEGER,
                           duckdb::LogicalType::INTEGER};

  auto two_ints = sirius::from_duckdb_vec(
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER});
  auto left_child =
    duckdb::make_uniq<sirius_physical_operator>(SiriusPhysicalOperatorType::PROJECTION, two_ints, 0);
  auto right_child =
    duckdb::make_uniq<sirius_physical_operator>(SiriusPhysicalOperatorType::PROJECTION, two_ints, 0);

  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.emplace_back(
    duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    comparison);

  f.hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
    *f.logical_join,
    std::move(left_child),
    std::move(right_child),
    sirius::wrap_join_conditions(std::move(conditions)),
    duckdb::JoinType::SINGLE,
    duckdb::vector<duckdb::idx_t>{},
    duckdb::vector<duckdb::idx_t>{},
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
    1000);

  size_t next_id = 0;
  number_operator_tree(*f.hash_join, next_id);
  return f;
}

memory_space* get_shared_mem_space()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

std::shared_ptr<cucascade::data_batch> make_key_payload_batch(memory_space& space,
                                                              const std::vector<int32_t>& keys,
                                                              const std::vector<int32_t>& payload)
{
  return make_two_column_batch<int32_t, int32_t>(
    space, keys, payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);
}

std::shared_ptr<cucascade::data_batch> make_nullable_key_payload_batch(
  memory_space& space,
  const std::vector<int32_t>& keys,
  const std::vector<bool>& key_valid,
  const std::vector<int32_t>& payload)
{
  auto k = make_numeric_batch_with_nulls<int32_t>(space, keys, key_valid, cudf::type_id::INT32);
  auto p = make_numeric_batch<int32_t>(space, payload, cudf::type_id::INT32);
  return concatenate_batches_horizontal({k, p}, space);
}

/// One output row: left key, left payload, right payload (nullopt when NULL-padded).
using out_row = std::tuple<int32_t, int32_t, std::optional<int32_t>>;

/// The gather maps of a hash join carry no row order, so sort by left key before asserting.
std::vector<out_row> rows_sorted_by_left_key(const operator_data& outputs)
{
  auto const& batches = dynamic_cast<const pipelineable_operator_data&>(outputs).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto view = sirius::get_cudf_table_view(*batches[0]);
  REQUIRE(view.num_columns() == 4);

  auto lk  = copy_column_to_host<int32_t>(view.column(0));
  auto lp  = copy_column_to_host<int32_t>(view.column(1));
  auto rp  = copy_column_to_host<int32_t>(view.column(3));
  auto rpv = copy_validity_to_host(view.column(3));

  std::vector<out_row> rows;
  for (size_t i = 0; i < lk.size(); ++i) {
    rows.emplace_back(lk[i], lp[i], rpv[i] ? std::optional<int32_t>{rp[i]} : std::nullopt);
  }
  std::sort(rows.begin(), rows.end(), [](auto const& a, auto const& b) {
    return std::get<0>(a) < std::get<0>(b);
  });
  return rows;
}

std::unique_ptr<operator_data> run(single_join_fixture& f,
                                   std::shared_ptr<cucascade::data_batch> left,
                                   std::shared_ptr<cucascade::data_batch> right)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{std::move(left), std::move(right)};
  return f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());
}

}  // namespace

TEST_CASE("sirius_physical_hash_join single join - duplicate build keys yield one row per probe row",
          "[physical_single_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  auto left = make_key_payload_batch(*space, {10, 20, 30}, {1, 2, 3});
  // Key 20 matches three build rows; SINGLE keeps exactly one of them.
  auto right = make_key_payload_batch(*space, {20, 20, 20}, {100, 200, 300});

  auto f    = create_single_join();
  auto rows = rows_sorted_by_left_key(*run(f, left, right));

  REQUIRE(rows.size() == 3);
  CHECK(rows[0] == out_row{10, 1, std::nullopt});
  CHECK(std::get<0>(rows[1]) == 20);
  CHECK(std::get<1>(rows[1]) == 2);
  REQUIRE(std::get<2>(rows[1]).has_value());
  auto const picked = *std::get<2>(rows[1]);
  CHECK((picked == 100 || picked == 200 || picked == 300));
  CHECK(rows[2] == out_row{30, 3, std::nullopt});
}

TEST_CASE("sirius_physical_hash_join single join - unmatched probe rows are NULL-padded",
          "[physical_single_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  auto left  = make_key_payload_batch(*space, {10, 20, 30}, {1, 2, 3});
  auto right = make_key_payload_batch(*space, {99, 20}, {9, 7});

  auto f    = create_single_join();
  auto rows = rows_sorted_by_left_key(*run(f, left, right));

  REQUIRE(rows == std::vector<out_row>{{10, 1, std::nullopt}, {20, 2, 7}, {30, 3, std::nullopt}});
}

TEST_CASE("sirius_physical_hash_join single join - empty build side NULL-pads every probe row",
          "[physical_single_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  auto left  = make_key_payload_batch(*space, {10, 20, 30}, {1, 2, 3});
  auto right = make_key_payload_batch(*space, {}, {});

  auto f    = create_single_join();
  auto rows = rows_sorted_by_left_key(*run(f, left, right));

  REQUIRE(rows == std::vector<out_row>{
                    {10, 1, std::nullopt}, {20, 2, std::nullopt}, {30, 3, std::nullopt}});
}

TEST_CASE("sirius_physical_hash_join single join - IS NOT DISTINCT FROM matches NULL keys",
          "[physical_single_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  // Left key 0 is NULL; the build has a NULL key and a duplicate 30.
  auto left  = make_nullable_key_payload_batch(*space, {0, 10, 30}, {false, true, true}, {1, 2, 3});
  auto right = make_nullable_key_payload_batch(
    *space, {0, 30, 30}, {false, true, true}, {7, 8, 9});

  auto f    = create_single_join(duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM);
  auto rows = rows_sorted_by_left_key(*run(f, left, right));

  REQUIRE(rows.size() == 3);
  // The NULL left key sorts first (its stored value is 0) and matches the NULL build key.
  CHECK(rows[0] == out_row{0, 1, 7});
  CHECK(rows[1] == out_row{10, 2, std::nullopt});
  CHECK(std::get<0>(rows[2]) == 30);
  REQUIRE(std::get<2>(rows[2]).has_value());
  auto const picked = *std::get<2>(rows[2]);
  CHECK((picked == 8 || picked == 9));
}

TEST_CASE("sirius_physical_hash_join single join - planner screens admit SINGLE on equality keys",
          "[physical_single_join]")
{
  REQUIRE(sirius_physical_hash_join::is_join_type_supported(duckdb::JoinType::SINGLE));

  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.emplace_back(
    duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
    duckdb::ExpressionType::COMPARE_EQUAL);
  auto wrapped = sirius::wrap_join_conditions(std::move(conditions));
  REQUIRE(sirius_physical_hash_join::are_conditions_supported(wrapped, duckdb::JoinType::SINGLE));
}
