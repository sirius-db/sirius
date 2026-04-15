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

#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_hash_join.hpp>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

//===----------------------------------------------------------------------===//
// Fixture helpers
//===----------------------------------------------------------------------===//

/**
 * @brief Holds the LogicalComparisonJoin and hash join needed for mark join tests.
 * The logical_join must outlive the hash_join because hash_join stores op.types by reference.
 */
struct mark_join_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;
};

/**
 * @brief Create a mark join operator with two INT32 key columns (left col[0] = right col[0]).
 * Left child has types {INTEGER, INTEGER} (key + payload), right child has {INTEGER} (key only).
 */
mark_join_fixture create_mark_join()
{
  mark_join_fixture f;

  f.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::MARK);
  f.logical_join->types = {
    duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::BOOLEAN};

  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER},
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    0);

  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition cond;
  cond.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(cond));

  f.hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
    *f.logical_join,
    std::move(left_child),
    std::move(right_child),
    std::move(conditions),
    duckdb::JoinType::MARK,
    duckdb::vector<duckdb::idx_t>{},        // left_projection_map (empty = all)
    duckdb::vector<duckdb::idx_t>{},        // right_projection_map (not used by MARK)
    duckdb::vector<duckdb::LogicalType>{},  // delim_types
    1000,
    nullptr);

  return f;
}

memory_space* get_shared_mem_space()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Mark join tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_hash_join mark join - partial match", "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {10, 20, 30, 40, 50};
  std::vector<int32_t> left_payload = {1, 2, 3, 4, 5};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Only {20, 40} exist on the right — rows 1 and 3 should be marked
  std::vector<int32_t> right_ids = {20, 40};
  auto right_batch = make_numeric_batch<int32_t>(*space, right_ids, cudf::type_id::INT32);

  auto f = create_mark_join();
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs = f.hash_join->execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  auto out_view =
    outputs->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table().view();
  REQUIRE(out_view.num_columns() == 3);
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == left_ids);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(1)) == left_payload);
  REQUIRE(copy_column_to_host<bool>(out_view.column(2)) ==
          std::vector<bool>{false, true, false, true, false});
}

TEST_CASE("sirius_physical_hash_join mark join - all rows match", "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {10, 20, 30};
  std::vector<int32_t> left_payload = {1, 2, 3};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Right contains every left key — all marks should be true
  std::vector<int32_t> right_ids = {10, 20, 30};
  auto right_batch = make_numeric_batch<int32_t>(*space, right_ids, cudf::type_id::INT32);

  auto f = create_mark_join();
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs = f.hash_join->execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  auto out_view =
    outputs->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table().view();
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == left_ids);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(1)) == left_payload);
  REQUIRE(copy_column_to_host<bool>(out_view.column(2)) == std::vector<bool>{true, true, true});
}

TEST_CASE("sirius_physical_hash_join mark join - no rows match", "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {10, 20, 30};
  std::vector<int32_t> left_payload = {1, 2, 3};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Right has completely disjoint keys — all marks should be false
  std::vector<int32_t> right_ids = {40, 50, 60};
  auto right_batch = make_numeric_batch<int32_t>(*space, right_ids, cudf::type_id::INT32);

  auto f = create_mark_join();
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs = f.hash_join->execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  auto out_view =
    outputs->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table().view();
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == left_ids);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(1)) == left_payload);
  REQUIRE(copy_column_to_host<bool>(out_view.column(2)) == std::vector<bool>{false, false, false});
}

TEST_CASE("sirius_physical_hash_join mark join - empty right side", "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {10, 20, 30};
  std::vector<int32_t> left_payload = {1, 2, 3};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Empty right table — semi_indices will be empty, all marks should be false
  std::vector<int32_t> right_ids = {};
  auto right_batch = make_numeric_batch<int32_t>(*space, right_ids, cudf::type_id::INT32);

  auto f = create_mark_join();
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs = f.hash_join->execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  auto out_view =
    outputs->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table().view();
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == left_ids);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(1)) == left_payload);
  REQUIRE(copy_column_to_host<bool>(out_view.column(2)) == std::vector<bool>{false, false, false});
}

TEST_CASE("sirius_physical_hash_join mark join - duplicate keys on right side",
          "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {10, 20, 30};
  std::vector<int32_t> left_payload = {1, 2, 3};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Right has key 20 repeated three times — left row 1 should still get mark=true exactly once
  std::vector<int32_t> right_ids = {20, 20, 20};
  auto right_batch = make_numeric_batch<int32_t>(*space, right_ids, cudf::type_id::INT32);

  auto f = create_mark_join();
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs = f.hash_join->execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  auto out_view =
    outputs->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table().view();
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == left_ids);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(1)) == left_payload);
  REQUIRE(copy_column_to_host<bool>(out_view.column(2)) == std::vector<bool>{false, true, false});
}
