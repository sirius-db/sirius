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

#include "helper/type_conversions.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_nested_loop_join.hpp>

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

struct nlj_projection_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_nested_loop_join> nlj;
};

struct projected_nlj_result {
  std::unique_ptr<operator_data> outputs;
  cudf::table_view view;
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
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                duckdb::LogicalType::INTEGER}),
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
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
    sirius::wrap_join_conditions(std::move(conditions)),
    duckdb::JoinType::MARK,
    duckdb::vector<duckdb::idx_t>{},  // left_projection_map (empty = all)
    duckdb::vector<duckdb::idx_t>{},  // right_projection_map (not used by MARK)
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),  // delim_types
    1000,
    nullptr);

  // The engine freezes every hash join's dynamic-filter plan after planning and before any task
  // runs; a test that constructs the join directly must invoke the same freeze (this join has no
  // builder, so it receives the canonical disabled plan). Without this, the join's runtime
  // claim path would fail loudly — by design — when the first build batch arrives.
  {
    sirius::op::sirius_physical_hash_join* producers[] = {f.hash_join.get()};
    sirius::op::freeze_or_verify_dynamic_filter_plans(producers);
  }

  return f;
}

memory_space* get_shared_mem_space()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

std::shared_ptr<cucascade::data_batch> make_three_int32_batch(memory_space& space,
                                                              const std::vector<int32_t>& col0,
                                                              const std::vector<int32_t>& col1,
                                                              const std::vector<int32_t>& col2)
{
  auto b0 = make_numeric_batch<int32_t>(space, col0, cudf::type_id::INT32);
  auto b1 = make_numeric_batch<int32_t>(space, col1, cudf::type_id::INT32);
  auto b2 = make_numeric_batch<int32_t>(space, col2, cudf::type_id::INT32);
  return concatenate_batches_horizontal({b0, b1, b2}, space);
}

nlj_projection_fixture create_projected_nlj(duckdb::JoinType join_type,
                                            duckdb::ExpressionType comparison)
{
  nlj_projection_fixture f;

  f.logical_join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
  if (join_type == duckdb::JoinType::MARK) {
    f.logical_join->types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::BOOLEAN};
  } else {
    f.logical_join->types = {duckdb::LogicalType::INTEGER};
  }

  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
      duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER}),
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);

  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition cond;
  cond.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.comparison = comparison;
  conditions.push_back(std::move(cond));

  f.nlj = duckdb::make_uniq<sirius_physical_nested_loop_join>(
    *f.logical_join,
    std::move(left_child),
    std::move(right_child),
    sirius::wrap_join_conditions(std::move(conditions)),
    join_type,
    1000,
    duckdb::vector<std::size_t>{1},  // left_projection_map: output only payload column
    duckdb::vector<std::size_t>{});

  return f;
}

projected_nlj_result execute_projected_nlj(sirius_physical_nested_loop_join& nlj,
                                           std::shared_ptr<cucascade::data_batch> left,
                                           std::shared_ptr<cucascade::data_batch> right)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{std::move(left), std::move(right)};
  auto outputs = nlj.execute(pipelineable_operator_data(inputs), cudf::get_default_stream());
  auto const& output_data = dynamic_cast<const pipelineable_operator_data&>(*outputs);
  REQUIRE(output_data.get_data_batches().size() == 1);
  auto view = sirius::get_cudf_table_view(*output_data.get_data_batches()[0]);
  return projected_nlj_result{std::move(outputs), view};
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
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
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
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
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
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
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
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
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
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == left_ids);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(1)) == left_payload);
  REQUIRE(copy_column_to_host<bool>(out_view.column(2)) == std::vector<bool>{false, true, false});
}

TEST_CASE("sirius_physical_hash_join mark join - build-on-left (cudf::mark_join) path",
          "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {10, 20, 30, 40};
  std::vector<int32_t> left_payload = {1, 2, 3, 4};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Right (probe) side is larger than the left (output) side; only {20, 40} match.
  std::vector<int32_t> right_ids = {20, 40, 11, 12, 13, 14, 15, 16};
  auto right_batch = make_numeric_batch<int32_t>(*space, right_ids, cudf::type_id::INT32);

  auto f = create_mark_join();
  // Force the adaptive switch: with ratio 1.0 and right_rows (8) >= left_rows (4), the join must
  // build on the left via cudf::mark_join and probe with the right. Output must stay identical to
  // the filtered_join path.
  f.hash_join->mark_join_build_switch_ratio = 1.0;

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(out_view.num_columns() == 3);
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == left_ids);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(1)) == left_payload);
  REQUIRE(copy_column_to_host<bool>(out_view.column(2)) ==
          std::vector<bool>{false, true, false, true});
}

//===----------------------------------------------------------------------===//
// Nested-loop join projection-map regression tests
//===----------------------------------------------------------------------===//

TEST_CASE("sirius_physical_nested_loop_join MARK honors the left projection map",
          "[physical_nested_loop_join][projection][mark]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  auto left  = make_three_int32_batch(*space,
                                     /*key*/ {1, 2, 3},
                                     /*payload selected by left_projection_map*/ {10, 20, 30},
                                     /*unprojected sentinel*/ {100, 200, 300});
  auto right = make_numeric_batch<int32_t>(*space, {3}, cudf::type_id::INT32);

  auto f = create_projected_nlj(duckdb::JoinType::MARK, duckdb::ExpressionType::COMPARE_LESSTHAN);
  auto result   = execute_projected_nlj(*f.nlj, left, right);
  auto out_view = result.view;

  REQUIRE(out_view.num_columns() == 2);
  REQUIRE(out_view.num_rows() == 3);
  REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == std::vector<int32_t>{10, 20, 30});
  REQUIRE(copy_column_to_host<bool>(out_view.column(1)) == std::vector<bool>{true, true, false});
}

TEST_CASE("sirius_physical_nested_loop_join MARK empty side honors the left projection map",
          "[physical_nested_loop_join][projection][mark]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  SECTION("empty right side marks projected left rows false")
  {
    auto left  = make_three_int32_batch(*space, {1, 2, 3}, {10, 20, 30}, {100, 200, 300});
    auto right = make_numeric_batch<int32_t>(*space, {}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::MARK, duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_columns() == 2);
    REQUIRE(out_view.num_rows() == 3);
    REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == std::vector<int32_t>{10, 20, 30});
    REQUIRE(copy_column_to_host<bool>(out_view.column(1)) ==
            std::vector<bool>{false, false, false});
  }

  SECTION("empty left side still exposes the projected-left-plus-mark schema")
  {
    auto left  = make_three_int32_batch(*space, {}, {}, {});
    auto right = make_numeric_batch<int32_t>(*space, {1, 2}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::MARK, duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_columns() == 2);
    REQUIRE(out_view.num_rows() == 0);
  }
}

TEST_CASE("sirius_physical_nested_loop_join SEMI and ANTI honor the left projection map",
          "[physical_nested_loop_join][projection][semi][anti]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  SECTION("SEMI normal path projects only selected left columns")
  {
    auto left  = make_three_int32_batch(*space, {1, 2, 3}, {10, 20, 30}, {100, 200, 300});
    auto right = make_numeric_batch<int32_t>(*space, {2}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::SEMI, duckdb::ExpressionType::COMPARE_EQUAL);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_columns() == 1);
    REQUIRE(out_view.num_rows() == 1);
    REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == std::vector<int32_t>{20});
  }

  SECTION("ANTI normal path projects only selected left columns")
  {
    auto left  = make_three_int32_batch(*space, {1, 2, 3}, {10, 20, 30}, {100, 200, 300});
    auto right = make_numeric_batch<int32_t>(*space, {2}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::ANTI, duckdb::ExpressionType::COMPARE_EQUAL);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_columns() == 1);
    REQUIRE(out_view.num_rows() == 2);
    REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == std::vector<int32_t>{10, 30});
  }

  SECTION("SEMI empty-side arm keeps the projected schema")
  {
    auto left  = make_three_int32_batch(*space, {1, 2, 3}, {10, 20, 30}, {100, 200, 300});
    auto right = make_numeric_batch<int32_t>(*space, {}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::SEMI, duckdb::ExpressionType::COMPARE_EQUAL);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_columns() == 1);
    REQUIRE(out_view.num_rows() == 0);
  }

  SECTION("ANTI empty-side arm projects all preserved left rows")
  {
    auto left  = make_three_int32_batch(*space, {1, 2, 3}, {10, 20, 30}, {100, 200, 300});
    auto right = make_numeric_batch<int32_t>(*space, {}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::ANTI, duckdb::ExpressionType::COMPARE_EQUAL);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_columns() == 1);
    REQUIRE(out_view.num_rows() == 3);
    REQUIRE(copy_column_to_host<int32_t>(out_view.column(0)) == std::vector<int32_t>{10, 20, 30});
  }
}
