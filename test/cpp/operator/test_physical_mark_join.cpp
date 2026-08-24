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
#include <duckdb/common/exception.hpp>
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

//! Depth-first, root-first numbering of a bare operator tree, standing in for
//! pipeline::assign_operator_ids in fixtures that never build pipelines.
void number_operator_tree(sirius::op::sirius_physical_operator& op, size_t& next_id)
{
  op.operator_id = next_id++;
  for (auto& child : op.children) {
    if (child) { number_operator_tree(*child, next_id); }
  }
}

struct nlj_projection_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_nested_loop_join> nlj;
};

struct projected_nlj_result {
  std::unique_ptr<operator_data> outputs;
  cudf::table_view view;
};

/**
 * @brief Create a mark join operator with two INT32 key columns joined by @p comparison
 * (left col[0] <comparison> right col[0]).
 * Left child has types {INTEGER, INTEGER} (key + payload), right child has {INTEGER} (key only).
 */
mark_join_fixture create_mark_join(
  duckdb::ExpressionType comparison = duckdb::ExpressionType::COMPARE_EQUAL)
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
  cond.comparison = comparison;
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
    1000);

  // No pipelines exist in this fixture, so the converter's assign_operator_ids never runs.
  // Number the tree here — operator code reads get_operator_id(), which rejects the sentinel.
  size_t next_id = 0;
  number_operator_tree(*f.hash_join, next_id);

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

// Two-condition NLJ MARK fixture: (left.col0 <cmp0> right.col0) AND (left.col2 <cmp1> right.col1).
// Left child {INT,INT,INT} (col0, payload col1, col2), right child {INT,INT}. Outputs the payload
// column plus the mark. Used to exercise three-valued conjunction semantics (NULL AND FALSE=FALSE)
// and mixed null-safe/null-propagating conjunctions.
nlj_projection_fixture create_projected_nlj_two_cond(duckdb::ExpressionType comparison0,
                                                     duckdb::ExpressionType comparison1)
{
  nlj_projection_fixture f;

  f.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::MARK);
  f.logical_join->types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::BOOLEAN};

  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
      duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER}),
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                duckdb::LogicalType::INTEGER}),
    0);

  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition cond0;
  cond0.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond0.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond0.comparison = comparison0;
  conditions.push_back(std::move(cond0));
  duckdb::JoinCondition cond1;
  cond1.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 2);
  cond1.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 1);
  cond1.comparison = comparison1;
  conditions.push_back(std::move(cond1));

  f.nlj = duckdb::make_uniq<sirius_physical_nested_loop_join>(
    *f.logical_join,
    std::move(left_child),
    std::move(right_child),
    sirius::wrap_join_conditions(std::move(conditions)),
    duckdb::JoinType::MARK,
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

// Issue #1076: MARK joins must emit a NULL mark (not false) for an unmatched left row when the
// build/right side contains a NULL join key. left {1,2,3} vs right {2, NULL} -> [NULL, true, NULL].
TEST_CASE("sirius_physical_hash_join mark join - right side has NULL key", "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {1, 2, 3};
  std::vector<int32_t> left_payload = {10, 20, 30};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Right key column = {2, NULL}: only 2 is a real key; the NULL taints every non-match to NULL.
  auto right_batch =
    make_numeric_batch_with_nulls<int32_t>(*space, {2, 0}, {true, false}, cudf::type_id::INT32);

  auto f = create_mark_join();
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(out_view.num_rows() == 3);

  auto mark = out_view.column(2);
  REQUIRE(mark.null_count() == 2);
  REQUIRE(copy_validity_to_host(mark) == std::vector<bool>{false, true, false});
  REQUIRE(copy_column_to_host<bool>(mark)[1] == true);  // the one matched row
}

// Issue #1076: an unmatched left row with a NULL probe key is NULL, not false, even when the build
// side has no NULL key. left {1, NULL, 2} vs right {2} -> [false, NULL, true].
TEST_CASE("sirius_physical_hash_join mark join - probe side has NULL key", "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  // Left key column carries a NULL at row 1; payload is a plain column.
  auto left_key = make_numeric_batch_with_nulls<int32_t>(
    *space, {1, 0, 2}, {true, false, true}, cudf::type_id::INT32);
  auto left_payload = make_numeric_batch<int32_t>(*space, {10, 20, 30}, cudf::type_id::INT32);
  auto left_batch   = concatenate_batches_horizontal({left_key, left_payload}, *space);

  auto right_batch = make_numeric_batch<int32_t>(*space, {2}, cudf::type_id::INT32);

  auto f = create_mark_join();
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(out_view.num_rows() == 3);

  auto mark = out_view.column(2);
  REQUIRE(mark.null_count() == 1);
  REQUIRE(copy_validity_to_host(mark) == std::vector<bool>{true, false, true});
  auto values = copy_column_to_host<bool>(mark);
  REQUIRE(values[0] == false);  // 1 has no match, probe valid, right clean -> false
  REQUIRE(values[2] == true);   // 2 matches -> true
}

// Issue #1076: the same NULL semantics must hold on the build-on-left (cudf::mark_join) path.
TEST_CASE("sirius_physical_hash_join mark join - right NULL key on build-on-left path",
          "[physical_mark_join]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  std::vector<int32_t> left_ids     = {10, 20, 30, 40};
  std::vector<int32_t> left_payload = {1, 2, 3, 4};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Right (probe) side larger than left to trigger the switch; {20, 40} match and a NULL is
  // present.
  auto right_batch =
    make_numeric_batch_with_nulls<int32_t>(*space,
                                           {20, 40, 0, 11, 12, 13, 14},
                                           {true, true, false, true, true, true, true},
                                           cudf::type_id::INT32);

  auto f                                    = create_mark_join();
  f.hash_join->mark_join_build_switch_ratio = 1.0;

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());

  auto out_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(out_view.num_rows() == 4);

  auto mark = out_view.column(2);
  REQUIRE(mark.null_count() == 2);
  REQUIRE(copy_validity_to_host(mark) == std::vector<bool>{false, true, false, true});
}

// A MARK join must run in BUILD_PROBE mode, which is mutually exclusive with MIXED_JOIN. A MARK
// join carrying both an equality and an inequality condition (the shape that would otherwise select
// MIXED_JOIN) is therefore rejected at construction rather than silently mis-executed. No GPU is
// needed: the constructor makes the mode decision from the conditions alone.
TEST_CASE("sirius_physical_hash_join mark join - mixed conditions are unsupported",
          "[physical_mark_join]")
{
  auto logical_join   = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::MARK);
  logical_join->types = {
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

  // Equality (left.col0 = right.col0) + inequality (left.col1 < right.col0) => mixed shape.
  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition eq;
  eq.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  eq.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  eq.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(eq));
  duckdb::JoinCondition lt;
  lt.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 1);
  lt.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  lt.comparison = duckdb::ExpressionType::COMPARE_LESSTHAN;
  conditions.push_back(std::move(lt));

  REQUIRE_THROWS_AS(duckdb::make_uniq<sirius_physical_hash_join>(
                      *logical_join,
                      std::move(left_child),
                      std::move(right_child),
                      sirius::wrap_join_conditions(std::move(conditions)),
                      duckdb::JoinType::MARK,
                      duckdb::vector<duckdb::idx_t>{},
                      duckdb::vector<duckdb::idx_t>{},
                      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
                      1000),
                    std::runtime_error);
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

// Issue #1119: the NLJ (conditional) MARK path must emit a NULL mark for an unmatched left row
// when the predicate was never TRUE but was UNKNOWN (NULL) for some right row.
TEST_CASE("sirius_physical_nested_loop_join MARK emits NULL under three-valued logic",
          "[physical_nested_loop_join][projection][mark]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  // Predicate is left.col0 < right.col0 — a pure inequality, the shape that reaches the NLJ in
  // production plans.
  auto make_left = [&](const std::vector<int32_t>& a,
                       const std::vector<bool>& a_valid,
                       const std::vector<int32_t>& payload) {
    auto col0 = a_valid.empty() ? make_numeric_batch<int32_t>(*space, a, cudf::type_id::INT32)
                                : make_numeric_batch_with_nulls<int32_t>(
                                    *space, a, a_valid, cudf::type_id::INT32);
    auto col1 = make_numeric_batch<int32_t>(*space, payload, cudf::type_id::INT32);
    auto col2 =
      make_numeric_batch<int32_t>(*space, std::vector<int32_t>(a.size(), 0), cudf::type_id::INT32);
    return concatenate_batches_horizontal({col0, col1, col2}, *space);
  };

  SECTION("right NULL taints an otherwise-false row to NULL")
  {
    auto left = make_left({1, 5, 10}, {}, {10, 20, 30});
    auto right =
      make_numeric_batch_with_nulls<int32_t>(*space, {8, 0}, {true, false}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::MARK, duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_rows() == 3);
    auto mark = out_view.column(1);
    REQUIRE(mark.null_count() == 1);
    REQUIRE(copy_validity_to_host(mark) == std::vector<bool>{true, true, false});
    auto values = copy_column_to_host<bool>(mark);
    REQUIRE(values[0] == true);
    REQUIRE(values[1] == true);
  }

  SECTION("NULL probe key is NULL, not false")
  {
    auto left  = make_left({1, 0, 20}, {true, false, true}, {10, 20, 30});
    auto right = make_numeric_batch<int32_t>(*space, {10}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::MARK, duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    REQUIRE(out_view.num_rows() == 3);
    auto mark = out_view.column(1);
    REQUIRE(mark.null_count() == 1);
    REQUIRE(copy_validity_to_host(mark) == std::vector<bool>{true, false, true});
    auto values = copy_column_to_host<bool>(mark);
    REQUIRE(values[0] == true);   // 1 < 10
    REQUIRE(values[2] == false);  // 20 < 10 is definitively false
  }

  SECTION("definitely-false rows with no nulls stay false")
  {
    auto left  = make_left({20, 30}, {}, {10, 20});
    auto right = make_numeric_batch<int32_t>(*space, {10}, cudf::type_id::INT32);

    auto f = create_projected_nlj(duckdb::JoinType::MARK, duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    auto mark = out_view.column(1);
    REQUIRE(mark.null_count() == 0);
    REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{false, false});
  }
}

// Issue #1119: with a conjunction, a NULL in one comparison must be absorbed by a FALSE in another
// (NULL AND FALSE == FALSE) — a naive "any referenced null -> NULL" would be wrong here.
TEST_CASE("sirius_physical_nested_loop_join MARK conjunction absorbs NULL with FALSE",
          "[physical_nested_loop_join][projection][mark]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  // Predicate: (left.col0 < right.col0) AND (left.col2 < right.col1).
  auto make_left = [&](int32_t a, int32_t payload, int32_t c) {
    auto col0 = make_numeric_batch<int32_t>(*space, {a}, cudf::type_id::INT32);
    auto col1 = make_numeric_batch<int32_t>(*space, {payload}, cudf::type_id::INT32);
    auto col2 = make_numeric_batch<int32_t>(*space, {c}, cudf::type_id::INT32);
    return concatenate_batches_horizontal({col0, col1, col2}, *space);
  };
  auto make_right = [&](const std::vector<int32_t>& b,
                        const std::vector<bool>& b_valid,
                        const std::vector<int32_t>& d,
                        const std::vector<bool>& d_valid) {
    auto col0 = make_numeric_batch_with_nulls<int32_t>(*space, b, b_valid, cudf::type_id::INT32);
    auto col1 = make_numeric_batch_with_nulls<int32_t>(*space, d, d_valid, cudf::type_id::INT32);
    return concatenate_batches_horizontal({col0, col1}, *space);
  };

  SECTION("every right row has a FALSE conjunct -> mark FALSE despite the NULLs")
  {
    auto left = make_left(/*a*/ 100, /*payload*/ 10, /*c*/ 100);
    // r0=(b=NULL,d=5): (100<NULL)=NULL AND (100<5)=FALSE -> FALSE
    // r1=(b=10,d=NULL): (100<10)=FALSE AND (100<NULL)=NULL -> FALSE
    auto right = make_right({0, 10}, {false, true}, {5, 0}, {true, false});

    auto f        = create_projected_nlj_two_cond(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                           duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    auto mark = out_view.column(1);
    REQUIRE(mark.null_count() == 0);
    REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{false});
  }

  SECTION("a right row that is TRUE-then-NULL (no FALSE conjunct) yields NULL")
  {
    auto left = make_left(/*a*/ 100, /*payload*/ 10, /*c*/ 1);
    // r0=(b=NULL,d=5): (100<NULL)=NULL AND (1<5)=TRUE -> NULL, and there is no TRUE match anywhere.
    auto right = make_right({0}, {false}, {5}, {true});

    auto f        = create_projected_nlj_two_cond(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                           duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    auto mark = out_view.column(1);
    REQUIRE(mark.null_count() == 1);
    REQUIRE(copy_validity_to_host(mark) == std::vector<bool>{false});
  }
}

// Issue #1119 (review): the null-safe comparisons IS [NOT] DISTINCT FROM yield a definite
// TRUE/FALSE for NULL operands, never UNKNOWN — their rows must not be tainted into NULL marks.
TEST_CASE("sirius_physical_nested_loop_join MARK null-safe comparisons yield definite marks",
          "[physical_nested_loop_join][projection][mark]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  // Left col0 (the key) = {NULL, 5}, payload = {10, 20}; predicate col0 <cmp> right.col0.
  // Every section expects a mark column with no NULLs.
  auto make_left = [&](const std::vector<int32_t>& a,
                       const std::vector<bool>& a_valid,
                       const std::vector<int32_t>& payload) {
    auto col0 = make_numeric_batch_with_nulls<int32_t>(*space, a, a_valid, cudf::type_id::INT32);
    auto col1 = make_numeric_batch<int32_t>(*space, payload, cudf::type_id::INT32);
    auto col2 =
      make_numeric_batch<int32_t>(*space, std::vector<int32_t>(a.size(), 0), cudf::type_id::INT32);
    return concatenate_batches_horizontal({col0, col1, col2}, *space);
  };
  auto run = [&](duckdb::ExpressionType cmp, std::shared_ptr<cucascade::data_batch> right) {
    auto left = make_left({0, 5}, {false, true}, {10, 20});  // col0 = {NULL, 5}
    auto f    = create_projected_nlj(duckdb::JoinType::MARK, cmp);
    return execute_projected_nlj(*f.nlj, std::move(left), std::move(right));
  };

  SECTION("IS NOT DISTINCT FROM, right {5}: NULL vs 5 FALSE, 5 vs 5 TRUE")
  {
    auto result = run(duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM,
                      make_numeric_batch<int32_t>(*space, {5}, cudf::type_id::INT32));
    auto mark   = result.view.column(1);
    REQUIRE(mark.null_count() == 0);
    REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{false, true});
  }

  SECTION("IS NOT DISTINCT FROM, right {NULL}: NULL vs NULL TRUE, 5 vs NULL FALSE")
  {
    auto result =
      run(duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM,
          make_numeric_batch_with_nulls<int32_t>(*space, {0}, {false}, cudf::type_id::INT32));
    auto mark = result.view.column(1);
    REQUIRE(mark.null_count() == 0);
    REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{true, false});
  }

  SECTION("IS DISTINCT FROM, right {5}: NULL vs 5 TRUE, 5 vs 5 FALSE")
  {
    auto result = run(duckdb::ExpressionType::COMPARE_DISTINCT_FROM,
                      make_numeric_batch<int32_t>(*space, {5}, cudf::type_id::INT32));
    auto mark   = result.view.column(1);
    REQUIRE(mark.null_count() == 0);
    REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{true, false});
  }

  SECTION("IS DISTINCT FROM, right {NULL}: NULL vs NULL FALSE, 5 vs NULL TRUE")
  {
    auto result =
      run(duckdb::ExpressionType::COMPARE_DISTINCT_FROM,
          make_numeric_batch_with_nulls<int32_t>(*space, {0}, {false}, cudf::type_id::INT32));
    auto mark = result.view.column(1);
    REQUIRE(mark.null_count() == 0);
    REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{false, true});
  }
}

// Issue #1119 (review): the realistic reachable path — a mixed conjunction that pairs a null-safe
// comparison (IS NOT DISTINCT FROM) with a null-propagating one (<). Only the propagating conjunct
// may be null-tainted; the null-safe conjunct must keep its definite TRUE/FALSE.
TEST_CASE("sirius_physical_nested_loop_join MARK mixed null-safe + null-propagating conjunction",
          "[physical_nested_loop_join][projection][mark]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  // Predicate: (left.col0 IS NOT DISTINCT FROM right.col0) AND (left.col2 < right.col1).
  auto make_left = [&](int32_t a, bool a_valid, int32_t payload, int32_t c, bool c_valid) {
    auto col0 =
      make_numeric_batch_with_nulls<int32_t>(*space, {a}, {a_valid}, cudf::type_id::INT32);
    auto col1 = make_numeric_batch<int32_t>(*space, {payload}, cudf::type_id::INT32);
    auto col2 =
      make_numeric_batch_with_nulls<int32_t>(*space, {c}, {c_valid}, cudf::type_id::INT32);
    return concatenate_batches_horizontal({col0, col1, col2}, *space);
  };
  auto make_right = [&](int32_t b, int32_t d) {
    auto col0 = make_numeric_batch<int32_t>(*space, {b}, cudf::type_id::INT32);
    auto col1 = make_numeric_batch<int32_t>(*space, {d}, cudf::type_id::INT32);
    return concatenate_batches_horizontal({col0, col1}, *space);
  };

  SECTION("null-safe conjunct is a definite FALSE -> whole predicate FALSE, mark FALSE")
  {
    // (NULL IS NOT DISTINCT FROM 5) = FALSE, (1 < 100) = TRUE -> FALSE AND TRUE = FALSE.
    auto left  = make_left(/*a*/ 0, /*a_valid*/ false, /*payload*/ 10, /*c*/ 1, /*c_valid*/ true);
    auto right = make_right(/*b*/ 5, /*d*/ 100);

    auto f        = create_projected_nlj_two_cond(duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM,
                                           duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    auto mark = out_view.column(1);
    REQUIRE(mark.null_count() == 0);
    REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{false});
  }

  SECTION("null-safe conjunct TRUE but propagating conjunct UNKNOWN -> mark NULL")
  {
    // (5 IS NOT DISTINCT FROM 5) = TRUE, (NULL < 100) = UNKNOWN -> TRUE AND UNKNOWN = UNKNOWN.
    auto left  = make_left(/*a*/ 5, /*a_valid*/ true, /*payload*/ 10, /*c*/ 0, /*c_valid*/ false);
    auto right = make_right(/*b*/ 5, /*d*/ 100);

    auto f        = create_projected_nlj_two_cond(duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM,
                                           duckdb::ExpressionType::COMPARE_LESSTHAN);
    auto result   = execute_projected_nlj(*f.nlj, left, right);
    auto out_view = result.view;

    auto mark = out_view.column(1);
    REQUIRE(mark.null_count() == 1);
    REQUIRE(copy_validity_to_host(mark) == std::vector<bool>{false});
  }
}

// DuckDB's PhysicalNestedLoopJoin::IsSupported admits RIGHT_SEMI, RIGHT_ANTI and SINGLE, which
// used to survive planning and then hit the execute switch's `default:` throw mid-query.
TEST_CASE("sirius_physical_nested_loop_join rejects join types it cannot execute",
          "[physical_nested_loop_join][join_type]")
{
  using JT = duckdb::JoinType;

  for (auto join_type : {JT::INNER, JT::LEFT, JT::RIGHT, JT::SEMI, JT::ANTI, JT::MARK, JT::OUTER}) {
    INFO("join type " << duckdb::JoinTypeToString(join_type));
    REQUIRE(sirius_physical_nested_loop_join::is_join_type_supported(join_type));
  }

  for (auto join_type : {JT::RIGHT_SEMI, JT::RIGHT_ANTI, JT::SINGLE}) {
    INFO("join type " << duckdb::JoinTypeToString(join_type));
    REQUIRE_FALSE(sirius_physical_nested_loop_join::is_join_type_supported(join_type));
    // The DuckDB-mirroring condition check must agree, so neither entry point can admit them.
    REQUIRE_FALSE(sirius_physical_nested_loop_join::is_supported(
      duckdb::vector<sirius::join_condition>{}, join_type));
    // Backstop: constructing one anyway throws inside plan generation (which falls back to CPU)
    // rather than aborting the query from execute().
    REQUIRE_THROWS_AS(create_projected_nlj(join_type, duckdb::ExpressionType::COMPARE_EQUAL),
                      duckdb::NotImplementedException);
  }
}

// SINGLE reaches neither GPU join: both dispatches fall through to a mid-query throw.
// are_conditions_supported now folds in the join-type screen so the planner rejects it.
TEST_CASE("sirius_physical_hash_join rejects join types it cannot execute",
          "[physical_hash_join][join_type]")
{
  using JT = duckdb::JoinType;

  for (auto join_type : {JT::INNER,
                         JT::LEFT,
                         JT::RIGHT,
                         JT::SEMI,
                         JT::ANTI,
                         JT::RIGHT_SEMI,
                         JT::RIGHT_ANTI,
                         JT::MARK,
                         JT::OUTER}) {
    INFO("join type " << duckdb::JoinTypeToString(join_type));
    REQUIRE(sirius_physical_hash_join::is_join_type_supported(join_type));
  }
  REQUIRE_FALSE(sirius_physical_hash_join::is_join_type_supported(JT::SINGLE));

  // A plain equality condition the hash join would otherwise accept.
  auto make_conditions = [](duckdb::ExpressionType comparison) {
    duckdb::vector<duckdb::JoinCondition> conditions;
    duckdb::JoinCondition cond;
    cond.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
    cond.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
    cond.comparison = comparison;
    conditions.push_back(std::move(cond));
    return sirius::wrap_join_conditions(std::move(conditions));
  };

  auto equal_conditions = make_conditions(duckdb::ExpressionType::COMPARE_EQUAL);
  REQUIRE(sirius_physical_hash_join::are_conditions_supported(equal_conditions, JT::INNER));
  REQUIRE(sirius_physical_hash_join::are_conditions_supported(equal_conditions, JT::MARK));
  // Rejected on the join type alone, not the conditions.
  REQUIRE_FALSE(sirius_physical_hash_join::are_conditions_supported(equal_conditions, JT::SINGLE));

  // An all-null-safe MARK is supported: one EQUAL covers every key and the marks are definite.
  auto null_safe_conditions = make_conditions(duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM);
  REQUIRE(sirius_physical_hash_join::are_conditions_supported(null_safe_conditions, JT::INNER));
  REQUIRE(sirius_physical_hash_join::are_conditions_supported(null_safe_conditions, JT::MARK));

  // Mixing the two is not, since one null_equality cannot express both policies.
  duckdb::vector<duckdb::JoinCondition> mixed;
  for (auto comparison :
       {duckdb::ExpressionType::COMPARE_EQUAL, duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM}) {
    duckdb::JoinCondition cond;
    auto const idx = static_cast<duckdb::idx_t>(mixed.size());
    cond.left      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, idx);
    cond.right     = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, idx);
    cond.comparison = comparison;
    mixed.push_back(std::move(cond));
  }
  auto mixed_conditions = sirius::wrap_join_conditions(std::move(mixed));
  REQUIRE(sirius_physical_hash_join::are_conditions_supported(mixed_conditions, JT::INNER));
  REQUIRE_FALSE(sirius_physical_hash_join::are_conditions_supported(mixed_conditions, JT::MARK));
}

// An all-null-safe MARK runs under EQUAL: NULL matches NULL and no comparison is UNKNOWN, so the
// mark column carries no nulls. Under the previous UNEQUAL pin a NULL probe key produced NULL.
TEST_CASE("sirius_physical_hash_join MARK with all null-safe keys emits definite marks",
          "[physical_hash_join][mark][nulls]")
{
  auto* space = get_shared_mem_space();
  REQUIRE(space);

  // compare_nulls()/mark_is_null_safe() are internal, so assert the semantics instead: NULL
  // matching NULL proves EQUAL, a null-free mark column proves definiteness.
  auto f = create_mark_join(duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM);

  // Left keys {NULL, 5, 7} with payload; right (build) keys {NULL, 5}.
  auto left_key = make_numeric_batch_with_nulls<int32_t>(
    *space, {0, 5, 7}, {false, true, true}, cudf::type_id::INT32);
  auto left_payload = make_numeric_batch<int32_t>(*space, {10, 20, 30}, cudf::type_id::INT32);
  auto left         = concatenate_batches_horizontal({left_key, left_payload}, *space);
  auto right =
    make_numeric_batch_with_nulls<int32_t>(*space, {0, 5}, {false, true}, cudf::type_id::INT32);

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left, right};
  auto outputs =
    f.hash_join->execute(pipelineable_operator_data(inputs), cudf::get_default_stream());
  auto const& output_data = dynamic_cast<const pipelineable_operator_data&>(*outputs);
  REQUIRE(output_data.get_data_batches().size() == 1);
  auto view = sirius::get_cudf_table_view(*output_data.get_data_batches()[0]);

  // NULL matches NULL, 5 matches 5, 7 matches nothing -- and the miss is FALSE, never NULL.
  auto mark = view.column(view.num_columns() - 1);
  REQUIRE(mark.null_count() == 0);
  REQUIRE(copy_column_to_host<bool>(mark) == std::vector<bool>{true, true, false});
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
