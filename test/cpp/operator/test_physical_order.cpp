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
#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_order.hpp>

using namespace sirius::op;
using sirius::op::operator_data;
using sirius::op::pipelineable_operator_data;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

namespace {

duckdb::BoundOrderByNode make_order(
  duckdb::idx_t col_idx,
  duckdb::LogicalType type,
  duckdb::OrderType dir         = duckdb::OrderType::ASCENDING,
  duckdb::OrderByNullType nulls = duckdb::OrderByNullType::NULLS_LAST)
{
  return duckdb::BoundOrderByNode(
    dir, nulls, duckdb::make_uniq<duckdb::BoundReferenceExpression>(std::move(type), col_idx));
}

std::shared_ptr<data_batch> make_1col_batch(memory_space& space, const std::vector<int64_t>& vals)
{
  return make_numeric_batch<int64_t>(space, vals, cudf::type_id::INT64);
}

std::shared_ptr<data_batch> make_2col_batch(memory_space& space,
                                            const std::vector<int64_t>& col0,
                                            const std::vector<int64_t>& col1)
{
  return make_two_column_batch<int64_t, int64_t>(
    space, col0, col1, cudf::type_id::INT64, std::nullopt);
}

std::shared_ptr<data_batch> make_3col_batch(memory_space& space,
                                            const std::vector<int64_t>& col0,
                                            const std::vector<int64_t>& col1,
                                            const std::vector<int64_t>& col2)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();
  auto size   = static_cast<cudf::size_type>(col0.size());

  auto make_col = [&](const std::vector<int64_t>& vals) {
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, size, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpy(col->mutable_view().data<int64_t>(),
               vals.data(),
               sizeof(int64_t) * vals.size(),
               cudaMemcpyHostToDevice);
    return col;
  };

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_col(col0));
  cols.push_back(make_col(col1));
  cols.push_back(make_col(col2));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto gpu_repr =
    std::make_unique<gpu_table_representation>(std::move(table), space, cudf::get_default_stream());
  auto batch_id = ::sirius::get_next_batch_id();
  return data_batch::make(batch_id, std::move(gpu_repr));
}

}  // namespace

// ---------------------------------------------------------------------------
// 1-column tests
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_order sorts 1 column ascending", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch = make_1col_batch(*space, {5, 1, 9, 3, 7, 2, 8});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));

  std::vector<int64_t> expected{1, 2, 3, 5, 7, 8, 9};
  REQUIRE(col0 == expected);
}

TEST_CASE("sirius_physical_order sorts 1 column descending", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch = make_1col_batch(*space, {5, 1, 9, 3, 7, 2, 8});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::DESCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));

  std::vector<int64_t> expected{9, 8, 7, 5, 3, 2, 1};
  REQUIRE(col0 == expected);
}

// ---------------------------------------------------------------------------
// 2-column tests
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_order sorts by col0, returns both columns", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  //                          col0           col1 (payload)
  auto batch = make_2col_batch(*space, {3, 1, 4, 1, 5}, {30, 10, 40, 11, 50});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0, 1};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));

  // Stable sort not guaranteed — only check order keys are sorted
  REQUIRE(std::is_sorted(out_c0.begin(), out_c0.end()));
  REQUIRE(out_c0.size() == 5);
  REQUIRE(out_c1.size() == 5);
}

TEST_CASE("sirius_physical_order sorts by 2 keys (asc, desc)", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  //                          col0           col1
  auto batch = make_2col_batch(*space, {2, 2, 1, 1, 3}, {10, 20, 30, 40, 50});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));
  orders.push_back(make_order(
    1, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::DESCENDING));

  duckdb::vector<duckdb::idx_t> projections{0, 1};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));

  // Expected: (1,40), (1,30), (2,20), (2,10), (3,50)
  std::vector<int64_t> expected_c0{1, 1, 2, 2, 3};
  std::vector<int64_t> expected_c1{40, 30, 20, 10, 50};
  REQUIRE(out_c0 == expected_c0);
  REQUIRE(out_c1 == expected_c1);
}

TEST_CASE("sirius_physical_order projects only col1 when sorting by col0", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch = make_2col_batch(*space, {3, 1, 2}, {30, 10, 20});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  // Only project col1 (payload), not col0 (sort key)
  duckdb::vector<duckdb::idx_t> projections{1};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  REQUIRE(view.num_columns() == 1);

  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  std::vector<int64_t> expected{10, 20, 30};
  REQUIRE(out_c0 == expected);
}

// ---------------------------------------------------------------------------
// 3-column tests
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_order 3 columns, sort by col0 asc, return all", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch = make_3col_batch(*space, {3, 1, 2}, {30, 10, 20}, {300, 100, 200});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0, 1, 2};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));
  auto out_c2 = copy_column_to_host<int64_t>(view.column(2));

  REQUIRE(out_c0 == std::vector<int64_t>{1, 2, 3});
  REQUIRE(out_c1 == std::vector<int64_t>{10, 20, 30});
  REQUIRE(out_c2 == std::vector<int64_t>{100, 200, 300});
}

TEST_CASE("sirius_physical_order 3 columns, sort by col0 asc + col1 desc, return all",
          "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  //  col0  col1  col2
  //   1     20   100
  //   1     10   200
  //   2     30   300
  //   2     40   400
  //   3     50   500
  auto batch =
    make_3col_batch(*space, {1, 1, 2, 2, 3}, {20, 10, 30, 40, 50}, {100, 200, 300, 400, 500});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));
  orders.push_back(make_order(
    1, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::DESCENDING));

  duckdb::vector<duckdb::idx_t> projections{0, 1, 2};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));
  auto out_c2 = copy_column_to_host<int64_t>(view.column(2));

  // Expected: (1,20,100), (1,10,200), (2,40,400), (2,30,300), (3,50,500)
  REQUIRE(out_c0 == std::vector<int64_t>{1, 1, 2, 2, 3});
  REQUIRE(out_c1 == std::vector<int64_t>{20, 10, 40, 30, 50});
  REQUIRE(out_c2 == std::vector<int64_t>{100, 200, 400, 300, 500});
}

TEST_CASE("sirius_physical_order 3 columns, sort by col0, project col1 and col2 only",
          "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch = make_3col_batch(*space, {3, 1, 2}, {30, 10, 20}, {300, 100, 200});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{1, 2};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  REQUIRE(view.num_columns() == 2);

  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));

  REQUIRE(out_c0 == std::vector<int64_t>{10, 20, 30});
  REQUIRE(out_c1 == std::vector<int64_t>{100, 200, 300});
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_order handles multiple batches", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch1 = make_1col_batch(*space, {5, 3, 1});
  auto batch2 = make_1col_batch(*space, {4, 2, 6});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = op.execute(pipelineable_operator_data({batch1, batch2}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 2);

  // Each batch is independently sorted
  auto t1 = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col1 = copy_column_to_host<int64_t>(t1.column(0));
  REQUIRE(col1 == std::vector<int64_t>{1, 3, 5});

  auto t2 = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[1]);
  auto col2 = copy_column_to_host<int64_t>(t2.column(0));
  REQUIRE(col2 == std::vector<int64_t>{2, 4, 6});
}

TEST_CASE("sirius_physical_order handles null input batches", "[physical_order]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch = make_1col_batch(*space, {3, 1, 2});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, batch, nullptr};
  auto out = op.execute(pipelineable_operator_data(inputs), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));
  REQUIRE(col0 == std::vector<int64_t>{1, 2, 3});
}

TEST_CASE("sirius_physical_order returns empty for all-null inputs", "[physical_order]")
{
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, nullptr};
  auto out = op.execute(pipelineable_operator_data(inputs), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().empty());
}

// ---------------------------------------------------------------------------
// Run coalescing: which batches form one sorted run
// ---------------------------------------------------------------------------

namespace {

using parked_batch = sirius_physical_order::parked_batch;

parked_batch parked(uint64_t id, std::size_t bytes, std::size_t rows, int device = 0)
{
  return parked_batch{.id = id, .bytes = bytes, .rows = rows, .device = device};
}

sort_run_limits limits_of(std::size_t bytes)
{
  sort_run_limits limits;
  limits.max_run_bytes = bytes;
  return limits;
}

std::vector<std::size_t> indices(std::initializer_list<std::size_t> values) { return values; }

}  // namespace

TEST_CASE("sirius_physical_order run planning waits for a full group", "[physical_order][coalesce]")
{
  std::vector<parked_batch> batches{parked(1, 40, 4), parked(2, 40, 4)};
  // 80 of 100 bytes: more may arrive, so no group yet.
  CHECK(sirius_physical_order::plan_next_run(batches, limits_of(100), false, false).empty());
  // The producer finished: the remainder is one group.
  CHECK(sirius_physical_order::plan_next_run(batches, limits_of(100), false, true) ==
        indices({0, 1}));
  // Nothing parked.
  CHECK(sirius_physical_order::plan_next_run({}, limits_of(100), false, true).empty());
}

TEST_CASE("sirius_physical_order run planning closes a group at the byte limit",
          "[physical_order][coalesce]")
{
  // The third batch would take the group past the limit: it starts the next group.
  std::vector<parked_batch> batches{parked(1, 40, 4), parked(2, 40, 4), parked(3, 40, 4)};
  CHECK(sirius_physical_order::plan_next_run(batches, limits_of(100), false, false) ==
        indices({0, 1}));
  // A group that reaches the limit exactly is complete without waiting for another batch.
  CHECK(sirius_physical_order::plan_next_run(batches, limits_of(80), false, false) ==
        indices({0, 1}));
  // A batch larger than the limit is a group of one.
  std::vector<parked_batch> big{parked(1, 500, 4), parked(2, 10, 1)};
  CHECK(sirius_physical_order::plan_next_run(big, limits_of(100), false, false) == indices({0}));
  // A small batch followed by an oversized one: the small one closes alone.
  std::vector<parked_batch> small_big{parked(1, 10, 1), parked(2, 500, 4)};
  CHECK(sirius_physical_order::plan_next_run(small_big, limits_of(100), false, false) ==
        indices({0}));
}

TEST_CASE("sirius_physical_order run planning respects the row and string limits",
          "[physical_order][coalesce]")
{
  auto limits         = limits_of(1000);
  limits.max_run_rows = 10;
  std::vector<parked_batch> batches{parked(1, 10, 6), parked(2, 10, 3), parked(3, 10, 2)};
  // 6 + 3 rows fit, 2 more do not.
  CHECK(sirius_physical_order::plan_next_run(batches, limits, false, false) == indices({0, 1}));
  // An unknown row count does not close a group; execute() checks the real tables.
  std::vector<parked_batch> unknown{
    parked_batch{.id = 1, .bytes = 10, .rows = std::nullopt, .device = 0}, parked(2, 10, 3)};
  CHECK(sirius_physical_order::plan_next_run(unknown, limits, false, false).empty());

  // With a string column the decoded bytes of a group stay within the string limit.
  auto string_limits                    = limits_of(1000);
  string_limits.max_string_column_bytes = 50;
  std::vector<parked_batch> strings{parked(1, 30, 1), parked(2, 30, 1)};
  CHECK(sirius_physical_order::plan_next_run(strings, string_limits, true, false) == indices({0}));
  CHECK(sirius_physical_order::plan_next_run(strings, string_limits, false, false).empty());
}

TEST_CASE("sirius_physical_order run planning groups batches by device",
          "[physical_order][coalesce]")
{
  // Device 0 and device 1 alternate; device 1 fills first.
  std::vector<parked_batch> batches{parked(1, 30, 1, 0),
                                    parked(2, 60, 1, 1),
                                    parked(3, 30, 1, 0),
                                    parked(4, 60, 1, 1),
                                    parked(5, 30, 1, -1)};
  CHECK(sirius_physical_order::plan_next_run(batches, limits_of(100), false, false) ==
        indices({1}));
  // At the end the remainder is packed in repository order, whatever its device.
  std::vector<parked_batch> tail{parked(1, 30, 1, 0), parked(3, 30, 1, 1), parked(5, 30, 1, -1)};
  CHECK(sirius_physical_order::plan_next_run(tail, limits_of(100), false, false).empty());
  CHECK(sirius_physical_order::plan_next_run(tail, limits_of(100), false, true) ==
        indices({0, 1, 2}));
  CHECK(sirius_physical_order::plan_next_run(tail, limits_of(70), false, true) == indices({0, 1}));
}

TEST_CASE("sirius_physical_order run planning consumes every batch exactly once",
          "[physical_order][coalesce]")
{
  // Drain a mixed repository the way the task creator does and check the partition of the ids.
  std::vector<parked_batch> remaining;
  for (uint64_t id = 0; id < 97; ++id) {
    remaining.push_back(parked(id, 7 + (id * 13) % 50, 1 + id % 9, static_cast<int>(id % 3) - 1));
  }
  auto limits         = limits_of(120);
  limits.max_run_rows = 40;
  std::vector<uint64_t> claimed;
  std::size_t groups = 0;
  while (!remaining.empty()) {
    auto const group = sirius_physical_order::plan_next_run(remaining, limits, false, true);
    REQUIRE_FALSE(group.empty());
    std::size_t bytes = 0;
    std::size_t rows  = 0;
    for (auto const index : group) {
      bytes += remaining[index].bytes;
      rows += *remaining[index].rows;
      claimed.push_back(remaining[index].id);
    }
    if (group.size() > 1) {
      CHECK(bytes <= limits.max_run_bytes);
      CHECK(rows <= limits.max_run_rows);
    }
    for (auto it = group.rbegin(); it != group.rend(); ++it) {
      remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(*it));
    }
    ++groups;
  }
  std::sort(claimed.begin(), claimed.end());
  REQUIRE(claimed.size() == 97);
  for (uint64_t id = 0; id < 97; ++id) {
    CHECK(claimed[id] == id);
  }
  CHECK(groups < 97);
}

TEST_CASE("sirius_physical_order run planning bounds strings by their own bytes",
          "[physical_order][coalesce]")
{
  auto limits                    = limits_of(1000);
  limits.max_string_column_bytes = 100;
  auto const with_strings = [](uint64_t id, std::size_t bytes, std::size_t string_bytes) {
    auto batch         = parked(id, bytes, 10);
    batch.string_bytes = string_bytes;
    return batch;
  };
  // 300 decoded bytes each, 40 of them characters: two batches fit under the string limit, the
  // third starts the next group.
  std::vector<parked_batch> batches{
    with_strings(1, 300, 40), with_strings(2, 300, 40), with_strings(3, 300, 40)};
  CHECK(sirius_physical_order::plan_next_run(batches, limits, true, false) == indices({0, 1}));
  // Without a string column the bound does not apply and the byte limit closes the group.
  CHECK(sirius_physical_order::plan_next_run(batches, limits, false, false).empty());
  CHECK(sirius_physical_order::plan_next_run(batches, limits, false, true) == indices({0, 1, 2}));
  // A group that reaches the string limit exactly is complete.
  std::vector<parked_batch> exact{with_strings(1, 300, 50), with_strings(2, 300, 50)};
  CHECK(sirius_physical_order::plan_next_run(exact, limits, true, false) == indices({0, 1}));
  // A batch without a bound of its own counts with all its bytes.
  std::vector<parked_batch> unbounded{with_strings(1, 300, 40), parked(2, 300, 10)};
  CHECK(sirius_physical_order::plan_next_run(unbounded, limits, true, false) == indices({0}));
}

TEST_CASE("sirius_physical_order run planning holds a group back while its GPU sorts enough",
          "[physical_order][coalesce]")
{
  auto limits                 = limits_of(100);
  limits.max_concurrent_bytes = 200;
  std::vector<parked_batch> batches{parked(1, 60, 1, 0), parked(2, 60, 1, 0), parked(3, 60, 1, 0)};

  using sorting = sirius_physical_order::sorting_bytes_by_device;
  // Nothing sorting, or room for the group: it is claimed.
  CHECK(sirius_physical_order::plan_next_run(batches, limits, false, false, sorting{}) ==
        indices({0}));
  CHECK(sirius_physical_order::plan_next_run(batches, limits, false, false, sorting{{0, 140}}) ==
        indices({0}));
  // No room on device 0: its group waits, also when the producer has finished.
  CHECK(sirius_physical_order::plan_next_run(batches, limits, false, false, sorting{{0, 141}})
          .empty());
  CHECK(
    sirius_physical_order::plan_next_run(batches, limits, false, true, sorting{{0, 141}}).empty());
  // What another device sorts does not count.
  CHECK(sirius_physical_order::plan_next_run(batches, limits, false, false, sorting{{1, 500}}) ==
        indices({0}));
  // A device that is not sorting takes a group larger than the limit.
  std::vector<parked_batch> big{parked(1, 900, 1, 0)};
  CHECK(sirius_physical_order::plan_next_run(big, limits, false, false, sorting{{0, 0}}) ==
        indices({0}));
  CHECK(sirius_physical_order::plan_next_run(big, limits, false, false, sorting{{0, 1}}).empty());

  // The group of a busy device is passed over for the complete group of another one.
  std::vector<parked_batch> two_devices{parked(1, 60, 1, 0),
                                        parked(2, 60, 1, 0),
                                        parked(3, 60, 1, 1),
                                        parked(4, 60, 1, 0),
                                        parked(5, 60, 1, 1)};
  CHECK(sirius_physical_order::plan_next_run(
          two_devices, limits, false, false, sorting{{0, 190}}) == indices({2}));
  // Without the limit the planning is unchanged.
  CHECK(sirius_physical_order::plan_next_run(
          two_devices, limits_of(100), false, false, sorting{{0, 190}}) == indices({0}));

  // The tail counts against the device holding most of it.
  std::vector<parked_batch> tail{parked(1, 30, 1, 0), parked(2, 40, 1, 1), parked(3, 20, 1, -1)};
  CHECK(sirius_physical_order::run_device(tail, {0, 1, 2}) == 1);
  CHECK(sirius_physical_order::run_device(tail, {2}) == -1);
  CHECK(sirius_physical_order::plan_next_run(tail, limits, false, true, sorting{{1, 150}}).empty());
  CHECK(sirius_physical_order::plan_next_run(tail, limits, false, true, sorting{{0, 150}}) ==
        indices({0, 1, 2}));
}

TEST_CASE("sirius_physical_order run limits from the engine sizes", "[physical_order][coalesce]")
{
  auto const gib = std::size_t{1} << 30;
  // Two full runs at a time: a run is half the share, the share bounds the sorts running at
  // once, and the producer is held back at twice the share of every GPU.
  auto const limits = sirius::op::make_sort_run_limits(80 * gib, 1, 8, 0.25, 2, true);
  REQUIRE(limits);
  CHECK(limits->max_run_bytes == 10 * gib);
  CHECK(limits->max_concurrent_bytes == 20 * gib);
  CHECK(limits->max_backlog_bytes == 40 * gib);
  CHECK(sirius::op::make_sort_run_limits(80 * gib, 4, 8, 0.25, 2, true)->max_backlog_bytes ==
        160 * gib);
  // The backlog limit exceeds what the groups still being filled can hold (one per GPU and one
  // for batches that are not on a GPU), so a complete group exists whenever it is reached.
  for (std::size_t gpus : {1, 2, 4, 8}) {
    for (std::size_t runs : {1, 2, 8}) {
      auto const l = sirius::op::make_sort_run_limits(80 * gib, gpus, 8, 0.25, runs, true);
      REQUIRE(l);
      CHECK(l->max_backlog_bytes >= (gpus + 1) * l->max_run_bytes);
    }
  }
  // 0 concurrent runs: the previous rule, one run per executor thread and no bound.
  auto const per_thread = sirius::op::make_sort_run_limits(80 * gib, 1, 8, 0.25, 0, false);
  REQUIRE(per_thread);
  CHECK(per_thread->max_run_bytes == 20 * gib / 8);
  CHECK(per_thread->max_concurrent_bytes == 0);
  CHECK(per_thread->max_backlog_bytes == 0);
  CHECK_FALSE(sirius::op::make_sort_run_limits(0, 1, 8, 0.25, 2, true));
  CHECK_FALSE(sirius::op::make_sort_run_limits(4, 1, 8, 0.25, 2, true));
}

TEST_CASE("sirius_physical_order bytes per row that are not string characters",
          "[physical_order][coalesce]")
{
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR));
  auto const logical = sirius::from_duckdb_vec(std::move(types));
  // 8 + 4 and a 32-bit offset per string.
  CHECK(sirius_physical_order::non_character_row_bytes(logical, {}) == 16);
  // A narrower physical carrier counts at its own width, a wider one at the logical width.
  CHECK(sirius_physical_order::non_character_row_bytes(logical,
                                                       {cudf::data_type{cudf::type_id::INT32},
                                                        cudf::data_type{cudf::type_id::INT64},
                                                        cudf::data_type{cudf::type_id::STRING}}) ==
        12);
}

TEST_CASE("sirius_physical_order coalescing sorts a task's batches as one run",
          "[physical_order][coalesce]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto make_op = [](sort_run_limits limits) {
    duckdb::vector<duckdb::LogicalType> types;
    types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
    types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(make_order(
      0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::DESCENDING));
    duckdb::vector<duckdb::idx_t> projections{0, 1};
    auto op = std::make_unique<sirius_physical_order>(
      sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);
    op->enable_run_coalescing(limits);
    return op;
  };
  auto make_input = [&] {
    return std::vector<std::shared_ptr<data_batch>>{make_2col_batch(*space, {5, 3, 1}, {50, 30, 10}),
                                                    nullptr,
                                                    make_2col_batch(*space, {4, 2, 6}, {40, 20, 60}),
                                                    make_2col_batch(*space, {9, 0}, {90, 0})};
  };
  auto column = [](const operator_data& out, std::size_t batch, cudf::size_type col) {
    auto table = sirius::get_cudf_table_view(
      *dynamic_cast<const pipelineable_operator_data&>(out).get_data_batches()[batch]);
    return copy_column_to_host<int64_t>(table.column(col));
  };

  SECTION("one run when the limits allow")
  {
    auto op  = make_op(limits_of(1 << 20));
    auto out = op->execute(pipelineable_operator_data(make_input()), cudf::get_default_stream());
    REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);
    CHECK(column(*out, 0, 0) == std::vector<int64_t>{9, 6, 5, 4, 3, 2, 1, 0});
    // The payload follows its key.
    CHECK(column(*out, 0, 1) == std::vector<int64_t>{90, 60, 50, 40, 30, 20, 10, 0});
    auto const counters = op->counters();
    CHECK(counters.tasks == 1);
    CHECK(counters.input_batches == 3);
    CHECK(counters.runs == 1);
    CHECK(counters.concatenated_bytes > 0);
  }

  SECTION("the row limit splits the task into several runs")
  {
    auto limits         = limits_of(1 << 20);
    limits.max_run_rows = 6;
    auto op             = make_op(limits);
    auto out = op->execute(pipelineable_operator_data(make_input()), cudf::get_default_stream());
    // 3 + 3 rows, then 2: every row is in exactly one run and every run is sorted.
    REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 2);
    CHECK(column(*out, 0, 0) == std::vector<int64_t>{6, 5, 4, 3, 2, 1});
    CHECK(column(*out, 1, 0) == std::vector<int64_t>{9, 0});
    CHECK(op->counters().runs == 2);
  }

  SECTION("a single batch is sorted without a concatenation")
  {
    auto op = make_op(limits_of(1 << 20));
    auto out =
      op->execute(pipelineable_operator_data({make_2col_batch(*space, {2, 3, 1}, {20, 30, 10})}),
                  cudf::get_default_stream());
    REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);
    CHECK(column(*out, 0, 0) == std::vector<int64_t>{3, 2, 1});
    CHECK(op->counters().concatenated_bytes == 0);
  }

  SECTION("batches with different column types are separate runs")
  {
    auto op = make_op(limits_of(1 << 20));
    std::vector<std::shared_ptr<data_batch>> mixed{
      make_2col_batch(*space, {2, 3, 1}, {20, 30, 10}),
      make_two_column_batch<int64_t, int32_t>(
        *space, {7, 8}, {70, 80}, cudf::type_id::INT32)};
    auto out = op->execute(pipelineable_operator_data(mixed), cudf::get_default_stream());
    REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 2);
    CHECK(column(*out, 0, 0) == std::vector<int64_t>{3, 2, 1});
    CHECK(column(*out, 1, 0) == std::vector<int64_t>{8, 7});
  }
}

TEST_CASE("sirius_physical_order rejects a zero run size", "[physical_order][coalesce]")
{
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT)));
  sirius_physical_order op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), duckdb::vector<duckdb::idx_t>{0}, 0);
  CHECK_FALSE(op.run_coalescing_enabled());
  CHECK_THROWS_AS(op.enable_run_coalescing(sort_run_limits{}), std::invalid_argument);
}
