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

#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/planner/bound_query_node.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_order.hpp>

using namespace sirius::op;
using sirius::op::operator_data;
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

  auto gpu_repr = std::make_unique<gpu_table_representation>(std::move(table), space);
  auto batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<data_batch>(batch_id, std::move(gpu_repr));
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));

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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));

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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch1, batch2}));
  REQUIRE(out->get_data_batches().size() == 2);

  // Each batch is independently sorted
  auto t1   = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col1 = copy_column_to_host<int64_t>(t1.view().column(0));
  REQUIRE(col1 == std::vector<int64_t>{1, 3, 5});

  auto t2   = out->get_data_batches()[1]->get_data()->cast<gpu_table_representation>().get_table();
  auto col2 = copy_column_to_host<int64_t>(t2.view().column(0));
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, batch, nullptr};
  auto out = op.execute(operator_data(inputs));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));
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

  sirius_physical_order op(std::move(types), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, nullptr};
  auto out = op.execute(operator_data(inputs));
  REQUIRE(out->get_data_batches().empty());
}
