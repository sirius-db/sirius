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
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_merge_sort.hpp>
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

// Helper: sort a batch using sirius_physical_order so it can be used as merge input
std::shared_ptr<data_batch> sort_batch(const std::shared_ptr<data_batch>& batch,
                                       const duckdb::vector<duckdb::BoundOrderByNode>& orders,
                                       const duckdb::vector<duckdb::idx_t>& projections,
                                       const duckdb::vector<duckdb::LogicalType>& types)
{
  sirius_physical_order order_op(duckdb::vector<duckdb::LogicalType>(types),
                                 copy_orders(orders),
                                 duckdb::vector<duckdb::idx_t>(projections),
                                 0);
  auto result = order_op.execute(operator_data({batch}));
  REQUIRE(result->get_data_batches().size() == 1);
  return result->get_data_batches()[0];
}

}  // namespace

// ---------------------------------------------------------------------------
// 1-column tests
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_merge_sort merges 2 sorted 1-column batches ascending",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  // Two pre-sorted ascending batches
  auto batch1 = make_1col_batch(*space, {1, 3, 5, 7});
  auto batch2 = make_1col_batch(*space, {2, 4, 6, 8});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_merge_sort op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch1, batch2}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));

  std::vector<int64_t> expected{1, 2, 3, 4, 5, 6, 7, 8};
  REQUIRE(col0 == expected);
}

TEST_CASE("sirius_physical_merge_sort merges 3 sorted 1-column batches descending",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch1 = make_1col_batch(*space, {9, 6, 3});
  auto batch2 = make_1col_batch(*space, {8, 5, 2});
  auto batch3 = make_1col_batch(*space, {7, 4, 1});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::DESCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_merge_sort op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch1, batch2, batch3}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));

  std::vector<int64_t> expected{9, 8, 7, 6, 5, 4, 3, 2, 1};
  REQUIRE(col0 == expected);
}

// ---------------------------------------------------------------------------
// 2-column tests
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_merge_sort merges 2 sorted 2-column batches by col0 asc",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  // Pre-sorted by col0 ascending
  auto batch1 = make_2col_batch(*space, {1, 3, 5}, {10, 30, 50});
  auto batch2 = make_2col_batch(*space, {2, 4, 6}, {20, 40, 60});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0, 1};

  sirius_physical_merge_sort op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch1, batch2}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));

  REQUIRE(out_c0 == std::vector<int64_t>{1, 2, 3, 4, 5, 6});
  REQUIRE(out_c1 == std::vector<int64_t>{10, 20, 30, 40, 50, 60});
}

TEST_CASE("sirius_physical_merge_sort merges 2-column batches sorted by 2 keys",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));
  orders.push_back(make_order(
    1, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0, 1};

  // Pre-sort batches using the order operator
  auto raw1    = make_2col_batch(*space, {2, 1, 1, 3}, {20, 10, 5, 30});
  auto sorted1 = sort_batch(raw1, orders, projections, types);
  // sorted1: (1,5), (1,10), (2,20), (3,30)

  auto raw2    = make_2col_batch(*space, {1, 2, 3, 2}, {15, 25, 35, 1});
  auto sorted2 = sort_batch(raw2, orders, projections, types);
  // sorted2: (1,15), (2,1), (2,25), (3,35)

  sirius_physical_merge_sort op(duckdb::vector<duckdb::LogicalType>(types),
                                copy_orders(orders),
                                duckdb::vector<duckdb::idx_t>(projections),
                                0);

  auto out = op.execute(operator_data({sorted1, sorted2}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));

  // Merged: (1,5), (1,10), (1,15), (2,1), (2,20), (2,25), (3,30), (3,35)
  REQUIRE(out_c0 == std::vector<int64_t>{1, 1, 1, 2, 2, 2, 3, 3});
  REQUIRE(out_c1 == std::vector<int64_t>{5, 10, 15, 1, 20, 25, 30, 35});
}

// ---------------------------------------------------------------------------
// 3-column tests
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_merge_sort merges 3-column batches sorted by col0, returns all",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  // Pre-sorted by col0 ascending
  auto batch1 = make_3col_batch(*space, {1, 3, 5}, {10, 30, 50}, {100, 300, 500});
  auto batch2 = make_3col_batch(*space, {2, 4, 6}, {20, 40, 60}, {200, 400, 600});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0, 1, 2};

  sirius_physical_merge_sort op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch1, batch2}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));
  auto out_c2 = copy_column_to_host<int64_t>(view.column(2));

  REQUIRE(out_c0 == std::vector<int64_t>{1, 2, 3, 4, 5, 6});
  REQUIRE(out_c1 == std::vector<int64_t>{10, 20, 30, 40, 50, 60});
  REQUIRE(out_c2 == std::vector<int64_t>{100, 200, 300, 400, 500, 600});
}

TEST_CASE("sirius_physical_merge_sort 3 columns sorted by col0 asc + col1 desc",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

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

  // Pre-sort using the order operator
  auto raw1    = make_3col_batch(*space, {1, 2, 1}, {20, 10, 30}, {100, 200, 300});
  auto sorted1 = sort_batch(raw1, orders, projections, types);
  // sorted1: (1,30,300), (1,20,100), (2,10,200)

  auto raw2    = make_3col_batch(*space, {2, 1, 2}, {40, 25, 5}, {400, 500, 600});
  auto sorted2 = sort_batch(raw2, orders, projections, types);
  // sorted2: (1,25,500), (2,40,400), (2,5,600)

  sirius_physical_merge_sort op(duckdb::vector<duckdb::LogicalType>(types),
                                copy_orders(orders),
                                duckdb::vector<duckdb::idx_t>(projections),
                                0);

  auto out = op.execute(operator_data({sorted1, sorted2}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
  auto out_c0 = copy_column_to_host<int64_t>(view.column(0));
  auto out_c1 = copy_column_to_host<int64_t>(view.column(1));
  auto out_c2 = copy_column_to_host<int64_t>(view.column(2));

  // Merged by (col0 asc, col1 desc):
  // (1,30,300), (1,25,500), (1,20,100), (2,40,400), (2,10,200), (2,5,600)
  REQUIRE(out_c0 == std::vector<int64_t>{1, 1, 1, 2, 2, 2});
  REQUIRE(out_c1 == std::vector<int64_t>{30, 25, 20, 40, 10, 5});
  REQUIRE(out_c2 == std::vector<int64_t>{300, 500, 100, 400, 200, 600});
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("sirius_physical_merge_sort single batch passthrough", "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch = make_1col_batch(*space, {1, 2, 3});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_merge_sort op(std::move(types), std::move(orders), std::move(projections), 0);

  auto out = op.execute(operator_data({batch}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));
  REQUIRE(col0 == std::vector<int64_t>{1, 2, 3});
}

TEST_CASE("sirius_physical_merge_sort skips null batches", "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch1 = make_1col_batch(*space, {1, 3, 5});
  auto batch2 = make_1col_batch(*space, {2, 4, 6});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_merge_sort op(std::move(types), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, batch1, nullptr, batch2, nullptr};
  auto out = op.execute(operator_data(inputs));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));
  REQUIRE(col0 == std::vector<int64_t>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("sirius_physical_merge_sort returns empty for all-null inputs", "[physical_merge_sort]")
{
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_merge_sort op(std::move(types), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, nullptr};
  auto out = op.execute(operator_data(inputs));
  REQUIRE(out->get_data_batches().empty());
}

TEST_CASE("sirius_physical_merge_sort constructed from sirius_physical_order",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto batch1 = make_1col_batch(*space, {1, 3, 5});
  auto batch2 = make_1col_batch(*space, {2, 4, 6});

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(
    0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::OrderType::ASCENDING));

  duckdb::vector<duckdb::idx_t> projections{0};

  sirius_physical_order order_op(duckdb::vector<duckdb::LogicalType>(types),
                                 copy_orders(orders),
                                 duckdb::vector<duckdb::idx_t>(projections),
                                 0);

  sirius_physical_merge_sort op(&order_op);

  auto out = op.execute(operator_data({batch1, batch2}));
  REQUIRE(out->get_data_batches().size() == 1);

  auto table = out->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto col0  = copy_column_to_host<int64_t>(table.view().column(0));
  REQUIRE(col0 == std::vector<int64_t>{1, 2, 3, 4, 5, 6});
}
