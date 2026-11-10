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
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius/exception.hpp"
#include "utils/test_physical_operator.hpp"

#include <cudf/concatenate.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_merge_sort.hpp>
#include <op/sirius_physical_order.hpp>
#include <op/sirius_physical_sort_partition.hpp>
#include <op/sirius_physical_sort_sample.hpp>

#include <algorithm>
#include <functional>
#include <random>

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

// Helper: sort a batch using sirius_physical_order so it can be used as merge input
std::shared_ptr<data_batch> sort_batch(const std::shared_ptr<data_batch>& batch,
                                       const duckdb::vector<duckdb::BoundOrderByNode>& orders,
                                       const duckdb::vector<duckdb::idx_t>& projections,
                                       const duckdb::vector<duckdb::LogicalType>& types)
{
  sirius_physical_order order_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>(types)),
    types.size(),
    copy_orders(orders),
    duckdb::vector<duckdb::idx_t>(projections),
    0);
  auto result = order_op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches().size() == 1);
  return dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches()[0];
}

}  // namespace

// Drive the same claim/prepare/execute/publish lifecycle as production. Each round
// claims every currently available task, then completes them in reverse claim order,
// so publication order differs from claim order. Returns final chunks in position order.
std::vector<std::shared_ptr<data_batch>> run_ranges(
  sirius_physical_merge_sort& merge, std::vector<std::vector<std::shared_ptr<data_batch>>> ranges)
{
  cucascade::shared_data_repository_manager repositories;
  cucascade::shared_data_repository input, output;
  sirius::op::test_physical_operator destination(SiriusPhysicalOperatorType::FILTER, {}, 0);
  merge.operator_id = 1;
  merge.register_refinement(repositories);
  merge.add_port("default",
                 std::make_unique<sirius_physical_operator::port>(
                   MemoryBarrierType::FULL, &input, nullptr, nullptr));
  destination.add_port("default",
                       std::make_unique<sirius_physical_operator::port>(
                         MemoryBarrierType::FULL, &output, nullptr, nullptr));
  merge.add_next_port_after_sink({&destination, "default"});
  for (std::size_t range = 0; range < ranges.size(); ++range) {
    for (auto const& batch : ranges[range]) {
      if (batch) { input.add_data_batch(batch, range); }
    }
  }
  while (true) {
    std::vector<std::unique_ptr<operator_data>> claimed;
    while (auto work = merge.get_next_task_input_data()) {
      claimed.push_back(std::move(work));
    }
    if (claimed.empty()) { break; }
    for (auto it = claimed.rbegin(); it != claimed.rend(); ++it) {
      auto& work = dynamic_cast<pipelineable_operator_data&>(**it);
      work.prepare_for_processing(get_default_gpu_space(), default_stream());
      auto result = merge.execute(work, default_stream());
      default_stream().sync();
      merge.sink(*result, default_stream());
      default_stream().sync();
      merge.on_task_output_published(*result);
    }
  }
  REQUIRE(merge.all_ports_empty());
  std::vector<std::shared_ptr<data_batch>> results;
  if (output.all_empty()) { return results; }
  auto positions = output.get_positions();
  std::sort(positions.begin(), positions.end());
  for (auto position : positions) {
    results.push_back(output.pop_data_batch_by_position(position));
  }
  return results;
}

std::unique_ptr<operator_data> run_merge(sirius_physical_merge_sort& merge,
                                         std::vector<std::shared_ptr<data_batch>> batches)
{
  auto results = run_ranges(merge, {std::move(batches)});
  if (results.size() <= 1) {
    return std::make_unique<pipelineable_operator_data>(std::move(results));
  }
  std::vector<cudf::table_view> views;
  for (auto const& batch : results) {
    views.push_back(sirius::get_cudf_table_view(*batch));
  }
  auto* space    = get_default_gpu_space();
  auto table     = cudf::concatenate(views, default_stream(), get_resource_ref(*space));
  auto collected = sirius::make_data_batch(std::move(table), *space, default_stream(), {});
  default_stream().sync();
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<data_batch>>{std::move(collected)});
}

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

  sirius_physical_merge_sort op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = run_merge(op, {batch1, batch2});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));

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

  sirius_physical_merge_sort op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = run_merge(op, {batch1, batch2, batch3});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));

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

  sirius_physical_merge_sort op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = run_merge(op, {batch1, batch2});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
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

  sirius_physical_merge_sort op(sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>(types)),
                                copy_orders(orders),
                                duckdb::vector<duckdb::idx_t>(projections),
                                0);

  auto out = run_merge(op, {sorted1, sorted2});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
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

  sirius_physical_merge_sort op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = run_merge(op, {batch1, batch2});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
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

  sirius_physical_merge_sort op(sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>(types)),
                                copy_orders(orders),
                                duckdb::vector<duckdb::idx_t>(projections),
                                0);

  auto out = run_merge(op, {sorted1, sorted2});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
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

  sirius_physical_merge_sort op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  auto out = run_merge(op, {batch});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));
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

  sirius_physical_merge_sort op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, batch1, nullptr, batch2, nullptr};
  auto out = run_merge(op, inputs);
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));
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

  sirius_physical_merge_sort op(
    sirius::from_duckdb_vec(std::move(types)), std::move(orders), std::move(projections), 0);

  std::vector<std::shared_ptr<data_batch>> inputs{nullptr, nullptr};
  auto out = run_merge(op, inputs);
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().empty());
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

  sirius_physical_order order_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>(types)),
    types.size(),
    copy_orders(orders),
    duckdb::vector<duckdb::idx_t>(projections),
    0);

  sirius_physical_merge_sort op(&order_op);

  auto out = run_merge(op, {batch1, batch2});
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  auto col0 = copy_column_to_host<int64_t>(table.column(0));
  REQUIRE(col0 == std::vector<int64_t>{1, 2, 3, 4, 5, 6});
}

// ---------------------------------------------------------------------------
// Refinement of oversized ranges
// ---------------------------------------------------------------------------

namespace {

std::unique_ptr<sirius_physical_merge_sort> make_bigint_merge(duckdb::OrderType direction,
                                                              std::size_t chunk_bytes)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), direction));
  auto op = std::make_unique<sirius_physical_merge_sort>(
    duckdb::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::BIGINT)},
    std::move(orders),
    duckdb::vector<std::size_t>{0},
    0);
  op->chunk_bytes = chunk_bytes;
  return op;
}

std::vector<int64_t> sorted_values(std::mt19937_64& rng, std::size_t rows, int64_t range, bool desc)
{
  std::uniform_int_distribution<int64_t> dist(0, range);
  std::vector<int64_t> values(rows);
  for (auto& value : values) {
    value = dist(rng);
  }
  std::sort(values.begin(), values.end());
  if (desc) { std::reverse(values.begin(), values.end()); }
  return values;
}

// Every chunk is bounded by the byte target's row equivalent; concatenation is ordered.
std::vector<int64_t> collect_chunks(std::vector<std::shared_ptr<data_batch>> const& chunks,
                                    std::size_t max_rows)
{
  std::vector<int64_t> all;
  for (auto const& chunk : chunks) {
    auto values = copy_column_to_host<int64_t>(sirius::get_cudf_table_view(*chunk).column(0));
    CHECK(!values.empty());
    CHECK(values.size() <= max_rows);
    all.insert(all.end(), values.begin(), values.end());
  }
  return all;
}

}  // namespace

TEST_CASE("merge sort refines an oversized range into bounded ordered chunks",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  std::mt19937_64 rng(42);
  std::vector<std::shared_ptr<data_batch>> inputs;
  std::vector<int64_t> expected;
  // Unequal contributions, many duplicate keys.
  for (std::size_t rows : {1000, 37, 600, 1500, 1}) {
    auto values = sorted_values(rng, rows, 300, false);
    expected.insert(expected.end(), values.begin(), values.end());
    inputs.push_back(make_1col_batch(*space, values));
  }
  std::sort(expected.begin(), expected.end());

  std::size_t const chunk_bytes = 2048;
  auto op                       = make_bigint_merge(duckdb::OrderType::ASCENDING, chunk_bytes);
  auto chunks                   = run_ranges(*op, {inputs});
  CHECK(chunks.size() > 1);
  REQUIRE(collect_chunks(chunks, chunk_bytes / sizeof(int64_t)) == expected);
}

TEST_CASE("merge sort refines all-equal keys using contribution coordinates",
          "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  std::vector<std::shared_ptr<data_batch>> inputs;
  for (int i = 0; i < 3; ++i) {
    inputs.push_back(make_1col_batch(*space, std::vector<int64_t>(500, 7)));
  }
  std::size_t const chunk_bytes = 1024;
  auto op                       = make_bigint_merge(duckdb::OrderType::ASCENDING, chunk_bytes);
  auto chunks                   = run_ranges(*op, {inputs});
  CHECK(chunks.size() > 1);
  REQUIRE(collect_chunks(chunks, chunk_bytes / sizeof(int64_t)) == std::vector<int64_t>(1500, 7));
}

TEST_CASE("merge sort keeps range order when only some ranges are refined", "[physical_merge_sort]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  std::mt19937_64 rng(7);
  // Descending: range 0 holds the largest keys. Range 1 fits; ranges 0 and 2 do not.
  auto shift = [](std::vector<int64_t> values, int64_t offset) {
    for (auto& value : values) {
      value += offset;
    }
    return values;
  };
  std::vector<std::vector<int64_t>> range_values{shift(sorted_values(rng, 900, 99, true), 2000),
                                                 shift(sorted_values(rng, 20, 99, true), 1000),
                                                 sorted_values(rng, 700, 99, true)};
  std::vector<std::vector<std::shared_ptr<data_batch>>> ranges(3);
  std::vector<int64_t> expected;
  for (std::size_t r = 0; r < range_values.size(); ++r) {
    auto const& values = range_values[r];
    auto half          = values.size() / 2;
    // Two sorted contributions per range.
    std::vector<int64_t> a(values.begin(), values.begin() + half);
    std::vector<int64_t> b(values.begin() + half, values.end());
    ranges[r].push_back(make_1col_batch(*space, a));
    ranges[r].push_back(make_1col_batch(*space, b));
    expected.insert(expected.end(), values.begin(), values.end());
  }
  std::sort(expected.begin(), expected.end(), std::greater<>{});

  std::size_t const chunk_bytes = 1600;
  auto op                       = make_bigint_merge(duckdb::OrderType::DESCENDING, chunk_bytes);
  auto chunks                   = run_ranges(*op, ranges);
  CHECK(chunks.size() > 3);
  REQUIRE(collect_chunks(chunks, chunk_bytes / sizeof(int64_t)) == expected);
}

TEST_CASE("sort lowering shares executable comparison arguments", "[sort_comparison][cpu]")
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(1,
                              duckdb::LogicalType::BIGINT,
                              duckdb::OrderType::DESCENDING,
                              duckdb::OrderByNullType::NULLS_LAST));
  orders.push_back(make_order(0,
                              duckdb::LogicalType::BIGINT,
                              duckdb::OrderType::ASCENDING,
                              duckdb::OrderByNullType::NULLS_FIRST));
  sirius_physical_order local({sirius::logical_type::make(sirius::type_id::BIGINT),
                               sirius::logical_type::make(sirius::type_id::BIGINT)},
                              2,
                              std::move(orders),
                              {0, 1},
                              0);
  sirius_physical_sort_sample sample(&local);
  sirius_physical_sort_partition partition(&local);
  sirius_physical_merge_sort merge(&local);
  REQUIRE(sample.comparison == local.comparison);
  REQUIRE(partition.comparison == local.comparison);
  REQUIRE(merge.comparison == local.comparison);
  CHECK(local.comparison->key_columns == std::vector<int>{1, 0});
  CHECK(local.comparison->directions ==
        std::vector<cudf::order>{cudf::order::DESCENDING, cudf::order::ASCENDING});
  CHECK(local.comparison->null_precedence ==
        std::vector<cudf::null_order>{cudf::null_order::BEFORE, cudf::null_order::BEFORE});
}

TEST_CASE("ORDER BY expansion retains hidden keys through the shared sort chain",
          "[sort_comparison][cpu]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);
  auto bigint = sirius::logical_type::make(sirius::type_id::BIGINT);
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(1,
                              duckdb::LogicalType::BIGINT,
                              duckdb::OrderType::DESCENDING,
                              duckdb::OrderByNullType::NULLS_FIRST));
  duckdb::unique_ptr<sirius_physical_operator> plan =
    duckdb::make_uniq<sirius_physical_order>(duckdb::vector<sirius::logical_type>{bigint},
                                             2,
                                             std::move(orders),
                                             duckdb::vector<std::size_t>{0},
                                             10);
  plan->children.push_back(duckdb::make_uniq<test_physical_operator>(
    SiriusPhysicalOperatorType::INVALID, duckdb::vector<sirius::logical_type>{bigint, bigint}, 10));
  sirius::planner::sirius_physical_plan_generator generator(*connection.context);
  generator.insert_gpu_pipeline_operators(plan);
  REQUIRE(plan->type == SiriusPhysicalOperatorType::MERGE_SORT);
  auto& merge = plan->Cast<sirius_physical_merge_sort>();
  REQUIRE(merge.children.size() == 1);
  auto& partition = merge.children[0]->Cast<sirius_physical_sort_partition>();
  CHECK(merge.input_barrier_for(partition) == MemoryBarrierType::FULL);
  auto& sample = partition.children[0]->Cast<sirius_physical_sort_sample>();
  auto& local  = sample.children[0]->Cast<sirius_physical_order>();
  CHECK(merge.types.size() == 1);
  CHECK(partition.types.size() == 2);
  CHECK(sample.types.size() == 2);
  CHECK(local.types.size() == 2);
  CHECK(local.projections == duckdb::vector<std::size_t>{0, 1});
  CHECK(local.comparison == sample.comparison);
  CHECK(local.comparison == partition.comparison);
  CHECK(local.comparison == merge.comparison);
  CHECK(local.comparison->key_columns == std::vector<int>{1});
}

TEST_CASE("sort stages reject out-of-schema keys during construction", "[sort_comparison][cpu]")
{
  auto bigint = sirius::logical_type::make(sirius::type_id::BIGINT);
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(1, duckdb::LogicalType::BIGINT));
  CHECK_THROWS_AS(sort_comparison_config(orders, 1), sirius::invalid_input_exception);
  CHECK_NOTHROW(sort_comparison_config(orders, 2));
  CHECK_THROWS_AS(sirius_physical_order({bigint}, 1, copy_orders(orders), {0}, 0),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(sirius_physical_merge_sort({bigint}, copy_orders(orders), {0}, 0),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(sirius_physical_sort_partition({bigint}, copy_orders(orders), {0}, 0),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(sirius_physical_sort_sample({bigint}, copy_orders(orders), 0),
                  sirius::invalid_input_exception);
}

TEST_CASE("sort comparison lowers absolute SQL null precedence for both directions",
          "[sort_comparison][cpu]")
{
  auto direction = GENERATE(duckdb::OrderType::ASCENDING, duckdb::OrderType::DESCENDING);
  auto nulls = GENERATE(duckdb::OrderByNullType::NULLS_FIRST, duckdb::OrderByNullType::NULLS_LAST);
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::BIGINT, direction, nulls));
  sort_comparison_config config(orders, 1);
  bool before =
    (nulls == duckdb::OrderByNullType::NULLS_FIRST) != (direction == duckdb::OrderType::DESCENDING);
  CHECK(config.null_precedence.front() ==
        (before ? cudf::null_order::BEFORE : cudf::null_order::AFTER));
  CHECK_THROWS_AS(sort_comparison_config(orders, 0), sirius::invalid_input_exception);
}

TEST_CASE("ORDER BY common consumer rejects nested payloads", "[sort_comparison][cpu]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);

  auto types = sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
    duckdb::LogicalType::BIGINT, duckdb::LogicalType::LIST(duckdb::LogicalType::BIGINT)});
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::BIGINT));
  duckdb::unique_ptr<sirius_physical_operator> plan = duckdb::make_uniq<sirius_physical_order>(
    types, 2, std::move(orders), duckdb::vector<std::size_t>{0, 1}, 10);
  plan->children.push_back(
    duckdb::make_uniq<test_physical_operator>(SiriusPhysicalOperatorType::INVALID, types, 10));
  sirius::planner::sirius_physical_plan_generator generator(*connection.context);
  CHECK_THROWS_AS(generator.insert_gpu_pipeline_operators(plan), sirius::not_implemented_exception);
}
