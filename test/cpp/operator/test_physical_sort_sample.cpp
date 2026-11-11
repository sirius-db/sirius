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
#include "utils/test_physical_operator.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_order.hpp>
#include <op/sirius_physical_sort_partition.hpp>
#include <op/sirius_physical_sort_sample.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius/exception.hpp>
#include <sirius_config.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;
using sirius::op::pipelineable_operator_data;

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

class mock_gpu_pipeline : public sirius::pipeline::sirius_pipeline {
 public:
  explicit mock_gpu_pipeline(const sirius::pipeline::pipeline_build_context& ctx)
    : sirius_pipeline(ctx), _finished(false)
  {
  }

  void set_finished(bool finished) { _finished = finished; }

  bool is_pipeline_finished() const override { return _finished; }

 private:
  bool _finished;
};

sirius_physical_sort_sample make_sort_sample(uint64_t sort_sample_bytes,
                                             uint64_t max_partition_bytes = 0)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::INTEGER));

  return sirius_physical_sort_sample(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    std::move(orders),
    1000,
    sort_sample_bytes,
    max_partition_bytes);
}

void add_int_batches(cucascade::shared_data_repository& repo,
                     memory_space& space,
                     int num_batches,
                     std::size_t rows_per_batch)
{
  for (int b = 0; b < num_batches; ++b) {
    std::vector<int32_t> values(rows_per_batch);
    std::iota(values.begin(), values.end(), static_cast<int32_t>(b * rows_per_batch));
    repo.add_data_batch(make_numeric_batch<int32_t>(space, values, cudf::type_id::INT32), 0);
  }
}

void attach_port(sirius_physical_sort_sample& sample_op,
                 cucascade::shared_data_repository& repo,
                 std::shared_ptr<mock_gpu_pipeline> src_pipeline = nullptr)
{
  auto port           = std::make_unique<sirius_physical_operator::port>();
  port->type          = MemoryBarrierType::PIPELINE;
  port->repo          = &repo;
  port->src_pipeline  = std::move(src_pipeline);
  port->dest_pipeline = nullptr;
  sample_op.add_port("input", std::move(port));
}

std::size_t batch_count(const operator_data& data)
{
  return dynamic_cast<const pipelineable_operator_data&>(data).get_data_batches().size();
}

int32_t read_int32_column(const cudf::column_view& col, cudf::size_type row)
{
  std::vector<int32_t> host(col.size());
  REQUIRE(cudaMemcpy(host.data(),
                     col.data<int32_t>(),
                     host.size() * sizeof(int32_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host[row];
}

}  // namespace

TEST_CASE("sirius_physical_sort_sample stops sampling at sort_sample_bytes threshold",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold         = 6144;
  constexpr int num_batches            = 5;
  constexpr std::size_t rows_per_batch = 1000;

  auto sample_op = make_sort_sample(threshold);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  add_int_batches(*repo, *space, num_batches, rows_per_batch);
  attach_port(sample_op, *repo);

  auto result = sample_op.get_next_task_input_data();
  REQUIRE(result != nullptr);
  REQUIRE(batch_count(*result) < static_cast<std::size_t>(num_batches));
  REQUIRE(batch_count(*result) >= 1);
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);

  (void)sample_op.execute(*result, cudf::get_default_stream());
  REQUIRE(sample_op.boundaries_computed());

  std::size_t total_batches_returned = batch_count(*result);
  while (true) {
    auto next = sample_op.get_next_task_input_data();
    if (!next) { break; }
    total_batches_returned += batch_count(*next);
  }

  REQUIRE(total_batches_returned == static_cast<std::size_t>(num_batches));
}

TEST_CASE("sirius_physical_sort_sample get_next_task_hint waits for sample bytes",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold         = 6144;
  constexpr std::size_t rows_per_batch = 1000;

  auto sample_op = make_sort_sample(threshold);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(false);

  // get_next_task_hint() reports the upstream pipeline's first operator as the producer.
  auto upstream_producer = duckdb::make_uniq<sirius::op::test_physical_operator>(
    SiriusPhysicalOperatorType::ORDER_BY,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000);
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*src_pipeline, *upstream_producer);

  attach_port(sample_op, *repo, src_pipeline);

  add_int_batches(*repo, *space, 1, rows_per_batch);
  auto waiting_hint = sample_op.get_next_task_hint();
  REQUIRE(waiting_hint.has_value());
  REQUIRE(waiting_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(waiting_hint->producer == upstream_producer.get());

  add_int_batches(*repo, *space, 1, rows_per_batch);
  auto ready_hint = sample_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);
  REQUIRE(ready_hint->producer == &sample_op);
}

TEST_CASE("sirius_physical_sort_sample is ready when upstream finished with empty repo",
          "[physical_sort_sample]")
{
  auto sample_op = make_sort_sample(100000);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(true);
  attach_port(sample_op, *repo, src_pipeline);

  // Upstream finished but produced no batches — must not block waiting for sample bytes.
  auto hint = sample_op.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == &sample_op);

  // No batches to pull; execute() should still be able to mark boundaries done.
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);
}

TEST_CASE("sirius_physical_sort_sample get_next_task_hint returns nullopt while scheduled",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold         = 6144;
  constexpr std::size_t rows_per_batch = 1000;

  auto sample_op = make_sort_sample(threshold);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  add_int_batches(*repo, *space, 2, rows_per_batch);
  attach_port(sample_op, *repo);

  REQUIRE(sample_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  REQUIRE(sample_op.get_next_task_input_data() != nullptr);
  REQUIRE_FALSE(sample_op.get_next_task_hint().has_value());
}

TEST_CASE(
  "sirius_physical_sort_sample pulls all remaining data when upstream finished below threshold",
  "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold         = 100000;
  constexpr int num_batches            = 3;
  constexpr std::size_t rows_per_batch = 1000;

  auto sample_op = make_sort_sample(threshold);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  add_int_batches(*repo, *space, num_batches, rows_per_batch);

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(true);
  attach_port(sample_op, *repo, src_pipeline);

  auto ready_hint = sample_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);

  auto result = sample_op.get_next_task_input_data();
  REQUIRE(result != nullptr);
  REQUIRE(batch_count(*result) == static_cast<std::size_t>(num_batches));
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);

  (void)sample_op.execute(*result, cudf::get_default_stream());
  REQUIRE(sample_op.boundaries_computed());
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);
}

TEST_CASE("sirius_physical_sort_sample merges pre-sorted batches for partition boundaries",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr std::size_t rows_per_batch = 50;
  std::vector<int32_t> high_keys(rows_per_batch);
  std::iota(high_keys.begin(), high_keys.end(), 50);
  std::vector<int32_t> low_keys(rows_per_batch);
  std::iota(low_keys.begin(), low_keys.end(), 0);

  auto batch_high = make_numeric_batch<int32_t>(*space, high_keys, cudf::type_id::INT32);
  auto batch_low  = make_numeric_batch<int32_t>(*space, low_keys, cudf::type_id::INT32);

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::INTEGER));

  // Force two partitions from estimated total size vs max_partition_bytes.
  sirius_physical_sort_sample sample_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    std::move(orders),
    1000,
    1'000'000,
    2500,
    sirius::config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  auto output = sample_op.execute(pipelineable_operator_data({batch_high, batch_low}),
                                  cudf::get_default_stream());
  REQUIRE(output != nullptr);
  REQUIRE(sample_op.boundaries_computed());
  REQUIRE(sample_op.get_num_partitions() == 2);
  REQUIRE(sample_op.get_partition_boundaries().num_rows() == 1);

  const auto boundary_key =
    read_int32_column(sample_op.get_partition_boundaries().view().column(0), 0);
  REQUIRE(boundary_key == 50);
}

TEST_CASE("sirius_physical_sort_sample sizes a complete input from its actual bytes",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr std::size_t rows_per_batch = 30;
  std::vector<int32_t> values(rows_per_batch);
  std::iota(values.begin(), values.end(), 0);
  auto batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
  std::size_t batch_bytes;
  {
    auto read_only = batch->to_read_only();
    batch_bytes    = read_only.get_data()->get_size_in_bytes();
  }

  auto sample_op = make_sort_sample(/*sort_sample_bytes=*/batch_bytes,
                                    /*max_partition_bytes=*/batch_bytes - 1);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  repo->add_data_batch(std::move(batch), 0);

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(true);
  attach_port(sample_op, *repo, src_pipeline);

  REQUIRE(sample_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  REQUIRE(repo->all_empty());

  (void)sample_op.execute(*input, cudf::get_default_stream());
  CHECK(sample_op.get_num_partitions() == 2);
  CHECK(sample_op.get_partition_boundaries().num_rows() == 1);
}

TEST_CASE("sirius_physical_sort_sample extrapolates partial input without duplicate boundaries",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr std::size_t rows_per_batch = 30;
  bool upstream_finished               = false;
  int num_batches                      = 1;
  bool expect_repo_empty               = true;
  SECTION("upstream is unfinished")
  {
    upstream_finished = false;
    num_batches       = 1;
    expect_repo_empty = true;
  }
  SECTION("upstream is finished with an unsampled batch")
  {
    upstream_finished = true;
    num_batches       = 2;
    expect_repo_empty = false;
  }

  std::vector<int32_t> values(rows_per_batch);
  std::iota(values.begin(), values.end(), 0);
  auto first_batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
  std::size_t batch_bytes;
  {
    auto read_only = first_batch->to_read_only();
    batch_bytes    = read_only.get_data()->get_size_in_bytes();
  }

  auto sample_op = make_sort_sample(/*sort_sample_bytes=*/batch_bytes,
                                    /*max_partition_bytes=*/batch_bytes - 1);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  repo->add_data_batch(std::move(first_batch), 0);
  if (num_batches > 1) { add_int_batches(*repo, *space, num_batches - 1, rows_per_batch); }

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(upstream_finished);
  attach_port(sample_op, *repo, src_pipeline);

  REQUIRE(sample_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  REQUIRE(repo->all_empty() == expect_repo_empty);

  (void)sample_op.execute(*input, cudf::get_default_stream());
  CHECK(sample_op.get_num_partitions() == rows_per_batch);
  CHECK(sample_op.get_partition_boundaries().num_rows() == rows_per_batch - 1);
}

// ---------------------------------------------------------------------------
// Boundary key prefix (boundary_key_count)
// ---------------------------------------------------------------------------

namespace {

/// Two INT32 columns; @p key_valid[i] == false makes column 0 NULL at row i.
std::shared_ptr<cucascade::data_batch> make_prefix_batch(
  memory_space& space,
  std::vector<int32_t> const& key,
  std::vector<bool> const& key_valid,
  std::vector<int32_t> const& value,
  cudf::type_id key_type = cudf::type_id::INT32)
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();
  auto size   = static_cast<cudf::size_type>(key.size());

  auto null_mask = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    if (!key_valid[i]) {
      cudf::set_null_mask(
        static_cast<cudf::bitmask_type*>(null_mask.data()), i, i + 1, false, stream);
      ++null_count;
    }
  }
  auto key_col = cudf::make_numeric_column(
    cudf::data_type{key_type}, size, std::move(null_mask), null_count, stream, mr);
  if (key_type == cudf::type_id::INT64) {
    std::vector<int64_t> wide(key.begin(), key.end());
    REQUIRE(cudaMemcpy(key_col->mutable_view().data<int64_t>(),
                       wide.data(),
                       sizeof(int64_t) * wide.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  } else {
    REQUIRE(cudaMemcpy(key_col->mutable_view().data<int32_t>(),
                       key.data(),
                       sizeof(int32_t) * key.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  auto value_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, size, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpy(value_col->mutable_view().data<int32_t>(),
                     value.data(),
                     sizeof(int32_t) * value.size(),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(key_col));
  cols.push_back(std::move(value_col));
  return ::sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                   space,
                                   stream,
                                   ::sirius::telemetry::batch_telemetry_info{});
}

duckdb::vector<duckdb::BoundOrderByNode> two_key_orders(duckdb::OrderType prefix_dir,
                                                        duckdb::OrderByNullType prefix_nulls)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::INTEGER, prefix_dir, prefix_nulls));
  orders.push_back(make_order(1, duckdb::LogicalType::INTEGER));
  return orders;
}

auto two_int_types()
{
  return sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                     duckdb::LogicalType::INTEGER});
}

/// Sort @p batch on @p orders with the ORDER_BY operator, as the sort chain does before sampling.
std::shared_ptr<cucascade::data_batch> locally_sorted(
  std::shared_ptr<cucascade::data_batch> batch,
  duckdb::vector<duckdb::BoundOrderByNode> const& orders)
{
  sirius_physical_order order_op(two_int_types(), copy_orders(orders), {0, 1}, 0);
  auto result =
    order_op.execute(pipelineable_operator_data({std::move(batch)}), cudf::get_default_stream());
  auto const& batches = dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches();
  REQUIRE(batches.size() == 1);
  return batches[0];
}

/// A sample operator over two INT32 keys that cuts the input into as many ranges as it can.
sirius_physical_sort_sample make_fine_two_key_sample(
  duckdb::vector<duckdb::BoundOrderByNode> orders, std::size_t estimated_cardinality)
{
  return sirius_physical_sort_sample(two_int_types(),
                                     std::move(orders),
                                     estimated_cardinality,
                                     /*sort_sample_bytes=*/1'000'000,
                                     /*max_partition_bytes=*/1);
}

}  // namespace

TEST_CASE("sirius_physical_sort_sample rejects invalid boundary key counts",
          "[physical_sort_sample]")
{
  auto sample_op = make_fine_two_key_sample(
    two_key_orders(duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST), 10);
  CHECK(sample_op.boundary_key_count() == 2);
  CHECK_THROWS_AS(sample_op.set_boundary_key_count(0), std::invalid_argument);
  CHECK_THROWS_AS(sample_op.set_boundary_key_count(3), std::invalid_argument);
  sample_op.set_boundary_key_count(1);
  CHECK(sample_op.boundary_key_count() == 1);
  sample_op.set_boundary_key_count(2);
  CHECK(sample_op.boundary_key_count() == 2);
}

TEST_CASE("sirius_physical_sort_sample computes distinct boundaries on its boundary keys",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // 100 rows, prefix key i / 10 (10 distinct values), second key i.
  constexpr int rows = 100;
  std::vector<int32_t> key(rows), value(rows);
  for (int i = 0; i < rows; ++i) {
    key[i]   = i / 10;
    value[i] = i;
  }
  auto orders = two_key_orders(duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST);
  auto batch  = make_prefix_batch(*space, key, std::vector<bool>(rows, true), value);

  SECTION("default count uses every key and keeps distinct boundaries")
  {
    auto sample_op = make_fine_two_key_sample(copy_orders(orders), rows);
    (void)sample_op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
    REQUIRE(sample_op.get_num_partitions() == rows);
    auto const& boundaries = sample_op.get_partition_boundaries();
    CHECK(boundaries.num_columns() == 2);
    CHECK(boundaries.num_rows() == rows - 1);
  }

  SECTION("a prefix of one key holds one column and drops repeated values")
  {
    auto sample_op = make_fine_two_key_sample(copy_orders(orders), rows);
    sample_op.set_boundary_key_count(1);
    (void)sample_op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
    auto const& boundaries = sample_op.get_partition_boundaries();
    REQUIRE(boundaries.num_columns() == 1);
    auto host = copy_column_to_host<int32_t>(boundaries.view().column(0));
    // Boundary indices are 1..99, so every prefix value 0..9 appears exactly once.
    CHECK(host == std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    CHECK(sample_op.get_num_partitions() == host.size() + 1);
  }

  SECTION("one prefix value covering most of the sample leaves few boundaries")
  {
    std::vector<int32_t> skewed(rows, 5);
    skewed[rows - 1]  = 9;
    auto skewed_batch = make_prefix_batch(*space, skewed, std::vector<bool>(rows, true), value);
    auto sample_op    = make_fine_two_key_sample(copy_orders(orders), rows);
    sample_op.set_boundary_key_count(1);
    (void)sample_op.execute(pipelineable_operator_data({skewed_batch}), cudf::get_default_stream());
    auto host = copy_column_to_host<int32_t>(sample_op.get_partition_boundaries().view().column(0));
    CHECK(host == std::vector<int32_t>{5, 9});
    CHECK(sample_op.get_num_partitions() == 3);
  }

  SECTION("repeated boundaries are dropped on the full key too")
  {
    // Every row repeats its full key ten times, so the 99 quantiles hold 10 distinct rows.
    std::vector<int32_t> repeated(rows, 0);
    auto repeated_batch = make_prefix_batch(*space, key, std::vector<bool>(rows, true), repeated);
    auto sample_op      = make_fine_two_key_sample(copy_orders(orders), rows);
    (void)sample_op.execute(pipelineable_operator_data({repeated_batch}),
                            cudf::get_default_stream());
    auto const& boundaries = sample_op.get_partition_boundaries();
    REQUIRE(boundaries.num_columns() == 2);
    CHECK(copy_column_to_host<int32_t>(boundaries.view().column(0)) ==
          std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    CHECK(sample_op.get_num_partitions() == 11);
  }
}

TEST_CASE("sirius_physical_sort_partition never splits tied rows on the full key",
          "[physical_sort_sample][physical_sort_partition]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  // One INT32 key with five distinct values over three sorted batches, cut on the full key.
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::INTEGER));
  auto types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int b = 0; b < 3; ++b) {
    std::vector<int32_t> values(100);
    for (int i = 0; i < 100; ++i) {
      values[i] = (b * 100 + i) % 5;
    }
    std::sort(values.begin(), values.end());
    batches.push_back(make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32));
  }
  sirius_physical_sort_sample sample_op(types, copy_orders(orders), 300, 1'000'000, 1);
  (void)sample_op.execute(pipelineable_operator_data(batches), cudf::get_default_stream());
  CHECK(sample_op.boundary_key_count() == 1);
  // Boundaries 0..4: range 0 is empty, since the first boundary is the smallest value.
  CHECK(sample_op.get_num_partitions() == 6);

  sirius_physical_sort_partition partition_op(types, copy_orders(orders), {0}, 0);
  partition_op.set_sample_op(&sample_op);
  std::map<int32_t, std::set<std::size_t>> ranges_of_value;
  for (auto const& batch : batches) {
    auto output =
      partition_op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
    auto const& slices = dynamic_cast<const sort_partition_output&>(*output);
    for (std::size_t i = 0; i < slices.get_data_batches().size(); ++i) {
      auto table = sirius::get_cudf_table_view(*slices.get_data_batches()[i]);
      for (auto value : copy_column_to_host<int32_t>(table.column(0))) {
        ranges_of_value[value].insert(slices.partitions[i]);
      }
    }
  }
  REQUIRE(ranges_of_value.size() == 5);
  for (auto const& [value, ranges] : ranges_of_value) {
    INFO("value " << value);
    CHECK(ranges.size() == 1);
  }
}

TEST_CASE("sirius_physical_sort_partition never splits rows tied on a key prefix",
          "[physical_sort_sample][physical_sort_partition]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto prefix_dir   = duckdb::OrderType::ASCENDING;
  auto prefix_nulls = duckdb::OrderByNullType::NULLS_FIRST;
  SECTION("ascending, nulls first") {}
  SECTION("ascending, nulls last") { prefix_nulls = duckdb::OrderByNullType::NULLS_LAST; }
  SECTION("descending, nulls first") { prefix_dir = duckdb::OrderType::DESCENDING; }
  SECTION("descending, nulls last")
  {
    prefix_dir   = duckdb::OrderType::DESCENDING;
    prefix_nulls = duckdb::OrderByNullType::NULLS_LAST;
  }
  auto orders = two_key_orders(prefix_dir, prefix_nulls);

  // Three batches, prefix key i % 7 with NULLs at i % 11 == 0, second key unique.
  constexpr int num_batches = 3;
  constexpr int rows        = 200;
  std::vector<std::shared_ptr<cucascade::data_batch>> sorted;
  for (int b = 0; b < num_batches; ++b) {
    std::vector<int32_t> key(rows), value(rows);
    std::vector<bool> valid(rows);
    for (int i = 0; i < rows; ++i) {
      int const global = b * rows + i;
      key[i]           = global % 7;
      valid[i]         = global % 11 != 0;
      value[i]         = global;
    }
    sorted.push_back(locally_sorted(make_prefix_batch(*space, key, valid, value), orders));
  }

  auto sample_op = make_fine_two_key_sample(copy_orders(orders), num_batches * rows);
  sample_op.set_boundary_key_count(1);
  (void)sample_op.execute(pipelineable_operator_data(sorted), cudf::get_default_stream());
  REQUIRE(sample_op.get_num_partitions() > 1);
  // Eight distinct prefix values including NULL.
  CHECK(sample_op.get_num_partitions() <= 9);

  sirius_physical_sort_partition partition_op(two_int_types(), copy_orders(orders), {0, 1}, 0);
  partition_op.set_sample_op(&sample_op);

  std::map<std::optional<int32_t>, std::set<std::size_t>> ranges_of_prefix;
  std::size_t total_rows = 0;
  for (auto const& batch : sorted) {
    auto output =
      partition_op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
    auto const& slices      = dynamic_cast<const sort_partition_output&>(*output);
    auto const& out_batches = slices.get_data_batches();
    REQUIRE(out_batches.size() == slices.partitions.size());
    for (std::size_t i = 0; i < out_batches.size(); ++i) {
      REQUIRE(slices.partitions[i] < sample_op.get_num_partitions());
      auto table = sirius::get_cudf_table_view(*out_batches[i]);
      auto keys  = copy_column_to_host<int32_t>(table.column(0));
      auto valid = table.column(0).nullable() ? copy_validity_to_host(table.column(0))
                                              : std::vector<bool>(keys.size(), true);
      total_rows += keys.size();
      for (std::size_t r = 0; r < keys.size(); ++r) {
        std::optional<int32_t> prefix = valid[r] ? std::optional<int32_t>{keys[r]} : std::nullopt;
        ranges_of_prefix[prefix].insert(slices.partitions[i]);
      }
    }
  }
  CHECK(total_rows == num_batches * rows);
  CHECK(ranges_of_prefix.size() == 8);
  CHECK(ranges_of_prefix.count(std::nullopt) == 1);
  for (auto const& [prefix, ranges] : ranges_of_prefix) {
    INFO("prefix " << (prefix ? std::to_string(*prefix) : std::string{"NULL"}));
    CHECK(ranges.size() == 1);
  }
}

TEST_CASE("sirius_physical_sort_partition rejects a key type that differs from the boundaries",
          "[physical_sort_sample][physical_sort_partition]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr int rows = 20;
  std::vector<int32_t> key(rows), value(rows);
  std::iota(key.begin(), key.end(), 0);
  std::iota(value.begin(), value.end(), 0);
  std::vector<bool> valid(rows, true);
  auto orders = two_key_orders(duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST);

  auto sample_op = make_fine_two_key_sample(copy_orders(orders), rows);
  sample_op.set_boundary_key_count(1);
  (void)sample_op.execute(
    pipelineable_operator_data({make_prefix_batch(*space, key, valid, value)}),
    cudf::get_default_stream());
  REQUIRE(sample_op.get_num_partitions() > 1);

  sirius_physical_sort_partition partition_op(two_int_types(), copy_orders(orders), {0, 1}, 0);
  partition_op.set_sample_op(&sample_op);
  auto wide = make_prefix_batch(*space, key, valid, value, cudf::type_id::INT64);
  CHECK_THROWS_AS(
    partition_op.execute(pipelineable_operator_data({wide}), cudf::get_default_stream()),
    sirius::internal_exception);
}
