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

#include "helper/cuda_launch_error.hpp"
#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"
#include "sort_range_test_utils.hpp"
#include "utils/scan_resource_test_utils.hpp"
#include "utils/test_physical_operator.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_sort_sample.hpp>
#include <op/sort_range_count.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_config.hpp>

#include <algorithm>
#include <numeric>
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

/// As pipeline 1 does: the sorted batch gives its key rows, then goes to the input port.
void feed(sirius_physical_sort_sample& sample_op,
          cucascade::shared_data_repository& repo,
          std::shared_ptr<cucascade::data_batch> batch)
{
  sirius::test::sort_range::keep_key_rows(sample_op, {batch});
  repo.add_data_batch(std::move(batch), 0);
}

void add_int_batches(sirius_physical_sort_sample& sample_op,
                     cucascade::shared_data_repository& repo,
                     memory_space& space,
                     int num_batches,
                     std::size_t rows_per_batch)
{
  for (int b = 0; b < num_batches; ++b) {
    std::vector<int32_t> values(rows_per_batch);
    std::iota(values.begin(), values.end(), static_cast<int32_t>(b * rows_per_batch));
    feed(sample_op, repo, make_numeric_batch<int32_t>(space, values, cudf::type_id::INT32));
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

/// Records each request and returns one partition.
class recording_boundaries final : public cross_rank_sort_boundaries {
 public:
  sort_boundaries make(request const& request, ::cuda::stream_ref) override
  {
    key_columns.push_back(request.sample_keys.num_columns());
    num_parts.push_back(request.num_parts);
    return {};
  }

  std::vector<cudf::size_type> key_columns;
  std::vector<std::size_t> num_parts;
};

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

TEST_CASE("sirius_physical_sort_sample starts the boundary task once sort_sample_bytes arrived",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold         = 6144;
  constexpr int num_batches            = 5;
  constexpr std::size_t rows_per_batch = 1000;

  auto sample_op = make_sort_sample(threshold);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  attach_port(sample_op, *repo);

  // One batch is below the setting.
  add_int_batches(sample_op, *repo, *space, 1, rows_per_batch);
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);

  // The boundary task carries one batch; its sample is every batch handed over.
  add_int_batches(sample_op, *repo, *space, 1, rows_per_batch);
  auto result = sample_op.get_next_task_input_data();
  REQUIRE(result != nullptr);
  REQUIRE(batch_count(*result) == 1);
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);

  (void)sample_op.execute(*result, cudf::get_default_stream());
  REQUIRE(sample_op.boundaries_computed());
  CHECK(sample_op.get_range_sample_stats().sample_batches == 2);
  CHECK(sample_op.get_range_sample_stats().sample_rows == 2 * rows_per_batch);
  CHECK_FALSE(sample_op.get_range_sample_stats().complete_input);

  // Later batches are not sampled and pass through one per task.
  add_int_batches(sample_op, *repo, *space, num_batches - 2, rows_per_batch);
  CHECK(sample_op.sampled_input().batches == 2);
  std::size_t total_batches_returned = batch_count(*result);
  while (true) {
    auto next = sample_op.get_next_task_input_data();
    if (!next) { break; }
    CHECK(batch_count(*next) == 1);
    total_batches_returned += batch_count(*next);
  }

  REQUIRE(total_batches_returned == static_cast<std::size_t>(num_batches));
}

TEST_CASE("sirius_physical_sort_sample counts a batch handed over twice once",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto const stream = cudf::get_default_stream();

  constexpr std::size_t rows = 1000;
  std::vector<int32_t> values(rows);
  std::iota(values.begin(), values.end(), 0);
  auto batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
  std::size_t batch_bytes;
  {
    auto read_only = batch->to_read_only();
    batch_bytes    = read_only.get_data()->get_size_in_bytes();
  }

  auto sample_op = make_sort_sample(/*sort_sample_bytes=*/2 * batch_bytes);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  attach_port(sample_op, *repo);

  // A retried task hands its batch over again.
  sirius::test::sort_range::keep_key_rows(sample_op, {batch}, stream);
  sirius::test::sort_range::keep_key_rows(sample_op, {batch}, stream);
  CHECK(sample_op.sampled_input().batches == 1);
  CHECK(sample_op.sampled_input().rows == rows);
  CHECK(sample_op.sampled_input().bytes == batch_bytes);

  // Key rows alone do not start the boundary task: a batch has to wait in the port.
  auto second = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
  sirius::test::sort_range::keep_key_rows(sample_op, {second}, stream);
  CHECK(sample_op.sampled_input().batches == 2);
  CHECK(sample_op.get_next_task_input_data() == nullptr);

  repo->add_data_batch(batch, 0);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  (void)sample_op.execute(*input, stream);
  CHECK(sample_op.get_range_sample_stats().sample_batches == 2);
  CHECK(sample_op.get_range_sample_stats().sample_rows == 2 * rows);
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

  add_int_batches(sample_op, *repo, *space, 1, rows_per_batch);
  auto waiting_hint = sample_op.get_next_task_hint();
  REQUIRE(waiting_hint.has_value());
  REQUIRE(waiting_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(waiting_hint->producer == upstream_producer.get());

  add_int_batches(sample_op, *repo, *space, 1, rows_per_batch);
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
  src_pipeline->set_finished(false);
  attach_port(sample_op, *repo, src_pipeline);

  SECTION("not distributed")
  {
    src_pipeline->set_finished(true);
    // Upstream finished but produced no batches — must not block waiting for sample bytes.
    auto hint = sample_op.get_next_task_hint();
    REQUIRE(hint.has_value());
    REQUIRE(hint->hint == TaskCreationHint::READY);
    REQUIRE(hint->producer == &sample_op);

    // No batches to pull; execute() should still be able to mark boundaries done.
    REQUIRE(sample_op.get_next_task_input_data() == nullptr);
    CHECK_FALSE(sample_op.runs_without_input());
  }

  SECTION("with a cross-rank hook: a boundary task with no input, which asks the hook")
  {
    auto hook = std::make_shared<recording_boundaries>();
    sample_op.set_cross_rank_boundaries(hook);
    CHECK(sample_op.cross_rank_boundaries() == hook.get());

    // The boundary task is owed before upstream finishes, but cannot run yet.
    CHECK_FALSE(sample_op.all_ports_empty());
    CHECK(sample_op.get_next_task_input_data() == nullptr);

    src_pipeline->set_finished(true);
    auto hint = sample_op.get_next_task_hint();
    REQUIRE(hint.has_value());
    CHECK(hint->hint == TaskCreationHint::READY);

    auto input = sample_op.get_next_task_input_data();
    REQUIRE(input != nullptr);
    CHECK(batch_count(*input) == 0);
    CHECK(sample_op.runs_without_input());
    CHECK(sample_op.get_next_task_input_data() == nullptr);

    // The boundary task is claimed: the hook can no longer change.
    CHECK_THROWS_AS(sample_op.set_cross_rank_boundaries(std::make_shared<recording_boundaries>()),
                    std::logic_error);
    CHECK_THROWS_AS(sample_op.set_cross_rank_boundaries(nullptr), std::logic_error);
    CHECK(sample_op.cross_rank_boundaries() == hook.get());

    (void)sample_op.execute(*input, cudf::get_default_stream());
    CHECK(sample_op.boundaries_computed());
    CHECK(sample_op.get_num_partitions() == 1);
    CHECK_FALSE(sample_op.runs_without_input());
    CHECK_THROWS_AS(sample_op.set_cross_rank_boundaries(nullptr), std::logic_error);
    CHECK(sample_op.cross_rank_boundaries() == hook.get());
    REQUIRE(hook->key_columns.size() == 1);
    CHECK(hook->key_columns[0] == 0);
    CHECK(hook->num_parts[0] == 1);
  }
}

TEST_CASE("sirius_physical_sort_sample with no input ends with one partition, whatever its keys",
          "[physical_sort_sample]")
{
  // A key that is not a column reference fails only once there is a sample to cut.
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(
    duckdb::OrderType::ASCENDING,
    duckdb::OrderByNullType::NULLS_LAST,
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1)));
  sirius_physical_sort_sample sample_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    std::move(orders),
    1000);
  pipelineable_operator_data empty{std::vector<std::shared_ptr<cucascade::data_batch>>{}};
  REQUIRE_NOTHROW(sample_op.execute(empty, cudf::get_default_stream()));
  CHECK(sample_op.boundaries_computed());
  CHECK(sample_op.get_num_partitions() == 1);
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
  add_int_batches(sample_op, *repo, *space, 2, rows_per_batch);
  attach_port(sample_op, *repo);

  REQUIRE(sample_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  REQUIRE(sample_op.get_next_task_input_data() != nullptr);
  REQUIRE_FALSE(sample_op.get_next_task_hint().has_value());
}

TEST_CASE("sirius_physical_sort_sample samples all input when upstream finished below threshold",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr uint64_t threshold         = 100000;
  constexpr int num_batches            = 3;
  constexpr std::size_t rows_per_batch = 1000;

  auto sample_op = make_sort_sample(threshold);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  add_int_batches(sample_op, *repo, *space, num_batches, rows_per_batch);

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  attach_port(sample_op, *repo, src_pipeline);
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);

  src_pipeline->set_finished(true);
  auto ready_hint = sample_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  REQUIRE(ready_hint->hint == TaskCreationHint::READY);

  auto result = sample_op.get_next_task_input_data();
  REQUIRE(result != nullptr);
  REQUIRE(batch_count(*result) == 1);
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);

  (void)sample_op.execute(*result, cudf::get_default_stream());
  REQUIRE(sample_op.boundaries_computed());
  auto const& stats = sample_op.get_range_sample_stats();
  CHECK(stats.complete_input);
  CHECK(stats.est_source == "measured");
  CHECK(stats.sample_batches == static_cast<std::size_t>(num_batches));
  CHECK(stats.sample_rows == num_batches * rows_per_batch);

  // The batches the boundary task left behind pass through.
  for (int b = 1; b < num_batches; ++b) {
    auto next = sample_op.get_next_task_input_data();
    REQUIRE(next != nullptr);
    CHECK(batch_count(*next) == 1);
  }
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);
}

TEST_CASE("sirius_physical_sort_sample waits for all its input when sort_sample_bytes is 0",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto sample_op = make_sort_sample(/*sort_sample_bytes=*/0);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline      = std::make_shared<mock_gpu_pipeline>(build_ctx);
  auto upstream_producer = duckdb::make_uniq<sirius::op::test_physical_operator>(
    SiriusPhysicalOperatorType::ORDER_BY,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    1000);
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.add_pipeline_operator(*src_pipeline, *upstream_producer);
  attach_port(sample_op, *repo, src_pipeline);

  // However much input was handed over, the boundary task waits for the last of it.
  constexpr int num_batches = 3;
  add_int_batches(sample_op, *repo, *space, num_batches, 1000);
  auto waiting_hint = sample_op.get_next_task_hint();
  REQUIRE(waiting_hint.has_value());
  CHECK(waiting_hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK(sample_op.get_next_task_input_data() == nullptr);

  src_pipeline->set_finished(true);
  auto ready_hint = sample_op.get_next_task_hint();
  REQUIRE(ready_hint.has_value());
  CHECK(ready_hint->hint == TaskCreationHint::READY);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  (void)sample_op.execute(*input, cudf::get_default_stream());
  CHECK(sample_op.get_range_sample_stats().complete_input);
  CHECK(sample_op.get_range_sample_stats().sample_batches == static_cast<std::size_t>(num_batches));
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

  auto const stream = cudf::get_default_stream();
  sirius::test::sort_range::keep_key_rows(sample_op, {batch_high, batch_low}, stream);
  auto output = sample_op.execute(pipelineable_operator_data({batch_high, batch_low}), stream);
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
  feed(sample_op, *repo, std::move(batch));

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

TEST_CASE("sirius_physical_sort_sample sizes partial input without duplicate boundaries",
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
  SECTION("upstream is finished with a batch still waiting")
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
  feed(sample_op, *repo, std::move(first_batch));
  if (num_batches > 1) {
    add_int_batches(sample_op, *repo, *space, num_batches - 1, rows_per_batch);
  }

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(upstream_finished);
  attach_port(sample_op, *repo, src_pipeline);

  REQUIRE(sample_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  REQUIRE(repo->all_empty() == expect_repo_empty);

  (void)sample_op.execute(*input, cudf::get_default_stream());
  auto const expected_ranges = upstream_finished ? 3 : rows_per_batch;
  CHECK(sample_op.get_num_partitions() == expected_ranges);
  CHECK(sample_op.get_partition_boundaries().num_rows() == expected_ranges - 1);
}

TEST_CASE("sirius_physical_sort_sample scales planner bytes by sampled rows",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  constexpr std::size_t rows_per_batch = 30;
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::INTEGER));
  auto sample_op = sirius_physical_sort_sample(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    std::move(orders),
    /*estimated_cardinality=*/75,
    /*sort_sample_bytes=*/rows_per_batch * sizeof(int32_t),
    /*max_partition_bytes=*/100);
  auto repo = std::make_unique<cucascade::shared_data_repository>();
  add_int_batches(sample_op, *repo, *space, 1, rows_per_batch);

  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(false);
  attach_port(sample_op, *repo, src_pipeline);

  REQUIRE(sample_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  (void)sample_op.execute(*input, cudf::get_default_stream());
  CHECK(sample_op.get_num_partitions() == 3);
  CHECK(sample_op.get_range_sample_stats().est_source == "planner");
}

TEST_CASE("sirius_physical_sort_sample records key-row stride", "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  bool const capped = GENERATE(false, true);
  auto sample_op    = make_sort_sample(1000, 30000);
  if (capped) { sample_op.set_max_sample_key_rows(4096); }
  auto repo = std::make_unique<cucascade::shared_data_repository>();
  attach_port(sample_op, *repo);
  add_int_batches(sample_op, *repo, *space, 3, 20000);
  // The cap is fixed once a batch was handed over.
  CHECK_THROWS_AS(sample_op.set_max_sample_key_rows(8192), std::logic_error);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  (void)sample_op.execute(*input, cudf::get_default_stream());
  auto const& stats = sample_op.get_range_sample_stats();
  CHECK(stats.sample_rows == 60000);
  // Every row fits the default cap; 4096 rows leave room for every 16th row of each batch.
  CHECK(stats.sample_stride == (capped ? std::size_t{16} : std::size_t{1}));
  CHECK(stats.sample_key_rows == (capped ? std::size_t{3750} : std::size_t{60000}));
}

TEST_CASE("sirius_physical_sort_sample picks the same boundaries from all input in any order",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto const stream = cudf::get_default_stream();

  struct run {
    int32_t first;
    std::size_t rows;
  };
  std::vector<run> const runs{{0, 1000}, {10000, 300}, {20000, 2500}, {30000, 3}};
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  std::size_t bytes = 0;
  for (auto const& r : runs) {
    std::vector<int32_t> values(r.rows);
    std::iota(values.begin(), values.end(), r.first);
    batches.push_back(make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32));
    bytes += batches.back()->to_read_only().get_data()->get_size_in_bytes();
  }
  bool const reversed = GENERATE(false, true);
  if (reversed) { std::reverse(batches.begin(), batches.end()); }

  // A quarter of the input per range, from all the input.
  auto sample_op = make_sort_sample(/*sort_sample_bytes=*/0,
                                    /*max_partition_bytes=*/(bytes + 3) / 4);
  cucascade::shared_data_repository repo;
  auto source =
    std::make_shared<mock_gpu_pipeline>(sirius::pipeline::pipeline_build_context{nullptr});
  attach_port(sample_op, repo, source);
  for (auto const& batch : batches) {
    feed(sample_op, repo, batch);
  }
  REQUIRE(sample_op.get_next_task_input_data() == nullptr);

  source->set_finished(true);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  (void)sample_op.execute(*input, stream);

  auto const& stats = sample_op.get_range_sample_stats();
  CHECK(stats.complete_input);
  CHECK(stats.sample_batches == 4);
  CHECK(stats.sample_rows == 3803);
  CHECK(stats.sample_key_rows == 3803);
  CHECK(stats.sample_stride == 1);
  REQUIRE(sample_op.get_num_partitions() == 4);
  CHECK(copy_column_to_host<int32_t>(sample_op.get_partition_boundaries().view().column(0)) ==
        std::vector<int32_t>{950, 20601, 21552});
}

TEST_CASE("Sort range count respects bytes, rows, and sample size", "[physical_sort_sample]")
{
  CHECK(choose_sort_range_count(800, 100, 100, 100) == 8);
  CHECK(choose_sort_range_count(10, 6000000000ULL, 100, 6000000000ULL) == 6);
  CHECK(choose_sort_range_count(10, 6000000000ULL, 0, 6000000000ULL) == 6);
  CHECK(choose_sort_range_count(1000, 100, 100, 3) == 3);
}

TEST_CASE("Sort boundary diagnostic failure preserves completed distributed boundaries",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto const stream      = cudf::get_default_stream();
  bool const empty_input = GENERATE(false, true);
  // What the diagnostic's allocation throws: cudaSuccess stands for rmm::out_of_memory, any
  // other status for a thrust::system_error with that code.
  auto const status = GENERATE(
    cudaSuccess, cudaErrorLaunchOutOfResources, cudaErrorInvalidValue, cudaErrorIllegalAddress);
  // A task retries on out-of-memory and on retryable launch errors.
  bool const task_would_retry = status == cudaSuccess || sirius::is_retryable_launch_error(status);

  // Arm the allocator only after the hook succeeds, so the failure is in the diagnostic,
  // not in sampling or boundary construction. Like the real collective, the hook is one-shot.
  // The upstream is the space's allocator: a reference to the current resource would point
  // into the slot that the guard below replaces.
  sirius::test::tracked_scan_resource resource(space->get_default_allocator(), stream);
  if (status != cudaSuccess) {
    resource.state->failure = std::make_exception_ptr(
      thrust::system_error(status, thrust::cuda_category(), "injected diagnostic failure"));
  }
  class one_shot_boundaries final : public cross_rank_sort_boundaries {
   public:
    one_shot_boundaries(std::unique_ptr<cudf::table> boundaries,
                        std::shared_ptr<sirius::test::tracked_scan_resource::counters> counters)
      : boundaries_(std::move(boundaries)), counters_(std::move(counters))
    {
    }
    sort_boundaries make(request const&, ::cuda::stream_ref) override
    {
      if (++calls != 1) { throw std::logic_error("boundary collective repeated"); }
      counters_->fail_at = counters_->attempts;
      return {3, std::move(boundaries_)};
    }
    std::size_t calls = 0;

   private:
    std::unique_ptr<cudf::table> boundaries_;
    std::shared_ptr<sirius::test::tracked_scan_resource::counters> counters_;
  };

  auto boundary_batch = make_numeric_batch<int32_t>(*space, {10, 20}, cudf::type_id::INT32);
  auto hook           = std::make_shared<one_shot_boundaries>(
    std::make_unique<cudf::table>(
      sirius::get_cudf_table_view(*boundary_batch), stream, space->get_default_allocator()),
    resource.state);
  auto sample = make_sort_sample(1024, 40);
  sample.set_cross_rank_boundaries(hook);
  cucascade::shared_data_repository repo;
  if (!empty_input) { add_int_batches(sample, repo, *space, 1, 30); }
  auto source =
    std::make_shared<mock_gpu_pipeline>(sirius::pipeline::pipeline_build_context{nullptr});
  source->set_finished(true);
  attach_port(sample, repo, source);
  auto input = sample.get_next_task_input_data();
  REQUIRE(input != nullptr);

  // Restore the process-wide resource even if a REQUIRE fails.
  struct resource_guard {
    ::cuda::mr::any_resource<::cuda::mr::device_accessible> previous;
    ~resource_guard() { (void)cudf::set_current_device_resource(std::move(previous)); }
  } guard{cudf::set_current_device_resource(
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource})};

  if (!task_would_retry) {
    // Any other error still propagates.
    REQUIRE_THROWS_AS(sample.execute(*input, stream), thrust::system_error);
    CHECK(resource.state->attempts > resource.state->fail_at);
    CHECK_FALSE(sample.boundaries_computed());
    return;
  }

  std::unique_ptr<operator_data> output;
  REQUIRE_NOTHROW(output = sample.execute(*input, stream));
  CHECK(resource.state->attempts > resource.state->fail_at);
  REQUIRE(sample.boundaries_computed());
  CHECK(sample.get_num_partitions() == 3);
  CHECK_FALSE(sample.get_range_sample_stats().repeated_boundaries.has_value());
  REQUIRE(sample.get_partition_boundaries().num_rows() == 2);
  CHECK(read_int32_column(sample.get_partition_boundaries().view().column(0), 0) == 10);
  CHECK(read_int32_column(sample.get_partition_boundaries().view().column(0), 1) == 20);
  CHECK(batch_count(*output) == (empty_input ? 0 : 1));

  // Completed operators pass through on subsequent execution without invoking the hook again.
  REQUIRE_NOTHROW(sample.execute(*input, stream));
  CHECK(hook->calls == 1);
}
