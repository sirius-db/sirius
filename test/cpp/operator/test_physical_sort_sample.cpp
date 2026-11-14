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

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_sort_sample.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_config.hpp>

#include <numeric>
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
  REQUIRE(sample_op.num_boundary_rows() == 1);

  const auto boundary_key = read_int32_column(
    sample_op.get_partition_boundaries(*space, cudf::get_default_stream()).column(0), 0);
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
  CHECK(sample_op.num_boundary_rows() == 1);
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
  CHECK(sample_op.num_boundary_rows() == rows_per_batch - 1);
}

// A reader on another stream must see complete boundaries as soon as they are published, even
// while the boundary task's own stream still has work queued.
TEST_CASE("sirius_physical_sort_sample publishes boundaries that are complete on every stream",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  // The streams must outlive everything allocated on them.
  rmm::cuda_stream task_stream;
  rmm::cuda_stream reader_stream;

  constexpr std::size_t rows_per_batch = 3000;
  std::vector<int32_t> values(rows_per_batch);
  std::iota(values.begin(), values.end(), 1000);
  auto batch = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
  std::size_t batch_bytes;
  {
    auto read_only = batch->to_read_only();
    batch_bytes    = read_only.get_data()->get_size_in_bytes();
  }

  // Three partitions over the complete input: boundaries are rows 1000 and 2000.
  auto sample_op = make_sort_sample(/*sort_sample_bytes=*/batch_bytes,
                                    /*max_partition_bytes=*/batch_bytes / 3 + 1);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  repo->add_data_batch(std::move(batch), 0);
  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(true);
  attach_port(sample_op, *repo, src_pipeline);
  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);

  // Give the task's stream a backlog, as the merge of the sample batches does in a real query:
  // work queued by execute() then runs only after it.
  rmm::device_buffer scratch(std::size_t{256} << 20, task_stream.view());
  for (int i = 0; i < 40; ++i) {
    REQUIRE(cudaMemsetAsync(scratch.data(), i, scratch.size(), task_stream.value()) == cudaSuccess);
  }

  (void)sample_op.execute(*input, task_stream.view());
  REQUIRE(sample_op.boundaries_computed());
  REQUIRE(sample_op.get_num_partitions() == 3);
  REQUIRE(sample_op.num_boundary_rows() == 2);

  // Read the boundaries the way another task does: at once, on a different stream.
  auto boundaries = sample_op.get_partition_boundaries(*space, reader_stream.view());
  REQUIRE(boundaries.num_columns() == 1);
  REQUIRE(boundaries.num_rows() == 2);
  std::vector<int32_t> host(2, -1);
  REQUIRE(cudaMemcpyAsync(host.data(),
                          boundaries.column(0).data<int32_t>(),
                          host.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          reader_stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);
  CHECK(host[0] == 2000);
  CHECK(host[1] == 3000);

  // A second request on the same GPU returns the same device copy.
  auto again = sample_op.get_partition_boundaries(*space, reader_stream.view());
  CHECK(again.column(0).data<int32_t>() == boundaries.column(0).data<int32_t>());

  REQUIRE(cudaStreamSynchronize(task_stream.value()) == cudaSuccess);
  sample_op.clear_partition_boundaries();
  CHECK(sample_op.num_boundary_rows() == 0);
}

namespace {

constexpr std::size_t kGiB = std::size_t{1} << 30;

sort_partition_sizing_input sizing_input(std::size_t data_gib,
                                         std::size_t limit_gib,
                                         std::size_t concurrency,
                                         std::size_t gpus = 1)
{
  sort_partition_sizing_input in;
  in.data_bytes               = data_gib * kGiB;
  in.data_rows                = in.data_bytes / 16;
  in.avg_batch_bytes          = 128 * (std::size_t{1} << 20);
  in.gpu_capacity_bytes       = limit_gib * kGiB;
  in.gpu_available_bytes      = limit_gib * kGiB;
  in.memory_fraction          = 0.33;
  in.policy.merge_concurrency = concurrency;
  in.policy.num_gpus          = gpus;
  in.policy.windowed_only     = true;
  return in;
}

}  // namespace

TEST_CASE("sort partition sizing: an explicit partition size decides alone",
          "[physical_sort_sample][sort_partition_sizing]")
{
  auto in                         = sizing_input(32, 80, 4);
  in.max_partition_bytes_override = 4 * kGiB;
  auto sizing                     = compute_sort_partition_sizing(in);
  CHECK(sizing.num_partitions == 8);
  CHECK(sizing.max_partition_bytes == 4 * kGiB);
  CHECK(std::string(sizing.rule) == "max_sort_partition_bytes");

  // Also when it asks for fewer partitions than the automatic rule would give.
  in.max_partition_bytes_override = 64 * kGiB;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
  // And with auto sizing off.
  in.policy.auto_sizing           = false;
  in.max_partition_bytes_override = 10 * kGiB;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 4);
}

TEST_CASE("sort partition sizing: the previous rule uses the memory free at the moment",
          "[physical_sort_sample][sort_partition_sizing]")
{
  auto in               = sizing_input(32, 80, 4);
  in.policy.auto_sizing = false;
  // 0.33 x 80 GiB = 26.4 GiB per partition.
  CHECK(compute_sort_partition_sizing(in).num_partitions == 2);
  CHECK(std::string(compute_sort_partition_sizing(in).rule) == "fraction_of_free");
  // Half of the GPU is in use: the same data gets more partitions.
  in.gpu_available_bytes = 40 * kGiB;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 3);
  // Nothing free: no bound, one partition (unchanged behaviour).
  in.gpu_available_bytes = 0;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
  // The concurrency and row terms do not apply.
  in                    = sizing_input(160, 160, 16);
  in.policy.auto_sizing = false;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 4);
}

TEST_CASE("sort partition sizing: the memory term uses the GPU limit, not the free memory",
          "[physical_sort_sample][sort_partition_sizing]")
{
  // 0.33 x 160 GiB = 52.8 GiB per partition.
  auto in = sizing_input(64, 160, 0);
  CHECK(compute_sort_partition_sizing(in).num_partitions == 2);
  CHECK(std::string(compute_sort_partition_sizing(in).rule) == "fraction_of_limit");
  // The count does not move with the memory in use when the boundaries are computed.
  in.gpu_available_bytes = 30 * kGiB;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 2);
  // The fraction setting still scales it.
  in.memory_fraction = 0.125;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 4);
  // Unknown limit: fall back to the free memory.
  in                    = sizing_input(64, 160, 0);
  in.gpu_capacity_bytes = 0;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 2);
}

TEST_CASE("sort partition sizing: smaller partitions when the sort does not fit the GPUs",
          "[physical_sort_sample][sort_partition_sizing]")
{
  // The data and its sorted copy fit an 80 GiB GPU up to 40 GiB.
  CHECK(compute_sort_partition_sizing(sizing_input(40, 80, 0)).num_partitions == 2);
  auto sizing = compute_sort_partition_sizing(sizing_input(41, 80, 0));
  CHECK(sizing.num_partitions == 6);  // ceil(41 / (0.10 x 80))
  CHECK(std::string(sizing.rule) == "larger_than_gpu");
  CHECK(compute_sort_partition_sizing(sizing_input(160, 80, 0)).num_partitions == 20);
  // The free memory of the moment plays no part.
  auto in                = sizing_input(160, 80, 0);
  in.gpu_available_bytes = 20 * kGiB;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 20);
  // Two GPUs hold twice as much before partitions shrink.
  CHECK(compute_sort_partition_sizing(sizing_input(80, 80, 0, 2)).num_partitions == 4);
  CHECK(compute_sort_partition_sizing(sizing_input(81, 80, 0, 2)).num_partitions == 11);
  // A fraction setting below kSortPartitionSpillFraction is kept.
  in                 = sizing_input(160, 80, 0);
  in.memory_fraction = 0.05;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 40);
  CHECK(std::string(compute_sort_partition_sizing(in).rule) == "fraction_of_limit");
}

TEST_CASE("sort partition sizing: one partition per merge thread on several GPUs",
          "[physical_sort_sample][sort_partition_sizing]")
{
  // 32 GiB fits one partition of a 160 GiB GPU; four executor threads get four partitions.
  auto sizing = compute_sort_partition_sizing(sizing_input(32, 160, 4, 2));
  CHECK(sizing.num_partitions == 4);
  CHECK(std::string(sizing.rule) == "merge_concurrency");
  CHECK(compute_sort_partition_sizing(sizing_input(32, 160, 4, 4)).num_partitions == 4);
  // Not on one GPU.
  CHECK(compute_sort_partition_sizing(sizing_input(32, 160, 4)).num_partitions == 1);
  CHECK(compute_sort_partition_sizing(sizing_input(32, 80, 4)).num_partitions == 2);
  // More threads than kSortPartitionMaxConcurrency do not raise it further; fewer lower it.
  CHECK(compute_sort_partition_sizing(sizing_input(64, 160, 16, 2)).num_partitions ==
        kSortPartitionMaxConcurrency);
  CHECK(compute_sort_partition_sizing(sizing_input(32, 160, 2, 2)).num_partitions == 2);
  // Small sorts are not split: no partition below kSortPartitionMinConcurrencyBytes.
  CHECK(compute_sort_partition_sizing(sizing_input(1, 80, 4, 2)).num_partitions == 1);
  CHECK(compute_sort_partition_sizing(sizing_input(3, 80, 4, 2)).num_partitions == 1);
  CHECK(compute_sort_partition_sizing(sizing_input(4, 80, 4, 2)).num_partitions == 2);
  CHECK(compute_sort_partition_sizing(sizing_input(8, 80, 4, 2)).num_partitions == 4);
  // Small input batches are not cut into pieces below kSortPartitionMinPieceBytes.
  auto in            = sizing_input(32, 80, 4, 2);
  in.avg_batch_bytes = 3 * kSortPartitionMinPieceBytes;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 3);
  in.avg_batch_bytes = kSortPartitionMinPieceBytes / 2;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 2);  // the memory term
  // Only when every partition takes the windowed merge.
  in                      = sizing_input(32, 160, 4, 2);
  in.policy.windowed_only = false;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
  // A single thread, or an unknown width, adds nothing to the memory term.
  CHECK(compute_sort_partition_sizing(sizing_input(16, 80, 1, 2)).num_partitions == 1);
  CHECK(compute_sort_partition_sizing(sizing_input(16, 80, 0, 2)).num_partitions == 1);
  // The memory term still wins when it asks for more.
  CHECK(compute_sort_partition_sizing(sizing_input(160, 160, 4, 2)).num_partitions == 4);
  in                 = sizing_input(160, 160, 4, 2);
  in.memory_fraction = 0.1;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 10);
}

TEST_CASE(
  "sort partition sizing: auto mode keeps partitions that cannot be merged whole together "
  "above the one-shot threshold",
  "[physical_sort_sample][sort_partition_sizing]")
{
  // An 80 GiB GPU whose 'auto' route merges partitions of up to 13.2 GiB whole.
  auto const auto_input = [](std::size_t data_gib) {
    auto in                                 = sizing_input(data_gib, 80, 4);
    in.policy.windowed_only                 = false;
    in.policy.auto_one_shot_partition_bytes = static_cast<std::size_t>(13.2 * kGiB);
    return in;
  };

  // The memory term alone leaves partitions above the threshold.
  auto in     = auto_input(32);
  auto sizing = compute_sort_partition_sizing(in);
  CHECK(sizing.num_partitions == 2);
  CHECK(std::string(sizing.rule) == "fraction_of_limit");

  // Four 8 GiB partitions would be merged whole, four at a time: 4 x 3.5 x 8 GiB does not fit.
  in.memory_fraction = 0.125;
  sizing             = compute_sort_partition_sizing(in);
  CHECK(sizing.num_partitions == 2);
  CHECK(std::string(sizing.rule) == "windowed_route");
  CHECK(static_cast<double>(in.data_bytes / sizing.num_partitions) >=
        kSortPartitionWindowedRouteMargin *
          static_cast<double>(in.policy.auto_one_shot_partition_bytes));

  // Partitions whose one-shot merges fit together are kept.
  in                 = auto_input(16);
  in.memory_fraction = 0.05;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 4);

  // A sort that does not fit the GPU: as many partitions as stay clear of the threshold.
  CHECK(compute_sort_partition_sizing(auto_input(64)).num_partitions == 4);
  sizing = compute_sort_partition_sizing(auto_input(160));
  CHECK(sizing.num_partitions == 10);
  CHECK(std::string(sizing.rule) == "windowed_route");

  // Other modes, and an unknown threshold, are not affected.
  in                      = auto_input(160);
  in.policy.windowed_only = true;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 20);
  in                                      = auto_input(160);
  in.policy.auto_one_shot_partition_bytes = 0;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 20);
}

TEST_CASE("sort partition sizing: one-shot partitions stay under the cuDF row limit",
          "[physical_sort_sample][sort_partition_sizing]")
{
  // 5 billion 4-byte rows are 18.6 GiB: one partition by memory under a 160 GiB limit.
  auto in                 = sizing_input(1, 160, 4);
  in.data_rows            = 5'000'000'000;
  in.data_bytes           = in.data_rows * 4;
  in.policy.windowed_only = false;

  SECTION("auto mode, partition small enough for the one-shot merge")
  {
    in.policy.auto_one_shot_partition_bytes = 26 * kGiB;
    auto sizing                             = compute_sort_partition_sizing(in);
    CHECK(sizing.num_partitions == 4);  // ceil(5e9 / kSortPartitionMaxRows)
    CHECK(std::string(sizing.rule) == "row_limit");
    CHECK(in.data_rows / sizing.num_partitions < (std::size_t{1} << 31) - 1);
    // At or below the bound: no effect.
    in.data_rows  = kSortPartitionMaxRows;
    in.data_bytes = in.data_rows * 4;
    CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
    // Unknown row count: no effect.
    in.data_rows = 0;
    CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
  }
  SECTION("auto mode, partition merged in windows: left whole")
  {
    // Cutting it would move its pieces to the one-shot merge.
    in.policy.auto_one_shot_partition_bytes = 10 * kGiB;
    CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
    // Unknown threshold: left whole as well.
    in.policy.auto_one_shot_partition_bytes = 0;
    CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
  }
  SECTION("one_shot mode")
  {
    in.policy.one_shot_only = true;
    auto sizing             = compute_sort_partition_sizing(in);
    CHECK(sizing.num_partitions == 4);
    CHECK(std::string(sizing.rule) == "row_limit");
  }
  SECTION("windowed mode: no row limit")
  {
    in.policy.windowed_only = true;
    CHECK(compute_sort_partition_sizing(in).num_partitions == 1);
  }
}

TEST_CASE("sort partition sizing: one-shot partitions fit the GPU with their merge temporaries",
          "[physical_sort_sample][sort_partition_sizing]")
{
  // 36 GiB of 16-byte rows under 80 GiB: the fraction alone gives 2 partitions of 18 GiB, and two
  // one-shot merges of 3.5 x 18 GiB do not fit. One-shot sizing: 80 / (3.5 x 2) = 11.4 GiB.
  auto in                 = sizing_input(36, 80, 4);
  in.policy.windowed_only = false;
  in.policy.one_shot_only = true;
  auto sizing             = compute_sort_partition_sizing(in);
  CHECK(sizing.num_partitions == 4);
  CHECK(std::string(sizing.rule) == "one_shot_memory");
  auto const partition_bytes = in.data_bytes / sizing.num_partitions;
  auto const partition_rows  = in.data_rows / sizing.num_partitions;
  CHECK(2 * (3 * partition_bytes + 8 * partition_rows) <= in.gpu_capacity_bytes);
  // Wider rows need less per byte: 3 + 8/64 instead of 3.5.
  in.data_rows = in.data_bytes / 64;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 3);
  // Not applied to the windowed and automatic routes.
  in.policy.one_shot_only = false;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 2);
  // A sort that does not fit the GPU takes the larger of the two counts.
  in                      = sizing_input(160, 80, 4);
  in.policy.windowed_only = false;
  in.policy.one_shot_only = true;
  CHECK(compute_sort_partition_sizing(in).num_partitions == 20);
}
