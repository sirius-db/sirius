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

#include "data/convertible_data_batch.hpp"
#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"
#include "utils/test_physical_operator.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/order/bounded_sort.hpp>
#include <op/sirius_physical_sort_partition.hpp>
#include <op/sirius_physical_sort_sample.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_config.hpp>

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
  CHECK(sample_op.get_partition_bytes() == 2500);
  REQUIRE(sirius::get_cudf_table_view(*dynamic_cast<sort_sample_output&>(*output).boundaries)
            .num_rows() == 1);

  const auto boundary_key = read_int32_column(
    sirius::get_cudf_table_view(*dynamic_cast<sort_sample_output&>(*output).boundaries).column(0),
    0);
  REQUIRE(boundary_key == 50);

  // Replaying a boundary-producing task after a downstream compute failure keeps
  // the original ranges, even though the operator is already marked DONE.
  auto retried       = sample_op.execute(pipelineable_operator_data({batch_high, batch_low}),
                                   cudf::get_default_stream());
  auto& retry_output = dynamic_cast<sort_sample_output&>(*retried);
  REQUIRE(retry_output.boundaries);
  CHECK(sample_op.get_num_partitions() == 2);
  CHECK(read_int32_column(sirius::get_cudf_table_view(*retry_output.boundaries).column(0), 0) ==
        50);
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

  auto output = sample_op.execute(*input, cudf::get_default_stream());
  CHECK(sample_op.get_num_partitions() == 2);
  CHECK(sample_op.get_partition_bytes() == batch_bytes - 1);
  CHECK(sirius::get_cudf_table_view(*dynamic_cast<sort_sample_output&>(*output).boundaries)
          .num_rows() == 1);

  // Subsequent tasks include boundaries in admission and device preparation, but
  // never expose boundary rows as payload rows to partitioning.
  add_int_batches(*repo, *space, 1, 10);
  auto continuation = sample_op.get_next_task_input_data();
  REQUIRE(continuation);
  auto& prepared = dynamic_cast<pipelineable_operator_data&>(*continuation);
  REQUIRE(prepared.get_data_batches().size() == 2);
  continuation->prepare_for_processing(space, cudf::get_default_stream());
  auto forwarded = sample_op.execute(*continuation, cudf::get_default_stream());
  auto& sampled  = dynamic_cast<sort_sample_output&>(*forwarded);
  REQUIRE(sampled.boundaries);
  CHECK(sampled.get_data_batches().size() == 1);
  CHECK(sirius::get_cudf_table_view(*sampled.boundaries).num_rows() == 1);
}

TEST_CASE("sirius_physical_sort_sample sizes automatic partitions from runs and chunk size",
          "[physical_sort_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::INTEGER));
  // Two complete runs and a tiny chunk: the piece floor (two runs of 64 MiB) sets
  // the target, so this small input stays in one range. The whole of the test memory
  // space is the cap, which keeps it above the floor.
  sirius_physical_sort_sample sample_op(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    std::move(orders),
    1000,
    /*sort_sample_bytes=*/1'000'000,
    /*max_partition_bytes=*/0,
    /*max_partition_memory_fraction=*/1.0,
    nullptr,
    /*chunk_bytes=*/1);
  auto repo = std::make_unique<cucascade::shared_data_repository>();
  add_int_batches(*repo, *space, 2, 10);
  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto src_pipeline = std::make_shared<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(true);
  attach_port(sample_op, *repo, src_pipeline);

  auto input = sample_op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  REQUIRE(space->get_available_memory(cudf::get_default_stream()) >
          4 * sirius::op::sort_min_piece_bytes);
  sample_op.execute(*input, cudf::get_default_stream());
  CHECK(sample_op.get_num_partitions() == 1);
  CHECK(sample_op.get_partition_bytes() == 2 * sirius::op::sort_min_piece_bytes);
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

  auto output = sample_op.execute(*input, cudf::get_default_stream());
  CHECK(sample_op.get_num_partitions() == rows_per_batch);
  CHECK(sirius::get_cudf_table_view(*dynamic_cast<sort_sample_output&>(*output).boundaries)
          .num_rows() == rows_per_batch - 1);
}

TEST_CASE("registered sampled boundaries spill and route duplicate ranges on another GPU",
          "[multi_gpu][sort_boundaries]")
{
  int devices = 0;
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  if (devices < 2) { SKIP("requires two GPUs"); }
  std::optional<rmm::cuda_stream> remote_stream;
  auto manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto* gpu    = manager->get_memory_space(Tier::GPU, 0);
  auto* remote = manager->get_memory_space(Tier::GPU, 1);
  auto* host   = manager->get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu);
  REQUIRE(remote);
  REQUIRE(host);
  rmm::cuda_stream stream;
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{1}};
    remote_stream.emplace();
  }
  cucascade::shared_data_repository_manager repositories;
  auto sample        = make_sort_sample(16, 16);
  sample.operator_id = 1;
  sample.register_boundaries(repositories);
  cucascade::shared_data_repository input;
  attach_port(sample, input);
  input.add_data_batch(make_numeric_batch<int32_t>(*gpu, {0, 0, 4, 4}, cudf::type_id::INT32));
  auto first = sample.get_next_task_input_data();
  REQUIRE(first);
  auto computed = sample.execute(*first, stream);
  stream.synchronize();
  REQUIRE(sample.get_num_partitions() == 4);
  computed.reset();
  first.reset();
  auto* boundaries = repositories.get_repository(1, "sort_boundaries").get();
  sirius::convertible_data_batch_provider candidates(boundaries);
  auto candidate = candidates.get_next_convertible(gpu, true);
  REQUIRE(candidate);
  REQUIRE(candidate->convert({host}, stream, *manager, true).has_value());
  auto boundary = boundaries->get_data_batch_by_id(boundaries->get_batch_ids().front());
  {
    auto ro = boundary->to_read_only();
    CHECK(ro.get_memory_space() == host);
  }
  input.add_data_batch(make_numeric_batch<int32_t>(*gpu, {0, 1, 4, 8}, cudf::type_id::INT32));
  auto next = sample.get_next_task_input_data();
  REQUIRE(next);
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{1}};
    next->prepare_for_processing(remote, *remote_stream);
    auto forwarded = sample.execute(*next, *remote_stream);
    auto& sampled  = dynamic_cast<sort_sample_output&>(*forwarded);
    REQUIRE(sampled.boundaries);
    CHECK(sampled.boundaries->get_memory_space() == remote);
    sirius_physical_sort_partition partition(
      sample.types, copy_orders(sample.orders), {0}, 0, sample.comparison);
    partition.set_sample_op(&sample);
    auto output = partition.execute(*forwarded, *remote_stream);
    remote_stream->synchronize();
    auto& routed = dynamic_cast<sort_partition_output&>(*output);
    CHECK(routed.partitions == std::vector<std::size_t>{1, 3});
    REQUIRE(routed.get_data_batches().size() == 2);
    CHECK(copy_column_to_host<int32_t>(
            sirius::get_cudf_table_view(*routed.get_data_batches()[0]).column(0)) ==
          std::vector<int32_t>{0, 1});
    CHECK(copy_column_to_host<int32_t>(
            sirius::get_cudf_table_view(*routed.get_data_batches()[1]).column(0)) ==
          std::vector<int32_t>{4, 8});
  }
  next.reset();
  sample.clear_partition_boundaries();
  CHECK(boundaries->all_empty());
}
