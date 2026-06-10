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

#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
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

sirius_physical_sort_sample make_sort_sample(uint64_t sort_sample_bytes)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::LogicalType::INTEGER));

  return sirius_physical_sort_sample(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    std::move(orders),
    1000,
    sort_sample_bytes);
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
                 duckdb::shared_ptr<mock_gpu_pipeline> src_pipeline = nullptr)
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

  const sirius::pipeline::pipeline_build_context build_ctx{true};
  auto src_pipeline = duckdb::make_shared_ptr<mock_gpu_pipeline>(build_ctx);
  src_pipeline->set_finished(false);

  // get_next_task_hint() reports the upstream pipeline's first operator as the producer.
  auto upstream_producer = duckdb::make_uniq<sirius_physical_operator>(
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

  const sirius::pipeline::pipeline_build_context build_ctx{true};
  auto src_pipeline = duckdb::make_shared_ptr<mock_gpu_pipeline>(build_ctx);
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

  const sirius::pipeline::pipeline_build_context build_ctx{true};
  auto src_pipeline = duckdb::make_shared_ptr<mock_gpu_pipeline>(build_ctx);
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
