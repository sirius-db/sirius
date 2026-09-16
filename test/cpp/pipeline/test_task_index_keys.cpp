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

/**
 * @file test_task_index_keys.cpp
 * @brief index_keys_for is the single key extractor shared by the task_scheduler queue and every
 *        gpu_pipeline_executor queue. If the two disagreed on a task's query, a per-query drain
 *        would clear it from one queue and leave it in the other.
 */

#include "catch.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "query_id.hpp"
#include "utils/telemetry_utils.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace {

using sirius::make_query_id;
using sirius::pipeline::index_keys_for;

//! A task that is not a gpu_pipeline_task, to exercise the sentinel branch.
class plain_task : public sirius::parallel::itask {
 public:
  plain_task() : itask(/*task_id=*/1, nullptr, nullptr) {}
  void execute(rmm::cuda_stream_view /*stream*/) override {}
};

struct task_fixture {
  //! A pipeline needs a real source/sink: ~gpu_pipeline_task calls mark_task_completed(), which
  //! walks the pipeline's operators.
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> make_pipeline(
    sirius::query_id_t query_id, sirius::exec::queue_priority priority)
  {
    auto pipeline  = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
    auto& op       = *operators.emplace_back(std::make_unique<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::FILTER,
      duckdb::vector<sirius::logical_type>{},
      /*estimated_cardinality=*/0));
    op.operator_id = operators.size() - 1;

    sirius::pipeline::sirius_pipeline_build_state build_state;
    build_state.set_pipeline_source(*pipeline, op);
    build_state.set_pipeline_sink(*pipeline, &op, /*sink_pipeline_count=*/1);

    pipeline->set_query_id(query_id);
    pipeline->set_priority(priority);
    return pipeline;
  }

  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> make_task(
    const duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>& pipeline,
    sirius::exec::queue_priority priority)
  {
    auto global_state = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    global_state->set_priority(priority);

    auto op_data = std::make_unique<sirius::op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
    return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      /*task_id=*/1,
      std::vector<cucascade::shared_data_repository*>{},
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data)),
      std::move(global_state));
  }

  sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  //! Outlive every pipeline/task built from this fixture.
  std::vector<std::unique_ptr<sirius::op::sirius_physical_operator>> operators;
};

}  // namespace

TEST_CASE("index_keys_for takes the query id from the task's pipeline", "[task_index_keys]")
{
  task_fixture f;
  const auto query_id = make_query_id(7);
  auto pipeline       = f.make_pipeline(query_id, /*priority=*/42);
  auto task           = f.make_task(pipeline, /*priority=*/42);

  const auto keys = index_keys_for(*task);

  CHECK(keys.query_id == sirius::value_of(query_id));
  CHECK(keys.priority == 42);
}

TEST_CASE("index_keys_for does not unpack the query id from the priority", "[task_index_keys]")
{
  task_fixture f;
  // query_priority_bits masks the id to 31 bits, so a bit-31 id and 0 pack to the SAME priority
  // bits. Recovering the query from those bits would report 0 here, and a
  // drain(query_index{value_of(query_id)}) would then never match this task.
  const auto query_id = make_query_id(0x8000'0000U);
  REQUIRE(sirius::query_priority_bits(query_id) == sirius::query_priority_bits(make_query_id(0)));

  auto pipeline = f.make_pipeline(query_id, /*priority=*/sirius::query_priority_bits(query_id));
  auto task     = f.make_task(pipeline, /*priority=*/sirius::query_priority_bits(query_id));

  const auto keys = index_keys_for(*task);

  CHECK(keys.query_id == sirius::value_of(query_id));
  CHECK(keys.query_id != 0U);
}

TEST_CASE("index_keys_for reports the pipeline source's operator type", "[task_index_keys]")
{
  task_fixture f;
  auto pipeline = f.make_pipeline(make_query_id(3), /*priority=*/1);
  auto task     = f.make_task(pipeline, /*priority=*/1);

  const auto keys = index_keys_for(*task);
  CHECK(keys.operator_type == sirius::op::SiriusPhysicalOperatorType::FILTER);
}

TEST_CASE("index_keys_for gives a non-pipeline task sentinel keys", "[task_index_keys]")
{
  plain_task task;

  const auto keys = index_keys_for(task);

  // Max priority sorts it last; the sentinel query/device keys keep it out of any query's
  // bucket, so a per-query drain can never remove it.
  CHECK(keys.priority == std::numeric_limits<sirius::exec::queue_priority>::max());
  CHECK(keys.operator_type == sirius::op::SiriusPhysicalOperatorType::INVALID);
  CHECK(keys.query_id == 0U);
  CHECK(keys.device_id == sirius::exec::no_preferred_device);
}

TEST_CASE("index_keys_for defaults to no preferred device", "[task_index_keys]")
{
  task_fixture f;
  auto pipeline = f.make_pipeline(make_query_id(5), /*priority=*/9);
  auto task     = f.make_task(pipeline, /*priority=*/9);

  const auto keys = index_keys_for(*task);
  CHECK(keys.device_id == sirius::exec::no_preferred_device);
}
