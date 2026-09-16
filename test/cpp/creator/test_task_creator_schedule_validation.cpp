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
 * @file test_task_creator_schedule_validation.cpp
 * @brief schedule() keys every creation request by the operator's pipeline (query id +
 *        priority), so a pipeline-less operator has no valid key.
 *
 * Accepting one and substituting query 0 would not surface until much later: manager_loop()
 * finds no state registered for query 0 and drops the request, and the query then waits forever
 * for a task that was never created. These cases pin the loud failure instead.
 */

#include "catch.hpp"
#include "creator/config.hpp"
#include "creator/task_creator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "query_id.hpp"
#include "sirius/exception.hpp"

#include <memory>

namespace {

using sirius::creator::task_creator;
using sirius::creator::task_creator_config;

struct schedule_fixture {
  schedule_fixture()
    : memory_manager(sirius::test::operator_utils::initialize_memory_manager(1)),
      creator(task_creator_config{}, *memory_manager)
  {
  }

  //! An operator that no pipeline ever claimed, as an unplaced plan node would be.
  std::unique_ptr<sirius::op::sirius_physical_operator> make_operator()
  {
    auto op = std::make_unique<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::FILTER,
      duckdb::vector<sirius::logical_type>{},
      /*estimated_cardinality=*/0);
    op->operator_id = 0;
    return op;
  }

  //! Place @p op in a pipeline the way planner::query::build_indices does.
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> place(
    sirius::op::sirius_physical_operator& op, sirius::query_id_t query_id)
  {
    auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
    sirius::pipeline::sirius_pipeline_build_state build_state;
    build_state.set_pipeline_source(*pipeline, op);
    build_state.set_pipeline_sink(*pipeline, &op, /*sink_pipeline_count=*/1);
    pipeline->set_query_id(query_id);
    pipeline->set_priority(3);
    op.set_pipeline(pipeline);
    return pipeline;
  }

  sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager;
  task_creator creator;
};

}  // namespace

TEST_CASE("task_creator::schedule rejects a null operator", "[task_creator][schedule]")
{
  schedule_fixture f;

  REQUIRE_THROWS_AS(f.creator.schedule(nullptr), sirius::internal_exception);
}

TEST_CASE("task_creator::schedule rejects an operator with no pipeline", "[task_creator][schedule]")
{
  schedule_fixture f;
  auto op = f.make_operator();

  // The creation worker dereferences node->get_pipeline() unconditionally, so there is no
  // meaningful way to service this request; failing at the producer names the operator.
  REQUIRE_THROWS_AS(f.creator.schedule(op.get()), sirius::internal_exception);
}

TEST_CASE("task_creator::schedule accepts a placed operator", "[task_creator][schedule]")
{
  schedule_fixture f;
  const auto query_id = sirius::make_query_id(11);
  auto op             = f.make_operator();
  auto pipeline       = f.place(*op, query_id);

  REQUIRE_NOTHROW(f.creator.schedule(op.get()));

  // Drop the queued request before `op` dies: it holds a raw pointer into the plan.
  f.creator.drain_pending_tasks(query_id);
}

TEST_CASE("task_creator::schedule with an explicit query id still requires a pipeline",
          "[task_creator][schedule]")
{
  schedule_fixture f;
  auto op = f.make_operator();

  // The query id is supplied here, but the priority still comes from the pipeline, so the
  // overload is no laxer than the one-argument form.
  REQUIRE_THROWS_AS(f.creator.schedule(op.get(), sirius::make_query_id(11)),
                    sirius::internal_exception);
}
