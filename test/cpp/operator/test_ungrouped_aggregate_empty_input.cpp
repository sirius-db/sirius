/*
 * Copyright 2026, Sirius Contributors.
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

#include "creator/task_creator.hpp"
#include "cudf/cudf_utils.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "expression/ast/node.hpp"
#include "op/input_free_operator_data.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/lists/lists_column_view.hpp>

#include <catch.hpp>
#include <cucascade/data/data_repository_manager.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
using cucascade::memory::Tier;
using sirius::aggregate_id;
using sirius::logical_type;
using sirius::type_id;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;

namespace {

using expressions = duckdb::vector<std::unique_ptr<sirius::ast::node>>;
using schema      = duckdb::vector<logical_type>;

auto const bigint  = logical_type::make(type_id::BIGINT);
auto const integer = logical_type::make(type_id::INTEGER);

std::unique_ptr<sirius::ast::node> aggregate(aggregate_id kind,
                                             logical_type result,
                                             logical_type input = integer,
                                             bool distinct      = false)
{
  std::vector<std::unique_ptr<sirius::ast::node>> children;
  if (kind != aggregate_id::count_star) {
    children.push_back(std::make_unique<sirius::ast::node>(sirius::ast::reference{0, input}));
  }
  return std::make_unique<sirius::ast::node>(
    sirius::ast::aggregate{kind, std::move(children), result, distinct});
}

expressions count_star()
{
  expressions result;
  result.push_back(aggregate(aggregate_id::count_star, bigint));
  return result;
}

class hint_walker final : public sirius::creator::task_creator {
 public:
  using task_creator::get_operator_for_next_task;
  using task_creator::task_creator;
};

class capture_result final : public sirius_physical_operator {
 public:
  explicit capture_result(schema types)
    : sirius_physical_operator(SiriusPhysicalOperatorType::RESULT_COLLECTOR, std::move(types), 1)
  {
  }

  std::unique_ptr<operator_data> execute(operator_data const& data, ::cuda::stream_ref) override
  {
    if (fail_next) {
      fail_next = false;
      throw rmm::out_of_memory("injected downstream failure after aggregate finalization");
    }
    return std::make_unique<pipelineable_operator_data>(
      dynamic_cast<pipelineable_operator_data const&>(data).get_data_batches());
  }

  void sink(operator_data const& data, ::cuda::stream_ref) override
  {
    ++publications;
    batches = dynamic_cast<pipelineable_operator_data const&>(data).get_data_batches();
  }

  bool fail_next           = false;
  std::size_t publications = 0;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
};

struct merge_fixture {
  sirius::pipeline::pipeline_build_context context{nullptr};
  cucascade::shared_data_repository_manager repositories;
  sirius_physical_operator producer;
  sirius_physical_ungrouped_aggregate_merge merge;
  capture_result collector;
  std::shared_ptr<sirius_pipeline> upstream;
  std::shared_ptr<sirius_pipeline> pipeline;

  explicit merge_fixture(schema types = {bigint}, expressions aggregates = count_star())
    : merge(types, std::move(aggregates), 1, duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES),
      collector(std::move(types)),
      upstream(std::make_shared<sirius_pipeline>(context)),
      pipeline(std::make_shared<sirius_pipeline>(context))
  {
    sirius_pipeline_build_state state;
    state.set_pipeline_source(*upstream, producer);
    state.set_pipeline_operators(*upstream, {producer});
    state.set_pipeline_sink(*upstream, &producer, 0);
    upstream->set_pipeline_id(0);
    state.set_pipeline_source(*pipeline, merge);
    state.set_pipeline_operators(*pipeline, {merge, collector});
    state.set_pipeline_sink(*pipeline, &collector, 0);
    pipeline->set_pipeline_id(1);
    pipeline->add_dependency(upstream);
    producer.set_pipeline(upstream);
    merge.set_pipeline(pipeline);
    collector.set_pipeline(pipeline);
    sirius::pipeline::assign_operator_ids({upstream, pipeline});
    std::vector<sirius::pipeline::repository_wiring> wiring{
      {"default", MemoryBarrierType::FULL, &producer, upstream, pipeline}};
    sirius::pipeline::materialize_repository_wiring(wiring, repositories);
  }

  void close_input()
  {
    upstream->update_pipeline_status(false);
    REQUIRE(upstream->is_pipeline_finished());
  }
};

std::shared_ptr<cucascade::data_batch> only_batch(operator_data const& output)
{
  auto const& batches = dynamic_cast<pipelineable_operator_data const&>(output).get_data_batches();
  REQUIRE(batches.size() == 1);
  return batches.front();
}

}  // namespace

TEST_CASE("ungrouped merge claims one task after zero-batch input closes",
          "[ungrouped_empty_input][scheduling]")
{
  merge_fixture fixture;
  fixture.pipeline->update_pipeline_status(false);
  CHECK_FALSE(fixture.pipeline->is_pipeline_finished());
  CHECK_FALSE(fixture.merge.all_ports_empty());
  auto waiting = fixture.merge.get_next_task_hint();
  REQUIRE(waiting.has_value());
  CHECK(waiting->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  CHECK(waiting->producer == &fixture.producer);
  CHECK(fixture.merge.get_next_task_input_data() == nullptr);
  fixture.close_input();
  CHECK_FALSE(fixture.pipeline->is_pipeline_finished());

  std::atomic<int> claims{0};
  {
    std::vector<std::jthread> callers;
    for (int i = 0; i < 16; ++i) {
      callers.emplace_back([&] {
        if (auto hint = fixture.merge.get_next_task_hint();
            hint && hint->hint == TaskCreationHint::READY) {
          ++claims;
        }
      });
    }
  }
  CHECK(claims.load() == 1);
  fixture.pipeline->update_pipeline_status(false);
  CHECK_FALSE(fixture.pipeline->is_pipeline_finished());
  CHECK_FALSE(fixture.merge.all_ports_empty());
  auto input = fixture.merge.get_next_task_input_data();
  REQUIRE(input != nullptr);
  CHECK(dynamic_cast<input_free_operator_data*>(input.get()) != nullptr);
  CHECK(input->get_estimated_size_in_bytes() >= (std::size_t{1} << 20));
  CHECK(fixture.merge.all_ports_empty());
  CHECK(fixture.merge.get_next_task_input_data() == nullptr);
  CHECK_FALSE(fixture.merge.get_next_task_hint().has_value());
}

TEST_CASE("ungrouped merge waits for closure even with partials and issues no empty fallback",
          "[ungrouped_empty_input][scheduling]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  merge_fixture fixture;
  fixture.merge.push_data_batch("default",
                                make_numeric_batch<int64_t>(*space, {3}, cudf::type_id::INT64));
  fixture.merge.push_data_batch("default",
                                make_numeric_batch<int64_t>(*space, {4}, cudf::type_id::INT64));
  auto waiting = fixture.merge.get_next_task_hint();
  REQUIRE(waiting.has_value());
  CHECK(waiting->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  fixture.close_input();
  REQUIRE(fixture.merge.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = fixture.merge.get_next_task_input_data();
  REQUIRE(input != nullptr);
  REQUIRE(dynamic_cast<pipelineable_operator_data*>(input.get()) != nullptr);
  auto result = fixture.merge.execute(*input, default_stream());
  auto batch  = only_batch(*result);
  CHECK(copy_column_to_host<int64_t>(sirius::get_cudf_table_view(*batch).column(0)) ==
        std::vector<int64_t>{7});
  CHECK(fixture.merge.get_next_task_input_data() == nullptr);
  CHECK_FALSE(fixture.merge.get_next_task_hint().has_value());
}

TEST_CASE("ungrouped merge gives equivalent SQL results for zero batches and zero rows",
          "[ungrouped_empty_input][physical_ungrouped_aggregate]")
{
  auto const zero_batches = GENERATE(true, false);
  auto const input_type   = logical_type::make(GENERATE(type_id::INTEGER, type_id::BIGINT));
  auto manager            = sirius::test::operator_utils::initialize_memory_manager();
  auto* space             = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  schema types{bigint,
               bigint,
               bigint,
               bigint,
               input_type,
               input_type,
               input_type,
               logical_type::make(type_id::DOUBLE),
               bigint,
               logical_type::make(type_id::DOUBLE)};
  auto make_aggregates = [input_type] {
    expressions result;
    result.push_back(aggregate(aggregate_id::count_star, bigint));
    result.push_back(aggregate(aggregate_id::count, bigint, input_type));
    result.push_back(aggregate(aggregate_id::count, bigint, input_type, true));
    result.push_back(aggregate(aggregate_id::sum, bigint, input_type));
    result.push_back(aggregate(aggregate_id::min, input_type, input_type));
    result.push_back(aggregate(aggregate_id::max, input_type, input_type));
    result.push_back(aggregate(aggregate_id::first, input_type, input_type));
    result.push_back(aggregate(aggregate_id::avg, logical_type::make(type_id::DOUBLE), input_type));
    result.push_back(aggregate(aggregate_id::sum_no_overflow, bigint, input_type));
    result.push_back(
      aggregate(aggregate_id::stddev_samp, logical_type::make(type_id::DOUBLE), input_type));
    return result;
  };
  merge_fixture fixture(types, make_aggregates());
  if (!zero_batches) {
    sirius_physical_ungrouped_aggregate local(
      types, make_aggregates(), 1, duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
    auto input   = input_type.id() == type_id::INTEGER
                     ? make_numeric_batch<int32_t>(*space, {}, cudf::type_id::INT32)
                     : make_numeric_batch<int64_t>(*space, {}, cudf::type_id::INT64);
    auto partial = local.execute(pipelineable_operator_data({input}), default_stream());
    auto batch   = only_batch(*partial);
    auto view    = sirius::get_cudf_table_view(*batch);
    CHECK(view.column(3).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
    CHECK(view.column(7).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
    CHECK(view.column(10).type() == cudf::data_type{cudf::type_id::STRUCT});
    fixture.merge.push_data_batch("default", std::move(batch));
  }
  fixture.close_input();
  REQUIRE(fixture.merge.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = fixture.merge.get_next_task_input_data();
  REQUIRE(input != nullptr);
  input->prepare_for_processing(space, default_stream());
  auto result = fixture.merge.execute(*input, default_stream());
  auto batch  = only_batch(*result);
  auto view   = sirius::get_cudf_table_view(*batch);
  REQUIRE(view.num_rows() == 1);
  REQUIRE(view.num_columns() == static_cast<cudf::size_type>(types.size()));
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    CHECK(view.column(i).type() == sirius::require_cudf_type(types[i]));
    if (i < 3) {
      CHECK(view.column(i).null_count() == 0);
      CHECK(copy_column_to_host<int64_t>(view.column(i)) == std::vector<int64_t>{0});
    } else {
      CHECK(view.column(i).null_count() == 1);
    }
  }
}

TEST_CASE("ungrouped merge combines ordinary AVG and distinct-count partials once",
          "[ungrouped_empty_input][physical_ungrouped_aggregate]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto make_aggregates = [] {
    expressions result;
    result.push_back(aggregate(aggregate_id::avg, logical_type::make(type_id::DOUBLE)));
    result.push_back(aggregate(aggregate_id::count, bigint, integer, true));
    result.push_back(aggregate(aggregate_id::count_star, bigint));
    return result;
  };
  schema types{logical_type::make(type_id::DOUBLE), bigint, bigint};
  merge_fixture fixture(types, make_aggregates());
  sirius_physical_ungrouped_aggregate local(
    types, make_aggregates(), 1, duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
  for (auto const& values : std::vector<std::vector<int32_t>>{{1, 2, 1}, {2, 3}}) {
    auto input   = make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
    auto partial = local.execute(pipelineable_operator_data({input}), default_stream());
    fixture.merge.push_data_batch("default", only_batch(*partial));
  }
  fixture.close_input();
  REQUIRE(fixture.merge.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input  = fixture.merge.get_next_task_input_data();
  auto output = fixture.merge.execute(*input, default_stream());
  auto batch  = only_batch(*output);
  auto view   = sirius::get_cudf_table_view(*batch);
  REQUIRE(view.num_rows() == 1);
  CHECK(copy_column_to_host<double>(view.column(0)).front() == Catch::Approx(1.8));
  CHECK(copy_column_to_host<int64_t>(view.column(1)) == std::vector<int64_t>{3});
  CHECK(copy_column_to_host<int64_t>(view.column(2)) == std::vector<int64_t>{5});
  CHECK(fixture.merge.get_next_task_input_data() == nullptr);
}

TEST_CASE("ungrouped merge empty final row preserves physical carriers and nested schema",
          "[ungrouped_empty_input][array]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto const decimal   = logical_type::make_decimal(18, 2);
  auto const text      = logical_type::make(type_id::VARCHAR);
  auto const timestamp = logical_type::make(type_id::TIMESTAMP);
  auto const array     = logical_type::make_array(integer, 3);
  expressions aggregates;
  for (auto const& type : schema{integer, decimal, text, timestamp, array}) {
    aggregates.push_back(aggregate(aggregate_id::first, type, type));
  }
  merge_fixture fixture({integer, decimal, text, timestamp, array}, std::move(aggregates));
  std::vector<cudf::data_type> physical{cudf::data_type{cudf::type_id::INT8},
                                        cudf::data_type{cudf::type_id::DECIMAL32, -2},
                                        cudf::data_type{cudf::type_id::STRING},
                                        cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                        cudf::data_type{cudf::type_id::LIST}};
  fixture.merge.set_physical_types(physical);
  fixture.close_input();
  REQUIRE(fixture.merge.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = fixture.merge.get_next_task_input_data();
  REQUIRE_THROWS_AS(fixture.merge.execute(*input, default_stream()), std::runtime_error);
  input->prepare_for_processing(manager->get_memory_space(Tier::HOST, 0), default_stream());
  REQUIRE_THROWS_AS(fixture.merge.execute(*input, default_stream()), std::runtime_error);
  input->prepare_for_processing(space, default_stream());
  auto result = fixture.merge.execute(*input, default_stream());
  auto batch  = only_batch(*result);
  auto view   = sirius::get_cudf_table_view(*batch);
  REQUIRE(view.num_rows() == 1);
  REQUIRE(view.num_columns() == 5);
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    CHECK(view.column(i).type() == physical[i]);
    CHECK(view.column(i).null_count() == 1);
  }
  CHECK(cudf::lists_column_view{view.column(4)}.child().type() ==
        cudf::data_type{cudf::type_id::INT32});
}

TEST_CASE("ungrouped merge validates unsupported distinct aggregates on zero batches",
          "[ungrouped_empty_input]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  expressions aggregates;
  aggregates.push_back(aggregate(aggregate_id::sum, bigint, integer, true));
  merge_fixture fixture({bigint}, std::move(aggregates));
  fixture.close_input();
  REQUIRE(fixture.merge.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = fixture.merge.get_next_task_input_data();
  input->prepare_for_processing(space, default_stream());
  CHECK_THROWS_AS(fixture.merge.execute(*input, default_stream()),
                  sirius::not_implemented_exception);
}

TEST_CASE("ungrouped merge empty reservation estimate scales with output width",
          "[ungrouped_empty_input][scheduling]")
{
  constexpr std::size_t width = 1024;
  expressions aggregates;
  schema types(width, bigint);
  for (std::size_t i = 0; i < width; ++i) {
    aggregates.push_back(aggregate(aggregate_id::count_star, bigint));
  }
  merge_fixture fixture(std::move(types), std::move(aggregates));
  fixture.close_input();
  REQUIRE(fixture.merge.get_next_task_hint()->hint == TaskCreationHint::READY);
  auto input = fixture.merge.get_next_task_input_data();
  REQUIRE(input != nullptr);
  CHECK(input->get_estimated_size_in_bytes() >= width * 4096);
  CHECK(fixture.merge.no_history_peak_memory_estimate(
          {.num_batches = 1, .bytes = 42, .type = operator_data_type::PIPELINEABLE}) == 84);
}

TEST_CASE("ungrouped merge empty task retries OOM and publishes once before completion",
          "[ungrouped_empty_input][gpu_pipeline_task]")
{
  auto const failure = GENERATE(0, 1, 2);
  auto manager       = sirius::test::operator_utils::initialize_memory_manager();
  auto* space        = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  merge_fixture fixture;
  auto completion = std::make_shared<sirius::pipeline::completion_handler>();
  if (failure == 1) {
    auto injection          = std::make_shared<sirius::op::scan::test_injections>();
    injection->gpu_task_oom = 1;
    completion->injections  = std::move(injection);
  }
  fixture.collector.fail_next = failure == 2;
  auto completed              = completion->get_awaitable();
  fixture.pipeline->set_completion_handler(completion);
  auto global = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
    fixture.pipeline, sirius::test::make_test_telemetry_context());
  global->set_completion_handler(completion);
  sirius::exec::query_lifecycle_registry lifecycle;
  hint_walker creator(lifecycle, sirius::creator::task_creator_config{}, *manager);
  std::vector<std::shared_ptr<sirius_pipeline>> visited;
  auto before_close = creator.get_operator_for_next_task(&fixture.merge, visited);
  CHECK_FALSE(before_close.is_ready);
  CHECK(before_close.op == &fixture.producer);
  REQUIRE(visited.size() == 2);
  fixture.close_input();
  CHECK_FALSE(fixture.pipeline->is_pipeline_finished());
  visited.clear();
  auto ready = creator.get_operator_for_next_task(&fixture.merge, visited);
  REQUIRE(ready.is_ready);
  CHECK(ready.op == &fixture.merge);
  fixture.pipeline->update_pipeline_status(false);
  CHECK_FALSE(fixture.pipeline->is_pipeline_finished());
  auto make_task = [&](uint64_t id, std::unique_ptr<operator_data> input, std::size_t resume = 0) {
    auto local =
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input), resume);
    local->set_preferred_device_id(0);
    auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      id, std::vector<cucascade::shared_data_repository*>{}, std::move(local), global);
    auto estimate = task->get_estimated_reservation_size_info(space);
    CHECK(estimate.bytes_to_materialize_input == 0);
    // Match task_scheduler::schedule and gpu_pipeline_executor::manager_loop before execution.
    auto& fsm = task->telemetry_fsm();
    fsm.queued({.queue = {.target = quent::task_queue::TaskQueueId{quent::nil_uuid()},
                          .data   = {.entries = 1}}});
    fsm.reserving({
      .requested_bytes      = estimate.reservation_size,
      .input_basis          = estimate.input_basis,
      .peak_estimate        = estimate.peak_memory_estimate,
      .bytes_to_materialize = estimate.bytes_to_materialize_input,
      .manager_thread =
        {
          .target = quent::task_manager_loop_thread::TaskManagerLoopThreadId{quent::nil_uuid()},
          .data   = {},
        },
    });
    auto reservation = manager->request_reservation(
      cucascade::memory::any_memory_space_in_tier{Tier::GPU}, estimate.reservation_size);
    REQUIRE(reservation != nullptr);
    dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(task->local_state())
      ->set_reservation(std::move(reservation), estimate);
    return task;
  };
  auto retire_task = [](std::unique_ptr<sirius::pipeline::gpu_pipeline_task>& task, bool success) {
    task->set_telemetry_finalized();
    task->telemetry_fsm().finalizing({.success = success});
    task.reset();
  };
  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> task;
  {
    auto task_lock = fixture.pipeline->get_task_creation_lock();
    task           = make_task(1, fixture.merge.get_next_task_input_data());
  }
  CHECK(fixture.merge.all_ports_empty());
  fixture.pipeline->update_pipeline_status(false);
  CHECK_FALSE(fixture.pipeline->is_pipeline_finished());
  auto stream = space->acquire_stream();
  if (failure != 0) {
    std::unique_ptr<operator_data> retry_input;
    std::size_t resume = 0;
    try {
      task->execute(stream);
      FAIL("the task must encounter the injected OOM");
    } catch (sirius::pipeline::oom_reschedule_exception& error) {
      resume = error.get_resume_operator_index();
      CHECK(resume == (failure == 1 ? 0 : 1));
      retry_input = error.release_intermediate_data();
    }
    REQUIRE(retry_input != nullptr);
    CHECK(fixture.collector.publications == 0);
    CHECK_FALSE(fixture.pipeline->is_pipeline_finished());
    CHECK(fixture.merge.get_next_task_input_data() == nullptr);
    auto retry = make_task(2, std::move(retry_input), resume);
    retire_task(task, false);
    retry->execute(stream);
    retire_task(retry, true);
  } else {
    task->execute(stream);
    retire_task(task, true);
  }
  CHECK(fixture.collector.publications == 1);
  REQUIRE(fixture.collector.batches.size() == 1);
  auto view = sirius::get_cudf_table_view(*fixture.collector.batches.front());
  REQUIRE(view.num_rows() == 1);
  CHECK(copy_column_to_host<int64_t>(view.column(0), stream) == std::vector<int64_t>{0});
  CHECK(fixture.pipeline->is_pipeline_finished());
  CHECK(completed.wait_for(std::chrono::seconds{0}) == std::future_status::ready);
  CHECK_NOTHROW(completed.get());
}
