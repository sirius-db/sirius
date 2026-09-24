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

#include "catch.hpp"
#include "creator/task_creator.hpp"
#include "op/sirius_physical_union.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/types.hpp>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace {

using sirius::op::MemoryBarrierType;
using sirius::op::sirius_physical_operator;
using sirius::op::sirius_physical_union;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::op::TaskCreationHint;
using sirius::pipeline::pipeline_build_context;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;

class controllable_pipeline final : public sirius_pipeline {
 public:
  explicit controllable_pipeline(const pipeline_build_context& context) : sirius_pipeline(context)
  {
  }

  void set_finished(bool finished) { _finished.store(finished); }

  bool is_pipeline_finished() const override { return _finished.load(); }

 private:
  std::atomic<bool> _finished{false};
};

class recording_task_creator final : public sirius::creator::task_creator {
 public:
  recording_task_creator(sirius::memory::sirius_memory_reservation_manager& mem_mgr,
                         sirius::creator::request_type strategy)
    : task_creator(sirius::creator::task_creator_config{.strategy = strategy}, mem_mgr)
  {
  }

  void schedule(sirius_physical_operator* request) override
  {
    std::lock_guard<std::mutex> guard(_mutex);
    _scheduled.push_back(request);
  }

  std::size_t schedule_count(const sirius_physical_operator* request)
  {
    std::lock_guard<std::mutex> guard(_mutex);
    return static_cast<std::size_t>(std::count(_scheduled.begin(), _scheduled.end(), request));
  }

 private:
  std::mutex _mutex;
  std::vector<sirius_physical_operator*> _scheduled;
};

class union_fixture {
 private:
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> _memory_manager;
  recording_task_creator _creator;

 public:
  explicit union_fixture(
    std::size_t num_arms,
    sirius::creator::request_type strategy = sirius::creator::request_type::active)
    : _memory_manager(sirius::test::operator_utils::initialize_memory_manager()),
      _creator(*_memory_manager, strategy),
      union_op({}, 0),
      union_pipeline(std::make_shared<sirius_pipeline>(pipeline_build_context{nullptr, true}))
  {
    union_op.set_pipeline(union_pipeline);
    union_pipeline->set_task_creator(&_creator);

    sirius_pipeline_build_state build_state;
    for (std::size_t arm = 0; arm < num_arms; ++arm) {
      auto producer = duckdb::make_uniq<sirius_physical_operator>(
        SiriusPhysicalOperatorType::PROJECTION, duckdb::vector<sirius::logical_type>{}, 0);
      auto* producer_ptr = producer.get();
      union_op.children.push_back(std::move(producer));
      producers.push_back(producer_ptr);

      auto pipeline =
        std::make_shared<controllable_pipeline>(pipeline_build_context{nullptr, true});
      duckdb::vector<std::reference_wrapper<sirius_physical_operator>> operators;
      operators.emplace_back(*producer_ptr);
      build_state.set_pipeline_operators(*pipeline, std::move(operators));
      pipeline->set_pipeline_id(arm);
      pipeline->set_task_creator(&_creator);
      source_pipelines.push_back(pipeline);

      auto repository     = std::make_unique<cucascade::shared_data_repository>();
      auto port           = std::make_unique<sirius_physical_operator::port>();
      port->type          = MemoryBarrierType::PARTIAL;
      port->repo          = repository.get();
      port->src_pipeline  = pipeline;
      port->dest_pipeline = union_pipeline;
      union_op.add_port(sirius_physical_union::port_label(arm), std::move(port));
      repositories.push_back(std::move(repository));
    }
  }

  std::shared_ptr<cucascade::data_batch> make_batch(int32_t value)
  {
    auto* gpu_space =
      _memory_manager->get_memory_space(cucascade::memory::Tier::GPU, /*device_id=*/0);
    REQUIRE(gpu_space != nullptr);
    return sirius::test::operator_utils::make_numeric_batch<int32_t>(
      *gpu_space, {value}, cudf::type_id::INT32);
  }

  std::size_t self_schedule_count() { return _creator.schedule_count(&union_op); }

  std::size_t producer_schedule_count(std::size_t arm)
  {
    return _creator.schedule_count(producers.at(arm));
  }

  sirius_physical_union union_op;
  std::shared_ptr<sirius_pipeline> union_pipeline;
  std::vector<sirius_physical_operator*> producers;
  std::vector<std::shared_ptr<controllable_pipeline>> source_pipelines;
  std::vector<std::unique_ptr<cucascade::shared_data_repository>> repositories;
};

}  // namespace

TEST_CASE("physical_union nominates and drains only its active arm", "[physical_union]")
{
  union_fixture fixture(3);
  fixture.repositories[1]->add_data_batch(fixture.make_batch(1));
  fixture.repositories[2]->add_data_batch(fixture.make_batch(2));

  auto hint = fixture.union_op.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(hint->producer == fixture.producers[0]);
  REQUIRE_FALSE(fixture.union_op.get_next_task_hint().has_value());

  REQUIRE(fixture.union_op.get_next_task_input_data() == nullptr);
  REQUIRE(fixture.repositories[1]->total_size() == 1);
  REQUIRE(fixture.repositories[2]->total_size() == 1);

  fixture.repositories[0]->add_data_batch(fixture.make_batch(0));
  hint = fixture.union_op.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::READY);
  REQUIRE(hint->producer == &fixture.union_op);

  REQUIRE(fixture.union_op.get_next_task_input_data() != nullptr);
  REQUIRE(fixture.repositories[0]->total_size() == 0);
  REQUIRE(fixture.repositories[1]->total_size() == 1);
  REQUIRE_FALSE(fixture.union_op.get_next_task_hint().has_value());
}

TEST_CASE("physical_union skips finished empty arms", "[physical_union]")
{
  union_fixture fixture(3);
  fixture.source_pipelines[0]->set_finished(true);
  fixture.source_pipelines[1]->set_finished(true);

  auto hint = fixture.union_op.get_next_task_hint();
  REQUIRE(hint.has_value());
  REQUIRE(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(hint->producer == fixture.producers[2]);
  REQUIRE_FALSE(fixture.union_op.get_next_task_hint().has_value());
}

TEST_CASE("physical_union sends a full request for a pre-seeded arm only under lookahead",
          "[physical_union]")
{
  SECTION("active strategy: latch only, no enqueue")
  {
    union_fixture fixture(1);
    fixture.repositories[0]->add_data_batch(fixture.make_batch(0));

    REQUIRE(fixture.union_op.get_next_task_hint()->hint == TaskCreationHint::READY);
    REQUIRE(fixture.producer_schedule_count(0) == 0);
    REQUIRE(fixture.union_op.get_next_task_input_data() != nullptr);
    // Latched by the READY above: an empty live arm is not re-nominated.
    REQUIRE_FALSE(fixture.union_op.get_next_task_hint().has_value());
    REQUIRE(fixture.producer_schedule_count(0) == 0);
  }

  SECTION("lookahead strategy: latch and exactly one full request")
  {
    union_fixture fixture(1, sirius::creator::request_type::lookahead);
    fixture.repositories[0]->add_data_batch(fixture.make_batch(0));

    REQUIRE(fixture.union_op.get_next_task_hint()->hint == TaskCreationHint::READY);
    REQUIRE(fixture.producer_schedule_count(0) == 1);
    REQUIRE(fixture.union_op.get_next_task_hint()->hint == TaskCreationHint::READY);
    REQUIRE(fixture.producer_schedule_count(0) == 1);
    REQUIRE(fixture.union_op.get_next_task_input_data() != nullptr);
    REQUIRE_FALSE(fixture.union_op.get_next_task_hint().has_value());
    REQUIRE(fixture.producer_schedule_count(0) == 1);
  }
}

TEST_CASE("physical_union final pop schedules one handoff without advancing early",
          "[physical_union]")
{
  union_fixture fixture(2);
  fixture.source_pipelines[0]->set_finished(true);
  fixture.repositories[0]->add_data_batch(fixture.make_batch(0));
  fixture.repositories[0]->add_data_batch(fixture.make_batch(1));

  REQUIRE(fixture.union_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  REQUIRE(fixture.union_op.get_next_task_input_data() != nullptr);
  REQUIRE(fixture.repositories[0]->total_size() == 1);
  REQUIRE(fixture.self_schedule_count() == 0);

  REQUIRE(fixture.union_op.get_next_task_input_data() != nullptr);
  REQUIRE(fixture.repositories[0]->total_size() == 0);
  REQUIRE(fixture.self_schedule_count() == 1);

  auto handoff = fixture.union_op.get_next_task_hint();
  REQUIRE(handoff.has_value());
  REQUIRE(handoff->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(handoff->producer == fixture.producers[1]);
  REQUIRE_FALSE(fixture.union_op.get_next_task_hint().has_value());
  REQUIRE(fixture.self_schedule_count() == 1);
}

TEST_CASE("physical_union final arm does not enqueue a handoff", "[physical_union]")
{
  union_fixture fixture(1);
  fixture.source_pipelines[0]->set_finished(true);
  fixture.repositories[0]->add_data_batch(fixture.make_batch(0));

  REQUIRE(fixture.union_op.get_next_task_hint()->hint == TaskCreationHint::READY);
  REQUIRE(fixture.union_op.get_next_task_input_data() != nullptr);
  REQUIRE(fixture.self_schedule_count() == 0);
  REQUIRE_FALSE(fixture.union_op.get_next_task_hint().has_value());
}
