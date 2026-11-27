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

#include "utils/test_physical_operator.hpp"

#include <catch.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/repository_wiring.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <string_view>
#include <vector>

using sirius::op::MemoryBarrierType;
using sirius::op::sirius_physical_operator;
using sirius::pipeline::materialize_repository_wiring;
using sirius::pipeline::repository_wiring;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;

namespace {

// Test scaffolding: minimal pipeline_build_context, reused across cases.
class wiring_test_env {
 public:
  wiring_test_env() = default;

  std::shared_ptr<sirius_pipeline> make_pipeline()
  {
    auto pipeline = std::make_shared<sirius_pipeline>(build_ctx);
    pipelines.push_back(pipeline);
    return pipeline;
  }

  // Number every operator built so far, exactly as sirius_engine does between pipeline
  // conversion and repository wiring. materialize_repository_wiring reads operator ids,
  // so this must run first.
  void assign_ids() { sirius::pipeline::assign_operator_ids(pipelines); }

  sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  std::vector<std::shared_ptr<sirius_pipeline>> pipelines;
};

// Build a pipeline with the given sink and operator list. `operators` may be
// empty to test the "next_op = sink" code path.
std::shared_ptr<sirius_pipeline> build_pipeline(
  wiring_test_env& env,
  sirius_physical_operator* sink,
  const std::vector<sirius_physical_operator*>& operators,
  size_t pipeline_id)
{
  auto pipeline = env.make_pipeline();
  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_sink(*pipeline, sink, /*pipeline_idx=*/0);
  for (auto* op : operators) {
    build_state.add_pipeline_operator(*pipeline, *op);
  }
  pipeline->set_pipeline_id(pipeline_id);
  return pipeline;
}

}  // namespace

TEST_CASE("materialize: basic 4-arg-style wiring attaches port and records sink fanout",
          "[repository_wiring][materializer]")
{
  wiring_test_env env;
  cucascade::shared_data_repository_manager mgr;

  // Source pipeline: sink op pushes through "default" port at FULL barrier.
  sirius::op::test_physical_operator source_sink;
  auto src_pipeline = build_pipeline(env, &source_sink, /*operators=*/{}, /*pipeline_id=*/0);

  // Destination pipeline: first non-sink operator is the consumer.
  sirius::op::test_physical_operator dest_first_op;
  sirius::op::test_physical_operator dest_sink;
  auto dest_pipeline = build_pipeline(env, &dest_sink, {&dest_first_op}, /*pipeline_id=*/1);

  std::vector<repository_wiring> wirings = {{std::string_view{"default"},
                                             MemoryBarrierType::FULL,
                                             &source_sink,
                                             src_pipeline,
                                             dest_pipeline}};

  env.assign_ids();

  materialize_repository_wiring(wirings, mgr);

  // Port should land on dest_first_op (operators[0]), not on dest_sink.
  auto* port = dest_first_op.get_port("default");
  REQUIRE(port != nullptr);
  CHECK(port->type == MemoryBarrierType::FULL);
  CHECK(port->repo != nullptr);
  CHECK(port->src_pipeline == src_pipeline);
  CHECK(port->dest_pipeline == dest_pipeline);

  // Sink fanout records the same (next_op, port_id) pair.
  auto next_ports = source_sink.get_next_ports_after_sink();
  REQUIRE(next_ports.size() == 1);
  CHECK(next_ports[0].next_operator == &dest_first_op);
  CHECK(next_ports[0].next_operator_port_name == "default");

  // Repository was registered under the destination operator's id.
  CHECK(mgr.get_repository(dest_first_op.operator_id, "default").get() == port->repo);
}

TEST_CASE("materialize: empty operators routes port onto destination sink",
          "[repository_wiring][materializer]")
{
  wiring_test_env env;
  cucascade::shared_data_repository_manager mgr;

  sirius::op::test_physical_operator source_sink;
  auto src_pipeline = build_pipeline(env, &source_sink, {}, /*pipeline_id=*/0);

  // Destination pipeline has no intermediate operators; port should go on the sink.
  sirius::op::test_physical_operator dest_sink;
  auto dest_pipeline = build_pipeline(env, &dest_sink, /*operators=*/{}, /*pipeline_id=*/1);

  std::vector<repository_wiring> wirings = {{std::string_view{"scan"},
                                             MemoryBarrierType::PIPELINE,
                                             &source_sink,
                                             src_pipeline,
                                             dest_pipeline}};

  env.assign_ids();

  materialize_repository_wiring(wirings, mgr);

  auto* port = dest_sink.get_port("scan");
  REQUIRE(port != nullptr);
  CHECK(port->type == MemoryBarrierType::PIPELINE);
  CHECK(port->repo != nullptr);

  auto next_ports = source_sink.get_next_ports_after_sink();
  REQUIRE(next_ports.size() == 1);
  CHECK(next_ports[0].next_operator == &dest_sink);
}

TEST_CASE("materialize: explicit source op (5-arg-style) records fanout on the sub-operator",
          "[repository_wiring][materializer]")
{
  wiring_test_env env;
  cucascade::shared_data_repository_manager mgr;

  // The descriptor names a source_op other than the pipeline's sink, so
  // add_next_port_after_sink should land on it instead of on the sink.
  sirius::op::test_physical_operator pipeline_sink;
  sirius::op::test_physical_operator sub_emitter;
  auto src_pipeline = build_pipeline(env, &pipeline_sink, {}, /*pipeline_id=*/0);

  sirius::op::test_physical_operator dest_first_op;
  sirius::op::test_physical_operator dest_sink;
  auto dest_pipeline = build_pipeline(env, &dest_sink, {&dest_first_op}, /*pipeline_id=*/1);

  std::vector<repository_wiring> wirings = {{std::string_view{"default"},
                                             MemoryBarrierType::FULL,
                                             &sub_emitter,  // explicit source op
                                             src_pipeline,
                                             dest_pipeline}};

  env.assign_ids();

  materialize_repository_wiring(wirings, mgr);

  // Sub-emitter records the fanout, NOT pipeline_sink.
  auto sub_ports = sub_emitter.get_next_ports_after_sink();
  REQUIRE(sub_ports.size() == 1);
  CHECK(sub_ports[0].next_operator == &dest_first_op);

  auto sink_ports = pipeline_sink.get_next_ports_after_sink();
  CHECK(sink_ports.size() == 0);
}

TEST_CASE("materialize: multiple wirings to the same destination attach distinct ports",
          "[repository_wiring][materializer]")
{
  wiring_test_env env;
  cucascade::shared_data_repository_manager mgr;

  sirius::op::test_physical_operator probe_sink;
  auto probe_pipeline = build_pipeline(env, &probe_sink, {}, /*pipeline_id=*/0);

  sirius::op::test_physical_operator build_sink;
  auto build_pipeline_handle = build_pipeline(env, &build_sink, {}, /*pipeline_id=*/1);

  // Hash-join-like destination: receives "default" from probe and "build" from build side.
  sirius::op::test_physical_operator hash_join_op;
  sirius::op::test_physical_operator dest_sink;
  auto dest_pipeline = build_pipeline(env, &dest_sink, {&hash_join_op}, /*pipeline_id=*/2);

  std::vector<repository_wiring> wirings = {
    {std::string_view{"default"},
     MemoryBarrierType::PARTIAL,
     &probe_sink,
     probe_pipeline,
     dest_pipeline},
    {std::string_view{"build"},
     MemoryBarrierType::FULL,
     &build_sink,
     build_pipeline_handle,
     dest_pipeline},
  };

  env.assign_ids();

  materialize_repository_wiring(wirings, mgr);

  auto* default_port = hash_join_op.get_port("default");
  auto* build_port   = hash_join_op.get_port("build");
  REQUIRE(default_port != nullptr);
  REQUIRE(build_port != nullptr);
  CHECK(default_port->type == MemoryBarrierType::PARTIAL);
  CHECK(build_port->type == MemoryBarrierType::FULL);
  CHECK(default_port->src_pipeline == probe_pipeline);
  CHECK(build_port->src_pipeline == build_pipeline_handle);
  CHECK(default_port->repo != build_port->repo);  // Each port has its own repo.
}

namespace {

class finishable_pipeline : public sirius_pipeline {
 public:
  using sirius_pipeline::sirius_pipeline;
  bool is_pipeline_finished() const override { return finished; }
  bool finished = false;
};

}  // namespace

TEST_CASE("spill rank orders repositories by when their consumer can run",
          "[repository_wiring][materializer]")
{
  wiring_test_env env;
  cucascade::shared_data_repository_manager mgr;
  std::vector<std::shared_ptr<finishable_pipeline>> p;
  std::vector<sirius::op::test_physical_operator> sinks(5);
  sirius::op::test_physical_operator partition_op;
  sirius::op::test_physical_operator join_op;
  for (size_t i = 0; i < sinks.size(); ++i) {
    auto pipeline = std::make_shared<finishable_pipeline>(env.build_ctx);
    sirius_pipeline_build_state build_state;
    build_state.set_pipeline_sink(*pipeline, &sinks[i], /*pipeline_idx=*/0);
    if (i == 1) { build_state.add_pipeline_operator(*pipeline, partition_op); }
    if (i == 4) { build_state.add_pipeline_operator(*pipeline, join_op); }
    pipeline->set_pipeline_id(i);
    p.push_back(pipeline);
    env.pipelines.push_back(pipeline);
  }
  // Scheduled order: 0 scan -> 1 partition (FULL); 2 build and 3 probe producers -> 4 join.
  std::vector<repository_wiring> wirings = {
    {std::string_view{"default"}, MemoryBarrierType::FULL, &sinks[0], p[0], p[1]},
    {std::string_view{"build"}, MemoryBarrierType::PARTIAL, &sinks[2], p[2], p[4]},
    {std::string_view{"default"}, MemoryBarrierType::PARTIAL, &sinks[3], p[3], p[4]},
  };
  env.assign_ids();
  materialize_repository_wiring(wirings, mgr);
  auto rank = sirius::pipeline::make_spill_rank(wirings, env.pipelines, mgr);

  auto const* partition_input = partition_op.get_port("default")->repo;
  auto const* join_build      = join_op.get_port("build")->repo;
  auto const* join_probe      = join_op.get_port("default")->repo;
  CHECK(rank(partition_input) == 0);
  CHECK(rank(join_build) == 3);
  CHECK(rank(join_probe) == 3);

  // The join still waits for its probe producer once the build side is done.
  p[2]->finished = true;
  CHECK(rank(join_build) == 3);
  p[3]->finished = true;
  CHECK(rank(join_build) == -1);
  CHECK(rank(join_probe) == -1);

  p[0]->finished = true;
  CHECK(rank(partition_input) == -1);

  cucascade::shared_data_repository unknown;
  CHECK(rank(&unknown) == -1);
}
