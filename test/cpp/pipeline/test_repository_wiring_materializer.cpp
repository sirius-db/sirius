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

#include <catch.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <duckdb.hpp>
#include <helper/logical_type.hpp>
#include <op/sirius_physical_column_data_scan.hpp>
#include <op/sirius_physical_delim_join.hpp>
#include <op/sirius_physical_dummy_scan.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <op/sirius_physical_partition.hpp>
#include <pipeline/repository_wiring.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <stdexcept>
#include <string_view>
#include <vector>

using sirius::op::MemoryBarrierType;
using sirius::op::sirius_physical_operator;
using sirius::op::sirius_physical_partition;
using sirius::op::sirius_physical_right_delim_join;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::materialize_repository_wiring;
using sirius::pipeline::repository_wiring;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;

namespace {

// Test scaffolding: minimal pipeline_build_context, reused across cases.
class wiring_test_env {
 public:
  wiring_test_env() = default;

  duckdb::shared_ptr<sirius_pipeline> make_pipeline()
  {
    return duckdb::make_shared_ptr<sirius_pipeline>(build_ctx);
  }

  sirius::pipeline::pipeline_build_context build_ctx{nullptr};
};

// Build a pipeline with the given sink and operator list. `operators` may be
// empty to test the "next_op = sink" code path.
duckdb::shared_ptr<sirius_pipeline> build_pipeline(
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
  sirius_physical_operator source_sink;
  auto src_pipeline = build_pipeline(env, &source_sink, /*operators=*/{}, /*pipeline_id=*/0);

  // Destination pipeline: first non-sink operator is the consumer.
  sirius_physical_operator dest_first_op;
  sirius_physical_operator dest_sink;
  auto dest_pipeline = build_pipeline(env, &dest_sink, {&dest_first_op}, /*pipeline_id=*/1);

  std::vector<repository_wiring> wirings = {{std::string_view{"default"},
                                             MemoryBarrierType::FULL,
                                             &source_sink,
                                             src_pipeline,
                                             dest_pipeline}};

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

  sirius_physical_operator source_sink;
  auto src_pipeline = build_pipeline(env, &source_sink, {}, /*pipeline_id=*/0);

  // Destination pipeline has no intermediate operators; port should go on the sink.
  sirius_physical_operator dest_sink;
  auto dest_pipeline = build_pipeline(env, &dest_sink, /*operators=*/{}, /*pipeline_id=*/1);

  std::vector<repository_wiring> wirings = {{std::string_view{"scan"},
                                             MemoryBarrierType::PIPELINE,
                                             &source_sink,
                                             src_pipeline,
                                             dest_pipeline}};

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

  // Pipeline whose sink is a delim-join-style composite — the actual emitter is a
  // sub-operator, not the pipeline's sink. The descriptor carries the explicit
  // source_op, so add_next_port_after_sink should land on it instead of on the sink.
  sirius_physical_operator pipeline_sink;
  sirius_physical_operator sub_emitter;
  auto src_pipeline = build_pipeline(env, &pipeline_sink, {}, /*pipeline_id=*/0);

  sirius_physical_operator dest_first_op;
  sirius_physical_operator dest_sink;
  auto dest_pipeline = build_pipeline(env, &dest_sink, {&dest_first_op}, /*pipeline_id=*/1);

  std::vector<repository_wiring> wirings = {{std::string_view{"default"},
                                             MemoryBarrierType::FULL,
                                             &sub_emitter,  // explicit source op
                                             src_pipeline,
                                             dest_pipeline}};

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

  sirius_physical_operator probe_sink;
  auto probe_pipeline = build_pipeline(env, &probe_sink, {}, /*pipeline_id=*/0);

  sirius_physical_operator build_sink;
  auto build_pipeline_handle = build_pipeline(env, &build_sink, {}, /*pipeline_id=*/1);

  // Hash-join-like destination: receives "default" from probe and "build" from build side.
  sirius_physical_operator hash_join_op;
  sirius_physical_operator dest_sink;
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

TEST_CASE("materialize: delim join destinations use the generic path with no sibling side effects",
          "[repository_wiring][materializer]")
{
  // A delim join is a fan-out source, not a wiring destination; the materializer does not
  // special-case RIGHT/LEFT delim joins: no extra port is grafted onto a partition_join sibling,
  // and a LEFT delim join does not throw. If a delim join ever appears as a destination's first
  // operator, it is wired exactly like any other operator.
  wiring_test_env env;
  cucascade::shared_data_repository_manager mgr;
  duckdb::vector<sirius::logical_type> empty_types;

  sirius_physical_operator source_sink;
  auto src_pipeline = build_pipeline(env, &source_sink, {}, /*pipeline_id=*/0);

  auto dummy_join = duckdb::make_uniq<sirius_physical_operator>(SiriusPhysicalOperatorType::INVALID,
                                                                empty_types,
                                                                /*estimated_cardinality=*/0);
  dummy_join->children.push_back(duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::INVALID, empty_types, 0));
  dummy_join->children.push_back(duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::INVALID, empty_types, 0));

  auto right_delim = duckdb::make_uniq<sirius_physical_right_delim_join>(
    empty_types,
    std::move(dummy_join),
    duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
    /*estimated_cardinality=*/0,
    duckdb::optional_idx());

  // partition_join target — its `type` field is read by sirius_physical_partition's
  // ctor; NESTED_LOOP_JOIN takes a no-op branch.
  auto fake_parent = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, empty_types, 0);
  auto partition_join =
    duckdb::make_uniq<sirius_physical_partition>(empty_types, 0, fake_parent.get());
  right_delim->partition_join = partition_join.get();

  sirius_physical_operator unused_sink;
  auto dest_pipeline = build_pipeline(env, &unused_sink, {right_delim.get()}, /*pipeline_id=*/1);

  std::vector<repository_wiring> wirings = {{std::string_view{"build"},
                                             MemoryBarrierType::FULL,
                                             &source_sink,
                                             src_pipeline,
                                             dest_pipeline}};

  // Must not throw (the LEFT-delim "should never be a source" invariant is gone), and the port
  // lands only on the destination's operators[0].
  REQUIRE_NOTHROW(materialize_repository_wiring(wirings, mgr));

  auto* delim_port = right_delim->get_port("build");
  REQUIRE(delim_port != nullptr);
  CHECK(delim_port->repo != nullptr);

  // No sibling port is grafted onto partition_join.
  CHECK(partition_join->get_port_ids().empty());
}
