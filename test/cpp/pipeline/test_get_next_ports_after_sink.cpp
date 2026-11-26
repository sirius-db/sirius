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
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <vector>

using sirius::op::sirius_physical_operator;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;

TEST_CASE("yields nothing when no sink is set", "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});
  REQUIRE(not pipeline.get_sink());

  for ([[maybe_unused]] const auto& port : pipeline.get_next_ports_after_sink()) {
    FAIL("port found");
  }
}

TEST_CASE("returns the sink's ports", "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});

  sirius::op::test_physical_operator sink_op;
  sirius::op::test_physical_operator consumer_a;
  sirius::op::test_physical_operator consumer_b;
  sink_op.add_next_port_after_sink({&consumer_a, "port_a"});
  sink_op.add_next_port_after_sink({&consumer_b, "port_b"});

  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_sink(pipeline, &sink_op, 0);

  std::vector<sirius_physical_operator::next_port_info> ports;
  for (const auto& port : pipeline.get_next_ports_after_sink()) {
    ports.push_back(port);
  }
  REQUIRE(ports.size() == 2);
  CHECK(ports[0].next_operator == &consumer_a);
  CHECK(ports[0].next_operator_port_name == "port_a");
  CHECK(ports[1].next_operator == &consumer_b);
  CHECK(ports[1].next_operator_port_name == "port_b");
}

TEST_CASE("yields nothing for a sink with no ports", "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});

  sirius::op::test_physical_operator sink_op;
  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_sink(pipeline, &sink_op, 0);

  for ([[maybe_unused]] const auto& port : pipeline.get_next_ports_after_sink()) {
    FAIL("port found");
  }
}
