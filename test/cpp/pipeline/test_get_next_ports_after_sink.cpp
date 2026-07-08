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
#include <duckdb.hpp>
#include <expression/ast/node.hpp>
#include <helper/logical_type.hpp>
#include <op/sirius_physical_column_data_scan.hpp>
#include <op/sirius_physical_delim_join.hpp>
#include <op/sirius_physical_dummy_scan.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <op/sirius_physical_partition.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <memory>
#include <utility>
#include <vector>

using sirius::op::sirius_physical_column_data_scan;
using sirius::op::sirius_physical_dummy_scan;
using sirius::op::sirius_physical_grouped_aggregate;
using sirius::op::sirius_physical_left_delim_join;
using sirius::op::sirius_physical_operator;
using sirius::op::sirius_physical_partition;
using sirius::op::sirius_physical_right_delim_join;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;

namespace {

duckdb::unique_ptr<sirius_physical_operator> make_dummy_two_child_join()
{
  duckdb::vector<sirius::logical_type> empty_types;
  auto join = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::INVALID, empty_types, 0);
  join->children.push_back(duckdb::make_uniq<sirius_physical_dummy_scan>(empty_types, 0));
  join->children.push_back(duckdb::make_uniq<sirius_physical_dummy_scan>(empty_types, 0));
  return join;
}

}  // namespace

TEST_CASE("yields nothing when no sink is set", "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});
  REQUIRE(not pipeline.get_sink());

  for (const auto& port : pipeline.get_next_ports_after_sink()) {
    FAIL("port found");
  }
}

TEST_CASE("returns sink ports for standard (non-delim) sink",
          "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});

  sirius_physical_operator sink_op;
  sirius_physical_operator consumer_a;
  sirius_physical_operator consumer_b;
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

TEST_CASE("yields nothing for standard sink with no ports", "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});

  sirius_physical_operator sink_op;
  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_sink(pipeline, &sink_op, 0);

  for (const auto& port : pipeline.get_next_ports_after_sink()) {
    FAIL("port found");
  }
}

TEST_CASE("RIGHT_DELIM_JOIN: concatenates partition_join then distinct ports",
          "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});
  duckdb::vector<sirius::logical_type> empty_types;

  auto right_delim = duckdb::make_uniq<sirius_physical_right_delim_join>(
    empty_types,
    make_dummy_two_child_join(),
    duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
    0,
    duckdb::optional_idx());

  // sirius_physical_partition needs a parent_op whose `type` field is one of the supported
  // values. NESTED_LOOP_JOIN takes a no-op branch and avoids a downcast, so a standard
  // sirius_physical_operator with that type is enough.
  auto fake_parent = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, empty_types, 0);
  auto partition_join =
    duckdb::make_uniq<sirius_physical_partition>(empty_types, 0, fake_parent.get());

  auto distinct = duckdb::make_uniq<sirius_physical_grouped_aggregate>(
    empty_types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>>{},
    duckdb::vector<std::unique_ptr<sirius::ast::node>>{},
    0);

  sirius_physical_operator partition_consumer_x;
  sirius_physical_operator partition_consumer_y;
  sirius_physical_operator distinct_consumer;
  partition_join->add_next_port_after_sink({&partition_consumer_x, "port_x"});
  partition_join->add_next_port_after_sink({&partition_consumer_y, "port_y"});
  distinct->add_next_port_after_sink({&distinct_consumer, "distinct_port"});

  right_delim->partition_join = partition_join.get();
  right_delim->distinct       = std::move(distinct);

  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_sink(pipeline, right_delim.get(), 0);

  std::vector<sirius_physical_operator::next_port_info> ports;
  for (const auto& port : pipeline.get_next_ports_after_sink()) {
    ports.push_back(port);
  }
  REQUIRE(ports.size() == 3);
  CHECK(ports[0].next_operator == &partition_consumer_x);
  CHECK(ports[0].next_operator_port_name == "port_x");
  CHECK(ports[1].next_operator == &partition_consumer_y);
  CHECK(ports[1].next_operator_port_name == "port_y");
  CHECK(ports[2].next_operator == &distinct_consumer);
  CHECK(ports[2].next_operator_port_name == "distinct_port");
}

TEST_CASE("LEFT_DELIM_JOIN: concatenates column_data_scan then distinct ports",
          "[pipeline][get_next_ports_after_sink]")
{
  sirius_pipeline pipeline(sirius::pipeline::pipeline_build_context{nullptr});
  duckdb::vector<sirius::logical_type> empty_types;

  auto left_delim = duckdb::make_uniq<sirius_physical_left_delim_join>(
    empty_types,
    make_dummy_two_child_join(),
    duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
    0,
    duckdb::optional_idx());

  auto column_data_scan = duckdb::make_uniq<sirius_physical_column_data_scan>(
    empty_types, SiriusPhysicalOperatorType::COLUMN_DATA_SCAN, 0, static_cast<std::size_t>(0));

  auto distinct = duckdb::make_uniq<sirius_physical_grouped_aggregate>(
    empty_types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>>{},
    duckdb::vector<std::unique_ptr<sirius::ast::node>>{},
    0);

  sirius_physical_operator scan_consumer;
  sirius_physical_operator distinct_consumer_p;
  sirius_physical_operator distinct_consumer_q;
  column_data_scan->add_next_port_after_sink({&scan_consumer, "scan_port"});
  distinct->add_next_port_after_sink({&distinct_consumer_p, "distinct_p"});
  distinct->add_next_port_after_sink({&distinct_consumer_q, "distinct_q"});

  left_delim->column_data_scan = column_data_scan.get();
  left_delim->distinct         = std::move(distinct);

  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_sink(pipeline, left_delim.get(), 0);

  std::vector<sirius_physical_operator::next_port_info> ports;
  for (const auto& port : pipeline.get_next_ports_after_sink()) {
    ports.push_back(port);
  }
  REQUIRE(ports.size() == 3);
  CHECK(ports[0].next_operator == &scan_consumer);
  CHECK(ports[0].next_operator_port_name == "scan_port");
  CHECK(ports[1].next_operator == &distinct_consumer_p);
  CHECK(ports[1].next_operator_port_name == "distinct_p");
  CHECK(ports[2].next_operator == &distinct_consumer_q);
  CHECK(ports[2].next_operator_port_name == "distinct_q");
}
