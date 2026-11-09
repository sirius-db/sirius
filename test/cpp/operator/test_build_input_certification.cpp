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

/*
 * Host-only tests of `certify_decision`, the decision the build PARTITION and BUILD_FILTER share
 * before certifying their complete build input.
 */

#include "op/dynamic_filter/build_input_certification.hpp"
#include "op/sirius_physical_operator.hpp"

#include <catch.hpp>

#include <variant>

namespace {

namespace op = sirius::op;
using op::construction_decline;
using op::MemoryBarrierType;

/// A single-partition `default` FULL input whose finished source queued three batches.
constexpr op::build_port_facts k_ready{.data_ports             = 1,
                                       .single_port_is_default = true,
                                       .repository_partitions  = 1,
                                       .barrier                = MemoryBarrierType::FULL,
                                       .has_source_pipeline    = true,
                                       .source_finished        = true,
                                       .repository_batches     = 3,
                                       .has_port_directive     = false};

std::optional<construction_decline> decline_of(op::build_port_facts const& facts)
{
  auto const decision = op::certify_decision(facts);
  if (auto const* decline = std::get_if<construction_decline>(&decision)) { return *decline; }
  return std::nullopt;
}

}  // namespace

TEST_CASE("a finished single default FULL input proceeds to the ledger",
          "[dynamic_filter][certification]")
{
  CHECK(std::holds_alternative<op::certification_proceed>(op::certify_decision(k_ready)));
  auto no_batches               = k_ready;
  no_batches.repository_batches = 0;
  CHECK(std::holds_alternative<op::certification_proceed>(op::certify_decision(no_batches)));
}

TEST_CASE("an unfinished FULL source pends", "[dynamic_filter][certification]")
{
  auto unfinished            = k_ready;
  unfinished.source_finished = false;
  CHECK(std::holds_alternative<op::certification_pending>(op::certify_decision(unfinished)));
}

TEST_CASE("an input of the wrong shape declines at once", "[dynamic_filter][certification]")
{
  auto facts            = k_ready;
  facts.source_finished = false;  // A decline never waits for the source.
  SECTION("no data port") { facts.data_ports = 0; }
  SECTION("two data ports") { facts.data_ports = 2; }
  SECTION("a data port other than default") { facts.single_port_is_default = false; }
  SECTION("a partitioned repository") { facts.repository_partitions = 2; }
  SECTION("a PARTIAL barrier") { facts.barrier = MemoryBarrierType::PARTIAL; }
  SECTION("no source pipeline") { facts.has_source_pipeline = false; }
  CHECK(decline_of(facts) == construction_decline::input_shape);
}

TEST_CASE("a PIPELINE barrier declines before the source finishes",
          "[dynamic_filter][certification]")
{
  auto facts            = k_ready;
  facts.barrier         = MemoryBarrierType::PIPELINE;
  facts.source_finished = false;
  CHECK(decline_of(facts) == construction_decline::pipeline_barrier);
}

TEST_CASE("a late-materialization directive declines before the source finishes",
          "[dynamic_filter][certification]")
{
  auto facts               = k_ready;
  facts.has_port_directive = true;
  facts.source_finished    = false;
  CHECK(decline_of(facts) == construction_decline::port_directive);
}

TEST_CASE("decline checks run in a fixed order", "[dynamic_filter][certification]")
{
  auto facts               = k_ready;
  facts.has_port_directive = true;
  facts.barrier            = MemoryBarrierType::PIPELINE;
  CHECK(decline_of(facts) == construction_decline::pipeline_barrier);
  facts.data_ports = 2;
  CHECK(decline_of(facts) == construction_decline::input_shape);
}

TEST_CASE("every decline reason has a stable spelling", "[dynamic_filter][certification]")
{
  CHECK(op::to_string(construction_decline::input_shape) == "input_shape");
  CHECK(op::to_string(construction_decline::pipeline_barrier) == "pipeline_barrier");
  CHECK(op::to_string(construction_decline::port_directive) == "port_directive");
  CHECK(op::to_string(construction_decline::inventory_mismatch) == "inventory_mismatch");
  CHECK(op::to_string(construction_decline::unreadable_arrival) == "unreadable_arrival");
  CHECK(op::to_string(construction_decline::compressed_arrival) == "compressed_arrival");
}
