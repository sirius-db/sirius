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

#include "op/sirius_physical_partition.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "utils/test_physical_operator.hpp"

#include <catch.hpp>

#include <atomic>
#include <future>

using namespace sirius::op;

namespace {
auto one_column()
{
  return duckdb::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER)};
}

void attach(sirius_physical_operator& op, cucascade::shared_data_repository& repo)
{
  auto port  = std::make_unique<sirius_physical_operator::port>();
  port->repo = &repo;
  port->type = MemoryBarrierType::FULL;
  op.add_port("default", std::move(port));
}

/// A source pipeline that is not finished until the test says so.
class stub_pipeline : public sirius::pipeline::sirius_pipeline {
 public:
  using sirius_pipeline::sirius_pipeline;
  bool is_pipeline_finished() const override { return finished; }

  bool finished = false;
};
}  // namespace

TEST_CASE("partition output metadata preserves sparse destinations through reconstruction",
          "[partition_exchange][unit]")
{
  // Metadata only: null placeholders are never prepared or executed as batches.
  routed_operator_data output({nullptr, nullptr}, {1, 4});
  output.set_preferred_device_id(5);
  auto reconstructed = output.with_batches_preserving_rows({nullptr, nullptr});
  auto const* routed = dynamic_cast<routed_operator_data const*>(reconstructed.get());
  REQUIRE(routed);
  CHECK(routed->destinations() == std::vector<std::size_t>{1, 4});
  CHECK(routed->get_preferred_device_id() == 5);
  CHECK_THROWS_AS(output.with_batches_preserving_rows({nullptr}), std::invalid_argument);
}

TEST_CASE("concurrent partition requests share one layout decision", "[partition_exchange][unit]")
{
  sirius_physical_partition left(one_column(), 0, partition_spec{{0}});
  sirius_physical_partition right(one_column(), 0, partition_spec{{0}});
  sirius::op::test_physical_operator target(
    SiriusPhysicalOperatorType::PROJECTION, one_column(), 0);
  cucascade::shared_data_repository left_repo, right_repo, output_repo;
  attach(left, left_repo);
  attach(right, right_repo);
  attach(target, output_repo);
  left.add_next_port_after_sink({&target, "default"});
  right.add_next_port_after_sink({&target, "default"});
  std::atomic<int> decisions{0};
  partition_exchange::bind({&left, &right}, 1, false, [&](auto const&, bool) {
    ++decisions;
    return std::vector<partition_layout>(2,
                                         {partition_mode::hash, partition_placement::unpinned(4)});
  });
  auto first  = std::async(std::launch::async, [&] { return left.get_next_task_input_data(); });
  auto second = std::async(std::launch::async, [&] { return right.get_next_task_input_data(); });
  CHECK_FALSE(first.get());
  CHECK_FALSE(second.get());
  CHECK(decisions == 1);
  REQUIRE(left.layout());
  REQUIRE(right.layout());
  CHECK(*left.layout()->placement == *right.layout()->placement);
  CHECK(output_repo.num_partitions() == 4);
}

TEST_CASE("a sizing input that outgrows a deferred scan stops being held",
          "[partition_exchange][unit]")
{
  sirius_physical_partition build(one_column(), 0, partition_spec{{0}});
  sirius_physical_partition probe(one_column(), 0, partition_spec{{0}});
  sirius::op::test_physical_operator target(
    SiriusPhysicalOperatorType::PROJECTION, one_column(), 0);
  sirius::op::test_physical_operator producer(
    SiriusPhysicalOperatorType::PROJECTION, one_column(), 0);
  cucascade::shared_data_repository build_repo, probe_repo, output_repo;
  attach(build, build_repo);
  attach(probe, probe_repo);
  attach(target, output_repo);
  build.add_next_port_after_sink({&target, "default"});
  probe.add_next_port_after_sink({&target, "default"});
  sirius::pipeline::pipeline_build_context context{nullptr};
  auto source = std::make_shared<stub_pipeline>(context);
  sirius::pipeline::sirius_pipeline_build_state state;
  state.add_pipeline_operator(*source, producer);
  build.get_port("default")->src_pipeline = source;

  std::vector<partition_input_stats> sized_from;
  partition_exchange::bind({&build, &probe}, 0, false, [&](auto const& stats, bool) {
    sized_from = stats;
    return std::vector<partition_layout>(2,
                                         {partition_mode::hash, partition_placement::unpinned(4)});
  });

  SECTION("a sizing input smaller than the scan is still collected whole")
  {
    build.exchange()->defer_scan(probe, 1);
    CHECK_FALSE(build.get_next_task_input_data());
    CHECK_FALSE(build.layout());
    CHECK_FALSE(build.exchange()->scan_holds());
  }

  SECTION("a sizing input as large as the scan gives way to it")
  {
    build.exchange()->defer_scan(probe, 0);
    CHECK_FALSE(build.get_next_task_input_data());
    REQUIRE(build.layout());
    REQUIRE(probe.layout());
    CHECK(build.exchange()->scan_holds());
    REQUIRE(sized_from.size() == 2);
    CHECK(sized_from[0].from_deferred_scan);
    CHECK_FALSE(sized_from[0].measured_rows);
    // Its barrier no longer holds it: it asks for its producer's next batch.
    auto const hint = build.get_next_task_hint();
    REQUIRE(hint);
    CHECK(hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA);
    CHECK(hint->producer == &producer);
  }

  SECTION("a complete sizing input sizes the layout itself")
  {
    source->finished = true;
    build.exchange()->defer_scan(probe, 0);
    CHECK_FALSE(build.get_next_task_input_data());
    REQUIRE(build.layout());
    CHECK_FALSE(build.exchange()->scan_holds());
    REQUIRE(sized_from.size() == 2);
    CHECK_FALSE(sized_from[0].from_deferred_scan);
  }

  SECTION("the sizing input cannot be the deferred scan")
  {
    build.exchange()->defer_scan(build, 0);
    CHECK_FALSE(build.get_next_task_input_data());
    CHECK_FALSE(build.layout());
  }
}
