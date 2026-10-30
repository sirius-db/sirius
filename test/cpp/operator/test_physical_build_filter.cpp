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

#include "dynamic_filter_accumulation_test_utils.hpp"
#include "op/dynamic_filter/sirius_physical_build_filter.hpp"
#include "partitioned_join_test_utils.hpp"
#include "planner/late_mat_plan_pass.hpp"

#include <catch.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace {

namespace acc  = sirius::test::accumulation;
namespace op   = sirius::op;
namespace util = sirius::test::operator_utils;

duckdb::vector<sirius::logical_type> build_types()
{
  return sirius::from_duckdb_vec(
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER, duckdb::LogicalType::BIGINT});
}

struct build_filter_fixture {
  acc::fixture gpu;
  op::sirius_physical_operator upstream{
    op::SiriusPhysicalOperatorType::PROJECTION, build_types(), 100};
  std::shared_ptr<util::controllable_pipeline> source =
    std::make_shared<util::controllable_pipeline>(false);
  cucascade::shared_data_repository input_repo;
  std::unique_ptr<op::sirius_physical_build_filter> filter;

  explicit build_filter_fixture(std::uint64_t cap = 256 * acc::mib)
  {
    gpu.session =
      std::make_unique<op::dynamic_filter_publication_session>(gpu.make_plan(cap), &gpu.stats);
    filter = std::make_unique<op::sirius_physical_build_filter>(build_types(), 100, *gpu.session);
    sirius::pipeline::sirius_pipeline_build_state state;
    state.add_pipeline_operator(*source, upstream);
    auto input          = std::make_unique<op::sirius_physical_operator::port>();
    input->type         = op::MemoryBarrierType::FULL;
    input->repo         = &input_repo;
    input->src_pipeline = source;
    filter->add_port("default", std::move(input));
  }

  acc::batch_ptr push(std::int64_t start, cudf::size_type rows = 32)
  {
    auto batch = gpu.make_batch(0, start, rows);
    filter->push_data_batch("default", batch);
    return batch;
  }

  void execute(op::operator_data const& input, acc::batch_ptr const& expected)
  {
    auto const stream = gpu.task_stream(0);
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
    acc::task_reservation task{gpu.gpu(0), 16 * acc::mib, stream};
    auto output = filter->execute(input, stream);
    stream.sync();
    auto const* batches = dynamic_cast<op::pipelineable_operator_data*>(output.get());
    REQUIRE(batches);
    REQUIRE(batches->get_data_batches().size() == 1);
    REQUIRE(batches->get_data_batches().front() == expected);
  }
};

void install_stage_materialization(build_filter_fixture& fixture)
{
  auto const pin    = std::make_shared<sirius::late_mat::pin_entry_handle>("build_filter_pin", 1);
  auto const origin = [&pin](std::uint32_t position) {
    return sirius::late_mat::column_origin{
      .handle = pin, .column_pos = position, .generation = pin->generation()};
  };
  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64}};
  auto pair =
    sirius::late_mat::make_defer_pair(schema, {0, 1}, schema, {0, 1}, {origin(0), origin(1)});
  REQUIRE(pair.valid());
  REQUIRE(sirius::planner::install_deferral(fixture.upstream, *fixture.filter, std::move(pair)));
}

}  // namespace

TEST_CASE("build filter publishes once after every complete-input batch contributes",
          "[dynamic_filter][build_filter]")
{
  build_filter_fixture fixture;
  std::vector<acc::batch_ptr> batches;
  for (int index = 0; index < 7; ++index) {
    batches.push_back(fixture.push(index * 100));
  }
  auto const before = fixture.gpu.allocated_bytes();
  auto hint         = fixture.filter->get_next_task_hint();
  REQUIRE(hint);
  REQUIRE(hint->hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(hint->producer == &fixture.upstream);
  REQUIRE_FALSE(fixture.filter->get_next_task_input_data());
  REQUIRE(fixture.input_repo.total_size() == batches.size());
  REQUIRE(fixture.gpu.allocated_bytes() == before);
  REQUIRE(fixture.gpu.stats.snapshot().publication_attempts == 0);

  fixture.source->finished = true;
  for (std::size_t index = 0; index < batches.size(); ++index) {
    auto input = fixture.filter->get_next_task_input_data();
    REQUIRE(input);
    fixture.execute(*input, batches[index]);
    if (index == 0) { fixture.execute(*input, batches[index]); }
    REQUIRE(fixture.gpu.channel->filter_count() == (index + 1 == batches.size() ? 2 : 0));
  }
  REQUIRE(fixture.input_repo.total_size() == 0);
  auto const stats = fixture.gpu.stats.snapshot();
  REQUIRE(stats.publication_attempts == 1);
  REQUIRE(stats.accumulation_expected_contributions == batches.size());
  REQUIRE(stats.accumulation_completed_contributions == batches.size());
  REQUIRE(stats.accumulation_duplicate_contributions == 1);
  REQUIRE(stats.accumulation_publications_finished == 1);
  for (auto const& batch : batches) {
    REQUIRE(fixture.gpu.possible_rows(0, batch, 0) == 32);
    REQUIRE(fixture.gpu.possible_rows(1, batch, 0) == 32);
  }

  fixture.filter->finalize_operator();
  REQUIRE(fixture.gpu.channel->snapshot().terminal());
  REQUIRE_FALSE(fixture.gpu.session->try_begin_accumulation(acc::fixture::certify(batches)));
  REQUIRE(fixture.gpu.channel->filter_count() == 2);
}

TEST_CASE("declined build filter forwards input and preserves the later publication path",
          "[dynamic_filter][build_filter]")
{
  build_filter_fixture fixture;
  auto const batch         = fixture.push(0);
  fixture.source->finished = true;

  SECTION("the input barrier cannot certify completeness")
  {
    fixture.filter->get_port("default")->type = op::MemoryBarrierType::PARTIAL;
  }
  SECTION("the arrival ledger does not cover every input batch")
  {
    fixture.input_repo.add_data_batch(fixture.gpu.make_batch(0, 100, 32));
  }
  SECTION("the input has no producer pipeline")
  {
    fixture.filter->get_port("default")->src_pipeline.reset();
  }
  SECTION("the stage must materialize deferred columns") { install_stage_materialization(fixture); }

  auto input = fixture.filter->get_next_task_input_data();
  REQUIRE(input);
  fixture.execute(*input, batch);
  REQUIRE_FALSE(fixture.gpu.session->accumulation_claimed());
  REQUIRE(fixture.gpu.channel->filter_count() == 0);
  REQUIRE(fixture.gpu.stats.snapshot().accumulations_skipped_inventory == 1);
  fixture.filter->finalize_operator();
  REQUIRE_FALSE(fixture.gpu.channel->snapshot().terminal());
  fixture.gpu.session->finish_input();
  REQUIRE(fixture.gpu.channel->snapshot().terminal());
}

TEST_CASE("build filter cancellation and allocation refusal preserve batches",
          "[dynamic_filter][build_filter]")
{
  auto const refuse_allocation = GENERATE(false, true);
  build_filter_fixture fixture(refuse_allocation ? 0 : 256 * acc::mib);
  auto const first         = fixture.push(0);
  auto const second        = fixture.push(100);
  fixture.source->finished = true;

  auto first_input = fixture.filter->get_next_task_input_data();
  REQUIRE(first_input);
  fixture.execute(*first_input, first);
  if (!refuse_allocation) { fixture.gpu.session->cancel(); }
  auto second_input = fixture.filter->get_next_task_input_data();
  REQUIRE(second_input);
  fixture.execute(*second_input, second);
  REQUIRE(fixture.gpu.channel->filter_count() == 0);
  REQUIRE(fixture.input_repo.total_size() == 0);
  fixture.filter->finalize_operator();
  REQUIRE(fixture.gpu.channel->snapshot().terminal() == !refuse_allocation);
}

TEST_CASE("a build filter budgets scratch rather than its resident input",
          "[dynamic_filter][build_filter][memory_estimate]")
{
  build_filter_fixture fixture;
  constexpr std::size_t input_bytes = 256 * acc::mib;
  auto const bound = fixture.filter->no_history_peak_memory_estimate({1, input_bytes});
  REQUIRE(bound > 0);
  REQUIRE(bound < acc::mib);
  auto const first         = fixture.push(0);
  fixture.source->finished = true;
  auto input               = fixture.filter->get_next_task_input_data();
  REQUIRE(input);
  auto const stream = fixture.gpu.task_stream(0);
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
    acc::task_reservation task{fixture.gpu.gpu(0), bound, stream};
    auto const peak = fixture.gpu.allocator(0).get_peak_allocated_bytes(stream);
    auto output     = fixture.filter->execute(*input, stream);
    stream.sync();
    REQUIRE(output);
    REQUIRE(fixture.gpu.allocator(0).get_peak_allocated_bytes(stream) <= peak + bound);
  }
  REQUIRE(fixture.gpu.channel->filter_count() == 2);
  REQUIRE(fixture.gpu.possible_rows(0, first, 0) == 32);

  install_stage_materialization(fixture);
  REQUIRE(fixture.filter->no_history_peak_memory_estimate({1, input_bytes}) == 2 * input_bytes);
}

TEST_CASE("a build filter scratch bound covers peer reduction",
          "[dynamic_filter][build_filter][memory_estimate][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  acc::fixture fixture({.devices = 2});
  if (!acc::has_pool_access(fixture.spaces)) { return; }
  fixture.session = std::make_unique<op::dynamic_filter_publication_session>(
    fixture.make_plan(256 * acc::mib), &fixture.stats);
  op::sirius_physical_build_filter filter{build_types(), 100, *fixture.session};
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(1, 1000, 300);
  REQUIRE(fixture.session->try_begin_accumulation(acc::fixture::certify({first, second})));
  fixture.contribute(first);
  fixture.contribute(second);

  auto const bound = filter.no_history_peak_memory_estimate({1, 256 * acc::mib});
  REQUIRE(bound == 16 * acc::mib);
  auto const stream = fixture.task_stream(1);
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{1}};
    acc::task_reservation task{fixture.gpu(1), bound, stream};
    auto const peak = fixture.allocator(1).get_peak_allocated_bytes(stream);
    fixture.session->publish_if_final(second->get_batch_id(), fixture.gpu(1), stream);
    stream.sync();
    REQUIRE(fixture.allocator(1).get_peak_allocated_bytes(stream) <= peak + bound);
  }
  REQUIRE(fixture.channel->filter_count() == 2);
  REQUIRE(fixture.possible_rows(0, first, 0) == 300);
  REQUIRE(fixture.possible_rows(0, second, 1) == 300);
}

// The probe partition sizes a right-family join. Give it an empty, finished input.
void add_finished_probe_input(op::sirius_physical_partition& probe,
                              cucascade::shared_data_repository& repository)
{
  auto input          = std::make_unique<op::sirius_physical_operator::port>();
  input->type         = op::MemoryBarrierType::FULL;
  input->repo         = &repository;
  input->src_pipeline = std::make_shared<util::controllable_pipeline>(true);
  probe.add_port("default", std::move(input));
}

TEST_CASE("right-family build partition finishes filter publication before requesting probe sizing",
          "[dynamic_filter][build_filter][scheduling]")
{
  auto tree   = util::make_partitioned_join({.join_type = duckdb::JoinType::RIGHT_SEMI});
  auto& build = *tree.build_partition;
  // No pipelines exist here, so number the operators the exchange reads.
  tree.join->operator_id            = 0;
  build.operator_id                 = 1;
  tree.probe_partition->operator_id = 2;
  cucascade::shared_data_repository probe_repository;
  add_finished_probe_input(*tree.probe_partition, probe_repository);
  auto publication = duckdb::make_uniq<op::sirius_physical_build_filter>(
    build.types, 100, tree.join->dynamic_filter_session());
  auto* const publication_ptr = publication.get();
  publication->children.push_back(std::move(build.children.front()));
  build.children.front() = std::move(publication);

  auto source = std::make_shared<util::controllable_pipeline>(false);
  sirius::pipeline::sirius_pipeline_build_state state;
  state.add_pipeline_operator(*source, *publication_ptr);
  cucascade::shared_data_repository repository;
  auto input          = std::make_unique<op::sirius_physical_operator::port>();
  input->type         = op::MemoryBarrierType::FULL;
  input->repo         = &repository;
  input->src_pipeline = source;
  build.add_port("default", std::move(input));

  auto hint = build.get_next_task_hint();
  REQUIRE(hint);
  REQUIRE(hint->hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(hint->producer == publication_ptr);
  source->finished = true;
  REQUIRE_FALSE(build.get_next_task_hint());
}

TEST_CASE("an empty build filter pipeline finishes without tasks and releases its partition",
          "[dynamic_filter][build_filter][scheduling]")
{
  auto const join_type =
    GENERATE(duckdb::JoinType::RIGHT, duckdb::JoinType::RIGHT_SEMI, duckdb::JoinType::RIGHT_ANTI);
  auto tree   = util::make_partitioned_join({.join_type = join_type});
  auto& build = *tree.build_partition;
  // No pipelines exist here, so number the operators the exchange reads.
  tree.join->operator_id            = 0;
  build.operator_id                 = 1;
  tree.probe_partition->operator_id = 2;
  cucascade::shared_data_repository probe_repository;
  add_finished_probe_input(*tree.probe_partition, probe_repository);
  auto publication = duckdb::make_uniq<op::sirius_physical_build_filter>(
    build.types, 0, tree.join->dynamic_filter_session());
  auto* const publication_ptr = publication.get();
  publication->children.push_back(std::move(build.children.front()));
  build.children.front() = std::move(publication);

  sirius::pipeline::sirius_pipeline_build_state state;
  auto upstream = std::make_shared<util::controllable_pipeline>(false);
  state.add_pipeline_operator(*upstream, *publication_ptr->children.front());
  cucascade::shared_data_repository upstream_repository;
  auto stage_input          = std::make_unique<op::sirius_physical_operator::port>();
  stage_input->type         = op::MemoryBarrierType::FULL;
  stage_input->repo         = &upstream_repository;
  stage_input->src_pipeline = upstream;
  publication_ptr->add_port("default", std::move(stage_input));

  auto publication_pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
    sirius::pipeline::pipeline_build_context{nullptr});
  state.set_pipeline_source(*publication_pipeline, *publication_ptr);
  state.set_pipeline_sink(*publication_pipeline, publication_ptr, 1);
  state.add_pipeline_operator(*publication_pipeline, *publication_ptr);
  publication_ptr->set_pipeline(publication_pipeline);
  cucascade::shared_data_repository partition_repository;
  auto partition_input          = std::make_unique<op::sirius_physical_operator::port>();
  partition_input->type         = op::MemoryBarrierType::FULL;
  partition_input->repo         = &partition_repository;
  partition_input->src_pipeline = publication_pipeline;
  build.add_port("default", std::move(partition_input));

  publication_pipeline->update_pipeline_status(false);
  REQUIRE_FALSE(publication_pipeline->is_pipeline_finished());
  auto hint = build.get_next_task_hint();
  REQUIRE(hint);
  REQUIRE(hint->producer == publication_ptr);
  auto upstream_hint = publication_ptr->get_next_task_hint();
  REQUIRE(upstream_hint);
  REQUIRE(upstream_hint->producer == publication_ptr->children.front().get());

  upstream->finished = true;
  REQUIRE_FALSE(publication_ptr->get_next_task_hint());
  // The task creator reevaluates every visited pipeline when a hint walk finds no work.
  publication_pipeline->update_pipeline_status(false);
  REQUIRE(publication_pipeline->is_pipeline_finished());
  REQUIRE(publication_pipeline->get_tasks_created() == 0);
  REQUIRE(publication_pipeline->get_tasks_completed() == 0);
  REQUIRE(publication_ptr->finalized.load());
  REQUIRE_FALSE(tree.join->dynamic_filter_session().accumulation_claimed());
  REQUIRE_FALSE(build.get_next_task_hint());
}
