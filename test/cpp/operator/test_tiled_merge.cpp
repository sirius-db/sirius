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

#include "catch.hpp"
#include "data/convertible_data_batch.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/telemetry_utils.hpp"
#include "utils/test_physical_operator.hpp"

#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/memory/disk_table.hpp>

#include <future>
#include <numeric>
#include <set>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
namespace {
duckdb::vector<duckdb::BoundOrderByNode> ascending()
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(
    duckdb::OrderType::ASCENDING,
    duckdb::OrderByNullType::NULLS_LAST,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0));
  return orders;
}
template <typename Base>
class injected_stage : public Base {
 public:
  using Base::Base;
  bool fail_compute{false};
  bool replace_output{false};
  bool replaced_single{false}, replaced_multiple{false};
  std::size_t attempts{0}, publications{0};
  void notify_task_attempt(operator_data& data, task_attempt_context attempt) override
  {
    ++attempts;
    Base::notify_task_attempt(data, attempt);
  }
  void on_task_output_published(operator_data const& output) override
  {
    ++publications;
    Base::on_task_output_published(output);
  }
  std::unique_ptr<operator_data> execute(operator_data const& input,
                                         ::cuda::stream_ref stream) override
  {
    auto output = Base::execute(input, stream);
    if (std::exchange(fail_compute, false)) {
      throw rmm::out_of_memory("injected before publication");
    }
    if (replace_output) {
      auto const& batches = dynamic_cast<pipelineable_operator_data const&>(*output);
      replaced_single |= batches.get_data_batches().size() == 1;
      replaced_multiple |= batches.get_data_batches().size() > 1;
      std::vector<std::shared_ptr<cucascade::data_batch>> replacements;
      std::vector<cucascade::read_only_data_batch> reads;
      try {
        for (auto const& batch : batches.get_data_batches()) {
          reads.push_back(batch->to_read_only());
          auto const& read = reads.back();
          auto* space      = read.get_memory_space();
          replacements.push_back(sirius::make_data_batch(
            std::make_unique<cudf::table>(
              sirius::get_cudf_table_view(read), stream, space->get_default_allocator()),
            *space,
            stream,
            this->batch_telemetry()));
          REQUIRE(replacements.back() != batch);
        }
        stream.sync();
        if (!replacements.empty()) {
          auto wrong_count = replacements;
          wrong_count.pop_back();
          CHECK_THROWS(batches.with_batches_preserving_rows(std::move(wrong_count)));
        }
        return batches.with_batches_preserving_rows(std::move(replacements));
      } catch (...) {
        cudaStreamSynchronize(stream.get());
        throw;
      }
    }
    return output;
  }
};

struct tiled_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu;
  rmm::cuda_stream stream;
  cucascade::shared_data_repository_manager repositories;
  sirius_physical_order local;
  injected_stage<sirius_physical_sort_normalize> normalize;
  injected_stage<sirius_physical_sort_cuts> cuts;
  injected_stage<sirius_physical_sort_extract> extract;
  injected_stage<sirius_physical_merge_sort> merge;
  std::vector<sirius_physical_sort_stage*> stages;
  sirius::op::test_physical_operator destination{SiriusPhysicalOperatorType::FILTER, {}, 0};
  cucascade::shared_data_repository output;
  cucascade::shared_data_repository* input;
  std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines;
  std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state>> globals;
  uint64_t task_id{0};
  std::size_t retry_floor{0};

  explicit tiled_fixture(std::size_t width = 1, std::size_t gpus = 1)
    : manager(sirius::test::operator_utils::initialize_memory_manager(gpus)),
      gpu(manager->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      local(duckdb::vector<sirius::logical_type>(
              width, sirius::logical_type::make(sirius::type_id::BIGINT)),
            width,
            ascending(),
            {},
            0),
      normalize(&local),
      cuts(&local),
      extract(&local),
      merge(&local),
      stages{&normalize, &cuts, &extract, &merge}
  {
    for (std::size_t i = 0; i < stages.size(); ++i) {
      auto& stage       = *stages[i];
      stage.operator_id = i + 1;
      stage.chunk_bytes = 24 * width;
      repositories.add_new_repository(
        i + 1, "default", std::make_unique<cucascade::shared_data_repository>());
      stage.add_port("default",
                     std::make_unique<sirius_physical_operator::port>(
                       MemoryBarrierType::FULL, repo(i), nullptr, nullptr));
      if (i) { stages[i - 1]->add_next_port_after_sink({&stage, "default"}); }
      auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
        sirius::pipeline::pipeline_build_context{nullptr});
      pipeline->set_pipeline_id(i);
      sirius::pipeline::sirius_pipeline_build_state build;
      build.set_pipeline_source(*pipeline, stage);
      build.add_pipeline_operator(*pipeline, stage);
      build.set_pipeline_sink(*pipeline, &stage, 0);
      globals.push_back(std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
        pipeline, sirius::test::make_test_telemetry_context()));
      pipelines.push_back(std::move(pipeline));
    }
    input = repo(0);
    destination.add_port("default",
                         std::make_unique<sirius_physical_operator::port>(
                           MemoryBarrierType::FULL, &output, nullptr, nullptr));
    merge.add_next_port_after_sink({&destination, "default"});
  }
  cucascade::shared_data_repository* repo(std::size_t stage)
  {
    return repositories.get_repository(stage + 1, "default").get();
  }
  void add(std::vector<int64_t> values, std::size_t range = 0)
  {
    input->add_data_batch(make_numeric_batch<int64_t>(*gpu, values, cudf::type_id::INT64), range);
  }
  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> task(
    std::size_t stage,
    std::unique_ptr<operator_data> data,
    cucascade::memory::memory_space* target = nullptr,
    bool at_ceiling                         = false)
  {
    if (!target) { target = gpu; }
    auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      ++task_id,
      repositories.get_repositories(),
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(data)),
      globals.at(stage));
    auto info = task->get_estimated_reservation_size_info(target);
    info.reservation_size =
      at_ceiling ? target->get_max_memory() : std::max(info.reservation_size, retry_floor);
    auto reservation =
      target->make_reservation_or_null(std::min(info.reservation_size, target->get_max_memory()));
    if (!reservation) {
      // Model downgrade before reservation; ordinary preparation restores only
      // this task's inputs afterward.
      auto candidates = repositories.get_repositories();
      candidates.push_back(&output);
      auto* host = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
      for (auto* repository : candidates) {
        sirius::convertible_data_batch_provider provider(repository);
        for (auto& candidate : provider.get_all_convertible(target, true, false)) {
          REQUIRE(candidate->convert({host}, stream, *manager, true).has_value());
          info = task->get_estimated_reservation_size_info(target);
          info.reservation_size =
            at_ceiling ? target->get_max_memory() : std::max(info.reservation_size, retry_floor);
          reservation = target->make_reservation_or_null(
            std::min(info.reservation_size, target->get_max_memory()));
          if (reservation) { break; }
        }
        if (reservation) { break; }
      }
    }
    REQUIRE(reservation);
    auto* local =
      dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(task->local_state());
    local->set_reservation(std::move(reservation), info);
    task->telemetry_fsm().queued(
      {.queue = {.target = quent::task_queue::TaskQueueId{quent::nil_uuid()},
                 .data   = {.entries = 1}}});
    task->telemetry_fsm().reserving(
      {.requested_bytes      = 0,
       .input_basis          = 0,
       .peak_estimate        = 0,
       .bytes_to_materialize = 0,
       .manager_thread       = {
               .target = quent::task_manager_loop_thread::TaskManagerLoopThreadId{quent::nil_uuid()},
               .data   = {}}});
    return task;
  }
  void drain_stage(std::size_t stage, bool reverse = true)
  {
    std::vector<std::unique_ptr<operator_data>> claimed;
    while (auto data = stages[stage]->get_next_task_input_data()) {
      claimed.push_back(std::move(data));
    }
    if (reverse) { std::reverse(claimed.begin(), claimed.end()); }
    for (auto& data : claimed) {
      auto item = task(stage, std::move(data));
      for (int retry = 0;; ++retry) {
        REQUIRE(retry < 24);
        try {
          item->execute(stream);
          break;
        } catch (sirius::pipeline::oom_reschedule_exception& error) {
          auto input = error.release_intermediate_data();
          dynamic_cast<pipelineable_operator_data&>(*input).remove_read_only_lock();
          retry_floor =
            dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(item->local_state())
              ->get_retry_reservation_floor();
          item.reset();
          item = task(stage, std::move(input));
        }
      }
      item.reset();
      retry_floor = 0;
    }
    CHECK(stages[stage]->all_ports_empty());
    CHECK(repo(stage)->all_empty());
    pipelines[stage]->update_pipeline_status(false);
    CHECK(pipelines[stage]->is_pipeline_finished());
  }
  void drain(std::size_t first = 0)
  {
    for (auto i = first; i < stages.size(); ++i) {
      drain_stage(i);
    }
  }
  std::vector<int64_t> rows()
  {
    std::vector<int64_t> values;
    for (auto position : output.get_positions()) {
      pipelineable_operator_data restored({output.get_data_batch_by_position(position)});
      restored.prepare_for_processing(gpu, stream);
      stream.synchronize();
      auto read  = restored.get_read_only_batches().front();
      auto chunk = copy_column_to_host<int64_t>(sirius::get_cudf_table_view(read).column(0));
      values.insert(values.end(), chunk.begin(), chunk.end());
    }
    return values;
  }
};
}  // namespace

TEST_CASE("sort pipelines preserve range and tile order under reversed completion", "[tiled_merge]")
{
  tiled_fixture f;
  f.input->set_num_partitions(5);
  for (int64_t i = 1; i <= 3; ++i) {
    f.add({i, i + 3, i + 6}, 1);
    f.add({i + 9, i + 12, i + 15}, 3);
  }
  f.add({}, 4);
  f.drain();
  std::vector<int64_t> expected(18);
  std::iota(expected.begin(), expected.end(), 1);
  CHECK(f.rows() == expected);
  CHECK(f.output.get_ordering() == cucascade::data_repository_ordering::ORDERED);
  for (auto position : f.output.get_positions()) {
    CHECK(*sirius::get_batch_rows_and_bytes(*f.output.get_data_batch_by_position(position)).rows <=
          3);
  }
  // Finishing tiles backwards creates batch IDs in the opposite order. Positions,
  // not IDs or insertion order, must reconstruct SQL order.
  auto positions = f.output.get_positions();
  REQUIRE(positions.size() > 1);
  CHECK(f.output.get_data_batch_by_position(positions.front())->get_batch_id() >
        f.output.get_data_batch_by_position(positions.back())->get_batch_id());
}

TEST_CASE("sort pipelines handle empty single and duplicate runs", "[tiled_merge]")
{
  auto count = GENERATE(0, 1, 101);
  tiled_fixture f;
  f.add(std::vector<int64_t>(count, 7));
  f.add(std::vector<int64_t>(count, 7));
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>(2 * count, 7));
}

TEST_CASE("normalization and coalescing freeze runs before cut planning", "[tiled_merge]")
{
  tiled_fixture f;
  f.normalize.chunk_bytes = f.cuts.chunk_bytes = 64;
  for (int64_t i = 0; i < 32; ++i) {
    f.add({i});
  }
  std::vector<int64_t> large(1200);
  std::iota(large.begin(), large.end(), 100);
  f.add(large, 2);
  f.drain_stage(0);
  CHECK(f.repo(1)->get_positions(0).size() == 4);
  for (auto p : {0, 2}) {
    for (auto position : f.repo(1)->get_positions(p)) {
      CHECK(sirius::get_batch_rows_and_bytes(*f.repo(1)->get_data_batch_by_position(position, p))
              .bytes <= 64);
    }
  }
  f.drain(1);
  std::vector<int64_t> expected(32);
  std::iota(expected.begin(), expected.end(), 0);
  expected.insert(expected.end(), large.begin(), large.end());
  CHECK(f.rows() == expected);
}

TEST_CASE("sort metadata survives batch replacement and compute retry in every stage",
          "[tiled_merge]")
{
  tiled_fixture f;
  f.normalize.fail_compute = f.cuts.fail_compute = f.extract.fail_compute = f.merge.fail_compute =
    true;
  f.normalize.replace_output = f.cuts.replace_output = f.extract.replace_output =
    f.merge.replace_output                           = true;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  CHECK(f.normalize.attempts > f.normalize.publications);
  CHECK(f.cuts.attempts > f.cuts.publications);
  CHECK(f.extract.attempts > f.extract.publications);
  CHECK(f.merge.attempts > f.merge.publications);
  CHECK(f.cuts.replaced_multiple);
  CHECK(f.merge.replaced_single);
}

TEST_CASE("projection is applied only after tile merge", "[tiled_merge]")
{
  tiled_fixture f(2);
  f.merge.set_final_projections({1}, {sirius::logical_type::make(sirius::type_id::BIGINT)});
  f.input->add_data_batch(
    make_two_column_batch<int64_t, int64_t>(*f.gpu, {1, 3, 5}, {60, 40, 20}, cudf::type_id::INT64));
  f.input->add_data_batch(
    make_two_column_batch<int64_t, int64_t>(*f.gpu, {2, 4, 6}, {50, 30, 10}, cudf::type_id::INT64));
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>{60, 50, 40, 30, 20, 10});
}

TEST_CASE("matched cuts and runs and extracted pieces survive independent spilling",
          "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  f.drain_stage(0);
  f.drain_stage(1);
  auto* host = f.manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  auto spill = [&](cucascade::shared_data_repository* repo) {
    sirius::convertible_data_batch_provider provider(repo);
    auto candidates = provider.get_all_convertible(f.gpu, true, false);
    REQUIRE_FALSE(candidates.empty());
    for (auto& candidate : candidates) {
      REQUIRE(candidate->convert({host}, f.stream, *f.manager, false).has_value());
    }
  };
  spill(f.repo(2));
  f.drain_stage(2);
  spill(f.repo(3));
  f.drain_stage(3);
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST_CASE("concurrent extraction claims consume each run cuts pair once", "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  f.drain_stage(0);
  f.drain_stage(1);
  std::vector<std::future<std::unique_ptr<operator_data>>> claims;
  for (int i = 0; i < 8; ++i) {
    claims.push_back(
      std::async(std::launch::async, [&] { return f.extract.get_next_task_input_data(); }));
  }
  std::set<uint64_t> ids;
  std::vector<std::unique_ptr<operator_data>> claimed;
  for (auto& claim : claims) {
    if (auto data = claim.get()) { claimed.push_back(std::move(data)); }
  }
  REQUIRE(claimed.size() == 3);
  CHECK_FALSE(f.extract.all_ports_empty());
  CHECK_FALSE(f.extract.get_next_task_hint());
  for (auto& data : claimed) {
    auto const& values = dynamic_cast<pipelineable_operator_data&>(*data).get_data_batches();
    REQUIRE(values.size() == 2);
    for (auto const& value : values) {
      CHECK(ids.insert(value->get_batch_id()).second);
    }
    auto task = f.task(2, std::move(data));
    task->execute(f.stream);
  }
  CHECK(f.extract.all_ports_empty());
  f.drain_stage(3);
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

namespace {
class rejecting_tiled_destination : public cucascade::shared_data_repository {
 public:
  void add_data_batch_at_position_unlocked(std::shared_ptr<cucascade::data_batch> const&,
                                           std::size_t,
                                           std::size_t) override
  {
    throw rmm::out_of_memory("injected final fan-out failure");
  }
};
}  // namespace
TEST_CASE("partial final publication is fatal after popping source pieces", "[tiled_merge]")
{
  tiled_fixture f;
  rejecting_tiled_destination rejecting;
  sirius::op::test_physical_operator consumer{SiriusPhysicalOperatorType::FILTER, {}, 0};
  consumer.add_port("default",
                    std::make_unique<sirius_physical_operator::port>(
                      MemoryBarrierType::FULL, &rejecting, nullptr, nullptr));
  f.merge.add_next_port_after_sink({&consumer, "default"});
  f.add({1, 2, 3});
  f.drain_stage(0);
  f.drain_stage(1);
  f.drain_stage(2);
  auto data = f.merge.get_next_task_input_data();
  REQUIRE(data);
  CHECK(f.repo(3)->all_empty());
  auto task = f.task(3, std::move(data));
  CHECK_THROWS_AS(task->execute(f.stream), rmm::out_of_memory);
  task.reset();
  CHECK(f.output.total_size() == 1);
  CHECK(f.repo(3)->all_empty());
  CHECK(f.merge.publications == 0);
  CHECK_FALSE(f.merge.all_ports_empty());
  CHECK_FALSE(f.merge.get_next_task_input_data());
}

TEST_CASE("sort stages restore matched inputs across GPUs", "[tiled_merge][multi_gpu]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) { SKIP("requires two GPUs"); }
  // device_buffer retains its allocation stream for eventual deallocation.
  // Declare the remote stream before the fixture so even failure unwinding
  // destroys all repository batches before destroying either GPU's stream.
  std::optional<rmm::cuda_stream> remote_stream;
  tiled_fixture f(1, 2);
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{1}};
    remote_stream.emplace();
  }
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  for (std::size_t stage = 0; stage < 4; ++stage) {
    int device_id = stage % 2;
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{device_id}};
    auto stream  = device_id ? ::cuda::stream_ref{*remote_stream} : ::cuda::stream_ref{f.stream};
    auto* target = f.manager->get_memory_space(cucascade::memory::Tier::GPU, device_id);
    while (auto data = f.stages[stage]->get_next_task_input_data()) {
      auto task = f.task(stage, std::move(data), target);
      task->execute(stream);
    }
    CHECK(f.repo(stage)->all_empty());
  }
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

namespace {
// Any accidental restoration in source planning fails this CPU-only test.
class no_sort_io : public cucascade::idisk_io_backend {
 public:
  void write(std::filesystem::path const&,
             void const*,
             std::size_t,
             std::size_t,
             ::cuda::stream_ref) override
  {
    throw std::logic_error("Unexpected sort metadata I/O");
  }
  void read(
    std::filesystem::path const&, void*, std::size_t, std::size_t, ::cuda::stream_ref) override
  {
    throw std::logic_error("Unexpected sort metadata I/O");
  }
  void write(std::filesystem::path const&, void const*, std::size_t, std::size_t) override
  {
    throw std::logic_error("Unexpected sort metadata I/O");
  }
  void read(std::filesystem::path const&, void*, std::size_t, std::size_t) override
  {
    throw std::logic_error("Unexpected sort metadata I/O");
  }
};
struct cut_metadata_fixture {
  cucascade::memory::memory_space space{
    cucascade::memory::disk_memory_space_config{0, 1ull << 20, "/tmp"},
    std::make_shared<no_sort_io>()};
  cucascade::shared_data_repository repo;
  sirius_physical_order local{
    {sirius::logical_type::make(sirius::type_id::BIGINT)}, 1, ascending(), {0}, 0};
  sirius_physical_sort_extract extract{&local};
  uint64_t id = 100;
  cut_metadata_fixture()
  {
    extract.add_port("default",
                     std::make_unique<sirius_physical_operator::port>(
                       MemoryBarrierType::FULL, &repo, nullptr, nullptr));
  }
  void add(std::size_t partition, std::size_t position, std::size_t rows)
  {
    auto table = std::make_unique<cucascade::memory::disk_table_allocation>(
      "", std::vector<cucascade::memory::column_metadata>{}, 0);
    table->columns.emplace_back();
    table->columns[0].num_rows = rows;
    // Reverse IDs deliberately: matching must use logical positions.
    repo.add_data_batch_at_position(
      cucascade::data_batch::make(
        id--, std::make_unique<cucascade::disk_data_representation>(std::move(table), space)),
      position,
      partition);
  }
};
}  // namespace

TEST_CASE("cut handoff plans tile positions from metadata and preserves replacement coordinates",
          "[tiled_merge][cpu]")
{
  cut_metadata_fixture f;
  f.add(1, 3, 2);
  f.add(1, 0, 10);
  f.add(1, 2, 5);
  f.add(1, 1, 2);
  f.add(4, 1, 0);
  f.add(4, 0, 1);  // no internal cuts means one tile
  auto first_id = f.repo.get_data_batch_by_position(0, 1)->get_batch_id();
  std::weak_ptr<cucascade::data_batch> first_batch = f.repo.get_data_batch_by_position(0, 1);
  std::vector<std::unique_ptr<operator_data>> claimed;
  while (auto data = f.extract.get_next_task_input_data()) {
    claimed.push_back(std::move(data));
  }
  REQUIRE(claimed.size() == 3);
  CHECK_FALSE(f.extract.get_next_task_hint());
  CHECK_FALSE(f.extract.all_ports_empty());
  CHECK(f.repo.all_empty());
  CHECK_FALSE(first_batch.expired());
  using data_type   = sirius_physical_sort_extract::input_data;
  auto const& first = dynamic_cast<data_type const&>(*claimed[0]);
  CHECK(first.run == 0);
  CHECK(first.first_tile == 0);
  CHECK(first.get_data_batches()[0]->get_batch_id() == first_id);
  auto const& second = dynamic_cast<data_type const&>(*claimed[1]);
  CHECK(second.run == 1);
  CHECK(second.first_tile == 0);
  auto const& last = dynamic_cast<data_type const&>(*claimed[2]);
  CHECK(last.first_tile == 3);
  auto replacement     = first.with_batches_preserving_rows(first.get_data_batches());
  auto const& replaced = dynamic_cast<data_type const&>(*replacement);
  CHECK(replaced.run == first.run);
  CHECK(replaced.first_tile == first.first_tile);
  // Simulate successful fenced publications in reverse order. Completion only
  // accounts for outstanding tasks; input ownership is independent.
  for (auto it = claimed.rbegin(); it != claimed.rend(); ++it) {
    f.extract.on_task_output_published(**it);
  }
  CHECK(f.repo.all_empty());
  CHECK(f.extract.all_ports_empty());
  // No outstanding tasks remain: a further callback must not underflow the counter.
  CHECK_THROWS_AS(f.extract.on_task_output_published(*replacement), sirius::internal_exception);
}

TEST_CASE("cut handoff rejects missing or incompatible pairs before claiming work",
          "[tiled_merge][cpu]")
{
  cut_metadata_fixture f;
  SECTION("missing cut table") { f.add(0, 0, 5); }
  SECTION("gap in pair positions")
  {
    f.add(0, 0, 5);
    f.add(0, 3, 2);
  }
  SECTION("different tile counts")
  {
    f.add(0, 0, 5);
    f.add(0, 1, 2);
    f.add(0, 2, 5);
    f.add(0, 3, 3);
  }
  auto before = f.repo.total_size();
  CHECK_THROWS_AS(f.extract.get_next_task_input_data(), sirius::internal_exception);
  CHECK(f.repo.total_size() == before);
  CHECK_FALSE(f.extract.all_ports_empty());
}

TEST_CASE("partial extraction publication is fatal after popping its run and cuts", "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  f.drain_stage(0);
  f.drain_stage(1);
  auto before = f.repo(2)->total_size();
  auto data   = f.extract.get_next_task_input_data();
  REQUIRE(data);
  CHECK(f.repo(2)->total_size() == before - 2);
  // Run zero contributes to tiles zero, one and two. Publish tile zero, then
  // fail at tile one: the runtime must not report successful completion.
  f.repo(3)->add_data_batch_at_position(
    make_numeric_batch<int64_t>(*f.gpu, {99}, cudf::type_id::INT64), 0, 1);
  auto task = f.task(2, std::move(data));
  CHECK_THROWS_AS(task->execute(f.stream), cucascade::repository_duplicate_error);
  task.reset();
  CHECK(f.repo(3)->get_data_batch_by_position(0, 0));
  CHECK(f.repo(2)->total_size() == before - 2);
  CHECK(f.extract.publications == 0);
  CHECK_FALSE(f.extract.all_ports_empty());
}

TEST_CASE("whole partition sort stages pop inputs and retain task ownership", "[tiled_merge][cpu]")
{
  auto kind = GENERATE(0, 1, 2);
  cut_metadata_fixture f;
  f.add(1, 1, 5);
  f.add(1, 0, 5);
  f.add(3, 0, 1);
  auto first_id  = f.repo.get_data_batch_by_position(0, 1)->get_batch_id();
  auto second_id = f.repo.get_data_batch_by_position(1, 1)->get_batch_id();
  std::weak_ptr<cucascade::data_batch> first = f.repo.get_data_batch_by_position(0, 1);
  cucascade::shared_data_repository unordered;
  std::unique_ptr<sirius_physical_sort_stage> stage;
  if (kind == 0) {
    stage = std::make_unique<sirius_physical_sort_normalize>(&f.local);
    for (std::size_t p = 0; p < f.repo.num_partitions(); ++p) {
      while (auto batch = f.repo.pop_next_data_batch(p)) {
        unordered.add_data_batch(std::move(batch), p);
      }
    }
  } else if (kind == 1) {
    stage = std::make_unique<sirius_physical_sort_cuts>(&f.local);
  } else {
    stage = std::make_unique<sirius_physical_merge_sort>(&f.local);
  }
  auto* repo = kind == 0 ? &unordered : &f.repo;
  stage->add_port("default",
                  std::make_unique<sirius_physical_operator::port>(
                    MemoryBarrierType::FULL, repo, nullptr, nullptr));
  auto first_task = stage->get_next_task_input_data();
  REQUIRE(first_task);
  CHECK(repo->empty(1));
  CHECK_FALSE(repo->empty(3));
  auto const& data = dynamic_cast<sirius_physical_sort_stage::task_data const&>(*first_task);
  CHECK(data.partition == 1);
  REQUIRE(data.get_data_batches().size() == 2);
  if (kind == 0) {
    // Normalization consumes unordered contributions; only membership matters.
    std::set<uint64_t> ids;
    for (auto const& batch : data.get_data_batches()) {
      ids.insert(batch->get_batch_id());
    }
    auto expected = std::set<uint64_t>{first_id, second_id};
    CHECK(ids == expected);
  } else {
    CHECK(data.get_data_batches()[0]->get_batch_id() == first_id);
    CHECK(data.get_data_batches()[1]->get_batch_id() == second_id);
  }
  auto replacement = data.with_batches_preserving_rows(data.get_data_batches());
  first_task.reset();
  CHECK_FALSE(first.expired());
  auto last_task = stage->get_next_task_input_data();
  REQUIRE(last_task);
  CHECK(dynamic_cast<sirius_physical_sort_stage::task_data const&>(*last_task).partition == 3);
  CHECK(repo->all_empty());
  CHECK_FALSE(stage->get_next_task_input_data());
  CHECK_FALSE(stage->get_next_task_hint());
  CHECK_FALSE(stage->all_ports_empty());
  // Completion tracks tasks only; the empty repository needs no retirement.
  stage->on_task_output_published(*last_task);
  CHECK_FALSE(stage->all_ports_empty());
  stage->on_task_output_published(*replacement);
  CHECK(stage->all_ports_empty());
  replacement.reset();
  CHECK(first.expired());
}

TEST_CASE("extraction output replacement preserves sparse tile destinations", "[tiled_merge][cpu]")
{
  cut_metadata_fixture f;
  f.add(0, 0, 5);
  f.add(0, 1, 2);
  auto claimed      = f.extract.get_next_task_input_data();
  auto const& input = dynamic_cast<sirius_physical_sort_extract::input_data const&>(*claimed);
  sirius_physical_sort_extract::output_data output(input.get_data_batches(), 7, {3, 5});
  auto replacement = output.with_batches_preserving_rows(output.get_data_batches());
  cucascade::shared_data_repository destination;
  sirius::op::test_physical_operator consumer{SiriusPhysicalOperatorType::FILTER, {}, 0};
  consumer.add_port("default",
                    std::make_unique<sirius_physical_operator::port>(
                      MemoryBarrierType::FULL, &destination, nullptr, nullptr));
  f.extract.add_next_port_after_sink({&consumer, "default"});
  f.extract.sink(*replacement, ::cuda::stream_ref{cudaStream_t{nullptr}});
  CHECK(destination.get_data_batch_by_position(7, 3) == output.get_data_batches()[0]);
  CHECK(destination.get_data_batch_by_position(7, 5) == output.get_data_batches()[1]);
  CHECK(destination.empty(4));
  f.extract.on_task_output_published(*replacement);
  CHECK(f.extract.all_ports_empty());
  CHECK_THROWS_AS(f.extract.on_task_output_published(output), sirius::internal_exception);
}

TEST_CASE("sort stages accept repositories that have never received input", "[tiled_merge][cpu]")
{
  cut_metadata_fixture f;
  sirius_physical_sort_normalize normalize{&f.local};
  sirius_physical_sort_cuts cuts{&f.local};
  sirius_physical_merge_sort merge{&f.local};
  for (auto* stage : std::vector<sirius_physical_sort_stage*>{&normalize, &cuts, &merge}) {
    stage->add_port("default",
                    std::make_unique<sirius_physical_operator::port>(
                      MemoryBarrierType::FULL, &f.repo, nullptr, nullptr));
  }
  for (auto* stage :
       std::vector<sirius_physical_sort_stage*>{&normalize, &cuts, &f.extract, &merge}) {
    CHECK_FALSE(stage->get_next_task_input_data());
    CHECK_FALSE(stage->get_next_task_hint());
    CHECK(stage->all_ports_empty());
  }
}
