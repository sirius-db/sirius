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

// Lifecycle tests for MERGE_SORT refinement, driven through real pipeline tasks:
// reservations, preparation, OOM reschedules, sink and publication.

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
using step = sirius_physical_merge_sort::step;

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

class injected_merge : public sirius_physical_merge_sort {
 public:
  using sirius_physical_merge_sort::sirius_physical_merge_sort;
  // Fail the first attempt of every task after it computes, before publication.
  bool fail_each_task{false};
  bool replace_output{false};
  std::set<step> replaced, executed;
  std::set<operator_data const*> failed;
  std::size_t attempts{0}, publications{0}, coalesce_runs{0};
  void notify_task_attempt(operator_data& data, task_attempt_context attempt) override
  {
    ++attempts;
    sirius_physical_merge_sort::notify_task_attempt(data, attempt);
  }
  void on_task_output_published(operator_data const& output) override
  {
    ++publications;
    sirius_physical_merge_sort::on_task_output_published(output);
  }
  std::unique_ptr<operator_data> execute(operator_data const& input,
                                         ::cuda::stream_ref stream) override
  {
    auto kind = dynamic_cast<work const&>(input).kind;
    executed.insert(kind);
    coalesce_runs += kind == step::coalesce;
    auto output = sirius_physical_merge_sort::execute(input, stream);
    if (fail_each_task && failed.insert(&input).second) {
      throw rmm::out_of_memory("injected before publication");
    }
    if (!replace_output) { return output; }
    auto const& batches = dynamic_cast<result const&>(*output);
    replaced.insert(batches.kind);
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
          batch_telemetry()));
        REQUIRE(replacements.back() != batch);
      }
      stream.sync();
      return batches.with_batches_preserving_rows(std::move(replacements));
    } catch (...) {
      cudaStreamSynchronize(stream.get());
      throw;
    }
  }
};

struct refinement_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu;
  rmm::cuda_stream stream;
  cucascade::shared_data_repository_manager repositories;
  sirius_physical_order local;
  injected_merge merge;
  sirius::op::test_physical_operator destination{SiriusPhysicalOperatorType::FILTER, {}, 0};
  cucascade::shared_data_repository output;
  cucascade::shared_data_repository* input;
  std::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global;
  uint64_t task_id{0};
  std::size_t retry_floor{0};
  std::size_t spilled{0};

  // The chunk target holds three BIGINT rows per column, so small inputs refine.
  explicit refinement_fixture(std::size_t width = 1, std::size_t gpus = 1)
    : manager(sirius::test::operator_utils::initialize_memory_manager(gpus)),
      gpu(manager->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      local(duckdb::vector<sirius::logical_type>(
              width, sirius::logical_type::make(sirius::type_id::BIGINT)),
            width,
            ascending(),
            {},
            0),
      merge(&local)
  {
    merge.operator_id = 1;
    merge.chunk_bytes = 24 * width;
    repositories.add_new_repository(
      1, "default", std::make_unique<cucascade::shared_data_repository>());
    input = repositories.get_repository(1, "default").get();
    merge.add_port("default",
                   std::make_unique<sirius_physical_operator::port>(
                     MemoryBarrierType::FULL, input, nullptr, nullptr));
    merge.register_refinement(repositories);
    pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
      sirius::pipeline::pipeline_build_context{nullptr});
    pipeline->set_pipeline_id(0);
    sirius::pipeline::sirius_pipeline_build_state build;
    build.set_pipeline_source(*pipeline, merge);
    build.add_pipeline_operator(*pipeline, merge);
    build.set_pipeline_sink(*pipeline, &merge, 0);
    global = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    destination.add_port("default",
                         std::make_unique<sirius_physical_operator::port>(
                           MemoryBarrierType::FULL, &output, nullptr, nullptr));
    merge.add_next_port_after_sink({&destination, "default"});
  }
  void add(std::vector<int64_t> values, std::size_t range = 0)
  {
    input->add_data_batch(make_numeric_batch<int64_t>(*gpu, values, cudf::type_id::INT64), range);
  }
  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> task(
    std::unique_ptr<operator_data> data, cucascade::memory::memory_space* target = nullptr)
  {
    if (!target) { target = gpu; }
    auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      ++task_id,
      repositories.get_repositories(),
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(data)),
      global);
    auto info             = task->get_estimated_reservation_size_info(target);
    info.reservation_size = std::max(info.reservation_size, retry_floor);
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
          info                  = task->get_estimated_reservation_size_info(target);
          info.reservation_size = std::max(info.reservation_size, retry_floor);
          reservation           = target->make_reservation_or_null(
            std::min(info.reservation_size, target->get_max_memory()));
          if (reservation) { break; }
        }
        if (reservation) { break; }
      }
    }
    REQUIRE(reservation);
    auto* state =
      dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(task->local_state());
    state->set_reservation(std::move(reservation), info);
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
  void run(std::unique_ptr<operator_data> data,
           cucascade::memory::memory_space* target,
           ::cuda::stream_ref on)
  {
    auto item = task(std::move(data), target);
    for (int retry = 0;; ++retry) {
      REQUIRE(retry < 24);
      try {
        item->execute(on);
        break;
      } catch (sirius::pipeline::oom_reschedule_exception& error) {
        auto retry_input = error.release_intermediate_data();
        dynamic_cast<pipelineable_operator_data&>(*retry_input).remove_read_only_lock();
        retry_floor =
          dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(item->local_state())
            ->get_retry_reservation_floor();
        item.reset();
        item = task(std::move(retry_input), target);
      }
    }
    item.reset();
    retry_floor = 0;
  }
  // Claim everything currently available, then complete it in reverse claim order.
  std::size_t round(cucascade::memory::memory_space* target = nullptr,
                    std::optional<::cuda::stream_ref> on    = std::nullopt)
  {
    std::vector<std::unique_ptr<operator_data>> claimed;
    while (auto data = merge.get_next_task_input_data()) {
      claimed.push_back(std::move(data));
    }
    std::reverse(claimed.begin(), claimed.end());
    for (auto& data : claimed) {
      run(std::move(data), target, on.value_or(::cuda::stream_ref{stream}));
    }
    return claimed.size();
  }
  void spill_all()
  {
    auto* host = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
    for (auto* repository : repositories.get_repositories()) {
      sirius::convertible_data_batch_provider provider(repository);
      for (auto& candidate : provider.get_all_convertible(gpu, true, false)) {
        REQUIRE(candidate->convert({host}, stream, *manager, false).has_value());
        ++spilled;
      }
    }
  }
  void finish()
  {
    CHECK(merge.all_ports_empty());
    for (auto* repository : repositories.get_repositories()) {
      CHECK(repository->all_empty());
    }
    pipeline->update_pipeline_status(false);
    CHECK(pipeline->is_pipeline_finished());
  }
  void drain(bool spill_between_rounds = false)
  {
    while (round()) {
      if (spill_between_rounds) { spill_all(); }
    }
    finish();
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

TEST_CASE("merge refinement preserves range order under reversed completion", "[merge_refinement]")
{
  refinement_fixture f;
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
  CHECK(f.output.get_positions().size() >= 6);
  for (auto position : f.output.get_positions()) {
    CHECK(*sirius::get_batch_rows_and_bytes(*f.output.get_data_batch_by_position(position)).rows <=
          3);
  }
}

TEST_CASE("merge refinement handles empty single and duplicate runs", "[merge_refinement]")
{
  auto count = GENERATE(0, 1, 101);
  refinement_fixture f;
  f.add(std::vector<int64_t>(count, 7));
  f.add(std::vector<int64_t>(count, 7));
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>(2 * count, 7));
}

TEST_CASE("refinement metadata survives batch replacement and compute retry in every step",
          "[merge_refinement]")
{
  refinement_fixture f;
  f.merge.fail_each_task = true;
  f.merge.replace_output = true;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  CHECK(f.merge.attempts > f.merge.publications);
  CHECK((f.merge.replaced == std::set<step>{step::merge, step::sample, step::plan, step::cut}));
}

TEST_CASE("many single-row contributions coalesce before sampling", "[merge_refinement]")
{
  // Twelve one-row contributions with a three-row target would sample every row; greedy
  // groups of three merge into four runs, which then sample and cut normally.
  refinement_fixture f;
  f.merge.fail_each_task = true;
  f.merge.replace_output = true;
  for (int64_t value : {7, 3, 11, 0, 5, 9, 1, 10, 4, 8, 2, 6}) {
    f.add({value});
  }
  f.drain();
  std::vector<int64_t> expected(12);
  std::iota(expected.begin(), expected.end(), 0);
  CHECK(f.rows() == expected);
  CHECK(f.merge.executed.contains(step::coalesce));
  CHECK(f.merge.replaced.contains(step::coalesce));
  CHECK(f.merge.executed.contains(step::cut));
  CHECK(f.merge.attempts > f.merge.publications);
  for (auto position : f.output.get_positions()) {
    CHECK(*sirius::get_batch_rows_and_bytes(*f.output.get_data_batch_by_position(position)).rows <=
          3);
  }
}

TEST_CASE("coalescing moves single contributions without a task", "[merge_refinement]")
{
  // Alternating 1-row and 3-row contributions: each 3-row one fills a group alone, so
  // only adjacent 1-row pairs are merged by tasks.
  refinement_fixture f;
  for (int64_t i = 0; i < 6; ++i) {
    f.add({10 * i});
    f.add({10 * i + 1});
    f.add({10 * i + 2, 10 * i + 3, 10 * i + 4});
  }
  f.drain();
  std::vector<int64_t> expected;
  for (int64_t i = 0; i < 6; ++i) {
    for (int64_t j = 0; j < 5; ++j) {
      expected.push_back(10 * i + j);
    }
  }
  CHECK(f.rows() == expected);
  // Six 1-row pairs are merged; the six 3-row contributions move without tasks.
  CHECK(f.merge.coalesce_runs == 6);
}

TEST_CASE("merge refinement applies the final projection only to merged leaves",
          "[merge_refinement]")
{
  refinement_fixture f(2);
  f.merge.set_final_projections({1}, {sirius::logical_type::make(sirius::type_id::BIGINT)});
  f.input->add_data_batch(
    make_two_column_batch<int64_t, int64_t>(*f.gpu, {1, 3, 5}, {60, 40, 20}, cudf::type_id::INT64));
  f.input->add_data_batch(
    make_two_column_batch<int64_t, int64_t>(*f.gpu, {2, 4, 6}, {50, 30, 10}, cudf::type_id::INT64));
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>{60, 50, 40, 30, 20, 10});
}

TEST_CASE("queued refinement state survives spilling between steps", "[merge_refinement]")
{
  refinement_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  f.drain(true);
  CHECK(f.spilled > 0);
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST_CASE("concurrent refinement claims take each contribution once", "[merge_refinement]")
{
  refinement_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  std::vector<std::future<std::unique_ptr<operator_data>>> claims;
  for (int i = 0; i < 8; ++i) {
    claims.push_back(
      std::async(std::launch::async, [&] { return f.merge.get_next_task_input_data(); }));
  }
  std::set<uint64_t> ids;
  std::vector<std::unique_ptr<operator_data>> claimed;
  for (auto& claim : claims) {
    if (auto data = claim.get()) { claimed.push_back(std::move(data)); }
  }
  // The range is oversized; each contribution fills a sampling group on its own.
  REQUIRE(claimed.size() == 3);
  CHECK_FALSE(f.merge.all_ports_empty());
  CHECK_FALSE(f.merge.get_next_task_hint());
  for (auto& data : claimed) {
    auto const& work = dynamic_cast<sirius_physical_merge_sort::work const&>(*data);
    CHECK((work.kind == step::sample));
    REQUIRE(work.get_data_batches().size() == 1);
    CHECK(ids.insert(work.get_data_batches()[0]->get_batch_id()).second);
  }
  for (auto& data : claimed) {
    f.run(std::move(data), f.gpu, f.stream);
  }
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

namespace {
class rejecting_destination : public cucascade::shared_data_repository {
 public:
  void add_data_batch_at_position_unlocked(std::shared_ptr<cucascade::data_batch> const&,
                                           std::size_t,
                                           std::size_t) override
  {
    throw rmm::out_of_memory("injected final fan-out failure");
  }
};
}  // namespace

TEST_CASE("partial final publication is fatal after claiming a leaf", "[merge_refinement]")
{
  refinement_fixture f;
  rejecting_destination rejecting;
  sirius::op::test_physical_operator consumer{SiriusPhysicalOperatorType::FILTER, {}, 0};
  consumer.add_port("default",
                    std::make_unique<sirius_physical_operator::port>(
                      MemoryBarrierType::FULL, &rejecting, nullptr, nullptr));
  f.merge.add_next_port_after_sink({&consumer, "default"});
  f.merge.chunk_bytes = std::size_t{1} << 20;  // one leaf merge, no refinement
  f.add({1, 2, 3});
  auto data = f.merge.get_next_task_input_data();
  REQUIRE(data);
  CHECK((dynamic_cast<sirius_physical_merge_sort::work const&>(*data).kind == step::merge));
  CHECK(f.input->all_empty());
  auto task = f.task(std::move(data));
  CHECK_THROWS_AS(task->execute(f.stream), rmm::out_of_memory);
  task.reset();
  CHECK(f.output.total_size() == 1);
  CHECK(f.merge.publications == 0);
  CHECK_FALSE(f.merge.all_ports_empty());
  CHECK_FALSE(f.merge.get_next_task_input_data());
}

TEST_CASE("merge refinement restores shared splitters across GPUs", "[merge_refinement][multi_gpu]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) { SKIP("requires two GPUs"); }
  // device_buffer retains its allocation stream for eventual deallocation.
  // Declare the remote stream before the fixture so even failure unwinding
  // destroys all repository batches before destroying either GPU's stream.
  std::optional<rmm::cuda_stream> remote_stream;
  refinement_fixture f(1, 2);
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{1}};
    remote_stream.emplace();
  }
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  for (int index = 0;; ++index) {
    int device_id = index % 2;
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{device_id}};
    auto stream  = device_id ? ::cuda::stream_ref{*remote_stream} : ::cuda::stream_ref{f.stream};
    auto* target = f.manager->get_memory_space(cucascade::memory::Tier::GPU, device_id);
    if (!f.round(target, stream)) { break; }
  }
  f.finish();
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

namespace {
// Any restoration during ingestion or claiming fails this CPU-only test.
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
struct metadata_fixture {
  cucascade::memory::memory_space space{
    cucascade::memory::disk_memory_space_config{0, 1ull << 20, "/tmp"},
    std::make_shared<no_sort_io>()};
  cucascade::shared_data_repository_manager repositories;
  cucascade::shared_data_repository repo;
  sirius_physical_order local{
    {sirius::logical_type::make(sirius::type_id::BIGINT)}, 1, ascending(), {0}, 0};
  sirius_physical_merge_sort merge{&local};
  uint64_t id = 100;
  metadata_fixture()
  {
    merge.operator_id = 1;
    merge.add_port("default",
                   std::make_unique<sirius_physical_operator::port>(
                     MemoryBarrierType::FULL, &repo, nullptr, nullptr));
  }
  std::shared_ptr<cucascade::data_batch> add(std::size_t partition, std::size_t rows)
  {
    auto table = std::make_unique<cucascade::memory::disk_table_allocation>(
      "", std::vector<cucascade::memory::column_metadata>{}, 0);
    table->columns.emplace_back();
    table->columns[0].num_rows = rows;
    auto batch                 = cucascade::data_batch::make(
      id--, std::make_unique<cucascade::disk_data_representation>(std::move(table), space));
    repo.add_data_batch(batch, partition);
    return batch;
  }
};
}  // namespace

TEST_CASE("merge ingestion reads metadata only and claims fitting ranges whole",
          "[merge_refinement][cpu]")
{
  metadata_fixture f;
  f.merge.register_refinement(f.repositories);
  auto first  = f.add(1, 5);
  auto second = f.add(1, 5);
  f.add(3, 1);
  f.add(3, 0);  // empty contributions are dropped at ingestion
  std::weak_ptr<cucascade::data_batch> first_batch = first;
  std::set<uint64_t> expected_ids{first->get_batch_id(), second->get_batch_id()};
  first.reset();
  second.reset();

  auto first_task = f.merge.get_next_task_input_data();
  REQUIRE(first_task);
  CHECK(f.repo.all_empty());
  auto const& work = dynamic_cast<sirius_physical_merge_sort::work const&>(*first_task);
  CHECK((work.kind == step::merge));
  CHECK(work.get_position() == (std::size_t{1} << 62) / 4);  // range 1 of 4
  CHECK(work.ordinals == std::vector<uint64_t>{0, 1});
  std::set<uint64_t> ids;
  for (auto const& batch : work.get_data_batches()) {
    ids.insert(batch->get_batch_id());
  }
  CHECK(ids == expected_ids);
  auto replacement     = work.with_batches_preserving_rows(work.get_data_batches());
  auto const& replaced = dynamic_cast<sirius_physical_merge_sort::work const&>(*replacement);
  CHECK((replaced.kind == work.kind));
  CHECK(replaced.node == work.node);
  CHECK(replaced.ordinals == work.ordinals);
  CHECK(replaced.get_position() == work.get_position());

  auto last_task = f.merge.get_next_task_input_data();
  REQUIRE(last_task);
  auto const& last = dynamic_cast<sirius_physical_merge_sort::work const&>(*last_task);
  CHECK(last.get_data_batches().size() == 1);
  CHECK_FALSE(f.merge.get_next_task_input_data());
  CHECK_FALSE(f.merge.get_next_task_hint());
  CHECK_FALSE(f.merge.all_ports_empty());

  // Publication, not input ownership, retires a node. Simulate it in reverse order.
  using result = sirius_physical_merge_sort::result;
  result last_done({}, step::merge, last.node, {}, {}, last.get_position());
  result first_done({}, step::merge, work.node, {}, {}, work.get_position());
  f.merge.on_task_output_published(last_done);
  CHECK_FALSE(f.merge.all_ports_empty());
  first_task.reset();
  CHECK_FALSE(first_batch.expired());  // still owned by the replacement input
  f.merge.on_task_output_published(first_done);
  CHECK(f.merge.all_ports_empty());
  replacement.reset();
  CHECK(first_batch.expired());
  CHECK_THROWS_AS(f.merge.on_task_output_published(first_done), sirius::internal_exception);
}

TEST_CASE("merge accepts a repository that has never received input", "[merge_refinement][cpu]")
{
  metadata_fixture f;
  CHECK_FALSE(f.merge.get_next_task_input_data());
  CHECK_FALSE(f.merge.get_next_task_hint());
  CHECK(f.merge.all_ports_empty());
}
