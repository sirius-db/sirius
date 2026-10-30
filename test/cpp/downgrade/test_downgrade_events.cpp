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

// sirius
#include "data/convertible_data.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "event/downgraded_batch.hpp"
#include "event/query_event_publisher.hpp"
#include "event/query_event_subscriber.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "query_id.hpp"
// data utilities
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <utils/telemetry_utils.hpp>
#include <utils/utils.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf
#include <cudf/utilities/default_stream.hpp>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace sirius::parallel;
using cucascade::memory::Tier;

namespace {

/// Every batch_downgraded event delivered to it, in delivery order. Drains its mailbox on stop,
/// so reading the events after the downgrade request has returned sees all of them.
class downgrade_recorder final : public sirius::event::query_event_subscriber {
 public:
  struct entry {
    sirius::query_id_t query;
    sirius::event::downgraded_batch batch;
  };

  explicit downgrade_recorder(sirius::event::query_event_publisher& publisher)
    : query_event_subscriber(
        publisher, {sirius::event::event_type::batch_downgraded}, /*drain_on_stop=*/true)
  {
    start();
  }

  ~downgrade_recorder() override { stop(); }

  [[nodiscard]] std::string_view name() const noexcept override { return "downgrade-recorder"; }

  /// Stop the subscriber, delivering everything already published, and return what arrived.
  std::vector<entry> collect()
  {
    stop();
    std::lock_guard lock(_mutex);
    return _entries;
  }

 private:
  void on_batch_downgraded(sirius::event::event_id_t,
                           sirius::event::timestamp_t,
                           sirius::query_id_t query_id,
                           sirius::event::downgraded_batch const& batch) noexcept override
  {
    std::lock_guard lock(_mutex);
    _entries.push_back({query_id, batch});
  }

  std::mutex _mutex;
  std::vector<entry> _entries;
};

/// A per-process directory under the temp directory, removed on scope exit even when a REQUIRE
/// aborts the test case.
class scoped_temp_dir {
 public:
  explicit scoped_temp_dir(std::string const& name)
    : _path(std::filesystem::temp_directory_path() / (name + "_" + std::to_string(::getpid())))
  {
  }
  ~scoped_temp_dir()
  {
    std::error_code ec;
    std::filesystem::remove_all(_path, ec);
  }
  scoped_temp_dir(scoped_temp_dir const&)            = delete;
  scoped_temp_dir& operator=(scoped_temp_dir const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const noexcept { return _path; }

 private:
  std::filesystem::path _path;
};

/// A convertible_data whose convert() records the given moves (when recording is on) and then
/// throws, like a multi-batch conversion that fails part-way.
class partially_failing_convertible final : public sirius::convertible_data {
 public:
  explicit partially_failing_convertible(std::vector<sirius::event::downgraded_batch> moves)
    : _moves(std::move(moves))
  {
  }

  std::optional<std::vector<std::size_t>> convert(
    const std::vector<const cucascade::memory::memory_space*>&,
    ::cuda::stream_ref,
    sirius::memory::sirius_memory_reservation_manager&,
    bool) override
  {
    if (recording()) {
      for (auto const& move : _moves) {
        note_converted(move);
      }
    }
    throw std::runtime_error("conversion failed after moving some batches");
  }

  std::size_t bytes_in_space(cucascade::memory::memory_space*) const override { return 0; }

 private:
  std::vector<sirius::event::downgraded_batch> _moves;
};

/// Memory manager with one GPU, one HOST space and, when @p disk_dir is non-empty, a DISK space.
std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_manager(
  std::filesystem::path const& disk_dir = {})
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2ull << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_numa_region_capacity(4ull << 30)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(0.75);
  if (!disk_dir.empty()) {
    std::filesystem::create_directories(disk_dir);
    builder.set_disk_mounting_point(0, 4ull << 30, disk_dir.string());
  }

  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
  sirius::converter_registry::initialize();
  return manager;
}

std::shared_ptr<cucascade::data_batch> make_gpu_batch(cucascade::memory::memory_space& gpu_space,
                                                      std::size_t num_rows = 1000)
{
  auto stream                                            = cudf::get_default_stream();
  std::vector<cudf::data_type> col_types                 = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {std::make_pair(0, 100000)};
  auto table = sirius::create_cudf_table_with_random_data(
    num_rows, col_types, ranges, stream, gpu_space.get_default_allocator());
  return sirius::make_data_batch(
    std::move(table), gpu_space, stream, sirius::telemetry::batch_telemetry_info{});
}

Tier batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

std::size_t batch_size(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_data()->get_size_in_bytes();
}

downgrade_executor make_executor(
  sirius::exec::query_lifecycle_registry& lifecycle,
  sirius::data::data_repository_manager_registry& repo_registry,
  sirius::memory::sirius_memory_reservation_manager& mgr,
  cucascade::memory::memory_space& source,
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask>* task_queue = nullptr)
{
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period = std::chrono::milliseconds{0}};
  return downgrade_executor(
    lifecycle, config, repo_registry, source.get_id(), &source, mgr, task_queue);
}

/// Repository holding @p batches, registered under @p query; returns its identity token.
std::uintptr_t add_repository(sirius::data::data_repository_manager_registry& repo_registry,
                              sirius::query_id_t query,
                              std::vector<std::shared_ptr<cucascade::data_batch>> const& batches)
{
  auto& manager = *repo_registry.create_for_query(query);
  auto repo     = std::make_unique<cucascade::shared_data_repository>();
  auto token    = reinterpret_cast<std::uintptr_t>(repo.get());
  for (auto const& batch : batches) {
    repo->add_data_batch(batch);
  }
  manager.add_new_repository(1, "out", std::move(repo));
  return token;
}

/// Global state carrying a scheduling priority, a preferred device and, optionally, an
/// operator-less pipeline of a query. The pipeline is detached on destruction because a task whose
/// pipeline has no operators cannot be destroyed while attached; declare it after the queue.
struct task_global_state {
  std::shared_ptr<sirius::pipeline::gpu_pipeline_task_global_state> state;

  task_global_state(std::optional<sirius::query_id_t> query,
                    sirius::exec::queue_priority priority,
                    int preferred_device)
  {
    std::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
    if (query) {
      const sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
      pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(build_ctx);
      pipeline->set_query_id(*query);
    }
    state = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
      std::move(pipeline), sirius::test::make_test_telemetry_context());
    state->set_priority(priority);
    state->set_preferred_device_id(preferred_device);
  }

  ~task_global_state() { state->set_pipeline(nullptr); }

  task_global_state(const task_global_state&)            = delete;
  task_global_state& operator=(const task_global_state&) = delete;

  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> make_task(
    uint64_t task_id, std::vector<std::shared_ptr<cucascade::data_batch>> batches) const
  {
    auto op_data = std::make_unique<sirius::op::pipelineable_operator_data>(std::move(batches));
    auto local =
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data));
    return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      task_id, std::vector<cucascade::shared_data_repository*>{}, std::move(local), state);
  }
};

downgrade_recorder::entry const& find_entry(std::vector<downgrade_recorder::entry> const& entries,
                                            cucascade::data_batch const& batch)
{
  auto it = std::find_if(entries.begin(), entries.end(), [&](auto const& e) {
    return e.batch.batch_id == batch.get_batch_id();
  });
  REQUIRE(it != entries.end());
  return *it;
}

}  // namespace

TEST_CASE("downgrade_record_query_id prefers the pipeline's id, then the priority's",
          "[downgrade_events]")
{
  sirius::event::downgraded_batch rec;
  REQUIRE_FALSE(downgrade_record_query_id(rec).has_value());

  rec.task_priority = sirius::query_priority_bits(sirius::make_query_id(9)) | 5;
  REQUIRE(downgrade_record_query_id(rec) == sirius::make_query_id(9));

  // The pipeline's id wins over the query packed into the priority.
  auto const pipeline_query = sirius::make_query_id(12);
  rec.task_query_id         = pipeline_query;
  REQUIRE(downgrade_record_query_id(rec) == pipeline_query);

  rec.task_priority.reset();
  REQUIRE(downgrade_record_query_id(rec) == pipeline_query);
}

TEST_CASE("publish_downgraded_batches reports the moves of a conversion that threw",
          "[downgrade_events]")
{
  auto const owner_query = sirius::make_query_id(3);
  auto const task_query  = sirius::make_query_id(4);
  sirius::event::downgraded_batch attributed;
  attributed.batch_id      = 11;
  attributed.task_query_id = task_query;
  sirius::event::downgraded_batch unattributed;
  unattributed.batch_id = 12;

  auto mgr              = make_manager();
  auto convert_and_fail = [&mgr](sirius::convertible_data& candidate) {
    REQUIRE_THROWS_AS(candidate.convert({}, cudf::get_default_stream(), *mgr, true),
                      std::runtime_error);
  };

  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  downgrade_recorder recorder(*publisher);

  partially_failing_convertible recorded({attributed, unattributed});
  recorded.record_conversions(true);
  convert_and_fail(recorded);
  REQUIRE(recorded.converted_batches().size() == 2);

  // Without a publisher nothing is reported, whatever was recorded.
  CHECK(publish_downgraded_batches(nullptr, recorded, owner_query) == 0);
  // A caller-supplied owner attributes every record.
  CHECK(publish_downgraded_batches(publisher.get(), recorded, owner_query) == 2);
  // Otherwise each record is attributed on its own, and one that cannot be is skipped.
  CHECK(publish_downgraded_batches(publisher.get(), recorded, std::nullopt) == 1);

  // With recording off, a failed conversion leaves nothing to publish.
  partially_failing_convertible unrecorded({attributed});
  convert_and_fail(unrecorded);
  CHECK(unrecorded.converted_batches().empty());
  CHECK(publish_downgraded_batches(publisher.get(), unrecorded, owner_query) == 0);

  auto const entries = recorder.collect();
  REQUIRE(entries.size() == 3);
  CHECK(entries[0].query == owner_query);
  CHECK(entries[0].batch.batch_id == 11);
  CHECK(entries[1].query == owner_query);
  CHECK(entries[1].batch.batch_id == 12);
  CHECK(entries[2].query == task_query);
  CHECK(entries[2].batch.batch_id == 11);
}

TEST_CASE("repository downgrades publish one event per batch under the owning query",
          "[downgrade_events]")
{
  auto mgr        = make_manager();
  auto* gpu_space = mgr->get_memory_space(Tier::GPU, 0);
  auto* host      = mgr->get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(host != nullptr);

  auto const q1 = sirius::make_query_id(1);
  auto const q2 = sirius::make_query_id(2);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(q1);
  lifecycle.open_query(q2);
  sirius::data::data_repository_manager_registry repo_registry;
  auto a           = make_gpu_batch(*gpu_space);
  auto b           = make_gpu_batch(*gpu_space, 2000);
  auto c           = make_gpu_batch(*gpu_space);
  auto const tok_1 = add_repository(repo_registry, q1, {a, b});
  auto const tok_2 = add_repository(repo_registry, q2, {c});
  auto const sizes = std::vector<std::size_t>{batch_size(*a), batch_size(*b), batch_size(*c)};

  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  downgrade_recorder recorder(*publisher);
  auto executor = make_executor(lifecycle, repo_registry, *mgr, *gpu_space);
  executor.set_query_event_publisher(*publisher);
  executor.start();
  REQUIRE(executor.request_free_memory_and_wait(1ull << 30) == sizes[0] + sizes[1] + sizes[2]);
  executor.stop();

  auto const entries = recorder.collect();
  REQUIRE(entries.size() == 3);

  struct expected {
    cucascade::data_batch* batch;
    sirius::query_id_t query;
    std::uintptr_t repository;
    std::size_t bytes;
  };
  for (auto const& want : {expected{a.get(), q1, tok_1, sizes[0]},
                           expected{b.get(), q1, tok_1, sizes[1]},
                           expected{c.get(), q2, tok_2, sizes[2]}}) {
    REQUIRE(batch_tier(*want.batch) == Tier::HOST);
    auto const& e = find_entry(entries, *want.batch);
    CHECK(e.query == want.query);
    CHECK(e.batch.batch.lock().get() == want.batch);
    CHECK(e.batch.from == gpu_space->get_id());
    CHECK(e.batch.to == host->get_id());
    CHECK(e.batch.bytes == want.bytes);
    CHECK(e.batch.source == sirius::event::downgrade_source::repository);
    CHECK(e.batch.repository == want.repository);
    CHECK(e.batch.partition == 0);
    CHECK_FALSE(e.batch.task_priority.has_value());
    CHECK_FALSE(e.batch.preferred_device.has_value());
    CHECK_FALSE(e.batch.task_query_id.has_value());
  }
}

TEST_CASE("task-queue downgrades publish under the task's query", "[downgrade_events]")
{
  auto mgr        = make_manager();
  auto* gpu_space = mgr->get_memory_space(Tier::GPU, 0);
  auto* host      = mgr->get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(host != nullptr);

  auto const attributed = sirius::make_query_id(7);
  auto const fallback   = sirius::make_query_id(9);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(attributed);
  // A task without a pipeline is queued under query 0, so its return to the queue needs it open.
  lifecycle.open_query(sirius::make_query_id(0));
  lifecycle.open_query(fallback);
  sirius::data::data_repository_manager_registry repo_registry;

  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue(
    &sirius::pipeline::index_keys_for);
  auto const attributed_priority = sirius::query_priority_bits(attributed) | 3;
  auto const fallback_priority   = sirius::query_priority_bits(fallback) | 4;
  // Declared after the queue so the pipelines are detached before the queue destroys the tasks.
  task_global_state with_pipeline(attributed, attributed_priority, /*preferred_device=*/0);
  task_global_state without_pipeline(std::nullopt, fallback_priority, /*preferred_device=*/0);

  auto moved1 = make_gpu_batch(*gpu_space);
  auto moved2 = make_gpu_batch(*gpu_space, 1500);
  auto moved3 = make_gpu_batch(*gpu_space, 500);
  auto const gpu_bytes =
    std::vector<std::size_t>{batch_size(*moved1), batch_size(*moved2), batch_size(*moved3)};
  REQUIRE(queue.push(with_pipeline.make_task(1, {moved1, moved2})));
  REQUIRE(queue.push(without_pipeline.make_task(2, {moved3})));

  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  downgrade_recorder recorder(*publisher);
  auto executor = make_executor(lifecycle, repo_registry, *mgr, *gpu_space, &queue);
  executor.set_query_event_publisher(*publisher);
  executor.start();
  REQUIRE(executor.request_free_memory_and_wait(1ull << 30) > 0);
  executor.stop();

  REQUIRE(queue.size() == 2);
  auto const entries = recorder.collect();
  REQUIRE(entries.size() == 3);

  auto const& e1 = find_entry(entries, *moved1);
  auto const& e2 = find_entry(entries, *moved2);
  auto const& e3 = find_entry(entries, *moved3);
  CHECK(e1.batch.conversion_seq < e2.batch.conversion_seq);
  for (auto const* e : {&e1, &e2}) {
    CHECK(e->query == attributed);
    CHECK(e->batch.task_query_id == attributed);
    CHECK(e->batch.task_priority == attributed_priority);
  }
  CHECK(e3.query == fallback);
  CHECK_FALSE(e3.batch.task_query_id.has_value());
  CHECK(e3.batch.task_priority == fallback_priority);

  struct expected {
    downgrade_recorder::entry const* entry;
    cucascade::data_batch* batch;
    std::size_t bytes;
  };
  for (auto const& want : {expected{&e1, moved1.get(), gpu_bytes[0]},
                           expected{&e2, moved2.get(), gpu_bytes[1]},
                           expected{&e3, moved3.get(), gpu_bytes[2]}}) {
    auto const* e = want.entry;
    REQUIRE(batch_tier(*want.batch) == Tier::HOST);
    CHECK(e->batch.batch.lock().get() == want.batch);
    CHECK(e->batch.from == gpu_space->get_id());
    CHECK(e->batch.to == host->get_id());
    CHECK(e->batch.bytes == want.bytes);
    CHECK(e->batch.source == sirius::event::downgrade_source::task_queue);
    CHECK(e->batch.repository == 0);
    CHECK(e->batch.preferred_device == 0);
  }
}

TEST_CASE("a HOST to DISK downgrade publishes a second, later record", "[downgrade_events]")
{
  scoped_temp_dir const disk_dir("sirius_test_downgrade_events");
  auto mgr         = make_manager(disk_dir.path());
  auto* gpu_space  = mgr->get_memory_space(Tier::GPU, 0);
  auto* host       = mgr->get_memory_space(Tier::HOST, 0);
  auto disk_spaces = mgr->get_memory_spaces_for_tier(Tier::DISK);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(host != nullptr);
  REQUIRE(disk_spaces.size() == 1);

  auto const q = sirius::make_query_id(1);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(q);
  sirius::data::data_repository_manager_registry repo_registry;
  auto batch       = make_gpu_batch(*gpu_space);
  auto const token = add_repository(repo_registry, q, {batch});

  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  downgrade_recorder recorder(*publisher);
  auto gpu_executor  = make_executor(lifecycle, repo_registry, *mgr, *gpu_space);
  auto host_executor = make_executor(lifecycle, repo_registry, *mgr, *host);
  gpu_executor.set_query_event_publisher(*publisher);
  host_executor.set_query_event_publisher(*publisher);
  gpu_executor.start();
  host_executor.start();

  REQUIRE(gpu_executor.request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(batch_tier(*batch) == Tier::HOST);
  auto const host_bytes = batch_size(*batch);
  REQUIRE(host_executor.request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(batch_tier(*batch) == Tier::DISK);
  host_executor.stop();
  gpu_executor.stop();

  auto const entries = recorder.collect();
  REQUIRE(entries.size() == 2);
  auto const& to_host = entries[0].batch.from == gpu_space->get_id() ? entries[0] : entries[1];
  auto const& to_disk = &to_host == &entries[0] ? entries[1] : entries[0];
  CHECK(to_host.batch.to == host->get_id());
  CHECK(to_disk.batch.from == host->get_id());
  CHECK(to_disk.batch.to == disk_spaces.front()->get_id());
  CHECK(to_disk.batch.bytes == host_bytes);
  CHECK(to_host.batch.conversion_seq < to_disk.batch.conversion_seq);
  for (auto const* e : {&to_host, &to_disk}) {
    CHECK(e->query == q);
    CHECK(e->batch.batch_id == batch->get_batch_id());
    CHECK(e->batch.repository == token);
    CHECK(e->batch.source == sirius::event::downgrade_source::repository);
  }
}

TEST_CASE("downgrades publish nothing while no one subscribes", "[downgrade_events]")
{
  auto mgr        = make_manager();
  auto* gpu_space = mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto const q = sirius::make_query_id(1);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(q);
  sirius::data::data_repository_manager_registry repo_registry;
  auto unobserved = make_gpu_batch(*gpu_space);
  add_repository(repo_registry, q, {unobserved});

  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  auto executor  = make_executor(lifecycle, repo_registry, *mgr, *gpu_space);
  executor.set_query_event_publisher(*publisher);
  executor.start();

  REQUIRE_FALSE(publisher->has_subscribers(sirius::event::event_type::batch_downgraded));
  REQUIRE(executor.request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(batch_tier(*unobserved) == Tier::HOST);

  // The decision is taken per request: a subscriber that arrives later sees later moves only.
  auto observed = make_gpu_batch(*gpu_space);
  add_repository(repo_registry, sirius::make_query_id(2), {observed});
  lifecycle.open_query(sirius::make_query_id(2));
  downgrade_recorder recorder(*publisher);
  REQUIRE(executor.request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(batch_tier(*observed) == Tier::HOST);
  executor.stop();

  auto const entries = recorder.collect();
  REQUIRE(entries.size() == 1);
  CHECK(entries.front().batch.batch_id == observed->get_batch_id());
  CHECK(entries.front().query == sirius::make_query_id(2));
}
