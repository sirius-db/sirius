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
#include "data/convertible_data.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "event/downgraded_batch.hpp"
#include "event/event.hpp"
#include "event/query_event_publisher.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "memory/device_preload_budget.hpp"
#include "memory/memory_wait_board.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/partition_placement.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "parallel/task.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "preload/candidate_store.hpp"
#include "preload/preload_executor.hpp"
#include "preload/preload_policy.hpp"
#include "preload/preload_streams.hpp"
#include "query_id.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/telemetry_utils.hpp"
#include "utils/utils.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

using cucascade::memory::Tier;
using namespace sirius::preload;
using namespace std::chrono_literals;

namespace {

constexpr std::size_t gib = std::size_t{1} << 30;
constexpr std::size_t mib = std::size_t{1} << 20;

/// Shape of the memory spaces a fixture builds.
struct fixture_options {
  int gpus{1};
  std::size_t gpu_capacity{gib};
  /// The reservation limit as a fraction of capacity.
  double reservation_fraction{1.0};
  double trigger{0.9};
  double stop{0.8};
  /// Track GPU reservations per stream instead of per thread.
  bool per_stream{false};
};

/// A pipeline whose FULL barrier the test opens by hand.
class barrier_pipeline : public sirius::pipeline::sirius_pipeline {
 public:
  barrier_pipeline() : sirius_pipeline(sirius::pipeline::pipeline_build_context{nullptr, true}) {}

  void finish() { _finished.store(true); }

  [[nodiscard]] bool is_pipeline_finished() const override { return _finished.load(); }

 private:
  std::atomic<bool> _finished{false};
};

/// A consumer that drains one partition per task.
class partition_consumer : public sirius::op::sirius_physical_partition_consumer_operator {
 public:
  partition_consumer()
    : sirius_physical_partition_consumer_operator(
        sirius::op::SiriusPhysicalOperatorType::HASH_JOIN, {}, 0)
  {
  }
};

/// A plain operator, for repositories whose consumer is not partition-draining.
class plain_operator : public sirius::op::sirius_physical_operator {
 public:
  plain_operator()
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::PROJECTION, {}, 0)
  {
  }
};

/// Blocks promotions inside promote_one until released, and counts them.
struct promotion_gate {
  std::mutex mu;
  std::condition_variable cv;
  bool open{false};
  int entered{0};

  void enter()
  {
    std::unique_lock lk{mu};
    ++entered;
    cv.notify_all();
    cv.wait(lk, [&] { return open; });
  }

  bool wait_entered(int n, std::chrono::milliseconds limit = 20s)
  {
    std::unique_lock lk{mu};
    return cv.wait_for(lk, limit, [&] { return entered >= n; });
  }

  void release()
  {
    std::lock_guard lk{mu};
    open = true;
    cv.notify_all();
  }

  int count()
  {
    std::lock_guard lk{mu};
    return entered;
  }
};

/**
 * GPU, HOST and DISK spaces; a publisher, a lifecycle, a task queue, a wait board and preload
 * stream pools; and helpers to make spilled batches, queued tasks and the events preload reacts
 * to. The executor is built by each test with its own configuration.
 */
struct executor_fixture {
  fixture_options opts;
  std::filesystem::path disk_dir{std::filesystem::temp_directory_path() /
                                 ("sirius_test_preload_executor_" + std::to_string(::getpid()))};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  std::vector<cucascade::memory::memory_space*> gpus;
  cucascade::memory::memory_space* host{nullptr};
  cucascade::memory::memory_space* disk{nullptr};
  std::shared_ptr<sirius::event::query_event_publisher> publisher{
    std::make_shared<sirius::event::query_event_publisher>()};
  sirius::exec::query_lifecycle_registry lifecycle;
  std::shared_ptr<sirius::memory::memory_wait_board> board;
  std::shared_ptr<preload_streams> streams;
  std::vector<int> device_ids;
  std::shared_ptr<sirius::memory::device_preload_budget> budget;
  rmm::cuda_stream stream_owner{};
  ::cuda::stream_ref stream{stream_owner.value()};
  sirius::exec::multi_index_priority_queue<sirius::parallel::itask> queue{
    &sirius::pipeline::index_keys_for};
  std::uint64_t task_id{1};
  std::atomic<std::uint64_t> expected_events{0};
  std::unique_ptr<preload_executor> exec;

  explicit executor_fixture(fixture_options o = {}) : opts(o)
  {
    std::filesystem::create_directories(disk_dir);
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(static_cast<std::size_t>(opts.gpus))
      .set_gpu_usage_limit(opts.gpu_capacity)
      .set_reservation_fraction_per_gpu(opts.reservation_fraction)
      .set_downgrade_fractions_per_gpu(opts.trigger, opts.stop)
      .track_reservation_per_stream(opts.per_stream)
      .set_per_numa_region_capacity(4 * gib)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, 4 * gib, disk_dir.string());
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    std::vector<int> devices;
    for (int d = 0; d < opts.gpus; ++d) {
      gpus.push_back(mgr->get_memory_space(Tier::GPU, d));
      REQUIRE(gpus.back() != nullptr);
      devices.push_back(d);
    }
    host = mgr->get_memory_space(Tier::HOST, 0);
    disk = mgr->get_memory_space(Tier::DISK, 0);
    REQUIRE(host != nullptr);
    REQUIRE(disk != nullptr);
    board      = std::make_shared<sirius::memory::memory_wait_board>(opts.gpus - 1);
    streams    = std::make_shared<preload_streams>(devices);
    device_ids = devices;
    budget     = std::make_shared<sirius::memory::device_preload_budget>(device_ids);
  }

  ~executor_fixture()
  {
    if (exec) { exec->stop(); }
    exec.reset();
    while (queue.try_pop()) {}
    std::error_code ec;
    std::filesystem::remove_all(disk_dir, ec);
  }

  executor_fixture(executor_fixture const&)            = delete;
  executor_fixture& operator=(executor_fixture const&) = delete;

  [[nodiscard]] cucascade::memory::memory_space& gpu() const { return *gpus.front(); }

  /// The capacity / downgrade-stop info preload is told, derived from this fixture's configurator.
  /// The pinned cuCascade exposes no get_capacity()/get_downgrade_stop_threshold().
  [[nodiscard]] space_capacity_fn space_capacity() const
  {
    auto const o = opts;
    return [o](cucascade::memory::memory_space_id id) -> space_capacity_info {
      switch (id.tier) {
        case Tier::GPU: return {o.gpu_capacity, o.stop};
        // set_per_numa_region_capacity(4 GiB); default downgrade stop fraction is 0.65.
        default: return {4 * gib, 0.65};
      }
    };
  }

  [[nodiscard]] std::size_t gpu_stop_threshold() const
  {
    return static_cast<std::size_t>(static_cast<double>(opts.gpu_capacity) * opts.stop);
  }

  [[nodiscard]] std::size_t gpu_trigger_threshold() const
  {
    return static_cast<std::size_t>(static_cast<double>(opts.gpu_capacity) * opts.trigger);
  }

  /// Device preloading enabled, no head margin, one copy per lane.
  [[nodiscard]] static preload_executor_config config()
  {
    preload_executor_config cfg;
    cfg.concurrent_preloads      = 1;
    cfg.head_margin_tasks        = 0;
    cfg.device_preloading_budget = 0.20;
    return cfg;
  }

  void build(preload_executor_config cfg = config(),
             hw_caps_override caps       = {false, false},
             bool start                  = true)
  {
    // Device preloading is on when a budget is set and the GPU spaces are not per-stream.
    bool const device_pre = cfg.device_preloading_budget > 0.0 && !opts.per_stream;
    exec                  = std::make_unique<preload_executor>(
      *publisher,
      std::move(cfg),
      preload_dependencies{.lifecycle                = lifecycle,
                                            .memory                   = *mgr,
                                            .topology                 = nullptr,
                                            .task_queue               = &queue,
                                            .wait_board               = board,
                                            .streams                  = device_pre ? streams : nullptr,
                                            .device_budget            = device_pre ? budget : nullptr,
                                            .pipeline_threads_per_gpu = 1,
                                            .space_capacity           = space_capacity(),
                                            .caps                     = caps});
    exec->enable_promotion_log_for_testing();
    if (start) { exec->start(); }
  }

  /// An RAII guard for a gated test: on any exit path (success, a failed REQUIRE, an exception),
  /// it opens @p gate and stops/joins the executor while the gate is still alive, so no worker is
  /// left blocked in @c promotion_gate::enter() when the gate is destroyed.
  [[nodiscard]] auto arm_gate(promotion_gate& gate)
  {
    return absl::Cleanup{[this, &gate] {
      gate.release();
      if (exec) { exec->stop(); }
    }};
  }

  /// Open query @p id in the lifecycle and in preload.
  sirius::query_id_t open_query(std::uint32_t id,
                                std::vector<repo_info> repos = {},
                                std::vector<int> active      = {0})
  {
    auto const q = sirius::make_query_id(id);
    lifecycle.open_query(q);
    exec->prepare_for_query(q, std::move(repos), std::move(active));
    return q;
  }

  /// Priority of pipeline @p rank of query @p q.
  [[nodiscard]] static sirius::exec::queue_priority priority(sirius::query_id_t q,
                                                             std::int64_t rank)
  {
    return sirius::query_priority_bits(q) | rank;
  }

  [[nodiscard]] std::shared_ptr<cucascade::data_batch> make_gpu_batch(std::size_t rows)
  {
    std::vector<cudf::data_type> types{cudf::data_type{cudf::type_id::INT32}};
    std::vector<std::optional<std::pair<int, int>>> ranges{std::make_pair(0, 100000)};
    auto table = sirius::create_cudf_table_with_random_data(
      rows, types, ranges, stream, gpu().get_default_allocator());
    stream.sync();
    return sirius::make_data_batch(
      std::move(table), gpu(), stream, sirius::telemetry::batch_telemetry_info{});
  }

  /// Move @p batch to @p space, the way a downgrade does.
  void move_to(std::shared_ptr<cucascade::data_batch> const& batch,
               cucascade::memory::memory_space& space)
  {
    auto mut = batch->to_mutable();
    if (cudaEvent_t const writer = mut.get_data()->get_writer_event(); writer != nullptr) {
      cucascade::cuda::cuda_event_view{writer}.wait(stream);
    }
    auto& registry = sirius::converter_registry::get();
    if (space.get_tier() == Tier::HOST) {
      mut.convert_to<cucascade::host_data_representation>(registry, &space, stream);
    } else {
      mut.convert_to<cucascade::disk_data_representation>(registry, &space, stream);
    }
    stream.sync();
  }

  [[nodiscard]] std::shared_ptr<cucascade::data_batch> spilled(cucascade::memory::memory_space& to,
                                                               std::size_t rows = 4096)
  {
    auto batch = make_gpu_batch(rows);
    move_to(batch, to);
    return batch;
  }

  /// A record of @p batch moving GPU:0 -> where it is now.
  [[nodiscard]] static sirius::event::downgraded_batch record(
    std::shared_ptr<cucascade::data_batch> const& batch,
    sirius::event::downgrade_source source = sirius::event::downgrade_source::task_queue,
    std::uintptr_t repository              = 0,
    std::size_t partition                  = 0,
    std::optional<std::size_t> bytes       = std::nullopt)
  {
    sirius::event::downgraded_batch rec;
    rec.batch          = batch;
    rec.batch_id       = batch->get_batch_id();
    rec.conversion_seq = sirius::g_conversion_seq.fetch_add(1) + 1;
    rec.from           = cucascade::memory::memory_space_id{Tier::GPU, 0};
    {
      auto ro   = batch->to_read_only();
      rec.to    = ro.get_memory_space()->get_id();
      rec.bytes = bytes.value_or(ro.get_data()->get_size_in_bytes());
    }
    rec.source     = source;
    rec.repository = repository;
    rec.partition  = partition;
    return rec;
  }

  void downgraded(sirius::query_id_t q, sirius::event::downgraded_batch const& rec)
  {
    ++expected_events;
    publisher->publish_batch_downgraded(q, rec);
  }

  void completed(sirius::query_id_t q,
                 int gpu_id                          = 0,
                 sirius::event::task_outcome outcome = sirius::event::task_outcome::succeeded)
  {
    ++expected_events;
    publisher->publish_task_completed(q, 0, gpu_id, outcome);
  }

  void wait_ended(sirius::query_id_t q, int gpu_id = 0)
  {
    ++expected_events;
    publisher->publish_memory_wait_ended_for_task(q, 0, gpu_id, 0);
  }

  void task_created(sirius::query_id_t q)
  {
    ++expected_events;
    publisher->publish_task_created(q, 0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0);
  }

  void pipeline_closed(sirius::query_id_t q)
  {
    ++expected_events;
    publisher->publish_pipeline_closed(q, 0, 0);
  }

  void awaiting(int gpu_id = 0)
  {
    ++expected_events;
    publisher->publish_executor_awaiting_task(gpu_id);
  }

  void queue_empty()
  {
    ++expected_events;
    publisher->publish_task_queue_empty();
  }

  void settle() { REQUIRE(exec->wait_until_quiescent_for_testing(30s, expected_events)); }

  /// Wait until every published event has been handled, without waiting for quiescence.
  bool handled(std::chrono::milliseconds limit = 20s)
  {
    return eventually([&] { return exec->events_handled_for_testing() >= expected_events; }, limit);
  }

  /// Poll @p done until it holds or @p limit passes.
  template <typename Pred>
  static bool eventually(Pred done, std::chrono::milliseconds limit = 20s)
  {
    auto const deadline = std::chrono::steady_clock::now() + limit;
    while (!done()) {
      if (std::chrono::steady_clock::now() > deadline) { return false; }
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }

  [[nodiscard]] std::shared_ptr<sirius::pipeline::gpu_pipeline_task_global_state> pipeline_state(
    sirius::exec::queue_priority p) const
  {
    auto state = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
      nullptr, sirius::test::make_test_telemetry_context());
    state->set_priority(p);
    return state;
  }

  /// Queue a task of @p state with inputs @p batches, preferring @p device (none when empty).
  void push_task(std::shared_ptr<sirius::pipeline::gpu_pipeline_task_global_state> const& state,
                 std::vector<std::shared_ptr<cucascade::data_batch>> batches,
                 std::optional<int> device = 0)
  {
    auto data  = std::make_unique<sirius::op::pipelineable_operator_data>(std::move(batches));
    auto local = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(data));
    if (device) { local->set_preferred_device_id(*device); }
    REQUIRE(queue.push(std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      task_id++, std::vector<cucascade::shared_data_repository*>{}, std::move(local), state)));
  }

  /// Reserve GPU memory so that its consumed bytes reach @p target.
  [[nodiscard]] std::unique_ptr<cucascade::memory::reservation> fill_gpu_to(std::size_t target)
  {
    auto& space         = gpu();
    auto const consumed = consumed_bytes(space.get_max_memory(), space.get_available_memory());
    REQUIRE(target > consumed);
    // Available memory is measured against the capacity, which may exceed the limit.
    auto const excess = space.get_available_memory() > space.get_max_memory()
                          ? space.get_available_memory() - space.get_max_memory()
                          : 0;
    auto res          = space.make_reservation_or_null(excess + (target - consumed));
    REQUIRE(res != nullptr);
    return res;
  }

  [[nodiscard]] std::size_t gpu_consumed() const
  {
    return consumed_bytes(gpu().get_max_memory(), gpu().get_available_memory());
  }

  [[nodiscard]] std::size_t committed_ceiling(preload_executor_config const& cfg) const
  {
    return gpu_ceilings_bytes(cfg,
                              {opts.gpu_capacity, gpu().get_max_memory(), gpu_stop_threshold()})
      .committed;
  }
};

Tier tier_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

std::uint64_t stat(preload_executor const& exec, counter c) { return exec.stats()[c]; }

std::vector<std::uint64_t> promoted_ids(preload_executor const& exec)
{
  std::vector<std::uint64_t> out;
  for (auto const& r : exec.promotion_log_for_testing()) {
    if (r.outcome == promotion_outcome::promoted) { out.push_back(r.batch_id); }
  }
  return out;
}

repo_info repo_of(cucascade::data_repository& repo,
                  sirius::exec::queue_priority dest_priority,
                  sirius::op::sirius_physical_partition_consumer_operator const* consumer = nullptr,
                  std::vector<std::weak_ptr<sirius::pipeline::sirius_pipeline>> full      = {})
{
  repo_info info;
  info.token              = reinterpret_cast<std::uintptr_t>(&repo);
  info.repo               = &repo;
  info.dest_priority      = dest_priority;
  info.partition_consumer = consumer;
  info.full_sources       = std::move(full);
  return info;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// 1-3: waking only on events
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: nothing happens without events, a downgrade event promotes",
                 "[preload_executor]")
{
  build();
  auto const q     = open_query(1);
  auto const batch = spilled(*host);
  push_task(pipeline_state(priority(q, 1)), {batch});

  std::this_thread::sleep_for(100ms);
  CHECK(stat(*exec, counter::passes) == 0);
  CHECK(tier_of(*batch) == Tier::HOST);

  downgraded(q, record(batch));
  settle();
  CHECK(tier_of(*batch) == Tier::GPU);
  CHECK(stat(*exec, counter::promoted_h2g) == 1);
  CHECK(promoted_ids(*exec) == std::vector<std::uint64_t>{batch->get_batch_id()});
  CHECK(exec->outstanding_bytes_for_testing(0) > 0);
  // One job, however many passes saw the item.
  CHECK(stat(*exec, counter::items_planned) == 1);
  CHECK(stat(*exec, counter::class_a_items) == 1);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: above the ceiling nothing retries until memory frees",
                 "[preload_executor]")
{
  auto const cfg = config();
  build(cfg);
  auto const q     = open_query(1);
  auto const batch = spilled(*host);
  push_task(pipeline_state(priority(q, 1)), {batch});

  // Above the committed ceiling, below the trigger.
  auto const ceiling = committed_ceiling(cfg);
  auto filler        = fill_gpu_to(ceiling + 16 * mib);
  REQUIRE_FALSE(gpu().should_downgrade_memory());

  downgraded(q, record(batch));
  settle();
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(stat(*exec, counter::refused_ceiling) >= 1);

  auto const before = exec->stats();
  std::this_thread::sleep_for(200ms);
  auto const after = exec->stats();
  CHECK(after.values == before.values);
  CHECK(tier_of(*batch) == Tier::HOST);

  filler.reset();
  completed(q);
  settle();
  CHECK(tier_of(*batch) == Tier::GPU);
  CHECK(stat(*exec, counter::promoted_h2g) == 1);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: a running promotion counts once against the ceiling",
                 "[preload_executor]")
{
  auto cfg                = config();
  cfg.concurrent_preloads = 2;
  build(cfg);
  promotion_gate gate;
  auto gate_guard = arm_gate(gate);
  exec->set_promotion_hook_for_testing([&](promotion_job const&) { gate.enter(); });
  auto const q     = open_query(1);
  auto const state = pipeline_state(priority(q, 1));
  auto const first = spilled(*host);
  auto const other = spilled(*host);
  push_task(state, {first});
  push_task(state, {other});

  // Room for both estimates plus half of one. Counting the parked job's bytes twice, once as
  // its reservation and once as in flight, would leave room for only half of the second.
  std::size_t const bytes = 8 * mib;
  auto const est          = reserve_estimate(bytes);
  auto const filler       = fill_gpu_to(committed_ceiling(cfg) - 2 * est - est / 2);
  REQUIRE_FALSE(gpu().should_downgrade_memory());

  // The hook runs once the first job's reservation is made: it stays parked there.
  downgraded(q, record(first, sirius::event::downgrade_source::task_queue, 0, 0, bytes));
  if (!gate.wait_entered(1)) {
    gate.release();
    FAIL("the first promotion never started");
  }
  downgraded(q, record(other, sirius::event::downgrade_source::task_queue, 0, 0, bytes));
  CHECK(gate.wait_entered(2));
  CHECK(stat(*exec, counter::refused_ceiling) == 0);

  gate.release();
  settle();
  CHECK(tier_of(*first) == Tier::GPU);
  CHECK(tier_of(*other) == Tier::GPU);
  CHECK(stat(*exec, counter::promoted_h2g) == 2);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: board hold-offs block until their wake-up event",
                 "[preload_executor]")
{
  build();
  auto const q     = open_query(1);
  auto const state = pipeline_state(priority(q, 1));

  SECTION("a stalled reserve phase")
  {
    auto const batch = spilled(*host);
    push_task(state, {batch});
    std::optional<sirius::memory::memory_wait_board::waiter> waiter{board->enter_wait(0)};
    downgraded(q, record(batch));
    settle();
    CHECK(tier_of(*batch) == Tier::HOST);
    CHECK(stat(*exec, counter::refused_held_off) >= 1);

    auto const passes = stat(*exec, counter::passes);
    waiter.reset();
    std::this_thread::sleep_for(100ms);
    CHECK(tier_of(*batch) == Tier::HOST);  // the release alone is not an event
    CHECK(stat(*exec, counter::passes) == passes);

    wait_ended(q);
    settle();
    CHECK(tier_of(*batch) == Tier::GPU);
  }

  SECTION("a pending OOM retry, which a sibling's success does not clear")
  {
    auto const batch = spilled(*host);
    push_task(state, {batch});
    std::optional<sirius::memory::memory_wait_board::oom_token> token{board->hold_oom(0)};
    downgraded(q, record(batch));
    settle();
    CHECK(tier_of(*batch) == Tier::HOST);

    completed(q);  // a sibling task succeeded
    settle();
    CHECK(tier_of(*batch) == Tier::HOST);

    token.reset();  // the retried task is destroyed...
    completed(q);   // ...and its success is published
    settle();
    CHECK(tier_of(*batch) == Tier::GPU);
  }
}

// ---------------------------------------------------------------------------------------------
// 4: contention
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: a contended batch is retried only after a queue event",
                 "[preload_executor]")
{
  auto cfg              = config();
  cfg.max_lock_attempts = 3;
  build(cfg);
  auto const q     = open_query(1);
  auto const batch = spilled(*host);
  push_task(pipeline_state(priority(q, 1)), {batch});

  {
    auto const reader = batch->to_read_only();  // a consumer holds the batch
    downgraded(q, record(batch));
    settle();
    CHECK(stat(*exec, counter::contended) == 1);

    std::this_thread::sleep_for(100ms);
    CHECK(stat(*exec, counter::contended) == 1);

    task_created(q);
    settle();
    CHECK(stat(*exec, counter::contended) == 2);

    task_created(q);
    settle();
    CHECK(stat(*exec, counter::contended) == 3);
    CHECK(stat(*exec, counter::dropped_lock_attempts) == 1);

    task_created(q);
    settle();
    CHECK(stat(*exec, counter::contended) == 3);
  }
  task_created(q);
  settle();
  CHECK(tier_of(*batch) == Tier::HOST);
}

// ---------------------------------------------------------------------------------------------
// 5-6: ordering and readiness
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: promotions follow the plan order and skip the head",
                 "[preload_executor]")
{
  auto cfg                           = config();
  cfg.head_margin_tasks              = 2;
  cfg.lookahead_tasks                = 8;
  cfg.repository_window              = 2;
  cfg.repository_partition_lookahead = 1;
  build(cfg);

  cucascade::data_repository repo;
  partition_consumer consumer;
  consumer.set_placement(std::make_shared<sirius::op::partition_placement const>(
    sirius::op::partition_placement::round_robin(4, {0})));
  auto const q1 = open_query(1, {repo_of(repo, priority(sirius::make_query_id(1), 3), &consumer)});
  auto const q2 = open_query(2);

  // Exact bucket: five tasks of q1; positions 0 and 1 are the head margin.
  std::vector<std::shared_ptr<cucascade::data_batch>> exact;
  auto const exact_state = pipeline_state(priority(q1, 5));
  for (int i = 0; i < 5; ++i) {
    exact.push_back(spilled(*host));
    push_task(exact_state, {exact.back()});
  }
  // Shared bucket: four tasks of q2, at positions 5..8 behind the GPU's own bucket.
  std::vector<std::shared_ptr<cucascade::data_batch>> shared;
  auto const shared_state = pipeline_state(priority(q2, 1));
  for (int i = 0; i < 4; ++i) {
    shared.push_back(spilled(*host));
    push_task(shared_state, {shared.back()}, std::nullopt);
  }
  // A ready partition consumer: partition 0 is drained next, partition 1 is in the lookahead.
  std::vector<std::shared_ptr<cucascade::data_batch>> parts;
  for (std::size_t p = 0; p < 4; ++p) {
    parts.push_back(spilled(*host));
    repo.add_data_batch(parts.back(), p);
  }

  {
    // Hold the GPU off while every record arrives, so one pass sees them all.
    std::optional<sirius::memory::memory_wait_board::waiter> waiter{board->enter_wait(0)};
    for (auto const& b : exact) {
      downgraded(q1, record(b));
    }
    for (auto const& b : shared) {
      downgraded(q2, record(b));
    }
    for (std::size_t p = 0; p < parts.size(); ++p) {
      downgraded(q1,
                 record(parts[p],
                        sirius::event::downgrade_source::repository,
                        reinterpret_cast<std::uintptr_t>(&repo),
                        p));
    }
    settle();
    CHECK(promoted_ids(*exec).empty());
  }
  wait_ended(q1);
  settle();

  std::vector<std::uint64_t> const expected{parts[1]->get_batch_id(),
                                            exact[2]->get_batch_id(),
                                            exact[3]->get_batch_id(),
                                            exact[4]->get_batch_id(),
                                            shared[0]->get_batch_id(),
                                            shared[1]->get_batch_id(),
                                            shared[2]->get_batch_id(),
                                            shared[3]->get_batch_id()};
  CHECK(promoted_ids(*exec) == expected);
  CHECK(tier_of(*exact[0]) == Tier::HOST);
  CHECK(tier_of(*exact[1]) == Tier::HOST);
  CHECK(tier_of(*parts[0]) == Tier::HOST);
  CHECK(tier_of(*parts[2]) == Tier::HOST);
  CHECK(tier_of(*parts[3]) == Tier::HOST);
  CHECK(stat(*exec, counter::skipped_creator_next) >= 1);
  for (auto const& r : exec->promotion_log_for_testing()) {
    CHECK(r.k == (r.batch_id == parts[1]->get_batch_id() ? klass::B : klass::A));
  }
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: consumers behind a FULL barrier get staging only",
                 "[preload_executor]")
{
  auto cfg                           = config();
  cfg.repository_window              = 2;
  cfg.repository_partition_lookahead = 1;
  build(cfg);

  auto const plain_source = std::make_shared<barrier_pipeline>();
  auto const part_source  = std::make_shared<barrier_pipeline>();
  cucascade::data_repository plain_repo;
  cucascade::data_repository part_repo;
  partition_consumer consumer;
  consumer.set_placement(std::make_shared<sirius::op::partition_placement const>(
    sirius::op::partition_placement::round_robin(3, {0})));
  auto const qid = sirius::make_query_id(1);
  auto const q   = open_query(1,
                              {repo_of(plain_repo, priority(qid, 2), nullptr, {plain_source}),
                               repo_of(part_repo, priority(qid, 3), &consumer, {part_source})});

  // Not ready: a DISK batch is staged, a HOST batch is left alone.
  auto const plain_disk = spilled(*disk);
  auto const plain_host = spilled(*host);
  plain_repo.add_data_batch(plain_disk, 0);
  plain_repo.add_data_batch(plain_host, 0);
  // Not ready, partitioned: DISK batches at the front of partitions 0..2.
  std::vector<std::shared_ptr<cucascade::data_batch>> parts;
  for (std::size_t p = 0; p < 3; ++p) {
    parts.push_back(spilled(*disk));
    part_repo.add_data_batch(parts.back(), p);
  }

  auto const token = [](cucascade::data_repository& r) {
    return reinterpret_cast<std::uintptr_t>(&r);
  };
  downgraded(q, record(plain_disk, sirius::event::downgrade_source::repository, token(plain_repo)));
  downgraded(q, record(plain_host, sirius::event::downgrade_source::repository, token(plain_repo)));
  for (std::size_t p = 0; p < parts.size(); ++p) {
    downgraded(q,
               record(parts[p], sirius::event::downgrade_source::repository, token(part_repo), p));
  }
  settle();
  CHECK(tier_of(*plain_disk) == Tier::HOST);
  CHECK(tier_of(*plain_host) == Tier::HOST);
  // At most a window of staged batches per repository, in partition order.
  CHECK(tier_of(*parts[0]) == Tier::HOST);
  CHECK(tier_of(*parts[1]) == Tier::HOST);
  CHECK(tier_of(*parts[2]) == Tier::DISK);
  CHECK(stat(*exec, counter::promoted_d2h) == 3);
  CHECK(stat(*exec, counter::promoted_h2g) == 0);
  CHECK(stat(*exec, counter::promoted_d2g) == 0);
  // More events change nothing while the staged batches wait for their consumer.
  completed(q);
  settle();
  CHECK(tier_of(*parts[2]) == Tier::DISK);

  // Barriers clear: the unpartitioned consumer is the creator's next pop and is left alone; the
  // partitioned one gets the partition after its next one.
  plain_source->finish();
  part_source->finish();
  pipeline_closed(q);
  settle();
  CHECK(tier_of(*plain_disk) == Tier::HOST);
  CHECK(tier_of(*plain_host) == Tier::HOST);
  CHECK(tier_of(*parts[0]) == Tier::HOST);
  CHECK(tier_of(*parts[1]) == Tier::GPU);
  CHECK(tier_of(*parts[2]) == Tier::DISK);
  CHECK(stat(*exec, counter::promoted_h2g) == 1);
}

// ---------------------------------------------------------------------------------------------
// 7: bounces and caps
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: bounces are counted and capped per batch",
                 "[preload_executor]")
{
  auto cfg                     = config();
  cfg.max_promotions_per_batch = 2;
  build(cfg);
  auto const q     = open_query(1);
  auto const batch = spilled(*host);
  push_task(pipeline_state(priority(q, 1)), {batch});

  downgraded(q, record(batch));
  settle();
  REQUIRE(tier_of(*batch) == Tier::GPU);

  move_to(batch, *host);  // downgraded again before its task ran
  downgraded(q, record(batch));
  settle();
  CHECK(stat(*exec, counter::bounce) == 1);
  CHECK(tier_of(*batch) == Tier::GPU);  // class A: one bounce bans speculation only

  move_to(batch, *host);
  downgraded(q, record(batch));
  settle();
  CHECK(stat(*exec, counter::bounce) == 2);
  CHECK(stat(*exec, counter::dropped_promotion_cap) == 1);
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(stat(*exec, counter::promoted_h2g) == 2);
  CHECK(exec->outstanding_bytes_for_testing(0) == 0);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: a downgrade during the copy is a bounce, not a promotion",
                 "[preload_executor]")
{
  build();
  auto const q     = open_query(1);
  auto const batch = spilled(*host);
  push_task(pipeline_state(priority(q, 1)), {batch});

  std::atomic<bool> fired{false};
  std::atomic<bool> merged{false};
  exec->set_promotion_hook_for_testing([&](promotion_job const& job) {
    if (job.batch_id != batch->get_batch_id() || fired.exchange(true)) { return; }
    // A newer record lands while the worker is between its reservation and the copy.
    downgraded(q, record(batch));
    merged = handled();
  });
  downgraded(q, record(batch));
  settle();
  REQUIRE(merged);
  CHECK(stat(*exec, counter::bounce) == 1);
  CHECK(stat(*exec, counter::promoted_h2g) == 1);
  CHECK(exec->outstanding_bytes_for_testing(0) == 0);
}

// ---------------------------------------------------------------------------------------------
// 8-9: retire and stop
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: retire with a promotion in flight",
                 "[preload_executor]")
{
  build();
  auto const q     = open_query(1);
  auto const batch = spilled(*host);
  push_task(pipeline_state(priority(q, 1)), {batch});
  promotion_gate gate;
  auto gate_guard = arm_gate(gate);
  exec->set_promotion_hook_for_testing([&](promotion_job const&) { gate.enter(); });

  downgraded(q, record(batch));
  REQUIRE(gate.wait_entered(1));

  lifecycle.quiesce(q);
  exec->retire_query(q);
  auto waiter = std::async(std::launch::async, [&] { lifecycle.wait_for_work(q); });
  CHECK(waiter.wait_for(100ms) == std::future_status::timeout);  // the lane holds a lease
  gate.release();
  REQUIRE(waiter.wait_for(20s) == std::future_status::ready);
  waiter.get();
  settle();

  CHECK(stat(*exec, counter::late_outcome_after_retire) == 1);
  CHECK(exec->outstanding_bytes_for_testing(0) == 0);
  exec->forget_query(q);
  exec->forget_query(q);
  CHECK_NOTHROW(lifecycle.close(q));
  CHECK_NOTHROW(lifecycle.clear());

  // Events of the retired query create nothing.
  auto const other = spilled(*host);
  downgraded(q, record(other));
  settle();
  CHECK(stat(*exec, counter::dropped_unknown_query) == 1);
  CHECK(tier_of(*other) == Tier::HOST);
}

TEST_CASE_METHOD(executor_fixture, "preload executor: stop and destruction", "[preload_executor]")
{
  SECTION("destroyed without start")
  {
    build(config(), {false, false}, /*start=*/false);
    exec.reset();
  }

  SECTION("stopped with a promotion in flight")
  {
    build();
    auto const q     = open_query(1);
    auto const batch = spilled(*host);
    push_task(pipeline_state(priority(q, 1)), {batch});
    promotion_gate gate;
    auto gate_guard = arm_gate(gate);
    exec->set_promotion_hook_for_testing([&](promotion_job const&) { gate.enter(); });
    downgraded(q, record(batch));
    REQUIRE(gate.wait_entered(1));

    auto stopper = std::async(std::launch::async, [&] { exec->stop(); });
    CHECK(stopper.wait_for(100ms) == std::future_status::timeout);  // joins the running copy
    gate.release();
    REQUIRE(stopper.wait_for(20s) == std::future_status::ready);
    stopper.get();
    CHECK(tier_of(*batch) == Tier::GPU);
    exec->stop();   // idempotent
    exec->start();  // terminal: no restart
    CHECK(exec->lane_threads_for_testing().size() == 2);
  }
}

// ---------------------------------------------------------------------------------------------
// 10: lanes
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: lane concurrency and in-flight bytes stay under their caps",
                 "[preload_executor]")
{
  // Hold every promotion inside a gate so the overlap is observed deterministically (no sleep):
  // @p want_overlap workers must be blocked in the gate at once before it is released.
  auto const run = [&](int cap,
                       double inflight_fraction,
                       std::uint32_t id,
                       int want_overlap,
                       bool expect_inflight_refusal) {
    auto cfg                  = config();
    cfg.concurrent_preloads   = static_cast<std::size_t>(cap);
    cfg.lookahead_tasks       = 16;
    cfg.max_inflight_fraction = inflight_fraction;
    cfg.max_batch_fraction    = inflight_fraction;
    build(cfg);
    promotion_gate gate;
    auto gate_guard = arm_gate(gate);
    exec->set_promotion_hook_for_testing([&](promotion_job const&) { gate.enter(); });
    auto const q     = open_query(id);
    auto const state = pipeline_state(priority(q, 1));
    std::vector<std::shared_ptr<cucascade::data_batch>> batches;
    {
      std::optional<sirius::memory::memory_wait_board::waiter> waiter{board->enter_wait(0)};
      for (int i = 0; i < 12; ++i) {
        batches.push_back(spilled(*host));
        push_task(state, {batches.back()});
        downgraded(q, record(batches.back()));
      }
      settle();  // held off: no copies start, so settle does not deadlock on the gate
    }
    wait_ended(q);  // release the hold-off: workers now enter the gate and block there
    // Deterministic overlap: wait until want_overlap workers are blocked in the gate together.
    REQUIRE(gate.wait_entered(want_overlap));
    if (expect_inflight_refusal) {
      // The in-flight-bytes cap refuses the rest while the first few are held in the gate.
      REQUIRE(eventually([&] { return exec->stats()[counter::refused_inflight] >= 1; }));
    }
    gate.release();
    settle();
    for (auto const& b : batches) {
      CHECK(tier_of(*b) == Tier::GPU);
    }
    auto const s = exec->stats();
    exec->stop();
    exec.reset();
    while (queue.try_pop()) {}
    return s;
  };

  SECTION("cap 1")
  {
    auto const s = run(1, 0.10, 1, /*want_overlap=*/1, /*expect_inflight_refusal=*/false);
    CHECK(s.lane_concurrency_max.at(0) == 1);
  }
  SECTION("cap 5")
  {
    auto const s = run(5, 0.10, 2, /*want_overlap=*/2, /*expect_inflight_refusal=*/false);
    CHECK(s.lane_concurrency_max.at(0) <= 5);
    CHECK(s.lane_concurrency_max.at(0) >= 2);
  }
  SECTION("in-flight bytes")
  {
    double const fraction = 0.004;
    auto const s   = run(5, fraction, 3, /*want_overlap=*/1, /*expect_inflight_refusal=*/true);
    auto const cap = fraction_of(fraction, opts.gpu_capacity);
    CHECK(s.gpu_inflight_bytes_max.at(0) <= cap);
    CHECK(s.lane_concurrency_max.at(0) < 5);
    CHECK(s[counter::refused_inflight] >= 1);
  }
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: auto lane widths (C2C sizes the GPU lane; DISK is always 1)",
                 "[preload_executor]")
{
  auto cfg                = config();
  cfg.concurrent_preloads = std::nullopt;  // auto: C2C sizes the GPU lane; DISK is 1 at this pin
  SECTION("no C2C: both lanes are 1")
  {
    build(cfg, {false, false});
    CHECK(exec->lane_threads_for_testing() == std::vector<int>{1, 1});
  }
  SECTION("GDS does not raise the DISK lane (no GDS path at this pin)")
  {
    build(cfg, {false, true});
    CHECK(exec->lane_threads_for_testing() == std::vector<int>{1, 1});
  }
  SECTION("C2C raises the HOST->GPU lane")
  {
    build(cfg, {true, false});
    CHECK(exec->lane_threads_for_testing() == std::vector<int>{5, 1});
  }
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: DISK->GPU without GDS shares the HOST->GPU tokens",
                 "[preload_executor]")
{
  bool const gds          = GENERATE(false, true);
  auto cfg                = config();
  cfg.concurrent_preloads = 1;
  build(cfg, {false, gds});
  auto const q     = open_query(1);
  auto const state = pipeline_state(priority(q, 1));
  auto const h2g   = spilled(*host);
  auto const d2g   = spilled(*disk);
  promotion_gate gate;
  auto gate_guard = arm_gate(gate);
  std::atomic<bool> disk_started{false};
  exec->set_promotion_hook_for_testing([&](promotion_job const& job) {
    if (job.batch_id == h2g->get_batch_id()) {
      gate.enter();
    } else {
      disk_started = true;
    }
  });
  push_task(state, {h2g});
  push_task(state, {d2g});
  downgraded(q, record(h2g));
  REQUIRE(gate.wait_entered(1));
  downgraded(q, record(d2g));
  REQUIRE(handled());

  if (gds) {
    auto const deadline = std::chrono::steady_clock::now() + 20s;
    while (!disk_started && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    CHECK(disk_started);
  } else {
    // Once the DISK item is planned, only the missing token keeps a worker from taking it.
    REQUIRE(eventually([&] { return stat(*exec, counter::passes) >= 2; }));
    std::this_thread::sleep_for(200ms);
    CHECK_FALSE(disk_started);  // the HOST->GPU copy holds the only token
  }
  gate.release();
  settle();
  CHECK(disk_started);
  CHECK(tier_of(*h2g) == Tier::GPU);
  CHECK(tier_of(*d2g) == Tier::GPU);
  CHECK(stat(*exec, counter::promoted_d2g) == 1);
}

// ---------------------------------------------------------------------------------------------
// 12: race with the consumer's own preparation
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: racing lock_and_prepare_batch never clones or leaks",
                 "[preload_executor]")
{
  build();
  auto const q             = open_query(1);
  auto const state         = pipeline_state(priority(q, 1));
  auto const baseline      = gpu().get_available_memory();
  constexpr int iterations = 1000;
  for (int i = 0; i < iterations; ++i) {
    auto const batch = spilled(*host, 256);
    auto const id    = batch->get_batch_id();
    push_task(state, {batch});
    downgraded(q, record(batch));
    {
      auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, &gpu(), stream);
      REQUIRE(prepared.has_value());
      REQUIRE(std::holds_alternative<sirius::pipeline::lock_to_existing_batch>(*prepared));
      CHECK(std::get<sirius::pipeline::lock_to_existing_batch>(*prepared)
              .ro_lock.get_memory_space()
              ->get_tier() == Tier::GPU);
    }
    settle();
    CHECK(batch->get_batch_id() == id);
    CHECK(tier_of(*batch) == Tier::GPU);
    while (queue.try_pop()) {}
  }
  stream.sync();
  CHECK(gpu().get_available_memory() == baseline);
  CHECK(stat(*exec, counter::failed) == 0);
}

// ---------------------------------------------------------------------------------------------
// 13-14: lost wake-ups and stale ready lists
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: an event during a refused attempt forces one retry",
                 "[preload_executor]")
{
  bool const event_during_attempt = GENERATE(true, false);
  auto const cfg                  = config();
  build(cfg);
  auto const q = open_query(1);
  // The record understates the batch, so the planner sees room for its estimate while the
  // worker, after growing to the exact need, is refused by the ceiling.
  auto const batch = spilled(*host, 1 << 20);
  push_task(pipeline_state(priority(q, 1)), {batch});
  auto filler = fill_gpu_to(committed_ceiling(cfg) - 3 * mib);

  std::atomic<int> attempts{0};
  std::atomic<bool> event_handled{!event_during_attempt};
  exec->set_promotion_hook_for_testing([&](promotion_job const&) {
    if (attempts.fetch_add(1) == 0 && event_during_attempt) {
      completed(q);
      event_handled = handled();
    }
  });
  downgraded(q, record(batch, sirius::event::downgrade_source::task_queue, 0, 0, 1));
  settle();
  REQUIRE(event_handled);
  // Quiescent and without a timer: nothing more can happen until the next event.
  CHECK(attempts == (event_during_attempt ? 2 : 1));
  CHECK(stat(*exec, counter::gated) == static_cast<std::uint64_t>(attempts.load()));
  CHECK(tier_of(*batch) == Tier::HOST);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: after a refusal no lane retries the space before an event",
                 "[preload_executor]")
{
  auto const cfg = config();
  build(cfg, {false, true});  // with GDS the DISK lane also promotes to the GPU concurrently
  auto const q     = open_query(1);
  auto const state = pipeline_state(priority(q, 1));
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 3; ++i) {
    batches.push_back(spilled(*host, 1 << 20));
    batches.push_back(spilled(*disk, 1 << 20));
  }
  auto filler = fill_gpu_to(committed_ceiling(cfg) - 3 * mib);
  std::atomic<int> attempts{0};
  exec->set_promotion_hook_for_testing([&](promotion_job const& job) {
    if (job.t.tier == Tier::GPU) { ++attempts; }
  });
  {
    std::optional<sirius::memory::memory_wait_board::waiter> waiter{board->enter_wait(0)};
    for (auto const& b : batches) {
      push_task(state, {b});
      downgraded(q, record(b, sirius::event::downgrade_source::task_queue, 0, 0, 1));
    }
    settle();
  }
  wait_ended(q);
  settle();
  // At most one job per lane was dispatched before the first refusal gated the GPU.
  auto const first = attempts.load();
  CHECK(first >= 1);
  CHECK(first <= 2);
  // Quiescent: no lane may try the GPU again before the next memory event.
  REQUIRE(exec->wait_until_quiescent_for_testing(30s, expected_events));
  CHECK(attempts == first);

  completed(q);
  settle();
  CHECK(attempts > first);
  CHECK(attempts <= first + 2);
}

// ---------------------------------------------------------------------------------------------
// Staging pressure and budget consistency
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: staging held off by GPU pressure resumes on a downgrade event",
                 "[preload_executor]")
{
  build();
  auto const plain_source = std::make_shared<barrier_pipeline>();
  cucascade::data_repository repo;
  auto const qid   = sirius::make_query_id(1);
  auto const q     = open_query(1, {repo_of(repo, priority(qid, 2), nullptr, {plain_source})});
  auto const batch = spilled(*disk);
  repo.add_data_batch(batch, 0);

  // The NUMA-local GPU is at its trigger, with no task waiting on the board.
  auto filler = fill_gpu_to(gpu_trigger_threshold() + mib);
  REQUIRE(gpu().should_downgrade_memory());
  downgraded(
    q,
    record(
      batch, sirius::event::downgrade_source::repository, reinterpret_cast<std::uintptr_t>(&repo)));
  settle();
  CHECK(tier_of(*batch) == Tier::DISK);
  CHECK(stat(*exec, counter::refused_held_off) >= 1);

  // The downgrade frees GPU memory and reports the batch it moved.
  filler.reset();
  auto const spilled_by_downgrade = spilled(*host);
  downgraded(q,
             record(spilled_by_downgrade,
                    sirius::event::downgrade_source::repository,
                    reinterpret_cast<std::uintptr_t>(&repo)));
  settle();
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(stat(*exec, counter::promoted_d2h) == 1);
}

TEST_CASE(
  "preload executor: planner budget agrees with admission when the limit is below "
  "capacity",
  "[preload_executor]")
{
  executor_fixture f{fixture_options{.reservation_fraction = 0.75, .trigger = 0.9, .stop = 0.8}};
  auto cfg                      = executor_fixture::config();
  cfg.device_preloading_ceiling = 0.3;  // a ceiling consumed bytes can actually reach
  f.build(cfg);
  auto const q       = f.open_query(1);
  auto const state   = f.pipeline_state(executor_fixture::priority(q, 1));
  auto const ceiling = f.committed_ceiling(cfg);
  REQUIRE(ceiling > 0);
  REQUIRE(ceiling < f.gpu().get_max_memory());

  SECTION("room for the batch: planned and admitted")
  {
    auto const batch = f.spilled(*f.host);
    f.push_task(state, {batch});
    auto filler = f.fill_gpu_to(ceiling - 8 * mib);
    f.downgraded(q, f.record(batch));
    f.settle();
    CHECK(tier_of(*batch) == Tier::GPU);
    CHECK(stat(*f.exec, counter::gated) == 0);
    CHECK(stat(*f.exec, counter::no_memory) == 0);
  }
  SECTION("no room: refused by the planner, never attempted")
  {
    auto const batch = f.spilled(*f.host);
    f.push_task(state, {batch});
    auto filler = f.fill_gpu_to(ceiling - mib / 2);
    std::atomic<int> attempts{0};
    f.exec->set_promotion_hook_for_testing([&](promotion_job const&) { ++attempts; });
    f.downgraded(q, f.record(batch));
    f.settle();
    CHECK(tier_of(*batch) == Tier::HOST);
    CHECK(attempts == 0);
    CHECK(stat(*f.exec, counter::refused_ceiling) >= 1);
  }
}

// ---------------------------------------------------------------------------------------------
// Staging-only mode and the device preloading budget
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: staging-only mode stages a DISK batch on HOST",
                 "[preload_executor]")
{
  auto cfg                     = config();
  cfg.concurrent_preloads      = 1;
  cfg.device_preloading_budget = 0.0;  // explicit width, no budget: DISK->HOST staging only
  build(cfg);
  // No GPU HOST->GPU lane exists (device preloading off): only the DISK lane. Dispatching a
  // staged item must resolve to that lane's actual index, not `gpu_count + disk_index`.
  REQUIRE(exec->lane_threads_for_testing() == std::vector<int>{1});
  auto const q     = open_query(1);
  auto const batch = spilled(*disk);
  push_task(pipeline_state(priority(q, 1)), {batch});
  downgraded(q, record(batch));
  settle();
  CHECK(tier_of(*batch) == Tier::HOST);  // staged, never promoted to a GPU
  CHECK(stat(*exec, counter::promoted_d2h) == 1);
  CHECK(stat(*exec, counter::promoted_h2g) == 0);
  CHECK(stat(*exec, counter::promoted_d2g) == 0);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: a claim releases or refuses the device preloading budget",
                 "[preload_executor]")
{
  build();  // device preloading on, budget 0.20
  auto const q = open_query(1);

  SECTION("claim before TAKE drops the job as claimed and charges nothing")
  {
    auto const batch = spilled(*host);
    push_task(pipeline_state(priority(q, 1)), {batch});
    // Hold off the GPU so no worker takes the job yet; the candidate is still watched.
    std::optional<sirius::memory::memory_wait_board::waiter> hold{board->enter_wait(0)};
    downgraded(q, record(batch));
    settle();
    REQUIRE(budget->watched_count() >= 1);
    // A task claims the batch before preload ever takes it.
    std::array<std::uint64_t, 1> const ids{batch->get_batch_id()};
    budget->claim(ids);
    hold.reset();
    wait_ended(q);  // reopen the GPU and trigger a pass
    settle();
    CHECK(tier_of(*batch) == Tier::HOST);  // never promoted
    CHECK(stat(*exec, counter::claimed) >= 1);
    CHECK(budget->outstanding(0) == 0);
    CHECK(stat(*exec, counter::promoted_h2g) == 0);
  }

  SECTION("claim during the copy releases immediately; landing does not recharge")
  {
    promotion_gate gate;
    auto gate_guard = arm_gate(gate);
    exec->set_promotion_hook_for_testing([&](promotion_job const&) { gate.enter(); });
    auto const batch = spilled(*host);
    push_task(pipeline_state(priority(q, 1)), {batch});
    downgraded(q, record(batch));
    REQUIRE(gate.wait_entered(1));        // the worker took the job (charged) and is in the hook
    REQUIRE(budget->outstanding(0) > 0);  // charged at TAKE
    std::array<std::uint64_t, 1> const ids{batch->get_batch_id()};
    budget->claim(ids);                  // a task claims it mid-copy
    CHECK(budget->outstanding(0) == 0);  // released immediately
    gate.release();
    settle();
    CHECK(tier_of(*batch) == Tier::GPU);  // the copy still landed
    CHECK(budget->outstanding(0) == 0);   // landing did not recharge
    CHECK(budget->claimed_releases() >= 1);
  }

  SECTION("claim after the conversion completes but before the outcome is installed")
  {
    promotion_gate gate;
    auto gate_guard = arm_gate(gate);
    // The post-convert hook runs once the conversion is done and before the outcome installs.
    exec->set_post_convert_hook_for_testing([&](promotion_job const&) { gate.enter(); });
    auto const batch = spilled(*host);
    push_task(pipeline_state(priority(q, 1)), {batch});
    downgraded(q, record(batch));
    REQUIRE(gate.wait_entered(1));        // conversion complete, outcome not yet installed
    REQUIRE(budget->outstanding(0) > 0);  // still charged from TAKE
    std::array<std::uint64_t, 1> const ids{batch->get_batch_id()};
    budget->claim(ids);
    CHECK(budget->outstanding(0) == 0);  // released in exactly this window
    gate.release();
    settle();
    CHECK(tier_of(*batch) == Tier::GPU);
    CHECK(budget->outstanding(0) == 0);  // installing the outcome did not recharge
    CHECK(budget->claimed_releases() >= 1);
  }

  SECTION("claim after landing releases the persisted charge")
  {
    auto const batch = spilled(*host);
    push_task(pipeline_state(priority(q, 1)), {batch});
    downgraded(q, record(batch));
    settle();
    REQUIRE(tier_of(*batch) == Tier::GPU);
    REQUIRE(budget->outstanding(0) > 0);  // the charge persists after landing until claimed
    std::array<std::uint64_t, 1> const ids{batch->get_batch_id()};
    budget->claim(ids);
    CHECK(budget->outstanding(0) == 0);
    CHECK(budget->claimed_releases() >= 1);
  }
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: the budget window fits two in-flight jobs but not three",
                 "[preload_executor]")
{
  auto cfg                = config();
  cfg.concurrent_preloads = 3;  // three lanes, so three copies could run at once
  // One batch's reserve estimate is about 1 MiB (with_slack of a 4096-row INT32 batch); a 2.5 MiB
  // budget fits two but not three. If in-flight bytes were double-counted, only one would fit.
  cfg.device_preloading_budget = 2.5 / 1024.0;
  cfg.max_inflight_fraction    = 1.0;  // keep the physical in-flight cap from binding first
  cfg.max_batch_fraction       = 1.0;
  build(cfg);
  promotion_gate gate;
  auto gate_guard = arm_gate(gate);
  exec->set_promotion_hook_for_testing([&](promotion_job const&) { gate.enter(); });
  auto const q     = open_query(1);
  auto const state = pipeline_state(priority(q, 1));
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 3; ++i) {
    batches.push_back(spilled(*host));
    push_task(state, {batches.back()});
    downgraded(q, record(batches.back()));
  }
  // Exactly two charges fit the window and reach the copy; the third is refused by the window.
  REQUIRE(gate.wait_entered(2));
  REQUIRE(eventually([&] { return stat(*exec, counter::refused_window) >= 1; }));
  CHECK(gate.count() == 2);  // never three while the two are held
  gate.release();
  settle();
}

// ---------------------------------------------------------------------------------------------
// 7: bans
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: the first OOM bans the GPU and the batch is staged instead",
                 "[preload_executor]")
{
  build();
  auto const q     = open_query(1);
  auto const batch = spilled(*disk);
  push_task(pipeline_state(priority(q, 1)), {batch});
  std::atomic<int> gpu_attempts{0};
  exec->set_promotion_hook_for_testing([&](promotion_job const& job) {
    if (job.t.tier != Tier::GPU) { return; }
    ++gpu_attempts;
    throw rmm::out_of_memory("preload test: copy overran its reservation");
  });

  downgraded(q, record(batch));
  settle();
  // The first OOM bans the GPU for the candidate's lifetime: no retry, then HOST staging only.
  CHECK(gpu_attempts == 1);
  CHECK(stat(*exec, counter::oom) == 1);
  CHECK(stat(*exec, counter::gpu_banned) == 1);
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(stat(*exec, counter::promoted_d2h) == 1);

  completed(q);
  settle();
  CHECK(gpu_attempts == 1);
  CHECK(tier_of(*batch) == Tier::HOST);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: a batch too big for the GPU is staged instead",
                 "[preload_executor]")
{
  auto cfg               = config();
  cfg.max_batch_fraction = 0.002;  // about 2 MiB of a 1 GiB GPU
  build(cfg);
  auto const q     = open_query(1);
  auto const batch = spilled(*disk, 1 << 20);  // 4 MiB
  push_task(pipeline_state(priority(q, 1)), {batch});

  // The record understates the batch, so only the worker finds it too big.
  downgraded(q, record(batch, sirius::event::downgrade_source::task_queue, 0, 0, 1));
  settle();
  CHECK(stat(*exec, counter::too_big) == 1);
  CHECK(stat(*exec, counter::gpu_banned) == 1);
  CHECK(tier_of(*batch) == Tier::HOST);
  CHECK(stat(*exec, counter::promoted_d2h) == 1);
  CHECK(stat(*exec, counter::promoted_d2g) == 0);

  completed(q);
  settle();
  CHECK(stat(*exec, counter::too_big) == 1);
  CHECK(tier_of(*batch) == Tier::HOST);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: a speculative batch that bounced is not promoted again",
                 "[preload_executor]")
{
  build();
  cucascade::data_repository repo;
  partition_consumer consumer;
  consumer.set_placement(std::make_shared<sirius::op::partition_placement const>(
    sirius::op::partition_placement::round_robin(2, {0})));
  auto const qid   = sirius::make_query_id(1);
  auto const q     = open_query(1, {repo_of(repo, priority(qid, 2), &consumer)});
  auto const token = reinterpret_cast<std::uintptr_t>(&repo);
  std::vector<std::shared_ptr<cucascade::data_batch>> parts;
  for (std::size_t p = 0; p < 2; ++p) {
    parts.push_back(spilled(*host));
    repo.add_data_batch(parts.back(), p);
  }
  for (std::size_t p = 0; p < parts.size(); ++p) {
    downgraded(q, record(parts[p], sirius::event::downgrade_source::repository, token, p));
  }
  settle();
  REQUIRE(tier_of(*parts[1]) == Tier::GPU);
  CHECK(exec->outstanding_speculative_bytes_for_testing(0) > 0);

  move_to(parts[1], *host);
  downgraded(q, record(parts[1], sirius::event::downgrade_source::repository, token, 1));
  settle();
  CHECK(stat(*exec, counter::bounce) == 1);
  CHECK(exec->outstanding_speculative_bytes_for_testing(0) == 0);

  completed(q);
  settle();
  CHECK(tier_of(*parts[1]) == Tier::HOST);
  CHECK(stat(*exec, counter::promoted_h2g) == 1);
}

// ---------------------------------------------------------------------------------------------
// Windows, scan budget, staging switch and ledger verdicts
// ---------------------------------------------------------------------------------------------

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: bytes in flight count against the lookahead window",
                 "[preload_executor]")
{
  auto cfg                = config();
  cfg.concurrent_preloads = 2;
  // About 1.5 MiB: one batch's reservation fits, two do not.
  cfg.device_preloading_budget = 1.5 / 1024.0;
  build(cfg);
  auto const q     = open_query(1);
  auto const state = pipeline_state(priority(q, 1));
  auto const first = spilled(*host);
  auto const other = spilled(*host);
  push_task(state, {first});
  push_task(state, {other});
  promotion_gate gate;
  auto gate_guard = arm_gate(gate);
  exec->set_promotion_hook_for_testing([&](promotion_job const&) { gate.enter(); });

  downgraded(q, record(first));
  REQUIRE(gate.wait_entered(1));
  // A pass while the first copy is parked: its bytes already fill the window.
  downgraded(q, record(other));
  REQUIRE(eventually([&] { return stat(*exec, counter::refused_window) >= 1; }));
  CHECK(gate.count() == 1);

  gate.release();
  settle();
  CHECK(tier_of(*first) == Tier::GPU);
  CHECK(tier_of(*other) == Tier::HOST);
  CHECK(stat(*exec, counter::promoted_h2g) == 1);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: the repository scan budget reaches every repository",
                 "[preload_executor]")
{
  auto cfg              = config();
  cfg.repository_window = 1;
  build(cfg);
  constexpr std::size_t n_repos = 70;  // more than one pass scans
  auto const source             = std::make_shared<barrier_pipeline>();
  auto const qid                = sirius::make_query_id(1);
  std::vector<std::unique_ptr<cucascade::data_repository>> repos;
  std::vector<repo_info> infos;
  for (std::size_t i = 0; i < n_repos; ++i) {
    repos.push_back(std::make_unique<cucascade::data_repository>());
    infos.push_back(repo_of(*repos.back(), priority(qid, 2), nullptr, {source}));
  }
  auto const q = open_query(1, infos);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (auto const& repo : repos) {
    batches.push_back(spilled(*disk));
    repo->add_data_batch(batches.back(), 0);
  }
  for (std::size_t i = 0; i < n_repos; ++i) {
    downgraded(q,
               record(batches[i],
                      sirius::event::downgrade_source::repository,
                      reinterpret_cast<std::uintptr_t>(repos[i].get())));
  }
  settle();
  for (auto const& b : batches) {
    CHECK(tier_of(*b) == Tier::HOST);
  }
  CHECK(stat(*exec, counter::promoted_d2h) == n_repos);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: repeated idle events ask for one pass",
                 "[preload_executor]")
{
  build();
  auto const q     = open_query(1);
  auto const batch = spilled(*host);
  // A candidate with no queued consumer: passes plan nothing, and it stays a candidate.
  downgraded(q, record(batch));
  settle();
  auto const passes = stat(*exec, counter::passes);

  // The scheduler publishes these on every turn of its loop while a device cannot take a task.
  for (int i = 0; i < 50; ++i) {
    awaiting(0);
    queue_empty();
  }
  settle();
  CHECK(stat(*exec, counter::passes) == passes + 1);

  // Any other event re-arms the idle wake-up.
  task_created(q);
  settle();
  awaiting(0);
  settle();
  CHECK(stat(*exec, counter::passes) == passes + 3);
  CHECK(tier_of(*batch) == Tier::HOST);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: stage_to_host off disables DISK->HOST staging",
                 "[preload_executor]")
{
  auto cfg          = config();
  cfg.stage_to_host = false;
  build(cfg);
  auto const source = std::make_shared<barrier_pipeline>();
  cucascade::data_repository repo;
  auto const qid   = sirius::make_query_id(1);
  auto const q     = open_query(1, {repo_of(repo, priority(qid, 2), nullptr, {source})});
  auto const batch = spilled(*disk);
  repo.add_data_batch(batch, 0);
  downgraded(
    q,
    record(
      batch, sirius::event::downgrade_source::repository, reinterpret_cast<std::uintptr_t>(&repo)));
  settle();
  CHECK(stat(*exec, counter::passes) >= 1);
  CHECK(tier_of(*batch) == Tier::DISK);
  CHECK(stat(*exec, counter::promoted_d2h) == 0);
}

TEST_CASE_METHOD(executor_fixture,
                 "preload executor: a consumed batch is a hit, a claimed speculative one turns A",
                 "[preload_executor]")
{
  SECTION("class A: its task ran")
  {
    build();
    auto const q     = open_query(1);
    auto const batch = spilled(*host);
    push_task(pipeline_state(priority(q, 1)), {batch});
    downgraded(q, record(batch));
    settle();
    REQUIRE(tier_of(*batch) == Tier::GPU);
    REQUIRE(exec->outstanding_bytes_for_testing(0) > 0);
    CHECK(stat(*exec, counter::hit) == 0);

    while (queue.try_pop()) {}  // the task ran: its inputs are unsubscribed
    completed(q);
    settle();
    CHECK(stat(*exec, counter::hit) == 1);
    CHECK(exec->outstanding_bytes_for_testing(0) == 0);
  }

  SECTION("class B: a task claimed it")
  {
    build();
    cucascade::data_repository repo;
    partition_consumer consumer;
    consumer.set_placement(std::make_shared<sirius::op::partition_placement const>(
      sirius::op::partition_placement::round_robin(2, {0})));
    auto const qid   = sirius::make_query_id(1);
    auto const q     = open_query(1, {repo_of(repo, priority(qid, 2), &consumer)});
    auto const token = reinterpret_cast<std::uintptr_t>(&repo);
    std::vector<std::shared_ptr<cucascade::data_batch>> parts;
    for (std::size_t p = 0; p < 2; ++p) {
      parts.push_back(spilled(*host));
      repo.add_data_batch(parts.back(), p);
      downgraded(q, record(parts.back(), sirius::event::downgrade_source::repository, token, p));
    }
    settle();
    REQUIRE(tier_of(*parts[1]) == Tier::GPU);
    auto const outstanding = exec->outstanding_bytes_for_testing(0);
    REQUIRE(exec->outstanding_speculative_bytes_for_testing(0) == outstanding);

    push_task(pipeline_state(priority(q, 2)), {parts[1]});  // subscribes it
    task_created(q);
    settle();
    CHECK(exec->outstanding_bytes_for_testing(0) == outstanding);
    CHECK(exec->outstanding_speculative_bytes_for_testing(0) == 0);
  }
}

// ---------------------------------------------------------------------------------------------
// Start-up checks
// ---------------------------------------------------------------------------------------------

namespace {

bool logged(sirius::test::scoped_recording_log_sink const& sink, std::string_view text)
{
  auto const records = sink.records();
  return std::any_of(records.begin(), records.end(), [&](auto const& r) {
    return r.level == sirius::log::level::warn && r.message.find(text) != std::string::npos;
  });
}

}  // namespace

TEST_CASE("preload executor: start-up warnings", "[preload_executor]")
{
  sirius::test::scoped_recording_log_sink sink{"warn"};

  SECTION("negative core ids are skipped")
  {
    executor_fixture f;
    auto cfg              = executor_fixture::config();
    cfg.cpu_affinity_list = {-1, 0};
    f.build(cfg);
    CHECK(logged(sink, "ignoring negative core ids"));
    auto const q     = f.open_query(1);
    auto const batch = f.spilled(*f.host);
    f.push_task(f.pipeline_state(executor_fixture::priority(q, 1)), {batch});
    f.downgraded(q, f.record(batch));
    f.settle();
    CHECK(tier_of(*batch) == Tier::GPU);
  }
  SECTION("zero ceilings")
  {
    executor_fixture f;
    auto cfg                              = executor_fixture::config();
    cfg.gpu_speculative_headroom_fraction = 1.0;  // speculative ceiling 0
    cfg.host_headroom_fraction            = 1.0;  // host staging ceiling 0
    f.build(cfg);
    CHECK(logged(sink, "speculative ceiling is 0"));
    CHECK(logged(sink, "staging ceiling is 0"));
  }
  SECTION("per-stream reservation tracking disables device preloading")
  {
    // The fixture's build() passes no streams or budget board when the GPU space is per-stream,
    // so the executor runs staging only: a single DISK lane, no GPU HOST->GPU lane.
    executor_fixture f{fixture_options{.per_stream = true}};
    f.build();
    CHECK(f.exec->lane_threads_for_testing().size() == 1);
  }
}

// ---------------------------------------------------------------------------------------------
// 11: several GPUs
// ---------------------------------------------------------------------------------------------

TEST_CASE("preload executor: unpinned work on several GPUs is only staged",
          "[preload_executor][multi_gpu]")
{
  int devices = 0;
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  if (devices < 2) { SKIP("needs two GPUs"); }
  executor_fixture f{fixture_options{.gpus = 2}};
  f.build();
  auto const q         = f.open_query(1, {}, {0, 1});
  auto const state     = f.pipeline_state(executor_fixture::priority(q, 1));
  auto const on_disk   = f.spilled(*f.disk);
  auto const on_host   = f.spilled(*f.host);
  auto const preferred = f.spilled(*f.host);
  f.push_task(state, {on_disk}, std::nullopt);
  f.push_task(state, {on_host}, std::nullopt);
  f.push_task(state, {preferred}, 1);
  f.downgraded(q, f.record(on_disk));
  f.downgraded(q, f.record(on_host));
  f.downgraded(q, f.record(preferred));
  f.settle();
  CHECK(tier_of(*on_disk) == Tier::HOST);
  CHECK(tier_of(*on_host) == Tier::HOST);
  {
    auto ro = preferred->to_read_only();
    CHECK(ro.get_memory_space()->get_id() == cucascade::memory::memory_space_id{Tier::GPU, 1});
  }
  CHECK(stat(*f.exec, counter::promoted_d2h) == 1);
  CHECK(stat(*f.exec, counter::promoted_h2g) == 1);
  CHECK(stat(*f.exec, counter::promoted_d2g) == 0);
}

// ---------------------------------------------------------------------------------------------
// Repository discovery
// ---------------------------------------------------------------------------------------------

TEST_CASE("preload executor: repositories of an operator", "[preload_executor]")
{
  auto const producer          = std::make_shared<barrier_pipeline>();
  auto const consumer_pipeline = std::make_shared<barrier_pipeline>();
  consumer_pipeline->set_priority(42);
  cucascade::data_repository full_repo;
  cucascade::data_repository pipelined_repo;
  partition_consumer op;
  auto add_port = [&](std::string const& id,
                      sirius::op::MemoryBarrierType type,
                      cucascade::shared_data_repository* repo) {
    auto port           = std::make_unique<sirius::op::sirius_physical_operator::port>();
    port->type          = type;
    port->repo          = repo;
    port->src_pipeline  = producer;
    port->dest_pipeline = consumer_pipeline;
    op.add_port(id, std::move(port));
  };
  add_port("build", sirius::op::MemoryBarrierType::FULL, &full_repo);
  add_port("probe", sirius::op::MemoryBarrierType::PIPELINE, &pipelined_repo);
  add_port("dependency", sirius::op::MemoryBarrierType::FULL, nullptr);

  auto const repos = repositories_of(op);
  REQUIRE(repos.size() == 2);
  for (auto const& r : repos) {
    CHECK(r.dest_priority == 42);
    CHECK(r.partition_consumer == &op);
    CHECK(r.full_sources.size() == 2);
  }
  CHECK(repos[0].repo != repos[1].repo);
}
