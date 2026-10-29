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

// The preload executor driven by the two events it consumes, task_downgraded and task_popped,
// published by hand: no scheduler, no downgrade executor. Each test waits until every event it
// published has been handled and the workers are idle, then reads the counters, the budget and
// where the batches are.

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "event/downgraded_task.hpp"
#include "event/query_event_publisher.hpp"
#include "event/query_event_subscriber.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "preload/config.hpp"
#include "preload/preload_executor.hpp"
#include "preload/preload_streams.hpp"
#include "preload/promotion.hpp"
#include "query_id.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/utils.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using cucascade::memory::Tier;
using namespace sirius::preload;
using namespace std::chrono_literals;

namespace {

constexpr std::size_t gib = std::size_t{1} << 30;

/// A tasks_left at or below any pipeline thread count used here: the gate is open.
constexpr std::size_t short_queue = 0;
/// A tasks_left above every pipeline thread count used here: the gate is closed.
constexpr std::size_t deep_queue = 100;
/// The pipeline every candidate belongs to unless a test says otherwise.
constexpr std::size_t candidate_pipeline = 1;
/// The pipeline every pop belongs to unless a test says otherwise.
constexpr std::size_t other_pipeline = 2;

using batch_ptr = std::shared_ptr<cucascade::data_batch>;

/// Holds promotions inside promote_one, on the worker, until released; counts them.
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

  bool wait_entered(int n, std::chrono::milliseconds limit = 30s)
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

/// A subscriber of its own to the two events the executor consumes; counts what it receives.
class event_probe final : public sirius::event::query_event_subscriber {
 public:
  explicit event_probe(sirius::event::query_event_publisher& publisher)
    : query_event_subscriber(
        publisher,
        {sirius::event::event_type::task_downgraded, sirius::event::event_type::task_popped})
  {
    start();
  }

  ~event_probe() override { stop(); }

  event_probe(event_probe const&)            = delete;
  event_probe& operator=(event_probe const&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override { return "preload_probe"; }

  /// At least @p n events have arrived, by @p limit.
  bool wait_for(std::uint64_t n, std::chrono::milliseconds limit = 30s)
  {
    std::unique_lock lk{_mu};
    return _cv.wait_for(lk, limit, [&] { return _received >= n; });
  }

 private:
  void on_task_downgraded(sirius::event::event_id_t,
                          sirius::event::timestamp_t,
                          sirius::query_id_t,
                          sirius::event::downgraded_task const&) noexcept override
  {
    received();
  }

  void on_task_popped(sirius::event::event_id_t,
                      sirius::event::timestamp_t,
                      sirius::query_id_t,
                      std::uint64_t,
                      std::size_t,
                      int,
                      std::size_t) noexcept override
  {
    received();
  }

  void received()
  {
    std::lock_guard lk{_mu};
    ++_received;
    _cv.notify_all();
  }

  std::mutex _mu;
  std::condition_variable _cv;
  std::uint64_t _received{0};
};

struct fixture_options {
  int gpus{1};
  std::size_t gpu_capacity{gib};
  /// Reservable fraction of the capacity. Below 1, reservations hit the limit while consumed
  /// bytes (capacity-based) are still 0.
  double gpu_reservation{1.0};
  double gpu_trigger{0.9};
  double gpu_stop{0.8};
};

/**
 * GPU, HOST and DISK spaces, a publisher, a lifecycle with one open query and preload stream
 * pools; helpers to spill batches, to publish the two events and to wait for them. Each test
 * builds the executor with its own configuration.
 */
struct executor_fixture {
  fixture_options opts;
  sirius::test::scratch_dir disk_dir{"preload_executor"};
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  std::vector<cucascade::memory::memory_space*> gpus;
  cucascade::memory::memory_space* host{nullptr};
  cucascade::memory::memory_space* disk{nullptr};
  std::shared_ptr<sirius::event::query_event_publisher> publisher{
    std::make_shared<sirius::event::query_event_publisher>()};
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::query_id_t const query{sirius::make_query_id(1)};
  std::shared_ptr<preload_streams> streams;
  rmm::cuda_stream stream_owner{};
  ::cuda::stream_ref stream{stream_owner.value()};
  std::uint64_t published{0};
  std::unique_ptr<preload_executor> exec;

  explicit executor_fixture(fixture_options o = {}) : opts(o)
  {
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(static_cast<std::size_t>(opts.gpus))
      .set_gpu_usage_limit(opts.gpu_capacity)
      .set_reservation_fraction_per_gpu(opts.gpu_reservation)
      .set_downgrade_fractions_per_gpu(opts.gpu_trigger, opts.gpu_stop)
      .track_reservation_per_stream(false)
      .set_per_numa_region_capacity(4 * gib)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, 4 * gib, disk_dir.path().string());
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
    streams = std::make_shared<preload_streams>(devices);
    lifecycle.open_query(query);
  }

  ~executor_fixture()
  {
    if (exec) { exec->stop(); }
    exec.reset();
  }

  executor_fixture(executor_fixture const&)            = delete;
  executor_fixture& operator=(executor_fixture const&) = delete;

  [[nodiscard]] cucascade::memory::memory_space& gpu() const { return *gpus.front(); }

  /// What preload is told about each space; the pinned cuCascade exposes neither value.
  [[nodiscard]] space_capacity_fn space_capacity() const
  {
    auto const o = opts;
    return [o](cucascade::memory::memory_space_id id) -> space_capacity_info {
      if (id.tier == Tier::GPU) { return {o.gpu_capacity, o.gpu_stop}; }
      // set_per_numa_region_capacity(4 GiB); the configurator's default HOST stop fraction.
      return {4 * gib, 0.65};
    };
  }

  /// Device promotion with a generous budget; no DISK->HOST staging.
  [[nodiscard]] static preload_executor_config config()
  {
    preload_executor_config cfg;
    cfg.device_budget = 0.2;
    return cfg;
  }

  [[nodiscard]] preload_dependencies deps(int pipeline_threads, bool with_streams)
  {
    return preload_dependencies{.lifecycle                = lifecycle,
                                .memory                   = *mgr,
                                .topology                 = nullptr,
                                .streams                  = with_streams ? streams : nullptr,
                                .pipeline_threads_per_gpu = pipeline_threads,
                                .space_capacity           = space_capacity()};
  }

  /// Build and start the executor. @p hook runs inside every promotion (see
  /// set_promotion_hook_for_testing); it must outlive the executor.
  void build(preload_executor_config const& cfg,
             int pipeline_threads                           = 2,
             std::function<void(promotion_job const&)> hook = {})
  {
    exec = std::make_unique<preload_executor>(
      *publisher, cfg, deps(pipeline_threads, cfg.device_budget > 0.0));
    if (hook) { exec->set_promotion_hook_for_testing(std::move(hook)); }
    exec->start();
  }

  /// On every exit path, opens @p gate and stops the executor while the gate is alive, so no
  /// worker is left blocked in the gate.
  [[nodiscard]] auto arm_gate(promotion_gate& gate)
  {
    return absl::Cleanup{[this, &gate] {
      gate.release();
      if (exec) { exec->stop(); }
    }};
  }

  [[nodiscard]] batch_ptr make_gpu_batch(std::size_t rows = 4096)
  {
    std::vector<cudf::data_type> types{cudf::data_type{cudf::type_id::INT32}};
    std::vector<std::optional<std::pair<int, int>>> ranges{std::make_pair(0, 100000)};
    auto table = sirius::create_cudf_table_with_random_data(
      rows, types, ranges, stream, gpu().get_default_allocator());
    stream.sync();
    return sirius::make_data_batch(
      std::move(table), gpu(), stream, sirius::telemetry::batch_telemetry_info{});
  }

  /// Move @p batch to @p space as a host or disk table, the way a downgrade does.
  void move_to(batch_ptr const& batch, cucascade::memory::memory_space& space)
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

  /// A GPU batch of @p rows moved to @p space.
  [[nodiscard]] batch_ptr spilled(cucascade::memory::memory_space& space, std::size_t rows = 4096)
  {
    auto batch = make_gpu_batch(rows);
    move_to(batch, space);
    return batch;
  }

  /// @p batch as a task_downgraded input, where it is now.
  [[nodiscard]] static sirius::event::downgraded_input input(batch_ptr const& batch)
  {
    auto ro = batch->to_read_only();
    return {.batch     = batch,
            .batch_id  = batch->get_batch_id(),
            .to        = ro.get_memory_space()->get_id(),
            .gpu_bytes = ro.get_data()->get_size_in_bytes()};
  }

  /// The priority of the task of @p owner (default: the fixture's query) that pops @p rank-th.
  [[nodiscard]] sirius::exec::queue_priority priority(
    std::uint64_t rank, std::optional<sirius::query_id_t> owner = std::nullopt) const
  {
    return static_cast<sirius::exec::queue_priority>(
      sirius::query_priority_bits(owner.value_or(query)) | static_cast<std::int64_t>(rank));
  }

  /// A task_downgraded payload for task @p task_id (pop order: its id) with @p batches.
  [[nodiscard]] sirius::event::downgraded_task task(
    std::uint64_t task_id,
    std::vector<batch_ptr> const& batches,
    std::size_t pipeline                    = candidate_pipeline,
    std::optional<int> device               = 0,
    std::optional<sirius::query_id_t> owner = std::nullopt) const
  {
    sirius::event::downgraded_task t{.task_id          = task_id,
                                     .pipeline_id      = pipeline,
                                     .preferred_device = device,
                                     .priority         = priority(task_id, owner),
                                     .inputs           = {}};
    for (auto const& b : batches) {
      t.inputs.push_back(input(b));
    }
    return t;
  }

  /// Publish task_downgraded with @p t for @p owner (default: the fixture's query).
  void downgraded(sirius::event::downgraded_task t,
                  std::optional<sirius::query_id_t> owner = std::nullopt)
  {
    ++published;
    publisher->publish_task_downgraded(owner.value_or(query), std::move(t));
  }

  /// Publish task_downgraded for task @p task_id (pop order: its id) with @p batches.
  void downgraded(std::uint64_t task_id,
                  std::vector<batch_ptr> const& batches,
                  std::size_t pipeline                    = candidate_pipeline,
                  std::optional<int> device               = 0,
                  std::optional<sirius::query_id_t> owner = std::nullopt)
  {
    downgraded(task(task_id, batches, pipeline, device, owner), owner);
  }

  /// Publish task_popped for task @p task_id on GPU @p gpu_id.
  void popped(std::uint64_t task_id,
              std::size_t tasks_left,
              std::size_t pipeline = other_pipeline,
              int gpu_id           = 0)
  {
    ++published;
    publisher->publish_task_popped(query, task_id, pipeline, gpu_id, tasks_left);
  }

  /// Every published event has been handled.
  [[nodiscard]] bool handled()
  {
    auto const deadline = std::chrono::steady_clock::now() + 30s;
    while (exec->events_handled_for_testing() < published) {
      if (std::chrono::steady_clock::now() > deadline) { return false; }
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }

  /// Every published event has been handled and every credit it gave has been spent.
  void settle()
  {
    REQUIRE(handled());
    REQUIRE(exec->wait_idle_for_testing(30s));
  }

  [[nodiscard]] std::size_t outstanding(cucascade::memory::memory_space const& space) const
  {
    return exec->outstanding_for_testing(space.get_id());
  }

  /// Reserve GPU memory so that only @p left more bytes can be reserved.
  [[nodiscard]] std::unique_ptr<cucascade::memory::reservation> leave_reservable(std::size_t left)
  {
    auto& space = gpu();
    // Available memory is measured against the capacity, reservations against the limit.
    auto const allocated = opts.gpu_capacity - space.get_available_memory();
    REQUIRE(space.get_max_memory() > allocated + left);
    auto res = space.make_reservation_or_null(space.get_max_memory() - allocated - left);
    REQUIRE(res != nullptr);
    return res;
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
};

Tier tier_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

cucascade::memory::memory_space_id space_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_id();
}

std::size_t size_of(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_data()->get_size_in_bytes();
}

/// What a GPU promotion of @p batch, where it is now, reserves and then holds.
std::size_t gpu_need(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return with_slack(promotion_peak_bytes(*ro.get_data()));
}

/// The budget the executor charges when it selects @p batch: its size with slack.
std::size_t estimate(cucascade::data_batch& batch)
{
  return with_slack(std::max<std::size_t>(size_of(batch), 1));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// The gate and the credits
// ---------------------------------------------------------------------------------------------

TEST_CASE("preload executor: a pop promotes only past the queue-depth gate",
          "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config(), /*pipeline_threads=*/2);
  auto const a = f.spilled(*f.host);
  f.downgraded(1, {a});

  // A deep queue of another pipeline: the candidate is far from running.
  f.popped(100, /*tasks_left=*/3);
  f.settle();
  CHECK(f.exec->counters().gated == 1);
  CHECK(f.exec->counters().promoted_to_gpu == 0);
  CHECK(tier_of(*a) == Tier::HOST);
  CHECK(f.outstanding(f.gpu()) == 0);

  SECTION("a queue no deeper than the pipeline threads") { f.popped(101, /*tasks_left=*/2); }
  SECTION("a deep queue of the candidate's own pipeline")
  {
    f.popped(101, /*tasks_left=*/3, candidate_pipeline);
  }
  f.settle();
  auto const c = f.exec->counters();
  CHECK(c.pops == 2);
  CHECK(c.gated == 1);
  CHECK(c.promoted_to_gpu == 1);
  CHECK(c.promoted_to_host == 0);
  REQUIRE(tier_of(*a) == Tier::GPU);
  CHECK(space_of(*a) == f.gpu().get_id());
  // The promoted input holds its exact bytes until its task is popped.
  CHECK(c.bytes_to_gpu > 0);
  CHECK(f.outstanding(f.gpu()) == c.bytes_to_gpu);

  f.popped(1, deep_queue);
  f.settle();
  CHECK(f.outstanding(f.gpu()) == 0);
  CHECK(f.exec->candidate_count_for_testing() == 0);
  CHECK(f.exec->counters().no_candidate == 1);
}

TEST_CASE("preload executor: one pop promotes at most one input", "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());
  auto const a = f.spilled(*f.host);
  auto const b = f.spilled(*f.host);
  f.downgraded(1, {a, b});

  f.popped(100, short_queue);
  f.settle();
  CHECK(f.exec->counters().promoted_to_gpu == 1);
  // Inputs go up in the order they were spilled.
  CHECK(tier_of(*a) == Tier::GPU);
  CHECK(tier_of(*b) == Tier::HOST);

  f.popped(101, short_queue);
  f.settle();
  CHECK(f.exec->counters().promoted_to_gpu == 2);
  CHECK(tier_of(*b) == Tier::GPU);
  CHECK(f.outstanding(f.gpu()) == f.exec->counters().bytes_to_gpu);
}

TEST_CASE("preload executor: the task that pops first goes up first, whenever it was spilled",
          "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());
  auto const runs_later = f.spilled(*f.host);
  auto const runs_first = f.spilled(*f.host);
  // Task 1 is spilled first, but task 2 has the smaller priority: it pops, and runs, first.
  auto spilled_first      = f.task(1, {runs_later});
  spilled_first.priority  = f.priority(9);
  auto spilled_second     = f.task(2, {runs_first});
  spilled_second.priority = f.priority(3);
  f.downgraded(std::move(spilled_first));
  f.downgraded(std::move(spilled_second));

  f.popped(100, short_queue);
  f.settle();
  CHECK(f.exec->counters().promoted_to_gpu == 1);
  CHECK(tier_of(*runs_first) == Tier::GPU);
  CHECK(tier_of(*runs_later) == Tier::HOST);

  f.popped(101, short_queue);
  f.settle();
  CHECK(f.exec->counters().promoted_to_gpu == 2);
  CHECK(tier_of(*runs_later) == Tier::GPU);
}

TEST_CASE("preload executor: pops during a promotion wait as credits, capped at the threads",
          "[preload][preload_executor]")
{
  promotion_gate gate;
  executor_fixture f;
  f.build(executor_fixture::config(), /*pipeline_threads=*/2, [&gate](promotion_job const&) {
    gate.enter();
  });
  auto const release = f.arm_gate(gate);
  std::vector<batch_ptr> inputs;
  for (int i = 0; i < 6; ++i) {
    inputs.push_back(f.spilled(*f.host));
  }
  f.downgraded(1, inputs);

  f.popped(100, short_queue);
  REQUIRE(gate.wait_entered(1));
  // Five pops while the worker copies: only the newest two are kept. The three oldest would be
  // gated (a deep queue of another pipeline) and the two newest promote, so keeping the oldest
  // instead would promote nothing more.
  for (std::uint64_t t = 101; t <= 103; ++t) {
    f.popped(t, deep_queue);
  }
  for (std::uint64_t t = 104; t <= 105; ++t) {
    f.popped(t, short_queue);
  }
  REQUIRE(f.handled());
  CHECK(f.exec->counters().pops == 6);
  gate.release();
  REQUIRE(f.exec->wait_idle_for_testing(30s));

  CHECK(gate.count() == 3);
  CHECK(f.exec->counters().gated == 0);
  CHECK(f.exec->counters().promoted_to_gpu == 3);
  CHECK(std::ranges::count_if(inputs,
                              [](batch_ptr const& b) { return tier_of(*b) == Tier::GPU; }) == 3);
}

// ---------------------------------------------------------------------------------------------
// Budget, ceiling and host staging
// ---------------------------------------------------------------------------------------------

TEST_CASE("preload executor: the device budget caps promoted bytes until their task is popped",
          "[preload][preload_executor]")
{
  executor_fixture f;
  auto const a      = f.spilled(*f.host);
  auto const b      = f.spilled(*f.host);
  auto const need_a = gpu_need(*a);
  auto const need_b = gpu_need(*b);
  // Room for either estimate alone, but not for B's estimate on top of what A holds.
  std::size_t const budget = std::max({estimate(*a), estimate(*b), need_a}) + 64 * 1024;
  REQUIRE(need_a + estimate(*b) > budget);
  auto cfg          = executor_fixture::config();
  cfg.device_budget = static_cast<double>(budget) / static_cast<double>(f.opts.gpu_capacity);
  f.build(cfg);
  f.downgraded(1, {a});
  f.downgraded(2, {b});

  f.popped(100, short_queue);
  f.settle();
  REQUIRE(tier_of(*a) == Tier::GPU);
  CHECK(f.outstanding(f.gpu()) == need_a);

  f.popped(101, short_queue);
  f.settle();
  CHECK(f.exec->counters().no_room == 1);
  CHECK(tier_of(*b) == Tier::HOST);
  CHECK(f.outstanding(f.gpu()) == need_a);

  // A's task is popped: its bytes leave the budget, and the same pop's credit promotes B.
  f.popped(1, short_queue);
  f.settle();
  CHECK(tier_of(*b) == Tier::GPU);
  CHECK(f.outstanding(f.gpu()) == need_b);
  CHECK(f.exec->counters().promoted_to_gpu == 2);
}

TEST_CASE("preload executor: a GPU above its ceiling gets nothing; the ceiling is clamped",
          "[preload][preload_executor]")
{
  executor_fixture f;
  sirius::test::scoped_recording_log_sink logs{"warn"};
  auto cfg = executor_fixture::config();
  // Above the GPU's downgrade stop fraction (0.8): clamped to it, with one warning.
  cfg.device_ceiling = 0.9;
  f.build(cfg);
  CHECK(std::ranges::count_if(logs.records(), [](auto const& r) {
          return r.message.find("device_ceiling") != std::string::npos;
        }) == 1);

  auto const a = f.spilled(*f.host);
  f.downgraded(1, {a});
  // Between the clamped ceiling and the configured one.
  auto fill = f.fill_gpu_to(static_cast<std::size_t>(0.85 * static_cast<double>(gib)));
  f.popped(100, short_queue);
  f.settle();
  CHECK(f.exec->counters().no_room == 1);
  CHECK(tier_of(*a) == Tier::HOST);
  CHECK(f.outstanding(f.gpu()) == 0);

  fill.reset();
  f.popped(101, short_queue);
  f.settle();
  CHECK(tier_of(*a) == Tier::GPU);
}

TEST_CASE("preload executor: a DISK input the GPU cannot take is staged to HOST",
          "[preload][preload_executor]")
{
  executor_fixture f;
  auto const on_host = f.spilled(*f.host);
  auto const on_disk = f.spilled(*f.disk);
  // A staging reserves the batch's size on DISK.
  auto const disk_bytes = size_of(*on_disk);
  auto cfg              = executor_fixture::config();
  cfg.host_budget       = 0.1;
  std::unique_ptr<cucascade::memory::reservation> fill;

  SECTION("no device budget")
  {
    cfg.device_budget = 0.0;
    f.build(cfg);
  }
  SECTION("the GPU above its ceiling")
  {
    f.build(cfg);
    fill = f.fill_gpu_to(static_cast<std::size_t>(0.6 * static_cast<double>(gib)));
  }

  f.downgraded(1, {on_host, on_disk});
  f.popped(100, short_queue);
  f.settle();
  auto c = f.exec->counters();
  CHECK(c.promoted_to_host == 1);
  CHECK(c.promoted_to_gpu == 0);
  // A HOST input with no room on the GPU stays where it is.
  CHECK(tier_of(*on_host) == Tier::HOST);
  CHECK(space_of(*on_disk) == f.host->get_id());
  CHECK(c.bytes_to_host == disk_bytes);
  CHECK(f.outstanding(*f.host) == disk_bytes);
  CHECK(f.outstanding(f.gpu()) == 0);

  // Nothing left that the GPU or HOST can take.
  f.popped(101, short_queue);
  f.settle();
  c = f.exec->counters();
  CHECK(c.no_room == 1);
  CHECK(c.promoted_to_host == 1);
  CHECK(tier_of(*on_host) == Tier::HOST);

  // Popping the task returns the staging budget.
  f.popped(1, deep_queue);
  f.settle();
  CHECK(f.outstanding(*f.host) == 0);
}

// ---------------------------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------------------------

TEST_CASE("preload executor: a candidate is promoted only on a GPU it may run on",
          "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());

  auto const elsewhere = f.spilled(*f.host);
  f.downgraded(1, {elsewhere}, candidate_pipeline, /*device=*/1);
  f.popped(100, short_queue);
  f.settle();
  CHECK(f.exec->counters().no_candidate == 1);
  CHECK(tier_of(*elsewhere) == Tier::HOST);

  auto const anywhere = f.spilled(*f.host);
  f.downgraded(2, {anywhere}, candidate_pipeline, std::nullopt);
  f.popped(101, short_queue);
  f.settle();
  CHECK(f.exec->counters().promoted_to_gpu == 1);
  CHECK(tier_of(*anywhere) == Tier::GPU);
  CHECK(tier_of(*elsewhere) == Tier::HOST);
}

TEST_CASE("preload executor: each GPU's pop promotes to that GPU only what may run there",
          "[preload][preload_executor][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs two GPUs"); }
  executor_fixture f{fixture_options{.gpus = 2}};
  f.build(executor_fixture::config());
  auto const pinned   = f.spilled(*f.host);
  auto const unpinned = f.spilled(*f.host);
  f.downgraded(1, {pinned}, candidate_pipeline, /*device=*/0);
  f.downgraded(2, {unpinned}, candidate_pipeline, std::nullopt);

  f.popped(100, short_queue, other_pipeline, /*gpu_id=*/1);
  f.settle();
  CHECK(space_of(*unpinned) == f.gpus[1]->get_id());
  CHECK(tier_of(*pinned) == Tier::HOST);
  CHECK(f.outstanding(*f.gpus[1]) == f.exec->counters().bytes_to_gpu);
  CHECK(f.outstanding(*f.gpus[0]) == 0);

  f.popped(101, short_queue, other_pipeline, /*gpu_id=*/0);
  f.settle();
  CHECK(space_of(*pinned) == f.gpus[0]->get_id());
  CHECK(f.exec->counters().promoted_to_gpu == 2);
  CHECK(f.outstanding(*f.gpus[0]) > 0);
}

// ---------------------------------------------------------------------------------------------
// Re-downgrades, skipped inputs and stale events
// ---------------------------------------------------------------------------------------------

TEST_CASE("preload executor: a re-downgraded input releases its budget and is promoted again",
          "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());
  auto const a = f.spilled(*f.host);
  f.downgraded(1, {a});
  f.popped(100, short_queue);
  f.settle();
  REQUIRE(tier_of(*a) == Tier::GPU);
  REQUIRE(f.outstanding(f.gpu()) > 0);

  // The downgrade executor spills it again before its task runs.
  f.move_to(a, *f.host);
  auto const need = gpu_need(*a);
  f.downgraded(1, {a});
  f.settle();
  CHECK(f.outstanding(f.gpu()) == 0);
  CHECK(f.exec->candidate_count_for_testing() == 1);

  f.popped(101, short_queue);
  f.settle();
  CHECK(f.exec->counters().promoted_to_gpu == 2);
  CHECK(tier_of(*a) == Tier::GPU);
  CHECK(f.outstanding(f.gpu()) == need);
}

TEST_CASE("preload executor: a busy or failed input is retried only once downgraded again",
          "[preload][preload_executor]")
{
  std::atomic<bool> fail_next{false};
  executor_fixture f;
  f.build(executor_fixture::config(), 2, [&fail_next](promotion_job const&) {
    if (fail_next.exchange(false)) { throw std::runtime_error("injected copy failure"); }
  });
  auto const a = f.spilled(*f.host);
  f.downgraded(1, {a});

  SECTION("its batch lock is held")
  {
    auto const held = a->to_read_only();
    f.popped(100, short_queue);
    f.settle();
    CHECK(f.exec->counters().not_needed == 1);
  }
  SECTION("its copy fails")
  {
    fail_next = true;
    f.popped(100, short_queue);
    f.settle();
    CHECK(f.exec->counters().failed == 1);
  }
  CHECK(tier_of(*a) == Tier::HOST);
  CHECK(f.outstanding(f.gpu()) == 0);

  // Skipped until downgraded again, so a pop finds nothing to do.
  f.popped(101, short_queue);
  f.settle();
  CHECK(f.exec->counters().no_room == 1);
  CHECK(tier_of(*a) == Tier::HOST);

  f.downgraded(1, {a});
  f.popped(102, short_queue);
  f.settle();
  CHECK(f.exec->counters().promoted_to_gpu == 1);
  CHECK(tier_of(*a) == Tier::GPU);
}

TEST_CASE("preload executor: an input found already on the GPU is not tried again",
          "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());
  // Reported on HOST, but back on the GPU before preload gets to it.
  auto const back         = f.make_gpu_batch();
  auto const spilled      = f.spilled(*f.host);
  auto first              = f.task(1, {back});
  first.inputs.front().to = f.host->get_id();
  f.downgraded(std::move(first));
  f.downgraded(2, {spilled});

  f.popped(100, short_queue);
  f.settle();
  auto c = f.exec->counters();
  CHECK(c.not_needed == 1);
  CHECK(c.promoted_to_gpu == 0);
  CHECK(tier_of(*spilled) == Tier::HOST);
  CHECK(f.outstanding(f.gpu()) == 0);

  // Now known to be on the GPU, it is skipped: the next credit goes to the other task.
  f.popped(101, short_queue);
  f.settle();
  c = f.exec->counters();
  CHECK(c.not_needed == 1);
  CHECK(c.promoted_to_gpu == 1);
  CHECK(tier_of(*spilled) == Tier::GPU);
}

TEST_CASE("preload executor: an input with no room for its exact need waits until that fits",
          "[preload][preload_executor]")
{
  // Half the GPU is reservable, so reservations fail while nothing counts as consumed yet: only
  // the reservation and the budget refuse here, never the ceiling.
  executor_fixture f{fixture_options{.gpu_reservation = 0.5}};
  auto const big   = f.spilled(*f.host, std::size_t{1} << 20);
  auto const small = f.spilled(*f.host);
  // Big is reported as 1 byte, so it is first selected with the 1 MiB minimum estimate.
  auto const first_estimate = with_slack(1);
  auto const big_need       = gpu_need(*big);
  auto const raised         = with_slack(big_need);
  auto const small_estimate = estimate(*small);
  auto const small_need     = gpu_need(*small);
  auto const below_big      = std::max({first_estimate, small_estimate, small_need});
  REQUIRE(below_big < big_need);
  // Room in the budget for both first estimates, not for big's raised one.
  std::size_t const budget = (std::max(first_estimate, small_estimate) + raised) / 2;
  auto cfg                 = executor_fixture::config();
  cfg.device_budget        = static_cast<double>(budget) / static_cast<double>(f.opts.gpu_capacity);
  f.build(cfg);
  // Reservable: every estimate and small's need, not big's need.
  auto const blocker = f.leave_reservable((below_big + big_need) / 2);
  REQUIRE(f.gpu().make_reservation_or_null(big_need) == nullptr);

  auto big_task                     = f.task(1, {big});
  big_task.inputs.front().gpu_bytes = 1;
  f.downgraded(std::move(big_task));
  f.downgraded(2, {small});

  // Big pops first: its estimate reserves, its exact need does not.
  f.popped(100, short_queue);
  f.settle();
  auto c = f.exec->counters();
  CHECK(c.no_room == 1);
  CHECK(c.promoted_to_gpu == 0);
  CHECK(tier_of(*big) == Tier::HOST);
  CHECK(f.outstanding(f.gpu()) == 0);

  // Big's estimate is now its exact need, which the budget refuses: the credit goes to small
  // instead of to another doomed attempt on big.
  f.popped(101, short_queue);
  f.settle();
  c = f.exec->counters();
  CHECK(c.no_room == 1);
  CHECK(c.promoted_to_gpu == 1);
  CHECK(tier_of(*small) == Tier::GPU);
  CHECK(tier_of(*big) == Tier::HOST);
}

TEST_CASE("preload executor: a task popped or retired during its promotion keeps no budget",
          "[preload][preload_executor]")
{
  promotion_gate gate;
  executor_fixture f;
  f.build(executor_fixture::config(), 2, [&gate](promotion_job const&) { gate.enter(); });
  auto const release = f.arm_gate(gate);
  auto const a       = f.spilled(*f.host);
  auto const charged = estimate(*a);
  f.downgraded(1, {a});
  f.popped(100, short_queue);
  REQUIRE(gate.wait_entered(1));
  REQUIRE(f.handled());
  CHECK(f.outstanding(f.gpu()) == charged);

  SECTION("its task is popped")
  {
    f.popped(1, deep_queue);
    REQUIRE(f.handled());
  }
  SECTION("its query retires") { f.exec->retire_query(f.query); }
  CHECK(f.exec->candidate_count_for_testing() == 0);
  // The running job, not the gone candidate, holds the charge.
  CHECK(f.outstanding(f.gpu()) == charged);

  gate.release();
  f.settle();
  // The copy completes, its charge is released, and nothing is recorded as held.
  CHECK(f.exec->counters().promoted_to_gpu == 1);
  CHECK(tier_of(*a) == Tier::GPU);
  CHECK(f.outstanding(f.gpu()) == 0);
  CHECK(f.exec->candidate_count_for_testing() == 0);
}

TEST_CASE("preload executor: a task_downgraded delivered after its task's pop is dropped",
          "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());
  auto const a = f.spilled(*f.host);
  // The mailbox delivers one producer's events in order, so the reordering a downgrade worker
  // and the scheduler can produce is staged by calling the hooks with chosen event ids. Safe from
  // the test thread: this test publishes nothing, so the subscriber's worker never runs a hook
  // concurrently, and the hooks take the executor's mutex for everything they touch anyway.
  auto& hooks     = static_cast<sirius::event::query_event_subscriber&>(*f.exec);
  auto const task = f.task(7, {a});

  hooks.on_task_popped(1000, {}, f.query, 7, other_pipeline, 0, deep_queue);
  REQUIRE(f.exec->wait_idle_for_testing(30s));
  hooks.on_task_downgraded(999, {}, f.query, task);
  CHECK(f.exec->counters().stale_event == 1);
  CHECK(f.exec->candidate_count_for_testing() == 0);

  // Published after the pop: the task was queued and spilled again.
  hooks.on_task_downgraded(1001, {}, f.query, task);
  CHECK(f.exec->counters().stale_event == 1);
  CHECK(f.exec->candidate_count_for_testing() == 1);
}

TEST_CASE("preload executor: retire_query drops the query's candidates and their budget",
          "[preload][preload_executor]")
{
  executor_fixture f;
  auto const other = sirius::make_query_id(2);
  f.lifecycle.open_query(other);
  f.build(executor_fixture::config());
  auto const a = f.spilled(*f.host);
  auto const b = f.spilled(*f.host);
  auto const c = f.spilled(*f.host);
  f.downgraded(1, {a});
  f.downgraded(2, {b});
  f.downgraded(3, {c}, candidate_pipeline, 0, other);
  f.popped(100, short_queue);
  f.settle();
  REQUIRE(tier_of(*a) == Tier::GPU);
  REQUIRE(f.outstanding(f.gpu()) > 0);

  f.exec->retire_query(f.query);
  CHECK(f.exec->candidate_count_for_testing() == 1);
  CHECK(f.outstanding(f.gpu()) == 0);

  f.popped(101, short_queue);
  f.settle();
  CHECK(tier_of(*b) == Tier::HOST);
  CHECK(tier_of(*c) == Tier::GPU);
}

TEST_CASE("preload executor: a query that grants no work lease adds no candidate",
          "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());
  auto const a = f.spilled(*f.host);
  f.downgraded(1, {a}, candidate_pipeline, 0, sirius::make_query_id(99));  // never opened
  f.popped(100, short_queue);
  f.settle();
  CHECK(f.exec->candidate_count_for_testing() == 0);
  CHECK(f.exec->counters().no_candidate == 1);
  CHECK(tier_of(*a) == Tier::HOST);
}

// ---------------------------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------------------------

TEST_CASE("preload executor: construction validates its dependencies",
          "[preload][preload_executor]")
{
  executor_fixture f;
  auto missing_capacity           = f.deps(2, true);
  missing_capacity.space_capacity = {};
  CHECK_THROWS_AS(preload_executor(*f.publisher, executor_fixture::config(), missing_capacity),
                  std::invalid_argument);
  CHECK_THROWS_AS(preload_executor(*f.publisher, executor_fixture::config(), f.deps(2, false)),
                  std::invalid_argument);

  // Staging alone needs no streams.
  preload_executor_config staging;
  staging.host_budget = 0.1;
  CHECK_NOTHROW(preload_executor(*f.publisher, staging, f.deps(2, false)));
}

TEST_CASE("preload executor: stop is idempotent and final", "[preload][preload_executor]")
{
  executor_fixture f;
  f.build(executor_fixture::config());
  f.exec->stop();
  f.exec->stop();
  f.exec->start();  // a no-op once stopped
  CHECK_FALSE(f.exec->is_subscribed());

  // A live subscriber receives both events, so they were published; publishing enqueues into
  // every registered mailbox at once, and the stopped executor's was dropped.
  event_probe probe{*f.publisher};
  auto const a = f.spilled(*f.host);
  f.downgraded(1, {a});
  f.popped(100, short_queue);
  REQUIRE(probe.wait_for(2));
  CHECK(f.exec->events_handled_for_testing() == 0);
  CHECK(f.exec->counters().pops == 0);
  CHECK(f.exec->candidate_count_for_testing() == 0);
  CHECK(f.exec->wait_idle_for_testing(1s));
  CHECK(tier_of(*a) == Tier::HOST);
}

TEST_CASE("preload executor: destruction joins a running promotion once it ends",
          "[preload][preload_executor]")
{
  promotion_gate gate;
  executor_fixture f;
  f.build(executor_fixture::config(), 2, [&gate](promotion_job const&) { gate.enter(); });
  auto const release_on_exit = f.arm_gate(gate);
  auto const a               = f.spilled(*f.host);
  f.downgraded(1, {a});
  f.popped(100, short_queue);
  REQUIRE(gate.wait_entered(1));

  auto destroying = std::async(std::launch::async, [&f] { f.exec.reset(); });
  // Opens the gate before the future's destructor waits for the destruction.
  absl::Cleanup const open_gate = [&gate] { gate.release(); };
  CHECK(destroying.wait_for(100ms) == std::future_status::timeout);
  gate.release();
  REQUIRE(destroying.wait_for(30s) == std::future_status::ready);
  destroying.get();
  // The copy it was blocked in completed.
  CHECK(tier_of(*a) == Tier::GPU);
}
