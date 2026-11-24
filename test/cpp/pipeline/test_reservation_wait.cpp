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

// A task whose memory reservation cannot be granted at once waits for it on its worker thread,
// so the executor's manager loop keeps dispatching the tasks that fit; and a task that runs out
// of memory while its reservation covers the whole memory space fails instead of retrying.

#include "catch.hpp"
#include "creator/task_creator.hpp"
#include "exec/config.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/query_retirement_test_utils.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_scheduler.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/error.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

constexpr std::size_t mib          = 1024ULL * 1024;
constexpr std::size_t gpu_capacity = 1000 * mib;
constexpr std::size_t holder_bytes = 600 * mib;
constexpr std::size_t big_bytes    = 700 * mib;
constexpr std::size_t small_bytes  = 20 * mib;

/// Restores the process-wide reservation policy when the test ends.
struct policy_guard {
  sirius::exec::reservation_wait_policy saved =
    sirius::pipeline::gpu_pipeline_executor::reservation_policy();
  explicit policy_guard(sirius::exec::reservation_wait_policy policy)
  {
    sirius::pipeline::gpu_pipeline_executor::set_reservation_policy(policy);
  }
  ~policy_guard() { sirius::pipeline::gpu_pipeline_executor::set_reservation_policy(saved); }
};

class gate {
 public:
  void wait()
  {
    std::unique_lock lock(mutex_);
    cv_.wait_for(lock, 30s, [&] { return open_; });
  }
  void open()
  {
    std::lock_guard lock(mutex_);
    open_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool open_{false};
};

struct shared_counters : sirius::pipeline::sirius_pipeline_task_global_state {
  shared_counters()
    : sirius_pipeline_task_global_state(nullptr, sirius::test::make_test_telemetry_context())
  {
  }
  std::atomic<int> started{0};
  std::atomic<int> completed{0};
  std::atomic<int> small_completed{0};
  std::atomic<int> big_completed{0};
  std::atomic<int> ooms{0};
};

/// A task that asks for `reservation_bytes` and then runs `body`.
class sized_task : public sirius::pipeline::gpu_pipeline_task {
 public:
  using body_fn = std::function<void(sized_task&, ::cuda::stream_ref)>;

  sized_task(uint64_t task_id,
             std::size_t reservation_bytes,
             body_fn body,
             std::shared_ptr<shared_counters> counters,
             std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state> local = nullptr)
    : gpu_pipeline_task(task_id,
                        std::vector<cucascade::shared_data_repository*>{},
                        local ? std::move(local)
                              : std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
                                  std::make_unique<sirius::op::pipelineable_operator_data>(
                                    std::vector<std::shared_ptr<cucascade::data_batch>>{})),
                        counters),
      _reservation_bytes(reservation_bytes),
      _body(std::move(body)),
      _counters(std::move(counters))
  {
  }

  std::unique_ptr<sirius::op::operator_data> compute_task(::cuda::stream_ref) override
  {
    return nullptr;
  }
  void publish_output(sirius::op::operator_data&, ::cuda::stream_ref) override {}
  std::vector<sirius::op::sirius_physical_operator*> get_output_consumers() override { return {}; }

  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space*) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.input_basis      = _reservation_bytes;
    info.reservation_size = _reservation_bytes;
    return info;
  }

  void execute(::cuda::stream_ref stream) override
  {
    _counters->started.fetch_add(1);
    _body(*this, stream);
    _counters->completed.fetch_add(1);
  }

  std::unique_ptr<gpu_pipeline_task> create_rescheduled_task(
    uint64_t task_id,
    std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local_state) override
  {
    auto typed = std::unique_ptr<sirius::pipeline::gpu_pipeline_task_local_state>(
      static_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(local_state.release()));
    return std::make_unique<sized_task>(
      task_id, _reservation_bytes, _body, _counters, std::move(typed));
  }

  [[nodiscard]] shared_counters& counters() { return *_counters; }
  [[nodiscard]] sirius::pipeline::gpu_pipeline_task_local_state& local()
  {
    return _local_state->cast<sirius::pipeline::gpu_pipeline_task_local_state>();
  }

 private:
  std::size_t _reservation_bytes;
  body_fn _body;
  std::shared_ptr<shared_counters> _counters;
};

struct fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu = nullptr;
  sirius::exec::query_lifecycle_registry lifecycle;
  std::unique_ptr<sirius::pipeline::task_scheduler> scheduler;
  std::unique_ptr<sirius::creator::task_creator> creator;
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();
  std::shared_ptr<shared_counters> counters = std::make_shared<shared_counters>();

  explicit fixture(int threads)
  {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(gpu_capacity)
      .set_reservation_fraction_per_gpu(1.0)
      .set_per_numa_region_capacity(256 * mib)
      .use_gpu_id_as_host_id()
      .track_reservation_per_stream(false)
      .set_reservation_fraction_per_numa_region(0.75);
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    gpu     = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    sirius::exec::thread_pool_config config;
    config.num_threads        = threads;
    config.thread_name_prefix = "reserve-wait";
    lifecycle.open_query(sirius::make_query_id(0));
    scheduler = std::make_unique<sirius::pipeline::task_scheduler>(
      lifecycle, config, *manager, sirius::test::make_test_telemetry_context());
    creator = std::make_unique<sirius::creator::task_creator>(
      lifecycle, sirius::creator::task_creator_config{}, *manager);
    creator->set_task_scheduler(*scheduler);
    scheduler->set_task_creator(*creator);
    counters->set_completion_handler(completion);
    scheduler->start();
  }

  ~fixture()
  {
    if (scheduler) { scheduler->stop(); }
  }

  void schedule(uint64_t id, std::size_t bytes, sized_task::body_fn body)
  {
    scheduler->schedule(std::make_unique<sized_task>(id, bytes, std::move(body), counters));
  }

  void retire()
  {
    sirius::test::retire_scheduler_query(
      lifecycle, *scheduler, sirius::make_query_id(0), false, creator.get());
  }
};

sirius::exec::reservation_wait_policy on_worker()
{
  sirius::exec::reservation_wait_policy policy;
  policy.wait_on_worker = true;
  return policy;
}

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds limit = 10s)
{
  auto const deadline = std::chrono::steady_clock::now() + limit;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(2ms);
  }
  return true;
}

sized_task::body_fn hold_until(gate& release)
{
  return [&release](sized_task&, ::cuda::stream_ref) { release.wait(); };
}

sized_task::body_fn count_small()
{
  return [](sized_task& task, ::cuda::stream_ref) { task.counters().small_completed.fetch_add(1); };
}

sized_task::body_fn count_big()
{
  return [](sized_task& task, ::cuda::stream_ref) { task.counters().big_completed.fetch_add(1); };
}

}  // namespace

TEST_CASE("A task waiting for its reservation does not stall the GPU executor",
          "[gpu_pipeline_executor][reservation_wait]")
{
  policy_guard policy{on_worker()};
  fixture f(3);
  gate release;

  // The holder keeps 600 of 1000 MiB reserved; the big task (700 MiB) cannot be reserved until
  // the holder finishes; the small ones (20 MiB) fit all along.
  f.schedule(0, holder_bytes, hold_until(release));
  REQUIRE(wait_until([&] { return f.counters->started.load() == 1; }));
  f.schedule(1, big_bytes, count_big());
  for (uint64_t id = 2; id < 6; ++id) {
    f.schedule(id, small_bytes, count_small());
  }

  // The small tasks run while the big one waits and the holder still holds its memory.
  CHECK(wait_until([&] { return f.counters->small_completed.load() == 4; }));
  CHECK(f.counters->big_completed.load() == 0);
  CHECK(f.gpu->get_total_reserved_memory() >= holder_bytes);

  release.open();
  CHECK(wait_until([&] { return f.counters->big_completed.load() == 1; }));
  CHECK(wait_until([&] { return f.counters->completed.load() == 6; }));
  CHECK_FALSE(f.completion->has_error());
  f.retire();
}

TEST_CASE("With wait_on_worker off a waiting reservation holds back later tasks",
          "[gpu_pipeline_executor][reservation_wait]")
{
  sirius::exec::reservation_wait_policy blocking;
  blocking.wait_on_worker = false;
  policy_guard policy{blocking};
  fixture f(3);
  gate release;

  f.schedule(0, holder_bytes, hold_until(release));
  REQUIRE(wait_until([&] { return f.counters->started.load() == 1; }));
  f.schedule(1, big_bytes, count_big());
  // The big task reaches the manager before the small ones are queued.
  std::this_thread::sleep_for(100ms);
  for (uint64_t id = 2; id < 6; ++id) {
    f.schedule(id, small_bytes, count_small());
  }

  // The manager loop waits with the big task, so nothing else is dispatched.
  std::this_thread::sleep_for(300ms);
  CHECK(f.counters->small_completed.load() == 0);
  CHECK(f.counters->big_completed.load() == 0);

  release.open();
  CHECK(wait_until([&] { return f.counters->completed.load() == 6; }));
  CHECK(f.counters->big_completed.load() == 1);
  CHECK(f.counters->small_completed.load() == 4);
  CHECK_FALSE(f.completion->has_error());
  f.retire();
}

TEST_CASE("A waiting task is overtaken by a bounded number of later tasks",
          "[gpu_pipeline_executor][reservation_wait]")
{
  auto bounded             = on_worker();
  bounded.max_bypass_tasks = 2;
  bounded.max_bypass_time  = 60s;
  policy_guard policy{bounded};
  fixture f(3);
  gate release;

  f.schedule(0, holder_bytes, hold_until(release));
  REQUIRE(wait_until([&] { return f.counters->started.load() == 1; }));
  f.schedule(1, big_bytes, count_big());
  // Let the big task start waiting before the small ones arrive, so each of them counts.
  std::this_thread::sleep_for(100ms);
  for (uint64_t id = 2; id < 8; ++id) {
    f.schedule(id, small_bytes, count_small());
  }

  // Two small tasks overtake the big one; then the manager dispatches nothing until it is
  // served.
  CHECK(wait_until([&] { return f.counters->small_completed.load() == 2; }));
  std::this_thread::sleep_for(300ms);
  CHECK(f.counters->small_completed.load() == 2);
  CHECK(f.counters->big_completed.load() == 0);

  release.open();
  CHECK(wait_until([&] { return f.counters->completed.load() == 8; }));
  CHECK(f.counters->big_completed.load() == 1);
  CHECK(f.counters->small_completed.load() == 6);
  CHECK_FALSE(f.completion->has_error());
  f.retire();
}

TEST_CASE("Every task waiting for memory at once still makes progress",
          "[gpu_pipeline_executor][reservation_wait]")
{
  policy_guard policy{on_worker()};
  fixture f(3);
  gate release;

  // Every worker but the holder's waits for memory. Once the holder is done the two big tasks
  // get their reservations one after the other (700 + 700 MiB do not fit together).
  f.schedule(0, holder_bytes, hold_until(release));
  REQUIRE(wait_until([&] { return f.counters->started.load() == 1; }));
  f.schedule(1, big_bytes, count_big());
  f.schedule(2, big_bytes, count_big());
  std::this_thread::sleep_for(100ms);
  CHECK(f.counters->big_completed.load() == 0);
  release.open();
  CHECK(wait_until([&] { return f.counters->big_completed.load() == 2; }));
  CHECK_FALSE(f.completion->has_error());
  f.retire();
}

TEST_CASE("An OOM under a whole-space reservation fails the task instead of retrying",
          "[gpu_pipeline_executor][reservation_wait][oom]")
{
  bool const fail_fast = GENERATE(true, false);
  sirius::exec::reservation_wait_policy configured;
  configured.fail_fast_on_full_space_oom = fail_fast;
  policy_guard policy{configured};
  fixture f(1);

  // Asks for more than the space holds (clamped to all of it) and then allocates beyond it.
  auto always_oom = [](sized_task& task, ::cuda::stream_ref) {
    task.counters().ooms.fetch_add(1);
    throw sirius::pipeline::oom_reschedule_exception(
      std::move(task.local()._input_data), 0, "OOM at operator TEST (index 0)");
  };
  auto result = f.completion->get_awaitable();
  f.schedule(0, 4 * gpu_capacity, always_oom);

  if (fail_fast) {
    REQUIRE(result.wait_for(30s) == std::future_status::ready);
    try {
      result.get();
      FAIL("the query did not fail");
    } catch (std::exception const& error) {
      std::string const message = error.what();
      CHECK(message.find("cannot fit in GPU memory") != std::string::npos);
      CHECK(message.find("OOM at operator TEST") != std::string::npos);
      CHECK(message.find(std::to_string(gpu_capacity)) != std::string::npos);
    }
    // One retry, then the failure.
    CHECK(f.counters->ooms.load() == 2);
  } else {
    // Without the check the task keeps retrying (up to the retry limit).
    CHECK(wait_until([&] { return f.counters->ooms.load() >= 5; }, 30s));
    CHECK_FALSE(f.completion->has_error());
  }
  f.retire();
}
