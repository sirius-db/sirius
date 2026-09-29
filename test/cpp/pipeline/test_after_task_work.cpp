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
#include "creator/task_creator.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_operator.hpp"
#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "parallel/after_task_work.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>

#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace acc      = sirius::test::accumulation;
using after_work   = sirius::parallel::after_task_work;
using operator_ptr = std::unique_ptr<sirius::op::operator_data>;

/**
 * @brief A pass-through operator whose task-input hook and execute are injectable.
 */
class hook_operator : public sirius::op::sirius_physical_operator {
 public:
  hook_operator()
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::FILTER,
                               sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
                               0)
  {
  }

  std::string get_name() const override { return "hook_operator"; }

  after_work observe_task_input(sirius::op::operator_data const& input,
                                ::cuda::stream_ref stream) override
  {
    return on_observe ? on_observe(input, stream) : after_work{};
  }

  operator_ptr execute(sirius::op::operator_data const& input, ::cuda::stream_ref stream) override
  {
    if (on_execute) { on_execute(stream); }
    auto const& data = dynamic_cast<sirius::op::pipelineable_operator_data const&>(input);
    return std::make_unique<sirius::op::pipelineable_operator_data>(data.get_data_batches());
  }

  void sink(sirius::op::operator_data const&, ::cuda::stream_ref) override {}
  bool is_sink() const override { return true; }

  std::function<after_work(sirius::op::operator_data const&, ::cuda::stream_ref)> on_observe;
  std::function<void(::cuda::stream_ref)> on_execute;
};

/**
 * @brief What a recording job observed when it ran, or that it was destroyed uninvoked.
 */
struct job_record {
  std::mutex mutex;
  std::condition_variable changed;
  bool hold               = false;  // The job blocks until released
  bool started            = false;
  bool finished           = false;
  int invocations         = 0;
  int destroyed_uninvoked = 0;
  std::thread::id thread;
  cudaStream_t stream{};
  bool tracked                = true;
  std::size_t reserved_bytes  = ~std::size_t{0};
  std::size_t tasks_created   = 0;
  std::size_t tasks_completed = 0;
  bool throws                 = false;
  /// When set, the job copies it into `consumer_schedules_at_start` as its first step.
  std::atomic<std::size_t> const* consumer_schedules = nullptr;
  std::size_t consumer_schedules_at_start            = 0;

  void release()
  {
    std::scoped_lock lock(mutex);
    hold = false;
    changed.notify_all();
  }

  void wait_started()
  {
    std::unique_lock lock(mutex);
    changed.wait(lock, [&] { return started; });
  }
};

struct recording_job {
  std::shared_ptr<job_record> record;
  sirius::pipeline::sirius_pipeline const* pipeline;
  cucascade::memory::memory_space* space;

  recording_job(std::shared_ptr<job_record> log,
                sirius::pipeline::sirius_pipeline const* owner,
                cucascade::memory::memory_space* gpu) noexcept
    : record{std::move(log)}, pipeline{owner}, space{gpu}
  {
  }
  recording_job(recording_job&&) noexcept = default;
  ~recording_job()
  {
    if (record) {
      std::scoped_lock lock(record->mutex);
      ++record->destroyed_uninvoked;
    }
  }

  void operator()(::cuda::stream_ref stream) &&
  {
    auto const log = std::move(record);
    std::unique_lock lock(log->mutex);
    if (log->consumer_schedules != nullptr) {
      log->consumer_schedules_at_start = log->consumer_schedules->load();
    }
    ++log->invocations;
    log->thread = std::this_thread::get_id();
    log->stream = stream.get();
    log->tracked =
      space->get_memory_resource_of<cucascade::memory::Tier::GPU>()->is_stream_tracked(stream);
    log->reserved_bytes  = space->get_total_reserved_memory();
    log->tasks_created   = pipeline->get_tasks_created();
    log->tasks_completed = pipeline->get_tasks_completed();
    log->started         = true;
    log->changed.notify_all();
    log->changed.wait(lock, [&] { return !log->hold; });
    log->finished = true;
    log->changed.notify_all();
    if (log->throws) { throw std::runtime_error{"after-task work failed on purpose"}; }
  }
};

/**
 * @brief A task creator that only counts the consumer schedule requests it receives.
 */
class counting_task_creator final : public sirius::creator::task_creator {
 public:
  explicit counting_task_creator(sirius::memory::sirius_memory_reservation_manager& manager)
    : task_creator(sirius::creator::task_creator_config{}, manager)
  {
  }

  void schedule(sirius::op::sirius_physical_operator* request) override
  {
    if (request != nullptr) { scheduled.fetch_add(1); }
  }

  std::atomic<std::size_t> scheduled{0};
};

/**
 * @brief One GPU, a two-worker executor, and a one-operator pipeline whose task input is one GPU
 * batch.
 */
struct harness {
  acc::fixture memory;
  std::unique_ptr<counting_task_creator> creator;  // Outlives the executor
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> requests;
  std::unique_ptr<sirius::pipeline::gpu_pipeline_executor> executor;
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();
  std::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::unique_ptr<hook_operator> source        = std::make_unique<hook_operator>();
  std::unique_ptr<hook_operator> hook          = std::make_unique<hook_operator>();
  std::unique_ptr<hook_operator> consumer      = std::make_unique<hook_operator>();
  std::unique_ptr<hook_operator> consumer_sink = std::make_unique<hook_operator>();
  std::shared_ptr<sirius::pipeline::sirius_pipeline> downstream;
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global;
  std::atomic<std::uint64_t> next_task_id{1};
  std::atomic<int> executed{0};

  explicit harness(bool per_stream = false) : memory(1, per_stream)
  {
    sirius::pipeline::pipeline_build_context const build{nullptr, true};
    pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(build);
    pipeline->set_pipeline_id(7);
    sirius::pipeline::sirius_pipeline_build_state state;
    state.set_pipeline_source(*pipeline, *source);
    state.add_pipeline_operator(*pipeline, *hook);
    state.set_pipeline_sink(*pipeline, *hook, 1);
    std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines{pipeline};
    sirius::pipeline::assign_operator_ids(pipelines);
    global = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    global->set_completion_handler(completion);
  }

  ~harness()
  {
    if (executor) { executor->stop(); }
    requests.close();
  }

  /**
   * @brief Adds a downstream pipeline whose source consumes this pipeline's output, and a counting
   * task creator that the executor sends its consumer schedule requests to.
   */
  void add_consumer()
  {
    sirius::pipeline::pipeline_build_context const build{nullptr, true};
    downstream = std::make_shared<sirius::pipeline::sirius_pipeline>(build);
    downstream->set_pipeline_id(8);
    sirius::pipeline::sirius_pipeline_build_state state;
    state.set_pipeline_source(*downstream, *consumer);
    state.set_pipeline_sink(*downstream, *consumer_sink, 1);
    downstream->add_dependency(pipeline);
    REQUIRE(pipeline->get_output_consumers().size() == 1);
    creator = std::make_unique<counting_task_creator>(*memory.manager);
  }

  void start_executor()
  {
    sirius::exec::thread_pool_config config;
    config.num_threads        = 2;
    config.thread_name_prefix = "after-task-test";
    executor                  = std::make_unique<sirius::pipeline::gpu_pipeline_executor>(
      config,
      &memory.gpu(0),
      requests.make_publisher(),
      nullptr,
      sirius::test::make_test_telemetry_context());
    if (creator) { executor->set_task_creator(creator.get()); }
    executor->start();
  }

  [[nodiscard]] std::unique_ptr<sirius::pipeline::gpu_pipeline_task> make_task(acc::batch_ptr batch)
  {
    std::vector<acc::batch_ptr> batches{std::move(batch)};
    return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      next_task_id++,
      std::vector<cucascade::shared_data_repository*>{},
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
        std::make_unique<sirius::op::pipelineable_operator_data>(std::move(batches))),
      global);
  }

  /// Executes a task on this thread as the executor's success exit does, including the runner.
  void run_inline(acc::batch_ptr batch, std::size_t reservation_bytes = 64 * acc::mib)
  {
    auto task        = make_task(std::move(batch));
    auto info        = task->get_estimated_reservation_size_info(&memory.gpu(0));
    auto reservation = memory.gpu(0).make_reservation_or_null(reservation_bytes);
    REQUIRE(reservation);
    auto* local =
      dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(task->local_state());
    REQUIRE(local != nullptr);
    local->set_reservation(std::move(reservation), info);
    auto const stream = memory.task_stream(0);
    task->execute(stream);
    auto work = task->take_after_task_work();
    task.reset();
    sirius::pipeline::gpu_pipeline_executor::run_after_task_work(std::move(work), stream, nullptr);
  }

  template <class Predicate>
  void wait_until(Predicate&& done)
  {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (!done()) {
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
  }

  /// A hook that returns one recording job on its first call and nothing afterwards.
  void record_once(std::shared_ptr<job_record> const& record,
                   std::thread::id* hook_thread,
                   cudaStream_t* hook_stream)
  {
    auto issued      = std::make_shared<std::atomic<bool>>(false);
    hook->on_observe = [this, record, issued, hook_thread, hook_stream](
                         sirius::op::operator_data const&, ::cuda::stream_ref stream) {
      if (issued->exchange(true)) { return after_work{}; }
      if (hook_thread != nullptr) { *hook_thread = std::this_thread::get_id(); }
      if (hook_stream != nullptr) { *hook_stream = stream.get(); }
      return after_work{recording_job{record, pipeline.get(), &memory.gpu(0)}};
    };
  }
};

/// A hook that contributes the task's one input batch to @p session, as the build PARTITION does.
std::function<after_work(sirius::op::operator_data const&, ::cuda::stream_ref)> contributing_hook(
  sirius::op::dynamic_filter_publication_session& session)
{
  return [&session](sirius::op::operator_data const& input, ::cuda::stream_ref stream) {
    auto const& data = dynamic_cast<sirius::op::pipelineable_operator_data const&>(input);
    auto batches     = data.get_read_only_batches();
    auto job = session.contribute(data.original_batch_ids().front(), batches.front(), stream);
    return job ? after_work{std::move(job)} : after_work{};
  };
}

static_assert(noexcept(sirius::pipeline::gpu_pipeline_executor::run_after_task_work(
  std::declval<after_work>(), std::declval<::cuda::stream_ref>(), nullptr)));

}  // namespace

TEST_CASE("after-task work runs once on the worker after the task's success epilogue",
          "[pipeline][after_task_work]")
{
  bool const per_stream = GENERATE(false, true);
  harness test(per_stream);
  auto record = std::make_shared<job_record>();
  std::thread::id hook_thread;
  cudaStream_t hook_stream{};
  test.record_once(record, &hook_thread, &hook_stream);
  test.start_executor();
  test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
  test.wait_until([&] {
    std::scoped_lock lock(record->mutex);
    return record->finished;
  });
  std::scoped_lock lock(record->mutex);
  REQUIRE(record->invocations == 1);
  REQUIRE(record->destroyed_uninvoked == 0);
  REQUIRE(record->thread == hook_thread);
  REQUIRE(record->stream == hook_stream);
  REQUIRE_FALSE(record->tracked);
  REQUIRE(record->reserved_bytes == 0);
  REQUIRE(record->tasks_completed == 1);
}

TEST_CASE("after-task work starts only after the task's consumers were scheduled",
          "[pipeline][after_task_work]")
{
  harness test;
  test.add_consumer();
  auto record                = std::make_shared<job_record>();
  record->consumer_schedules = &test.creator->scheduled;
  test.record_once(record, nullptr, nullptr);
  test.start_executor();
  test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
  test.wait_until([&] {
    std::scoped_lock lock(record->mutex);
    return record->finished;
  });
  test.executor->wait_and_validate_empty();
  std::scoped_lock lock(record->mutex);
  REQUIRE(record->invocations == 1);
  // Downstream never waits on the work: its consumer was scheduled, and the task destroyed, first.
  REQUIRE(record->consumer_schedules_at_start == 1);
  REQUIRE(record->tasks_completed == 1);
  REQUIRE(test.creator->scheduled.load() == 1);
}

TEST_CASE("after-task work runs after the retry of a reschedule was scheduled",
          "[pipeline][after_task_work]")
{
  harness test;
  auto record = std::make_shared<job_record>();
  test.record_once(record, nullptr, nullptr);
  auto oom_once         = std::make_shared<std::atomic<bool>>(true);
  test.hook->on_execute = [&test, oom_once](::cuda::stream_ref) {
    if (oom_once->exchange(false)) { throw rmm::out_of_memory{"forced out of memory"}; }
    ++test.executed;
  };
  test.start_executor();
  test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
  test.wait_until([&] {
    std::scoped_lock lock(record->mutex);
    return record->finished && test.executed.load() == 1;
  });
  std::scoped_lock lock(record->mutex);
  REQUIRE(record->invocations == 1);
  REQUIRE(record->tasks_created == 2);  // The retry exists before the work runs.
  REQUIRE_FALSE(record->tracked);
  REQUIRE_FALSE(test.completion->has_error());
}

TEST_CASE("after-task work is destroyed uninvoked on a fatal exit or a completed query",
          "[pipeline][after_task_work]")
{
  harness test;
  auto record = std::make_shared<job_record>();
  test.record_once(record, nullptr, nullptr);
  SECTION("fatal task exit")
  {
    test.hook->on_execute = [](::cuda::stream_ref) { throw std::runtime_error{"fatal operator"}; };
    test.start_executor();
    test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
    test.wait_until([&] {
      std::scoped_lock lock(record->mutex);
      return record->destroyed_uninvoked == 1;
    });
    REQUIRE(test.completion->has_error());
  }
  SECTION("query already completed")
  {
    test.hook->on_execute = [&test](::cuda::stream_ref) { ++test.executed; };
    test.completion->mark_completed();
    test.start_executor();
    test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
    test.wait_until([&] {
      std::scoped_lock lock(record->mutex);
      return record->destroyed_uninvoked == 1;
    });
    REQUIRE(test.executed.load() == 1);
    REQUIRE(test.pipeline->get_tasks_completed() == 1);
  }
  std::scoped_lock lock(record->mutex);
  REQUIRE(record->invocations == 0);
}

TEST_CASE("a throwing after-task work is logged and the retry still runs",
          "[pipeline][after_task_work]")
{
  harness test;
  auto record    = std::make_shared<job_record>();
  record->throws = true;
  test.record_once(record, nullptr, nullptr);
  auto oom_once         = std::make_shared<std::atomic<bool>>(true);
  test.hook->on_execute = [&test, oom_once](::cuda::stream_ref) {
    if (oom_once->exchange(false)) { throw rmm::out_of_memory{"forced out of memory"}; }
    ++test.executed;
  };
  test.start_executor();
  test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
  test.wait_until([&] { return test.executed.load() == 1; });
  test.executor->wait_and_validate_empty();
  std::scoped_lock lock(record->mutex);
  REQUIRE(record->invocations == 1);
  REQUIRE_FALSE(test.completion->has_error());
}

TEST_CASE("query completion while after-task work runs waits for it without validation errors",
          "[pipeline][after_task_work]")
{
  harness test;
  auto record  = std::make_shared<job_record>();
  record->hold = true;
  test.record_once(record, nullptr, nullptr);
  test.hook->on_execute = [&test](::cuda::stream_ref) { ++test.executed; };
  test.start_executor();
  test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
  record->wait_started();

  SECTION("success requested meanwhile")
  {
    test.completion->mark_completed();
    auto joined = std::async(std::launch::async, [&] { test.executor->wait_and_validate_empty(); });
    REQUIRE(joined.wait_for(std::chrono::milliseconds{100}) == std::future_status::timeout);
    record->release();
    REQUIRE_NOTHROW(joined.get());
  }
  SECTION("an error reported meanwhile")
  {
    test.completion->report_error("the query failed elsewhere");
    auto drained = std::async(std::launch::async, [&] { test.executor->drain_and_wait(); });
    REQUIRE(drained.wait_for(std::chrono::milliseconds{100}) == std::future_status::timeout);
    record->release();
    drained.get();
    // The executor serves the next query.
    auto next_global = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
      test.pipeline, sirius::test::make_test_telemetry_context());
    test.global = next_global;
    test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
    test.wait_until([&] { return test.executed.load() == 2; });
  }
  std::scoped_lock lock(record->mutex);
  REQUIRE(record->finished);
}

TEST_CASE(
  "a query error during a real accumulation drains, releases its storage, and frees the worker",
  "[pipeline][dynamic_filter][after_task_work]")
{
  // The gated task is either the first contributor (the attempt is collecting when the error
  // arrives) or the final one (its publishing job is pending in the task).
  bool const gate_final = GENERATE(false, true);
  acc::load_accumulation_kernels(1);
  harness test;
  auto const first    = test.memory.make_batch(0, 0, 2000);
  auto const second   = test.memory.make_batch(0, 5000, 2000);
  auto const baseline = test.memory.allocated_bytes()[0];
  REQUIRE(test.memory.begin({first, second}));

  // The gated task's inserts, and so its post-operator sync, wait until the test opens the gate.
  auto gate             = std::make_shared<std::unique_ptr<acc::stream_gate>>();
  auto gated_observed   = std::make_shared<std::atomic<bool>>(false);
  auto contribute       = contributing_hook(*test.memory.session);
  auto const gated_id   = (gate_final ? second : first)->get_batch_id();
  test.hook->on_observe = [contribute, gate, gated_observed, gated_id](
                            sirius::op::operator_data const& input, ::cuda::stream_ref stream) {
    auto const& data = dynamic_cast<sirius::op::pipelineable_operator_data const&>(input);
    bool const gated = data.original_batch_ids().front() == gated_id;
    if (gated) { *gate = std::make_unique<acc::stream_gate>(stream); }
    auto work = contribute(input, stream);
    if (gated) { gated_observed->store(true); }
    return work;
  };
  test.hook->on_execute = [&test](::cuda::stream_ref) { ++test.executed; };
  test.start_executor();
  if (gate_final) {
    test.executor->schedule(test.make_task(first));
    test.wait_until([&] { return test.executed.load() == 1; });
  }
  test.executor->schedule(test.make_task(gate_final ? second : first));
  test.wait_until([&] { return gated_observed->load(); });

  // The engine's error sequence: report, cancel the query's publications, then drain.
  test.completion->report_error("the query failed elsewhere");
  test.memory.session->cancel();
  auto drained = std::async(std::launch::async, [&] { test.executor->drain_and_wait(); });
  REQUIRE(drained.wait_for(std::chrono::milliseconds{100}) == std::future_status::timeout);
  (*gate)->open();
  drained.get();

  auto const counters = test.memory.stats.snapshot();
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE(counters.filters_pushed == 0);
  REQUIRE(test.memory.channel->snapshot().terminal());
  test.memory.session.reset();
  REQUIRE(test.memory.allocated_bytes()[0] == baseline);
  REQUIRE_FALSE(test.memory.allocator(0).is_stream_tracked(test.memory.task_stream(0)));

  // drain_and_wait returned, so no worker slot is held; the executor still serves new tasks.
  test.hook->on_observe = nullptr;
  test.global           = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    test.pipeline, sirius::test::make_test_telemetry_context());
  auto const executed = test.executed.load();
  test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
  test.executor->schedule(test.make_task(test.memory.make_batch(0, 0, 100)));
  test.wait_until([&] { return test.executed.load() == executed + 2; });
}

TEST_CASE("accumulated publication at a reschedule exit publishes once beside the retry",
          "[pipeline][dynamic_filter][after_task_work]")
{
  harness test;
  auto const first  = test.memory.make_batch(0, 0, 2000);
  auto const second = test.memory.make_batch(0, 5000, 2000);
  REQUIRE(test.memory.begin({first, second}));
  test.hook->on_observe = contributing_hook(*test.memory.session);
  auto oom_final        = std::make_shared<std::atomic<int>>(0);
  test.hook->on_execute = [&test, oom_final](::cuda::stream_ref) {
    // The second task (the final contributor) runs out of memory once, after contributing.
    if (oom_final->fetch_add(1) == 1) { throw rmm::out_of_memory{"forced out of memory"}; }
    ++test.executed;
  };
  test.start_executor();
  test.executor->schedule(test.make_task(first));
  test.wait_until([&] { return test.executed.load() == 1; });
  test.executor->schedule(test.make_task(second));
  test.wait_until([&] {
    return test.executed.load() == 2 &&
           test.memory.stats.snapshot().accumulation_publications_finished == 1;
  });
  test.executor->wait_and_validate_empty();
  auto const counters = test.memory.stats.snapshot();
  REQUIRE(counters.accumulation_duplicate_contributions == 1);
  REQUIRE(counters.accumulation_completed_contributions == 2);
  REQUIRE(counters.filters_pushed == 2);
  REQUIRE(counters.accumulations_skipped_error == 0);
  REQUIRE_FALSE(test.completion->has_error());
  test.memory.require_members({first, second});
}

TEST_CASE("accumulation leaves partition task memory accounting unchanged",
          "[dynamic_filter][memory]")
{
  // Runs the same two tasks through a pipeline with and without accumulation and compares what each
  // task's reservation observed.
  auto const run = [](bool accumulate, int variant) {
    harness test;
    auto const first    = test.memory.make_batch(0, 0, 50'000);
    auto const second   = test.memory.make_batch(0, 100'000, 50'000);
    auto const baseline = test.memory.allocated_bytes()[0];
    if (accumulate) {
      REQUIRE(test.memory.begin({first, second}));
      if (variant == 2) {
        // A decline in the task: the retired builder is released only by the settle job.
        test.hook->on_observe = [&test](sirius::op::operator_data const&, ::cuda::stream_ref) {
          auto job = test.memory.session->decline_accumulation(
            sirius::op::accumulation_decline::CONTRIBUTION_UNACCOUNTABLE);
          return job ? after_work{std::move(job)} : after_work{};
        };
      } else if (variant == 3) {
        // A validation failure in the task: an ID the inventory does not list.
        test.hook->on_observe = [&test](sirius::op::operator_data const& input,
                                        ::cuda::stream_ref stream) {
          auto const& data = dynamic_cast<sirius::op::pipelineable_operator_data const&>(input);
          auto batches     = data.get_read_only_batches();
          auto job = test.memory.session->contribute(~std::uint64_t{0}, batches.front(), stream);
          return job ? after_work{std::move(job)} : after_work{};
        };
      } else {
        test.hook->on_observe = contributing_hook(*test.memory.session);
      }
    }
    test.hook->on_execute = [&test](::cuda::stream_ref stream) {
      auto& allocator = test.memory.allocator(0);
      void* scratch   = allocator.allocate(stream, 4 * acc::mib, 256);
      stream.sync();
      allocator.deallocate(stream, scratch, 4 * acc::mib, 256);
    };
    test.run_inline(first);
    if (variant != 0) { test.run_inline(second); }
    auto const input_bytes = sirius::get_cudf_table_view(*first).num_rows() * 12;
    auto estimate          = test.global->get_memory_history().estimate_peak_memory(input_bytes);
    REQUIRE(estimate);
    if (accumulate && variant >= 2) {
      // The released partials leave the space at its pre-accumulation total.
      REQUIRE(test.memory.allocated_bytes()[0] == baseline);
    }
    REQUIRE_FALSE(test.memory.allocator(0).is_stream_tracked(test.memory.task_stream(0)));
    return std::pair{*estimate, test.global->get_memory_history().size()};
  };
  // 0: a non-final contribution; 1: the final one; 2: a decline in the task; 3: a validation
  // failure in the task.
  auto const variant = GENERATE(0, 1, 2, 3);
  REQUIRE(run(true, variant) == run(false, variant));
}

TEST_CASE("an out-of-memory retry floor ignores accumulation", "[dynamic_filter][memory]")
{
  auto const floor_after_oom = [](bool accumulate) {
    harness test;
    auto const only = test.memory.make_batch(0, 0, 50'000);
    if (accumulate) {
      REQUIRE(test.memory.begin({only}));
      test.hook->on_observe = contributing_hook(*test.memory.session);
    }
    test.hook->on_execute = [](::cuda::stream_ref) { throw rmm::out_of_memory{"forced"}; };
    auto task             = test.make_task(only);
    auto info             = task->get_estimated_reservation_size_info(&test.memory.gpu(0));
    auto reservation      = test.memory.gpu(0).make_reservation_or_null(64 * acc::mib);
    REQUIRE(reservation);
    auto* local =
      dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(task->local_state());
    REQUIRE(local != nullptr);
    local->set_reservation(std::move(reservation), info);
    REQUIRE_THROWS_AS(task->execute(test.memory.task_stream(0)),
                      sirius::pipeline::oom_reschedule_exception);
    auto const floor = local->get_retry_reservation_floor();
    auto work        = task->take_after_task_work();
    task.reset();
    sirius::pipeline::gpu_pipeline_executor::run_after_task_work(
      std::move(work), test.memory.task_stream(0), nullptr);
    if (accumulate) {
      REQUIRE(test.memory.stats.snapshot().accumulation_publications_finished == 1);
    }
    return floor;
  };
  REQUIRE(floor_after_oom(true) == floor_after_oom(false));
}

TEST_CASE("published accumulated filters stay globally accounted until the channel releases them",
          "[dynamic_filter][memory]")
{
  bool const per_stream = GENERATE(false, true);
  harness test(per_stream);
  auto const first    = test.memory.make_batch(0, 0, 20'000);
  auto const second   = test.memory.make_batch(0, 50'000, 20'000);
  auto const baseline = test.memory.allocated_bytes()[0];
  REQUIRE(test.memory.begin({first, second}));
  auto const partials = test.memory.allocated_bytes()[0];
  REQUIRE(partials > baseline);
  REQUIRE(test.memory.gpu(0).get_total_reserved_memory() == 0);
  test.hook->on_observe = contributing_hook(*test.memory.session);
  test.run_inline(first);
  test.run_inline(second);
  // The published replicas are the partial arrays; the scratch-free publication left no tracker
  // behind.
  REQUIRE(test.memory.stats.snapshot().accumulation_publications_finished == 1);
  REQUIRE(test.memory.allocated_bytes()[0] == partials);
  REQUIRE(test.memory.gpu(0).get_total_reserved_memory() == 0);
  REQUIRE_FALSE(test.memory.allocator(0).is_stream_tracked(test.memory.task_stream(0)));
  test.memory.session.reset();
  test.memory.channel.reset();
  REQUIRE(test.memory.allocated_bytes()[0] == baseline);
}
