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
#include "duckdb/common/exception.hpp"
#include "op/scan/iceberg_delete_set.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "scan_manager/native_reader_executor.hpp"
#include "scan_manager/preparation_coordinator.hpp"
#include "scan_manager/preparation_fixture.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
using namespace sirius::scan_manager;
using namespace std::chrono_literals;
using namespace sirius::test::preparation;

namespace {
bool gate_is_available(std::shared_ptr<preparation_gate> const& gate)
{
  bool available = false;
  std::thread probe([&] {
    auto deadline = std::chrono::steady_clock::now() + 100ms;
    do {
      std::unique_lock lock(gate->mutex, std::try_to_lock);
      if (lock.owns_lock()) {
        available = true;
        return;
      }
      std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
  });
  probe.join();
  return available;
}
}  // namespace

TEST_CASE("Non-Iceberg adapter source uses one coordinator claim path and releases the last split",
          "[scan_preparation][coordinator][adapter_contract]")
{
  fixture f(1, 1);
  auto out = std::make_shared<sink>();
  std::atomic<unsigned> claims{0};
  std::atomic<unsigned> reads{0};
  auto source_claim = [&]() -> preparation_coordinator::claim_result {
    auto id = claims.fetch_add(1);
    if (id >= 2) return {};
    return preparation_coordinator::job{[&, id](preparation_stop_token stop) {
      if (stop.stop_requested())
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      ++reads;
      return std::make_unique<tagged>(static_cast<int>(id), 1);
    }};
  };
  f.source(std::make_shared<accumulator>(1), out, std::move(source_claim));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));
  out->consume();
  REQUIRE(out->await(2));
  out->consume();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(claims == 3);  // two real claims plus the single exhausted probe
  CHECK(reads == 2);
  CHECK(out->count() == 2);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Coordinator admission is shared once across adapter units",
          "[scan_preparation][coordinator][adapter_contract]")
{
  fixture f(1, 2);
  auto out  = std::make_shared<sink>();
  auto unit = f.coordinator.admit_unit({101, 1}, {});
  REQUIRE(unit);
  REQUIRE(f.coordinator.admit_unit({101, 1}, {}) == unit);

  auto claim = [unit, count = 0]() mutable -> preparation_coordinator::claim_result {
    if (count == 2) return {};
    auto id = count++;
    return preparation_coordinator::job{[id] { return std::make_unique<tagged>(id, 1); }, unit};
  };
  f.source(std::make_shared<accumulator>(1), out, std::move(claim));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));
  out->consume();
  REQUIRE(out->await(2));
  out->consume();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Default and zero job limits allow submission while the worker is blocked",
          "[scan_preparation][coordinator]")
{
  preparation_config config;
  if (GENERATE(false, true)) config.max_inflight_metadata_jobs = 0;
  fixture f(1, 2, k_interrupt_check_interval, config.resolve(1));
  auto blocked       = f.make_hold();
  auto third         = f.make_hold();
  auto out           = std::make_shared<sink>();
  size_t submissions = 0;
  f.coordinator.before_submission_for_testing([&] {
    if (++submissions == 3) third->wait();
  });
  f.source(std::make_shared<accumulator>(3), out, finite_jobs(3, [=](int id) {
             if (id == 0) blocked->wait();
             return std::make_unique<tagged>(id);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  REQUIRE(third->await());
  CHECK(f.coordinator.snapshot().jobs_peak == 3);
  third->release();
  blocked->release();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(out->count() == 1);
  CHECK_FALSE(f.owner_error);
}

TEST_CASE("Ready partial batches publish while every preparation worker is blocked",
          "[scan_preparation][coordinator]")
{
  fixture f(2, 4, 500ms);
  f.coordinator.set_interrupt_check([] { return false; });
  auto a   = std::make_shared<accumulator>();
  auto out = std::make_shared<sink>();
  auto h1 = f.make_hold(), h2 = f.make_hold();

  f.source(a, out, finite_jobs(3, [=](int id) {
             if (id == 1) h1->wait();
             if (id == 2) h2->wait();
             return std::make_unique<tagged>(id);
           }));
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(h1->await());
  REQUIRE(h2->await());
  REQUIRE(out->await(1));
  CHECK_FALSE(out->closed.load());
  auto stats = f.coordinator.snapshot();
  CHECK(stats.jobs_peak == 2);
  CHECK(stats.partial_emissions == 1);
  CHECK(stats.max_residence >= 20ms);
  CHECK(stats.max_deadline_lateness < 100ms);
  INFO("residence us=" << stats.max_residence.count()
                       << ", deadline late us=" << stats.max_deadline_lateness.count());
}
TEST_CASE("First retained data arms an absolute deadline and pruned results never reset it",
          "[scan_preparation][coordinator]")
{
  accumulator a;
  auto origin = accumulator::clock::time_point{} + 100ms;
  a.fake_now  = origin;
  tagged first(1);
  size_t cursor = 0;
  REQUIRE(a.advance(first, cursor, 1).finished);
  REQUIRE(a.first_retained_time());
  CHECK(*a.first_retained_time() == origin);
  for (int i = 1; i < 10; ++i) {
    a.fake_now = origin + i * 2ms;
    tagged later(i, i % 2);
    cursor = 0;
    a.advance(later, cursor, 1);
    CHECK(*a.first_retained_time() == origin);
  }
  REQUIRE(a.partial_emit());
  CHECK_FALSE(a.first_retained_time());
  a.fake_now = origin + 40ms;
  cursor     = 0;
  a.advance(first, cursor, 1);
  CHECK(*a.first_retained_time() == origin + 40ms);
  a.partial_emit();
  CHECK_FALSE(a.partial_emit());
  CHECK_FALSE(a.first_retained_time());
}
TEST_CASE("Continuous small completion events cannot starve the deadline",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  std::atomic<int> claimed{0};
  f.source(std::make_shared<accumulator>(), out, [&]() -> preparation_coordinator::claim_result {
    int id = claimed++;
    if (id >= 100) return {};
    return preparation_coordinator::job{[id] {
                                          std::this_thread::sleep_for(2ms);
                                          return std::make_unique<tagged>(id, id % 3 ? 1 : 0);
                                        },
                                        {}};
  });
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));
  CHECK(claimed.load() < 100);
  CHECK(f.coordinator.snapshot().partial_emissions > 0);
}
TEST_CASE("A partial emission is not EOS and subsequent input forms another batch",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  auto h   = f.make_hold();

  f.source(std::make_shared<accumulator>(), out, finite_jobs(2, [=](int id) {
             if (id == 1) h->wait();
             return std::make_unique<tagged>(id);
           }));
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(h->await());
  REQUIRE(out->await(1));
  CHECK_FALSE(out->closed.load());
  out->consume();
  h->release();
  REQUIRE(out->await(2));
  REQUIRE(out->await_closed());
  CHECK(out->published == std::vector<int>{0, 1});
  CHECK(out->preparations == 2);
}
TEST_CASE("Publication is refused after cancellation wins during input construction",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out          = std::make_shared<sink>();
  auto construction = f.make_hold();
  auto awaitable    = f.completion.get_awaitable();
  f.source(std::make_shared<accumulator>(1),
           out,
           single_job([] { return std::make_unique<tagged>(7); }),
           construction);
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(construction->await());
  f.coordinator.request_stop(stop_reason::user_cancel);
  REQUIRE(awaitable.wait_for(100ms) == std::future_status::ready);
  CHECK_THROWS(awaitable.get());
  construction->release();
  f.finish();
  CHECK(out->count() == 0);
  CHECK(out->preparations == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Normal EOS does not report worker operation_canceled as a query error",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out      = std::make_shared<sink>();
  auto claiming = f.make_hold();
  std::promise<void> stopped;
  auto notified = stopped.get_future();
  std::promise<void> started;
  auto ready   = started.get_future();
  bool claimed = false;
  f.source(std::make_shared<accumulator>(1), out, [&]() -> preparation_coordinator::claim_result {
    if (std::exchange(claimed, true)) {
      claiming->wait();
      return {};
    }
    preparation_coordinator::job job{
      [&](preparation_stop_token stop) -> std::unique_ptr<sirius::op::scan::scan_info> {
        started.set_value();
        while (!stop.stop_requested())
          std::this_thread::yield();
        stopped.set_value();
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      }};
    return job;
  });
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(ready.wait_for(1s) == std::future_status::ready);
  REQUIRE(claiming->await());
  f.coordinator.request_stop(stop_reason::normal_eos);
  CHECK(notified.wait_for(1s) == std::future_status::ready);
  claiming->release();
  f.finish();
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("Normal EOS still reports a real late worker error", "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  std::promise<void> started;
  auto ready   = started.get_future();
  bool claimed = false;
  f.source(std::make_shared<accumulator>(1), out, [&]() -> preparation_coordinator::claim_result {
    if (std::exchange(claimed, true)) return {};
    preparation_coordinator::job job{
      [&started](preparation_stop_token stop) -> std::unique_ptr<sirius::op::scan::scan_info> {
        started.set_value();
        while (!stop.stop_requested())
          std::this_thread::yield();
        throw std::runtime_error("late worker failure");
      }};
    return job;
  });
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(ready.wait_for(1s) == std::future_status::ready);
  f.coordinator.request_stop(stop_reason::normal_eos);
  f.finish();
  CHECK(f.completion.has_error());
}

TEST_CASE("Unresolved typed input forbids scan input construction until the whole unit is ready",
          "[scan_preparation][coordinator]")
{
  fixture f(2, 4);
  auto out = std::make_shared<sink>();
  required_input_set required;
  required.set(2);
  required.set(3);
  auto unit = f.coordinator.admit_unit({3, 1}, required);
  REQUIRE(unit);
  std::promise<void> returned;
  auto stage = returned.get_future();
  f.source(std::make_shared<accumulator>(1),
           out,
           single_job(
             [&] {
               unit->complete_input(checkpoint_input{7});
               returned.set_value();
               return std::make_unique<tagged>(1);
             },
             unit));
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(stage.wait_for(1s) == std::future_status::ready);
  CHECK(f.coordinator.unit_state_snapshot({3, 1})->state == unit_state::pending);
  CHECK(out->count() == 0);
  REQUIRE(unit->complete_input(
    visibility_input{std::make_shared<sirius::op::scan::iceberg_delete_set const>("file")}));
  REQUIRE(out->await(1));
  CHECK(unit->record().state == unit_state::ready);
  f.coordinator.request_stop(stop_reason::normal_eos);
  CHECK_FALSE(f.coordinator.admit_unit({3, 2}, required));
  CHECK(unit->record().state == unit_state::ready);
}
TEST_CASE(
  "A full output window bounds jobs, results and file fan-out without blocking worker settlement",
  "[scan_preparation][coordinator]")
{
  fixture f(2, 1);
  auto out = std::make_shared<sink>();
  std::atomic<int> next{0};
  f.source(std::make_shared<accumulator>(1), out, [&]() -> preparation_coordinator::claim_result {
    int id = next++;
    if (id >= 1000) return {};
    return preparation_coordinator::job{[id] { return std::make_unique<tagged>(id, 1000); }, {}};
  });
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));
  std::this_thread::sleep_for(25ms);
  CHECK(out->count() == 1);
  CHECK(next.load() >= 1);
  CHECK(next.load() <= 2);
  auto stats = f.coordinator.snapshot();
  CHECK(stats.output_peak <= 1);
  CHECK(stats.results_peak <= 2);
  CHECK(stats.jobs_peak <= 2);
  f.coordinator.request_stop(stop_reason::normal_eos);
  f.finish();
  CHECK(out->count() == 1);
}
TEST_CASE("Claim and partial submission failures release reserved completion slots",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out       = std::make_shared<sink>();
  auto read      = f.make_hold();
  auto awaitable = f.completion.get_awaitable();
  std::atomic<int> submitted{0};

  std::string expected_error;
  SECTION("claim throws")
  {
    expected_error = "claim injected";
    f.source(std::make_shared<accumulator>(), out, []() -> preparation_coordinator::claim_result {
      throw std::runtime_error("claim injected");
    });
  }
  SECTION("second submission throws while a real read is running")
  {
    expected_error = "submit injected";
    f.coordinator.before_submission_for_testing([&] {
      if (++submitted == 2) {
        if (!read->await()) throw std::runtime_error("first read did not enter");
        throw std::runtime_error("submit injected");
      }
    });
    f.source(std::make_shared<accumulator>(), out, finite_jobs(2, [read](int id) {
               read->wait();
               return std::make_unique<tagged>(id);
             }));
  }
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(awaitable.wait_for(1s) == std::future_status::ready);
  CHECK_THROWS_WITH(awaitable.get(), expected_error);
  read->release();
  f.finish();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Non-cancellable reads keep their owner alive until drain finishes",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  auto h   = f.make_hold();
  f.source(std::make_shared<accumulator>(), out, single_job([=] {
             h->wait();
             return std::make_unique<tagged>(1);
           }));
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(h->await());
  f.coordinator.request_stop(stop_reason::normal_eos);
  auto drained = std::async(std::launch::async, [&] { f.finish(); });
  CHECK(drained.wait_for(20ms) == std::future_status::timeout);
  h->release();
  REQUIRE(drained.wait_for(1s) == std::future_status::ready);
  drained.get();
  CHECK(out->count() == 0);
  CHECK(f.coordinator.snapshot().runs == 1);
}
TEST_CASE("GPU completion without a ready event still stops preparation and settles success",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out  = std::make_shared<sink>();
  auto h    = f.make_hold();
  auto done = f.completion.get_awaitable();
  f.source(std::make_shared<accumulator>(), out, single_job([=] {
             h->wait();
             return std::make_unique<tagged>(1);
           }));
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(h->await());
  f.completion.mark_completed();
  CHECK(done.wait_for(20ms) == std::future_status::timeout);
  h->release();
  REQUIRE(done.wait_for(1s) == std::future_status::ready);
  CHECK_NOTHROW(done.get());
  f.finish();
  CHECK(out->count() == 0);
}
TEST_CASE("Production defaults publish a small batch at the deadline while metadata is blocked",
          "[scan_preparation][coordinator]")
{
  auto options = preparation_config{}.resolve(2);
  REQUIRE(options.partial_batch);
  REQUIRE(options.partial_batch->max_residence == 2s);
  fixture f(2, 4, k_interrupt_check_interval, options);
  auto out       = std::make_shared<sink>();
  auto a         = std::make_shared<accumulator>();
  auto h         = f.make_hold();
  auto retained  = std::make_shared<std::promise<void>>();
  auto reached   = retained->get_future();
  auto notified  = std::make_shared<std::atomic<bool>>(false);
  a->on_retained = [retained, notified] {
    if (!notified->exchange(true)) retained->set_value();
  };
  f.source(a, out, finite_jobs(2, [=](int id) {
             if (id == 1) h->wait();
             return std::make_unique<tagged>(id);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(reached.wait_for(1s) == std::future_status::ready);
  REQUIRE(h->await());
  std::this_thread::sleep_for(30ms);
  CHECK(out->count() == 0);
  {
    std::unique_lock lock(out->mutex);
    REQUIRE(out->cv.wait_for(lock, 3s, [&] { return !out->published.empty(); }));
  }
  CHECK_FALSE(out->closed);
  CHECK(f.coordinator.snapshot().partial_emissions == 1);
  CHECK(f.coordinator.snapshot().max_residence >= 2s);
  h->release();
  REQUIRE(out->await_closed());
  CHECK(out->count() == 2);
}
TEST_CASE("Cancellation racing with a deadline never publishes after recognition",
          "[scan_preparation][coordinator]")
{
  auto use_interrupt_signal = GENERATE(false, true);
  auto size_triggered       = GENERATE(false, true);
  auto options              = preparation_config{}.resolve(2);
  options.partial_batch     = partial_batch_options{50, 20ms, size_triggered ? 1s : 20ms};
  for (int iteration = 0; iteration < 12; ++iteration) {
    std::atomic<bool> interrupted{false};
    fixture f(2, 4, k_interrupt_check_interval, options);
    if (use_interrupt_signal) {
      f.coordinator.set_interrupt_check([&] { return interrupted.load(); });
    }
    auto out = std::make_shared<sink>();
    auto h   = f.make_hold();

    f.source(std::make_shared<accumulator>(2), out, finite_jobs(2, [=](int id) {
               if (id == 1) h->wait();
               return std::make_unique<tagged>(id);
             }));
    auto query_guard = f.shutdown_guard();
    auto done        = f.completion.get_awaitable();
    f.start();
    REQUIRE(h->await());
    std::this_thread::sleep_for(std::chrono::milliseconds(15 + iteration));
    if (use_interrupt_signal) {
      interrupted = true;
    } else {
      f.coordinator.request_stop(stop_reason::user_cancel);
    }
    REQUIRE(done.wait_for(1s) == std::future_status::ready);
    CHECK_THROWS_AS(done.get(), duckdb::InterruptException);
    auto count = out->count();
    h->release();
    f.finish();
    CHECK(out->count() == count);
  }
}

TEST_CASE("A unit failed before execution start remains the original terminal failure",
          "[scan_preparation][coordinator]")
{
  fixture f;
  required_input_set required;
  required.set(static_cast<size_t>(required_input::checkpoint_iteration));
  auto unit = f.coordinator.admit_unit({12, 1}, required);
  REQUIRE(unit);
  preparation_failure error;
  error.cause    = sirius::transparent::late_failure_cause::reader_io;
  error.detail   = "pre-start read failure";
  error.original = std::make_exception_ptr(std::runtime_error(error.detail));
  REQUIRE(unit->fail(error));
  auto future      = f.completion.get_awaitable();
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(future.wait_for(1s) == std::future_status::ready);
  CHECK_THROWS_WITH(future.get(), "pre-start read failure");
  f.finish();
  CHECK_FALSE(f.coordinator.unit_state_snapshot({12, 1}));
  CHECK(unit->record().state == unit_state::failed);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}
TEST_CASE("GPU completion cannot hide an accepted preparation failure",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out  = std::make_shared<sink>();
  auto read = f.make_hold();
  f.source(std::make_shared<accumulator>(),
           out,
           single_job([read]() -> std::unique_ptr<sirius::op::scan::scan_info> {
             read->wait();
             throw std::runtime_error("accepted read failed after GPU done");
           }));
  auto future      = f.completion.get_awaitable();
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(read->await());
  f.completion.mark_completed();
  CHECK(future.wait_for(5ms) == std::future_status::timeout);
  read->release();
  REQUIRE(future.wait_for(1s) == std::future_status::ready);
  CHECK_THROWS_WITH(future.get(), "accepted read failed after GPU done");
  f.finish();
  CHECK(out->count() == 0);
}

TEST_CASE("Later scans filling their lookahead cannot block the first scan",
          "[scan_preparation][coordinator]")
{
  fixture f(2, 2);
  auto first = std::make_shared<sink>(), later = std::make_shared<sink>();
  auto blocked = f.make_hold();
  f.source(std::make_shared<accumulator>(1), first, single_job([blocked] {
             blocked->wait();
             return std::make_unique<tagged>(1);
           }));
  f.source(std::make_shared<accumulator>(1), later, single_job([] {
             return std::make_unique<tagged>(2, 100);
           }));
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  REQUIRE(later->await(1));
  // No later pipeline consumption while execution still needs the first scan.
  std::this_thread::sleep_for(20ms);
  CHECK(later->count() == 2);
  blocked->release();
  REQUIRE(first->await(1));
  first->consume();
  CHECK(f.coordinator.snapshot().output_peak <= 3);
}

TEST_CASE("Cancellation or startup failure before run drains without starting jobs",
          "[scan_preparation][coordinator]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  sirius::pipeline::completion_handler completion;
  preparation_coordinator c(completion, dispatcher, preparation_config{}.resolve(1));
  auto done      = completion.get_awaitable();
  bool cancelled = false;
  c.arm();
  SECTION("user cancellation")
  {
    cancelled = true;
    c.request_stop(stop_reason::user_cancel);
  }
  SECTION("consumer startup failure")
  {
    completion.report_error(
      std::make_exception_ptr(std::runtime_error("consumer startup injected")));
  }
  c.drain();
  REQUIRE(done.wait_for(100ms) == std::future_status::ready);
  if (cancelled)
    CHECK_THROWS_AS(done.get(), duckdb::InterruptException);
  else
    CHECK_THROWS_WITH(done.get(), "consumer startup injected");
  CHECK(c.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  CHECK(c.snapshot().runs == 0);
  CHECK(c.snapshot().jobs_peak == 0);
}

TEST_CASE("Expired partial batch parks at a full output window and resumes on consumption",
          "[scan_preparation][coordinator]")
{
  struct retaining_coalescer : accumulator {
    retaining_coalescer() : accumulator(1) {}
    cursor_step advance(sirius::op::scan::scan_info& input, size_t& cursor, size_t quantum) override
    {
      auto step = accumulator::advance(input, cursor, quantum);
      if (step.batch && !step.finished) {
        cap           = 1000;
        auto tail     = accumulator::advance(input, cursor, 1);
        step.finished = tail.finished;
      }
      return step;
    }
  };
  fixture f(2, 1);
  auto out   = std::make_shared<sink>();
  auto later = f.make_hold();

  f.source(std::make_shared<retaining_coalescer>(), out, finite_jobs(2, [=](int id) {
             if (id == 1) later->wait();
             return std::make_unique<tagged>(id, id == 0 ? 2 : 0);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));
  std::this_thread::sleep_for(30ms);
  auto before = f.coordinator.snapshot().wakes;
  std::this_thread::sleep_for(25ms);
  CHECK(f.coordinator.snapshot().wakes <= before + 2);
  CHECK(out->count() == 1);
  out->consume();
  REQUIRE(out->await(2));
  CHECK(f.coordinator.snapshot().partial_emissions == 1);
  CHECK(f.coordinator.snapshot().max_residence >= 50ms);
  CHECK(f.coordinator.snapshot().output_peak <= 1);
  CHECK_FALSE(out->closed);
}
TEST_CASE("Construction and publication failures drain once and preserve the first error",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out            = std::make_shared<sink>();
  auto coalescer      = std::make_shared<accumulator>(1);
  bool fail_construct = false, fail_close = false, normal_close_failure = false;
  bool fail_initialize = false, fail_enqueue = false;
  auto destroyed_outside_gate = std::make_shared<std::atomic<bool>>(false);
  struct observed_input : sirius::op::scan::scan_operator_input {
    using scan_operator_input::scan_operator_input;
    std::function<void()> on_destroy;
    ~observed_input() override
    {
      if (on_destroy) on_destroy();
    }
  };
  size_t publications        = 0;
  std::string expected_error = "push injected";
  SECTION("construction throws")
  {
    fail_construct = true;
    expected_error = "construct injected";
  }
  SECTION("pre-commit failure") {}
  SECTION("closed connector rejects the queue commit")
  {
    fail_enqueue   = true;
    expected_error = "prepared scan published after connector close";
    out->connector->close();
  }
  SECTION("prefetch initialization throws")
  {
    fail_initialize = true;
    expected_error  = "scan input prefetch initialization already attempted";
  }
  SECTION("tail publication throws followed by a close failure")
  {
    fail_close          = true;
    expected_error      = "tail publication failed";
    coalescer->cap      = 1000;
    auto frozen         = accumulator::clock::now();
    coalescer->fake_now = frozen;
    f.coordinator.clock_for_testing([frozen] { return frozen; });
  }
  SECTION("normal close throws")
  {
    fail_close = normal_close_failure = true;
    expected_error                    = "secondary close failure";
  }
  preparation_coordinator::source source;
  source.coalescer = coalescer;
  source.try_claim = single_job([] { return std::make_unique<tagged>(1); });
  source.construct = [&](std::unique_ptr<sirius::op::scan::scan_info> batch) {
    if (fail_construct) throw std::runtime_error("construct injected");
    auto input = std::make_unique<observed_input>(
      std::move(batch), nullptr, 0, std::nullopt, observed_input::prefetch_start::deferred);
    input->on_destroy = [destroyed_outside_gate, gate = f.coordinator.publication_gate()] {
      *destroyed_outside_gate = gate_is_available(gate);
    };
    if (fail_initialize) input->prepare_prefetch();
    return preparation_coordinator::publication{std::move(input), 0};
  };
  f.coordinator.before_publication_for_testing([&] {
    ++publications;
    if (!normal_close_failure && !fail_enqueue) throw std::runtime_error(expected_error);
  });
  source.close = [out, fail_close] {
    std::lock_guard lock(out->mutex);
    ++out->close_calls;
    if (fail_close) throw std::runtime_error("secondary close failure");
    out->closed = true;
    out->cv.notify_all();
  };
  f.register_source(std::move(source), out);
  auto done  = f.completion.get_awaitable();
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(done.wait_for(1s) == std::future_status::ready);
  CHECK_THROWS_WITH(done.get(), expected_error);
  f.finish();
  f.coordinator.drain();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  if (fail_close)
    CHECK_THROWS_WITH(f.completion.throw_if_preparation_cleanup_failed(), expected_error);
  else
    CHECK_NOTHROW(f.completion.throw_if_preparation_cleanup_failed());
  if (normal_close_failure) out->consume();
  CHECK(out->count() == (normal_close_failure ? 1 : 0));
  CHECK(publications == (fail_construct || fail_initialize ? 0 : 1));
  CHECK(destroyed_outside_gate->load() == !fail_construct);
  CHECK(out->close_calls == 1);
}

TEST_CASE("An accepted failure callback drains before GPU completion can publish success",
          "[scan_preparation][coordinator][callback_drain]")
{
  fixture f;
  auto out      = std::make_shared<sink>();
  auto callback = f.make_hold();
  required_input_set required;
  required.set(static_cast<size_t>(required_input::visibility));
  auto unit = f.coordinator.admit_unit({91, 1}, required);
  std::promise<void> metadata;
  auto ready = metadata.get_future();
  f.source(std::make_shared<accumulator>(1),
           out,
           single_job(
             [&] {
               metadata.set_value();
               return std::make_unique<tagged>(1);
             },
             unit));
  auto done    = f.completion.get_awaitable();
  auto cleanup = f.shutdown_guard();
  f.start();
  REQUIRE(ready.wait_for(1s) == std::future_status::ready);
  auto gate = f.coordinator.publication_gate();
  {
    std::lock_guard lock(gate->mutex);
    auto report          = gate->report_failure;
    gate->report_failure = [report, callback](preparation_failure const& failure) {
      callback->wait();
      report(failure);
    };
  }
  auto failure = std::async(std::launch::async, [&] {
    return unit->fail(preparation_failure{
      sirius::transparent::late_failure_cause::reader_io,
      {},
      "accepted callback failure",
      std::make_exception_ptr(std::runtime_error("accepted callback failure"))});
  });
  REQUIRE(callback->await());
  f.completion.mark_completed();
  // Failure may already be delivered by the ready-result path. Success and
  // physical quiescence must still wait for the accepted callback to return.
  auto status = done.wait_for(20ms);
  if (status == std::future_status::ready)
    CHECK_THROWS_WITH(done.get(), "accepted callback failure");
  CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  callback->release();
  REQUIRE(failure.get());
  if (status != std::future_status::ready) {
    REQUIRE(done.wait_for(1s) == std::future_status::ready);
    CHECK_THROWS_WITH(done.get(), "accepted callback failure");
  }
  f.finish();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  CHECK(out->count() == 0);
}

TEST_CASE("Exhausted enumeration cannot close before accepted out-of-order results drain",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out   = std::make_shared<sink>();
  auto first = f.make_hold(), last = f.make_hold();

  f.source(std::make_shared<accumulator>(1), out, finite_jobs(2, [=](int id) {
             (id == 0 ? first : last)->wait();
             return std::make_unique<tagged>(id);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(first->await());
  REQUIRE(last->await());
  last->release();
  REQUIRE(out->await(1));
  CHECK_FALSE(out->closed);
  out->consume();
  first->release();
  REQUIRE(out->await(2));
  REQUIRE(out->await_closed());
  CHECK(out->published == std::vector<int>{1, 0});
}
TEST_CASE("One lookahead batch per scan preserves progress with one worker",
          "[scan_preparation][coordinator]")
{
  fixture f(1, 1);
  auto first = std::make_shared<sink>(), last = std::make_shared<sink>();
  f.source(std::make_shared<accumulator>(1), first, single_job([] {
             return std::make_unique<tagged>(1);
           }));
  f.source(
    std::make_shared<accumulator>(1), last, single_job([] { return std::make_unique<tagged>(2); }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(first->await(1));
  REQUIRE(last->await(1));
  first->consume();
  REQUIRE(first->await_closed());
  REQUIRE(last->await(1));
  last->consume();
  REQUIRE(last->await_closed());
  f.finish();
  CHECK(f.coordinator.snapshot().jobs_peak == 1);
  CHECK(f.coordinator.snapshot().results_peak <= 1);
  CHECK(f.coordinator.snapshot().output_peak <= 2);
}

TEST_CASE("Physical external users delay quiescence after success or cancellation",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  f.source(
    std::make_shared<accumulator>(), out, [] { return preparation_coordinator::claim_result{}; });
  auto use     = f.coordinator.external_use();
  auto done    = f.completion.get_awaitable();
  auto cleanup = std::shared_ptr<fixture>(&f, [&](fixture* ptr) {
    use.reset();
    ptr->stop();
  });
  f.start();
  REQUIRE(out->await_closed());
  SECTION("GPU completion cannot release an outstanding borrower")
  {
    f.completion.mark_completed();
    CHECK(done.wait_for(20ms) == std::future_status::timeout);
  }
  SECTION("Cancellation reports promptly but still drains the borrower")
  {
    f.coordinator.request_stop(stop_reason::user_cancel);
    REQUIRE(done.wait_for(1s) == std::future_status::ready);
    CHECK_THROWS_AS(done.get(), duckdb::InterruptException);
  }
  CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  CHECK_FALSE(f.coordinator.external_use());
  use.reset();
  f.finish();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  f.coordinator.drain();
  CHECK(out->close_calls == 1);
}

TEST_CASE("Cancelling a pending unit synchronizes with and wakes the query owner",
          "[scan_preparation][coordinator]")
{
  fixture f(1, 1);
  auto out = std::make_shared<sink>();
  required_input_set required;
  required.set(static_cast<size_t>(required_input::visibility));
  auto unit = f.coordinator.admit_unit({72, 1}, required);
  f.source(std::make_shared<accumulator>(1),
           out,
           single_job([] { return std::make_unique<tagged>(1); }, unit));
  auto cleanup = f.shutdown_guard();
  f.start();
  auto deadline = std::chrono::steady_clock::now() + 1s;
  while (!f.coordinator.snapshot().results_peak && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  REQUIRE(f.coordinator.snapshot().results_peak == 1);
  auto gate = f.coordinator.publication_gate();
  std::promise<void> cancellation_entered;
  auto entered = cancellation_entered.get_future();
  std::unique_lock lock(gate->mutex);
  auto cancellation = std::async(std::launch::async, [&] {
    cancellation_entered.set_value();
    return unit->cancel();
  });
  CHECK(entered.wait_for(1s) == std::future_status::ready);
  CHECK(cancellation.wait_for(20ms) == std::future_status::timeout);
  lock.unlock();
  REQUIRE(cancellation.get());
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(unit->record().state == unit_state::cancelled);
  CHECK_FALSE(unit->cancel());
}

TEST_CASE(
  "Temporary-memory exhaustion delays claims without closing the source; cancellation drains "
  "charged results",
  "[scan_preparation][coordinator][ledger]")
{
  auto cancel  = GENERATE(false, true);
  auto permits = GENERATE(1u, 2u);
  fixture f;
  auto out            = std::make_shared<sink>();
  auto blocked        = f.make_hold();
  auto second_blocked = f.make_hold();
  sirius::scan_manager::test::test_reservation_provider provider;
  provider.grant_bytes = 32 + 16 * permits;
  preparation_ledger ledger(provider);
  std::array envelope{scan_envelope{{{{16, 0, 0}, 16}, {{16, 0, 0}, 16}}, {}, true}};
  REQUIRE(
    ledger.admit(envelope, memory_space_id(cucascade::memory::Tier::HOST, 0), permits).n_permits ==
    permits);
  struct input : tagged {
    std::shared_ptr<sirius::op::scan::iceberg_delete_set const> deletes;
    input(int id, decltype(deletes) set) : tagged(id), deletes(std::move(set)) {}
  };
  int next = 0;
  preparation_coordinator::source source;
  source.coalescer = std::make_shared<accumulator>(1);
  source.try_claim = [&]() -> preparation_coordinator::claim_result {
    if (next >= 2) return {};
    auto key      = unit_key{1, next + 1u};
    auto acquired = ledger.try_acquire_unit(key, 16);
    if (!acquired) return preparation_pending{};
    auto id     = next++;
    auto permit = std::make_shared<w_permit>(std::move(acquired));
    return preparation_coordinator::job{
      [&, id, permit] {
        auto allocator = ledger.allocator(*permit);
        auto positions = allocator.allocate_retained(16);
        auto* data     = reinterpret_cast<int64_t*>(positions.data());
        data[0]        = 1;
        data[1]        = 3;
        auto set       = std::make_shared<sirius::op::scan::iceberg_delete_set const>(
          "file", std::move(positions), 2, 1);
        if (id == 0) blocked->wait();
        if (id == 1 && permits == 2) second_blocked->wait();
        return std::make_unique<input>(id, std::move(set));
      },
      {}};
  };
  source.construct = [](auto batch) { return make_publication(std::move(batch), 0); };
  source.close     = [out] {
    std::lock_guard lock(out->mutex);
    out->closed = true;
    out->cv.notify_all();
  };
  f.register_source(std::move(source), out);
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  if (permits == 2) REQUIRE(second_blocked->await());
  CHECK(ledger.permits_in_flight() == permits);
  CHECK(out->count() == 0);
  CHECK_FALSE(out->closed);
  if (cancel) f.coordinator.request_stop(stop_reason::user_cancel);
  blocked->release();
  second_blocked->release();
  if (!cancel) {
    REQUIRE(out->await(1));
    out->consume();
    REQUIRE(out->await(2));
    out->consume();
    REQUIRE(out->await_closed());
  }
  f.finish();
  f.coordinator.drain();
  CHECK(out->count() == (cancel ? 0 : 2));
  CHECK(ledger.permits_in_flight() == 0);
  CHECK(provider.seen->allocated_bytes == 0);
}

TEST_CASE("Observed interruption stops claims and publication while owned work drains",
          "[scan_preparation][coordinator][interrupt]")
{
  enum class point { waiting_for_worker, constructing_input, submitting_work };
  auto interruption =
    GENERATE(point::waiting_for_worker, point::constructing_input, point::submitting_work);
  CAPTURE(static_cast<int>(interruption));
  fixture f(1, 1);
  auto out     = std::make_shared<sink>();
  auto blocked = f.make_hold();
  std::atomic<bool> interrupted{false};
  std::atomic<unsigned> claims{0}, reads{0};
  f.coordinator.set_interrupt_check([&] { return interrupted.load(); });
  if (interruption == point::submitting_work)
    f.coordinator.before_submission_for_testing([&] { interrupted = true; });
  f.source(
    std::make_shared<accumulator>(1),
    out,
    [&]() -> preparation_coordinator::claim_result {
      if (claims++ != 0) return {};
      return preparation_coordinator::job{[&] {
                                            ++reads;
                                            if (interruption == point::waiting_for_worker)
                                              blocked->wait();
                                            return std::make_unique<tagged>(7);
                                          },
                                          {}};
    },
    interruption == point::constructing_input ? blocked : nullptr);
  auto guard = f.shutdown_guard();
  auto done  = f.completion.get_awaitable();
  f.start();
  if (interruption != point::submitting_work) {
    REQUIRE(blocked->await());
    interrupted = true;
    if (interruption == point::constructing_input) blocked->release();
  }
  REQUIRE(done.wait_for(1s) == std::future_status::ready);
  CHECK_THROWS_AS(done.get(), duckdb::InterruptException);
  CHECK_FALSE(f.coordinator.admit_unit({92, 2}, {}));
  CHECK_FALSE(f.coordinator.external_use());
  CHECK(claims == 1);
  if (interruption == point::waiting_for_worker)
    CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  blocked->release();
  f.finish();
  CHECK(out->count() == 0);
  CHECK(claims == 1);
  CHECK(reads == (interruption == point::submitting_work ? 0 : 1));
  CHECK(out->preparations == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  CHECK_THROWS_AS(f.coordinator.set_interrupt_check({}), std::logic_error);
}

TEST_CASE("Lookahead tracks work across result coalescing and publication until consumption",
          "[scan_preparation][coordinator]")
{
  auto rows_per_file                 = GENERATE(1, 4);
  auto lookahead                     = GENERATE(1, 2, 3);
  auto options                       = preparation_config{}.resolve(1);
  options.max_inflight_metadata_jobs = 1;
  options.lookahead_batches          = lookahead;
  fixture f(1, lookahead, k_interrupt_check_interval, options);
  auto out       = std::make_shared<sink>();
  auto completed = std::make_shared<std::atomic<size_t>>(0);
  f.source(std::make_shared<accumulator>(8), out, finite_jobs(100, [=](int id) {
             ++*completed;
             return std::make_unique<tagged>(id, rows_per_file);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(lookahead));
  std::this_thread::sleep_for(20ms);
  CHECK(completed->load() == lookahead * 8 / rows_per_file);
  CHECK(out->count() == lookahead);
  out->consume();
  REQUIRE(out->await(lookahead + 1));
  CHECK(completed->load() == (lookahead + 1) * 8 / rows_per_file);
  CHECK(f.coordinator.snapshot().jobs_peak == 1);
}

TEST_CASE("A file larger than lookahead streams through bounded output and reaches EOS",
          "[scan_preparation][coordinator]")
{
  auto options              = preparation_config{}.resolve(1);
  options.lookahead_batches = 2;
  fixture f(1, 2, k_interrupt_check_interval, options);
  auto out = std::make_shared<sink>();
  f.source(std::make_shared<accumulator>(8), out, single_job([] {
             return std::make_unique<tagged>(0, 41);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(2));
  std::this_thread::sleep_for(20ms);
  CHECK(out->count() == 2);
  for (size_t consumed = 0; consumed < 6; ++consumed) {
    REQUIRE(out->await(consumed + 1));
    out->consume();
  }
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(out->count() == 6);
  CHECK(f.coordinator.snapshot().output_peak <= 2);
}

TEST_CASE("Pruned preparation results do not consume the lookahead window",
          "[scan_preparation][coordinator]")
{
  auto options              = preparation_config{}.resolve(2);
  options.lookahead_batches = 1;
  fixture f(2, 1, k_interrupt_check_interval, options);
  auto out = std::make_shared<sink>();
  f.source(std::make_shared<accumulator>(8), out, finite_jobs(100, [](int id) {
             return std::make_unique<tagged>(id, id == 99 ? 1 : 0);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(out->count() == 1);
}

TEST_CASE("Size-based partial emission respects fill and minimum residence",
          "[scan_preparation][coordinator][partial_batch]")
{
  auto bytes               = GENERATE(49, 50, 100);
  auto options             = preparation_config{}.resolve(2);
  options.partial_batch    = partial_batch_options{};
  options.drain_quantum    = GENERATE(1u, 100u);
  options.max_control_work = 1;
  fixture f(2, 2, k_interrupt_check_interval, options);
  auto a        = std::make_shared<accumulator>(100);
  auto out      = std::make_shared<sink>();
  auto retained = f.make_hold(), blocked = f.make_hold();
  std::atomic<int> milliseconds{100};
  auto clock = [&] {
    return accumulator::clock::time_point{} + std::chrono::milliseconds(milliseconds.load());
  };
  a->fake_clock = clock;
  f.coordinator.clock_for_testing(clock);
  a->on_retained = [a = a.get(), bytes, retained] {
    if (a->retained + 1 == static_cast<size_t>(bytes)) retained->wait();
  };
  f.source(a, out, finite_jobs(2, [=](int id) {
             if (id == 1) blocked->wait();
             return std::make_unique<tagged>(id, id ? 0 : bytes);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(retained->await());
  milliseconds = 149;
  retained->release();
  REQUIRE(blocked->await());
  if (bytes == 100) {
    REQUIRE(out->await(1));  // Normal full batches do not wait for the minimum residence.
  } else {
    std::this_thread::sleep_for(20ms);
    CHECK(out->count() == 0);
    milliseconds = 150;
    f.coordinator.wake_for_testing();
    if (bytes == 50) {
      REQUIRE(out->await(1));
    } else {
      std::this_thread::sleep_for(20ms);
      CHECK(out->count() == 0);
      milliseconds = 2100;
      f.coordinator.wake_for_testing();
      REQUIRE(out->await(1));
    }
  }
  CHECK_FALSE(out->closed.load());
  {
    std::lock_guard lock(out->mutex);
    CHECK(out->pending_bytes.front() == bytes);
  }
  auto stats = f.coordinator.snapshot();
  CHECK(stats.normal_emissions == (bytes == 100 ? 1 : 0));
  CHECK(stats.size_emissions == (bytes == 50 ? 1 : 0));
  CHECK(stats.partial_emissions == (bytes == 100 ? 0 : 1));
}

TEST_CASE("Size-based partial emission waits for consumption but timeout can use spare capacity",
          "[scan_preparation][coordinator][partial_batch]")
{
  auto timeout          = GENERATE(false, true);
  auto options          = preparation_config{}.resolve(1);
  options.partial_batch = partial_batch_options{};
  options.drain_quantum = 100;
  fixture f(1, 2, k_interrupt_check_interval, options);
  auto a    = std::make_shared<accumulator>(100);
  auto out  = std::make_shared<sink>();
  auto next = f.make_hold(), retained = f.make_hold(), tail = f.make_hold();
  std::atomic<int> milliseconds{100};
  auto clock = [&] {
    return accumulator::clock::time_point{} + std::chrono::milliseconds(milliseconds.load());
  };
  a->fake_clock = clock;
  f.coordinator.clock_for_testing(clock);
  std::atomic<bool> second{false};
  a->on_retained = [&] {
    if (second && a->retained == 49) retained->wait();
  };
  f.source(a, out, finite_jobs(3, [=](int id) {
             if (id == 1) next->wait();
             if (id == 2) tail->wait();
             return std::make_unique<tagged>(id, id == 0 ? 100 : id == 1 ? 50 : 0);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));
  second = true;
  next->release();
  REQUIRE(retained->await());
  milliseconds = 150;
  retained->release();
  std::this_thread::sleep_for(20ms);
  CHECK(out->count() == 1);
  if (timeout) {
    milliseconds = 2100;
    f.coordinator.wake_for_testing();
  } else {
    out->consume();  // The existing callback must wake the owner without new metadata.
  }
  REQUIRE(out->await(2));
  CHECK_FALSE(out->closed.load());
}

TEST_CASE("Partial batch deadlines wake while all preparation workers are held",
          "[scan_preparation][coordinator][partial_batch]")
{
  auto bytes            = GENERATE(1, 50);
  auto options          = preparation_config{}.resolve(1);
  options.partial_batch = partial_batch_options{50, 10ms, 40ms};
  fixture f(1, 2, k_interrupt_check_interval, options);
  auto out     = std::make_shared<sink>();
  auto blocked = f.make_hold();
  f.source(std::make_shared<accumulator>(100), out, finite_jobs(2, [=](int id) {
             if (id == 1) blocked->wait();
             return std::make_unique<tagged>(id, id ? 0 : bytes);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  REQUIRE(out->await(1));
  auto stats = f.coordinator.snapshot();
  CHECK(stats.partial_emissions == 1);
  CHECK_FALSE(out->closed.load());
}

TEST_CASE("Claim jobs transfer their move-only captures and execute at most once",
          "[scan_preparation][coordinator][claim]")
{
  static_assert(!std::is_copy_constructible_v<preparation_coordinator::job>);
  struct resource {
    int& releases;
    ~resource() { ++releases; }
  };
  int releases = 0;
  int runs     = 0;
  {
    preparation_coordinator::job job{
      [owned = std::make_unique<resource>(releases), &runs](preparation_stop_token) {
        ++runs;
        return std::make_unique<tagged>(42);
      }};
    auto moved = std::move(job);
    CHECK_THROWS_AS(job.run({}), std::logic_error);
    auto output = moved.run({});
    CHECK(dynamic_cast<tagged&>(*output).id == 42);
    CHECK_THROWS_AS(moved.run({}), std::logic_error);
    CHECK(releases == 1);
  }
  CHECK(runs == 1);
  CHECK(releases == 1);
  {
    preparation_coordinator::job dropped{
      [owned = std::make_unique<resource>(releases), &runs](preparation_stop_token) {
        ++runs;
        return std::make_unique<tagged>(0);
      }};
  }
  CHECK(runs == 1);
  CHECK(releases == 2);
}

TEST_CASE("Pending claims do not block ready sources or become exhaustion",
          "[scan_preparation][coordinator][claim][native_lane]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::metadata_worker, domain::native_reader);
  CAPTURE(execution);
  native_reader_executor native(1, 2);
  fixture f(1, 1, 5s, std::nullopt, &native);
  auto first  = std::make_shared<sink>();
  auto second = std::make_shared<sink>();
  std::atomic<bool> ready{false};
  std::function<void()> wake;
  f.source(
    std::make_shared<accumulator>(1),
    first,
    [&, claimed = false]() mutable -> preparation_coordinator::claim_result {
      if (!ready) return preparation_pending{};
      if (std::exchange(claimed, true)) return preparation_exhausted{};
      return preparation_coordinator::job{[] { return std::make_unique<tagged>(1); }};
    },
    {},
    [&](auto callback) { wake = std::move(callback); },
    execution);
  f.source(std::make_shared<accumulator>(1),
           second,
           single_job([] { return std::make_unique<tagged>(2); }),
           {},
           {},
           execution);
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(second->await(1));
  CHECK_FALSE(first->closed);
  CHECK(first->count() == 0);
  second->consume();
  ready = true;
  wake();
  REQUIRE(first->await(1));
  first->consume();
  REQUIRE(first->await_closed());
  REQUIRE(second->await_closed());
  f.finish();
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("A readiness wake during an unlocked claim is not lost before sleep",
          "[scan_preparation][coordinator][claim]")
{
  fixture f(1, 1, 5s);
  auto out = std::make_shared<sink>();
  std::function<void()> wake;
  unsigned calls = 0;
  f.source(
    std::make_shared<accumulator>(1),
    out,
    [&]() -> preparation_coordinator::claim_result {
      if (++calls == 1) {
        wake();  // Arrives before try_claim returns pending.
        return preparation_pending{};
      }
      if (calls > 2) return preparation_exhausted{};
      return preparation_coordinator::job{[] { return std::make_unique<tagged>(3); }};
    },
    {},
    [&](auto callback) { wake = std::move(callback); });
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));  // Must not wait for the five-second interrupt poll.
  out->consume();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(calls == 3);
}

TEST_CASE("A dispatcher that drops accepted work cannot turn it into successful empty input",
          "[scan_preparation][coordinator][claim]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  std::atomic<int> runs{0};
  f.source(std::make_shared<accumulator>(1), out, single_job([&] {
             ++runs;
             return std::make_unique<tagged>(1);
           }));
  f.dispatcher.request_stop();
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(runs == 0);
  CHECK(out->count() == 0);
  CHECK(f.completion.has_error());
}

TEST_CASE("Discarded results and failed enqueues release temporary allocations outside the gate",
          "[scan_preparation][coordinator][temporary_release]")
{
  bool enqueue_failure = GENERATE(false, true);
  sirius::scan_manager::test::test_reservation_provider provider;
  auto ledger = std::make_shared<preparation_ledger>(provider);
  std::array envelope{scan_envelope{{{{16, 0, 0}, 16}}, {}, true}};
  REQUIRE(ledger->admit(envelope, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  fixture f;
  auto out     = std::make_shared<sink>();
  auto blocked = f.make_hold();
  auto gate    = f.coordinator.publication_gate();
  ledger->bind_wakeup(preparation_wakeup{gate});
  charged_block rescued;
  bool released_outside_gate = true;
  struct temporary_input : tagged {
    charged_block storage;
    std::function<void(charged_block&)> check_release;
    temporary_input(charged_block bytes, std::function<void(charged_block&)> check)
      : tagged(1), storage(std::move(bytes)), check_release(std::move(check))
    {
    }
    temporary_input(temporary_input&& other)
      : tagged(other.id, other.rows),
        storage(std::move(other.storage)),
        check_release(std::move(other.check_release))
    {
    }
    ~temporary_input() override
    {
      if (storage) check_release(storage);
    }
  };
  struct moving_coalescer : accumulator {
    cursor_step advance(sirius::op::scan::scan_info& input, size_t&, size_t) override
    {
      produced = true;
      return {std::make_unique<temporary_input>(std::move(dynamic_cast<temporary_input&>(input))),
              true};
    }
  };
  if (enqueue_failure) f.coordinator.fail_next_batch_enqueue_for_testing();
  f.source(std::make_shared<moving_coalescer>(), out, single_job([&] {
             auto permit = ledger->try_acquire_unit({1, 1}, 16);
             auto bytes  = ledger->allocator(permit).allocate(8);
             ledger->retire_unit({1, 1});
             blocked->wait();
             return std::make_unique<temporary_input>(
               std::move(bytes), [&](charged_block& storage) {
                 released_outside_gate = gate_is_available(gate);
                 // Report a regression instead of deadlocking while trying to wake the gate.
                 if (!released_outside_gate) rescued = std::move(storage);
               });
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  if (!enqueue_failure) f.coordinator.request_stop(stop_reason::normal_eos);
  blocked->release();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(released_outside_gate);
  rescued.release();
  CHECK(out->count() == 0);
  CHECK(f.completion.has_error() == enqueue_failure);
  if (enqueue_failure)
    CHECK_THROWS_AS(std::rethrow_exception(f.completion.terminal_error()), std::bad_alloc);
  CHECK(ledger->permits_in_flight() == 0);
  CHECK(ledger->tracked_units() == 0);
  CHECK(provider.seen->allocated_bytes == 0);
}

TEST_CASE("Failure callback credit covers the last physical input release",
          "[scan_preparation][coordinator][failure_release]")
{
  fixture f;
  auto out     = std::make_shared<sink>();
  auto release = f.make_hold();
  required_input_set required;
  required.set(static_cast<size_t>(required_input::visibility));
  required.set(static_cast<size_t>(required_input::checkpoint_iteration));
  auto unit  = f.coordinator.admit_unit({94, 1}, required);
  auto input = std::shared_ptr<sirius::op::scan::iceberg_delete_set const>(
    new sirius::op::scan::iceberg_delete_set("file"), [release](auto* value) {
      release->wait();
      delete value;
    });
  REQUIRE(unit->complete_input(visibility_input{std::move(input)}));
  std::promise<void> started;
  auto ready = started.get_future();
  f.source(std::make_shared<accumulator>(1),
           out,
           single_job(
             [&] {
               started.set_value();
               return std::make_unique<tagged>(1);
             },
             unit));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(ready.wait_for(1s) == std::future_status::ready);
  auto failure = std::async(std::launch::async, [&] {
    return unit->fail(
      preparation_failure{sirius::transparent::late_failure_cause::reader_io,
                          {},
                          "input release failure",
                          std::make_exception_ptr(std::runtime_error("input release failure"))});
  });
  CHECK(release->await());
  auto gate = f.coordinator.publication_gate();
  {
    std::lock_guard lock(gate->mutex);
    CHECK(gate->callbacks == 1);
  }
  CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  release->release();
  REQUIRE(failure.get());
  f.finish();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE(
  "Prepared publication keeps its concrete type and cancellation can win before queue commit",
  "[scan_preparation][coordinator][publication]")
{
  using input = sirius::op::scan::scan_operator_input;
  static_assert(
    std::is_same_v<decltype(preparation_coordinator::publication::input)::element_type, input>);
  fixture f;
  auto out = std::make_shared<sink>();
  f.source(
    std::make_shared<accumulator>(1), out, single_job([] { return std::make_unique<tagged>(1); }));
  bool prepared = false;
  f.coordinator.before_publication_for_testing([&] {
    // The hook runs outside the gate; cancellation must be able to acquire it.
    prepared = true;
    f.coordinator.request_stop(stop_reason::normal_eos);
  });
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(prepared);
  CHECK(out->count() == 0);
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("Prefetch preparation and visibility each execute at most once",
          "[scan_preparation][coordinator][publication]")
{
  using input = sirius::op::scan::scan_operator_input;
  input prepared(
    std::make_shared<tagged>(1), nullptr, 0, std::nullopt, input::prefetch_start::deferred);
  CHECK_THROWS_AS(prepared.publish_prefetch(), std::logic_error);
  REQUIRE_NOTHROW(prepared.prepare_prefetch());
  CHECK_THROWS_AS(prepared.prepare_prefetch(), std::logic_error);
  REQUIRE_NOTHROW(prepared.publish_prefetch());
  CHECK_THROWS_AS(prepared.publish_prefetch(), std::logic_error);
}

TEST_CASE("Coordinator startup gate prevents work before consumers are ready",
          "[scan_preparation][coordinator]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::metadata_worker, domain::query_owner, domain::native_reader);
  native_reader_executor executor(1, 1);
  fixture f(2, 4, k_interrupt_check_interval, std::nullopt, &executor);
  unsigned threads = 0;
  f.coordinator.before_thread_for_testing([&] { ++threads; });
  auto done = f.completion.get_awaitable();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::constructed);
  auto out = std::make_shared<sink>();
  std::atomic<size_t> claims{0};
  f.source(
    std::make_shared<accumulator>(),
    out,
    [&]() -> preparation_coordinator::claim_result {
      ++claims;
      return preparation_exhausted{};
    },
    {},
    {},
    execution);
  CHECK(f.coordinator.snapshot().runner == std::thread::id{});
  f.coordinator.arm();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::armed);
  CHECK(f.coordinator.snapshot().runs == 0);
  CHECK(claims == 0);
  CHECK(threads == (execution == domain::metadata_worker ? 0 : 1));
  SECTION("open startup gate")
  {
    f.coordinator.start();
    f.coordinator.drain();
    CHECK(claims == 1);
    CHECK(f.coordinator.snapshot().runs == 1);
    auto stats = f.coordinator.snapshot();
    CHECK(stats.owner == std::this_thread::get_id());
    CHECK((stats.runner == stats.owner) == (execution == domain::metadata_worker));
    CHECK(stats.runner != std::thread::id{});
    CHECK(done.wait_for(0ms) == std::future_status::timeout);
    auto waiting = std::chrono::steady_clock::now();
    CHECK_FALSE(f.coordinator.advance(10ms));
    CHECK(std::chrono::steady_clock::now() - waiting >= 10ms);
    f.completion.mark_completed();
    REQUIRE(done.wait_for(100ms) == std::future_status::ready);
    CHECK_NOTHROW(done.get());
    CHECK_THROWS_AS(f.coordinator.start(), std::logic_error);
  }
  SECTION("cancel before startup gate")
  {
    f.coordinator.request_stop(stop_reason::normal_eos);
    f.coordinator.drain();
    CHECK(claims == 0);
    CHECK(f.coordinator.snapshot().runs == 0);
  }
  CHECK(out->close_calls == 1);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  f.coordinator.drain();
}

TEST_CASE("Coordinator thread creation failure closes the preparation barrier exactly once",
          "[scan_preparation][coordinator]")
{
  native_reader_executor executor(1, 1);
  fixture f(2, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto out = std::make_shared<sink>();
  f.source(
    std::make_shared<accumulator>(),
    out,
    [] { return preparation_exhausted{}; },
    {},
    {},
    preparation_coordinator::execution_domain::native_reader);
  auto done = f.completion.get_awaitable();
  f.coordinator.before_thread_for_testing(
    [] { throw std::runtime_error("thread allocation failed"); });
  CHECK_THROWS_WITH(f.coordinator.arm(), "thread allocation failed");
  CHECK(f.coordinator.snapshot().runs == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  CHECK(out->close_calls == 1);
  REQUIRE(done.wait_for(100ms) == std::future_status::ready);
  CHECK_THROWS_WITH(done.get(), "thread allocation failed");
  f.coordinator.drain();
  CHECK(out->close_calls == 1);
}

TEST_CASE("Query owner can remain blocked while coordinator publishes and retires",
          "[scan_preparation][coordinator]")
{
  native_reader_executor executor(1, 1);
  fixture f(2, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto out = std::make_shared<sink>();
  f.source(std::make_shared<accumulator>(1),
           out,
           finite_jobs(1, [](int id) { return std::make_unique<tagged>(id); }),
           {},
           {},
           preparation_coordinator::execution_domain::native_reader);
  f.coordinator.arm();
  f.coordinator.start();
  // The query owner waits here: neither control work nor publication may need it.
  REQUIRE(out->await_closed());
  CHECK(out->count() == 1);
  f.coordinator.drain();
  auto stats = f.coordinator.snapshot();
  CHECK(stats.owner == std::this_thread::get_id());
  CHECK(stats.runner != stats.owner);
  CHECK(stats.publisher == stats.runner);
}

TEST_CASE("Consumers may finish before coordinator startup without a false startup error",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto done = f.completion.get_awaitable();
  f.coordinator.arm();
  f.completion.mark_completed();
  f.coordinator.drain();
  REQUIRE(done.wait_for(100ms) == std::future_status::ready);
  CHECK_NOTHROW(done.get());
  CHECK_NOTHROW(f.coordinator.start());
  CHECK_THROWS_AS(f.coordinator.start(), std::logic_error);
  CHECK(f.coordinator.snapshot().runs == 0);
}

TEST_CASE("Concurrent retirement waits for one physical join and one source close",
          "[scan_preparation][coordinator]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::metadata_worker, domain::native_reader);
  native_reader_executor executor(1, 1);
  fixture f(2, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto blocked = f.make_hold();
  auto out     = std::make_shared<sink>();
  f.source(std::make_shared<accumulator>(1),
           out,
           finite_jobs(1,
                       [blocked](int id) {
                         blocked->wait();
                         return std::make_unique<tagged>(id);
                       }),
           {},
           {},
           execution);
  std::promise<void> first_entered, second_entered;
  std::future<void> first, second;
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  f.coordinator.request_stop(stop_reason::normal_eos);
  first  = std::async(std::launch::async, [&] {
    first_entered.set_value();
    f.coordinator.drain();
  });
  second = std::async(std::launch::async, [&] {
    second_entered.set_value();
    f.coordinator.drain();
  });
  first_entered.get_future().wait();
  second_entered.get_future().wait();
  CHECK(first.wait_for(0ms) == std::future_status::timeout);
  CHECK(second.wait_for(0ms) == std::future_status::timeout);
  blocked->release();
  f.finish();
  first.get();
  second.get();
  CHECK(out->close_calls == 1);
  CHECK(out->count() == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Coordinator rejects self join before acquiring the retirement mutex",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  std::atomic<bool> rejected{false};
  f.coordinator.before_publication_for_testing([&] {
    try {
      f.coordinator.drain();
    } catch (std::logic_error const&) {
      rejected = true;
    }
  });
  f.source(std::make_shared<accumulator>(1), out, finite_jobs(1, [](int id) {
             return std::make_unique<tagged>(id);
           }));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(rejected);
  CHECK(out->count() == 1);
}

TEST_CASE("Startup cancellation close hook cannot self join the coordinator",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  std::atomic<bool> rejected{false};
  preparation_coordinator::source source;
  source.coalescer = std::make_shared<accumulator>();
  source.try_claim = [] { return preparation_exhausted{}; };
  source.construct = [](auto) { return preparation_coordinator::publication{}; };
  source.close     = [&] {
    try {
      f.coordinator.drain();
    } catch (std::logic_error const&) {
      rejected = true;
    }
  };
  f.register_source(std::move(source), out);
  f.coordinator.arm();
  f.coordinator.request_stop(stop_reason::normal_eos);
  f.coordinator.drain();
  CHECK(rejected);
  CHECK(f.coordinator.snapshot().runs == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Reader work captures and close stay on the assigned thread across pulls",
          "[scan_preparation][provider_lane][native_lane]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::query_owner, domain::native_reader);
  native_reader_executor executor(2, 4);
  fixture f(1, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto out = std::make_shared<sink>();
  std::vector<std::thread::id> workers, disposers;
  bool closed_after_release = false;
  auto source               = make_source(execution);
  source.try_claim          = [&, next = 0]() mutable -> preparation_coordinator::claim_result {
    if (next == 3) return preparation_exhausted{};
    auto id      = next++;
    auto capture = std::shared_ptr<int>(new int, [&](int* p) {
      f.coordinator.snapshot();  // Capture disposal must not hold the publication gate.
      disposers.push_back(std::this_thread::get_id());
      delete p;
    });
    return preparation_coordinator::job{[&, id, capture = std::move(capture)] {
      workers.push_back(std::this_thread::get_id());
      return std::make_unique<tagged>(id, id == 0 ? 0 : 1);
    }};
  };
  source.close = [&] {
    closed_after_release = disposers.size() == 3;
    out->close();
  };
  auto guard = f.shutdown_guard();
  f.start([&] { f.register_source(std::move(source), out); });
  REQUIRE(out->await_closed());
  f.finish();
  REQUIRE(workers.size() == 3);
  CHECK(disposers == workers);
  CHECK(closed_after_release);
  CHECK((workers.front() == f.coordinator.snapshot().owner) == (execution == domain::query_owner));
  CHECK(workers.front() != f.coordinator.snapshot().runner);
  CHECK(std::ranges::all_of(workers, [&](auto id) { return id == workers.front(); }));
  CHECK(out->closing_thread == workers.front());
  CHECK(out->close_calls == 1);
  CHECK(out->count() == 2);  // A zero-row pull does not exhaust the reader.
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("Blocked provider operation does not occupy metadata capacity or coordinator",
          "[scan_preparation][provider_lane]")
{
  fixture f(1, 2);
  auto blocked  = f.make_hold();
  auto provider = std::make_shared<sink>();
  auto metadata = std::make_shared<sink>();
  auto guard    = f.shutdown_guard();
  f.start([&] {
    f.source(std::make_shared<accumulator>(1),
             provider,
             single_job([blocked] {
               blocked->wait();
               return std::make_unique<tagged>(1);
             }),
             {},
             {},
             preparation_coordinator::execution_domain::query_owner);
    f.source(std::make_shared<accumulator>(1), metadata, single_job([] {
               return std::make_unique<tagged>(2);
             }));
  });
  REQUIRE(blocked->await());
  REQUIRE(metadata->await(1));
  CHECK(provider->count() == 0);
  blocked->release();
  f.finish();
  CHECK(provider->count() == 1);
}

TEST_CASE("Provider zero row result is not exhaustion and lookahead blocks further pulls",
          "[scan_preparation][provider_lane]")
{
  fixture f(1, 1);
  auto out = std::make_shared<sink>();
  std::atomic<size_t> calls{0};
  auto guard = f.shutdown_guard();
  f.start([&] {
    f.source(std::make_shared<accumulator>(1),
             out,
             finite_jobs(3,
                         [&](int id) {
                           ++calls;
                           return std::make_unique<tagged>(id, id == 0 ? 0 : 1);
                         }),
             {},
             {},
             preparation_coordinator::execution_domain::query_owner);
  });
  REQUIRE(out->await(1));
  CHECK(calls == 2);
  out->consume();
  REQUIRE(out->await(2));
  CHECK(calls == 3);
  // Terminal close after stop cannot wait for this full output window to be consumed.
  f.coordinator.request_stop(stop_reason::normal_eos);
  f.finish();
  CHECK(out->close_calls == 1);
}

TEST_CASE("Cancelling an active provider waits for its return and suppresses late output",
          "[scan_preparation][provider_lane]")
{
  fixture f;
  auto blocked = f.make_hold();
  auto out     = std::make_shared<sink>();
  std::atomic<bool> saw_stop{false};
  auto guard = f.shutdown_guard();
  f.start([&] {
    f.source(
      std::make_shared<accumulator>(1),
      out,
      [&, claimed = false]() mutable -> preparation_coordinator::claim_result {
        if (std::exchange(claimed, true)) return preparation_exhausted{};
        return preparation_coordinator::job{[&](preparation_stop_token stop) {
          blocked->wait();
          saw_stop = stop.stop_requested();
          return std::make_unique<tagged>(1);
        }};
      },
      {},
      {},
      preparation_coordinator::execution_domain::query_owner);
  });
  REQUIRE(blocked->await());
  f.coordinator.request_stop(stop_reason::user_cancel);
  CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  CHECK(out->close_calls == 0);
  blocked->release();
  f.finish();
  CHECK(saw_stop);
  CHECK(out->count() == 0);
  CHECK(out->close_calls == 1);
  CHECK(f.completion.has_error());
}

TEST_CASE("Retired provider results and wakeups cannot affect another attempt",
          "[scan_preparation][provider_lane][attempt_isolation]")
{
  const bool late_error = GENERATE(false, true);
  auto released         = std::make_shared<std::atomic<bool>>(false);
  auto finish_current   = std::make_shared<std::atomic<bool>>(false);
  auto old_wakeup       = std::make_shared<preparation_wakeup>();
  auto current_wakeup   = std::make_shared<preparation_wakeup>();
  auto old              = std::make_unique<fixture>();
  fixture current;
  auto old_out                  = std::make_shared<sink>();
  auto current_out              = std::make_shared<sink>();
  auto blocked                  = old->make_hold();
  auto owner                    = std::shared_ptr<int>(new int(1), [released](int* value) {
    delete value;
    *released = true;
  });
  std::weak_ptr<int> weak_owner = owner;
  auto old_guard                = old->shutdown_guard();
  auto current_guard            = current.shutdown_guard();
  old->start([&, owner]() mutable {
    auto source      = make_source(preparation_coordinator::execution_domain::query_owner,
                              std::make_shared<accumulator>(1));
    source.try_claim = single_job([owner, blocked, late_error, completion = &old->completion] {
      blocked->wait();
      if (late_error)
        completion->report_error(std::make_exception_ptr(std::runtime_error("late old error")));
      else
        completion->mark_completed();
      return std::make_unique<tagged>(101);
    });
    source.construct = [](auto) -> preparation_coordinator::publication {
      throw std::runtime_error("retired attempt reached publication");
    };
    source.close = [owner, old_out]() mutable {
      // The provider session owner remains valid until Q finishes close.
      owner.reset();
      old_out->close();
    };
    source.bind_wakeup = [old_wakeup](preparation_wakeup wake) { *old_wakeup = std::move(wake); };
    old->register_source(std::move(source), old_out);
    owner.reset();  // Only the claimed job and Q close retain the session owner.
  });
  owner.reset();
  REQUIRE(blocked->await());
  old->coordinator.request_stop(stop_reason::user_cancel);
  auto first_error = old->completion.terminal_error();
  REQUIRE(first_error);
  CHECK_FALSE(weak_owner.expired());
  CHECK_FALSE(released->load());
  CHECK(old_out->close_calls == 0);

  current.start([&] {
    current.source(
      std::make_shared<accumulator>(1),
      current_out,
      [finish_current, claimed = false]() mutable -> preparation_coordinator::claim_result {
        if (!std::exchange(claimed, true))
          return preparation_coordinator::job{[] { return std::make_unique<tagged>(202); }};
        if (finish_current->load()) return preparation_exhausted{};
        return preparation_pending{};
      },
      {},
      [current_wakeup](preparation_wakeup wake) { *current_wakeup = std::move(wake); },
      preparation_coordinator::execution_domain::query_owner);
  });
  REQUIRE(current_out->await(1));
  // A's result and completion arrive only after B has published its own output.
  blocked->release();
  old->finish();
  REQUIRE_FALSE(old->owner_error);
  CHECK(old->completion.terminal_error() == first_error);
  CHECK(old_out->count() == 0);
  CHECK(old_out->close_calls == 1);
  CHECK(old_out->closing_thread == old->coordinator.snapshot().owner);
  CHECK(weak_owner.expired());
  CHECK(released->load());
  old_guard.reset();
  old.reset();
  (*old_wakeup)();  // An escaped notification remains safe after A's destruction.

  CHECK_FALSE(current.completion.has_error());
  CHECK(current_out->count() == 1);
  CHECK(current_out->close_calls == 0);
  CHECK(current.coordinator.snapshot().phase == preparation_coordinator::lifecycle::running);
  {
    std::lock_guard lock(current_out->mutex);
    CHECK(current_out->published == std::vector<int>{202});
  }
  current_out->consume();
  *finish_current = true;
  (*current_wakeup)();
  REQUIRE(current_out->await_closed());
  current.finish();
  CHECK_FALSE(current.owner_error);
  CHECK_FALSE(current.completion.has_error());
  CHECK(current_out->close_calls == 1);
  CHECK(current_out->count() == 1);
}

TEST_CASE("Reader close after startup failure preserves the first error and returns resources",
          "[scan_preparation][provider_lane][native_lane]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::query_owner, domain::native_reader);
  CAPTURE(execution);
  native_reader_executor native(1, 1);
  fixture f(1, 1, k_interrupt_check_interval, std::nullopt, &native);
  auto out                 = std::make_shared<sink>();
  size_t closes            = 0;
  bool self_drain_rejected = false;
  std::thread::id closer;
  auto source      = make_source(execution, std::make_shared<accumulator>());
  source.try_claim = [] { return preparation_exhausted{}; };
  source.construct = [](auto) { return preparation_coordinator::publication{}; };
  source.close     = [&] {
    ++closes;
    closer = std::this_thread::get_id();
    try {
      f.coordinator.drain();
    } catch (std::logic_error const&) {
      self_drain_rejected = true;
    }
    throw std::runtime_error("provider close error");
  };
  f.register_source(std::move(source), out);
  f.coordinator.before_thread_for_testing(
    [] { throw std::runtime_error("thread allocation failed"); });
  auto done = f.completion.get_awaitable();
  CHECK_THROWS_WITH(f.coordinator.arm(), "thread allocation failed");
  f.coordinator.drain();
  CHECK(closes == 1);
  CHECK(self_drain_rejected);
  CHECK((closer == std::this_thread::get_id()) == (execution == domain::query_owner));
  auto reused = native.reserve(1);
  CHECK(reused.size() == 1);
  CHECK_THROWS_WITH(done.get(), "thread allocation failed");
  CHECK_THROWS_WITH(f.completion.throw_if_preparation_cleanup_failed(), "thread allocation failed");
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Unpumped provider requests stay bounded and cancellation disposes captures before close",
          "[scan_preparation][provider_lane]")
{
  bool const before_dispatch = GENERATE(false, true);
  CAPTURE(before_dispatch);
  fixture f;
  auto submitted = f.make_hold();
  auto out       = std::make_shared<sink>();
  std::atomic<size_t> claims{0};
  std::promise<void> revisited;
  auto progressed = revisited.get_future();
  bool ran = false, released = false, closed_after_release = false;
  std::thread::id destroyed;
  auto capture     = std::shared_ptr<int>(new int, [&](int* p) {
    f.coordinator.snapshot();
    destroyed = std::this_thread::get_id();
    released  = true;
    delete p;
  });
  auto source      = make_source(preparation_coordinator::execution_domain::query_owner,
                            std::make_shared<accumulator>());
  source.try_claim = [capture = std::move(capture),
                      &ran,
                      &claims,
                      claimed = false]() mutable -> preparation_coordinator::claim_result {
    ++claims;
    if (std::exchange(claimed, true)) return preparation_exhausted{};
    return preparation_coordinator::job{[capture = std::move(capture), &ran] {
      ran = true;
      return std::make_unique<tagged>(1);
    }};
  };
  source.construct = [](auto) { return preparation_coordinator::publication{}; };
  source.close     = [&] { closed_after_release = released; };
  f.register_source(std::move(source), out);
  if (before_dispatch)
    f.coordinator.before_submission_for_testing([submitted] { submitted->wait(); });
  // Two visits ensure the coordinator rechecks the queued reader without Q pumping it.
  f.source(std::make_shared<accumulator>(),
           std::make_shared<sink>(),
           [&, visits = 0]() mutable -> preparation_coordinator::claim_result {
             if (++visits == 1) f.coordinator.wake_for_testing();
             if (visits == 2) revisited.set_value();
             return preparation_pending{};
           });
  auto guard = f.shutdown_guard();
  f.coordinator.arm();
  f.coordinator.start();
  if (before_dispatch)
    REQUIRE(submitted->await());
  else
    REQUIRE(progressed.wait_for(1s) == std::future_status::ready);
  CHECK(claims == 1);
  CHECK_FALSE(ran);
  CHECK_FALSE(released);
  f.coordinator.request_stop(stop_reason::normal_eos);
  CHECK(f.coordinator.advance());
  CHECK(released);
  CHECK_FALSE(ran);
  CHECK(destroyed == std::this_thread::get_id());
  submitted->release();
  f.coordinator.drain();
  CHECK(closed_after_release);
  CHECK(claims == 1);
  CHECK(out->count() == 0);
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("Reader close waits for the control driver to finish an active claim",
          "[scan_preparation][coordinator][claim]")
{
  fixture f;
  auto reader   = std::make_shared<sink>();
  auto metadata = std::make_shared<sink>();
  auto claiming = f.make_hold();
  auto guard    = f.shutdown_guard();
  f.source(
    std::make_shared<accumulator>(1),
    reader,
    []() -> preparation_coordinator::claim_result { return preparation_pending{}; },
    {},
    {},
    preparation_coordinator::execution_domain::query_owner);
  f.source(std::make_shared<accumulator>(1),
           metadata,
           [claiming]() -> preparation_coordinator::claim_result {
             claiming->wait();
             return preparation_exhausted{};
           });
  f.coordinator.arm();
  f.coordinator.start();
  REQUIRE(claiming->await());
  f.coordinator.request_stop(stop_reason::normal_eos);
  // Pumping Q can dispose returned jobs, but cannot close a reader ahead of the driver.
  f.coordinator.advance();
  CHECK(reader->close_calls == 0);
  CHECK(metadata->close_calls == 0);
  claiming->release();
  f.coordinator.drain();
  CHECK(reader->close_calls == 1);
  CHECK(reader->closing_thread == std::this_thread::get_id());
  CHECK(metadata->close_calls == 1);
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("External provider retirement waits for Q close without blocking Q drain",
          "[scan_preparation][provider_lane]")
{
  bool before_arm = GENERATE(false, true);
  fixture f;
  auto out     = std::make_shared<sink>();
  auto running = f.make_hold();
  auto closing = f.make_hold();
  std::thread::id closer;
  std::future<void> external;
  auto guard           = f.shutdown_guard();
  auto register_source = [&] {
    auto source      = make_source(preparation_coordinator::execution_domain::query_owner,
                              std::make_shared<accumulator>());
    source.try_claim = single_job([running] {
      running->wait();
      return std::make_unique<tagged>(1);
    });
    source.construct = [](auto) { return preparation_coordinator::publication{}; };
    source.close     = [&] {
      closer = std::this_thread::get_id();
      closing->wait();
    };
    f.register_source(std::move(source), out);
  };
  if (before_arm) {
    f.owner = std::thread([&] {
      register_source();
      running->wait();
      f.coordinator.drain();
    });
  } else {
    f.start(register_source);
  }
  auto owner = f.owner.get_id();
  REQUIRE(running->await());
  f.coordinator.request_stop(stop_reason::normal_eos);
  external = std::async(std::launch::async, [&] { f.coordinator.drain(); });
  CHECK(external.wait_for(10ms) == std::future_status::timeout);
  running->release();
  REQUIRE(closing->await());
  CHECK(external.wait_for(10ms) == std::future_status::timeout);
  CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  closing->release();
  REQUIRE(external.wait_for(1s) == std::future_status::ready);
  external.get();
  f.finish();
  CHECK(closer == owner);
  CHECK(out->count() == 0);
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("Provider pump rejects a different owner and recursive execution or drain",
          "[scan_preparation][provider_lane]")
{
  fixture f;
  auto out            = std::make_shared<sink>();
  size_t rejected     = 0;
  auto reject_reentry = [&] {
    try {
      f.coordinator.advance();
    } catch (std::logic_error const&) {
      ++rejected;
    }
    try {
      f.coordinator.drain();
    } catch (std::logic_error const&) {
      ++rejected;
    }
  };
  auto source      = make_source(preparation_coordinator::execution_domain::query_owner,
                            std::make_shared<accumulator>(1));
  source.try_claim = single_job([&] {
    reject_reentry();
    return std::make_unique<tagged>(1);
  });
  source.close     = reject_reentry;
  f.register_source(std::move(source), out);
  auto foreign = std::async(std::launch::async, [&] {
    try {
      f.coordinator.advance();
    } catch (std::logic_error const&) {
      return true;
    }
    return false;
  });
  CHECK(foreign.get());
  f.coordinator.arm();
  f.coordinator.start();
  f.coordinator.drain();
  CHECK(rejected == 4);
  CHECK(out->count() == 1);
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("Retirement racing arm stops the startup gate before waiting for its runner",
          "[scan_preparation][coordinator]")
{
  native_reader_executor executor(1, 1);
  fixture f(2, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto creating = f.make_hold();
  auto out      = std::make_shared<sink>();
  f.source(std::make_shared<accumulator>(1),
           out,
           single_job([] { return std::make_unique<tagged>(1); }),
           {},
           {},
           preparation_coordinator::execution_domain::native_reader);
  f.coordinator.before_thread_for_testing([creating] { creating->wait(); });
  std::future<void> arming, retiring;
  auto guard = f.shutdown_guard();
  arming     = std::async(std::launch::async, [&] { f.coordinator.arm(); });
  REQUIRE(creating->await());
  retiring = std::async(std::launch::async, [&] { f.coordinator.drain(); });
  CHECK(retiring.wait_for(10ms) == std::future_status::timeout);
  creating->release();
  arming.get();
  REQUIRE(retiring.wait_for(1s) == std::future_status::ready);
  retiring.get();
  CHECK(out->count() == 0);
  CHECK(out->close_calls == 1);
  CHECK(f.coordinator.snapshot().runs == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Blocked native reader leaves metadata provider and coordinator capacity available",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(1, 4);
  fixture f(1, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto blocked  = f.make_hold();
  auto native   = std::make_shared<sink>();
  auto waiting  = std::make_shared<sink>();
  auto metadata = std::make_shared<sink>();
  auto provider = std::make_shared<sink>();
  std::atomic<size_t> waiting_claims{0};
  auto guard = f.shutdown_guard();
  f.start([&] {
    f.source(std::make_shared<accumulator>(1),
             native,
             single_job([blocked] {
               blocked->wait();
               return std::make_unique<tagged>(1);
             }),
             {},
             {},
             preparation_coordinator::execution_domain::native_reader);
    f.source(
      std::make_shared<accumulator>(1),
      waiting,
      [&, claimed = false]() mutable -> preparation_coordinator::claim_result {
        ++waiting_claims;
        if (std::exchange(claimed, true)) return preparation_exhausted{};
        return preparation_coordinator::job{[] { return std::make_unique<tagged>(2); }};
      },
      {},
      {},
      preparation_coordinator::execution_domain::native_reader);
    f.source(std::make_shared<accumulator>(1), metadata, single_job([] {
               return std::make_unique<tagged>(3);
             }));
    f.source(std::make_shared<accumulator>(1),
             provider,
             single_job([] { return std::make_unique<tagged>(4); }),
             {},
             {},
             preparation_coordinator::execution_domain::query_owner);
  });
  REQUIRE(blocked->await());
  REQUIRE(metadata->await(1));
  REQUIRE(provider->await(1));
  CHECK_THROWS_WITH(executor.stop(), "cannot stop native executor with active readers");
  CHECK(waiting_claims <= 1);
  CHECK(native->count() == 0);
  blocked->release();
  REQUIRE(waiting->await_closed());
  f.finish();
  CHECK(native->count() == 1);
  CHECK(waiting->count() == 1);
  CHECK_FALSE(f.completion.has_error());
}

TEST_CASE("Native executor is shared and stopping one attempt does not stop its neighbor",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(2, 4);
  fixture first(1, 4, k_interrupt_check_interval, std::nullopt, &executor);
  fixture second(1, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto blocked  = first.make_hold();
  auto native   = std::make_shared<sink>();
  auto waiting  = std::make_shared<sink>();
  auto metadata = std::make_shared<sink>();
  std::atomic<size_t> claims{0};
  first.source(std::make_shared<accumulator>(1),
               native,
               single_job([blocked] {
                 blocked->wait();
                 return std::make_unique<tagged>(1);
               }),
               {},
               {},
               preparation_coordinator::execution_domain::native_reader);
  second.source(
    std::make_shared<accumulator>(1),
    waiting,
    [&]() -> preparation_coordinator::claim_result {
      ++claims;
      return preparation_exhausted{};
    },
    {},
    {},
    preparation_coordinator::execution_domain::native_reader);
  second.source(std::make_shared<accumulator>(1), metadata, single_job([] {
                  return std::make_unique<tagged>(2);
                }));
  auto first_guard  = first.shutdown_guard();
  auto second_guard = second.shutdown_guard();
  first.start();
  REQUIRE(blocked->await());
  second.start();
  REQUIRE(metadata->await(1));
  second.coordinator.request_stop(stop_reason::normal_eos);
  second.finish();
  CHECK(claims <= 1);
  CHECK(waiting->close_calls == 1);
  CHECK(native->close_calls == 0);
  blocked->release();
  first.finish();
  CHECK(native->count() == 1);
  CHECK(native->close_calls == 1);
  CHECK_FALSE(first.completion.has_error());
}

TEST_CASE("Native stop waits for noninterruptible next and closes on its thread",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(1, 4);
  fixture f(1, 1, k_interrupt_check_interval, std::nullopt, &executor);
  auto out     = std::make_shared<sink>();
  auto blocked = f.make_hold();
  auto closing = f.make_hold();
  std::thread::id next_thread, close_thread, destroy_thread;
  std::atomic<bool> released{false};
  auto capture     = std::shared_ptr<int>(new int, [&](int* p) {
    f.coordinator.snapshot();
    destroy_thread = std::this_thread::get_id();
    released       = true;
    delete p;
  });
  auto source      = make_source(preparation_coordinator::execution_domain::native_reader,
                            std::make_shared<accumulator>(1));
  source.try_claim = [&,
                      capture = std::move(capture),
                      claimed = false]() mutable -> preparation_coordinator::claim_result {
    if (std::exchange(claimed, true)) return preparation_exhausted{};
    return preparation_coordinator::job{[&, capture = std::move(capture)] {
      next_thread = std::this_thread::get_id();
      blocked->wait();
      return std::make_unique<tagged>(1);
    }};
  };
  source.construct = [](auto) { return preparation_coordinator::publication{}; };
  source.close     = [&] {
    close_thread = std::this_thread::get_id();
    closing->wait();
  };
  f.register_source(std::move(source), out);
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  f.coordinator.request_stop(stop_reason::user_cancel);
  CHECK_FALSE(released);
  CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  blocked->release();
  REQUIRE(closing->await());
  CHECK(released);
  CHECK(next_thread == destroy_thread);
  CHECK(next_thread == close_thread);
  CHECK(f.coordinator.snapshot().phase != preparation_coordinator::lifecycle::quiescent);
  closing->release();
  f.finish();
  CHECK(out->count() == 0);
  CHECK(f.completion.has_error());
}

TEST_CASE("Native readers share a fixed worker while one source is backpressured",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(1, 2);
  fixture f(1, 1, k_interrupt_check_interval, std::nullopt, &executor);
  auto probe = std::make_shared<sink>();
  auto build = std::make_shared<sink>();
  std::atomic<size_t> pulls{0};
  f.source(std::make_shared<accumulator>(1),
           probe,
           finite_jobs(2,
                       [&](int id) {
                         ++pulls;
                         return std::make_unique<tagged>(id);
                       }),
           {},
           {},
           preparation_coordinator::execution_domain::native_reader);
  f.source(std::make_shared<accumulator>(1),
           build,
           single_job([] { return std::make_unique<tagged>(9); }),
           {},
           {},
           preparation_coordinator::execution_domain::native_reader);
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(probe->await(1));
  // Model a join whose probe cannot be consumed before the build has produced its rows.
  REQUIRE(build->await(1));
  CHECK(pulls == 1);
  probe->consume();
  build->consume();
  REQUIRE(probe->await(2));
  f.coordinator.request_stop(stop_reason::normal_eos);  // close needs no output credit
  f.finish();
  CHECK(pulls == 2);
  CHECK(probe->close_calls == 1);
  CHECK(build->close_calls == 1);
  CHECK(probe->closing_thread == build->closing_thread);
}

TEST_CASE("Native reader group admission fails without partial leases or neighbor cancellation",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(1, 2);
  fixture first(1, 4, k_interrupt_check_interval, std::nullopt, &executor);
  fixture second(1, 4, k_interrupt_check_interval, std::nullopt, &executor);
  auto blocked = first.make_hold();
  auto live    = std::make_shared<sink>();
  first.source(std::make_shared<accumulator>(1),
               live,
               single_job([blocked] {
                 blocked->wait();
                 return std::make_unique<tagged>(1);
               }),
               {},
               {},
               preparation_coordinator::execution_domain::native_reader);
  auto a = std::make_shared<sink>();
  auto b = std::make_shared<sink>();
  for (auto out : {a, b})
    second.source(std::make_shared<accumulator>(1),
                  out,
                  single_job([] { return std::make_unique<tagged>(2); }),
                  {},
                  {},
                  preparation_coordinator::execution_domain::native_reader);
  auto guard = first.shutdown_guard();
  first.start();
  REQUIRE(blocked->await());
  CHECK_THROWS_WITH(second.coordinator.arm(),
                    "native reader capacity unavailable for complete attempt");
  CHECK(a->close_calls == 0);
  CHECK(b->close_calls == 0);
  CHECK(a->count() == 0);
  // Failure did not retain the one otherwise available slot.
  auto free = executor.reserve(1);
  free.clear();
  CHECK(live->close_calls == 0);
  blocked->release();
  first.finish();
  CHECK(live->count() == 1);
  CHECK_FALSE(first.completion.has_error());
  auto all = executor.reserve(2);
  CHECK(all.size() == 2);
}

TEST_CASE("Native execution and close errors retain the first error and return the lease",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(1, 1);
  fixture f(1, 1, k_interrupt_check_interval, std::nullopt, &executor);
  auto out = std::make_shared<sink>();
  std::thread::id next_thread, close_thread;
  size_t rejected  = 0;
  auto source      = make_source(preparation_coordinator::execution_domain::native_reader,
                            std::make_shared<accumulator>());
  source.try_claim = single_job([&]() -> std::unique_ptr<sirius::op::scan::scan_info> {
    next_thread = std::this_thread::get_id();
    try {
      f.coordinator.drain();
    } catch (std::logic_error const&) {
      ++rejected;
    }
    throw std::runtime_error("native next failed");
  });
  source.construct = [](auto) { return preparation_coordinator::publication{}; };
  source.close     = [&] {
    close_thread = std::this_thread::get_id();
    try {
      f.coordinator.drain();
    } catch (std::logic_error const&) {
      ++rejected;
    }
    throw std::runtime_error("native close failed");
  };
  f.register_source(std::move(source), out);
  auto done = f.completion.get_awaitable();
  f.coordinator.arm();
  f.coordinator.start();
  f.coordinator.drain();
  CHECK_THROWS_WITH(done.get(), "native next failed");
  CHECK_THROWS_WITH(f.completion.throw_if_preparation_cleanup_failed(), "native next failed");
  CHECK(next_thread == close_thread);
  CHECK(rejected == 2);
  CHECK(out->count() == 0);
  auto reused = executor.reserve(1);
  CHECK(reused.size() == 1);
}

TEST_CASE(
  "Reader submission failure disposes accepted captures before close on its assigned thread",
  "[scan_preparation][provider_lane][native_lane]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::query_owner, domain::native_reader);
  CAPTURE(execution);
  native_reader_executor executor(1, 1);
  fixture f(1, 1, k_interrupt_check_interval, std::nullopt, &executor);
  auto out = std::make_shared<sink>();
  std::thread::id destroyed, closed;
  bool ran = false, closed_after_release = false;
  auto capture     = std::shared_ptr<int>(new int, [&](int* p) {
    f.coordinator.snapshot();
    destroyed = std::this_thread::get_id();
    delete p;
  });
  auto source      = make_source(execution, std::make_shared<accumulator>());
  source.try_claim = [&,
                      capture = std::move(capture),
                      claimed = false]() mutable -> preparation_coordinator::claim_result {
    if (std::exchange(claimed, true)) return preparation_exhausted{};
    return preparation_coordinator::job{[&, capture = std::move(capture)] {
      ran = true;
      return std::make_unique<tagged>(1);
    }};
  };
  source.construct = [](auto) { return preparation_coordinator::publication{}; };
  source.close     = [&] {
    closed               = std::this_thread::get_id();
    closed_after_release = destroyed == closed;
  };
  f.register_source(std::move(source), out);
  f.coordinator.before_submission_for_testing(
    [] { throw std::runtime_error("reader submit failed"); });
  auto done = f.completion.get_awaitable();
  f.coordinator.arm();
  f.coordinator.start();
  f.coordinator.drain();
  CHECK_THROWS_WITH(done.get(), "reader submit failed");
  CHECK_FALSE(ran);
  CHECK(closed_after_release);
  CHECK(closed != f.coordinator.snapshot().runner);
  CHECK((closed == f.coordinator.snapshot().owner) == (execution == domain::query_owner));
  auto reused = executor.reserve(1);
  CHECK(reused.size() == 1);
  CHECK(out->count() == 0);
}

TEST_CASE("Queued native work is disposed on its worker after cancellation",
          "[scan_preparation][native_lane]")
{
  native_reader_executor executor(1, 2);
  fixture f(1, 1, k_interrupt_check_interval, std::nullopt, &executor);
  auto blocked = f.make_hold();
  auto first   = std::make_shared<sink>();
  auto queued  = std::make_shared<sink>();
  std::promise<void> submitted;
  size_t submissions = 0;
  std::atomic<bool> ran{false};
  std::thread::id worker, destroyed;
  auto capture = std::shared_ptr<int>(new int, [&](int* p) {
    f.coordinator.snapshot();
    destroyed = std::this_thread::get_id();
    delete p;
  });
  f.source(std::make_shared<accumulator>(1),
           first,
           single_job([&] {
             worker = std::this_thread::get_id();
             blocked->wait();
             return std::make_unique<tagged>(1);
           }),
           {},
           {},
           preparation_coordinator::execution_domain::native_reader);
  f.source(
    std::make_shared<accumulator>(1),
    queued,
    [&,
     capture = std::move(capture),
     claimed = false]() mutable -> preparation_coordinator::claim_result {
      if (std::exchange(claimed, true)) return preparation_exhausted{};
      return preparation_coordinator::job{[&, capture = std::move(capture)] {
        ran = true;
        return std::make_unique<tagged>(2);
      }};
    },
    {},
    {},
    preparation_coordinator::execution_domain::native_reader);
  f.coordinator.before_submission_for_testing([&] {
    if (++submissions == 2) submitted.set_value();
  });
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  REQUIRE(submitted.get_future().wait_for(1s) == std::future_status::ready);
  f.coordinator.request_stop(stop_reason::normal_eos);
  blocked->release();
  f.finish();
  CHECK_FALSE(ran);
  CHECK(destroyed == worker);
  CHECK(queued->closing_thread == worker);
  CHECK(first->count() == 0);
  CHECK(queued->count() == 0);
  auto reused = executor.reserve(2);
  CHECK(reused.size() == 2);
}

TEST_CASE("Reader stop intent distinguishes cancellation failure and normal completion",
          "[scan_preparation][reader_stop]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::metadata_worker, domain::query_owner, domain::native_reader);
  // User cancellation, reported failure, explicit EOS, GPU/LIMIT, predicate, cleanup intent.
  auto trigger = GENERATE(0, 1, 2, 3, 4, 5);
  native_reader_executor native(1, 1);
  fixture f(1, 1, 1ms, std::nullopt, &native);
  auto out      = std::make_shared<sink>();
  auto blocked  = f.make_hold();
  auto terminal = f.completion.get_awaitable().share();
  std::promise<preparation_stop_token> started;
  auto token_future = started.get_future();
  std::promise<std::pair<std::optional<stop_reason>, bool>> notified;
  auto notification = notified.get_future();
  std::atomic<bool> interrupted{false};
  f.coordinator.set_interrupt_check([&] { return interrupted.load(); });
  auto guard = f.shutdown_guard();
  f.start([&] {
    f.source(
      std::make_shared<accumulator>(1),
      out,
      [&, claimed = false]() mutable -> preparation_coordinator::claim_result {
        if (std::exchange(claimed, true)) return preparation_exhausted{};
        return preparation_coordinator::job{[&](preparation_stop_token stop) {
          std::stop_callback observer(stop.token(), [&] {
            notified.set_value({stop.reason(), terminal.wait_for(0s) == std::future_status::ready});
          });
          started.set_value(stop);
          blocked->wait();
          return std::make_unique<tagged>(1);
        }};
      },
      {},
      {},
      execution);
  });
  REQUIRE(token_future.wait_for(1s) == std::future_status::ready);
  auto token = token_future.get();
  CHECK_FALSE(token.stop_requested());
  CHECK_FALSE(token.reason());
  switch (trigger) {
    case 0: f.coordinator.request_stop(stop_reason::user_cancel); break;
    case 1: f.completion.report_error("original execution failure"); break;
    case 2: f.coordinator.request_stop(stop_reason::normal_eos); break;
    case 3: f.completion.mark_completed(); break;
    case 4:
      interrupted = true;
      f.coordinator.wake_for_testing();
      break;
    case 5: f.coordinator.request_stop(stop_reason::failure_induced); break;
  }
  REQUIRE(notification.wait_for(1s) == std::future_status::ready);
  auto [reason, terminal_ready] = notification.get();
  auto expected                 = trigger == 0 || trigger == 4   ? stop_reason::user_cancel
                                  : trigger == 1 || trigger == 5 ? stop_reason::failure_induced
                                                                 : stop_reason::normal_eos;
  CHECK(reason == expected);
  CHECK(terminal_ready == (trigger == 0 || trigger == 1 || trigger == 4));
  CHECK(token.stop_requested());
  CHECK(token.reason() == expected);
  // A child query clearing its own interrupt flag cannot clear the attempt's intent.
  interrupted = false;
  f.coordinator.request_stop(stop_reason::normal_eos);
  CHECK(token.reason() == expected);
  CHECK(out->close_calls == 0);
  // Cleanup intent can precede the runner reporting the actual exception.
  if (trigger == 5) f.completion.report_error("original execution failure");
  blocked->release();
  f.finish();
  CHECK(out->count() == 0);
  CHECK(out->close_calls == 1);
  if (trigger == 0 || trigger == 4)
    CHECK_THROWS_AS(terminal.get(), duckdb::InterruptException);
  else if (trigger == 1 || trigger == 5)
    CHECK_THROWS_WITH(terminal.get(), "original execution failure");
  else {
    f.completion.mark_completed();
    CHECK_NOTHROW(terminal.get());
  }
}

TEST_CASE("Stop intent upgrades without replaying callbacks or replacing the terminal error",
          "[scan_preparation][reader_stop]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::query_owner, domain::native_reader);
  native_reader_executor native(1, 1);
  fixture f(1, 1, 1ms, std::nullopt, &native);
  auto out      = std::make_shared<sink>();
  auto blocked  = f.make_hold();
  auto terminal = f.completion.get_awaitable();
  std::promise<preparation_stop_token> started;
  auto token_future = started.get_future();
  std::atomic<size_t> notifications{0};
  auto guard = f.shutdown_guard();
  f.start([&] {
    f.source(
      std::make_shared<accumulator>(1),
      out,
      [&, claimed = false]() mutable -> preparation_coordinator::claim_result {
        if (std::exchange(claimed, true)) return preparation_exhausted{};
        return preparation_coordinator::job{[&](preparation_stop_token stop) {
          std::stop_callback observer(stop.token(), [&] { ++notifications; });
          started.set_value(stop);
          blocked->wait();
          return std::make_unique<tagged>(1);
        }};
      },
      {},
      {},
      execution);
  });
  REQUIRE(token_future.wait_for(1s) == std::future_status::ready);
  auto token = token_future.get();
  f.coordinator.request_stop(stop_reason::normal_eos);
  CHECK(token.reason() == stop_reason::normal_eos);
  f.completion.report_error("first failure");
  CHECK(token.reason() == stop_reason::failure_induced);
  f.coordinator.request_stop(stop_reason::user_cancel);
  CHECK(token.reason() == stop_reason::user_cancel);
  f.coordinator.request_stop(stop_reason::failure_induced);
  f.coordinator.request_stop(stop_reason::normal_eos);
  CHECK(token.reason() == stop_reason::user_cancel);
  CHECK(notifications == 1);
  blocked->release();
  f.finish();
  CHECK_THROWS_WITH(terminal.get(), "first failure");
  CHECK(token.reason() == stop_reason::user_cancel);
  CHECK(out->close_calls == 1);
}

TEST_CASE("Natural reader exhaustion retains a normal stop view after coordinator destruction",
          "[scan_preparation][reader_stop]")
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::query_owner, domain::native_reader);
  preparation_stop_token retained;
  {
    native_reader_executor native(1, 1);
    fixture f(1, 1, 1ms, std::nullopt, &native);
    auto out   = std::make_shared<sink>();
    auto guard = f.shutdown_guard();
    f.start([&] {
      f.source(
        std::make_shared<accumulator>(1),
        out,
        [&, claimed = false]() mutable -> preparation_coordinator::claim_result {
          if (std::exchange(claimed, true)) return preparation_exhausted{};
          return preparation_coordinator::job{[&](preparation_stop_token stop) {
            retained = stop;
            return std::unique_ptr<tagged>{};
          }};
        },
        {},
        {},
        execution);
    });
    // The fixture coalescer emits its final empty batch; it must not keep the reader open.
    REQUIRE(out->await(1));
    out->consume();
    REQUIRE(out->await_closed());
    f.finish();
    CHECK_FALSE(f.completion.has_error());
  }
  CHECK(retained.stop_requested());
  CHECK(retained.reason() == stop_reason::normal_eos);
}

TEST_CASE("One scan coordinates dependent lanes without sharing their output credits",
          "[scan_preparation][preparation_group]")
{
  using domain       = preparation_coordinator::execution_domain;
  auto reader_domain = GENERATE(domain::query_owner, domain::native_reader);
  native_reader_executor native(1, 1);
  fixture f(1, 1, 1ms, std::nullopt, &native);
  auto out = std::make_shared<sink>();
  std::atomic<bool> discovered{false}, finish_reader{false};
  preparation_wakeup wake;
  auto unit = f.coordinator.admit_unit(
    {10, 0}, required_input_set{}.set(static_cast<size_t>(required_input::checkpoint_iteration)));
  std::promise<void> file_prepared, metadata_closed;
  auto prepared = file_prepared.get_future(), closed = metadata_closed.get_future();
  std::thread::id reader, closer;
  auto guard = f.shutdown_guard();
  f.start([&] {
    auto metadata        = make_source(domain::metadata_worker);
    metadata.destination = out->connector;
    metadata.try_claim   = [&, claimed = false]() mutable -> preparation_coordinator::claim_result {
      if (!discovered) return preparation_pending{};
      if (std::exchange(claimed, true)) return preparation_exhausted{};
      return preparation_coordinator::job{[&] {
                                            file_prepared.set_value();
                                            return std::make_unique<tagged>(20);
                                          },
                                          unit};
    };
    metadata.close       = [&] { metadata_closed.set_value(); };
    auto provider        = make_source(reader_domain);
    provider.destination = out->connector;
    provider.bind_wakeup = [&](preparation_wakeup value) { wake = std::move(value); };
    provider.try_claim   = [&, next = 0]() mutable -> preparation_coordinator::claim_result {
      if (next == 2)
        return finish_reader ? preparation_coordinator::claim_result{preparation_exhausted{}}
                               : preparation_coordinator::claim_result{preparation_pending{}};
      return preparation_coordinator::job{[&, step = next++] {
        reader = std::this_thread::get_id();
        if (step == 0)
          discovered = true;
        else
          unit->complete_input(checkpoint_input{17});
        wake();
        return std::make_unique<tagged>(10 + step);
      }};
    };
    provider.close = [&] { closer = std::this_thread::get_id(); };
    f.outputs      = {out, out};
    std::vector<preparation_coordinator::source> lanes;
    lanes.push_back(std::move(metadata));
    lanes.push_back(std::move(provider));
    f.coordinator.add_sources(std::move(lanes), [out] { out->close(); });
  });
  REQUIRE(out->await(1));
  REQUIRE(prepared.wait_for(1s) == std::future_status::ready);
  CHECK(out->count() == 1);  // The file must wait for the provider's dependency.
  CHECK(unit->record().state == unit_state::pending);
  out->consume();  // Returns only the provider's credit, allowing its next pull.
  REQUIRE(out->await(3));
  out->consume();
  out->consume();
  REQUIRE(closed.wait_for(1s) == std::future_status::ready);
  CHECK_FALSE(out->connector->is_discovery_complete());
  CHECK(out->close_calls == 0);
  finish_reader = true;
  wake();
  f.finish();
  CHECK_FALSE(f.completion.has_error());
  CHECK(out->connector->is_closed());
  CHECK(out->close_calls == 1);
  CHECK(reader == closer);
  CHECK((reader == f.coordinator.snapshot().owner) == (reader_domain == domain::query_owner));
  auto published = out->published;
  std::ranges::sort(published);
  CHECK(published == std::vector<int>{10, 11, 20});
}

TEST_CASE("Grouped lanes close their shared output after cancellation and failures",
          "[scan_preparation][preparation_group]")
{
  enum class ending { cancel, read_failure, close_failure };
  auto mode = GENERATE(ending::cancel, ending::read_failure, ending::close_failure);
  fixture f(1, 1);
  auto terminal = f.completion.get_awaitable();
  auto out      = std::make_shared<sink>();
  auto blocked  = f.make_hold();
  std::promise<void> provider_closed;
  auto closed = provider_closed.get_future();
  std::atomic<size_t> metadata_closes{0};
  auto guard = f.shutdown_guard();
  f.start([&] {
    auto metadata        = make_source(preparation_coordinator::execution_domain::metadata_worker);
    metadata.destination = out->connector;
    metadata.try_claim   = single_job([&]() -> std::unique_ptr<sirius::op::scan::scan_info> {
      blocked->wait();
      if (mode == ending::read_failure) throw std::runtime_error("metadata failed");
      return std::make_unique<tagged>(1);
    });
    metadata.close       = [&] { ++metadata_closes; };
    auto coalescer       = std::make_shared<accumulator>(1);
    coalescer->produced  = true;  // This lane supplies no rows.
    auto provider = make_source(preparation_coordinator::execution_domain::query_owner, coalescer);
    provider.destination = out->connector;
    provider.try_claim   = [] {
      return preparation_coordinator::claim_result{preparation_exhausted{}};
    };
    provider.close = [&] {
      provider_closed.set_value();
      if (mode == ending::close_failure) throw std::runtime_error("provider close failed");
    };
    f.outputs = {out, out};
    std::vector<preparation_coordinator::source> lanes;
    lanes.push_back(std::move(metadata));
    lanes.push_back(std::move(provider));
    f.coordinator.add_sources(std::move(lanes), [out] { out->close(); });
  });
  REQUIRE(blocked->await());
  REQUIRE(closed.wait_for(1s) == std::future_status::ready);
  CHECK_FALSE(out->connector->is_discovery_complete());
  if (mode == ending::cancel) f.coordinator.request_stop(stop_reason::user_cancel);
  if (mode == ending::close_failure) REQUIRE(terminal.wait_for(1s) == std::future_status::ready);
  blocked->release();
  f.finish();
  CHECK(f.completion.has_error());
  CHECK(out->count() == 0);
  CHECK(out->close_calls == 1);
  CHECK(metadata_closes == 1);
  CHECK(out->connector->is_closed());
  if (mode != ending::cancel)
    CHECK(f.completion.failure().detail ==
          (mode == ending::read_failure ? "metadata failed" : "provider close failed"));
}

TEST_CASE("Rejected native group admission closes output without opening readers",
          "[scan_preparation][preparation_group]")
{
  native_reader_executor native(1, 1);
  fixture f(1, 1, 1ms, std::nullopt, &native);
  auto out     = std::make_shared<sink>();
  size_t calls = 0, closes = 0;
  std::vector<preparation_coordinator::source> lanes;
  for (int i = 0; i < 2; ++i) {
    auto source        = make_source(preparation_coordinator::execution_domain::native_reader);
    source.destination = out->connector;
    source.try_claim   = single_job([&] {
      ++calls;
      return std::make_unique<tagged>(1);
    });
    source.close       = [&] { ++closes; };
    lanes.push_back(std::move(source));
  }
  f.coordinator.add_sources(std::move(lanes), [out] { out->close(); });
  CHECK_THROWS_WITH(f.coordinator.arm(), "native reader capacity unavailable for complete attempt");
  f.coordinator.drain();
  CHECK(calls == 0);
  CHECK(closes == 0);
  CHECK(out->close_calls == 1);
  CHECK(out->connector->is_closed());
  auto reused = native.reserve(1);
  CHECK(reused.size() == 1);
}
