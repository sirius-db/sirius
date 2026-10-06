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
#include "scan_manager/preparation_coordinator.hpp"
#include "scan_manager/preparation_test_support.hpp"

#include <array>
#include <condition_variable>
#include <future>
#include <mutex>
using namespace sirius::scan_manager;
using namespace std::chrono_literals;
namespace {
struct hold {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false, released = false;
  void wait()
  {
    std::unique_lock l(mutex);
    entered = true;
    cv.notify_all();
    if (!cv.wait_for(l, 5s, [&] { return released; }))
      throw std::runtime_error("test hold expired");
  }
  bool await()
  {
    std::unique_lock l(mutex);
    return cv.wait_for(l, 1s, [&] { return entered; });
  }
  void release()
  {
    std::lock_guard l(mutex);
    released = true;
    cv.notify_all();
  }
};
struct tagged : sirius::op::scan::scan_info {
  int id;
  size_t rows;
  tagged(int i, size_t n = 1) : id(i), rows(n) {}
};
struct accumulator : sirius::op::scan::batch_coalescer {
  size_t cap, retained = 0;
  int last      = 0;
  bool produced = false;
  std::optional<clock::time_point> fake_now;
  std::function<clock::time_point()> fake_clock;
  std::function<void()> on_retained;
  explicit accumulator(size_t cap = 1000) : cap(cap) {}
  cursor_step advance(sirius::op::scan::scan_info& input, size_t& cursor, size_t quantum) override
  {
    auto& data   = dynamic_cast<tagged&>(input);
    size_t steps = 0;
    while (cursor < data.rows && steps++ < quantum) {
      note_retained(fake_clock ? fake_clock() : fake_now.value_or(clock::now()));
      if (on_retained) on_retained();
      ++retained;
      ++cursor;
      last = data.id;
      if (retained >= cap) return {partial_emit(), cursor == data.rows};
    }
    return {nullptr, cursor == data.rows};
  }
  std::unique_ptr<sirius::op::scan::scan_info> partial_emit() override
  {
    if (!retained) return {};
    auto result = std::make_unique<tagged>(last, retained);
    retained    = 0;
    produced    = true;
    clear_retained();
    return result;
  }
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> push(
    std::unique_ptr<sirius::op::scan::scan_info> input) override
  {
    size_t cursor = 0;
    std::vector<std::unique_ptr<sirius::op::scan::scan_info>> out;
    for (;;) {
      auto s = advance(*input, cursor, 1);
      if (s.batch) out.push_back(std::move(s.batch));
      if (s.finished) return out;
    }
  }
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> flush() override
  {
    std::vector<std::unique_ptr<sirius::op::scan::scan_info>> out;
    if (auto b = partial_emit()) out.push_back(std::move(b));
    if (!produced) {
      out.push_back(std::make_unique<tagged>(-1, 0));
      produced = true;
    }
    return out;
  }
};
struct sink {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<int> published;
  std::function<void()> consumed;
  std::atomic<bool> closed{false};
  std::atomic<size_t> close_calls{0};
  size_t count()
  {
    std::lock_guard l(mutex);
    return published.size();
  }
  bool await(size_t n)
  {
    std::unique_lock l(mutex);
    return cv.wait_for(l, 1s, [&] { return published.size() >= n; });
  }
  bool await_closed()
  {
    std::unique_lock l(mutex);
    return cv.wait_for(l, 1s, [&] { return closed.load(); });
  }
  void consume()
  {
    if (consumed) consumed();
  }
};
struct fixture {
  sirius::exec::static_thread_pool pool;
  sirius::exec::scoped_dispatcher dispatcher;
  sirius::pipeline::completion_handler completion;
  preparation_coordinator coordinator;
  std::vector<std::shared_ptr<hold>> holds;
  explicit fixture(size_t workers = 2, size_t slots = 4)
    : pool(static_cast<int>(workers)),
      dispatcher(pool),
      coordinator(
        completion, dispatcher, preparation_options{workers, slots, slots, workers, 1, 20ms})
  {
  }
  std::thread owner;
  std::exception_ptr owner_error;
  void start()
  {
    owner = std::thread([this] {
      try {
        coordinator.arm();
        coordinator.run_on_query_thread();
        coordinator.drain();
      } catch (...) {
        owner_error = std::current_exception();
        completion.report_error(owner_error);
        coordinator.drain();
      }
    });
  }
  void finish()
  {
    if (owner.joinable()) owner.join();
  }
  void stop()
  {
    for (auto& h : holds)
      h->release();
    coordinator.request_stop(stop_reason::normal_eos);
    finish();
  }
  std::shared_ptr<fixture> shutdown_guard()
  {
    return std::shared_ptr<fixture>(this, [](fixture* f) { f->stop(); });
  }
  ~fixture()
  {
    for (auto& h : holds)
      h->release();
    coordinator.request_stop(stop_reason::normal_eos);
    finish();
    if (coordinator.snapshot().phase == preparation_coordinator::lifecycle::constructed)
      coordinator.drain();
  }
  std::shared_ptr<hold> make_hold()
  {
    auto h = std::make_shared<hold>();
    holds.push_back(h);
    return h;
  }
  void source(std::shared_ptr<accumulator> a,
              std::shared_ptr<sink> out,
              std::function<std::optional<preparation_coordinator::job>()> claim,
              std::shared_ptr<hold> construction = {})
  {
    coordinator.add_source(
      {a,
       std::move(claim),
       [construction](std::unique_ptr<sirius::op::scan::scan_info> batch) {
         if (construction) construction->wait();
         auto input = std::make_unique<sirius::op::scan::scan_operator_input>(std::move(batch));
         return preparation_coordinator::publication{std::move(input), 0};
       },
       [out](preparation_coordinator::publication p) {
         auto& input = dynamic_cast<sirius::op::scan::scan_operator_input&>(*p.input);
         auto metadata =
           std::get<std::shared_ptr<sirius::op::scan::scan_info>>(input.materialization_info);
         auto& tag = dynamic_cast<tagged&>(*metadata);
         std::lock_guard l(out->mutex);
         out->published.push_back(tag.id);
         out->cv.notify_all();
       },
       [out] {
         std::lock_guard l(out->mutex);
         ++out->close_calls;
         out->closed = true;
         out->cv.notify_all();
       },
       [out](std::function<void()> consume) { out->consumed = std::move(consume); }});
  }
};
}  // namespace
TEST_CASE("Ready partial batches publish while every preparation worker is blocked",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto a   = std::make_shared<accumulator>();
  auto out = std::make_shared<sink>();
  auto h1 = f.make_hold(), h2 = f.make_hold();
  int next = 0;
  f.source(a, out, [&]() -> std::optional<preparation_coordinator::job> {
    int id = next++;
    if (id >= 3) return {};
    return preparation_coordinator::job{[=] {
                                          if (id == 1) h1->wait();
                                          if (id == 2) h2->wait();
                                          return std::make_unique<tagged>(id);
                                        },
                                        {}};
  });
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
  f.source(
    std::make_shared<accumulator>(), out, [&]() -> std::optional<preparation_coordinator::job> {
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
  int next = 0;
  f.source(
    std::make_shared<accumulator>(), out, [&]() -> std::optional<preparation_coordinator::job> {
      int id = next++;
      if (id >= 2) return {};
      return preparation_coordinator::job{[=] {
                                            if (id == 1) h->wait();
                                            return std::make_unique<tagged>(id);
                                          },
                                          {}};
    });
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
}
TEST_CASE("Publication is refused after cancellation wins during input construction",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out          = std::make_shared<sink>();
  auto construction = f.make_hold();
  int next          = 0;
  auto awaitable    = f.completion.get_awaitable();
  f.source(
    std::make_shared<accumulator>(1),
    out,
    [&]() -> std::optional<preparation_coordinator::job> {
      if (next++) return {};
      return preparation_coordinator::job{[] { return std::make_unique<tagged>(7); }, {}};
    },
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
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}
TEST_CASE("Unresolved typed input forbids scan input construction until the whole unit is ready",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  required_input_set required;
  required.set(2);
  required.set(3);
  auto unit = f.coordinator.admit_unit({3, 1}, required);
  REQUIRE(unit);
  int next = 0;
  std::promise<void> returned;
  auto stage = returned.get_future();
  f.source(
    std::make_shared<accumulator>(1), out, [&]() -> std::optional<preparation_coordinator::job> {
      if (next++) return {};
      return preparation_coordinator::job{[&] {
                                            unit->complete_input(checkpoint_input{7});
                                            returned.set_value();
                                            return std::make_unique<tagged>(1);
                                          },
                                          unit};
    });
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(stage.wait_for(1s) == std::future_status::ready);
  CHECK(f.coordinator.unit_state_snapshot({3, 1})->state == unit_state::pending);
  CHECK(out->count() == 0);
  REQUIRE(unit->complete_input(
    delete_set_input{std::make_shared<sirius::op::scan::iceberg_delete_set const>("file")}));
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
  f.source(
    std::make_shared<accumulator>(1), out, [&]() -> std::optional<preparation_coordinator::job> {
      int id = next++;
      if (id >= 1000) return {};
      return preparation_coordinator::job{[id] { return std::make_unique<tagged>(id, 1000); }, {}};
    });
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(out->await(1));
  std::this_thread::sleep_for(25ms);
  CHECK(out->count() == 1);
  CHECK(next.load() == 1);
  auto stats = f.coordinator.snapshot();
  CHECK(stats.output_peak <= 1);
  CHECK(stats.results_peak <= 1);
  CHECK(stats.jobs_peak <= 1);
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
  int next = 0;
  std::string expected_error;
  SECTION("claim throws")
  {
    expected_error = "claim injected";
    f.source(
      std::make_shared<accumulator>(), out, []() -> std::optional<preparation_coordinator::job> {
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
    f.source(
      std::make_shared<accumulator>(), out, [&]() -> std::optional<preparation_coordinator::job> {
        int id = next++;
        if (id >= 2) return {};
        return preparation_coordinator::job{[read, id] {
                                              read->wait();
                                              return std::make_unique<tagged>(id);
                                            },
                                            {}};
      });
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
  int next = 0;
  f.source(
    std::make_shared<accumulator>(), out, [&]() -> std::optional<preparation_coordinator::job> {
      if (next++) return {};
      return preparation_coordinator::job{[=] {
                                            h->wait();
                                            return std::make_unique<tagged>(1);
                                          },
                                          {}};
    });
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
  int next  = 0;
  auto done = f.completion.get_awaitable();
  f.source(
    std::make_shared<accumulator>(), out, [&]() -> std::optional<preparation_coordinator::job> {
      if (next++) return {};
      return preparation_coordinator::job{[=] {
                                            h->wait();
                                            return std::make_unique<tagged>(1);
                                          },
                                          {}};
    });
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
TEST_CASE("A deterministic clock drives the coordinator at the original absolute deadline",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out      = std::make_shared<sink>();
  auto a        = std::make_shared<accumulator>();
  auto h        = f.make_hold();
  int next      = 0;
  auto tick     = std::make_shared<std::atomic<int>>(0);
  auto origin   = accumulator::clock::now();
  auto clock    = [tick, origin] { return origin + std::chrono::milliseconds(tick->load()); };
  a->fake_clock = clock;
  f.coordinator.clock_for_testing(clock);
  auto retained  = std::make_shared<std::promise<void>>();
  auto reached   = retained->get_future();
  auto notified  = std::make_shared<std::atomic<bool>>(false);
  a->on_retained = [retained, notified] {
    if (!notified->exchange(true)) retained->set_value();
  };
  f.source(a, out, [&]() -> std::optional<preparation_coordinator::job> {
    int id = next++;
    if (id >= 2) return {};
    return preparation_coordinator::job{[=] {
                                          if (id == 1) h->wait();
                                          return std::make_unique<tagged>(id);
                                        },
                                        {}};
  });
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(reached.wait_for(1s) == std::future_status::ready);
  tick->store(19);
  f.coordinator.wake_for_testing();
  std::this_thread::sleep_for(5ms);
  CHECK(out->count() == 0);
  tick->store(20);
  f.coordinator.wake_for_testing();
  REQUIRE(out->await(1));
  CHECK(f.coordinator.snapshot().max_residence == 20ms);
  CHECK(f.coordinator.snapshot().max_deadline_lateness == 0ms);
  tick->store(100);
  f.coordinator.wake_for_testing();
  std::this_thread::sleep_for(5ms);
  CHECK(out->count() == 1);
}
TEST_CASE("Cancel racing with a deadline never publishes after stop returns",
          "[scan_preparation][coordinator]")
{
  for (int iteration = 0; iteration < 12; ++iteration) {
    fixture f;
    auto out = std::make_shared<sink>();
    auto h   = f.make_hold();
    int next = 0;
    f.source(
      std::make_shared<accumulator>(), out, [&]() -> std::optional<preparation_coordinator::job> {
        int id = next++;
        if (id >= 2) return {};
        return preparation_coordinator::job{[=] {
                                              if (id == 1) h->wait();
                                              return std::make_unique<tagged>(id);
                                            },
                                            {}};
      });
    auto query_guard = f.shutdown_guard();
    auto done        = f.completion.get_awaitable();
    f.start();
    REQUIRE(h->await());
    std::this_thread::sleep_for(std::chrono::milliseconds(15 + iteration));
    f.coordinator.request_stop(stop_reason::user_cancel);
    auto count = out->count();
    h->release();
    f.finish();
    CHECK(out->count() == count);
    REQUIRE(done.wait_for(1s) == std::future_status::ready);
    CHECK_THROWS_AS(done.get(), duckdb::InterruptException);
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
  CHECK(f.coordinator.unit_state_snapshot({12, 1})->state == unit_state::failed);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}
TEST_CASE("GPU completion cannot hide an accepted preparation failure",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out     = std::make_shared<sink>();
  auto read    = f.make_hold();
  bool claimed = false;
  f.source(
    std::make_shared<accumulator>(), out, [&]() -> std::optional<preparation_coordinator::job> {
      if (std::exchange(claimed, true)) return {};
      return preparation_coordinator::job{
        [read]() -> std::unique_ptr<sirius::op::scan::scan_info> {
          read->wait();
          throw std::runtime_error("accepted read failed after GPU done");
        },
        {}};
    });
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

TEST_CASE("Later scans cannot consume the last output credit needed by the first scan",
          "[scan_preparation][coordinator]")
{
  fixture f(2, 2);
  auto first = std::make_shared<sink>(), later = std::make_shared<sink>();
  auto blocked       = f.make_hold();
  bool first_claimed = false, later_claimed = false;
  f.source(
    std::make_shared<accumulator>(1), first, [&]() -> std::optional<preparation_coordinator::job> {
      if (std::exchange(first_claimed, true)) return {};
      return preparation_coordinator::job{[blocked] {
                                            blocked->wait();
                                            return std::make_unique<tagged>(1);
                                          },
                                          {}};
    });
  f.source(
    std::make_shared<accumulator>(1), later, [&]() -> std::optional<preparation_coordinator::job> {
      if (std::exchange(later_claimed, true)) return {};
      return preparation_coordinator::job{[] { return std::make_unique<tagged>(2, 100); }, {}};
    });
  auto query_guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  REQUIRE(later->await(1));
  // No later pipeline consumption while execution still needs the first scan.
  std::this_thread::sleep_for(20ms);
  CHECK(later->count() == 1);
  blocked->release();
  REQUIRE(first->await(1));
  first->consume();
  CHECK(f.coordinator.snapshot().output_peak <= 2);
}

TEST_CASE(
  "Query driver is armed before consumers and runs on its owner without waiting for success",
  "[scan_preparation][coordinator]")
{
  sirius::exec::static_thread_pool pool(1);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  sirius::pipeline::completion_handler completion;
  preparation_coordinator c(completion, dispatcher, preparation_config{}.resolve(1));
  auto done = completion.get_awaitable();
  CHECK(c.snapshot().phase == preparation_coordinator::lifecycle::constructed);
  c.arm();
  CHECK(c.snapshot().phase == preparation_coordinator::lifecycle::armed);
  c.run_on_query_thread();
  CHECK(c.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  CHECK(c.snapshot().owner == std::this_thread::get_id());
  CHECK(c.snapshot().runner == std::this_thread::get_id());
  CHECK(done.wait_for(0ms) == std::future_status::timeout);
  completion.mark_completed();
  REQUIRE(done.wait_for(100ms) == std::future_status::ready);
  done.get();
  CHECK_THROWS_AS(c.run_on_query_thread(), std::logic_error);
  c.drain();
  c.drain();
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
  int next   = 0;
  f.source(std::make_shared<retaining_coalescer>(),
           out,
           [&]() -> std::optional<preparation_coordinator::job> {
             int id = next++;
             if (id >= 2) return {};
             return preparation_coordinator::job{[=] {
                                                   if (id == 1) later->wait();
                                                   return std::make_unique<tagged>(id,
                                                                                   id == 0 ? 2 : 0);
                                                 },
                                                 {}};
           });
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
  auto out       = std::make_shared<sink>();
  auto coalescer = std::make_shared<accumulator>(1);
  bool claimed = false, fail_construct = false, fail_close = false, normal_close_failure = false;
  size_t publications        = 0;
  std::string expected_error = "push injected";
  SECTION("construction throws")
  {
    fail_construct = true;
    expected_error = "construct injected";
  }
  SECTION("visible queue push throws") {}
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
  source.claim     = [&]() -> std::optional<preparation_coordinator::job> {
    if (std::exchange(claimed, true)) return {};
    return preparation_coordinator::job{[] { return std::make_unique<tagged>(1); }, {}};
  };
  source.construct = [&](std::unique_ptr<sirius::op::scan::scan_info> batch) {
    if (fail_construct) throw std::runtime_error("construct injected");
    return preparation_coordinator::publication{
      std::make_unique<sirius::op::scan::scan_operator_input>(std::move(batch)), 0};
  };
  source.publish = [&](preparation_coordinator::publication) {
    ++publications;
    if (!normal_close_failure) throw std::runtime_error(expected_error);
  };
  source.close = [out, fail_close] {
    std::lock_guard lock(out->mutex);
    ++out->close_calls;
    if (fail_close) throw std::runtime_error("secondary close failure");
    out->closed = true;
    out->cv.notify_all();
  };
  source.bind_consumption = [out](std::function<void()> callback) {
    out->consumed = std::move(callback);
  };
  f.coordinator.add_source(std::move(source));
  auto done  = f.completion.get_awaitable();
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(done.wait_for(1s) == std::future_status::ready);
  CHECK_THROWS_WITH(done.get(), expected_error);
  f.finish();
  f.coordinator.drain();
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
  CHECK(out->count() == 0);
  CHECK(publications == (fail_construct ? 0 : 1));
  CHECK(out->close_calls == 1);
}

TEST_CASE("An accepted failure callback drains before GPU completion can publish success",
          "[scan_preparation][coordinator][callback_drain]")
{
  fixture f;
  auto out      = std::make_shared<sink>();
  auto callback = f.make_hold();
  required_input_set required;
  required.set(static_cast<size_t>(required_input::delete_set));
  auto unit    = f.coordinator.admit_unit({91, 1}, required);
  bool claimed = false;
  std::promise<void> metadata;
  auto ready = metadata.get_future();
  f.source(
    std::make_shared<accumulator>(1), out, [&]() -> std::optional<preparation_coordinator::job> {
      if (std::exchange(claimed, true)) return {};
      return preparation_coordinator::job{[&] {
                                            metadata.set_value();
                                            return std::make_unique<tagged>(1);
                                          },
                                          unit};
    });
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
  int next = 0;
  f.source(
    std::make_shared<accumulator>(1), out, [&]() -> std::optional<preparation_coordinator::job> {
      int id = next++;
      if (id >= 2) return {};
      return preparation_coordinator::job{[=] {
                                            (id == 0 ? first : last)->wait();
                                            return std::make_unique<tagged>(id);
                                          },
                                          {}};
    });
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
TEST_CASE("A single result and output slot preserves progress across multiple scans",
          "[scan_preparation][coordinator]")
{
  fixture f(1, 1);
  auto first = std::make_shared<sink>(), last = std::make_shared<sink>();
  bool first_claimed = false, last_claimed = false;
  f.source(
    std::make_shared<accumulator>(1), first, [&]() -> std::optional<preparation_coordinator::job> {
      if (std::exchange(first_claimed, true)) return {};
      return preparation_coordinator::job{[] { return std::make_unique<tagged>(1); }, {}};
    });
  f.source(
    std::make_shared<accumulator>(1), last, [&]() -> std::optional<preparation_coordinator::job> {
      if (std::exchange(last_claimed, true)) return {};
      return preparation_coordinator::job{[] { return std::make_unique<tagged>(2); }, {}};
    });
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(first->await(1));
  CHECK(last->count() == 0);
  first->consume();
  REQUIRE(first->await_closed());
  REQUIRE(last->await(1));
  last->consume();
  REQUIRE(last->await_closed());
  f.finish();
  CHECK(f.coordinator.snapshot().jobs_peak == 1);
  CHECK(f.coordinator.snapshot().results_peak <= 1);
  CHECK(f.coordinator.snapshot().output_peak == 1);
}

TEST_CASE("Physical external users delay quiescence after success or cancellation",
          "[scan_preparation][coordinator]")
{
  fixture f;
  auto out = std::make_shared<sink>();
  f.source(std::make_shared<accumulator>(), out, [] {
    return std::optional<preparation_coordinator::job>{};
  });
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
  fixture f;
  auto out = std::make_shared<sink>();
  required_input_set required;
  required.set(static_cast<size_t>(required_input::delete_set));
  auto unit    = f.coordinator.admit_unit({72, 1}, required);
  bool claimed = false;
  f.source(
    std::make_shared<accumulator>(1), out, [&]() -> std::optional<preparation_coordinator::job> {
      if (std::exchange(claimed, true)) return {};
      return preparation_coordinator::job{[] { return std::make_unique<tagged>(1); }, unit};
    });
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
  auto cancel = GENERATE(false, true);
  fixture f;
  auto out     = std::make_shared<sink>();
  auto blocked = f.make_hold();
  sirius::scan_manager::test::test_reservation_provider provider;
  provider.grant_bytes = 48;
  preparation_ledger ledger(provider);
  std::array envelope{scan_envelope{{{16, 0, 0, 16}, {16, 0, 0, 16}}, 0, 0, true}};
  REQUIRE(ledger.admit(envelope, memory_space_id(cucascade::memory::Tier::HOST, 0)).n_permits == 1);
  struct input : tagged {
    std::shared_ptr<sirius::op::scan::iceberg_delete_set const> deletes;
    input(int id, decltype(deletes) set) : tagged(id), deletes(std::move(set)) {}
  };
  int next = 0;
  w_permit pending;
  bool registered = false;
  preparation_coordinator::source source;
  source.coalescer = std::make_shared<accumulator>(1);
  source.can_claim = [&] {
    if (next >= 2) return true;
    if (!registered) {
      ledger.register_unit({1, next + 1u}, 16);
      registered = true;
    }
    if (!pending) pending = ledger.acquire_permit({1, next + 1u});
    return bool(pending);
  };
  source.claim = [&]() -> std::optional<preparation_coordinator::job> {
    if (next >= 2) return {};
    auto id     = next++;
    registered  = false;
    auto permit = std::make_shared<w_permit>(std::move(pending));
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
        return std::make_unique<input>(id, std::move(set));
      },
      {}};
  };
  source.construct = [](std::unique_ptr<sirius::op::scan::scan_info> batch) {
    return preparation_coordinator::publication{
      std::make_unique<sirius::op::scan::scan_operator_input>(std::move(batch)), 0};
  };
  source.publish = [out](preparation_coordinator::publication) {
    std::lock_guard lock(out->mutex);
    out->published.push_back(1);
    out->cv.notify_all();
  };
  source.close = [out] {
    std::lock_guard lock(out->mutex);
    out->closed = true;
    out->cv.notify_all();
  };
  source.bind_consumption = [out](std::function<void()> callback) {
    out->consumed = std::move(callback);
  };
  f.coordinator.add_source(std::move(source));
  auto guard = f.shutdown_guard();
  f.start();
  REQUIRE(blocked->await());
  CHECK(ledger.permits_in_flight() == 1);
  CHECK(out->count() == 0);
  CHECK_FALSE(out->closed);
  if (cancel) f.coordinator.request_stop(stop_reason::user_cancel);
  blocked->release();
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
