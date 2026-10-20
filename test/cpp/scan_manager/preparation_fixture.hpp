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

#pragma once

#include "scan_manager/preparation_coordinator.hpp"
#include "scan_manager/preparation_test_support.hpp"

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>

namespace sirius::test::preparation {
using namespace sirius::scan_manager;
using namespace std::chrono_literals;
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
  size_t preparation_bytes() const noexcept override { return rows; }
};
struct accumulator : sirius::op::scan::batch_coalescer {
  size_t cap, retained = 0;
  int last      = 0;
  bool produced = false;
  std::optional<clock::time_point> fake_now;
  std::function<clock::time_point()> fake_clock;
  std::function<void()> on_retained;
  explicit accumulator(size_t cap = 1000) : cap(cap) {}
  size_t target_batch_bytes() const noexcept override { return cap; }
  size_t retained_bytes() const noexcept override { return retained; }
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
  std::shared_ptr<split_connector> connector = std::make_shared<split_connector>();
  std::deque<size_t> pending_bytes;
  std::atomic<bool> closed{false};
  std::atomic<size_t> close_calls{0};
  std::thread::id closing_thread;
  std::atomic<size_t> preparations{0};
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
  void close()
  {
    std::lock_guard lock(mutex);
    ++close_calls;
    closing_thread = std::this_thread::get_id();
    closed         = true;
    cv.notify_all();
  }
  void consume()
  {
    {
      std::lock_guard l(mutex);
      if (pending_bytes.empty()) return;
      pending_bytes.pop_front();
    }
    if (!connector->get_next_split()) throw std::logic_error("missing published input");
  }
};
inline auto finite_jobs(size_t count,
                        std::function<std::unique_ptr<sirius::op::scan::scan_info>(int)> make_input)
{
  return [count,
          make_input = std::move(make_input),
          next       = 0]() mutable -> preparation_coordinator::claim_result {
    auto id = next++;
    if (static_cast<size_t>(id) >= count) return {};
    return preparation_coordinator::job{[make_input, id] { return make_input(id); }, {}};
  };
}
inline auto single_job(std::function<std::unique_ptr<sirius::op::scan::scan_info>()> make_input,
                       std::shared_ptr<preparation_unit> unit = {})
{
  return [job     = preparation_coordinator::job{std::move(make_input), std::move(unit)},
          claimed = false]() mutable -> preparation_coordinator::claim_result {
    if (std::exchange(claimed, true)) return {};
    return std::move(job);
  };
}
inline preparation_coordinator::publication make_publication(
  std::unique_ptr<sirius::op::scan::scan_info> batch, size_t bytes = 0)
{
  using input = sirius::op::scan::scan_operator_input;
  return {std::make_unique<input>(
            std::move(batch), nullptr, 0, std::nullopt, input::prefetch_start::deferred),
          bytes};
}
inline preparation_coordinator::source make_source(
  preparation_coordinator::execution_domain domain,
  std::shared_ptr<accumulator> coalescer = std::make_shared<accumulator>(1))
{
  preparation_coordinator::source source;
  source.coalescer = std::move(coalescer);
  source.execution = domain;
  source.construct = [](auto batch) { return make_publication(std::move(batch)); };
  return source;
}

struct fixture {
  sirius::exec::static_thread_pool pool;
  sirius::exec::scoped_dispatcher dispatcher;
  sirius::pipeline::completion_handler completion;
  preparation_coordinator coordinator;
  std::vector<std::shared_ptr<hold>> holds;
  std::vector<std::shared_ptr<sink>> outputs;
  explicit fixture(size_t workers                               = 2,
                   size_t lookahead                             = 4,
                   std::chrono::milliseconds interrupt_interval = k_interrupt_check_interval,
                   std::optional<preparation_options> options   = std::nullopt,
                   native_reader_executor* native               = nullptr)
    : pool(static_cast<int>(workers)),
      dispatcher(pool),
      coordinator(completion,
                  dispatcher,
                  options.value_or(preparation_options{workers,
                                                       lookahead,
                                                       workers,
                                                       1,
                                                       interrupt_interval,
                                                       0,
                                                       partial_batch_options{50, 20ms, 20ms}}),
                  native)
  {
    coordinator.after_publication_for_testing(
      [this](size_t id, auto const& metadata, size_t bytes) {
        auto out  = outputs.at(id);
        auto& tag = dynamic_cast<tagged&>(*metadata);
        std::lock_guard lock(out->mutex);
        ++out->preparations;
        out->published.push_back(tag.id);
        out->pending_bytes.push_back(bytes);
        out->cv.notify_all();
      });
  }
  void register_source(preparation_coordinator::source source, std::shared_ptr<sink> out)
  {
    outputs.push_back(out);
    source.destination = out->connector;
    coordinator.add_source(std::move(source));
  }
  std::thread owner;
  std::exception_ptr owner_error;
  void start(std::function<void()> register_on_owner = {})
  {
    owner = std::thread([this, register_on_owner = std::move(register_on_owner)] {
      try {
        if (register_on_owner) register_on_owner();
        coordinator.arm();
        coordinator.start();
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
    coordinator.drain();
  }
  std::shared_ptr<fixture> shutdown_guard()
  {
    return std::shared_ptr<fixture>(this, [](fixture* f) { f->stop(); });
  }
  ~fixture() { stop(); }
  std::shared_ptr<hold> make_hold()
  {
    auto h = std::make_shared<hold>();
    holds.push_back(h);
    return h;
  }
  void source(std::shared_ptr<accumulator> a,
              std::shared_ptr<sink> out,
              sirius::exec::invocable<preparation_coordinator::claim_result()> claim,
              std::shared_ptr<hold> construction                  = {},
              std::function<void(preparation_wakeup)> bind_wakeup = {},
              preparation_coordinator::execution_domain execution =
                preparation_coordinator::execution_domain::metadata_worker)
  {
    register_source({a,
                     std::move(claim),
                     [construction](std::unique_ptr<sirius::op::scan::scan_info> batch) {
                       if (construction) construction->wait();
                       return make_publication(std::move(batch));
                     },
                     out->connector,
                     [out] { out->close(); },
                     std::move(bind_wakeup),
                     execution},
                    out);
  }
};

}  // namespace sirius::test::preparation
