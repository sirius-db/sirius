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
#include "exec/cuda_event_completion_poll.hpp"

#include <atomic>
#include <deque>
#include <future>
#include <memory>
#include <optional>
#include <semaphore>
#include <thread>
#include <vector>

using sirius::exec::cuda_event_completion_poll;
using sirius::exec::retire_lane;

namespace {
template <class Result>
struct acquisition_attempt {
  Result operator()();
};

template <class Fn>
concept acquirable =
  requires(cuda_event_completion_poll& poll, Fn&& fn) { poll.acquire(std::forward<Fn>(fn)); };

// A contextual boolean test accepts explicit conversions and does not require !.
struct explicit_boolean {
  explicit operator bool() & { return true; }
  bool operator!() = delete;
};
struct not_boolean {};
static_assert(acquirable<acquisition_attempt<bool>>);
static_assert(acquirable<acquisition_attempt<int*>>);
static_assert(acquirable<acquisition_attempt<std::unique_ptr<int>>>);
static_assert(acquirable<acquisition_attempt<std::optional<int>>>);
static_assert(acquirable<acquisition_attempt<explicit_boolean>>);
static_assert(!acquirable<acquisition_attempt<bool&>>);
static_assert(!acquirable<acquisition_attempt<const bool&>>);
static_assert(!acquirable<acquisition_attempt<bool&&>>);
static_assert(!acquirable<acquisition_attempt<not_boolean>>);
static_assert(!acquirable<acquisition_attempt<void>>);
static_assert(!acquirable<int>);

// A controlled CUDA runtime: no timing assumptions, no poisoned device context.
// synchronize delivers callbacks only on success; errors deliberately leave
// them pending so tests can prove ownership was not released early.
struct controlled_stream {
  struct callback {
    cudaStreamCallback_t fn;
    void* data;
  };
  std::mutex mutex;
  std::deque<callback> callbacks;
  std::atomic<cudaError_t> install_error{cudaSuccess};
  std::atomic<cudaError_t> query_error{cudaSuccess};
  std::atomic<cudaError_t> sync_error{cudaSuccess};
  std::atomic<int> queries{0}, synchronizations{0};
  std::binary_semaphore* query_entered{nullptr};
  std::binary_semaphore* query_release{nullptr};

  cudaStream_t stream() { return reinterpret_cast<cudaStream_t>(this); }
  static controlled_stream& get(cudaStream_t s) { return *reinterpret_cast<controlled_stream*>(s); }
  static cudaError_t add(cudaStream_t s, cudaStreamCallback_t fn, void* data, unsigned)
  {
    auto& self = get(s);
    if (self.install_error.load() != cudaSuccess) { return self.install_error; }
    std::lock_guard lock(self.mutex);
    self.callbacks.push_back({fn, data});
    return cudaSuccess;
  }
  static cudaError_t query(cudaStream_t s)
  {
    auto& self = get(s);
    ++self.queries;
    if (self.query_entered) {
      self.query_entered->release();
      self.query_release->acquire();
    }
    return self.query_error;
  }
  static cudaError_t synchronize(cudaStream_t s)
  {
    auto& self = get(s);
    ++self.synchronizations;
    if (self.sync_error.load() != cudaSuccess) { return self.sync_error; }
    self.complete();
    return cudaSuccess;
  }
  void complete(cudaError_t status = cudaSuccess)
  {
    std::deque<callback> ready;
    {
      std::lock_guard lock(mutex);
      ready.swap(callbacks);
    }
    for (auto cb : ready) {
      cb.fn(stream(), status, cb.data);
    }
  }
  // Models externally established stop of device work AND callback delivery.
  void stop()
  {
    std::lock_guard lock(mutex);
    callbacks.clear();
  }
  static sirius::exec::detail::completion_cuda_api api() { return {add, query, synchronize}; }
};

template <class Fn>
void submit(retire_lane& lane, Fn&& fn)
{
  auto sub = lane.begin();
  sub.on_retire(std::forward<Fn>(fn));
  REQUIRE(sub.commit() == cudaSuccess);
}

struct real_stream {
  cudaStream_t value{};
  void* data{};
  real_stream()
  {
    REQUIRE(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaMalloc(&data, 4096) == cudaSuccess);
  }
  ~real_stream()
  {
    cudaStreamSynchronize(value);
    cudaFree(data);
    cudaStreamDestroy(value);
  }
};
}  // namespace

TEST_CASE("completion frontier does not retire undelivered work",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  int calls  = 0;
  submit(lane, [&](cudaError_t) noexcept { ++calls; });
  CHECK(poll.drain_all() == 0);
  CHECK(calls == 0);
  CHECK_FALSE(lane.idle());
  stream.complete();
  CHECK(poll.drain_all() == 1);
  CHECK(poll.drain_all() == 0);
  CHECK(calls == 1);
  CHECK(lane.idle());
}

TEST_CASE("completion submissions support empty grouped RAII and idempotent commit",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  {
    auto sub = lane.begin();
    CHECK(sub.commit() == cudaSuccess);
  }
  CHECK(lane.idle());
  int calls = 0;
  {
    auto sub = lane.begin();
    for (int i = 0; i < 3; ++i) {
      sub.on_retire([&](cudaError_t) noexcept { ++calls; });
    }
    CHECK(sub.commit() == cudaSuccess);
    CHECK(sub.commit() == cudaSuccess);
    CHECK_THROWS_AS(sub.on_retire([](cudaError_t) noexcept {}), std::logic_error);
  }
  CHECK(stream.callbacks.size() == 1);
  CHECK(lane.oldest_pending_locked() == 1);
  {
    auto sub = lane.begin();
    sub.on_retire([&](cudaError_t) noexcept { ++calls; });
  }
  REQUIRE(poll.quiesce() == cudaSuccess);
  CHECK(calls == 4);
}

TEST_CASE("completion rejects empty retirement functions without changing staged work",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  int calls  = 0;
  {
    auto sub = lane.begin();
    CHECK_THROWS_AS(sub.on_retire({}), std::invalid_argument);
    sirius::exec::retire_fn fn = [&](cudaError_t) noexcept { ++calls; };
    sub.on_retire(std::move(fn));
    CHECK_THROWS_AS(sub.on_retire(nullptr), std::invalid_argument);
    REQUIRE(sub.commit() == cudaSuccess);
  }
  CHECK(stream.callbacks.size() == 1);
  CHECK(poll.quiesce() == cudaSuccess);
  CHECK(calls == 1);
  CHECK(lane.idle());
}

TEST_CASE("acquire returns a move-only resource made available by retirement",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  std::unique_ptr<int> available;
  submit(poll.lane_for(stream.stream()),
         [resource = std::make_unique<int>(42), &available](cudaError_t) mutable noexcept {
           available = std::move(resource);
         });
  stream.complete();
  auto acquired = poll.acquire([&] { return std::move(available); });
  REQUIRE(acquired);
  CHECK(*acquired == 42);
  CHECK_FALSE(available);
}

TEST_CASE("completion preserves successful prefix and the first device failure",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  std::vector<cudaError_t> seen(3, cudaErrorUnknown);
  submit(lane, [&](cudaError_t e) noexcept { seen[0] = e; });
  stream.complete();
  submit(lane, [&](cudaError_t e) noexcept { seen[1] = e; });
  stream.complete(cudaErrorIllegalAddress);
  submit(lane, [&](cudaError_t e) noexcept { seen[2] = e; });
  stream.complete(cudaErrorLaunchFailure);
  CHECK(poll.drain_all() == 3);
  CHECK(seen[0] == cudaSuccess);
  CHECK(seen[1] == cudaErrorIllegalAddress);
  CHECK(seen[2] == cudaErrorIllegalAddress);
  CHECK(lane.fault_error() == cudaErrorIllegalAddress);
}

TEST_CASE("query errors never force cache retirement", "[exec][cuda_event_completion_poll]")
{
  controlled_stream a, b;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& la  = poll.lane_for(a.stream());
  auto& lb  = poll.lane_for(b.stream());
  int calls = 0;
  submit(la, [&](cudaError_t) noexcept { ++calls; });
  submit(lb, [&](cudaError_t) noexcept { ++calls; });
  a.query_error = cudaErrorIllegalAddress;
  CHECK_FALSE(poll.acquire([] { return false; }));
  CHECK(calls == 0);
  CHECK_FALSE(la.idle());
  CHECK_FALSE(lb.idle());
  a.complete(cudaErrorIllegalAddress);
  b.complete();
  CHECK(poll.drain_all() == 2);
}

TEST_CASE("failed callback installation retains ownership until safe",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  std::vector<cudaError_t> seen;
  seen.reserve(2);
  submit(lane, [&](cudaError_t e) noexcept { seen.push_back(e); });
  stream.install_error  = cudaErrorNotSupported;
  const bool sync_fails = GENERATE(false, true);
  stream.sync_error     = sync_fails ? cudaErrorIllegalAddress : cudaSuccess;
  {
    auto sub = lane.begin();
    sub.on_retire([&](cudaError_t e) noexcept { seen.push_back(e); });
    const auto expected = sync_fails ? cudaErrorIllegalAddress : cudaErrorNotSupported;
    CHECK(sub.commit() == expected);
    CHECK(sub.commit() == expected);
  }
  CHECK(lane.enqueue_error() == cudaErrorNotSupported);
  CHECK_THROWS_AS(lane.begin(), std::logic_error);
  if (sync_fails) {
    CHECK(poll.drain_all() == 0);
    CHECK(poll.quiesce() == cudaErrorIllegalAddress);
    CHECK(seen.empty());
    stream.sync_error = cudaSuccess;
    CHECK(poll.quiesce() == cudaSuccess);
  } else {
    CHECK(seen.empty());  // no inline invocation under the submission lock
    CHECK(poll.drain_all() == 2);
  }
  REQUIRE(seen.size() == 2);
  CHECK(seen[0] == cudaSuccess);
  CHECK(seen[1] == cudaErrorNotSupported);
}

TEST_CASE("terminal retirement delivers the exact requested failure once",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane       = poll.lane_for(stream.stream());
  int calls        = 0;
  cudaError_t seen = cudaSuccess;
  submit(lane, [&](cudaError_t e) noexcept {
    ++calls;
    seen = e;
  });
  stream.stop();
  CHECK(poll.fail_all(cudaErrorLaunchFailure) == 1);
  CHECK(poll.fail_all(cudaErrorUnknown) == 0);
  CHECK(calls == 1);
  CHECK(seen == cudaErrorLaunchFailure);
  CHECK(lane.idle());
  CHECK(lane.detached());
}

TEST_CASE("wait for progress retires a high ticket while a low ticket is blocked",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream slow, fast;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& a = poll.lane_for(slow.stream());
  auto& b = poll.lane_for(fast.stream());
  for (int i = 0; i < 100; ++i) {
    submit(b, [](cudaError_t) noexcept {});
  }
  fast.complete();
  REQUIRE(poll.drain_all() == 100);
  bool ready = false;
  submit(a, [](cudaError_t) noexcept {});
  submit(b, [&](cudaError_t) noexcept { ready = true; });
  fast.complete();
  CHECK(poll.wait_for_progress() == cuda_event_completion_poll::progress::made);
  CHECK(ready);
  CHECK_FALSE(a.idle());
  CHECK(b.idle());
}

TEST_CASE("concurrent drain preserves FIFO and quiesce waits for active invocation",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream, other;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane       = poll.lane_for(stream.stream());
  auto& other_lane = poll.lane_for(other.stream());
  std::binary_semaphore entered{0}, release{0};
  std::atomic<int> sequence{0};
  std::atomic<bool> ordered{false}, other_done{false};
  submit(lane, [&](cudaError_t) noexcept {
    entered.release();
    release.acquire();
    sequence = 1;
  });
  stream.complete();
  std::thread first([&] { lane.drain(); });
  entered.acquire();
  submit(lane, [&](cudaError_t) noexcept {
    ordered  = sequence.load() == 1;
    sequence = 2;
  });
  stream.complete();
  submit(other_lane, [&](cudaError_t) noexcept { other_done = true; });
  other.complete();
  CHECK(poll.drain_all() == 1);
  CHECK(other_done);
  CHECK(sequence == 0);
  CHECK_FALSE(lane.idle());
  auto quiesce = std::async(std::launch::async, [&] { return poll.quiesce(); });
  CHECK(quiesce.wait_for(std::chrono::milliseconds{20}) == std::future_status::timeout);
  release.release();
  first.join();
  CHECK(quiesce.get() == cudaSuccess);
  CHECK(ordered);
  CHECK(sequence == 2);
}

TEST_CASE("terminal retirement waits for an active drainer before failing pending work",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  std::binary_semaphore entered{0}, release{0};
  std::atomic<int> sequence{0};
  bool ordered     = false;
  cudaError_t seen = cudaSuccess;
  submit(lane, [&](cudaError_t) noexcept {
    entered.release();
    release.acquire();
    sequence = 1;
  });
  stream.complete();
  std::thread first([&] { lane.drain(); });
  entered.acquire();
  submit(lane, [&](cudaError_t error) noexcept {
    ordered  = sequence.load() == 1;
    sequence = 2;
    seen     = error;
  });
  stream.stop();
  auto cleanup =
    std::async(std::launch::async, [&] { return poll.fail_all(cudaErrorLaunchFailure); });
  CHECK(cleanup.wait_for(std::chrono::milliseconds{20}) == std::future_status::timeout);
  release.release();
  first.join();
  CHECK(cleanup.get() == 1);
  CHECK(ordered);
  CHECK(sequence == 2);
  CHECK(seen == cudaErrorLaunchFailure);
  CHECK(lane.idle());
}

TEST_CASE("detach fences in-flight query and forbids subsequent CUDA access",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  std::binary_semaphore entered{0}, release{0};
  stream.query_entered = &entered;
  stream.query_release = &release;
  auto query           = std::async(std::launch::async, [&] { return lane.poll_health(); });
  entered.acquire();
  auto detach = std::async(std::launch::async, [&] { poll.detach(); });
  CHECK(detach.wait_for(std::chrono::milliseconds{20}) == std::future_status::timeout);
  release.release();
  CHECK(query.get() == cudaSuccess);
  detach.get();
  CHECK(lane.poll_health() == cudaErrorInvalidResourceHandle);
  CHECK(poll.quiesce() == cudaSuccess);
  CHECK_THROWS_AS(lane.begin(), std::logic_error);
  CHECK_THROWS_AS(poll.lane_for(stream.stream()), std::logic_error);
  CHECK(stream.queries == 1);
  CHECK(stream.synchronizations == 0);
}

TEST_CASE("detach alone cannot authorize retirement", "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  int calls  = 0;
  submit(lane, [&](cudaError_t) noexcept { ++calls; });
  poll.detach();
  CHECK(poll.quiesce() == cudaErrorNotReady);
  CHECK(calls == 0);
  CHECK_FALSE(lane.idle());
  stream.complete();
  CHECK(poll.quiesce() == cudaSuccess);
  CHECK(calls == 1);
  CHECK(stream.synchronizations == 0);
}

TEST_CASE("multiple acquirers observe another drainer's completion",
          "[exec][cuda_event_completion_poll]")
{
  controlled_stream stream;
  cuda_event_completion_poll poll(controlled_stream::api());
  auto& lane = poll.lane_for(stream.stream());
  std::atomic<bool> available{false};
  std::atomic<int> attempts{0};
  submit(lane, [&](cudaError_t) noexcept { available = true; });
  auto acquire = [&] {
    return poll.acquire([&] {
      ++attempts;
      return available.load();
    });
  };
  auto a = std::async(std::launch::async, acquire);
  auto b = std::async(std::launch::async, acquire);
  while (attempts.load() < 4) {
    std::this_thread::yield();
  }
  stream.complete();
  poll.drain_all();
  CHECK(a.get());
  CHECK(b.get());
  CHECK(poll.wait_for_progress() == cuda_event_completion_poll::progress::none);
}

TEST_CASE("real CUDA streams retire hundreds of reads once in FIFO order",
          "[exec][cuda_event_completion_poll][gpu_execution]")
{
  const int streams = GENERATE(1, 8, 16);
  std::vector<std::unique_ptr<real_stream>> fixtures;
  for (int i = 0; i < streams; ++i) {
    fixtures.push_back(std::make_unique<real_stream>());
  }
  cuda_event_completion_poll poll;
  std::vector<int> retired(streams, 0);
  std::atomic<int> failures{0};
  constexpr int reads = 100;
  for (int read = 0; read < reads; ++read) {
    for (int i = 0; i < streams; ++i) {
      auto sub = poll.lane_for(fixtures[i]->value).begin();
      sub.on_retire([&, i, read](cudaError_t e) noexcept {
        if (e != cudaSuccess || retired[i]++ != read) { ++failures; }
      });
      REQUIRE(cudaMemsetAsync(fixtures[i]->data, read, 4096, sub.stream()) == cudaSuccess);
      REQUIRE(sub.commit() == cudaSuccess);
    }
  }
  CHECK(poll.quiesce() == cudaSuccess);
  CHECK(failures == 0);
  for (int count : retired) {
    CHECK(count == reads);
  }
  poll.detach();
}

TEST_CASE("unequal completion queues across 8 and 16 lanes drain independently",
          "[exec][cuda_event_completion_poll]")
{
  const int count = GENERATE(8, 16);
  std::vector<std::unique_ptr<controlled_stream>> streams;
  for (int i = 0; i < count; ++i) {
    streams.push_back(std::make_unique<controlled_stream>());
  }
  cuda_event_completion_poll poll(controlled_stream::api());
  std::vector<int> retired(count, 0);
  int expected = 0;
  for (int i = 0; i < count; ++i) {
    auto& lane = poll.lane_for(streams[i]->stream());
    for (int j = 0; j < 100 + i * 7; ++j) {
      submit(lane, [&, i](cudaError_t) noexcept { ++retired[i]; });
    }
    if (i % 2) {
      streams[i]->complete();
      expected += 100 + i * 7;
    }
  }
  CHECK(poll.drain_all() == expected);
  for (int i = 0; i < count; ++i) {
    CHECK(retired[i] == (i % 2 ? 100 + i * 7 : 0));
  }
  CHECK(poll.quiesce() == cudaSuccess);
  for (int i = 0; i < count; ++i) {
    CHECK(retired[i] == 100 + i * 7);
  }
}

TEST_CASE("concurrent producers preserve real CUDA ticket order",
          "[exec][cuda_event_completion_poll][gpu_execution]")
{
  real_stream stream;
  cuda_event_completion_poll poll;
  auto& lane = poll.lane_for(stream.value);
  int next = 0, retired = 0;
  std::atomic<int> errors{0};
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&] {
      for (int j = 0; j < 100; ++j) {
        auto sub        = lane.begin();
        const int index = next++;
        sub.on_retire([&, index](cudaError_t e) noexcept {
          if (e != cudaSuccess || retired++ != index) { ++errors; }
        });
        if (cudaMemsetAsync(stream.data, index, 4096, sub.stream()) != cudaSuccess ||
            sub.commit() != cudaSuccess) {
          ++errors;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  CHECK(poll.quiesce() == cudaSuccess);
  CHECK(retired == 400);
  CHECK(errors == 0);
}
