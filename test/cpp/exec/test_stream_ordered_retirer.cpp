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
#include "exec/stream_ordered_retirer.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <thread>
#include <vector>

using sirius::exec::no_pending_v;
using sirius::exec::retire_lane;
using sirius::exec::stream_ordered_retirer;

namespace {

/// A real stream plus a device buffer, so submissions carry actual work the
/// frontier has to pass rather than an empty callback.
class stream_fixture {
 public:
  stream_fixture()
  {
    REQUIRE(cudaStreamCreate(&_stream) == cudaSuccess);
    REQUIRE(cudaMalloc(&_dst, kBytes) == cudaSuccess);
    REQUIRE(cudaMallocHost(&_src, kBytes) == cudaSuccess);
  }

  ~stream_fixture()
  {
    cudaStreamSynchronize(_stream);
    cudaFreeHost(_src);
    cudaFree(_dst);
    cudaStreamDestroy(_stream);
  }

  stream_fixture(stream_fixture const&)            = delete;
  stream_fixture& operator=(stream_fixture const&) = delete;

  [[nodiscard]] cudaStream_t stream() const noexcept { return _stream; }

  /// Enqueue a copy so the ticket has something to sit behind.
  void enqueue_work() const
  {
    REQUIRE(cudaMemcpyAsync(_dst, _src, kBytes, cudaMemcpyHostToDevice, _stream) == cudaSuccess);
  }

 private:
  static constexpr std::size_t kBytes = 1 << 16;
  cudaStream_t _stream{};
  void* _dst{nullptr};
  void* _src{nullptr};
};

/// Drain until `pred` holds or the budget runs out.  The frontier is advanced
/// by a driver callback thread, so a completed copy is not instantly visible.
template <class Pred>
bool drain_until(stream_ordered_retirer& r, Pred pred, int budget = 2000)
{
  for (int i = 0; i < budget; ++i) {
    r.drain_all();
    if (pred()) { return true; }
    std::this_thread::sleep_for(std::chrono::microseconds{200});
  }
  return pred();
}

}  // namespace

TEST_CASE("a committed submission retires once the stream passes it",
          "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  std::atomic<int> retired{0};
  std::atomic<cudaError_t> seen{cudaErrorUnknown};

  {
    auto sub = lane.begin();
    fx.enqueue_work();
    sub.on_retire([&](cudaError_t status) noexcept {
      seen.store(status);
      retired.fetch_add(1);
    });
    CHECK(sub.commit() == cudaSuccess);
  }

  CHECK(drain_until(retirer, [&] { return retired.load() == 1; }));
  CHECK(seen.load() == cudaSuccess);
  CHECK(lane.idle());
}

TEST_CASE("an empty submission allocates no ticket", "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  {
    auto sub = lane.begin();  // nothing staged
    CHECK(sub.commit() == cudaSuccess);
  }

  // No ticket means nothing to wait on -- a lane that counted one here would
  // lag its frontier forever, since no callback was enqueued for it.
  CHECK(lane.idle());
  CHECK(lane.oldest_pending_hint() == no_pending_v);
  CHECK(retirer.drain_all() == 0);
}

TEST_CASE("batches retire in submission order", "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  constexpr int kBatches = 8;
  std::vector<int> order;
  std::mutex order_m;

  for (int i = 0; i < kBatches; ++i) {
    auto sub = lane.begin();
    fx.enqueue_work();
    sub.on_retire([&, i](cudaError_t) noexcept {
      std::lock_guard g(order_m);
      order.push_back(i);
    });
    CHECK(sub.commit() == cudaSuccess);
  }

  CHECK(drain_until(retirer, [&] {
    std::lock_guard g(order_m);
    return order.size() == kBatches;
  }));

  std::lock_guard g(order_m);
  REQUIRE(order.size() == kBatches);
  for (int i = 0; i < kBatches; ++i) {
    CHECK(order[i] == i);  // work completes in submission order, so retirement does too
  }
}

TEST_CASE("several fns in one submission share a ticket", "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  std::atomic<int> retired{0};
  {
    auto sub = lane.begin();
    fx.enqueue_work();
    for (int i = 0; i < 3; ++i) {
      sub.on_retire([&](cudaError_t) noexcept { retired.fetch_add(1); });
    }
    CHECK(sub.commit() == cudaSuccess);
    // One ticket for the batch, so one callback -- not three.
    CHECK(lane.oldest_pending_locked() == 1);
  }

  CHECK(drain_until(retirer, [&] { return retired.load() == 3; }));
  CHECK(lane.idle());
}

TEST_CASE("an uncommitted submission still retires", "[exec][retirer][gpu_execution]")
{
  // Abandoning a scope that already launched work would recycle buffers the
  // copy engines are still reading, so the destructor commits.
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  std::atomic<int> retired{0};
  {
    auto sub = lane.begin();
    fx.enqueue_work();
    sub.on_retire([&](cudaError_t) noexcept { retired.fetch_add(1); });
    // deliberately no commit()
  }

  CHECK(drain_until(retirer, [&] { return retired.load() == 1; }));
}

TEST_CASE("drain does nothing while the frontier has not moved", "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  std::atomic<int> retired{0};
  {
    auto sub = lane.begin();
    // A long copy chain, so the batch is still outstanding when we drain.
    for (int i = 0; i < 256; ++i) {
      fx.enqueue_work();
    }
    sub.on_retire([&](cudaError_t) noexcept { retired.fetch_add(1); });
    CHECK(sub.commit() == cudaSuccess);
  }

  CHECK(lane.oldest_pending_locked() == 1);
  retirer.drain_all();  // may or may not have completed; must not retire early
  CHECK(retired.load() <= 1);

  CHECK(drain_until(retirer, [&] { return retired.load() == 1; }));
}

TEST_CASE("quiesce synchronizes and retires everything", "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  std::atomic<int> retired{0};
  for (int i = 0; i < 4; ++i) {
    auto sub = lane.begin();
    fx.enqueue_work();
    sub.on_retire([&](cudaError_t) noexcept { retired.fetch_add(1); });
    CHECK(sub.commit() == cudaSuccess);
  }

  CHECK(retirer.quiesce() == cudaSuccess);
  CHECK(retired.load() == 4);  // nothing may be left outstanding after quiesce
  CHECK(lane.idle());
}

TEST_CASE("fail_all retires the backlog with the given status", "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  std::atomic<int> failed{0};
  {
    auto sub = lane.begin();
    for (int i = 0; i < 256; ++i) {
      fx.enqueue_work();
    }
    sub.on_retire([&](cudaError_t status) noexcept {
      if (status == cudaErrorUnknown) { failed.fetch_add(1); }
    });
    CHECK(sub.commit() == cudaSuccess);
  }

  // Terminal recovery: stop the device first, exactly as the header requires,
  // or the fns would hand back buffers still being read.
  REQUIRE(cudaStreamSynchronize(fx.stream()) == cudaSuccess);
  retirer.fail_all(cudaErrorUnknown);

  CHECK(failed.load() + static_cast<int>(lane.idle()) >= 1);
  CHECK(lane.idle());
}

TEST_CASE("acquire drains until the resource is available", "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  // A one-slot "pool" the retirement hands back, so acquire() must wait for the
  // frontier rather than fail.
  std::atomic<bool> available{false};
  {
    auto sub = lane.begin();
    fx.enqueue_work();
    sub.on_retire([&](cudaError_t) noexcept { available.store(true); });
    CHECK(sub.commit() == cudaSuccess);
  }

  auto* got = retirer.acquire([&]() -> int* {
    static int slot = 42;
    return available.load() ? &slot : nullptr;
  });

  REQUIRE(got != nullptr);
  CHECK(*got == 42);
}

TEST_CASE("acquire reports failure instead of hanging when nothing is outstanding",
          "[exec][retirer][gpu_execution]")
{
  stream_fixture fx;
  stream_ordered_retirer retirer;
  static_cast<void>(retirer.lane_for(fx.stream()));

  // Nothing was ever submitted, so no frontier can advance; acquire must return
  // the falsy value rather than block forever.
  auto* got = retirer.acquire([]() -> int* { return nullptr; });
  CHECK(got == nullptr);
}

TEST_CASE("a detached lane never touches its stream again", "[exec][retirer][gpu_execution]")
{
  // A lane holds a raw cudaStream_t, and an owner that does not control stream
  // lifetime (the prefetching cache: callers pass a stream in per read) can
  // reach teardown after the stream is gone.  Synchronizing a dangling handle
  // faults inside the driver rather than returning an error, so detach() has to
  // make every later call stream-free.
  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(s);

  std::atomic<int> retired{0};
  {
    auto sub = lane.begin();
    sub.on_retire([&](cudaError_t) noexcept { retired.fetch_add(1); });
    CHECK(sub.commit() == cudaSuccess);
  }
  REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);

  // The stream's owner is finished with it -- which is what made the work safe
  // to stop waiting on in the first place.
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);

  retirer.detach();
  CHECK(lane.detached());
  CHECK(lane.poll_health() == cudaErrorInvalidResourceHandle);

  // Would be a segfault on an attached lane.
  CHECK(retirer.quiesce() == cudaSuccess);
  CHECK(retired.load() == 1);
  CHECK(lane.idle());
  // ~stream_ordered_retirer quiesces again; it must stay stream-free too.
}

TEST_CASE("lane_for is stable per stream", "[exec][retirer][gpu_execution]")
{
  stream_fixture a;
  stream_fixture b;
  stream_ordered_retirer retirer;

  auto& la1 = retirer.lane_for(a.stream());
  auto& la2 = retirer.lane_for(a.stream());
  auto& lb  = retirer.lane_for(b.stream());

  CHECK(&la1 == &la2);
  CHECK(&la1 != &lb);
  CHECK(la1.stream() == a.stream());
  CHECK(lb.stream() == b.stream());
}

TEST_CASE("lanes on separate streams retire independently", "[exec][retirer][gpu_execution]")
{
  stream_fixture a;
  stream_fixture b;
  stream_ordered_retirer retirer;

  std::atomic<int> ra{0};
  std::atomic<int> rb{0};

  {
    auto& lane = retirer.lane_for(a.stream());
    auto sub   = lane.begin();
    a.enqueue_work();
    sub.on_retire([&](cudaError_t) noexcept { ra.fetch_add(1); });
    CHECK(sub.commit() == cudaSuccess);
  }
  {
    auto& lane = retirer.lane_for(b.stream());
    auto sub   = lane.begin();
    b.enqueue_work();
    sub.on_retire([&](cudaError_t) noexcept { rb.fetch_add(1); });
    CHECK(sub.commit() == cudaSuccess);
  }

  CHECK(drain_until(retirer, [&] { return ra.load() == 1 && rb.load() == 1; }));
}

TEST_CASE("concurrent submitters on one lane keep ticket order", "[exec][retirer][gpu_execution]")
{
  // begin() serializes submitters, so tickets are handed out in the same order
  // their callbacks reach the stream -- the invariant the whole design rests on.
  stream_fixture fx;
  stream_ordered_retirer retirer;
  auto& lane = retirer.lane_for(fx.stream());

  constexpr int kThreads   = 4;
  constexpr int kPerThread = 16;
  std::atomic<int> retired{0};

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&] {
      for (int i = 0; i < kPerThread; ++i) {
        auto sub = lane.begin();
        fx.enqueue_work();
        sub.on_retire([&](cudaError_t) noexcept { retired.fetch_add(1); });
        CHECK(sub.commit() == cudaSuccess);
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }

  CHECK(drain_until(retirer, [&] { return retired.load() == kThreads * kPerThread; }));
  CHECK(lane.idle());
}
