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

#include "op/scan/prefetched_ranges.hpp"
#include "utils/sirius_test_env.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using sirius::op::scan::device_stream_pool;
using sirius::op::scan::prefetched_ranges;

/// What the hooks observed; shared so the hooks outlive the object under test.
struct hook_log {
  std::atomic<int> claims{0};
  std::atomic<int> releases{0};
  std::atomic<int> reason{-1};
  std::atomic<std::size_t> bytes{0};
  std::atomic<std::size_t> reserved{0};
};

constexpr std::size_t k_bytes = 4096;

/// One entry holding one @p bytes buffer allocated on @p stream.
std::vector<prefetched_ranges::device_ranges> make_ranges(::cuda::stream_ref stream,
                                                          std::size_t bytes = k_bytes)
{
  std::vector<prefetched_ranges::device_ranges> ranges(1);
  ranges[0].buffers.emplace_back(bytes, stream);
  ranges[0].spans.emplace_back(static_cast<std::uint8_t const*>(ranges[0].buffers[0].data()),
                               ranges[0].buffers[0].size());
  return ranges;
}

std::shared_ptr<prefetched_ranges> make_prefetch(int device,
                                                 device_stream_pool::lease lease,
                                                 std::shared_ptr<hook_log> log,
                                                 std::size_t bytes = k_bytes)
{
  auto ranges = make_ranges(lease.view, bytes);
  return std::make_shared<prefetched_ranges>(
    device,
    std::move(lease),
    std::move(ranges),
    bytes,
    bytes * 2,
    [log]() noexcept { log->claims.fetch_add(1); },
    [log](prefetched_ranges::release_event e) noexcept {
      log->releases.fetch_add(1);
      log->reason.store(static_cast<int>(e.reason));
      log->bytes.store(e.bytes);
      log->reserved.store(e.reserved_bytes);
    });
}

}  // namespace

TEST_CASE("device_stream_pool hands out round-robin leases that share the pool",
          "[prefetched_ranges][device_stream_pool]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto pool = device_stream_pool::create(0, 3);
  REQUIRE(pool->size() == 3);
  CHECK(pool->device() == 0);

  auto const a = pool->acquire();
  auto const b = pool->acquire();
  auto const c = pool->acquire();
  auto const d = pool->acquire();
  CHECK(a.owner == pool);
  CHECK(a.view.get() != b.view.get());
  CHECK(b.view.get() != c.view.get());
  CHECK(a.view.get() == d.view.get());

  // A zero count still yields a usable stream.
  CHECK(device_stream_pool::create(0, 0)->size() == 1);
}

TEST_CASE("prefetched_ranges claim releases the budget only", "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto log  = std::make_shared<hook_log>();
  auto pool = device_stream_pool::create(0, 1);
  auto pr   = make_prefetch(0, pool->acquire(), log);
  CHECK(pr->total_bytes() == k_bytes);
  CHECK(pr->reserved_bytes() == 2 * k_bytes);
  REQUIRE_FALSE(pr->claimed());

  pr->claim();
  CHECK(pr->claimed());
  CHECK(log->claims.load() == 1);
  // Claiming frees no buffer and reports no release.
  CHECK(log->releases.load() == 0);
  CHECK(pr->entry_count() == 1);
  CHECK_FALSE(pr->taken());

  // Idempotent.
  pr->claim();
  CHECK(log->claims.load() == 1);

  // Still takeable after a claim: the claim did not give up the buffers.
  REQUIRE(pr->set_ready());
  rmm::cuda_stream consumer;
  auto taken = pr->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  CHECK(log->claims.load() == 1);
  CHECK(log->releases.load() == 1);
  CHECK(log->reason.load() == static_cast<int>(prefetched_ranges::release_reason::consumed));
  CHECK(log->reserved.load() == 2 * k_bytes);
  taken->clear();
  consumer.synchronize();
  pr.reset();
  CHECK(log->claims.load() == 1);
  CHECK(log->releases.load() == 1);
}

TEST_CASE("prefetched_ranges destruction claims then releases once", "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto pool = device_stream_pool::create(0, 1);

  SECTION("unconsumed")
  {
    auto log = std::make_shared<hook_log>();
    {
      auto pr = make_prefetch(0, pool->acquire(), log);
      REQUIRE(pr->set_ready());
    }
    CHECK(log->claims.load() == 1);
    CHECK(log->releases.load() == 1);
    CHECK(log->reason.load() == static_cast<int>(prefetched_ranges::release_reason::unconsumed));
    CHECK(log->bytes.load() == k_bytes);
  }

  SECTION("failed")
  {
    auto log = std::make_shared<hook_log>();
    {
      auto pr = make_prefetch(0, pool->acquire(), log);
      REQUIRE(pr->set_failed(std::make_exception_ptr(std::runtime_error("io"))));
      CHECK_FALSE(pr->set_ready());  // already left pending
      CHECK(pr->current() == prefetched_ranges::state::failed);
      CHECK(pr->peek().empty());
    }
    CHECK(log->claims.load() == 1);
    CHECK(log->releases.load() == 1);
    CHECK(log->reason.load() == static_cast<int>(prefetched_ranges::release_reason::failed));
  }

  SECTION("claimed first")
  {
    auto log = std::make_shared<hook_log>();
    {
      auto pr = make_prefetch(0, pool->acquire(), log);
      pr->claim();
    }
    CHECK(log->claims.load() == 1);
    CHECK(log->releases.load() == 1);
  }
}

TEST_CASE("prefetched_ranges take refuses until ready", "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto log  = std::make_shared<hook_log>();
  auto pool = device_stream_pool::create(0, 1);
  auto pr   = make_prefetch(0, pool->acquire(), log);
  rmm::cuda_stream consumer;

  CHECK(pr->wait_for(std::chrono::milliseconds{1}) == prefetched_ranges::state::pending);
  CHECK_FALSE(pr->take(::cuda::stream_ref{consumer.value()}).has_value());
  CHECK(log->claims.load() == 0);
  CHECK(log->releases.load() == 0);

  REQUIRE(pr->set_ready());
  CHECK(pr->peek().size() == 1);
  CHECK(pr->take(::cuda::stream_ref{consumer.value()}).has_value());
  CHECK(pr->taken());
  // Exactly once.
  CHECK_FALSE(pr->take(::cuda::stream_ref{consumer.value()}).has_value());
  CHECK(log->releases.load() == 1);
}

TEST_CASE("prefetched_ranges take orders the consumer stream after the producer fill",
          "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  constexpr std::size_t big = 64UL << 20;
  auto log                  = std::make_shared<hook_log>();
  auto pool                 = device_stream_pool::create(0, 1);
  auto lease                = pool->acquire();
  auto const producer       = lease.view;

  auto ranges = make_ranges(producer, big);
  auto* data  = static_cast<std::uint8_t*>(ranges[0].buffers[0].data());
  auto pr     = std::make_shared<prefetched_ranges>(
    0,
    std::move(lease),
    std::move(ranges),
    big,
    big,
    [log]() noexcept { log->claims.fetch_add(1); },
    [log](prefetched_ranges::release_event) noexcept { log->releases.fetch_add(1); });

  // Keep the producer stream busy before the fill, so a consumer that did not wait on the ready
  // event would read the buffer before it is written.
  rmm::device_buffer scratch(big, producer);
  for (int i = 0; i < 16; ++i) {
    REQUIRE(cudaMemsetAsync(scratch.data(), i, big, producer.get()) == cudaSuccess);
  }
  REQUIRE(cudaMemsetAsync(data, 0xAB, big, producer.get()) == cudaSuccess);
  // set_ready() only records the event: it does not synchronize the producer stream.
  REQUIRE(pr->set_ready());

  rmm::cuda_stream consumer;
  auto taken = pr->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  REQUIRE(taken->size() == 1);
  REQUIRE((*taken)[0].spans.size() == 1);
  CHECK((*taken)[0].spans[0].data() == data);

  // Read back on the consumer stream only: correct only if take() ordered it after the fill.
  std::vector<std::uint8_t> host(big);
  REQUIRE(cudaMemcpyAsync(host.data(), data, big, cudaMemcpyDeviceToHost, consumer.value()) ==
          cudaSuccess);
  consumer.synchronize();
  CHECK(std::all_of(host.begin(), host.end(), [](std::uint8_t v) { return v == 0xAB; }));

  taken->clear();
  consumer.synchronize();
  scratch.resize(0, producer);
  REQUIRE(cudaStreamSynchronize(producer.get()) == cudaSuccess);
}

TEST_CASE("prefetched_ranges keeps the stream pool alive past the pool handle",
          "[prefetched_ranges][device_stream_pool]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto log = std::make_shared<hook_log>();
  std::weak_ptr<device_stream_pool const> weak;
  std::shared_ptr<prefetched_ranges> pr;
  {
    auto pool = device_stream_pool::create(0, 2);
    weak      = pool;
    pr        = make_prefetch(0, pool->acquire(), log);
  }
  // The owner handle is gone; the lease inside the prefetch still pins the streams.
  CHECK_FALSE(weak.expired());
  REQUIRE(pr->set_ready());

  // An untaken buffer is freed on the producer stream, which must still exist.
  pr.reset();
  CHECK(weak.expired());
  CHECK(log->claims.load() == 1);
  CHECK(log->releases.load() == 1);

  // A consumed prefetch no longer needs the pool: the taken buffers outlive it.
  std::optional<std::vector<prefetched_ranges::device_ranges>> taken;
  rmm::cuda_stream consumer;
  {
    auto pool = device_stream_pool::create(0, 1);
    auto p2   = make_prefetch(0, pool->acquire(), log);
    REQUIRE(p2->set_ready());
    taken = p2->take(::cuda::stream_ref{consumer.value()});
    REQUIRE(taken.has_value());
  }
  taken.reset();
  consumer.synchronize();
}

TEST_CASE("prefetched_ranges refuses a wrong-device take", "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto log  = std::make_shared<hook_log>();
  auto pool = device_stream_pool::create(0, 1);
  rmm::cuda_stream consumer;

  SECTION("flagged by the consumer")
  {
    {
      auto pr = make_prefetch(0, pool->acquire(), log);
      REQUIRE(pr->set_ready());
      pr->flag_wrong_device();
      CHECK(pr->peek().empty());
      CHECK_FALSE(pr->take(::cuda::stream_ref{consumer.value()}).has_value());
      CHECK(log->releases.load() == 0);
    }
    CHECK(log->claims.load() == 1);
    CHECK(log->releases.load() == 1);
    CHECK(log->reason.load() == static_cast<int>(prefetched_ranges::release_reason::wrong_device));
  }
}

TEST_CASE("prefetched_ranges refuses a take on a different current device",
          "[prefetched_ranges][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  // Ensure device 0 is current regardless of what a previous multi-GPU test left behind.
  rmm::cuda_set_device_raii bind0{rmm::cuda_device_id{0}};
  auto log   = std::make_shared<hook_log>();
  auto pool1 = device_stream_pool::create(1, 1);
  rmm::cuda_stream consumer;
  {
    auto pr = [&] {
      rmm::cuda_set_device_raii bind{rmm::cuda_device_id{1}};
      return make_prefetch(1, pool1->acquire(), log);
    }();
    REQUIRE(pr->set_ready());
    // This thread's current device is not 1: take() must not hand out buffers it would free on
    // the wrong device.
    CHECK_FALSE(pr->take(::cuda::stream_ref{consumer.value()}).has_value());
    CHECK(pr->peek().empty());
  }
  CHECK(log->releases.load() == 1);
  CHECK(log->reason.load() == static_cast<int>(prefetched_ranges::release_reason::wrong_device));
}

// settle() is noexcept and builds the abandon error under a try/catch so an allocation failure
// cannot terminate; that failure path is not injected at runtime here (no practical allocator
// failure hook), so this covers the normal abandon transition and the waiter wake-up.
TEST_CASE("prefetched_ranges abandon settles failed and wakes a waiter", "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto log  = std::make_shared<hook_log>();
  auto pool = device_stream_pool::create(0, 1);
  auto pr   = make_prefetch(0, pool->acquire(), log);

  std::atomic<int> seen{-1};
  std::thread waiter([&] { seen.store(static_cast<int>(pr->wait_for(std::chrono::seconds{30}))); });
  REQUIRE(pr->abandon());
  waiter.join();
  CHECK(seen.load() == static_cast<int>(prefetched_ranges::state::failed));
  CHECK(pr->abandoned());
  CHECK(pr->failure() != nullptr);
  CHECK_FALSE(pr->abandon());
  CHECK_FALSE(pr->set_ready());

  pr.reset();
  CHECK(log->releases.load() == 1);
  CHECK(log->reason.load() == static_cast<int>(prefetched_ranges::release_reason::failed));
}

TEST_CASE("an untaken prefetch frees its buffers under the shared allocation mutex",
          "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto log   = std::make_shared<hook_log>();
  auto pool  = device_stream_pool::create(0, 1);
  auto mutex = std::make_shared<std::mutex>();
  auto lease = pool->acquire();
  auto pr    = std::make_shared<prefetched_ranges>(
    0,
    lease,
    make_ranges(lease.view),
    k_bytes,
    k_bytes,
    [log]() noexcept { log->claims.fetch_add(1); },
    [log](prefetched_ranges::release_event) noexcept { log->releases.fetch_add(1); },
    mutex);

  // While another issuer holds the mutex (it is attaching or resetting the per-stream tracker),
  // the last reference's destruction must wait to free the untaken buffers...
  std::unique_lock<std::mutex> issuer(*mutex);
  std::atomic<bool> freed{false};
  std::thread dropper([&] {
    pr.reset();
    freed.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  CHECK_FALSE(freed.load());
  // ... but the hooks do not take the mutex: the claim already ran, before the free.
  CHECK(log->claims.load() == 1);

  issuer.unlock();
  dropper.join();
  CHECK(freed.load());
  CHECK(log->releases.load() == 1);
}

TEST_CASE("a consumed prefetch and a never-ready one do not deadlock on the allocation mutex",
          "[prefetched_ranges]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto log   = std::make_shared<hook_log>();
  auto pool  = device_stream_pool::create(0, 1);
  auto mutex = std::make_shared<std::mutex>();
  rmm::cuda_stream consumer;

  auto build = [&] {
    auto lease = pool->acquire();
    return std::make_shared<prefetched_ranges>(
      0,
      lease,
      make_ranges(lease.view),
      k_bytes,
      k_bytes,
      [log]() noexcept { log->claims.fetch_add(1); },
      [log](prefetched_ranges::release_event) noexcept { log->releases.fetch_add(1); },
      mutex);
  };

  // Consumed: the buffers left, so the destructor frees nothing and needs no lock -- it must not
  // block while the mutex is held elsewhere.
  auto taken_from = build();
  REQUIRE(taken_from->set_ready());
  auto taken = taken_from->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  {
    std::lock_guard<std::mutex> held(*mutex);
    taken_from.reset();  // would self-deadlock if the destructor locked unconditionally
  }
  taken->clear();
  consumer.synchronize();

  // A setup failure drops the prefetch on the issuer's own thread AFTER it released the mutex:
  // destruction locks and unlocks cleanly.
  {
    std::unique_lock<std::mutex> alloc(*mutex);
    auto pending = build();
    alloc.unlock();
    pending.reset();
  }
  CHECK(log->releases.load() == 2);
}
