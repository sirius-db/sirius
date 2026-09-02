/*
 * Copyright 2025, Sirius Contributors.
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

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <exec/exchange_staging_arena.hpp>
#include <sirius/exception.hpp>

#include <atomic>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

using sirius::exec::exchange_staging_arena;

namespace {
constexpr std::uint64_t kMiB = 1u << 20;
}

// ============================================================================
// ARENA-1: leases are aligned, and a released gap is immediately reusable
// ============================================================================

TEST_CASE("ARENA-1: lease/release bookkeeping and gap reuse", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);
  REQUIRE(arena.capacity() == kMiB);
  REQUIRE(arena.outstanding() == 0);

  // Offsets bump by the requested length rounded up to the alignment.
  auto a = arena.lease(100);
  auto b = arena.lease(4000);
  auto c = arena.lease(1);
  REQUIRE(a == 0);
  REQUIRE(b == 256);
  REQUIRE(c == 256 + 4096);
  REQUIRE(arena.outstanding() == 3);

  // The gap left by a NON-trailing release is immediately reusable -- this is the whole point
  // of the free list. Under the old bump allocator `d` landed past everything outstanding and
  // b's 4096 bytes stayed unreachable for the process lifetime.
  arena.release(b);
  auto d = arena.lease(1);
  REQUIRE(d == b);  // reuses b's block, address-ordered first fit
  REQUIRE(arena.outstanding() == 3);

  // The moment nothing is outstanding, the whole arena is free again from the base and fully
  // coalesced back into a single block.
  arena.release(a);
  arena.release(c);
  arena.release(d);
  REQUIRE(arena.outstanding() == 0);
  REQUIRE(arena.live_bytes() == 0);
  REQUIRE(arena.largest_free() == arena.capacity());
  REQUIRE(arena.lease(100) == 0);
  arena.release(0);

  // Peak concurrency, not lifetime total: three leases of 256 + 4096 + 256 were live at once.
  REQUIRE(arena.peak_live_bytes() == 256 + 4096 + 256);
}

// ============================================================================
// ARENA-2: every offset is aligned, and the region is plain device memory
// ============================================================================

TEST_CASE("ARENA-2: alignment and allocation type", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);

  REQUIRE(arena.base() != 0);
  REQUIRE(arena.base() % exchange_staging_arena::kAlignment == 0);
  for (auto len : {1ULL, 255ULL, 256ULL, 257ULL, 4097ULL}) {
    REQUIRE(arena.lease(len) % exchange_staging_arena::kAlignment == 0);
  }

  // The transport fast path hinges on this being an ordinary cudaMalloc device allocation —
  // pool memory would "work" while silently degrading ~220x over UCX cuda_ipc.
  cudaPointerAttributes attrs{};
  REQUIRE(cudaPointerGetAttributes(&attrs, reinterpret_cast<void*>(arena.base())) == cudaSuccess);
  REQUIRE(attrs.type == cudaMemoryTypeDevice);
}

// ============================================================================
// ARENA-3: the two refusal paths are distinct and each names what an operator
// can act on
// ============================================================================

TEST_CASE("ARENA-3: oversize and exhaustion are separate, actionable errors", "[staging_arena]")
{
  exchange_staging_arena arena(4096);
  auto a = arena.lease(1024);

  // Path 1 -- larger than the arena will EVER be. No amount of waiting or releasing helps, so
  // this must not be dressed up as "exhausted"; it is a sizing error, named as one.
  REQUIRE_THROWS_WITH(arena.lease(8192),
                      Catch::Contains("exceeds") && Catch::Contains("4096 byte capacity"));
  REQUIRE_THROWS_AS(arena.lease(8192), sirius::invalid_input_exception);

  // Path 2 -- fits the arena, does not fit what is free right now. This one IS transient, and
  // the message carries the fragmentation split (blocks / largest) plus the env var to raise.
  REQUIRE_THROWS_WITH(arena.lease(4000),
                      Catch::Contains("requested 4000") && Catch::Contains("4096 capacity") &&
                        Catch::Contains("largest") &&
                        Catch::Contains("SIRIUS_EXCHANGE_STAGING_BYTES"));

  // A refused lease must consume nothing -- checked against live bytes and the free list, not
  // against a head that a later full release would have reset anyway (which made the old
  // version of this assertion vacuous).
  REQUIRE(arena.outstanding() == 1);
  REQUIRE(arena.live_bytes() == 1024);
  REQUIRE(arena.total_free() == 4096 - 1024);
  arena.release(a);
  REQUIRE(arena.lease(4096) == 0);
}

// ============================================================================
// ARENA-4: misuse of lease/release is a defined error
// ============================================================================

TEST_CASE("ARENA-4: zero-length, unknown, and double release are rejected", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);

  REQUIRE_THROWS_AS(arena.lease(0), sirius::invalid_input_exception);

  auto a = arena.lease(100);
  REQUIRE_THROWS_AS(arena.release(a + 256), sirius::invalid_input_exception);
  arena.release(a);
  REQUIRE_THROWS_WITH(arena.release(a), Catch::Contains("not an outstanding lease"));
}

// ============================================================================
// ARENA-5: the unconfigured arena is a loud, named error
// ============================================================================

TEST_CASE("ARENA-5: require(nullptr) names the configuration knob", "[staging_arena]")
{
  REQUIRE_THROWS_WITH(
    exchange_staging_arena::require(nullptr),
    Catch::Contains("exchange staging arena not configured (set SIRIUS_EXCHANGE_STAGING_BYTES)"));

  exchange_staging_arena arena(kMiB);
  REQUIRE(&exchange_staging_arena::require(&arena) == &arena);
}

// ============================================================================
// ARENA-6: construction from the environment
// ============================================================================

TEST_CASE("ARENA-6: from_env honours the byte-suffix parser", "[staging_arena]")
{
  const auto* var = exchange_staging_arena::kCapacityEnvVar;

  unsetenv(var);
  REQUIRE(exchange_staging_arena::from_env() == nullptr);

  setenv(var, "4MiB", 1);
  auto arena = exchange_staging_arena::from_env();
  REQUIRE(arena != nullptr);
  REQUIRE(arena->capacity() == 4 * kMiB);

  setenv(var, "not-a-size", 1);
  REQUIRE_THROWS_WITH(exchange_staging_arena::from_env(),
                      Catch::Contains("SIRIUS_EXCHANGE_STAGING_BYTES"));

  setenv(var, "0", 1);
  REQUIRE_THROWS_AS(exchange_staging_arena::from_env(), sirius::invalid_input_exception);

  unsetenv(var);
}

// ============================================================================
// ARENA-7: a stuck lease pins only its own block, not the arena above it
// ============================================================================

// The regression this pins: reclamation used to happen only at zero outstanding, so ONE lease
// that was never released turned the arena into a monotonic bump allocator — every later
// query's staging traffic burned arena permanently and exhausted it within ~20 passing
// queries. The free list makes this structural rather than a special case: a released block
// goes straight back and coalesces, so traffic above a stuck lease reuses the same space
// forever no matter what order the releases arrive in.
TEST_CASE("ARENA-7: a stuck lease does not pin the space above it", "[staging_arena]")
{
  exchange_staging_arena arena(4096);
  auto stuck = arena.lease(256);
  REQUIRE(stuck == 0);

  // Steady-state traffic: each lease is released after its transmit. Every cycle must land on
  // the same offset -- address-ordered first fit always returns the lowest block that fits,
  // and the released block coalesces straight back into it.
  for (int i = 0; i < 100; ++i) {
    auto t = arena.lease(2048);
    REQUIRE(t == 256);
    arena.release(t);
  }
  REQUIRE(arena.outstanding() == 1);

  // Free bytes are back to baseline: everything above the stuck lease is grantable at once.
  auto big = arena.lease(4096 - 256);
  REQUIRE(big == 256);
  arena.release(big);

  // Out-of-order releases coalesce: z and y merge with the tail into one block starting at y.
  auto x = arena.lease(256);
  auto y = arena.lease(256);
  auto z = arena.lease(256);
  REQUIRE(z == 256 * 3);
  arena.release(z);
  arena.release(y);
  REQUIRE(arena.lease(256) == y);  // first fit lands at the start of the merged block
  arena.release(y);
  arena.release(x);

  arena.release(stuck);
  REQUIRE(arena.outstanding() == 0);
  REQUIRE(arena.lease(4096) == 0);
  arena.release(0);
}

// ============================================================================
// ARENA-8: capacity conservation — the invariant that makes drift impossible
// ============================================================================
TEST_CASE("ARENA-8: free + live == capacity at every step", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);
  auto check = [&] { REQUIRE(arena.total_free() + arena.live_bytes() == arena.capacity()); };

  check();
  std::vector<std::uint64_t> held;
  for (int i = 0; i < 16; ++i) {
    held.push_back(arena.lease(1000 + i * 137));
    check();
  }
  // Release in an order chosen to be neither LIFO nor FIFO.
  for (std::size_t i = 0; i < held.size(); ++i) {
    arena.release(held[(i * 7) % held.size()]);
    check();
  }
  REQUIRE(arena.outstanding() == 0);
  REQUIRE(arena.live_bytes() == 0);
  REQUIRE(arena.total_free() == arena.capacity());
  // Fully coalesced back to one block.
  REQUIRE(arena.largest_free() == arena.capacity());
}

// ============================================================================
// ARENA-9: adjacent frees coalesce, forward and backward
// ============================================================================
TEST_CASE("ARENA-9: neighbouring released blocks merge", "[staging_arena]")
{
  exchange_staging_arena arena(4096);
  auto a = arena.lease(1024);
  auto b = arena.lease(1024);
  auto c = arena.lease(1024);
  REQUIRE(arena.largest_free() == 1024);  // only the tail

  // Releasing the middle block alone cannot yet satisfy a 2048 lease.
  arena.release(b);
  REQUIRE(arena.largest_free() == 1024);

  // Releasing its successor merges b + c + tail -- coalescing forward.
  arena.release(c);
  REQUIRE(arena.largest_free() == 1024 + 1024 + 1024);

  // And backward, back to a single whole-arena block.
  arena.release(a);
  REQUIRE(arena.largest_free() == 4096);
  REQUIRE(arena.total_free() == 4096);
  REQUIRE(arena.lease(4096) == 0);
}

// ============================================================================
// ARENA-10: out-of-order steady state does not drift
// ============================================================================
// The defect this pins: with a bump head, every out-of-order release leaves a gap while every
// new lease advances the head, so the head tracks LIFETIME TOTALS rather than concurrency. A
// CN whose receive path grants concurrently and releases out of order therefore drifts toward
// exhaustion no matter how little is actually live. Here 8 leases of 4 KiB are live at any
// moment — 32 KiB of a 1 MiB arena — and the loop runs 10,000 cycles. The bump allocator
// exhausted within the first few hundred; MEASURED on this box before the change.
TEST_CASE("ARENA-10: out-of-order lease/release never drifts", "[staging_arena]")
{
  constexpr std::uint64_t kCapacity  = kMiB;
  constexpr std::uint64_t kLeaseSize = 4096;
  constexpr int kConcurrent          = 8;
  constexpr int kCycles              = 10000;

  exchange_staging_arena arena(kCapacity);
  std::vector<std::uint64_t> live;
  for (int i = 0; i < kConcurrent; ++i) {
    live.push_back(arena.lease(kLeaseSize));
  }

  // Deterministic pseudo-shuffle: release a rotating non-adjacent slot each cycle and
  // immediately re-lease, so the live set size is constant but the release order is not.
  std::uint64_t idx = 0;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    idx = (idx + 5) % kConcurrent;  // 5 is coprime with 8: visits every slot, never in order
    arena.release(live[idx]);
    live[idx] = arena.lease(kLeaseSize);  // must not throw
    REQUIRE(arena.live_bytes() == kConcurrent * kLeaseSize);
    REQUIRE(arena.total_free() + arena.live_bytes() == kCapacity);
  }

  REQUIRE(arena.outstanding() == kConcurrent);
  REQUIRE(arena.peak_live_bytes() == kConcurrent * kLeaseSize);
  for (auto offset : live) {
    arena.release(offset);
  }
  REQUIRE(arena.largest_free() == kCapacity);
}

// ============================================================================
// ARENA-11: live leases never alias, verified through the device memory itself
// ============================================================================
TEST_CASE("ARENA-11: concurrent leases address disjoint device memory", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);
  constexpr std::size_t kLen = 4096;
  std::vector<std::uint64_t> offsets;
  for (int i = 0; i < 16; ++i) {
    offsets.push_back(arena.lease(kLen));
  }

  // Stamp each lease with its own byte pattern...
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    auto* p = reinterpret_cast<void*>(arena.base() + offsets[i]);
    REQUIRE(cudaMemset(p, static_cast<int>(i + 1), kLen) == cudaSuccess);
  }
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // ...and read every one back. Any overlap shows up as a wrong pattern.
  std::vector<unsigned char> host(kLen);
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    const auto* p = reinterpret_cast<const void*>(arena.base() + offsets[i]);
    REQUIRE(cudaMemcpy(host.data(), p, kLen, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(host.front() == static_cast<unsigned char>(i + 1));
    REQUIRE(host.back() == static_cast<unsigned char>(i + 1));
  }

  // Release a subset and re-lease: the reused space must still be disjoint from what is live.
  for (std::size_t i = 0; i < offsets.size(); i += 2) {
    arena.release(offsets[i]);
  }
  std::vector<std::uint64_t> reused;
  for (std::size_t i = 0; i < offsets.size() / 2; ++i) {
    reused.push_back(arena.lease(kLen));
  }
  for (auto offset : reused) {
    auto* p = reinterpret_cast<void*>(arena.base() + offset);
    REQUIRE(cudaMemset(p, 0xEE, kLen) == cudaSuccess);
  }
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  // The odd-indexed leases were never released and must be untouched.
  for (std::size_t i = 1; i < offsets.size(); i += 2) {
    const auto* p = reinterpret_cast<const void*>(arena.base() + offsets[i]);
    REQUIRE(cudaMemcpy(host.data(), p, kLen, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(host.front() == static_cast<unsigned char>(i + 1));
  }
}

// ============================================================================
// ARENA-12: the allocator is safe under concurrent lease/release
// ============================================================================
// `lease`/`release` are called from transport, RPC and engine threads concurrently; nothing in
// the suite covered that until now.
TEST_CASE("ARENA-12: concurrent lease/release keeps its invariants", "[staging_arena]")
{
  exchange_staging_arena arena(8 * kMiB);
  constexpr int kThreads = 8;
  constexpr int kIters   = 2000;
  std::atomic<int> failures{0};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&arena, &failures] {
      for (int i = 0; i < kIters; ++i) {
        try {
          auto a = arena.lease(1024);
          auto b = arena.lease(4096);
          if (a % exchange_staging_arena::kAlignment != 0) { ++failures; }
          if (a == b) { ++failures; }
          arena.release(a);
          arena.release(b);
        } catch (const std::exception&) {
          ++failures;  // an 8 MiB arena with 8 threads x 5 KiB must never exhaust
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  REQUIRE(failures.load() == 0);
  REQUIRE(arena.outstanding() == 0);
  REQUIRE(arena.live_bytes() == 0);
  REQUIRE(arena.largest_free() == arena.capacity());
}

// ============================================================================
// ARENA-13: an oversized request is refused, not wrapped
// ============================================================================
TEST_CASE("ARENA-13: align_up overflow cannot alias a live lease", "[staging_arena]")
{
  exchange_staging_arena arena(kMiB);
  auto live = arena.lease(1024);

  // align_up wraps to 0 for these; without the guard the request would slip past the fit scan
  // and register a zero-length lease that aliases whatever is allocated next.
  REQUIRE_THROWS_AS(arena.lease(std::numeric_limits<std::uint64_t>::max()),
                    sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(arena.lease(std::numeric_limits<std::uint64_t>::max() - 100),
                    sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(arena.lease(2 * kMiB), sirius::invalid_input_exception);

  REQUIRE(arena.outstanding() == 1);
  REQUIRE(arena.live_bytes() == 1024);
  arena.release(live);
}
