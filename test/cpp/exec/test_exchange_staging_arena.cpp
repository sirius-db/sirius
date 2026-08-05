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

#include <cstdlib>

using sirius::exec::exchange_staging_arena;

namespace {
constexpr std::uint64_t kMiB = 1u << 20;
}

// ============================================================================
// ARENA-1: leases bump by aligned offsets; the head resets only at zero
// outstanding
// ============================================================================

TEST_CASE("ARENA-1: lease/release bookkeeping and reset-at-zero", "[staging_arena]")
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

  // Releasing SOME leases must not move the head: earlier offsets stay claimed by design (bump
  // allocator, no free list), so the next lease lands past everything ever handed out.
  arena.release(b);
  auto d = arena.lease(1);
  REQUIRE(d == c + 256);
  REQUIRE(arena.outstanding() == 3);

  // The moment nothing is outstanding, the whole arena is free again from the base.
  arena.release(a);
  arena.release(c);
  arena.release(d);
  REQUIRE(arena.outstanding() == 0);
  REQUIRE(arena.lease(100) == 0);
  arena.release(0);

  // The watermark records the deepest head ever reached, surviving the reset.
  REQUIRE(arena.high_water() == c + 256 + 256);
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
// ARENA-3: exhaustion is a loud error naming requested/free/capacity
// ============================================================================

TEST_CASE("ARENA-3: exhaustion names requested, free, and capacity", "[staging_arena]")
{
  exchange_staging_arena arena(4096);
  auto a = arena.lease(1024);

  REQUIRE_THROWS_WITH(arena.lease(8192),
                      Catch::Contains("requested 8192") && Catch::Contains("3072 free") &&
                        Catch::Contains("4096 capacity") &&
                        Catch::Contains("SIRIUS_EXCHANGE_STAGING_BYTES"));
  REQUIRE_THROWS_AS(arena.lease(8192), sirius::invalid_input_exception);

  // A refused lease consumes nothing.
  REQUIRE(arena.outstanding() == 1);
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
