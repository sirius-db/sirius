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
#include "exec/exchange_ring.hpp"
#include "sirius/exception.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <set>
#include <vector>

using sirius::exec::frame_ring;

namespace {
constexpr std::size_t MiB = std::size_t{1} << 20;
}

TEST_CASE("a frame ring is one cudaMalloc region of fixed frames", "[exchange_ring]")
{
  frame_ring ring(0, 4, 2 * MiB);
  CHECK(ring.frames() == 4);
  CHECK(ring.frame_bytes() == 2 * MiB);
  CHECK(ring.bytes() == 8 * MiB);
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(ring.base())) ==
          cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeDevice);
  CHECK(attributes.device == 0);
  for (std::uint32_t i = 0; i < 4; ++i) {
    CHECK(ring.frame_address(i) == ring.base() + i * 2 * MiB);
  }
  CHECK_THROWS_AS(ring.frame_address(4), sirius::invalid_input_exception);
}

TEST_CASE("a frame ring leases every frame once and never blocks", "[exchange_ring]")
{
  frame_ring ring(0, 3, MiB);
  std::set<std::uint32_t> leased;
  for (int i = 0; i < 3; ++i) {
    auto frame = ring.lease();
    REQUIRE(frame.has_value());
    CHECK(leased.insert(*frame).second);
  }
  CHECK_FALSE(ring.lease().has_value());
  CHECK(ring.leased() == 3);
  CHECK(ring.peak_leased() == 3);

  // Any returned frame serves the next lease, whatever the order: fixed frames cannot fragment.
  ring.release(1);
  auto again = ring.lease();
  REQUIRE(again.has_value());
  CHECK(*again == 1);
  ring.release(0);
  ring.release(2);
  ring.release(1);
  CHECK(ring.leased() == 0);
  CHECK(ring.peak_leased() == 3);
}

TEST_CASE("a frame ring refuses a frame it did not lease", "[exchange_ring]")
{
  frame_ring ring(0, 2, MiB);
  CHECK_THROWS_AS(ring.release(0), sirius::invalid_input_exception);
  auto frame = ring.lease();
  REQUIRE(frame.has_value());
  ring.release(*frame);
  CHECK_THROWS_AS(ring.release(*frame), sirius::invalid_input_exception);
  CHECK_THROWS_AS(ring.release(7), sirius::invalid_input_exception);
}

TEST_CASE("a frame ring refuses a bad geometry", "[exchange_ring]")
{
  CHECK_THROWS_AS(frame_ring(0, 0, MiB), sirius::invalid_input_exception);
  CHECK_THROWS_AS(frame_ring(0, 2, 0), sirius::invalid_input_exception);
  CHECK_THROWS_AS(frame_ring(0, 2, 1000), sirius::invalid_input_exception);
}

TEST_CASE("a frame ring's frame count rounds down, at least one", "[exchange_ring]")
{
  CHECK(frame_ring::frames_in(std::size_t{1} << 30, 64 * MiB) == 16);
  CHECK(frame_ring::frames_in(1536 * MiB, 64 * MiB) == 24);
  CHECK(frame_ring::frames_in(100 * MiB, 64 * MiB) == 1);
  CHECK(frame_ring::frames_in(10 * MiB, 64 * MiB) == 1);
}
