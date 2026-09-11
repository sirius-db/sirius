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

// Cache hits and fragmented reactor buffers produce many small H2D copies.
// device_copy_batch exists to collect those descriptors for one driver call
// without pretending adjacent pointers belong to one allocation. These tests
// make both requirements visible: invalid/empty pieces disappear, while valid
// pieces remain distinct and arrive at the right destination offsets.

#include "catch.hpp"
#include "cuda/device_copy_batch.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

TEST_CASE("device_copy_batch filters empty descriptors and resets for reuse",
          "[cuda][device_copy_batch][gpu_execution]")
{
  sirius::cuda::device_copy_batch batch;
  std::uint8_t byte{};

  batch.add(nullptr, &byte, 1);
  batch.add(&byte, nullptr, 1);
  batch.add(&byte, &byte, 0);

  CHECK(batch.empty());
  CHECK(batch.count() == 0);
  CHECK(batch.bytes() == 0);
  CHECK(batch.enqueue(rmm::cuda_stream_default) == cudaSuccess);

  batch.add(&byte, &byte, 1);
  REQUIRE(batch.count() == 1);
  REQUIRE(batch.bytes() == 1);
  batch.clear();

  CHECK(batch.empty());
  CHECK(batch.count() == 0);
  CHECK(batch.bytes() == 0);
}

TEST_CASE("device_copy_batch copies separate source allocations in one submission",
          "[cuda][device_copy_batch][gpu_execution]")
{
  constexpr std::size_t piece_size = 64;
  std::uint8_t* first{};
  std::uint8_t* second{};
  REQUIRE(cudaMallocHost(&first, piece_size) == cudaSuccess);
  REQUIRE(cudaMallocHost(&second, piece_size) == cudaSuccess);

  std::fill_n(first, piece_size, std::uint8_t{0x31});
  std::fill_n(second, piece_size, std::uint8_t{0x72});

  rmm::cuda_stream stream;
  rmm::device_buffer destination{2 * piece_size, stream.view()};

  sirius::cuda::device_copy_batch batch;
  batch.reserve(2);
  batch.add(destination.data(), first, piece_size);
  batch.add(static_cast<std::uint8_t*>(destination.data()) + piece_size, second, piece_size);

  // The two pieces stay separate even though their destination ranges touch:
  // the sources came from independent allocations and must not be fused across
  // an allocation boundary.
  REQUIRE(batch.count() == 2);
  REQUIRE(batch.bytes() == 2 * piece_size);
  REQUIRE(batch.enqueue(stream.view()) == cudaSuccess);
  stream.synchronize();

  std::array<std::uint8_t, 2 * piece_size> result{};
  REQUIRE(cudaMemcpy(result.data(), destination.data(), result.size(), cudaMemcpyDeviceToHost) ==
          cudaSuccess);

  CHECK(std::ranges::all_of(std::span{result}.first(piece_size),
                            [](auto byte) { return byte == 0x31; }));
  CHECK(std::ranges::all_of(std::span{result}.last(piece_size),
                            [](auto byte) { return byte == 0x72; }));

  REQUIRE(cudaFreeHost(second) == cudaSuccess);
  REQUIRE(cudaFreeHost(first) == cudaSuccess);
}
