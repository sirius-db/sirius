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

#include "catch.hpp"
#include "data/chunked_spill_copy.hpp"
#include "data/spill_chunked_converters.hpp"

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

using sirius::spill::chunked_copy_batcher;
using sirius::spill::copy_op;

namespace {

/// Fake source/destination addresses; the batcher never dereferences them.
void* fake_ptr(std::uintptr_t v) { return reinterpret_cast<void*>(v); }

struct captured_chunk {
  std::vector<copy_op> ops;
  std::size_t bytes;
};

std::vector<captured_chunk> run_batcher(std::size_t chunk_bytes,
                                        const std::vector<std::size_t>& op_sizes)
{
  std::vector<captured_chunk> chunks;
  chunked_copy_batcher batcher(chunk_bytes, [&](std::span<const copy_op> ops) {
    captured_chunk c;
    c.ops.assign(ops.begin(), ops.end());
    c.bytes =
      std::accumulate(ops.begin(), ops.end(), std::size_t{0}, [](std::size_t a, const copy_op& op) {
        return a + op.size;
      });
    chunks.push_back(std::move(c));
  });
  std::uintptr_t addr = 0x1000;
  for (auto size : op_sizes) {
    batcher.add(fake_ptr(addr), fake_ptr(addr + 0x100000000ull), size);
    addr += size;
  }
  batcher.flush_pending();
  return chunks;
}

}  // namespace

TEST_CASE("chunked_copy_batcher flushes at chunk boundaries", "[chunked_spill_copy]")
{
  auto chunks = run_batcher(8, {4, 4, 4});
  REQUIRE(chunks.size() == 2);
  REQUIRE(chunks[0].bytes == 8);
  REQUIRE(chunks[1].bytes == 4);
}

TEST_CASE("chunked_copy_batcher splits ops larger than the chunk", "[chunked_spill_copy]")
{
  // No submission, and no op within one, may exceed the chunk size.
  auto chunks = run_batcher(8, {28});
  REQUIRE(chunks.size() == 4);  // 8 + 8 + 8 + 4
  std::size_t total = 0;
  for (auto const& c : chunks) {
    REQUIRE(c.bytes <= 8);
    for (auto const& op : c.ops) {
      REQUIRE(op.size <= 8);
    }
    total += c.bytes;
  }
  REQUIRE(total == 28);
  REQUIRE(chunks.back().bytes == 4);
}

TEST_CASE("chunked_copy_batcher split pieces are contiguous", "[chunked_spill_copy]")
{
  // Split pieces must tile the original [dst, dst+size) range in order.
  std::vector<copy_op> all_ops;
  chunked_copy_batcher batcher(10, [&](std::span<const copy_op> ops) {
    all_ops.insert(all_ops.end(), ops.begin(), ops.end());
  });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 25);
  batcher.flush_pending();

  REQUIRE(all_ops.size() == 3);  // 10 + 10 + 5
  std::uintptr_t expected_dst = 0x1000;
  std::uintptr_t expected_src = 0x2000;
  for (auto const& op : all_ops) {
    REQUIRE(reinterpret_cast<std::uintptr_t>(op.dst) == expected_dst);
    REQUIRE(reinterpret_cast<std::uintptr_t>(op.src) == expected_src);
    expected_dst += op.size;
    expected_src += op.size;
  }
  REQUIRE(expected_dst == 0x1000 + 25);
}

TEST_CASE("chunked_copy_batcher with chunk_bytes 0 flushes once", "[chunked_spill_copy]")
{
  auto chunks = run_batcher(0, {4, 4, 4, 4});
  REQUIRE(chunks.size() == 1);
  REQUIRE(chunks[0].bytes == 16);
}

TEST_CASE("chunked_copy_batcher ignores empty and null ops", "[chunked_spill_copy]")
{
  std::size_t flushes = 0;
  chunked_copy_batcher batcher(8, [&](std::span<const copy_op>) { ++flushes; });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 0);
  batcher.add(nullptr, fake_ptr(0x2000), 8);
  batcher.add(fake_ptr(0x1000), nullptr, 8);
  batcher.flush_pending();
  REQUIRE(flushes == 0);
  REQUIRE(batcher.bytes_added() == 0);
  REQUIRE(batcher.chunks_flushed() == 0);
}

TEST_CASE("chunked_copy_batcher reports stats", "[chunked_spill_copy]")
{
  std::size_t flushes = 0;
  chunked_copy_batcher batcher(8, [&](std::span<const copy_op>) { ++flushes; });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 20);
  batcher.flush_pending();
  REQUIRE(batcher.bytes_added() == 20);
  REQUIRE(batcher.chunks_flushed() == flushes);
  REQUIRE(batcher.chunks_flushed() == 3);  // 8 + 8 + 4
  REQUIRE(batcher.largest_submission_bytes() == 8);
}

TEST_CASE("chunked_copy_batcher flush_pending is idempotent", "[chunked_spill_copy]")
{
  std::size_t flushes = 0;
  chunked_copy_batcher batcher(8, [&](std::span<const copy_op>) { ++flushes; });
  batcher.add(fake_ptr(0x1000), fake_ptr(0x2000), 4);
  batcher.flush_pending();
  batcher.flush_pending();
  REQUIRE(flushes == 1);
}

TEST_CASE("register_chunked_spill_converters replaces the GPU->HOST pair", "[chunked_spill_copy]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  REQUIRE(
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::host_data_representation>());

  sirius::spill::register_chunked_spill_converters(registry, 1ull << 20);
  REQUIRE(
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::host_data_representation>());
  // The HOST->GPU restore direction stays on the builtin converter.
  REQUIRE(
    registry
      .has_converter<cucascade::host_data_representation, cucascade::gpu_table_representation>());

  // Registering again must not throw on the duplicate key.
  REQUIRE_NOTHROW(sirius::spill::register_chunked_spill_converters(registry, 2ull << 20));
}

TEST_CASE("register_chunked_spill_converters with 0 keeps the builtin converter",
          "[chunked_spill_copy]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  sirius::spill::register_chunked_spill_converters(registry, 0);
  REQUIRE(
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::host_data_representation>());
}
