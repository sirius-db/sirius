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

#include <catch.hpp>
#include <io/cache/types.hpp>
#include <io/io_request.hpp>
#include <io/types.hpp>
#include <io/uring/types.hpp>
#include <io/uring/uring_reactor.hpp>
#include <sys/uio.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <vector>

using sirius::io::grouped_coordinator;
using sirius::io::IO_BLOCK_SIZE;
using sirius::io::prepared_io_completion;
using sirius::io::range;
using sirius::io::cache::cached_chunk;
using sirius::io::uring::max_dynamic_io_size;
using sirius::io::uring::min_dynamic_io_size;
using sirius::io::uring::uring_io_op;
using sirius::io::uring::uring_reactor;
using sirius::io::uring::detail::dynamic_io_target;
using sirius::io::uring::detail::fill_remaining_iovecs;
using sirius::io::uring::detail::is_odirect_compatible;
using sirius::io::uring::detail::is_odirect_runtime_error;
using sirius::io::uring::detail::odirect_available;

namespace {

class aligned_bytes {
 public:
  explicit aligned_bytes(std::size_t bytes)
    : _data(static_cast<std::uint8_t*>(std::aligned_alloc(IO_BLOCK_SIZE, bytes)))
  {
    REQUIRE(_data != nullptr);
  }

  ~aligned_bytes() { std::free(_data); }

  aligned_bytes(aligned_bytes const&)            = delete;
  aligned_bytes& operator=(aligned_bytes const&) = delete;

  [[nodiscard]] std::uint8_t* get() const noexcept { return _data; }

 private:
  std::uint8_t* _data;
};

}  // namespace

static_assert(!std::is_move_constructible_v<uring_io_op>);
static_assert(!std::is_move_assignable_v<uring_io_op>);

TEST_CASE("io_uring dynamic chunk target is backlog and slot aware", "[uring_readv]")
{
  constexpr std::size_t block = 64UL << 10;

  CHECK(dynamic_io_target(1, 64, block) == min_dynamic_io_size);
  CHECK(dynamic_io_target(2UL << 20, 64, block) == 2UL << 20);
  CHECK(dynamic_io_target(64UL << 20, 64, block) == 4UL << 20);
  CHECK(dynamic_io_target(64UL << 20, 512, block) == max_dynamic_io_size);
  CHECK(dynamic_io_target(64UL << 20, 0, block) == 0);
  CHECK(dynamic_io_target(64UL << 20, 64, 0) == 0);
}

TEST_CASE("io_uring dynamic target honors fixed blocks and the physical cap", "[uring_readv]")
{
  constexpr std::size_t block = 1UL << 20;

  CHECK(dynamic_io_target(1, 64, block) == block);
  CHECK(dynamic_io_target(3 * block + 1, 64, block) == 4 * block);
  CHECK(dynamic_io_target(max_dynamic_io_size, 3, block) == 3 * block);
  CHECK(dynamic_io_target(max_dynamic_io_size, 1, 2 * max_dynamic_io_size) == max_dynamic_io_size);
}

TEST_CASE("io_uring O_DIRECT validation covers the entire readv", "[uring_readv]")
{
  aligned_bytes first{2 * IO_BLOCK_SIZE};
  aligned_bytes second{IO_BLOCK_SIZE};
  std::array<iovec, 2> buffers{iovec{first.get(), 2 * IO_BLOCK_SIZE},
                               iovec{second.get(), IO_BLOCK_SIZE}};

  CHECK(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE}, buffers));
  CHECK_FALSE(is_odirect_compatible(range{1, 3 * IO_BLOCK_SIZE}, buffers));
  CHECK_FALSE(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE - 1}, buffers));

  buffers[1].iov_base = second.get() + 1;
  CHECK_FALSE(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE}, buffers));
  buffers[1].iov_base = second.get();
  buffers[1].iov_len -= 1;
  CHECK_FALSE(is_odirect_compatible(range{0, 3 * IO_BLOCK_SIZE - 1}, buffers));
}

TEST_CASE("io_uring falls back when O_DIRECT is unavailable or rejected", "[uring_readv]")
{
  CHECK(odirect_available(true, 3));
  CHECK_FALSE(odirect_available(false, 3));
  CHECK_FALSE(odirect_available(true, -1));
  CHECK(is_odirect_runtime_error(EINVAL));
  CHECK(is_odirect_runtime_error(EOPNOTSUPP));
  CHECK_FALSE(is_odirect_runtime_error(EIO));
}

TEST_CASE("io_uring readv resume preserves byte order after short reads", "[uring_readv]")
{
  std::array<std::uint8_t, 600> storage{};
  std::array<iovec, 3> source{
    iovec{storage.data(), 100}, iovec{storage.data() + 100, 200}, iovec{storage.data() + 300, 300}};
  std::vector<iovec> remaining;

  SECTION("exact iovec boundary")
  {
    fill_remaining_iovecs(source, 100, remaining);
    REQUIRE(remaining.size() == 2);
    CHECK(remaining[0].iov_base == storage.data() + 100);
    CHECK(remaining[0].iov_len == 200);
  }

  SECTION("middle of an iovec")
  {
    fill_remaining_iovecs(source, 150, remaining);
    REQUIRE(remaining.size() == 2);
    CHECK(remaining[0].iov_base == storage.data() + 150);
    CHECK(remaining[0].iov_len == 150);
    CHECK(remaining[1].iov_base == storage.data() + 300);
    CHECK(remaining[1].iov_len == 300);
  }

  SECTION("all bytes consumed")
  {
    fill_remaining_iovecs(source, 600, remaining);
    CHECK(remaining.empty());
  }
}

TEST_CASE("io_uring cache callback precedes its coordinator credit", "[uring_readv]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(IO_BLOCK_SIZE, 1);
  auto future      = coordinator->get_future();
  cached_chunk chunk{0};

  std::size_t callback_count = 0;
  std::size_t credits_seen   = 0;
  bool callback_success      = false;
  cached_chunk* observed     = nullptr;
  auto completion            = std::make_shared<prepared_io_completion>(
    [&](std::span<cached_chunk* const> chunks, bool success) noexcept {
      ++callback_count;
      credits_seen     = coordinator->tasks_remaining();
      callback_success = success;
      if (chunks.size() == 1) observed = chunks.front();
    });

  auto op                 = std::make_unique<uring_io_op>();
  op->request.coordinator = coordinator;
  op->request.on_complete = completion;
  op->request.completion_chunks.push_back(&chunk);

  op->request.finish_success();
  op->request.finish_success();

  CHECK(callback_count == 1);
  CHECK(credits_seen == 1);
  CHECK(callback_success);
  CHECK(observed == &chunk);
  CHECK(std::move(future).get() == IO_BLOCK_SIZE);
}

TEST_CASE("io_uring physical expansion publishes cache fragments independently", "[uring_readv]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(2 * IO_BLOCK_SIZE, 1);
  auto future      = coordinator->get_future();
  coordinator->add_tasks(1);

  cached_chunk first{0};
  cached_chunk second{IO_BLOCK_SIZE};
  std::array<cached_chunk*, 2> published{};
  std::size_t published_count = 0;
  bool callbacks_valid        = true;

  auto completion = std::make_shared<prepared_io_completion>(
    [&](std::span<cached_chunk* const> chunks, bool success) noexcept {
      callbacks_valid =
        callbacks_valid && success && chunks.size() == 1 && published_count < published.size();
      if (chunks.size() == 1 && published_count < published.size()) {
        published[published_count++] = chunks.front();
      }
    });

  auto make_op = [&](cached_chunk* chunk) {
    auto op                 = std::make_unique<uring_io_op>();
    op->request.coordinator = coordinator;
    op->request.on_complete = completion;
    op->request.completion_chunks.push_back(chunk);
    return op;
  };

  auto first_op  = make_op(&first);
  auto second_op = make_op(&second);
  first_op->request.finish_success();

  CHECK(coordinator->tasks_remaining() == 1);
  REQUIRE(published_count == 1);
  CHECK(published.front() == &first);

  second_op->request.finish_success();
  CHECK(callbacks_valid);
  REQUIRE(published_count == 2);
  CHECK(published.back() == &second);
  CHECK(std::move(future).get() == 2 * IO_BLOCK_SIZE);
}

TEST_CASE("io_uring CUDA-stage failure may publish valid host data before error", "[uring_readv]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(IO_BLOCK_SIZE, 1);
  auto future      = coordinator->get_future();
  bool host_valid  = false;

  auto completion = std::make_shared<prepared_io_completion>(
    [&](std::span<cached_chunk* const>, bool success) noexcept { host_valid = success; });

  auto op                 = std::make_unique<uring_io_op>();
  op->request.coordinator = coordinator;
  op->request.on_complete = completion;
  op->request.finish_error(cudaErrorInvalidValue, true);

  CHECK(host_valid);
  CHECK_THROWS(std::move(future).get());
}

TEST_CASE("io_uring alignment widens and coalesces safely", "[uring_readv]")
{
  using range_info = cudf::io::text::byte_range_info;

  auto physical = uring_reactor::align_to_physical(range_info{5000, 9000}, 20'000);
  CHECK(physical.offset() == 4096);
  CHECK(physical.size() == 12'288);

  std::array<range_info, 4> input{
    range_info{8193, 100}, range_info{1, 100}, range_info{4095, 2}, range_info{0, 0}};
  auto merged = uring_reactor::align_and_coalesce(input);

  REQUIRE(merged.size() == 1);
  CHECK(merged[0].offset() == 0);
  CHECK(merged[0].size() == 12'288);
}
