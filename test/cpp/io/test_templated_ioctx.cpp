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
#include "io/templated_ioctx.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct fake_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{0};
};

class fake_object final : public sirius::io::io_object {
 public:
  fake_object(std::string path, std::size_t size) : _path(std::move(path)), _size(size) {}

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _path;
  std::size_t _size;
};

class fake_reactor {
 public:
  using io_object_type                  = fake_object;
  using reactor_config_type             = fake_config;
  static constexpr bool prefers_bulk_io = false;

  explicit fake_reactor(std::size_t backlog = 0) : backlog(backlog) {}

  [[nodiscard]] fake_config const& get_config() const noexcept { return config; }

  void enqueue(std::unique_ptr<sirius::io::grouped_io_request> request) noexcept
  {
    requests.push_back(std::move(request));
  }

  [[nodiscard]] std::size_t queued_bytes() const noexcept { return backlog; }

  std::size_t host_read(fake_object const&, std::size_t, std::size_t size, std::uint8_t*) const
  {
    return size;
  }

  void start() {}
  void shutdown() {}
  void interrupt() {}

  [[nodiscard]] static std::unique_ptr<fake_object> create_io_object(std::string path)
  {
    return std::make_unique<fake_object>(std::move(path), 4096);
  }

  [[nodiscard]] static bool supports(std::string_view) { return true; }

  [[nodiscard]] static std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

  fake_config config;
  std::size_t backlog;
  std::vector<std::unique_ptr<sirius::io::grouped_io_request>> requests;
};

class fake_context final : public sirius::io::templated_ioctx<fake_reactor> {
 public:
  using templated_ioctx::templated_ioctx;

  [[nodiscard]] sirius::io::io_context_type type() const noexcept override
  {
    return sirius::io::io_context_type::uring;
  }
};

void complete_request(sirius::io::grouped_io_request& request)
{
  while (!request.empty()) {
    static_cast<void>(request.take_front());
    request.coordinator->on_complete();
  }
}

}  // namespace

TEST_CASE("mixed dispatch selects two least-busy reactors and shares one coordinator",
          "[io][ioctx]")
{
  std::vector<std::unique_ptr<fake_reactor>> reactors;
  std::vector<fake_reactor*> raw;
  for (auto const backlog : {1000U, 10U, 20U, 500U}) {
    reactors.push_back(std::make_unique<fake_reactor>(backlog));
    raw.push_back(reactors.back().get());
  }
  fake_context context{std::move(reactors)};
  auto object = std::make_shared<fake_object>("fake", 4096);

  std::uint8_t byte{};
  std::vector<sirius::io::prepared_io_slice> slices;
  slices.emplace_back(sirius::io::range{0, 60}, sirius::io::host_buffer{&byte});
  slices.emplace_back(sirius::io::range{100, 40}, sirius::io::host_buffer{&byte});
  slices.emplace_back(sirius::io::range{200, 20}, sirius::io::host_buffer{&byte});

  auto future = context.mixed_readv_async_io(*object, std::move(slices));

  CHECK(raw[0]->requests.empty());
  REQUIRE(raw[1]->requests.size() == 1);
  REQUIRE(raw[2]->requests.size() == 1);
  CHECK(raw[3]->requests.empty());

  auto& first  = *raw[1]->requests.front();
  auto& second = *raw[2]->requests.front();
  CHECK(first.coordinator == second.coordinator);
  CHECK(first.remaining_bytes() == 60);
  CHECK(second.remaining_bytes() == 60);
  CHECK_FALSE(future.is_ready());

  complete_request(first);
  CHECK_FALSE(future.is_ready());
  complete_request(second);

  CHECK(future.is_ready());
  CHECK(std::move(future).get() == 120);
}

TEST_CASE("mixed dispatch with no reactors fails and releases prepared cache slices", "[io][ioctx]")
{
  fake_context context{std::vector<std::unique_ptr<fake_reactor>>{}};
  auto object = std::make_shared<fake_object>("fake", 4096);
  std::atomic<bool> callback_success{true};

  auto completion = std::make_shared<sirius::io::prepared_io_completion>(
    [&callback_success](std::span<sirius::io::cache::cached_chunk* const>, bool success) noexcept {
      callback_success.store(success, std::memory_order_release);
    });

  std::uint8_t byte{};
  sirius::io::prepared_io_slice slice{sirius::io::range{0, 64}, sirius::io::host_buffer{&byte}};
  slice.on_complete = completion;
  std::vector<sirius::io::prepared_io_slice> slices;
  slices.push_back(std::move(slice));

  auto future = context.mixed_readv_async_io(*object, std::move(slices));

  CHECK_FALSE(callback_success.load(std::memory_order_acquire));
  CHECK(future.is_ready());
  CHECK_THROWS_WITH(std::move(future).get(), "mixed_readv_async_io: no available reactors");
}
