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

#pragma once

// A backend whose IO the test holds, completes and fails by hand, for tests of the readahead's
// ticket protocol against a real fs_cache.  A Sirius-side copy of cuCascade's
// test/io/cache/controlled_reactor.hpp (not on the Sirius include path), plus the
// `after_enqueue` seam the ticket tests need: it runs on the issuing thread right after a request
// was queued, so a test can pause issue before its fan-in guard drops, or complete the request
// inline.

#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::test::controlled_io {

/// Matches the cache chunk size of a default `cache.mode: cucs` configuration, which
/// ioctx::initialize_cache checks against the reactor's staging block.
inline constexpr std::size_t chunk_size = std::size_t{1} << 20;

/// A backend-neutral object for the held backend: keyed by its path like a local file, with an
/// optional @p version reported as its validation tag and the base @c io_object identity (a
/// version is an object tag; without one the identity is scoped to this open, so independent opens
/// share no cached bytes).  The REST object type only carries it to the backend; none of its ETag
/// rules apply.
class controlled_object final : public cucascade::io::rest::rest_io_object {
 public:
  explicit controlled_object(std::string path,
                             std::size_t bytes   = 4 * chunk_size,
                             std::string version = {})
    : rest_io_object(std::move(path), "controlled", "object.bin", bytes),
      _version(std::move(version))
  {
  }

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override
  {
    return object_path();
  }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _version; }
  [[nodiscard]] cucascade::io::object_identity identity() const override
  {
    return io_object::identity();
  }

 private:
  std::string _version;
};

struct controlled_reactor_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{1};
  std::size_t staging_bytes{chunk_size};
};

class controlled_reactor {
 public:
  using io_object_type                  = cucascade::io::rest::rest_io_object;
  using reactor_config_type             = controlled_reactor_config;
  static constexpr bool prefers_bulk_io = false;

  explicit controlled_reactor(controlled_reactor_config config = {}) : _config(config) {}
  ~controlled_reactor() { shutdown(); }

  [[nodiscard]] reactor_config_type const& get_config() const noexcept { return _config; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return _config.staging_bytes; }
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return 0; }

  void enqueue(std::unique_ptr<cucascade::io::grouped_io_request> request) noexcept
  {
    if (!request) { return; }
    bool queued = false;
    {
      std::scoped_lock lock{_mutex};
      if (!_stopped) {
        _requests.push_back(std::move(request));
        queued = true;
      }
    }
    _cv.notify_all();
    if (!queued) {
      request->cancel_remaining(_cancel_error);
      return;
    }
    if (after_enqueue) { after_enqueue(); }
  }

  /// Configured before the issuing threads start, reset only after they stopped.
  std::function<void()> after_enqueue;

  /// The oldest queued request, waiting up to @p timeout for one; null when none arrived.
  [[nodiscard]] std::unique_ptr<cucascade::io::grouped_io_request> take_next(
    std::chrono::milliseconds timeout = std::chrono::seconds(2))
  {
    std::unique_lock lock{_mutex};
    _cv.wait_for(lock, timeout, [&] { return _stopped || !_requests.empty(); });
    if (_requests.empty()) { return {}; }
    auto result = std::move(_requests.front());
    _requests.pop_front();
    return result;
  }

  [[nodiscard]] std::size_t pending() const
  {
    std::scoped_lock lock{_mutex};
    return _requests.size();
  }

  std::size_t host_read(io_object_type const&, std::size_t, std::size_t, std::uint8_t*)
  {
    throw std::runtime_error("the controlled backend serves only queued reads");
  }

  void start() {}
  void interrupt() { _cv.notify_all(); }
  void shutdown() noexcept
  {
    std::deque<std::unique_ptr<cucascade::io::grouped_io_request>> abandoned;
    {
      std::scoped_lock lock{_mutex};
      _stopped = true;
      abandoned.swap(_requests);
    }
    _cv.notify_all();
    for (auto& request : abandoned) {
      request->cancel_remaining(_cancel_error);
    }
  }

  static std::unique_ptr<io_object_type> create_io_object(std::string path)
  {
    return std::make_unique<controlled_object>(std::move(path));
  }
  static bool supports(std::string_view) { return true; }
  static std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

 private:
  controlled_reactor_config _config;
  std::exception_ptr _cancel_error{std::make_exception_ptr(std::runtime_error("test shutdown"))};
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::unique_ptr<cucascade::io::grouped_io_request>> _requests;
  bool _stopped{false};
};

static_assert(cucascade::io::io_reactor_c<controlled_reactor>);

/// One reactor; reports itself as a REST backend (any type with a host cache works).
class controlled_ioctx final : public cucascade::io::templated_ioctx<controlled_reactor> {
 public:
  explicit controlled_ioctx(controlled_reactor_config config = {})
    : templated_ioctx(1, [config] { return std::make_unique<controlled_reactor>(config); })
  {
  }
  ~controlled_ioctx() override
  {
    pre_destroy();
    shutdown();
  }

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::restful;
  }

  [[nodiscard]] controlled_reactor& reactor() noexcept { return *_reactors.front(); }
};

/// Fill every slice of @p request with @p value and complete it successfully.
inline void complete_success(cucascade::io::grouped_io_request& request, std::uint8_t value = 0x5a)
{
  while (!request.empty()) {
    auto slice = request.take_front();
    if (slice.h_buffer.is_contiguous()) {
      std::fill_n(std::get<std::uint8_t*>(slice.h_buffer.buffer), slice.size(), value);
    } else {
      for (auto* chunk : slice.h_buffer.fragments()) {
        auto const begin         = std::max(slice.offset(), chunk->offset);
        auto const source_offset = begin - slice.offset();
        auto const chunk_offset  = begin - chunk->offset;
        if (source_offset >= slice.size() || chunk_offset >= chunk_size) { continue; }
        auto const bytes = std::min(slice.size() - source_offset, chunk_size - chunk_offset);
        std::fill_n(chunk->data + chunk_offset, bytes, value);
      }
    }
    if (slice.on_complete) { (*slice.on_complete)(slice.h_buffer.fragments(), true); }
    request.coordinator->on_complete();
  }
}

/// Fail every slice of @p request with an IO error.
inline void complete_failure(cucascade::io::grouped_io_request& request)
{
  while (!request.empty()) {
    auto slice = request.take_front();
    if (slice.on_complete) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    request.coordinator->report_error(std::make_error_code(std::errc::io_error));
  }
}

/// One request taken off the held backend.  Pending test IO must settle even when an assertion
/// unwinds the test, so a request still held at destruction fails.
struct held_request {
  std::unique_ptr<cucascade::io::grouped_io_request> request;

  ~held_request()
  {
    if (request) { complete_failure(*request); }
  }
  explicit operator bool() const { return request != nullptr; }
  cucascade::io::grouped_io_request& operator*() const { return *request; }
};

}  // namespace sirius::test::controlled_io
