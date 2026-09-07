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

#include "exec/semi_future.hpp"
#include "io/io_context.hpp"
#include "io/io_request.hpp"
#include "io/types.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::io {

namespace detail {

[[nodiscard]] inline int current_cuda_device()
{
  int id          = -1;
  cudaError_t err = cudaGetDevice(&id);
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("templated_ioctx: cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }
  return id;
}

}  // namespace detail

/**
 * @brief Contract implemented by worker-driven reactors.
 *
 * Logical prepared slices are the only asynchronous input. A reactor owns all
 * backend chunking, alignment, staging allocation, and physical-operation
 * lifecycle after enqueue.
 */
template <class R>
concept io_reactor_c = requires(R reactor,
                                typename R::io_object_type object,
                                std::unique_ptr<grouped_io_request> request,
                                std::size_t offset,
                                std::size_t size,
                                std::uint8_t* destination,
                                std::string path,
                                std::string_view path_view) {
  typename R::io_object_type;
  typename R::reactor_config_type;

  { reactor.get_config() } -> std::same_as<typename R::reactor_config_type const&>;
  { reactor.enqueue(std::move(request)) } noexcept;
  { reactor.queued_bytes() } noexcept -> std::convertible_to<std::size_t>;
  { reactor.host_read(object, offset, size, destination) } -> std::same_as<std::size_t>;
  { reactor.start() };
  { reactor.shutdown() };
  { reactor.interrupt() };
  {
    R::create_io_object(std::move(path))
  } -> std::same_as<std::unique_ptr<typename R::io_object_type>>;
  { R::supports(path_view) } -> std::same_as<bool>;
  {
    R::align_and_coalesce(std::span<cudf::io::text::byte_range_info const>{},
                          std::optional<std::size_t>{})
  } -> std::same_as<std::vector<cudf::io::text::byte_range_info>>;
};

template <class R>
concept reactor_declares_bulk_io_preference = requires {
  { R::prefers_bulk_io } -> std::convertible_to<bool>;
};

template <class R>
struct reactor_traits {
  // Every reactor satisfying the new contract accepts mixed prepared slices.
  static constexpr bool supports_device_read         = true;
  static constexpr bool supports_host_to_device_read = true;
  static constexpr bool supports_vector_host_read    = true;
  static constexpr bool supports_device_range_read   = true;
  static constexpr bool prefers_bulk_io              = [] {
    if constexpr (reactor_declares_bulk_io_preference<R>) {
      return static_cast<bool>(R::prefers_bulk_io);
    } else {
      return false;
    }
  }();
};

template <io_reactor_c Reactor>
class templated_ioctx : public ioctx {
 public:
  using reactor_type        = Reactor;
  using io_object_type      = typename Reactor::io_object_type;
  using reactor_config_type = typename Reactor::reactor_config_type;
  using reactor_traits_t    = reactor_traits<Reactor>;

  enum class io_op_type { host, host_async, device_async, host_vector_async };

  explicit templated_ioctx(std::vector<std::unique_ptr<Reactor>> reactors)
    : _reactors(std::move(reactors))
  {
    if (!_reactors.empty()) _config = _reactors.front()->get_config();
  }

  template <class Factory>
    requires std::invocable<Factory&> &&
             std::convertible_to<std::invoke_result_t<Factory&>, std::unique_ptr<Reactor>>
  templated_ioctx(std::size_t n_reactors, Factory factory)
  {
    _reactors.reserve(n_reactors);
    for (std::size_t i = 0; i < n_reactors; ++i) {
      _reactors.emplace_back(factory());
    }
    if (!_reactors.empty()) _config = _reactors.front()->get_config();
  }

  ~templated_ioctx() override
  {
    this->pre_destroy();
    shutdown();
  }

  void start() override
  {
    if (_started) return;
    for (auto& reactor : _reactors) {
      reactor->start();
    }
    _started = true;
  }

  void shutdown() noexcept override
  {
    for (auto& reactor : _reactors) {
      try {
        reactor->shutdown();
      } catch (std::exception const& error) {
        SIRIUS_LOG_ERROR("templated_ioctx: reactor shutdown failed: {}", error.what());
      } catch (...) {
        SIRIUS_LOG_ERROR("templated_ioctx: reactor shutdown failed: unknown error");
      }
    }
  }

  [[nodiscard]] bool supports(std::string_view path) const noexcept final
  {
    return Reactor::supports(path);
  }

  [[nodiscard]] bool supports_device_read() const noexcept final
  {
    return reactor_traits_t::supports_device_read;
  }

  [[nodiscard]] bool supports_host_to_device_read() const noexcept final
  {
    return reactor_traits_t::supports_host_to_device_read;
  }

  [[nodiscard]] bool supports_vector_host_read() const noexcept final
  {
    return reactor_traits_t::supports_vector_host_read;
  }

  [[nodiscard]] bool supports_device_range_read() const noexcept final
  {
    return reactor_traits_t::supports_device_range_read;
  }

  [[nodiscard]] bool prefers_bulk_io() const noexcept final
  {
    return reactor_traits_t::prefers_bulk_io;
  }

  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept final
  {
    return _config.min_alignment_requirement();
  }

  [[nodiscard]] std::size_t merge_gap_size() const noexcept final
  {
    return _config.merge_gap_size();
  }

  [[nodiscard]] std::size_t n_max_concurrent_scans() const noexcept final
  {
    return _config.n_max_concurrent_scans;
  }

  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges,
    std::optional<std::size_t> alignment = std::nullopt) const noexcept override
  {
    return Reactor::align_and_coalesce(ranges, alignment);
  }

  /**
   * @brief Select at most two least-backlogged reactors.
   *
   * Rotation breaks load ties and keeps synchronous reads from sticking to
   * reactor zero. queued_bytes is advisory; slot limits remain authoritative.
   */
  virtual std::vector<Reactor*> next_reactor([[maybe_unused]] io_object_type const& object,
                                             [[maybe_unused]] std::size_t n_slices,
                                             [[maybe_unused]] io_op_type type,
                                             [[maybe_unused]] int device_id = -1)
  {
    constexpr std::size_t dispatch_fanout = 2;
    auto const count                      = _reactors.size();
    if (count == 0) return {};

    auto const start = _next.fetch_add(1, std::memory_order_relaxed) % count;
    std::vector<std::pair<std::size_t, std::size_t>> ranked;
    ranked.reserve(count);
    for (std::size_t distance = 0; distance < count; ++distance) {
      auto const index = (start + distance) % count;
      ranked.emplace_back(_reactors[index]->queued_bytes(), distance);
    }

    auto const selected = std::min(count, dispatch_fanout);
    std::partial_sort(ranked.begin(), ranked.begin() + selected, ranked.end());

    std::vector<Reactor*> result;
    result.reserve(selected);
    for (std::size_t i = 0; i < selected; ++i) {
      result.push_back(_reactors[(start + ranked[i].second) % count].get());
    }
    return result;
  }

  std::size_t host_read_io(io_object const& object,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* destination) override
  {
    auto const& typed = as_typed(object);
    size = std::min(size, typed.size() > offset ? typed.size() - offset : std::size_t{0});
    if (size == 0) return 0;

    auto reactors = next_reactor(typed, 1, io_op_type::host);
    if (reactors.empty()) throw std::runtime_error("host_read_io: no available reactors");
    return reactors.front()->host_read(typed, offset, size, destination);
  }

  exec::semi_future<std::size_t> mixed_readv_async_io(
    io_object const& object, std::vector<prepared_io_slice>&& input_slices) noexcept override
  {
    if (input_slices.empty()) return exec::make_semi_future<std::size_t>(0);

    std::vector<prepared_io_slice> slices;
    try {
      auto const& typed = as_typed(object);
      auto owner        = object.shared_from_this();

      slices.reserve(input_slices.size());
      std::size_t total_bytes = 0;
      int device_id           = -1;

      for (auto& slice : input_slices) {
        if (slice.rng.size == 0 || slice.rng.offset >= typed.size()) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), true);
            slice.on_complete.reset();
          }
          continue;
        }

        slice.rng.size = std::min(slice.rng.size, typed.size() - slice.rng.offset);
        if (slice.rng.size > std::numeric_limits<std::size_t>::max() - total_bytes) {
          throw std::overflow_error("mixed read byte count overflow");
        }
        total_bytes += slice.rng.size;

        if (slice.has_device_request()) {
          if (device_id < 0) device_id = detail::current_cuda_device();
          if (slice.d_buffer.device_id < 0) slice.d_buffer.device_id = device_id;
        }
        slices.push_back(std::move(slice));
      }

      if (slices.empty()) return exec::make_semi_future<std::size_t>(0);

      auto reactors = next_reactor(typed, slices.size(), io_op_type::host_vector_async, device_id);
      auto coordinator = std::make_shared<grouped_coordinator>(total_bytes, slices.size());
      auto future      = coordinator->get_future();

      if (reactors.empty()) {
        auto error = std::make_exception_ptr(
          std::runtime_error("mixed_readv_async_io: no available reactors"));
        for (auto& slice : slices) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
          }
          coordinator->report_error(error);
        }
        return future;
      }

      try {
        auto const partition_count = std::min(reactors.size(), slices.size());
        std::vector<std::vector<prepared_io_slice>> partitions(partition_count);
        std::vector<std::size_t> partition_bytes(partition_count, 0);

        // Keep the originals until every queue entry has been allocated. If an
        // allocation fails, their callbacks can still release all claimed cache
        // chunks and every coordinator credit can be settled.
        for (auto const& slice : slices) {
          auto const smallest = static_cast<std::size_t>(
            std::min_element(partition_bytes.begin(), partition_bytes.end()) -
            partition_bytes.begin());
          partition_bytes[smallest] += slice.size();
          partitions[smallest].push_back(slice);
        }

        std::vector<std::unique_ptr<grouped_io_request>> requests;
        requests.reserve(partition_count);
        for (std::size_t i = 0; i < partition_count; ++i) {
          requests.push_back(
            grouped_io_request::create(owner, std::move(partitions[i]), coordinator));
        }

        // enqueue is noexcept by reactor contract, so once publication starts
        // ownership cannot be stranded between reactors.
        for (std::size_t i = 0; i < partition_count; ++i) {
          reactors[i]->enqueue(std::move(requests[i]));
        }
        return future;
      } catch (...) {
        auto const error = std::current_exception();
        for (auto& slice : slices) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
          }
          coordinator->report_error(error);
        }
        return future;
      }
    } catch (...) {
      auto const error      = std::current_exception();
      auto fail_unsubmitted = [](auto& pending) noexcept {
        for (auto& slice : pending) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
            slice.on_complete.reset();
          }
        }
      };
      fail_unsubmitted(slices);
      fail_unsubmitted(input_slices);
      return exec::make_semi_future<std::size_t>(error);
    }
  }

 protected:
  std::shared_ptr<io_object> create_io_object(std::string path) override
  {
    return std::shared_ptr<io_object>(Reactor::create_io_object(std::move(path)));
  }

  reactor_config_type _config{};
  std::vector<std::unique_ptr<Reactor>> _reactors;
  std::atomic<std::size_t> _next{0};
  bool _started{false};

 private:
  static io_object_type const& as_typed(io_object const& object)
  {
    auto const* typed = dynamic_cast<io_object_type const*>(&object);
    if (typed == nullptr) throw std::invalid_argument("I/O object belongs to another backend");
    return *typed;
  }
};

}  // namespace sirius::io
