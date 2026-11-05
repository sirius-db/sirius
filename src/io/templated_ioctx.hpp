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
#include "io/reactor_placement.hpp"
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
  { reactor.staging_block_size() } noexcept -> std::convertible_to<std::size_t>;
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

  explicit templated_ioctx(std::vector<std::unique_ptr<Reactor>> reactors,
                           device_reactor_indices devices = {})
    : _reactors(std::move(reactors)), _device_reactors(std::move(devices))
  {
    if (!_reactors.empty()) _config = _reactors.front()->get_config();
    for (auto const& [device, indices] : _device_reactors) {
      if (indices.empty()) throw std::invalid_argument("device has no I/O reactors");
      for (auto index : indices) {
        if (index >= _reactors.size()) throw std::invalid_argument("invalid I/O reactor index");
      }
    }
    // The cache contract exposes one staging block size per ioctx.
    for (auto const& reactor : _reactors) {
      if (reactor->staging_block_size() != staging_block_size()) {
        throw std::invalid_argument("I/O host pools must have equal staging block sizes");
      }
    }
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

  /// NUMA-local contexts must agree on the staging block size (checked at construction).
  [[nodiscard]] std::size_t staging_block_size() const noexcept final
  {
    return _reactors.empty() ? 0 : _reactors.front()->staging_block_size();
  }

  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges,
    std::optional<std::size_t> alignment = std::nullopt) const noexcept override
  {
    return Reactor::align_and_coalesce(ranges, alignment);
  }

  /**
   * @brief Select at most two least-backlogged reactors in the device's staging group.
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
    auto const locality                   = _device_reactors.find(device_id);
    auto const* indices = locality == _device_reactors.end() ? nullptr : &locality->second;
    auto const count    = indices ? indices->size() : _reactors.size();
    if (count == 0) return {};
    auto const reactor_index = [&](std::size_t i) { return indices ? (*indices)[i] : i; };
    auto const start         = _next.fetch_add(1, std::memory_order_relaxed) % count;
    std::vector<std::pair<std::size_t, std::size_t>> ranked;
    ranked.reserve(count);
    for (std::size_t distance = 0; distance < count; ++distance) {
      auto const index = reactor_index((start + distance) % count);
      ranked.emplace_back(_reactors[index]->queued_bytes(), distance);
    }
    auto const selected = std::min(count, dispatch_fanout);
    std::partial_sort(ranked.begin(), ranked.begin() + selected, ranked.end());
    std::vector<Reactor*> result;
    result.reserve(selected);
    for (std::size_t i = 0; i < selected; ++i) {
      result.push_back(_reactors[reactor_index((start + ranked[i].second) % count)].get());
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
        // A dropped slice reads nothing, so its completion always reports failure: reporting
        // success would publish never-filled cache chunks as `cached`.
        if (slice.rng.size == 0 || slice.rng.offset >= typed.size()) {
          if (slice.is_fragmented()) {
            SIRIUS_LOG_WARN(
              "templated_ioctx: dropping fragmented slice [{}, +{}) outside object '{}' of {} "
              "bytes; caller must clamp to EOF",
              slice.rng.offset,
              slice.rng.size,
              typed.object_path(),
              typed.size());
          }
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
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
          if (slice.d_buffer.device_id < 0) {
            if (device_id < 0) device_id = detail::current_cuda_device();
            slice.d_buffer.device_id = device_id;
          }
        }
        slices.push_back(std::move(slice));
      }

      if (slices.empty()) return exec::make_semi_future<std::size_t>(0);

      auto coordinator = std::make_shared<grouped_coordinator>(total_bytes, slices.size());
      auto future      = coordinator->get_future();
      try {
        // A mixed request can name several GPUs. Resolve and balance each device's
        // slices separately, retaining one coordinator and atomic publication.
        std::unordered_map<int, std::vector<Reactor*>> choices;
        std::vector<Reactor*> reactors;
        std::vector<std::vector<prepared_io_slice>> partitions;
        std::vector<std::size_t> partition_bytes;
        for (auto const& slice : slices) {
          int const target    = slice.has_device_request() ? slice.d_buffer.device_id : -1;
          auto [it, inserted] = choices.try_emplace(target);
          if (inserted) {
            it->second = next_reactor(typed, slices.size(), io_op_type::host_vector_async, target);
          }
          if (it->second.empty()) {
            throw std::runtime_error("mixed_readv_async_io: no available reactors");
          }
          std::size_t smallest = 0;
          std::size_t least    = std::numeric_limits<std::size_t>::max();
          for (auto* reactor : it->second) {
            auto found = std::find(reactors.begin(), reactors.end(), reactor);
            auto index = std::size_t(found - reactors.begin());
            if (found == reactors.end()) {
              reactors.push_back(reactor);
              partitions.emplace_back();
              partition_bytes.push_back(0);
            }
            if (partition_bytes[index] < least) {
              smallest = index;
              least    = partition_bytes[index];
            }
          }
          partition_bytes[smallest] += slice.size();
          partitions[smallest].push_back(slice);
        }
        std::vector<std::unique_ptr<grouped_io_request>> requests(reactors.size());
        for (std::size_t i = 0; i < reactors.size(); ++i) {
          if (!partitions[i].empty()) {
            requests[i] = grouped_io_request::create(owner, std::move(partitions[i]), coordinator);
          }
        }

        // enqueue is noexcept by reactor contract, so once publication starts
        // ownership cannot be stranded between reactors.
        for (std::size_t i = 0; i < reactors.size(); ++i) {
          if (requests[i]) reactors[i]->enqueue(std::move(requests[i]));
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
  device_reactor_indices _device_reactors;
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
