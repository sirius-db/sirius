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

#include "cuda/device_copy_batch.hpp"
#include "exec/semi_future.hpp"
#include "io/types.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <sys/uio.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::io {

/**
 * @brief Fan-in shared by every physical operation derived from one logical read.
 *
 * The initial task count is one credit per prepared slice. Before a reactor
 * expands a slice into N physical operations it adds N-1 credits, then every
 * success, failure, or cancellation settles exactly one credit. The first
 * error stops further dispatch immediately, but the future is not fulfilled
 * until all already-published operations have drained.
 */
class grouped_coordinator final {
 public:
  using error_type = std::variant<std::exception_ptr, cudaError_t, std::error_code>;

  grouped_coordinator(std::size_t bytes_requested, std::size_t initial_tasks)
    : _bytes_requested(bytes_requested), _tasks_remaining(initial_tasks)
  {
  }

  grouped_coordinator(grouped_coordinator const&)            = delete;
  grouped_coordinator& operator=(grouped_coordinator const&) = delete;

  [[nodiscard]] bool should_continue() const noexcept
  {
    return _continue.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool has_error() const noexcept { return !should_continue(); }

  [[nodiscard]] std::size_t bytes_requested() const noexcept { return _bytes_requested; }

  [[nodiscard]] std::size_t tasks_remaining() const noexcept
  {
    return _tasks_remaining.load(std::memory_order_acquire);
  }

  /**
   * @brief Add credits for a slice expansion.
   *
   * This must happen before any derived operation can become visible to a
   * worker. The original slice credit keeps the count non-zero while it is
   * being expanded.
   */
  void add_tasks(std::size_t count) noexcept
  {
    if (count == 0) return;

    auto current = _tasks_remaining.load(std::memory_order_acquire);
    while (current != 0) {
      assert(current <= std::numeric_limits<std::size_t>::max() - count);
      if (_tasks_remaining.compare_exchange_weak(
            current, current + count, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
    assert(false && "cannot expand a completed grouped I/O request");
  }

  void on_complete() noexcept { settle_one(); }

  void report_error(error_type const& error,
                    std::source_location loc = std::source_location::current()) noexcept
  {
    bool expected = true;
    if (_continue.compare_exchange_strong(
          expected, false, std::memory_order_acq_rel, std::memory_order_acquire)) {
      std::exception_ptr converted;
      try {
        converted = to_exception_ptr(error, loc);
        if (converted == nullptr) {
          converted = std::make_exception_ptr(std::runtime_error("unknown I/O error"));
        }
      } catch (...) {
        converted = std::current_exception();
      }

      std::lock_guard lock(_state_mutex);
      assert(_first_exception == nullptr);
      _first_exception = std::move(converted);
    }
    settle_one();
  }
  [[nodiscard]] exec::semi_future<std::size_t> get_future()
  {
    exec::semi_future<std::size_t> future;
    {
      std::lock_guard lock(_state_mutex);
      assert(!_future_taken && "grouped coordinator future may only be retrieved once");
      future        = _promise.get_semi_future();
      _future_taken = true;
    }
    resolve_if_ready();
    return future;
  }

 private:
  [[nodiscard]] static std::exception_ptr to_exception_ptr(error_type const& error,
                                                           std::source_location loc)
  {
    if (std::holds_alternative<std::exception_ptr>(error)) {
      return std::get<std::exception_ptr>(error);
    }
    if (std::holds_alternative<cudaError_t>(error)) {
      auto const value = std::get<cudaError_t>(error);
      return std::make_exception_ptr(
        std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(value)) + " at " +
                           loc.file_name() + ":" + std::to_string(loc.line())));
    }

    auto const value = std::get<std::error_code>(error);
    return std::make_exception_ptr(std::system_error(
      value, "System error at " + std::string(loc.file_name()) + ":" + std::to_string(loc.line())));
  }

  void settle_one() noexcept
  {
    auto current = _tasks_remaining.load(std::memory_order_acquire);
    while (current != 0) {
      if (_tasks_remaining.compare_exchange_weak(
            current, current - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        if (current == 1) resolve_if_ready();
        return;
      }
    }
    assert(false && "a grouped I/O task was completed more than once");
  }

  void resolve_if_ready() noexcept
  {
    std::exception_ptr error;
    {
      std::lock_guard lock(_state_mutex);
      if (_tasks_remaining.load(std::memory_order_acquire) != 0 || !_future_taken || _fulfilled) {
        return;
      }
      _fulfilled = true;
      error      = _first_exception;
    }

    if (error != nullptr) {
      _promise.set_exception(std::move(error));
    } else {
      _promise.set_value(_bytes_requested);
    }
  }

  std::size_t const _bytes_requested;
  std::atomic<std::size_t> _tasks_remaining;
  std::atomic<bool> _continue{true};

  mutable std::mutex _state_mutex;
  std::exception_ptr _first_exception;
  bool _future_taken{false};
  bool _fulfilled{false};
  exec::promise<std::size_t> _promise;
};

/**
 * @brief A queue entry containing logical slices and their shared fan-in.
 *
 * The object and fragment-pointer arrays are owned for the full asynchronous
 * lifetime. Reactors keep an active request locally and consume its slices in
 * order; they do not explode the group into queue entries before slot capacity
 * is known.
 */
class grouped_io_request final {
 public:
  static std::unique_ptr<grouped_io_request> create(
    std::shared_ptr<const io_object> object,
    std::vector<prepared_io_slice> slices,
    std::shared_ptr<grouped_coordinator> coordinator)
  {
    if (object == nullptr || coordinator == nullptr) {
      throw std::invalid_argument("grouped_io_request requires an object and coordinator");
    }
    return std::unique_ptr<grouped_io_request>(
      new grouped_io_request(std::move(object), std::move(slices), std::move(coordinator)));
  }

  static std::unique_ptr<grouped_io_request> create(std::shared_ptr<const io_object> object,
                                                    std::vector<prepared_io_slice> slices)
  {
    std::size_t bytes = 0;
    for (auto const& slice : slices) {
      if (slice.size() > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error("grouped I/O byte count overflow");
      }
      bytes += slice.size();
    }
    auto coordinator = std::make_shared<grouped_coordinator>(bytes, slices.size());
    return create(std::move(object), std::move(slices), std::move(coordinator));
  }

  [[nodiscard]] bool empty() const noexcept { return _next == slices.size(); }

  [[nodiscard]] std::size_t remaining_slices() const noexcept { return slices.size() - _next; }

  [[nodiscard]] std::size_t remaining_bytes() const noexcept
  {
    std::size_t bytes = 0;
    for (std::size_t i = _next; i < slices.size(); ++i) {
      bytes += slices[i].size();
    }
    return bytes;
  }

  [[nodiscard]] prepared_io_slice& front() noexcept
  {
    assert(!empty());
    return slices[_next];
  }

  [[nodiscard]] prepared_io_slice take_front() noexcept
  {
    assert(!empty());
    return std::move(slices[_next++]);
  }

  void cancel_remaining(grouped_coordinator::error_type const& error) noexcept
  {
    while (!empty()) {
      auto slice = take_front();
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
      coordinator->report_error(error);
    }
  }

  std::shared_ptr<const io_object> obj;
  std::vector<prepared_io_slice> slices;
  std::shared_ptr<grouped_coordinator> coordinator;

 private:
  grouped_io_request(std::shared_ptr<const io_object> object,
                     std::vector<prepared_io_slice> request_slices,
                     std::shared_ptr<grouped_coordinator> group)
    : obj(std::move(object)), slices(std::move(request_slices)), coordinator(std::move(group))
  {
  }

  std::size_t _next{0};
};

struct device_cpy_request {
  range req_rng;
  device_buffer d_buffer;
  int device_id{-1};

  /**
   * @brief Copy the logical request window out of physical I/O buffers.
   *
   * @p host_buf represents all bytes in @p io_rng in order. Aligned physical
   * over-read is skipped, fragmented sources are batched, and the optional
   * event is recorded after the final copy on the destination stream.
   */
  [[nodiscard]] cudaError_t copy_async(range io_rng,
                                       std::span<iovec const> host_buf,
                                       cudaEvent_t event = nullptr) const noexcept
  {
    try {
      if (d_buffer.data == nullptr) return cudaErrorInvalidValue;

      auto const copy_rng = intersect(req_rng, io_rng);
      if (copy_rng.empty()) return req_rng.empty() ? cudaSuccess : cudaErrorInvalidValue;

      int target_device = device_id >= 0 ? device_id : d_buffer.device_id;
      if (target_device < 0) {
        auto const status = cudaGetDevice(&target_device);
        if (status != cudaSuccess) return status;
      }
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{target_device}};

      std::size_t skip      = copy_rng.offset - io_rng.offset;
      std::size_t copied    = 0;
      auto* device_dst      = d_buffer.data + (copy_rng.offset - req_rng.offset);
      std::size_t remaining = copy_rng.size;

      sirius::cuda::device_copy_batch batch;
      batch.reserve(host_buf.size());
      for (auto const& entry : host_buf) {
        auto const length = entry.iov_len;
        if (skip >= length) {
          skip -= length;
          continue;
        }

        auto const available = length - skip;
        auto const bytes     = std::min(available, remaining);
        auto const* source   = static_cast<std::uint8_t const*>(entry.iov_base) + skip;
        batch.add(device_dst + copied, source, bytes);
        copied += bytes;
        remaining -= bytes;
        skip = 0;
        if (remaining == 0) break;
      }

      if (remaining != 0) return cudaErrorInvalidValue;
      auto const copy_status = batch.enqueue(d_buffer.stream);
      if (copy_status != cudaSuccess) return copy_status;
      return event == nullptr ? cudaSuccess : cudaEventRecord(event, d_buffer.stream.value());
    } catch (...) {
      return cudaErrorUnknown;
    }
  }
};

/**
 * @brief Backend-neutral physical operation produced by a reactor worker.
 *
 * Reactor-specific transfer state may be retained through @ref staging_owner.
 * Every terminal path must call exactly one of finish_success/finish_error;
 * the cache callback runs before the final coordinator decrement.
 */
struct io_op_request {
  std::shared_ptr<const io_object> obj;
  range io_rng;
  std::vector<iovec> iovecs;
  std::shared_ptr<void> staging_owner;
  std::unique_ptr<device_cpy_request> device_copy;
  std::shared_ptr<grouped_coordinator> coordinator;
  std::shared_ptr<prepared_io_completion> on_complete;
  std::vector<cache::cached_chunk*> completion_chunks;

  void finish_success() noexcept
  {
    if (_terminal.exchange(true, std::memory_order_acq_rel)) return;
    if (on_complete != nullptr) { (*on_complete)(completion_chunks, true); }
    coordinator->on_complete();
  }

  void finish_error(grouped_coordinator::error_type const& error,
                    bool host_data_valid = false) noexcept
  {
    if (_terminal.exchange(true, std::memory_order_acq_rel)) return;
    if (on_complete != nullptr) { (*on_complete)(completion_chunks, host_data_valid); }
    coordinator->report_error(error);
  }

  [[nodiscard]] bool terminal() const noexcept { return _terminal.load(std::memory_order_acquire); }

 private:
  std::atomic<bool> _terminal{false};
};

}  // namespace sirius::io
