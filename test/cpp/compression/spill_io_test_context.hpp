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

#include <cuda_runtime.h>

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/io/io_context.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::test {

/// Two deterministic pauses: before the inner backend can touch source buffers, and
/// after its future has drained but before the proxy resolves the caller's future.
struct spill_write_gate {
  std::promise<void> submitted_promise;
  std::promise<void> inner_completed_promise;
  std::shared_future<void> submitted       = submitted_promise.get_future().share();
  std::shared_future<void> inner_completed = inner_completed_promise.get_future().share();

  void allow_submit()
  {
    std::call_once(_submit_once, [&] { _submit_promise.set_value(); });
  }
  void allow_completion()
  {
    std::call_once(_complete_once, [&] { _complete_promise.set_value(); });
  }
  void release()
  {
    allow_submit();
    allow_completion();
  }
  void wait_to_submit() const { _submit.wait(); }
  void wait_to_complete() const { _complete.wait(); }

 private:
  std::promise<void> _submit_promise;
  std::promise<void> _complete_promise;
  std::shared_future<void> _submit   = _submit_promise.get_future().share();
  std::shared_future<void> _complete = _complete_promise.get_future().share();
  std::once_flag _submit_once;
  std::once_flag _complete_once;
};

/// Always releases a gate and joins, including when a Catch REQUIRE unwinds the test.
/// Unlike std::async's implicit join, teardown cannot deadlock on an unreleased gate.
template <typename Result>
class joined_spill_task {
 public:
  template <typename Function>
  joined_spill_task(std::shared_ptr<spill_write_gate> gate, Function&& function)
    : _gate(std::move(gate))
  {
    std::packaged_task<Result()> task(std::forward<Function>(function));
    result  = task.get_future();
    _thread = std::thread(std::move(task));
  }
  ~joined_spill_task()
  {
    _gate->release();
    if (_thread.joinable()) { _thread.join(); }
  }
  joined_spill_task(joined_spill_task const&)            = delete;
  joined_spill_task& operator=(joined_spill_task const&) = delete;

  std::future<Result> result;

 private:
  std::shared_ptr<spill_write_gate> _gate;
  std::thread _thread;
};

/// Cache-free proxy over a real DISK backend: forced cuFile (with POSIX fallback)
/// when compiled in, otherwise the automatic backend. Only write capabilities and
/// the write completion result are changed.
class spill_io_test_context final : public cucascade::io::ioctx {
 public:
  enum class completion { success, short_count, failure };

  struct observations {
    std::size_t calls{0};
    std::size_t host_slices{0};
    std::size_t device_slices{0};
    std::size_t requested_bytes{0};
    std::size_t header_bytes{0};
    std::size_t inner_bytes{0};
    std::size_t read_opens{0};
    std::size_t opened_read_size{0};
    std::size_t host_read_calls{0};
    std::size_t host_read_bytes{0};
    std::size_t device_read_calls{0};
    std::size_t device_read_bytes{0};
    std::string last_path;
  };

  spill_io_test_context()
#ifdef CUCASCADE_HAS_CUFILE
    : _inner(cucascade::make_disk_io_context({.backend = cucascade::disk_io_backend::cufile}))
#else
    : _inner(cucascade::make_disk_io_context())
#endif
  {
  }
  ~spill_io_test_context() override
  {
    // Gated tests own a joined_spill_task, which releases its gate before the
    // context can be destroyed. Drain frontend callbacks before the inner context.
    pre_destroy();
    if (_writer.joinable()) { _writer.join(); }
  }

  // Configure before submitting; these are never changed while a write is active.
  bool write_enabled{true};
  bool device_write_enabled{true};
  completion write_completion{completion::success};
  std::shared_ptr<spill_write_gate> gate;
  // One-shot, read-only object supplied by truncation tests. Configure before decoding.
  std::shared_ptr<cucascade::io::io_object> next_read_object;

  observations observed() const
  {
    std::lock_guard lock(_mutex);
    return _observed;
  }

  cucascade::io::io_context_type type() const noexcept override { return _inner->type(); }
  void start() override { _inner->start(); }
  void shutdown() noexcept override
  {
    if (gate) { gate->release(); }
    if (_writer.joinable()) { _writer.join(); }
    _inner->shutdown();
  }
  void warmup(std::string_view path) noexcept override { _inner->warmup(path); }
  bool supports(std::string_view path) const noexcept override { return _inner->supports(path); }
  bool supports_device_read() const noexcept override { return _inner->supports_device_read(); }
  bool supports_host_to_device_read() const noexcept override
  {
    return _inner->supports_host_to_device_read();
  }
  bool supports_vector_host_read() const noexcept override
  {
    return _inner->supports_vector_host_read();
  }
  bool supports_device_range_read() const noexcept override
  {
    return _inner->supports_device_range_read();
  }
  bool supports_write() const noexcept override
  {
    return write_enabled && _inner->supports_write();
  }
  bool supports_device_write() const noexcept override
  {
    return supports_write() && device_write_enabled && _inner->supports_device_write();
  }
  bool prefers_bulk_io() const noexcept override { return _inner->prefers_bulk_io(); }
  std::size_t min_alignment_requirement() const noexcept override
  {
    return _inner->min_alignment_requirement();
  }
  std::size_t merge_gap_size() const noexcept override { return _inner->merge_gap_size(); }
  std::size_t n_max_concurrent_scans() const noexcept override
  {
    return _inner->n_max_concurrent_scans();
  }
  std::size_t staging_block_size() const noexcept override { return _inner->staging_block_size(); }

  std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<const cucascade::io::byte_range> ranges,
    std::optional<std::size_t> alignment = std::nullopt) const noexcept override
  {
    return _inner->align_and_coalesce(ranges, alignment);
  }

  std::size_t host_read_io(const cucascade::io::io_object& obj,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* dst) override
  {
    auto const bytes = _inner->host_read_io(obj, offset, size, dst);
    std::lock_guard lock(_mutex);
    ++_observed.host_read_calls;
    _observed.host_read_bytes += bytes;
    return bytes;
  }
  cucascade::exec::semi_future<std::size_t> host_read_async_io(
    const cucascade::io::io_object& obj,
    std::size_t offset,
    std::size_t size,
    std::uint8_t* dst,
    cucascade::io::io_priority priority = cucascade::io::io_priority::automatic) noexcept override
  {
    return _inner->host_read_async_io(obj, offset, size, dst, priority);
  }
  cucascade::exec::semi_future<std::size_t> device_read_async_io(
    const cucascade::io::io_object& obj,
    std::size_t offset,
    std::size_t size,
    std::uint8_t* dst,
    ::cuda::stream_ref stream,
    cucascade::io::io_priority priority = cucascade::io::io_priority::automatic) noexcept override
  {
    try {
      {
        std::lock_guard lock(_mutex);
        ++_observed.device_read_calls;
      }
      return _inner->device_read_async_io(obj, offset, size, dst, stream, priority)
        .defer_value([this](std::size_t bytes) {
          std::lock_guard lock(_mutex);
          _observed.device_read_bytes += bytes;
          return bytes;
        });
    } catch (...) {
      return cucascade::exec::make_semi_future<std::size_t>(std::current_exception());
    }
  }
  cucascade::exec::semi_future<std::size_t> host_readv_async_io(
    const cucascade::io::io_object& obj,
    std::span<const cucascade::io::slice> slices,
    cucascade::io::io_priority priority = cucascade::io::io_priority::automatic) noexcept override
  {
    return _inner->host_readv_async_io(obj, slices, priority);
  }
  cucascade::exec::semi_future<std::size_t> device_readv_async_io(
    const cucascade::io::io_object& obj,
    std::span<const cucascade::io::slice> slices,
    ::cuda::stream_ref stream,
    cucascade::io::io_priority priority = cucascade::io::io_priority::automatic) noexcept override
  {
    return _inner->device_readv_async_io(obj, slices, stream, priority);
  }
  cucascade::exec::semi_future<std::size_t> mixed_readv_async_io(
    const cucascade::io::io_object& obj,
    std::vector<cucascade::io::prepared_io_slice>&& slices) noexcept override
  {
    return _inner->mixed_readv_async_io(obj, std::move(slices));
  }

  std::size_t host_write_io(const cucascade::io::io_object& obj,
                            std::size_t offset,
                            std::size_t size,
                            const std::uint8_t* src) override
  {
    return _inner->host_write_io(obj, offset, size, src);
  }
  void flush_io(const cucascade::io::io_object& obj) override { _inner->flush_io(obj); }
  cucascade::exec::semi_future<void> flush_async_io(
    const cucascade::io::io_object& obj, cucascade::io::io_priority priority) noexcept override
  {
    return _inner->flush_async_io(obj, priority);
  }

  cucascade::exec::semi_future<std::size_t> mixed_writev_async_io(
    const cucascade::io::io_object& obj,
    std::vector<cucascade::io::prepared_io_slice>&& slices) noexcept override
  {
    try {
      {
        std::lock_guard lock(_mutex);
        ++_observed.calls;
        for (auto const& slice : slices) {
          _observed.host_slices += slice.has_host_request();
          _observed.device_slices += slice.has_device_request();
          _observed.requested_bytes += slice.size();
          if (slice.is_host_request() && slice.offset() == 0) {
            _observed.header_bytes = slice.size();
          }
        }
      }
      auto const mode = write_completion;
      if (!gate) {
        return _inner->mixed_writev_async_io(obj, std::move(slices))
          .defer_value([this, mode](std::size_t bytes) { return finish(bytes, mode); });
      }

      if (_writer.joinable()) { _writer.join(); }
      int device = -1;
      if (cudaGetDevice(&device) != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice failed");
      }
      cucascade::exec::promise<std::size_t> promise;
      auto result = promise.get_semi_future();
      _writer     = std::thread([this,
                             &obj,
                             slices = std::move(slices),
                             pause  = gate,
                             device,
                             mode,
                             promise = std::move(promise)]() mutable {
        pause->submitted_promise.set_value();
        pause->wait_to_submit();
        std::size_t bytes = 0;
        std::exception_ptr error;
        try {
          if (cudaSetDevice(device) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed");
          }
          // Deliberately do not copy or inspect the source bytes before the gate.
          // A valid file proves the converter retained its header and encoded buffers.
          bytes = _inner->mixed_writev_async_io(obj, std::move(slices)).get();
          std::lock_guard lock(_mutex);
          _observed.inner_bytes = bytes;
        } catch (...) {
          error = std::current_exception();
        }
        pause->inner_completed_promise.set_value();
        pause->wait_to_complete();
        try {
          if (error) { std::rethrow_exception(error); }
          promise.set_value(finish(bytes, mode));
        } catch (...) {
          promise.set_exception(std::current_exception());
        }
      });
      return result;
    } catch (...) {
      cucascade::exec::promise<std::size_t> promise;
      auto result = promise.get_semi_future();
      promise.set_exception(std::current_exception());
      return result;
    }
  }

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path) override
  {
    auto object = next_read_object ? std::exchange(next_read_object, {})
                                   : _inner->open_io_object(std::move(path));
    std::lock_guard lock(_mutex);
    ++_observed.read_opens;
    _observed.opened_read_size = object->size();
    return object;
  }
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path,
                                                             cucascade::io::open_hint hint) override
  {
    return _inner->open_io_object(std::move(path), hint);
  }
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path,
                                                             std::uint64_t known_size) override
  {
    return _inner->open_io_object(std::move(path), known_size);
  }
  std::shared_ptr<cucascade::io::io_object> create_io_object_for_write(
    std::string path, cucascade::io::write_open_options options) override
  {
    {
      std::lock_guard lock(_mutex);
      _observed.last_path = path;
    }
    return _inner->open_io_object_for_write(std::move(path), options);
  }

 private:
  std::size_t finish(std::size_t bytes, completion mode)
  {
    {
      std::lock_guard lock(_mutex);
      _observed.inner_bytes = bytes;
    }
    if (mode == completion::failure) { throw std::runtime_error("injected spill write failure"); }
    return mode == completion::short_count && bytes != 0 ? bytes - 1 : bytes;
  }

  std::shared_ptr<cucascade::io::ioctx> _inner;
  mutable std::mutex _mutex;
  observations _observed;
  std::thread _writer;
};

}  // namespace sirius::test
