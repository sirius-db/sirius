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

#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

namespace sirius::scan_manager {

/**
 * @brief One `[uring_gauges]` DEBUG line for reactor @p index, or nullopt when the
 * reactor was idle over the window.
 *
 * @param index The reactor's position in @c uring_ioctx::reactor_gauges().
 * @param current This window's gauges (a @c take_gauges() snapshot).
 * @param previous The previous window's gauges of the same reactor, or nullptr for
 *        the first window (the cumulative started / submitted-bytes deltas are then 0).
 * @param seconds Length of the window, for the MiB/s rate (0 prints a rate of 0).
 */
[[nodiscard]] std::optional<std::string> format_gauges_line(
  std::size_t index,
  cucascade::io::uring::uring_reactor::gauges const& current,
  cucascade::io::uring::uring_reactor::gauges const* previous,
  double seconds);

/**
 * @brief Periodic DEBUG logger of a uring ioctx's per-reactor gauges.
 *
 * A thread named `uring_gauges` wakes every @ref sample_period and, only while the
 * log sink accepts DEBUG, polls @c uring_ioctx::reactor_gauges() and logs one
 * @ref format_gauges_line per non-idle reactor.  Above DEBUG it costs a timed
 * wakeup and nothing else, and does not restart the reactors' gauge windows.
 *
 * Owned by the scan manager, which stops it before releasing the ioctx.
 */
class uring_gauges_sampler {
 public:
  static constexpr std::chrono::milliseconds sample_period{250};

  explicit uring_gauges_sampler(std::shared_ptr<cucascade::io::uring::uring_ioctx> io_ctx);
  ~uring_gauges_sampler();

  uring_gauges_sampler(uring_gauges_sampler const&)            = delete;
  uring_gauges_sampler& operator=(uring_gauges_sampler const&) = delete;
  uring_gauges_sampler(uring_gauges_sampler&&)                 = delete;
  uring_gauges_sampler& operator=(uring_gauges_sampler&&)      = delete;

  /// Start the sampler thread.  No-op while it is running.
  void start();

  /// Stop and join the sampler thread.  Idempotent.
  void stop() noexcept;

 private:
  void run(std::stop_token const& stop_token);

  std::shared_ptr<cucascade::io::uring::uring_ioctx> _io_ctx;
  std::mutex _mutex;
  std::condition_variable_any _cv;
  /// Declared last: its destructor stops and joins the thread before the members
  /// it reads go away.
  std::jthread _thread;
};

}  // namespace sirius::scan_manager
