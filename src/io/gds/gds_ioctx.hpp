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

#include "io/gds/gds_config.hpp"
#include "io/gds/gds_stats.hpp"

#include <kvikio/compat_mode.hpp>
#include <kvikio/file_handle.hpp>

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

namespace sirius::io::gds {

/// A cuFile read did not return within `gds_config::read_timeout`, or reads
/// stayed slower than `gds_config::slow_read_ms_per_mib`.
class read_stalled_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// cuFile served a read with POSIX calls while `gds_config::require` is set.
class posix_fallback_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Local-file read backend that moves file bytes into GPU memory with
 * cuFile (GPUDirect Storage).
 *
 * A device read is cut into pieces of `gds_config::task_size` and each piece
 * is one blocking `kvikio::FileHandle::read` (`cuFileRead`) on a thread pool
 * this context owns. That is the call pattern of `kvikio::FileHandle::pread`,
 * which cuDF's own file source uses. The pool is ours rather than kvikIO's so
 * that a read returns at once and settles later (cuDF issues one read per
 * column chunk and expects them to overlap), and so that every piece in flight
 * can be timed.
 *
 * Host reads (Parquet footers) are plain `pread`.
 *
 * It reports `io_context_type::kvikio`: the scan manager returns its default
 * context for a path only when the registry's type for that path matches, and
 * cuCascade's enum has no value of its own for this backend.
 *
 * Reads that stay slower than `slow_read_ms_per_mib` mark the context slow: later
 * reads fail with `read_stalled_error` while those in flight finish. That is
 * the early sign of a storage path going bad, seen half a minute before a stall.
 *
 * A read in flight longer than `read_timeout` ends the process with status 70.
 * cuFile reads cannot be cancelled, and failing the request instead would let
 * the caller free a buffer cuFile is still writing to.
 */
class gds_ioctx final : public cucascade::io::ioctx {
 public:
  /// Reads one piece into device memory at @p dst and returns the bytes read.
  using device_reader = std::function<std::size_t(
    kvikio::FileHandle&, void* dst, std::size_t size, std::size_t file_offset)>;

  /// Everything but `compat_mode` exists for tests.
  struct options {
    /// OFF is GDS. ON makes kvikIO read with POSIX calls, which lets tests run
    /// the same code on machines without GDS.
    kvikio::CompatMode compat_mode{kvikio::CompatMode::OFF};
    /// Unset reads with `kvikio::FileHandle::read` (cuFile, or POSIX in compat mode).
    device_reader reader{};
    std::chrono::milliseconds watchdog_period{1000};
    /// Reads a watchdog period needs before their mean time is judged.
    std::size_t slow_read_min_reads{16};
    /// Called once when a read is stuck. Defaults to logging and `std::_Exit(70)`.
    std::function<void()> on_fatal_stall{};
  };

  struct io_counters {
    std::uint64_t device_bytes{0};
    std::uint64_t device_reads{0};
    std::uint64_t host_bytes{0};
    /// Time device reads spent inside cuFile, summed over reads.
    std::uint64_t device_read_ns{0};
    /// Time device reads waited for a pool thread, summed over reads.
    std::uint64_t device_wait_ns{0};
  };

  /// @throw std::runtime_error in GDS mode when cuFile does not report GDS as
  ///        available, or, with `require`, when it may fall back to POSIX.
  explicit gds_ioctx(gds_config config, options opts);
  explicit gds_ioctx(gds_config config) : gds_ioctx(config, options{}) {}
  ~gds_ioctx() override;

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::kvikio;
  }

  void shutdown() noexcept override;

  [[nodiscard]] bool supports(std::string_view path) const noexcept override;
  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return false; }

  [[nodiscard]] std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<const cucascade::io::byte_range> ranges,
    std::optional<std::size_t> alignment) const noexcept override;

  std::size_t host_read_io(const cucascade::io::io_object& obj,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* dst) override;

  cucascade::exec::semi_future<std::size_t> mixed_readv_async_io(
    const cucascade::io::io_object& obj,
    std::vector<cucascade::io::prepared_io_slice>&& slices) noexcept override;

  [[nodiscard]] gds_config const& config() const noexcept { return _config; }
  [[nodiscard]] bool stalled() const noexcept { return _state.load() == state::stalled; }
  [[nodiscard]] io_counters counters() const noexcept;
  /// cuFile's counters, when the loaded libcufile reports them.
  [[nodiscard]] std::optional<cufile_counters> cufile() const noexcept;

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path) override;

 private:
  enum class state : int { ok, stalled, slow, posix_fallback };
  struct request;

  void submit(std::function<void(std::size_t worker)> task);
  void worker_loop(std::size_t worker);
  void watchdog_loop();
  void run_piece(std::shared_ptr<request> const& req,
                 std::size_t slice_index,
                 std::size_t piece_offset,
                 std::size_t piece_size,
                 std::int64_t queued_at,
                 std::size_t worker);
  void settle(std::shared_ptr<request> const& req, std::exception_ptr error) noexcept;
  [[nodiscard]] std::exception_ptr failure() const;

  gds_config _config;
  options _options;
  bool _cufile_counters{false};
  cufile_counters _cufile_baseline{};

  std::atomic<state> _state{state::ok};
  std::atomic<std::uint64_t> _device_bytes{0};
  std::atomic<std::uint64_t> _device_reads{0};
  std::atomic<std::uint64_t> _host_bytes{0};
  std::atomic<std::uint64_t> _device_read_ns{0};
  std::atomic<std::uint64_t> _device_charged_bytes{0};
  std::atomic<std::uint64_t> _device_wait_ns{0};

  std::mutex _queue_mutex;
  std::condition_variable _queue_cv;
  /// Separate from `_queue_cv`: a shared one lets the watchdog swallow a worker's wake-up.
  std::condition_variable _watchdog_cv;
  std::deque<std::function<void(std::size_t)>> _queue;
  bool _stop{false};
  bool _watchdog_stop{false};

  std::mutex _live_mutex;
  std::unordered_set<std::shared_ptr<request>> _live;

  /// Start time of the read each worker is in, as steady-clock nanoseconds; 0 when idle.
  std::unique_ptr<std::atomic<std::int64_t>[]> _read_started;
  std::vector<std::thread> _workers;
  std::thread _watchdog;
};

}  // namespace sirius::io::gds
