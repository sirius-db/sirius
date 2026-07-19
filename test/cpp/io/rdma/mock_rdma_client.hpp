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

#include "io/rdma/rdma_transport_client.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace sirius::io::rdma {

/**
 * @brief In-memory @c s3_control_client test double.
 *
 * Serves seeded objects (HEAD size, ranged GET with Content-Range), with
 * one-shot response overrides (@c respond_status / @c fail_transport apply to
 * the NEXT call only, then clear), a gate that blocks @c range_get while
 * closed, and call/connection counters mirroring the production seam.
 * Thread-safe.
 */
class mock_s3_control_client final : public s3_control_client {
 public:
  void put_object(std::string bucket, std::string key, std::vector<std::uint8_t> bytes);

  /// The NEXT call reports this transport error (status 0), then clears.
  void fail_transport(std::string curl_error);
  /// The NEXT call reports this HTTP status with no bytes, then clears.
  void respond_status(long http_status);
  /// The next @p count range_get calls report @p http_status (no bytes),
  /// then normal service resumes — drives the host-plane retry: count 1
  /// exercises retry-then-succeed, count >= max-attempts exercises
  /// retry-exhaustion.
  void fail_next_n_range_gets(std::size_t count, long http_status);

  /// While closed, range_get() blocks (after being counted) until open_gate().
  void close_gate();
  void open_gate();

  [[nodiscard]] size_t heads_issued() const;
  [[nodiscard]] size_t range_gets_issued() const;

  [[nodiscard]] head_result head(const rx_route& route) override;
  [[nodiscard]] range_get_result range_get(const rx_route& route,
                                           size_t offset,
                                           size_t size,
                                           uint8_t* dst) override;
  [[nodiscard]] uint64_t attempts_total() const noexcept override;
  [[nodiscard]] uint64_t connections_total() const noexcept override;

 private:
  mutable std::mutex _mtx;
  std::condition_variable _gate_cv;
  std::map<std::pair<std::string, std::string>, std::vector<std::uint8_t>> _objects;
  std::optional<std::string> _next_transport_error;
  std::optional<long> _next_status;
  std::size_t _fail_next_range_gets{0};
  long _fail_next_status{0};
  bool _gate_closed{false};
  size_t _heads_issued{0};
  size_t _range_gets_issued{0};
  bool _connected{false};  // the first attempt "opens" the connection; reuse adds 0
  uint64_t _connections_total{0};
};

/**
 * @brief Shared-state @c rdma_data_session_factory test double.
 *
 * ALL test state lives on the factory: sessions handed to workers are thin
 * views over it, so tests configure everything before start() and observe
 * everything after teardown.  With no script or fault installed, a get()
 * serves the seeded object as a fully valid completion (exact bytes, HTTP
 * 200, tag "mock-reply-tag") and copies the payload into the destination
 * (host or device).  Precedence: scripted results (FIFO) over persistent
 * @c fail_gets (sent_unknown) over persistent @c short_write (completed with
 * fewer bytes) over the seeded serve.  A gate blocks get() (after counting)
 * while closed.  Thread-safe.
 */
class mock_rdma_data_session_factory final
  : public rdma_data_session_factory,
    public std::enable_shared_from_this<mock_rdma_data_session_factory> {
 public:
  void put_object(std::string bucket, std::string key, std::vector<std::uint8_t> bytes);
  void script_result(data_get_result result);
  void close_get_gate();
  void open_get_gate();
  void fail_gets(std::string transport_error);
  void short_write(size_t bytes);
  /// Subsequent acquire() calls throw std::runtime_error(what).
  void fail_acquire(std::string what);
  /// The first @p count acquire() calls succeed; every later one returns
  /// nullptr — count 0 fails the first worker's acquire, count = worker
  /// count leaves the workers whole and fails the arena registrar's acquire.
  void null_acquire_after(size_t count);
  /// The next @p count register_memory calls on a session throw.
  void fail_register(size_t count, std::string message);
  /// Sessions currently alive (constructed and not yet destroyed): proves a
  /// leaked arena registrar session is released rather than destroyed —
  /// checking deregister_count == 0 alone cannot.
  [[nodiscard]] size_t live_sessions() const;

  [[nodiscard]] size_t gets_issued() const;
  [[nodiscard]] size_t peak_concurrent_gets() const;
  [[nodiscard]] size_t register_count() const;
  [[nodiscard]] size_t deregister_count() const;
  [[nodiscard]] size_t acquired_total() const;
  [[nodiscard]] std::vector<std::thread::id> acquisition_thread_ids() const;

  [[nodiscard]] std::unique_ptr<rdma_data_session> acquire() override;

 private:
  friend class mock_rdma_data_session;
  data_get_result serve_get(const rx_route& route, size_t offset, size_t size, void* dst);
  void count_register();  // throws when fail_register is armed
  void count_deregister();
  void on_session_created() noexcept { _live_sessions.fetch_add(1, std::memory_order_relaxed); }
  void on_session_destroyed() noexcept { _live_sessions.fetch_sub(1, std::memory_order_relaxed); }

  mutable std::mutex _mtx;
  std::condition_variable _gate_cv;
  std::map<std::pair<std::string, std::string>, std::vector<std::uint8_t>> _objects;
  std::deque<data_get_result> _scripted;
  std::optional<std::string> _fail_message;
  std::optional<size_t> _short_write;
  std::optional<std::string> _fail_acquire;
  std::optional<size_t> _null_acquire_after;
  size_t _acquire_calls{0};
  size_t _fail_register_count{0};
  std::string _fail_register_message;
  std::atomic<size_t> _live_sessions{0};
  bool _gate_closed{false};
  size_t _gets_issued{0};
  // In-flight get accounting is atomic, not _mtx-guarded: the delivery copy
  // runs after the lock is released (below), so the RAII decrement must be
  // safe without the lock even if that copy throws.
  std::atomic<size_t> _concurrent{0};
  std::atomic<size_t> _peak_concurrent{0};
  size_t _register_count{0};
  size_t _deregister_count{0};
  std::vector<std::thread::id> _acquire_threads;
};

}  // namespace sirius::io::rdma
