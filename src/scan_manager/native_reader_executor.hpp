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

#include "scan_manager/preparation.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace sirius::scan_manager {
// Runtime-owned, bounded executor shared by attempts. A reader keeps thread affinity, not
// exclusive use of a worker. Idle readers cannot prevent other readers on that worker running.
class native_reader_executor {
  struct state;

 public:
  struct operation {
    virtual ~operation()        = default;
    virtual void run() noexcept = 0;
    // Valid CUDA device ordinal for run(), or -1 for host-only work. Workers have no fixed
    // device affinity; the executor selects this device for the duration of run() and of the
    // operation's destruction that follows it. An operation
    // that leaves device-resident output must record a writer event after its last write and hand
    // it over with the output; the consumer waits on that event on its own stream before reading.
    // Never synchronize the whole device.
    virtual int device_id() const noexcept { return -1; }
  };
  class lease {
   public:
    ~lease();
    lease(lease const&)            = delete;
    lease& operator=(lease const&) = delete;
    bool idle() const;
    // Caller owns an idle lease. No allocation after ownership transfer. Even cancellation
    // must run its cleanup operation: this executor never silently discards accepted work.
    void submit(std::unique_ptr<operation>) noexcept;

   private:
    friend class native_reader_executor;
    lease(state& owner, size_t index) : owner_(&owner), index_(index) {}
    state* owner_;
    size_t index_;
    bool active_ = false;
  };

  // No worker threads are started until the first nonempty reservation succeeds.
  native_reader_executor(size_t workers, size_t max_readers);
  // All attempts must retire and return their leases before the runtime destroys the executor.
  ~native_reader_executor();
  native_reader_executor(native_reader_executor const&)            = delete;
  native_reader_executor& operator=(native_reader_executor const&) = delete;

  // Runtime shutdown: requires all leases returned. Rejection leaves live readers usable.
  // Idempotent after success; no later reservation can reopen this executor.
  void stop();
  // Fault seam, installed before the first reservation. Called before each thread creation.
  void before_thread_for_testing(std::function<void(size_t)>);

  // Atomic group admission. Throws before assigning any slot when capacity is unavailable.
  // Waiting with a partial group could deadlock sources with downstream dependencies.
  std::vector<std::unique_ptr<lease>> reserve(size_t readers);
  // Keep the returned subscription alive while waiting for operation disposal/slot release.
  std::shared_ptr<preparation_wakeup> observe(preparation_wakeup);

 private:
  // Serializes startup/admission/shutdown. Workers and lease release never acquire it.
  std::mutex lifecycle_mutex_;
  std::unique_ptr<state> state_;
  std::function<void(size_t)> before_thread_;
  void start_workers();  // Caller holds lifecycle_mutex_; no slot lock across join.
};
}  // namespace sirius::scan_manager
