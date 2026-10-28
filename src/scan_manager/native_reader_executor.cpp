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

#include "scan_manager/native_reader_executor.hpp"

#include "transparent/replay_admission.hpp"

#include <rmm/cuda_device.hpp>

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace sirius::scan_manager {
struct native_reader_executor::state {
  struct slot {
    bool leased = false, running = false;
    std::unique_ptr<operation> pending;
  };
  using observers = std::vector<std::weak_ptr<preparation_wakeup>>;
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<slot> slots;
  std::vector<std::thread> workers;
  const size_t worker_count;
  bool stopping                          = false;
  std::shared_ptr<observers const> wakes = std::make_shared<observers>();
  state(size_t threads, size_t capacity) : slots(capacity), worker_count(threads) {}

  // Snapshot without allocating, then notify after unlocking. A wake only signals C; it
  // cannot run reader cleanup or invert the executor/publication-gate lock order.
  void notify() noexcept
  {
    std::shared_ptr<observers const> snapshot;
    {
      std::lock_guard lock(mutex);
      snapshot = wakes;
    }
    for (auto const& weak : *snapshot)
      if (auto wake = weak.lock()) (*wake)();
  }
  void run(size_t worker) noexcept
  {
    size_t cursor = worker;
    for (;;) {
      size_t selected = slots.size();
      std::unique_ptr<operation> work;
      {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] {
          if (stopping) return true;
          for (size_t n = 0; n < slots.size(); ++n) {
            auto i = (cursor + n) % slots.size();
            if (i % worker_count == worker && slots[i].pending) {
              selected = i;
              return true;
            }
          }
          return false;
        });
        if (stopping) return;
        auto& slot   = slots[selected];
        slot.running = true;
        work         = std::move(slot.pending);
      }
      if (auto const device = work->device_id(); device >= 0) {
        rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device}};
        work->run();
        work.reset();  // Captures that hold device resources are released on that device.
      } else {
        work->run();
        work.reset();  // The operation and its captures die on the assigned thread, unlocked.
      }
      {
        std::lock_guard lock(mutex);
        slots[selected].running = false;
        cursor                  = (selected + 1) % slots.size();
      }
      notify();
    }
  }
};

native_reader_executor::native_reader_executor(size_t workers, size_t max_readers)
{
  if (!workers || max_readers < workers)
    throw std::invalid_argument(
      "native executor needs positive workers and at least one slot per worker");
  state_ = std::make_unique<state>(workers, max_readers);
  state_->workers.reserve(workers);
}
void native_reader_executor::start_workers()
{
  if (!state_->workers.empty()) return;
  try {
    for (size_t i = 0; i < state_->worker_count; ++i) {
      if (before_thread_) before_thread_(i);
      state_->workers.emplace_back([this, i] { state_->run(i); });
    }
  } catch (...) {
    {
      std::lock_guard lock(state_->mutex);
      state_->stopping = true;
    }
    state_->cv.notify_all();
    for (auto& thread : state_->workers)
      thread.join();
    state_->workers.clear();
    // No lease was admitted. Restore the dormant state so a later attempt can retry.
    {
      std::lock_guard lock(state_->mutex);
      state_->stopping = false;
    }
    throw;
  }
}
native_reader_executor::~native_reader_executor() { stop(); }
void native_reader_executor::stop()
{
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  {
    std::lock_guard lock(state_->mutex);
    if (state_->stopping) return;
    // Do not disable workers while an attempt still needs them for close/disposal.
    if (std::ranges::any_of(state_->slots,
                            [](auto const& s) { return s.leased || s.running || s.pending; }))
      throw std::logic_error("cannot stop native executor with active readers");
    state_->stopping = true;
  }
  state_->cv.notify_all();
  for (auto& worker : state_->workers)
    worker.join();
  state_->workers.clear();
}
void native_reader_executor::before_thread_for_testing(std::function<void(size_t)> hook)
{
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  if (!state_->workers.empty() || state_->stopping)
    throw std::logic_error("native executor startup seam already frozen");
  before_thread_ = std::move(hook);
}
std::vector<std::unique_ptr<native_reader_executor::lease>> native_reader_executor::reserve(
  size_t count)
{
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  std::vector<std::unique_ptr<lease>> result;
  {
    std::lock_guard lock(state_->mutex);
    if (state_->stopping) throw std::runtime_error("native executor is stopped");
    if (std::ranges::count_if(state_->slots, [](auto const& s) { return !s.leased; }) < count)
      throw transparent::classified_execution_error(
        transparent::late_failure_cause::resource,
        "native reader capacity unavailable for complete attempt");
    result.reserve(count);
    // Construct every inactive handle before starting workers or committing any slot.
    for (size_t i = 0; i < state_->slots.size() && result.size() < count; ++i)
      if (!state_->slots[i].leased) result.emplace_back(new lease(*state_, i));
  }
  if (!count) return result;
  start_workers();
  {
    std::lock_guard lock(state_->mutex);
    for (auto& handle : result) {
      state_->slots[handle->index_].leased = true;
      handle->active_                      = true;
    }
  }
  return result;
}
std::shared_ptr<preparation_wakeup> native_reader_executor::observe(preparation_wakeup wake)
{
  auto subscription = std::make_shared<preparation_wakeup>(std::move(wake));
  std::lock_guard lock(state_->mutex);
  auto next = std::make_shared<state::observers>();
  next->reserve(state_->wakes->size() + 1);
  for (auto const& old : *state_->wakes)
    if (!old.expired()) next->push_back(old);
  next->push_back(subscription);
  state_->wakes = std::move(next);
  return subscription;
}
native_reader_executor::lease::~lease()
{
  if (!active_) return;
  {
    std::lock_guard lock(owner_->mutex);
    auto& slot = owner_->slots[index_];
    assert(!slot.running && !slot.pending);
    slot.leased = false;
  }
  owner_->notify();
}
bool native_reader_executor::lease::idle() const
{
  std::lock_guard lock(owner_->mutex);
  auto const& slot = owner_->slots[index_];
  return !slot.running && !slot.pending;
}
void native_reader_executor::lease::submit(std::unique_ptr<operation> task) noexcept
{
  {
    std::lock_guard lock(owner_->mutex);
    auto& slot = owner_->slots[index_];
    assert(active_ && task && !slot.pending && !slot.running && !owner_->stopping);
    slot.pending = std::move(task);
  }
  owner_->cv.notify_all();
}
}  // namespace sirius::scan_manager
