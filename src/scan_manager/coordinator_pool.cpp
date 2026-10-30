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

#include "scan_manager/coordinator_pool.hpp"

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <stdexcept>

namespace sirius::scan_manager {
struct coordinator_pool::entry {
  std::function<step_result()> step;
  std::function<void()> stop;
  std::function<void(std::exception_ptr)> fail;
  size_t index = 0, generation = 0;
  bool registered = false, runnable = false, executing = false, finished = false,
       stop_notifying = false;
  std::optional<clock::time_point> deadline;
};

struct coordinator_pool::state {
  static thread_local state* current_worker;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::shared_ptr<entry>> entries;
  size_t const worker_count;
  size_t started = 0, cursor = 0, next_generation = 0;
  bool stopping = false;
  explicit state(size_t workers) : worker_count(workers) {}

  void wake(size_t index, size_t generation) noexcept
  {
    std::lock_guard lock(mutex);
    if (index >= entries.size()) return;
    auto const& e = entries[index];
    if (!e || e->generation != generation || e->finished) return;
    e->runnable = true;
    e->deadline.reset();
    changed.notify_all();
  }

  void run() noexcept
  {
    current_worker = this;
    for (;;) {
      std::shared_ptr<entry> selected;
      {
        std::unique_lock lock(mutex);
        for (;;) {
          std::optional<clock::time_point> deadline;
          bool unfinished = false;
          auto now        = clock::now();
          for (size_t n = 0; n < entries.size(); ++n) {
            auto i        = (cursor + n) % entries.size();
            auto const& e = entries[i];
            if (!e || e->finished) continue;
            unfinished = true;
            if (e->executing) continue;
            if (e->runnable || (e->deadline && *e->deadline <= now)) {
              selected     = e;
              e->executing = true;
              e->runnable  = false;
              e->deadline.reset();
              cursor = (i + 1) % entries.size();
              break;
            }
            if (e->deadline && (!deadline || *e->deadline < *deadline)) deadline = e->deadline;
          }
          if (selected) break;
          if (stopping && !unfinished) {
            current_worker = nullptr;
            return;
          }
          if (deadline)
            changed.wait_until(lock, *deadline);
          else
            changed.wait(lock);
        }
      }
      step_result result{progress::runnable};
      try {
        result = selected->step();
      } catch (...) {
        // Failure belongs to this attempt. Its next step must still retire owned work.
        selected->fail(std::current_exception());
      }
      {
        std::lock_guard lock(mutex);
        selected->executing = false;
        if (result.next == progress::finished) {
          selected->finished = true;
          selected->runnable = false;
        } else if (result.next == progress::runnable || selected->runnable) {
          // A wake during the step wins over its parked result.
          selected->runnable = true;
        } else {
          selected->deadline = result.deadline;
        }
        // Drop the worker's ownership before acknowledging completion to drain waiters.
        selected.reset();
        changed.notify_all();
      }
    }
  }
};
thread_local coordinator_pool::state* coordinator_pool::state::current_worker = nullptr;

coordinator_pool::coordinator_pool(size_t workers) : state_(std::make_shared<state>(workers))
{
  workers_.reserve(workers);
}
coordinator_pool::~coordinator_pool() { stop(); }

void coordinator_pool::start_workers()
{
  if (!workers_.empty()) return;
  try {
    for (size_t i = 0; i < state_->worker_count; ++i) {
      if (before_thread_) before_thread_(i);
      workers_.emplace_back([st = state_] { st->run(); });
    }
    std::lock_guard lock(state_->mutex);
    state_->started = workers_.size();
  } catch (...) {
    {
      std::lock_guard lock(state_->mutex);
      state_->stopping = true;  // No registration has been accepted yet.
      state_->changed.notify_all();
    }
    for (auto& worker : workers_)
      worker.join();
    workers_.clear();
    {
      std::lock_guard lock(state_->mutex);
      state_->stopping = false;
    }
    throw;
  }
}

std::shared_ptr<coordinator_pool::registration> coordinator_pool::add(
  std::function<step_result()> step,
  std::function<void()> stop,
  std::function<void(std::exception_ptr)> fail)
{
  if (!step || !stop || !fail) throw std::invalid_argument("incomplete coordinator callbacks");
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  {
    std::lock_guard lock(state_->mutex);
    if (state_->stopping || !state_->worker_count)
      throw std::logic_error("coordinator pool is stopped or disabled");
  }
  auto e      = std::make_shared<entry>();
  e->step     = std::move(step);
  e->stop     = std::move(stop);
  e->fail     = std::move(fail);
  auto handle = std::shared_ptr<registration>(new registration(state_, e));
  start_workers();
  std::lock_guard lock(state_->mutex);
  auto slot = std::find(state_->entries.begin(), state_->entries.end(), nullptr);
  e->index  = static_cast<size_t>(slot - state_->entries.begin());
  if (slot == state_->entries.end())
    state_->entries.push_back(e);
  else
    *slot = e;
  e->generation = ++state_->next_generation;
  e->registered = true;
  return handle;
}

void coordinator_pool::stop()
{
  if (state::current_worker == state_.get())
    throw std::logic_error("coordinator worker cannot stop its own pool");
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  {
    std::lock_guard lock(state_->mutex);
    if (state_->stopping) return;
    state_->stopping = true;
    state_->changed.notify_all();
  }
  for (size_t i = 0;; ++i) {
    std::shared_ptr<entry> e;
    {
      std::lock_guard lock(state_->mutex);
      if (i == state_->entries.size()) break;
      e = state_->entries[i];
      if (!e || e->finished) continue;
      e->stop_notifying = true;
    }
    e->stop();  // Never invoke attempt code while holding the pool mutex.
    {
      std::lock_guard lock(state_->mutex);
      e->stop_notifying = false;
      if (!e->finished) e->runnable = true;
      e.reset();
      state_->changed.notify_all();
    }
  }
  for (auto& worker : workers_)
    worker.join();
  workers_.clear();
  std::lock_guard lock(state_->mutex);
  state_->started = 0;
}

size_t coordinator_pool::started_workers() const
{
  std::lock_guard lock(state_->mutex);
  return state_->started;
}
void coordinator_pool::before_thread_for_testing(std::function<void(size_t)> hook)
{
  std::lock_guard lifecycle_lock(lifecycle_mutex_);
  if (!workers_.empty() || state_->stopping) throw std::logic_error("pool startup seam frozen");
  before_thread_ = std::move(hook);
}
coordinator_pool::registration::registration(std::shared_ptr<state> st, std::shared_ptr<entry> e)
  : state_(std::move(st)), entry_(std::move(e))
{
}
coordinator_pool::registration::~registration()
{
  std::shared_ptr<entry> retired;
  {
    std::lock_guard lock(state_->mutex);
    if (!entry_->registered) return;
    assert(entry_->finished && !entry_->executing && !entry_->stop_notifying);
    retired = std::move(state_->entries[entry_->index]);
    state_->changed.notify_all();
  }
}
void coordinator_pool::wakeup::operator()() const noexcept
{
  if (auto st = state_.lock()) st->wake(index_, generation_);
}
coordinator_pool::wakeup coordinator_pool::registration::notifier() const noexcept
{
  wakeup result;
  result.state_      = state_;
  result.index_      = entry_->index;
  result.generation_ = entry_->generation;
  return result;
}
void coordinator_pool::registration::wake() noexcept { notifier()(); }
void coordinator_pool::registration::wait()
{
  if (state::current_worker == state_.get())
    throw std::logic_error("coordinator worker cannot wait for its own pool");
  std::unique_lock lock(state_->mutex);
  state_->changed.wait(lock, [&] { return entry_->finished && !entry_->stop_notifying; });
}
}  // namespace sirius::scan_manager
