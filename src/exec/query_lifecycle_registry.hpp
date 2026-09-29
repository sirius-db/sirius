/*
 * Copyright 2025, Sirius Contributors.
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

#include "query_id.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::exec {

enum class query_lifecycle_state : std::uint8_t { open, quiescing };

/// Captured at submission acquisition; callers must not re-read the registry to diagnose refusal.
enum class query_submission_status : std::uint8_t { accepted, quiescing, unknown };

/**
 * @brief Per-query submission gates and explicit work accounting.
 *
 * A submission guard registers a publisher atomically with checking that its query is open.
 * quiesce_and_wait_for_submissions() closes that gate and waits for admitted publishers BEFORE
 * callers drain queues. No lifecycle mutex is held while constructing, pushing or destroying work.
 * Every publication (including retries and transfers) must acquire a guard and retain it through
 * insertion. A refused publication must dispose of its work under the caller's existing ownership.
 *
 * A move-only work_lease can be taken from a submission and carried through queued/in-hand/running
 * work. Its count does not change on transfer. The control block can retain the execution's
 * physical plan independently of its front-end engine. Repository and scan-state ownership
 * stays in the runtime until publication, work and borrow retirement complete.
 *
 * Production tasks and creator requests retain leases through destruction and completion
 * callbacks. Spill candidates borrow their victim's query. Scan producers use their existing
 * scoped dispatcher join before providers are released. accepts_work() remains advisory.
 */
class query_lifecycle_registry {
  struct query_control {
    std::mutex mutex;
    std::condition_variable idle;
    query_lifecycle_state state{query_lifecycle_state::open};
    std::size_t submissions{0};
    std::size_t work{0};
    std::shared_ptr<void> resources;
  };

 public:
  class submission_guard;

  /// A claim on query-owned resources. Release only after the last callback/destructor/device use.
  /// The control block survives registry destruction; the execution owner must outlive its users.
  class work_lease {
   public:
    work_lease()                             = default;
    work_lease(const work_lease&)            = delete;
    work_lease& operator=(const work_lease&) = delete;
    work_lease(work_lease&& other) noexcept : control_(std::move(other.control_)) {}
    work_lease& operator=(work_lease&& other) noexcept
    {
      if (this != &other) {
        reset();
        control_ = std::move(other.control_);
      }
      return *this;
    }
    ~work_lease() { reset(); }

    explicit operator bool() const noexcept { return control_ != nullptr; }

    void reset() noexcept
    {
      if (auto control = std::exchange(control_, nullptr)) {
        bool notify;
        {
          std::lock_guard lock(control->mutex);
          --control->work;
          notify = control->state == query_lifecycle_state::quiescing && control->work == 0 &&
                   control->submissions == 0;
        }
        if (notify) { control->idle.notify_all(); }
      }
    }

   private:
    friend class query_lifecycle_registry;
    friend class submission_guard;
    // The acquiring operation increments the count while checking the gate under the same lock.
    explicit work_lease(std::shared_ptr<query_control> control) noexcept
      : control_(std::move(control))
    {
    }
    std::shared_ptr<query_control> control_;
  };

  /// Short-lived publisher registration. Keep through insertion (or abandoned-work destruction),
  /// never through task execution or a wait for worker/memory capacity. Destruction settles it.
  class submission_guard {
   public:
    submission_guard()                                   = default;
    submission_guard(const submission_guard&)            = delete;
    submission_guard& operator=(const submission_guard&) = delete;
    submission_guard(submission_guard&& other) noexcept
      : control_(std::move(other.control_)),
        owns_work_(std::exchange(other.owns_work_, false)),
        status_(std::exchange(other.status_, query_submission_status::unknown))
    {
    }
    submission_guard& operator=(submission_guard&& other) noexcept
    {
      if (this != &other) {
        finish();
        control_   = std::move(other.control_);
        owns_work_ = std::exchange(other.owns_work_, false);
        status_    = std::exchange(other.status_, query_submission_status::unknown);
      }
      return *this;
    }
    ~submission_guard() { finish(); }

    explicit operator bool() const noexcept { return control_ != nullptr; }
    [[nodiscard]] query_submission_status status() const noexcept { return status_; }

    /// Transfer the counted claim into a work item before publishing it. May be called once.
    [[nodiscard]] work_lease take_work_lease()
    {
      if (!control_ || !owns_work_) {
        throw std::logic_error("submission has no work lease to transfer");
      }
      owns_work_ = false;
      return work_lease{control_};
    }

    /// Publication is over. If no lease was transferred, release its work count as well.
    void finish() noexcept
    {
      if (auto control = std::exchange(control_, nullptr)) {
        bool notify;
        {
          std::lock_guard lock(control->mutex);
          // Both counts belong to this control block. Release them together, so a publisher
          // waiter cannot observe leftover untransferred work and we acquire the mutex once.
          if (std::exchange(owns_work_, false)) { --control->work; }
          --control->submissions;
          // This is the only transition here that can satisfy either wait predicate.
          notify = control->state == query_lifecycle_state::quiescing && control->submissions == 0;
        }
        if (notify) { control->idle.notify_all(); }
      }
    }

   private:
    friend class query_lifecycle_registry;
    explicit submission_guard(query_submission_status status) : status_(status) {}
    explicit submission_guard(std::shared_ptr<query_control> control) noexcept
      : control_(std::move(control)), owns_work_(true), status_(query_submission_status::accepted)
    {
    }
    std::shared_ptr<query_control> control_;
    // Avoid retaining two shared_ptr references until the work claim actually leaves the guard.
    bool owns_work_{false};
    query_submission_status status_{query_submission_status::unknown};
  };

  struct query_activity {
    std::size_t submissions{0};
    std::size_t work{0};
    std::shared_ptr<void> resources;
  };

  query_lifecycle_registry()                                           = default;
  query_lifecycle_registry(const query_lifecycle_registry&)            = delete;
  query_lifecycle_registry& operator=(const query_lifecycle_registry&) = delete;

  /// Register once before any producer runs. Duplicate registration is an error, including an
  /// attempt to reopen a quiescing query: reopening would invalidate cleanup's closed-gate proof.
  void open_query(sirius::query_id_t query_id)
  {
    std::lock_guard lock(mutex_);
    if (queries_.contains(query_id)) {
      throw std::logic_error("query lifecycle already registered");
    }
    queries_.emplace(query_id, std::make_shared<query_control>());
  }

  [[nodiscard]] submission_guard try_begin_submission(sirius::query_id_t query_id)
  {
    auto control = find(query_id);
    if (!control) { return submission_guard{query_submission_status::unknown}; }
    std::lock_guard lock(control->mutex);
    if (control->state != query_lifecycle_state::open) {
      return submission_guard{query_submission_status::quiescing};
    }
    ++control->submissions;
    ++control->work;
    return submission_guard{std::move(control)};
  }

  /// Register an independent resource borrower. For queue publication use try_begin_submission.
  [[nodiscard]] work_lease try_acquire_work(sirius::query_id_t query_id)
  {
    auto control = find(query_id);
    if (!control) { return {}; }
    std::lock_guard lock(control->mutex);
    if (control->state != query_lifecycle_state::open) { return {}; }
    ++control->work;
    return work_lease{std::move(control)};
  }

  /// Stop new submissions/borrows without waiting. Safe for a worker to call on its own query.
  /// Previously admitted publishers may still push until wait_for_submissions() has returned.
  void quiesce(sirius::query_id_t query_id)
  {
    if (auto control = find(query_id)) {
      std::lock_guard lock(control->mutex);
      control->state = query_lifecycle_state::quiescing;
    }
  }

  /// Query-owner barrier before queue drains. Never call while holding this query's submission
  /// guard, a queue lock, or another lock an admitted publisher needs. Unknown IDs are a no-op.
  void quiesce_and_wait_for_submissions(sirius::query_id_t query_id)
  {
    if (auto control = find(query_id)) {
      std::unique_lock lock(control->mutex);
      control->state = query_lifecycle_state::quiescing;
      control->idle.wait(lock, [&] { return control->submissions == 0; });
    }
  }

  /// Wait after quiesce, before draining queues. Exposed separately for cancellation coordinators.
  void wait_for_submissions(sirius::query_id_t query_id)
  {
    if (auto control = find(query_id)) {
      std::unique_lock lock(control->mutex);
      require_quiescing(*control);
      control->idle.wait(lock, [&] { return control->submissions == 0; });
    }
  }

  /// Wait after closing publication and disposing of queued work. This counts ONLY explicit
  /// leases, not unconverted users. Must not be called by a holder of one of this query's leases.
  void wait_for_work(sirius::query_id_t query_id)
  {
    if (auto control = find(query_id)) {
      std::unique_lock lock(control->mutex);
      require_quiescing(*control);
      control->idle.wait(lock, [&] { return control->submissions == 0 && control->work == 0; });
    }
  }

  /// Keep the execution's plan alive independently of its front-end engine. Registration
  /// precedes publication. Unknown IDs support standalone plan-building fixtures.
  void retain_resources(query_id_t query_id, std::shared_ptr<void> resources)
  {
    auto control = find(query_id);
    if (!control) { return; }
    std::lock_guard lock(control->mutex);
    if (control->state != query_lifecycle_state::open || control->resources) {
      throw std::logic_error("query resources already registered or retiring");
    }
    control->resources = std::move(resources);
  }

  /// Retire resources only after all asynchronous users, before repositories are cleared.
  /// Run destructors outside both locks: they can invoke callbacks into the runtime.
  void release_resources(query_id_t query_id)
  {
    std::shared_ptr<void> resources;
    if (auto control = find(query_id)) {
      std::lock_guard lock(control->mutex);
      require_quiescing(*control);
      require_idle(*control);
      resources = std::move(control->resources);
    }
  }

  /// Forget a retired query. Refuse to erase live accounting. Unknown IDs are a no-op, so failed
  /// initialization and repeated cleanup are supported. Existing drains are still mandatory.
  void close(sirius::query_id_t query_id)
  {
    std::shared_ptr<query_control> retired;
    {
      std::lock_guard registry_lock(mutex_);
      auto it = queries_.find(query_id);
      if (it == queries_.end()) { return; }
      retired = it->second;
      std::lock_guard lock(retired->mutex);
      require_idle(*retired);
      retired->state = query_lifecycle_state::quiescing;
      queries_.erase(it);
    }
  }

  /// Advisory snapshot only. Use a submission guard to authorize a subsequent queue push.
  [[nodiscard]] bool accepts_work(sirius::query_id_t query_id) const
  {
    return state(query_id) == query_lifecycle_state::open;
  }

  [[nodiscard]] std::optional<query_lifecycle_state> state(sirius::query_id_t query_id) const
  {
    auto control = find(query_id);
    if (!control) { return std::nullopt; }
    std::lock_guard lock(control->mutex);
    return control->state;
  }

  /// Diagnostics/test snapshot; an unknown query has zero activity.
  [[nodiscard]] query_activity activity(sirius::query_id_t query_id) const
  {
    auto control = find(query_id);
    if (!control) { return {}; }
    std::lock_guard lock(control->mutex);
    return {control->submissions, control->work};
  }

  [[nodiscard]] std::size_t size() const
  {
    std::lock_guard lock(mutex_);
    return queries_.size();
  }

  /// Shutdown: close every acquisition gate before stopping shared workers.
  void quiesce_all()
  {
    std::lock_guard registry_lock(mutex_);
    for (auto const& [id, control] : queries_) {
      std::lock_guard lock(control->mutex);
      control->state = query_lifecycle_state::quiescing;
    }
  }

  /// Runtime teardown only, after producers and workers have stopped. Close every gate before
  /// checking counts, including controls already obtained by callers racing a registry lookup.
  void clear()
  {
    decltype(queries_) retired;
    {
      std::lock_guard registry_lock(mutex_);
      for (auto const& [id, control] : queries_) {
        std::lock_guard lock(control->mutex);
        control->state = query_lifecycle_state::quiescing;
      }
      for (auto const& [id, control] : queries_) {
        std::lock_guard lock(control->mutex);
        require_idle(*control);
      }
      retired.swap(queries_);
    }
  }

 private:
  [[nodiscard]] std::shared_ptr<query_control> find(sirius::query_id_t query_id) const
  {
    // This mutex protects map membership, not the control's state/counters. Keep its scope
    // separate from the query lock so activity on one query does not hold up other lookups.
    std::lock_guard lock(mutex_);
    auto it = queries_.find(query_id);
    return it == queries_.end() ? nullptr : it->second;
  }

  static void require_quiescing(const query_control& control)
  {
    if (control.state != query_lifecycle_state::quiescing) {
      throw std::logic_error("query must be quiescing before waiting for activity");
    }
  }

  static void require_idle(const query_control& control)
  {
    if (control.submissions != 0 || control.work != 0) {
      throw std::logic_error("query lifecycle still has outstanding submissions or work");
    }
  }

  // Lock order: registry -> query control. Handles only lock their control; no queue or callback
  // runs under either mutex. Waiters release the registry lock before waiting on a query's CV.
  mutable std::mutex mutex_;
  std::map<sirius::query_id_t, std::shared_ptr<query_control>> queries_;
};

}  // namespace sirius::exec
