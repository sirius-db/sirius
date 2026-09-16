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

#include <absl/functional/any_invocable.h>

#include <functional>
#include <memory>
#include <stop_token>
#include <utility>

namespace sirius::exec {

// ---------------------------------------------------------------------------
// completion_token — opaque RAII handle owning a completion subscription
// ---------------------------------------------------------------------------
//
// Wraps a std::stop_callback bound to a completion_controller's stop_token.
// The (perfect-forwarded) callable and its bound arguments are type-erased, so
// completion_token is a single concrete type regardless of what was
// registered, and can be handed around behind a std::unique_ptr.  Destroying
// the token deregisters the callback.  It has no public API beyond
// construction — it exists only to own the subscription's lifetime.
//
// std::stop_callback gives the one-shot firing and the register-after-already-
// fired behaviour (fires immediately, on the constructing thread) for free.

class completion_token {
 public:
  completion_token() = default;

  template <typename Fn, typename... Args>
  completion_token(std::stop_token token, Fn&& fn, Args&&... args)
    : _cb(std::make_unique<impl_type>(
        std::move(token),
        [fn = std::forward<Fn>(fn), ... args = std::forward<Args>(args)]() mutable {
          std::invoke(fn, args...);
        }))
  {
  }

  // Naturally movable now because std::unique_ptr is movable!
  completion_token(completion_token&&) noexcept            = default;
  completion_token& operator=(completion_token&&) noexcept = default;

  completion_token(const completion_token&)            = delete;
  completion_token& operator=(const completion_token&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept { return _cb != nullptr; }

  [[nodiscard]] bool is_armed() const noexcept { return _cb != nullptr; }

 private:
  using impl_type = std::stop_callback<absl::AnyInvocable<void()>>;
  std::unique_ptr<impl_type> _cb;
};

// ---------------------------------------------------------------------------
// completion_controller — unbounded slot tracking with one-shot completion
// ---------------------------------------------------------------------------
//
// Hands out RAII `slot`s that track in-flight work; completion fires once,
// when the last slot has been released AND the producer has signalled it is
// done issuing work.
//
// Lifetime-based implementation: a shared ctrl_impl owns the stop_source, and
// its destructor requests stop.  Both the controller and every outstanding
// slot hold a shared_ptr to it, so ctrl_impl — and therefore the
// not-yet-stopped state — stays alive until:
//   - close() drops the controller's own reference, and
//   - the last slot is destroyed.
// Whichever happens last destroys ctrl_impl and fires the registered
// callbacks.  If close() is never called, the controller's own reference is
// dropped by its destructor instead, which serves as a backstop.
//
// Usage:
//     auto tok = controller.on_completion([] { /* all work drained */ });
//     for (auto& w : work) scheduler.enqueue(make_task(w, controller.acquire()));
//     controller.close();   // completion can now fire once the slots drain
//
// Threading: slot destruction (release) is thread-safe (atomic refcount), so
// slots may be dropped from any thread.  acquire()/close() mutate the
// controller's own shared_ptr and are expected to be called from the single
// producer thread (not concurrently with each other).  on_completion() and
// completed() read only the cached stop_token and are always safe.

class completion_controller {
 private:
  struct ctrl_impl {
    ~ctrl_impl() { _stop.request_stop(); }
    std::stop_source _stop;
  };

 public:
  // RAII handle representing one unit of in-flight work.  Holds a strong
  // reference to ctrl_impl, so an outstanding slot keeps completion pending.
  class slot {
   public:
    slot() = default;

    slot(slot&&) noexcept            = default;
    slot& operator=(slot&&) noexcept = default;

    slot(slot const&)            = delete;
    slot& operator=(slot const&) = delete;

    /// True if this slot holds a live reservation.
    explicit operator bool() const noexcept { return _impl != nullptr; }

   private:
    friend class completion_controller;
    explicit slot(std::shared_ptr<ctrl_impl> impl) noexcept : _impl(std::move(impl)) {}

    std::shared_ptr<ctrl_impl> _impl;
  };

  completion_controller() : _impl(std::make_shared<ctrl_impl>()), _token(_impl->_stop.get_token())
  {
  }

  completion_controller(completion_controller const&)            = delete;
  completion_controller& operator=(completion_controller const&) = delete;

  /// Hand out a slot, extending ctrl_impl's lifetime.  Never blocks.  After
  /// close(), no new work can be tracked, so this returns a disengaged slot.
  [[nodiscard]] slot acquire() { return slot{_impl}; }

  /// Signal that no more work will be issued by dropping the controller's own
  /// reference to ctrl_impl.  Completion fires now if no slots are
  /// outstanding, otherwise when the last one drains.  Idempotent.
  void close() noexcept { _impl.reset(); }

  /// Register @p fn (bound to @p args) to run exactly once when completion
  /// fires.  The returned handle owns the subscription: keep it alive for as
  /// long as the callback should stay armed; destroying it deregisters the
  /// callback.  If completion has already fired, the callback runs immediately
  /// on the calling thread before this returns.
  template <typename Fn, typename... Args>
  [[nodiscard]] completion_token on_completion(Fn&& fn, Args&&... args)
  {
    return completion_token(_token, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  /// True once completion has been signalled.
  [[nodiscard]] bool completed() const noexcept { return _token.stop_requested(); }

 private:
  std::shared_ptr<ctrl_impl> _impl;
  std::stop_token _token;
};

}  // namespace sirius::exec
