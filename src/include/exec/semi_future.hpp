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
 * this code has been minimally adopted from folly's SemiFuture implementation, with some
 * modifications to fit our needs.
 */

#pragma once

#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "exec/try.hpp"

#include <absl/functional/any_invocable.h>

namespace sirius::exec {

// Task type submitted to an executor.
using executor_func = absl::AnyInvocable<void() &&>;

// Concept: any type with a void enqueue(executor_func) member.
template <class exec_t>
concept executor_concept = requires(exec_t& e, executor_func f) {
  { e.enqueue(std::move(f)) } -> std::same_as<void>;
};

// Synchronous executor: runs each task immediately on the calling thread.
class inline_executor {
 public:
  void enqueue(executor_func task) { std::move(task)(); }

  static inline_executor* instance() noexcept
  {
    static inline_executor inst;
    return &inst;
  }
};

// ---------------------------------------------------------------------------
// Exceptions

class broken_promise_error : public std::runtime_error {
 public:
  broken_promise_error() : std::runtime_error("sf::promise destroyed without being satisfied") {}
};

class promise_already_satisfied_error : public std::logic_error {
 public:
  promise_already_satisfied_error() : std::logic_error("sf::promise already satisfied") {}
};

class future_already_retrieved_error : public std::logic_error {
 public:
  future_already_retrieved_error()
    : std::logic_error("sf::semi_future already retrieved from promise")
  {
  }
};

class future_invalid_error : public std::logic_error {
 public:
  future_invalid_error() : std::logic_error("sf::semi_future is invalid (moved-from or empty)") {}
};

// Used to disambiguate exception_type-tagged overloads.
template <class value_t>
struct tag_t {};
template <class value_t>
inline constexpr tag_t<value_t> tag{};

// Forward decls.
template <class value_t>
class semi_future;
template <class value_t>
class promise;
template <class value_t, executor_concept exec_t>
class future;

namespace detail {

// Detect std::unique_ptr specializations.  Used to keep the semi_future
// value-constructor from hijacking the internal `unique_ptr<state<T>>`
// constructor: when value_t is constructible from a pointer (e.g. bool, via
// unique_ptr's explicit operator bool), the forwarding-reference value
// constructor is an exact match and would otherwise win over the state
// constructor, silently collapsing a deferred chain into a ready_state.
template <class t>
struct is_unique_ptr : std::false_type {};
template <class t, class d>
struct is_unique_ptr<std::unique_ptr<t, d>> : std::true_type {};
template <class t>
inline constexpr bool is_unique_ptr_v = is_unique_ptr<t>::value;

template <class value_t>
struct is_semi_future : std::false_type {};
template <class value_t>
struct is_semi_future<semi_future<value_t>> : std::true_type {
  using value_type = value_t;
};
template <class value_t>
inline constexpr bool is_semi_future_v = is_semi_future<value_t>::value;

template <class value_t>
struct is_try : std::false_type {};
template <class value_t>
struct is_try<try_t<value_t>> : std::true_type {
  using value_type = value_t;
};
template <class value_t>
inline constexpr bool is_try_v = is_try<value_t>::value;

// Result type extraction: collapses try_t<wrap_t> and semi_future<wrap_t> down
// to wrap_t.
template <class r_t>
struct unwrap_callback_result {
  using type = r_t;
};
template <class wrap_t>
struct unwrap_callback_result<try_t<wrap_t>> {
  using type = wrap_t;
};
template <class wrap_t>
struct unwrap_callback_result<semi_future<wrap_t>> {
  using type = wrap_t;
};
template <class r_t>
using unwrap_callback_result_t = typename unwrap_callback_result<r_t>::type;

// ---------------------------------------------------------------------------
// Timed futex wait — blocks until atom != expected or deadline is reached.
//
// Uses FUTEX_WAIT_BITSET with an absolute CLOCK_MONOTONIC deadline (matching
// std::chrono::steady_clock), identical to folly's futexWaitUntil approach.
// The reinterpret_cast is safe on Linux: std::atomic<uint32_t> is a standard-
// layout type whose address equals the address of its contained uint32_t.
// Returns true if the value changed, false on timeout.

inline bool futex_wait_until(std::atomic<std::uint32_t>& atom,
                             std::uint32_t expected,
                             std::chrono::steady_clock::time_point deadline)
{
#ifdef __linux__
  static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t),
                "futex requires atomic<uint32_t> to have no overhead storage");
  // Linux futex constants (stable ABI; avoids pulling in <linux/futex.h>).
  constexpr unsigned futex_wait_bitset      = 9U;
  constexpr unsigned futex_private_flag     = 128U;
  constexpr unsigned futex_bitset_match_any = 0xFFFFFFFFU;

  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch());
  struct timespec ts{static_cast<time_t>(ns.count() / 1'000'000'000LL),
                     static_cast<long>(ns.count() % 1'000'000'000LL)};
  // FUTEX_WAIT_BITSET with no FUTEX_CLOCK_REALTIME → absolute CLOCK_MONOTONIC.
  syscall(SYS_futex,
          reinterpret_cast<std::uint32_t*>(&atom),
          static_cast<int>(futex_wait_bitset | futex_private_flag),
          expected,
          &ts,
          nullptr,
          futex_bitset_match_any);
  return errno != ETIMEDOUT;
#else
  // Non-Linux fallback: spin-yield (rare; add a platform futex port if needed).
  while (atom.load(std::memory_order_relaxed) == expected) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::yield();
  }
  return true;
#endif
}

// ---------------------------------------------------------------------------
// core: the shared slot between promise and semi_future/future.
//
// Lock-free FSM with four states, one-shot:
//
//                     set_try()        set_callback()
//     empty ─────────────────► has_result ────────────────► done
//        │                                                     ▲
//        │                                                     │
//        └──────── set_callback() ──► has_callback ── set_try()──┘
//
// Whichever of set_try() / set_callback() arrives second is the one that fires
// the callback (on its executor or inline). Pull-mode waiters spin on
// std::atomic<uint32_t>::wait, which compiles to a futex on Linux.
//
// Single-consumer model: either pull (wait+steal) OR push (set_callback), not
// both, per core instance.

template <class value_t>
class core {
 public:
  using callback = absl::AnyInvocable<void(try_t<value_t>&&) &&>;

  core()                       = default;
  core(core const&)            = delete;
  core& operator=(core const&) = delete;

  [[nodiscard]] bool ready() const noexcept
  {
    auto s = _state.load(std::memory_order_acquire);
    return s == has_result || s == done;
  }

  // Producer side: deposit the result. Fires the callback if one is waiting.
  void set_try(try_t<value_t>&& t)
  {
    _try                   = std::move(t);
    std::uint32_t expected = empty;
    if (_state.compare_exchange_strong(
          expected, has_result, std::memory_order_acq_rel, std::memory_order_acquire)) {
      // Pull-mode consumer (if any) is woken; push-mode set_callback (if it
      // arrives later) will fire inline.
      _state.notify_all();
      return;
    }
    // The only other valid prior state is has_callback: a callback was
    // already installed and is waiting for us to deliver the value.
    assert(expected == has_callback);
    run_and_complete();
  }

  // Consumer side (push mode): install a callback to fire when ready.
  void set_callback(callback cb)
  {
    _callback              = std::move(cb);
    std::uint32_t expected = empty;
    if (_state.compare_exchange_strong(
          expected, has_callback, std::memory_order_acq_rel, std::memory_order_acquire)) {
      // Producer hasn't fired yet; they'll do it when set_try() runs.
      return;
    }
    // Result was already deposited; fire now.
    assert(expected == has_result);
    run_and_complete();
  }

  // Pull-mode block until ready. Idempotent.
  void wait()
  {
    while (true) {
      auto s = _state.load(std::memory_order_acquire);
      if (s == has_result || s == done) { return; }
      _state.wait(s, std::memory_order_acquire);
    }
  }

  template <class rep, class period>
  bool wait_for(std::chrono::duration<rep, period> const& dur)
  {
    return wait_until(std::chrono::steady_clock::now() + dur);
  }

  bool wait_until(std::chrono::steady_clock::time_point deadline)
  {
    while (true) {
      auto s = _state.load(std::memory_order_acquire);
      if (s == has_result || s == done) { return true; }
      if (std::chrono::steady_clock::now() >= deadline) { return false; }
      // Blocks until _state != s or deadline passes; no busy spinning.
      if (!detail::futex_wait_until(_state, s, deadline)) { return false; }
    }
  }

  // Caller must have observed ready() == true. Single-shot; mutually
  // exclusive with set_callback().
  try_t<value_t> steal() noexcept(std::is_nothrow_move_constructible_v<try_t<value_t>>)
  {
    return std::move(_try);
  }

 private:
  static constexpr std::uint32_t empty        = 0;
  static constexpr std::uint32_t has_result   = 1;
  static constexpr std::uint32_t has_callback = 2;
  static constexpr std::uint32_t done         = 3;

  void run_and_complete()
  {
    // Both _try and _callback are set. The callback owns any executor dispatch
    // logic (captured at future<T,exec_t>::then_try time), so call it directly.
    auto cb = std::move(_callback);
    std::move(cb)(std::move(_try));
    _state.store(done, std::memory_order_release);
    _state.notify_all();
  }

  std::atomic<std::uint32_t> _state{empty};
  try_t<value_t> _try;
  callback _callback;
};

// ---------------------------------------------------------------------------
// state<value_t>: type-erased view of a chain that eventually yields a
// try_t<value_t>.
//
// Two consumption modes (mutually exclusive, exactly one per state):
//   * Pull:  await() + consume() — caller blocks and reads.
//   * Push:  set_callback(exe, cb) — cb is invoked with the final
//   try_t<value_t> when
//            the chain completes, scheduled on `exe` (or inline if nullptr).
//
// state chains route both modes recursively: each link wraps cb with its
// local transformation before installing on its upstream.

template <class value_t>
struct state {
  using callback = absl::AnyInvocable<void(try_t<value_t>&&) &&>;

  virtual ~state()                                                         = default;
  virtual void await()                                                     = 0;
  virtual bool await_until(std::chrono::steady_clock::time_point deadline) = 0;
  [[nodiscard]] virtual bool peek_ready() const noexcept                   = 0;
  virtual try_t<value_t> consume()                                         = 0;
  virtual void set_callback(callback cb)                                   = 0;
};

// Leaf state: backed by a core shared with a promise.
template <class value_t>
class leaf_state final : public state<value_t> {
 public:
  using callback = typename state<value_t>::callback;

  explicit leaf_state(std::shared_ptr<core<value_t>> core) noexcept : _core(std::move(core)) {}

  void await() override { _core->wait(); }
  bool await_until(std::chrono::steady_clock::time_point deadline) override
  {
    return _core->wait_until(deadline);
  }
  [[nodiscard]] bool peek_ready() const noexcept override { return _core->ready(); }
  try_t<value_t> consume() override
  {
    _core->wait();
    return _core->steal();
  }
  void set_callback(callback cb) override { _core->set_callback(std::move(cb)); }

 private:
  std::shared_ptr<core<value_t>> _core;
};

// Ready state: already-completed value, no synchronization needed.
template <class value_t>
class ready_state final : public state<value_t> {
 public:
  using callback = typename state<value_t>::callback;

  explicit ready_state(try_t<value_t> t) noexcept(
    std::is_nothrow_move_constructible_v<try_t<value_t>>)
    : _try(std::move(t))
  {
  }

  void await() override {}
  bool await_until(std::chrono::steady_clock::time_point) override { return true; }
  [[nodiscard]] bool peek_ready() const noexcept override { return true; }
  try_t<value_t> consume() override { return std::move(_try); }
  void set_callback(callback cb) override { std::move(cb)(std::move(_try)); }

 private:
  try_t<value_t> _try;
};

// Chained state: upstream state<upstream_t> + fn_t: try_t<upstream_t>&& ->
// try_t<value_t>.
template <class value_t, class upstream_t, class fn_t>
class deferred_state final : public state<value_t> {
 public:
  using callback = typename state<value_t>::callback;

  deferred_state(std::unique_ptr<state<upstream_t>> upstream, fn_t&& fn)
    : _upstream(std::move(upstream)), _fn(std::move(fn))
  {
  }

  void await() override { _upstream->await(); }
  bool await_until(std::chrono::steady_clock::time_point deadline) override
  {
    return _upstream->await_until(deadline);
  }
  [[nodiscard]] bool peek_ready() const noexcept override { return _upstream->peek_ready(); }
  try_t<value_t> consume() override
  {
    try {
      return std::invoke(std::move(_fn), _upstream->consume());
    } catch (...) {
      return try_t<value_t>(std::current_exception());
    }
  }

  void set_callback(callback cb) override
  {
    auto wrapper = [fn = std::move(_fn), cb = std::move(cb)](try_t<upstream_t>&& u) mutable {
      try_t<value_t> result;
      try {
        result = std::invoke(std::move(fn), std::move(u));
      } catch (...) {
        result = try_t<value_t>(std::current_exception());
      }
      std::move(cb)(std::move(result));
    };
    _upstream->set_callback(std::move(wrapper));
  }

 private:
  std::unique_ptr<state<upstream_t>> _upstream;
  fn_t _fn;
};

// Flat-map state: fn_t: try_t<upstream_t>&& -> semi_future<value_t>. Awaits the
// inner future, caches its result so wait/get are idempotent (pull-mode).
template <class value_t, class upstream_t, class fn_t>
class flat_map_state final : public state<value_t> {
 public:
  using callback = typename state<value_t>::callback;

  flat_map_state(std::unique_ptr<state<upstream_t>> upstream, fn_t&& fn)
    : _upstream(std::move(upstream)), _fn(std::move(fn))
  {
  }

  void await() override
  {
    if (_cached) { return; }
    try_t<upstream_t> in = _upstream->consume();
    try {
      semi_future<value_t> inner = std::invoke(std::move(_fn), std::move(in));
      _cached.emplace(std::move(inner).get_try());
    } catch (...) {
      _cached.emplace(try_t<value_t>(std::current_exception()));
    }
  }

  // Timed wait: block on upstream with the deadline. Once upstream is consumed
  // and fn invoked we cannot un-consume it, so the inner wait always blocks
  // without a timeout (it must complete to keep the cached result consistent).
  bool await_until(std::chrono::steady_clock::time_point deadline) override
  {
    if (_cached) { return true; }
    if (!_upstream->await_until(deadline)) { return false; }
    await();  // upstream is ready; run fn + wait on inner (no further timeout)
    return true;
  }

  [[nodiscard]] bool peek_ready() const noexcept override { return _cached.has_value(); }

  try_t<value_t> consume() override
  {
    if (!_cached) await();
    return std::move(*_cached);
  }

  void set_callback(callback cb) override;  // defined below

 private:
  std::unique_ptr<state<upstream_t>> _upstream;
  fn_t _fn;
  std::optional<try_t<value_t>> _cached;
};

// Helper to invoke fn_t with the value of a try_t<upstream_t>, accommodating
// upstream_t == void.
template <class fn_t, class upstream_t>
decltype(auto) invoke_with_value(fn_t&& f, try_t<upstream_t>&& t)
{
  if constexpr (std::is_void_v<upstream_t>) {
    (void)t;  // t carries no value when upstream_t is void
    return std::invoke(std::forward<fn_t>(f));
  } else {
    return std::invoke(std::forward<fn_t>(f), std::move(t).value());
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// semi_future<value_t>

template <class value_t>
class semi_future {
  static_assert(!std::is_reference_v<value_t>, "semi_future<value_t&> is not supported");

 public:
  using value_type = value_t;

  semi_future() noexcept = default;

  // Construct from a ready value (perfect forwarding, sans value_t==void).
  template <class upstream_t      = value_t,
            std::enable_if_t<!std::is_void_v<upstream_t> &&
                               std::is_constructible_v<value_t, upstream_t&&> &&
                               !std::is_same_v<std::decay_t<upstream_t>, semi_future<value_t>> &&
                               !detail::is_unique_ptr_v<std::decay_t<upstream_t>>,
                             int> = 0>
  /* implicit */ semi_future(upstream_t&& v)
    : _state(
        std::make_unique<detail::ready_state<value_t>>(try_t<value_t>(std::forward<upstream_t>(v))))
  {
  }

  semi_future(semi_future&&) noexcept            = default;
  semi_future& operator=(semi_future&&) noexcept = default;
  semi_future(semi_future const&)                = delete;
  semi_future& operator=(semi_future const&)     = delete;

  // ----- state inspection -----

  [[nodiscard]] bool valid() const noexcept { return _state != nullptr; }

  // True if the upstream value is ready. With deferred work in the chain,
  // peek_ready() does NOT imply the chained transformations have run; call
  // get()/get_try() to consume.
  [[nodiscard]] bool is_ready() const
  {
    throw_if_invalid();
    return _state->peek_ready();
  }

  // ----- consumption -----

  // Blocks until ready, runs the deferred chain inline, and returns value_t (or
  // rethrows the captured exception).
  value_t get() &&
  {
    throw_if_invalid();
    _state->await();
    auto t = _state->consume();
    _state.reset();
    return std::move(t).get();
  }

  // As above but with a timeout. Throws std::future_error / TimeoutError on
  // expiry. For minimal scope we throw std::runtime_error.
  template <class rep, class period>
  value_t get(std::chrono::duration<rep, period> const& dur) &&
  {
    throw_if_invalid();
    if (!wait_for(dur)) {
      _state.reset();
      throw std::runtime_error("sf::semi_future::get timed out");
    }
    auto t = _state->consume();
    _state.reset();
    return std::move(t).get();
  }

  // Returns the try_t (value-or-exception), never throws from the captured
  // exception itself.
  try_t<value_t> get_try() &&
  {
    throw_if_invalid();
    _state->await();
    auto t = _state->consume();
    _state.reset();
    return t;
  }

  // Block until ready. Does not consume the future.
  semi_future<value_t>& wait() &
  {
    throw_if_invalid();
    _state->await();
    return *this;
  }
  semi_future<value_t>&& wait() &&
  {
    throw_if_invalid();
    _state->await();
    return std::move(*this);
  }
  template <class rep, class period>
  bool wait(std::chrono::duration<rep, period> const& dur) &&
  {
    throw_if_invalid();
    return wait_for(dur);
  }

  // ----- composition -----

  // defer: fn_t takes try_t<value_t>&& and returns upstream_t,
  // try_t<upstream_t>, or semi_future<upstream_t>.
  template <class fn_t>
  auto defer(fn_t&& f) &&
  {
    throw_if_invalid();
    using decayed_f = std::decay_t<fn_t>;
    using r_t       = std::invoke_result_t<decayed_f, try_t<value_t>&&>;
    using inner_t   = detail::unwrap_callback_result_t<r_t>;

    if constexpr (detail::is_try_v<r_t>) {
      // fn_t: try_t<value_t>&& -> try_t<inner_t>. Use directly.
      auto st = std::make_unique<detail::deferred_state<inner_t, value_t, decayed_f>>(
        std::move(_state), std::forward<fn_t>(f));
      return semi_future<inner_t>(std::move(st));
    } else if constexpr (detail::is_semi_future_v<r_t>) {
      // fn_t: try_t<value_t>&& -> semi_future<inner_t>. Flatten.
      auto st = std::make_unique<detail::flat_map_state<inner_t, value_t, decayed_f>>(
        std::move(_state), std::forward<fn_t>(f));
      return semi_future<inner_t>(std::move(st));
    } else {
      // fn_t: try_t<value_t>&& -> inner_t (raw value, including void). Wrap
      // into try_t<inner_t>.
      auto wrapper = [fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable -> try_t<inner_t> {
        if constexpr (std::is_void_v<inner_t>) {
          std::invoke(std::move(fn), std::move(t));
          return try_t<void>::make_value();
        } else {
          return try_t<inner_t>(std::invoke(std::move(fn), std::move(t)));
        }
      };
      using wrap_t = decltype(wrapper);
      auto st      = std::make_unique<detail::deferred_state<inner_t, value_t, wrap_t>>(
        std::move(_state), std::move(wrapper));
      return semi_future<inner_t>(std::move(st));
    }
  }

  // defer_value: fn_t takes value_t&& (or no args if value_t == void). If
  // upstream produced an exception, the exception is propagated and fn_t is not
  // invoked.
  template <class fn_t>
  auto defer_value(fn_t&& f) &&
  {
    return std::move(*this).defer([fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable {
      if (t.has_exception()) { std::rethrow_exception(t.exception()); }
      return detail::invoke_with_value(std::move(fn), std::move(t));
    });
  }

  // defer_error (typed): invoke fn_t(exc_t const&) only if the captured
  // exception is (or derives from) exc_t. Otherwise propagate unchanged.
  template <class exception_type, class fn_t>
  semi_future<value_t> defer_error(tag_t<exception_type>, fn_t&& f) &&
  {
    return std::move(*this).defer(
      [fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable -> try_t<value_t> {
        if (!t.has_exception()) return std::move(t);
        try {
          std::rethrow_exception(t.exception());
        } catch (exception_type const& e) {
          if constexpr (std::is_void_v<value_t>) {
            std::invoke(std::move(fn), e);
            return try_t<void>::make_value();
          } else {
            return try_t<value_t>(std::invoke(std::move(fn), e));
          }
        } catch (...) {
          return std::move(t);
        }
      });
  }

  // defer_error (un-tagged): invoke fn_t(std::exception_ptr) on any exception.
  template <class fn_t>
  semi_future<value_t> defer_error(fn_t&& f) &&
  {
    return std::move(*this).defer(
      [fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable -> try_t<value_t> {
        if (!t.has_exception()) return std::move(t);
        if constexpr (std::is_void_v<value_t>) {
          std::invoke(std::move(fn), t.exception());
          return try_t<void>::make_value();
        } else {
          return try_t<value_t>(std::invoke(std::move(fn), t.exception()));
        }
      });
  }

  // defer_ensure: invoke fn_t() unconditionally before the result propagates.
  // If fn_t throws, the thrown exception replaces the original try_t.
  template <class fn_t>
  semi_future<value_t> defer_ensure(fn_t&& f) &&
  {
    return std::move(*this).defer(
      [fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable -> try_t<value_t> {
        std::invoke(std::move(fn));
        return std::move(t);
      });
  }

  // unit: discard the value, propagate exceptions, yield semi_future<void>.
  semi_future<void> unit() &&
  {
    return std::move(*this).defer([](try_t<value_t>&& t) -> try_t<void> {
      if (t.has_exception()) { return try_t<void>(t.exception()); }
      return try_t<void>::make_value();
    });
  }

  // Attach an executor: returns a future<value_t, exec_t>. Once via() returns,
  // the chain is "live" — when the upstream value lands, the deferred chain
  // runs and future-chained thenX stages are scheduled onto `exe`.
  template <executor_concept exec_t>
  future<value_t, exec_t> via(exec_t* exe) &&;  // defined below

  // Push-mode entry point. Installs `cb` as the chain's terminal callback;
  // when the chain produces its final try_t<value_t>, cb is invoked inline.
  // Any executor dispatch must be baked into cb by the caller.
  // Used by future<value_t, exec_t>::thenX and flat_map_state's flattening.
  template <class cb_t>
  void install_callback(cb_t&& cb) &&
  {
    throw_if_invalid();
    _state->set_callback(typename detail::state<value_t>::callback(std::forward<cb_t>(cb)));
    _state.reset();
  }

  // Internal: construct from a pre-built state. Users should not call this
  // directly; the state type lives in the detail namespace.
  explicit semi_future(std::unique_ptr<detail::state<value_t>> s) noexcept : _state(std::move(s)) {}

 private:
  template <class>
  friend class semi_future;
  template <class, executor_concept>
  friend class future;

  void throw_if_invalid() const
  {
    if (!_state) { throw future_invalid_error(); }
  }

  template <class rep, class period>
  bool wait_for(std::chrono::duration<rep, period> const& dur)
  {
    return _state->await_until(std::chrono::steady_clock::now() + dur);
  }

  std::unique_ptr<detail::state<value_t>> _state;
};

// ---------------------------------------------------------------------------
// promise<value_t>

template <class value_t>
class promise {
 public:
  promise() : _core(std::make_shared<detail::core<value_t>>()) {}

  promise(promise&&) noexcept            = default;
  promise& operator=(promise&&) noexcept = default;
  promise(promise const&)                = delete;
  promise& operator=(promise const&)     = delete;

  ~promise()
  {
    // If the producer dropped the promise without satisfying it, surface a
    // broken_promise_error to the consumer.
    if (_core && !_core->ready()) {
      _core->set_try(try_t<value_t>(std::make_exception_ptr(broken_promise_error())));
    }
  }

  semi_future<value_t> get_semi_future()
  {
    if (_retrieved) { throw future_already_retrieved_error(); }
    if (!_core) { throw promise_already_satisfied_error(); }
    _retrieved = true;
    return semi_future<value_t>(std::make_unique<detail::leaf_state<value_t>>(_core));
  }

  // Convenience: get a future<value_t, exec_t> directly, bound to `exe`.
  // Equivalent to get_semi_future().via(exe).
  template <executor_concept exec_t>
  future<value_t, exec_t> get_future(exec_t* exe);  // defined below

  template <class upstream_t = value_t, std::enable_if_t<!std::is_void_v<upstream_t>, int> = 0>
  void set_value(upstream_t&& v)
  {
    check_settable();
    _core->set_try(try_t<value_t>(std::forward<upstream_t>(v)));
    _core.reset();
  }

  template <class upstream_t = value_t, std::enable_if_t<std::is_void_v<upstream_t>, int> = 0>
  void set_value()
  {
    check_settable();
    _core->set_try(try_t<void>::make_value());
    _core.reset();
  }

  void set_exception(std::exception_ptr e)
  {
    check_settable();
    _core->set_try(try_t<value_t>(std::move(e)));
    _core.reset();
  }

  template <class exc_t>
  void set_exception(exc_t&& e)
  {
    set_exception(std::make_exception_ptr(std::forward<exc_t>(e)));
  }

  void set_try(try_t<value_t>&& t)
  {
    check_settable();
    _core->set_try(std::move(t));
    _core.reset();
  }

  // Invoke f and capture value-or-exception in this promise.
  template <class fn_t>
  void set_with(fn_t&& f)
  {
    check_settable();
    _core->set_try(make_try_with(std::forward<fn_t>(f)));
    _core.reset();
  }

  [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(_core); }

 private:
  void check_settable()
  {
    if (!_core) { throw promise_already_satisfied_error(); }
  }

  std::shared_ptr<detail::core<value_t>> _core;
  bool _retrieved = false;
};

// ---------------------------------------------------------------------------
// future<value_t, exec_t>: semi_future bound to a concrete executor type.
//
// exec_t must satisfy executor_concept (has enqueue(executor_func)).
// then_value/then_try/then_error/then_ensure install push-mode callbacks on the
// underlying chain. Each new stage gets its own task on the executor (the
// per-stage tasking comes from each thenX creating a fresh promise→future
// pair whose core schedules callbacks on the executor).

template <class value_t, executor_concept exec_t>
class future {
  static_assert(!std::is_reference_v<value_t>, "future<value_t&> is not supported");

 public:
  using value_type    = value_t;
  using executor_type = exec_t;

  future() noexcept = default;

  future(future&&) noexcept            = default;
  future& operator=(future&&) noexcept = default;
  future(future const&)                = delete;
  future& operator=(future const&)     = delete;

  [[nodiscard]] bool valid() const noexcept { return _state != nullptr; }
  exec_t* get_executor() const noexcept { return _executor; }

  [[nodiscard]] bool is_ready() const
  {
    throw_if_invalid();
    return _state->peek_ready();
  }

  // Convert back to a semi_future (drops the executor binding).
  semi_future<value_t> semi() &&
  {
    throw_if_invalid();
    auto sf = semi_future<value_t>(std::move(_state));
    return sf;
  }

  // Block and return the value. Uses pull-mode on the underlying chain.
  value_t get() &&
  {
    throw_if_invalid();
    _state->await();
    auto t = _state->consume();
    _state.reset();
    return std::move(t).get();
  }

  try_t<value_t> get_try() &&
  {
    throw_if_invalid();
    _state->await();
    auto t = _state->consume();
    _state.reset();
    return t;
  }

  future<value_t, exec_t>& wait() &
  {
    throw_if_invalid();
    _state->await();
    return *this;
  }
  future<value_t, exec_t>&& wait() &&
  {
    throw_if_invalid();
    _state->await();
    return std::move(*this);
  }

  // then_try: fn_t takes try_t<value_t>&& and returns upstream_t,
  // try_t<upstream_t>, or semi_future<upstream_t>.
  template <class fn_t>
  auto then_try(fn_t&& f) &&
  {
    throw_if_invalid();
    using decayed_f = std::decay_t<fn_t>;
    using r_t       = std::invoke_result_t<decayed_f, try_t<value_t>&&>;
    using inner_t   = detail::unwrap_callback_result_t<r_t>;

    promise<inner_t> p;
    future<inner_t, exec_t> next = p.get_semi_future().via(_executor);

    // Outer callback: runs wherever the upstream chain fires (inline on the
    // producer's thread). It immediately enqueues the real work on _executor,
    // so _executor->enqueue is a direct (non-virtual) call — no type erasure.
    exec_t* exe = _executor;
    _state->set_callback(
      [exe, p = std::move(p), fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable {
        exe->enqueue([p = std::move(p), fn = std::move(fn), t = std::move(t)]() mutable {
          try {
            if constexpr (detail::is_semi_future_v<r_t>) {
              semi_future<inner_t> inner = std::invoke(std::move(fn), std::move(t));
              std::move(inner).install_callback(
                [p = std::move(p)](try_t<inner_t>&& it) mutable { p.set_try(std::move(it)); });
            } else if constexpr (detail::is_try_v<r_t>) {
              p.set_try(std::invoke(std::move(fn), std::move(t)));
            } else if constexpr (std::is_void_v<inner_t>) {
              std::invoke(std::move(fn), std::move(t));
              p.set_value();
            } else {
              p.set_value(std::invoke(std::move(fn), std::move(t)));
            }
          } catch (...) {
            p.set_exception(std::current_exception());
          }
        });
      });
    _state.reset();
    return next;
  }

  // then_value: fn_t takes value_t (or no args for value_t==void); exceptions
  // propagate.
  template <class fn_t>
  auto then_value(fn_t&& f) &&
  {
    return std::move(*this).then_try([fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable {
      if (t.has_exception()) std::rethrow_exception(t.exception());
      return detail::invoke_with_value(std::move(fn), std::move(t));
    });
  }

  // then_error (typed): like defer_error but on the executor.
  template <class exception_type, class fn_t>
  future<value_t, exec_t> then_error(tag_t<exception_type>, fn_t&& f) &&
  {
    return std::move(*this).then_try(
      [fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable -> try_t<value_t> {
        if (!t.has_exception()) { return std::move(t); }
        try {
          std::rethrow_exception(t.exception());
        } catch (exception_type const& e) {
          if constexpr (std::is_void_v<value_t>) {
            std::invoke(std::move(fn), e);
            return try_t<void>::make_value();
          } else {
            return try_t<value_t>(std::invoke(std::move(fn), e));
          }
        } catch (...) {
          return std::move(t);
        }
      });
  }

  template <class fn_t>
  future<value_t, exec_t> then_error(fn_t&& f) &&
  {
    return std::move(*this).then_try(
      [fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable -> try_t<value_t> {
        if (!t.has_exception()) { return std::move(t); }
        if constexpr (std::is_void_v<value_t>) {
          std::invoke(std::move(fn), t.exception());
          return try_t<void>::make_value();
        } else {
          return try_t<value_t>(std::invoke(std::move(fn), t.exception()));
        }
      });
  }

  template <class fn_t>
  future<value_t, exec_t> then_ensure(fn_t&& f) &&
  {
    return std::move(*this).then_try(
      [fn = std::forward<fn_t>(f)](try_t<value_t>&& t) mutable -> try_t<value_t> {
        std::invoke(std::move(fn));
        return std::move(t);
      });
  }

  // Internal constructor — users go through semi_future::via or
  // promise::get_future.
  future(std::unique_ptr<detail::state<value_t>> s, exec_t* e) noexcept
    : _state(std::move(s)), _executor(e)
  {
  }

 private:
  void throw_if_invalid() const
  {
    if (!_state) { throw future_invalid_error(); }
  }

  std::unique_ptr<detail::state<value_t>> _state;
  exec_t* _executor = nullptr;
};

// ---------------------------------------------------------------------------
// Out-of-line definitions (need full semi_future/future types).

template <class value_t>
template <executor_concept exec_t>
future<value_t, exec_t> semi_future<value_t>::via(exec_t* exe) &&
{
  throw_if_invalid();
  return future<value_t, exec_t>(std::move(_state), exe);
}

template <class value_t>
template <executor_concept exec_t>
future<value_t, exec_t> promise<value_t>::get_future(exec_t* exe)
{
  return get_semi_future().via(exe);
}

namespace detail {

template <class value_t, class upstream_t, class fn_t>
void flat_map_state<value_t, upstream_t, fn_t>::set_callback(typename state<value_t>::callback cb)
{
  auto wrapper = [fn = std::move(_fn), cb = std::move(cb)](try_t<upstream_t>&& u) mutable {
    semi_future<value_t> inner;
    try {
      inner = std::invoke(std::move(fn), std::move(u));
    } catch (...) {
      std::move(cb)(try_t<value_t>(std::current_exception()));
      return;
    }
    std::move(inner).install_callback(std::move(cb));
  };
  _upstream->set_callback(std::move(wrapper));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Free constructors

template <class value_t>
semi_future<value_t> make_semi_future(try_t<value_t> t)
{
  return semi_future<value_t>(std::make_unique<detail::ready_state<value_t>>(std::move(t)));
}

template <class value_t,
          std::enable_if_t<!detail::is_try_v<std::decay_t<value_t>> &&
                             !detail::is_semi_future_v<std::decay_t<value_t>>,
                           int> = 0>
semi_future<std::decay_t<value_t>> make_semi_future(value_t&& v)
{
  using inner_t = std::decay_t<value_t>;
  return semi_future<inner_t>(
    std::make_unique<detail::ready_state<inner_t>>(try_t<inner_t>(std::forward<value_t>(v))));
}

template <class value_t = void>
semi_future<value_t> make_semi_future()
{
  if constexpr (std::is_void_v<value_t>) {
    return semi_future<void>(
      std::make_unique<detail::ready_state<void>>(try_t<void>::make_value()));
  } else {
    return semi_future<value_t>(
      std::make_unique<detail::ready_state<value_t>>(try_t<value_t>(value_t{})));
  }
}

// Invoke f immediately and capture value-or-exception in a ready semi_future.
template <class fn_t>
auto make_semi_future_with(fn_t&& f)
{
  using r_t = std::invoke_result_t<fn_t>;
  if constexpr (std::is_void_v<r_t>) {
    try {
      std::invoke(std::forward<fn_t>(f));
      return make_semi_future(try_t<void>::make_value());
    } catch (...) {
      return make_semi_future(try_t<void>(std::current_exception()));
    }
  } else {
    try {
      return make_semi_future(try_t<r_t>(std::invoke(std::forward<fn_t>(f))));
    } catch (...) {
      return make_semi_future(try_t<r_t>(std::current_exception()));
    }
  }
}

}  // namespace sirius::exec
