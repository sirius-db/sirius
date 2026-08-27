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

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>

namespace sirius::exec {

/// Identity of one producer feeding a stream. A fan-in stream (N remote senders into one
/// source) is only at end-of-stream once every expected sender has closed, so closes must
/// dedup by identity rather than by count.
using sender_id_t = std::uint32_t;

/// End-of-stream lifecycle for a single stream, kept deliberately separate from the queue.
///
/// A `cucascade::shared_data_repository` already *is* the queue of batches; what it lacks is
/// the stream's lifecycle — who is still producing, whether "no batch right now" means "wait"
/// or "the stream is over", and how a starved consumer gets woken. This class owns exactly
/// that, and nothing else: it holds no repository and no batches. The endpoint that owns the
/// repository passes emptiness in as a snapshot (`repo_empty`), so the repository lock and this
/// lock are never held together and cannot invert.
///
/// Push admission and close share this one lock, which is what makes the two central races
/// safe: no batch can be admitted after the stream has gone terminal, and a batch is always
/// registered in the repository *before* its waker fires.
///
/// Non-copyable and non-movable — it owns a mutex and a condition variable, so an operator
/// holds it as an in-place member.
class stream_lifecycle {
 public:
  /// What a consumer should do right now, given a snapshot of repository emptiness.
  enum class availability {
    HAS_DATA,       ///< At least one batch is pullable.
    WAITING,        ///< Nothing available yet, but more may still arrive.
    END_OF_STREAM,  ///< Every expected sender closed and the queue is empty. Terminal.
  };

  /// @param expected The full set of sender ids that must close before end-of-stream.
  ///                 `{0}` for a single producer; `{0, 1}` for a two-way fan-in. An empty set
  ///                 means the stream is terminal from construction (nothing will ever
  ///                 produce), which is a legitimate degenerate case.
  explicit stream_lifecycle(std::set<sender_id_t> expected);

  stream_lifecycle(const stream_lifecycle&)            = delete;
  stream_lifecycle& operator=(const stream_lifecycle&) = delete;
  stream_lifecycle(stream_lifecycle&&)                 = delete;
  stream_lifecycle& operator=(stream_lifecycle&&)      = delete;

  // -----------------------------------------------------------------------
  // Producer side
  // -----------------------------------------------------------------------

  /// Run `insert` (the caller's repository add) under this lock, then fire the armed waker.
  ///
  /// Holding the lock across the insert is the point: a close cannot interleave, so no batch
  /// lands after end-of-stream, and the batch is visible in the repository before any waker
  /// observes it. The waker is fired after unlocking so the callback may re-enter the endpoint.
  ///
  /// @return false when the stream is already terminal — `insert` was not run.
  bool admit(const std::function<void()>& insert);

  /// Record that `sender` has finished producing. Idempotent: a repeat close from the same
  /// sender is a no-op and cannot advance a fan-in stream on its own. Once every expected
  /// sender has closed the stream goes terminal, waking `wait()` and firing the
  /// end-of-stream hook.
  ///
  /// @throws sirius::invalid_input_exception when `sender` is not in the expected set — an
  ///         unexpected sender is a wiring bug, not something to silently count.
  void mark_sender_done(sender_id_t sender);

  // -----------------------------------------------------------------------
  // Consumer side
  // -----------------------------------------------------------------------

  /// Classify what the consumer should do, given a snapshot of repository emptiness.
  /// Data still queued always wins over terminal: end-of-stream is never reported while an
  /// accepted batch is still pullable.
  [[nodiscard]] availability classify(bool repo_empty) const;

  /// Terminal EOS predicate: every expected sender closed AND the queue is empty.
  [[nodiscard]] bool drained(bool repo_empty) const;

  /// Block until `classify(repo_empty())` is no longer WAITING, i.e. until a batch is pullable
  /// or the stream has ended. `repo_empty` is re-evaluated under this lock on every wake-up, so
  /// it must not take the lock of anything that could call back in here.
  ///
  /// For external (wrapper / test) threads only — never call it from a GPU worker.
  void wait(const std::function<bool()>& repo_empty);

  // -----------------------------------------------------------------------
  // Re-arm and completion hooks
  // -----------------------------------------------------------------------

  /// Arm a one-shot waker fired by the next successful `admit()`. Cleared once fired, so a
  /// consumer that goes back to WAITING must re-arm.
  ///
  /// `arm_if` is evaluated under this lock together with the arming, which is what closes the
  /// lost-wake race: either the predicate sees the batch a concurrent push just admitted, or
  /// the push has not happened yet and will fire the waker we are installing. Callbacks must
  /// not capture raw pointers to anything this lifecycle can outlive.
  ///
  /// @param arm_if Re-checked "am I really starved?" predicate, evaluated under the lock.
  /// @return true when the waker was armed; false when `arm_if` said the caller is no longer
  ///         starved (nothing was armed, and the caller should re-evaluate).
  bool arm_waker(std::function<void()> waker, const std::function<bool()>& arm_if);

  /// Register the hook fired exactly once when the stream goes terminal. Registering *after*
  /// the stream already ended fires it immediately, so a hook wired late (a pipeline attached
  /// after a race-y close) is never lost.
  void set_on_end_of_stream(std::function<void()> hook);

  // -----------------------------------------------------------------------
  // Observers (diagnostics / tests)
  // -----------------------------------------------------------------------

  /// True once every expected sender has closed, regardless of what is still queued.
  [[nodiscard]] bool terminal() const;

  /// True when `sender` has already closed.
  [[nodiscard]] bool sender_closed(sender_id_t sender) const;

 private:
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  const std::set<sender_id_t> _expected;
  std::set<sender_id_t> _closed;
  bool _terminal{false};
  std::function<void()> _waker;  ///< One-shot; cleared when fired.
  std::function<void()> _on_end_of_stream;
};

}  // namespace sirius::exec
