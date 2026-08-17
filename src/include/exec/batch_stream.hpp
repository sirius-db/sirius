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

#include <cucascade/data/data_repository.hpp>

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <set>

namespace sirius::exec {

/// Identity of one producer feeding a stream. A fan-in stream (N remote senders into one
/// source) is only at end-of-stream once every expected sender has closed, so closes must
/// dedup by identity rather than by count.
using sender_id_t = std::uint32_t;

/// One direction of batch flow: N declared senders push `data_batch`es into one
/// `shared_data_repository`; consumers pull, poll, or block. The repository is the queue; this
/// owns everything the repository lacks — who is still producing, whether "no batch right now"
/// means "wait" or "the stream is over", how a starved consumer gets woken, and how a producer
/// failure reaches the consumer.
///
/// The streaming source, the streaming sink, and the session all need this same pairing;
/// written once here, each of them is left with task-protocol glue.
///
/// The repository is borrowed: the caller creates and registers it with the memory manager, so
/// queued batches stay spillable. Hold the stream by `shared_ptr` — producer threads co-own it.
///
/// Admission, close, and every emptiness check share the one lock, which is what makes the races
/// safe: no batch is admitted after the stream goes terminal, no queued batch is ever reported as
/// end-of-stream, and a batch is in the repository before `on_data` announces it. The repository
/// is touched only under that lock — except `try_pull()`'s pop, so wait-then-pop is not atomic
/// and a blocking consumer loop must re-check after `wait()`.
///
/// Error semantics are P1–P4 (see `fail`). Stream-level guarantees S1–S5 (cited by call sites
/// and tests) collect the observable contracts of each mechanism:
/// - **S1 — admission ordering.** `push()` puts the batch in the repository before firing
///   `on_data`, and returns false once the stream is terminal. A consumer that saw EOS can never
///   be raced by a batch that was not yet visible when `on_data` fired.
/// - **S2 — poison dominates.** `fail()` fires `on_data` (P4) so a consumer parked on `WAITING`
///   wakes to collect the rethrow. The error is immediate (P1), first-wins (P2), and ends the
///   stream at once (P3) — a starved source would otherwise never discover a producer failure.
/// - **S3 — errored is never clean.** A stream with a pending error never returns
///   `END_OF_STREAM` or `drained()`. The only exit is the rethrow from `try_pull()`, never the
///   quiet success that would let a failed query finish as if it had worked.
/// - **S4 — rethrow beats pop.** `try_pull()` checks the pending error before popping; batches
///   queued behind a failure are never handed to the consumer.
/// - **S5 — wait-then-pop is not atomic.** `wait()` and the following `try_pull()` are two
///   separate critical sections; a blocking consumer loop must re-check after waking.
class batch_stream {
 public:
  /// What a consumer should do right now.
  enum class availability {
    HAS_DATA,       ///< At least one batch is pullable (or an error is waiting to rethrow).
    WAITING,        ///< Nothing available yet, but more may still arrive.
    END_OF_STREAM,  ///< Every expected sender closed, queue empty, no error. Terminal.
  };

  /// @param repo The queue. Must not be null.
  /// @param expected Every sender that must `close()` before end-of-stream — `{0}` for a single
  ///        producer, `{0 … N-1}` for an N-way fan-in. An empty set means the stream is terminal
  ///        from construction (nothing will ever produce), a legitimate degenerate case.
  /// @throws sirius::invalid_input_exception when `repo` is null.
  batch_stream(std::shared_ptr<cucascade::shared_data_repository> repo,
               std::set<sender_id_t> expected);

  batch_stream(const batch_stream&)            = delete;
  batch_stream& operator=(const batch_stream&) = delete;
  batch_stream(batch_stream&&)                 = delete;
  batch_stream& operator=(batch_stream&&)      = delete;

  // -----------------------------------------------------------------------
  // Producer side — any thread
  // -----------------------------------------------------------------------

  /// Insert `batch` into the repository, then fire `on_data` after unlocking.
  /// @return false when the stream is already terminal — the batch was refused, not queued.
  [[nodiscard]] bool push(std::shared_ptr<cucascade::data_batch> batch);

  /// Record that `sender` has finished producing cleanly. Idempotent: a repeat close from the
  /// same sender is a no-op and cannot advance a fan-in stream on its own. Once every expected
  /// sender has closed the stream goes terminal, waking `wait()` and firing the
  /// end-of-stream hook.
  ///
  /// @throws sirius::invalid_input_exception when `sender` is not in the expected set — an
  ///         unexpected sender is a wiring bug, not something to silently count.
  void close(sender_id_t sender);

  /// A producer died: poison the stream. Failure is stream-wide, not per-sender — it carries no
  /// identity and waits for nobody.
  ///
  /// - **P1 — immediate visibility.** `pending_error()` returns it as soon as this call returns.
  /// - **P2 — the first failure wins.** Later failures and clean closes never displace the
  ///   original cause.
  /// - **P3 — fail-fast terminal.** The stream ends at once rather than waiting for senders that
  ///   will never produce anything useful now.
  /// - **P4 — poison is data.** A pending error classifies as `HAS_DATA` even over an empty
  ///   queue, and is never `END_OF_STREAM` or `drained()`: an errored stream ends by rethrow at
  ///   the consumer, not by a quiet clean finish. Fires `on_data` to bring a parked consumer
  ///   back for it, as well as `on_end_of_stream`.
  ///
  /// @throws sirius::invalid_input_exception when `error` is null — a null failure is a bug at
  ///         the call site, not a clean close.
  void fail(std::exception_ptr error);

  // -----------------------------------------------------------------------
  // Consumer side
  // -----------------------------------------------------------------------

  /// Non-blocking: the next queued batch, or nullptr when nothing is queued.
  /// @throws the pending error, checked before the pop, so batches queued behind a failure are
  ///         never handed out. Every call rethrows; the error is never consumed.
  std::shared_ptr<cucascade::data_batch> try_pull();

  /// What the consumer should do right now. Queued data wins over terminal, so end-of-stream is
  /// never reported while an accepted batch is still pullable. A pending error wins over both
  /// (P4): it reads as `HAS_DATA` so the consumer comes back and collects the rethrow.
  [[nodiscard]] availability classify() const;

  /// True only at a clean end: every expected sender closed, the queue empty, no pending error
  /// (P4 — an errored stream never reports a clean end).
  [[nodiscard]] bool drained() const;

  /// The producer's error, or null. Never cleared once set.
  [[nodiscard]] std::exception_ptr pending_error() const;

  /// Block until a batch is pullable or the stream has ended (cleanly or with an error).
  /// For external (session / wrapper / test) threads only — never call it from a GPU worker.
  void wait();

  // -----------------------------------------------------------------------
  // Hooks. Both hold one slot, and both fire after unlocking, so a callback may re-enter the
  // stream. Neither may capture a raw pointer to anything this stream can outlive.
  // -----------------------------------------------------------------------

  /// Fired by every successful `push()`, and by the close that records an error. A consumer that
  /// found the queue empty has already been dropped by its driver; this is how it gets picked up
  /// again.
  void set_on_data(std::function<void()> hook);

  /// Fired when the stream goes terminal. A second registration replaces the first, which then
  /// never fires. Registering after the stream already ended fires the hook immediately,
  /// *inside this call*, so a late-wired hook is never lost.
  void set_on_end_of_stream(std::function<void()> hook);

  // -----------------------------------------------------------------------
  // Observers (diagnostics / tests)
  // -----------------------------------------------------------------------

  /// True once every expected sender has closed (or an error ended the stream), regardless of
  /// what is still queued.
  [[nodiscard]] bool terminal() const;

  /// True when `sender` has already closed.
  [[nodiscard]] bool sender_closed(sender_id_t sender) const;

  /// The queue, still owned by whoever created and registered it.
  [[nodiscard]] const std::shared_ptr<cucascade::shared_data_repository>& repository() const
  {
    return _repo;
  }

 private:
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::shared_ptr<cucascade::shared_data_repository> _repo;
  const std::set<sender_id_t> _expected;
  /// A set, not a counter: two closes from sender 0 must not stand in for senders {0, 1}.
  std::set<sender_id_t> _closed;
  bool _terminal{false};
  /// The producer's failure, if any. First non-null wins; never cleared.
  std::exception_ptr _error;
  std::function<void()> _on_data;
  std::function<void()> _on_end_of_stream;
};

}  // namespace sirius::exec
