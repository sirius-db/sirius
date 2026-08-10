/*
<<<<<<< HEAD
 * Copyright 2025, Sirius Contributors.
=======
 * Copyright 2026, Sirius Contributors.
>>>>>>> f6d96a5d (feat(exec): repository-backed streaming source with sender-aware EOS (#836))
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

/// Producer identity for sender-set EOS (set, not counter).
using sender_id_t = std::uint32_t;

/// Repository IS the queue. Owns sender-set EOS, terminal/error state, and wake hooks.
/// Contracts: S1–S5 / P1–P4 — see docs/super-sirius/streaming-sessions.md. Borrowed repo; hold
/// via shared_ptr. try_pull() pops outside the lock — S5: wait-then-pop is not atomic.
class batch_stream {
 public:
  enum class availability {
    HAS_DATA,       ///< Batch pullable, or a pending error (P4).
    WAITING,        ///< Empty, but more may arrive.
    END_OF_STREAM,  ///< All expected senders closed, queue empty, no error.
  };

  /// @param repo The queue. Must not be null.
  /// @param expected Senders that must `close()` before EOS. Empty → terminal from construction.
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

  /// Push into the repository then fire on_data (S1). @return false if terminal.
  [[nodiscard]] bool push(std::shared_ptr<cucascade::data_batch> batch);

  /// Sender-set EOS: idempotent per sender; terminal when all expected closed.
  /// @throws sirius::invalid_input_exception when `sender` is not in the expected set.
  void close(sender_id_t sender);

  /// Poison the stream (no sender identity). P1–P3: immediate, first-wins, fail-fast terminal.
  /// S2 / P4: classifies as HAS_DATA, fires on_data, and notifies wait().
  /// @throws sirius::invalid_input_exception when `error` is null.
  void fail(std::exception_ptr error);

  // -----------------------------------------------------------------------
  // Consumer side
  // -----------------------------------------------------------------------

  /// Next batch, or nullptr. @throws pending error before popping (S4); error is never cleared.
  std::shared_ptr<cucascade::data_batch> try_pull();

  /// Queued data / pending error (P4) beat EOS.
  [[nodiscard]] availability classify() const;

  /// Clean end only: all senders closed, queue empty, no error (S3).
  [[nodiscard]] bool drained() const;

  /// The producer's error, or null. Never cleared once set.
  [[nodiscard]] std::exception_ptr pending_error() const;

  /// Block until classify() != WAITING (S5: not atomic with try_pull — re-check after wake).
  /// External threads only — never from a GPU worker.
  void wait();

  // -----------------------------------------------------------------------
  // Hooks fire after unlock; may re-enter. Do not capture raw this.
  // -----------------------------------------------------------------------

<<<<<<< HEAD
<<<<<<< HEAD
  /// Fired by every successful `push()`, and by the close that records an error. A consumer that
  /// found the queue empty has already been dropped by its driver; this is how it gets picked up
  /// again.
=======
  /// Fired by every successful `push()`, and by `fail()` (P4). A consumer that found the queue
  /// empty has already been dropped by its driver; this is how it gets picked up again.
  ///
  /// Deliberately not one-shot: the hook stays installed for the life of the stream. A one-shot
  /// hook would have to be re-armed by the consumer, and a push landing between the fire and the
  /// re-arm would go unannounced — the starved consumer would never be picked up.
>>>>>>> f6d96a5d (feat(exec): repository-backed streaming source with sender-aware EOS (#836))
=======
  /// Fires on every successful push() and on fail() (P4). Not one-shot: a one-shot hook would
  /// need re-arming, and a push between fire and re-arm would go unannounced.
>>>>>>> 2844d76d (docs(op): enhance streaming operator documentation and clarify design invariants)
  void set_on_data(std::function<void()> hook);

  /// Replaces prior hook; if already terminal, fires immediately inside this call.
  void set_on_end_of_stream(std::function<void()> hook);

  // -----------------------------------------------------------------------
  // Observers (diagnostics / tests)
  // -----------------------------------------------------------------------

  /// True once every expected sender has closed (or an error ended the stream).
  [[nodiscard]] bool terminal() const;

  [[nodiscard]] bool sender_closed(sender_id_t sender) const;

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
  /// First non-null wins; never cleared.
  std::exception_ptr _error;
  std::function<void()> _on_data;
  std::function<void()> _on_end_of_stream;
};

}  // namespace sirius::exec
