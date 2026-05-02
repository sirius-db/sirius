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
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>

namespace sirius::op {
class operator_data;
}  // namespace sirius::op

namespace sirius::scan_manager {

/**
 * @brief Bridge between a scan-side producer (the scan manager) and a scan source operator.
 *
 * The connector is a lock-protected queue of pre-built splits. The producer side enqueues
 * splits as they become ready and calls close() when no more will arrive. The consumer
 * pulls splits via get_next_split(), which BLOCKS until either a split is available or
 * the connector has been closed and drained:
 *
 *   - returns std::nullopt           → connector is closed and drained, no more will arrive.
 *   - returns a non-null unique_ptr  → next split.
 *
 * Failure path: if the producer cannot complete the scan, it calls @c set_error to
 * record the exception and then @c close() to wake consumers. A consumer waiting in
 * (or arriving at) @c get_next_split observes the stored exception and rethrows it,
 * so the failure surfaces as a thrown exception rather than a silent end-of-stream.
 */
class split_connector {
 public:
  split_connector();
  ~split_connector();

  split_connector(const split_connector&)            = delete;
  split_connector& operator=(const split_connector&) = delete;
  split_connector(split_connector&&)                 = delete;
  split_connector& operator=(split_connector&&)      = delete;

  /// \brief Enqueue a ready split. Producer side. Wakes a waiting consumer.
  void push_split(std::unique_ptr<op::operator_data> split);

  /// \brief Mark the connector as closed: no more splits will be pushed. Idempotent.
  ///        Wakes all waiting consumers.
  void close();

  /// \brief Record a producer-side failure. Subsequent (and currently waiting)
  ///        @c get_next_split calls rethrow this exception instead of returning
  ///        @c std::nullopt. Idempotent: only the first error is kept; later
  ///        calls are dropped on the floor (one is enough to fail the query).
  ///        Callers should follow this with @c close() so consumers wake.
  void set_error(std::exception_ptr err);

  /// \brief Pull the next split, blocking until one is available or the connector
  ///        is closed and drained.
  /// \return std::nullopt when closed and drained without errors; the next split
  ///         otherwise.
  /// \throws The exception passed to @c set_error if the producer recorded one.
  std::optional<std::unique_ptr<op::operator_data>> get_next_split();

  /// \brief True iff close() has been called and the queue is drained.
  [[nodiscard]] bool is_closed() const;

  [[nodiscard]] bool has_more_splits() const;

 private:
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::unique_ptr<op::operator_data>> _splits;
  bool _closed{false};
  /// First producer-side failure, if any. Cleared by get_next_split when it
  /// rethrows so the same exception is delivered exactly once and the
  /// connector behaves like a closed-empty queue afterwards.
  std::exception_ptr _stored_error;
};

}  // namespace sirius::scan_manager
