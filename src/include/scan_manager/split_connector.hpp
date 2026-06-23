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

#include "op/sirius_physical_operator.hpp"

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>

namespace sirius::scan_manager {

class split_provider;

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
 *   - throws                         → producer surfaced an error via close(exception_ptr)
 *                                       and the queue is drained.
 *
 * Pushes are gated: only @ref split_provider may enqueue splits (via the
 * @c friend relationship and @ref split_provider::push_to_connector helper).
 * close() and the consumer-side methods remain public so the scan manager
 * (driver loop) and the scan operator can drive the lifecycle.
 */
class split_connector : public std::enable_shared_from_this<split_connector> {
 public:
  split_connector();
  ~split_connector();

  split_connector(const split_connector&)            = delete;
  split_connector& operator=(const split_connector&) = delete;
  split_connector(split_connector&&)                 = delete;
  split_connector& operator=(split_connector&&)      = delete;

  /// \brief Mark the connector as closed: no more splits will be pushed. Idempotent.
  ///        Wakes all waiting consumers.
  ///
  /// \param exception Optional exception captured by the producer. The first
  ///                  non-null exception passed across all close() calls is
  ///                  stored and rethrown by get_next_split() once the queue
  ///                  has been drained. Subsequent close() calls do not
  ///                  overwrite an already-stored exception.
  void close(std::exception_ptr const& exception = nullptr);

  /// \brief Pull the next split, blocking until one is available or the connector
  ///        is closed and drained.
  /// \return std::nullopt when closed and drained without error; the next split
  ///         otherwise.
  /// \throws The exception passed to close() (if any) once the queue is drained.
  std::optional<std::unique_ptr<op::operator_data>> get_next_split();

  /// \brief True iff close() has been called and the queue is drained.
  [[nodiscard]] bool is_closed() const;

  [[nodiscard]] bool has_more_splits() const;

 private:
  friend class load_balancing_scan_batch_coalecer;

  /// \brief Enqueue a ready split. Producer side. Wakes a waiting consumer.
  ///        Reachable only via @ref split_provider::push_to_connector so all
  ///        producers route through the provider's friendship channel.
  void push_split(std::unique_ptr<op::operator_data> split);

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::unique_ptr<op::operator_data>> _splits;
  bool _closed{false};
  std::exception_ptr _exception;
};

}  // namespace sirius::scan_manager
