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

#include <deque>
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
 * side (the scan operator) pulls splits via get_next_split() with three-state semantics:
 *
 *   - returned optional is std::nullopt          → connector is closed and drained.
 *   - returned optional contains a null pointer  → no split ready yet, retry later.
 *   - returned optional contains a non-null ptr  → next split.
 */
class split_connector {
 public:
  split_connector();
  ~split_connector();

  split_connector(const split_connector&)            = delete;
  split_connector& operator=(const split_connector&) = delete;
  split_connector(split_connector&&)                 = delete;
  split_connector& operator=(split_connector&&)      = delete;

  /// \brief Enqueue a ready split. Producer side.
  void push_split(std::unique_ptr<op::operator_data> split);

  /// \brief Mark the connector as closed: no more splits will be pushed. Idempotent.
  void close();

  /// \brief Pull the next split.
  /// \return std::nullopt when closed and drained; an optional containing a null
  ///         unique_ptr when open but currently empty; an optional containing the
  ///         next split otherwise.
  std::optional<std::unique_ptr<op::operator_data>> get_next_split();

  /// \brief True iff close() has been called and the queue is drained.
  [[nodiscard]] bool is_closed() const;

  /// \brief True iff a split is currently queued.
  [[nodiscard]] bool has_pending_split() const;

 private:
  mutable std::mutex _mutex;
  std::deque<std::unique_ptr<op::operator_data>> _splits;
  bool _closed{false};
};

}  // namespace sirius::scan_manager
