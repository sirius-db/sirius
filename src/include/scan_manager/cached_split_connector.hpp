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

#include <cucascade/data/data_batch.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Connector backed by a fixed vector of pre-pinned cached batches.
 *
 * Unlike split_connector — which is a producer/consumer queue — this connector
 * hands out entries from an immutable vector supplied at construction. Each
 * call to next_split() returns the entry at the current cursor and advances
 * the cursor atomically; once the cursor passes the end of the vector,
 * next_split() returns std::nullopt.
 */
class cached_split_connector {
 public:
  using cached_entry = std::shared_ptr<cucascade::data_batch>;

  explicit cached_split_connector(std::vector<cached_entry> entries) : _entries(std::move(entries))
  {
  }

  cached_split_connector(const cached_split_connector&)            = delete;
  cached_split_connector& operator=(const cached_split_connector&) = delete;
  cached_split_connector(cached_split_connector&&)                 = delete;
  cached_split_connector& operator=(cached_split_connector&&)      = delete;

  /// \brief Return the next cached entry and advance the cursor.
  /// \return std::nullopt when the cursor has reached the end of the vector.
  std::optional<cached_entry> next_split()
  {
    auto idx = _cursor.fetch_add(1, std::memory_order_relaxed);
    if (idx >= _entries.size()) { return std::nullopt; }
    return _entries[idx];
  }

  [[nodiscard]] std::size_t size() const { return _entries.size(); }

 private:
  const std::vector<cached_entry> _entries;
  std::atomic<std::size_t> _cursor{0};
};

}  // namespace sirius::scan_manager
