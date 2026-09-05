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

#include <cucascade/data/data_batch.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace sirius {

/**
 * @brief Registry of creator-declared reclaimable sizes for view-backed batches whose owner does
 * not release the viewed storage on drop (e.g. the scan's pinned-cache view forwards).
 *
 * declare() returns a token the creator stores inside the batch representation's owner, so the
 * entry is erased exactly when that representation is destroyed (conversion away or batch death);
 * a restored/owned successor representation therefore reports full reclaimability again. Batches
 * without an entry default to "reclaimable == representation size". Coverage is per
 * representation via the token: a path that extracts a view's owner and re-wraps it under a new
 * batch id gets the undeclared default unless it redeclares.
 */
class reclaim_ledger {
 public:
  /// Leaky singleton: tokens may fire during static teardown.
  static reclaim_ledger& instance()
  {
    static auto* ledger = new reclaim_ledger();
    return *ledger;
  }

  /**
   * @brief Record @p reclaimable_bytes for @p batch_id (sirius batch ids are never reused).
   *
   * @param batch_id The batch to declare for.
   * @param reclaimable_bytes Bytes actually freed by converting the batch away; must not exceed
   *                          the bytes attributed to the batch's representation.
   * @return A token that erases the entry when destroyed; store it in the representation's owner.
   */
  [[nodiscard]] std::shared_ptr<void> declare(uint64_t batch_id, std::size_t reclaimable_bytes)
  {
    // Token before entry: any throw unwinds the token, which retires an absent id (no-op), so an
    // entry can never outlive its token.
    std::shared_ptr<void> token{nullptr, [this, batch_id](void*) { retire(batch_id); }};
    std::lock_guard lock(_mutex);
    _entries.emplace(batch_id, reclaimable_bytes);
    return token;
  }

  /// Declared reclaimable bytes for @p batch_id, or nullopt when undeclared.
  std::optional<std::size_t> declared_bytes(uint64_t batch_id) const
  {
    std::lock_guard lock(_mutex);
    auto it = _entries.find(batch_id);
    if (it == _entries.end()) { return std::nullopt; }
    return it->second;
  }

 private:
  reclaim_ledger() = default;

  void retire(uint64_t batch_id)
  {
    std::lock_guard lock(_mutex);
    _entries.erase(batch_id);
  }

  // Lock-leaf: nothing is acquired while _mutex is held, so declare/lookup/retire are safe from
  // any context (retire runs inside convert_to under a batch's exclusive lock).
  mutable std::mutex _mutex;
  std::unordered_map<uint64_t, std::size_t> _entries;
};

/// Bytes actually freed in a batch's memory space by converting it away: the creator-declared
/// size when one is registered, otherwise the representation's full size.
inline std::size_t reclaimable_size_in_bytes(const cucascade::read_only_data_batch& ro)
{
  return reclaim_ledger::instance()
    .declared_bytes(ro.get_batch_id())
    .value_or(ro.get_data()->get_size_in_bytes());
}

}  // namespace sirius
