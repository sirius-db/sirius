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

#include <array>
#include <atomic>
#include <cstddef>
#include <format>
#include <string>

namespace sirius {

/// Why a spill sweep passed over a batch it found.
enum class spill_skip : std::size_t {
  borrow_refused,  ///< its query could not be borrowed (ending, or never opened)
  subscribed,      ///< a task or an exchange job holds it
  not_idle,        ///< locked or being read when the sweep looked
  wrong_tier,      ///< not in the space being relieved, or already in a target
  lock_busy,       ///< its exclusive lock was taken when a worker came to convert it
  no_target,       ///< no lower tier had room for it
};

inline constexpr std::size_t spill_skip_reasons = 6;

/// Batches and bytes passed over by reason. Bytes are those known at the time: a batch that
/// could not be read without its lock counts 0 bytes.
struct spill_skip_counts {
  std::array<std::atomic<std::size_t>, spill_skip_reasons> batches{};
  std::array<std::atomic<std::size_t>, spill_skip_reasons> bytes{};

  void add(spill_skip why, std::size_t size) noexcept
  {
    auto const i = static_cast<std::size_t>(why);
    batches[i].fetch_add(1, std::memory_order_relaxed);
    bytes[i].fetch_add(size, std::memory_order_relaxed);
  }

  void add(spill_skip_counts const& other) noexcept
  {
    for (std::size_t i = 0; i < spill_skip_reasons; ++i) {
      batches[i].fetch_add(other.batches[i].load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
      bytes[i].fetch_add(other.bytes[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
  }

  [[nodiscard]] std::size_t count(spill_skip why) const noexcept
  {
    return batches[static_cast<std::size_t>(why)].load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t total() const noexcept
  {
    std::size_t n = 0;
    for (auto const& b : batches) {
      n += b.load(std::memory_order_relaxed);
    }
    return n;
  }

  /// "subscribed 3 (1.50 GiB), lock_busy 1 (0.06 GiB)" for every reason seen, or "none".
  [[nodiscard]] std::string to_string() const
  {
    static constexpr std::array<char const*, spill_skip_reasons> names{
      "borrow_refused", "subscribed", "not_idle", "wrong_tier", "lock_busy", "no_target"};
    std::string out;
    for (std::size_t i = 0; i < spill_skip_reasons; ++i) {
      auto const n = batches[i].load(std::memory_order_relaxed);
      if (n == 0) { continue; }
      if (!out.empty()) { out += ", "; }
      out += std::format("{} {} ({:.2f} GiB)",
                         names[i],
                         n,
                         bytes[i].load(std::memory_order_relaxed) / double(1ULL << 30));
    }
    return out.empty() ? "none" : out;
  }
};

}  // namespace sirius
