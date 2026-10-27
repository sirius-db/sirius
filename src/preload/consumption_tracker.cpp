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

#include "preload/consumption_tracker.hpp"

#include <iterator>

namespace sirius::preload {

consumption_tracker& consumption_tracker::instance() noexcept
{
  static consumption_tracker tracker;
  return tracker;
}

void consumption_tracker::enable() noexcept { _enabled.fetch_add(1, std::memory_order_relaxed); }

void consumption_tracker::disable() noexcept
{
  if (_enabled.fetch_sub(1, std::memory_order_relaxed) != 1) { return; }
  std::lock_guard const lk{_mu};
  _batches.clear();
  _counts.clear();
}

void consumption_tracker::note_downgraded(query_id_t q, std::uint64_t batch_id) noexcept
{
  if (!enabled()) { return; }
  std::lock_guard const lk{_mu};
  if (auto const it = _batches.find(batch_id); it != _batches.end()) {
    it->second = {q, batch_state::downgraded};
    return;
  }
  if (_batches.size() >= max_entries) { return; }
  try {
    _batches.emplace(batch_id, entry{q, batch_state::downgraded});
  } catch (...) {  // untracked: the batch is simply not counted
  }
}

void consumption_tracker::note_promotion_started(std::uint64_t batch_id) noexcept
{
  if (!enabled()) { return; }
  std::lock_guard const lk{_mu};
  if (auto const it = _batches.find(batch_id); it != _batches.end()) {
    it->second.state = batch_state::promoting;
  }
}

void consumption_tracker::note_promotion_finished(std::uint64_t batch_id, bool promoted) noexcept
{
  if (!enabled()) { return; }
  std::lock_guard const lk{_mu};
  if (auto const it = _batches.find(batch_id);
      it != _batches.end() && it->second.state == batch_state::promoting) {
    it->second.state = promoted ? batch_state::promoted : batch_state::downgraded;
  }
}

std::optional<consumption_tracker::batch_state> consumption_tracker::begin_input(
  std::uint64_t batch_id) noexcept
{
  if (!enabled()) { return std::nullopt; }
  std::lock_guard const lk{_mu};
  auto const it = _batches.find(batch_id);
  if (it == _batches.end()) { return std::nullopt; }
  return it->second.state;
}

void consumption_tracker::end_input(std::uint64_t batch_id,
                                    batch_state before,
                                    bool upgraded_to_gpu,
                                    std::uint64_t lock_wait_ns,
                                    std::uint64_t prepare_ns) noexcept
{
  std::lock_guard const lk{_mu};
  auto const it = _batches.find(batch_id);
  if (it == _batches.end()) { return; }  // forgotten, or counted by another task
  auto const q = it->second.query;
  _batches.erase(it);
  try {
    auto& c = _counts[q];
    switch (classify(before, upgraded_to_gpu)) {
      case input_readiness::ready: ++c.ready; break;
      case input_readiness::waited:
        ++c.waited;
        c.waited_ns += lock_wait_ns;
        break;
      case input_readiness::upgraded:
        ++c.upgraded;
        c.upgraded_ns += prepare_ns;
        break;
    }
  } catch (...) {  // uncounted
  }
}

input_readiness_counts consumption_tracker::take_query(query_id_t q) noexcept
{
  std::lock_guard const lk{_mu};
  for (auto it = _batches.begin(); it != _batches.end();) {
    it = it->second.query == q ? _batches.erase(it) : std::next(it);
  }
  auto const it = _counts.find(q);
  if (it == _counts.end()) { return {}; }
  auto const counts = it->second;
  _counts.erase(it);
  return counts;
}

input_readiness consumption_tracker::classify(batch_state before, bool upgraded_to_gpu) noexcept
{
  if (upgraded_to_gpu) { return input_readiness::upgraded; }
  return before == batch_state::promoted ? input_readiness::ready : input_readiness::waited;
}

}  // namespace sirius::preload
