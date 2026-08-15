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

#include "op/groupby_surrogate_store.hpp"

#include "sirius/exception.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sirius::op {

void surrogate_deferral_store::reservation::commit(::cucascade::read_only_data_batch batch) &&
{
  if (batch.get_batch_id() != _batch_id) {
    throw sirius::internal_exception(
      "surrogate_deferral_store::commit: batch {} does not match the reserved id {} on the {} "
      "side",
      batch.get_batch_id(),
      _batch_id,
      to_string(_side));
  }
  _store->commit(_side, _batch_id, std::move(batch));
}

surrogate_deferral_store::reservation surrogate_deferral_store::reserve(join_side side,
                                                                        std::uint64_t batch_id,
                                                                        cudf::size_type rows)
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto& state = state_for(side);
  if (auto* existing = find_entry(state, batch_id); existing != nullptr) {
    if (existing->rows != rows) {
      throw sirius::internal_exception(
        "surrogate_deferral_store::reserve: batch {} on the {} side re-reserved with a different "
        "row count ({} vs {})",
        batch_id,
        to_string(side),
        existing->rows,
        rows);
    }
    return reservation{*this, side, batch_id, existing->base};
  }
  if (state.next_base > static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()) -
                          static_cast<std::int64_t>(rows)) {
    // Finalization gathers with an INT32 cudf gather map; refuse address spaces that overflow
    // it instead of computing garbage. (The planner declines on estimated cardinality; this is
    // the hard runtime backstop.) Checked before the entry is recorded, so a throwing reserve
    // leaves the store unchanged.
    throw std::runtime_error(
      std::format("groupby_surrogate_keys: deferred string source exceeds int32 row addressing; "
                  "disable the groupby_surrogate_keys setting for this query [side: {}, requested "
                  "rows: {}, next base: {}]",
                  to_string(side),
                  rows,
                  state.next_base));
  }
  std::int64_t const base = state.next_base;
  state.next_base += rows;
  state.entries.push_back(entry{batch_id, base, rows, std::nullopt});
  return reservation{*this, side, batch_id, base};
}

void surrogate_deferral_store::commit(join_side side,
                                      std::uint64_t batch_id,
                                      ::cucascade::read_only_data_batch batch)
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto* reserved = find_entry(state_for(side), batch_id);
  if (reserved == nullptr) {
    // Structurally prevented by the reservation token; kept as defense in depth.
    throw sirius::internal_exception(
      "surrogate_deferral_store::commit: batch {} on the {} side was never reserved",
      batch_id,
      to_string(side));
  }
  if (!reserved->batch) { reserved->batch = std::move(batch); }
}

std::vector<surrogate_deferral_store::retained_source> surrogate_deferral_store::snapshot(
  join_side side) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto const& state = state_for(side);
  std::vector<retained_source> out;
  out.reserve(state.entries.size());
  for (auto const& e : state.entries) {
    if (!e.batch) {
      throw sirius::internal_exception(
        "surrogate_deferral_store::snapshot: reserved source batch {} on the {} side was never "
        "committed (its producing task cannot have succeeded)",
        e.batch_id,
        to_string(side));
    }
    out.push_back(retained_source{e.base, e.rows, *e.batch});
  }
  return out;
}

surrogate_deferral_store::release_stats surrogate_deferral_store::release()
{
  std::lock_guard<std::mutex> lock(_mutex);
  release_stats stats{0, 0};
  for (auto* state : {&_left, &_right}) {
    for (auto& e : state->entries) {
      if (!e.batch) { continue; }
      ++stats.sources;
      if (auto const* data = e.batch->get_data(); data != nullptr) {
        stats.bytes += data->get_size_in_bytes();
      }
      e.batch.reset();
    }
  }
  return stats;
}

surrogate_deferral_store::entry* surrogate_deferral_store::find_entry(side_state& state,
                                                                      std::uint64_t batch_id)
{
  auto const it = std::ranges::find(state.entries, batch_id, &entry::batch_id);
  return it == state.entries.end() ? nullptr : &*it;
}

}  // namespace sirius::op
