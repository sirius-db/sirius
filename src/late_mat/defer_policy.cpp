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

#include "late_mat/defer_policy.hpp"

#include <unordered_map>

namespace sirius::late_mat {

char const* describe(defer_refusal r) noexcept
{
  switch (r) {
    case defer_refusal::installed: return "installed";
    case defer_refusal::too_little_value: return "deferred value below the floor";
    case defer_refusal::too_short_a_ride: return "too few port crossings";
    case defer_refusal::no_columns: return "no columns to defer";
    case defer_refusal::evicted: return "evicted by a wider bundle";
    case defer_refusal::second_bundle: return "a wider bundle already rides this scan";
    case defer_refusal::below_value_x_boundaries: return "value x crossings below the floor";
  }
  return "unknown";
}

std::int64_t defer_candidate::net_value_bytes(std::int64_t rowid_bytes) const noexcept
{
  if (columns.empty()) { return -rowid_bytes; }
  std::int64_t total = 0;
  for (auto const& c : columns) {
    total += c.value_bytes;
  }
  return total - carrier_bytes(rowid_bytes);
}

std::int64_t defer_candidate::carrier_bytes(std::int64_t rowid_bytes) const noexcept
{
  if (columns.empty()) { return rowid_bytes; }
  return rowid_bytes + (static_cast<std::int64_t>(columns.size()) - 1) * kPlaceholderBytes;
}

std::vector<defer_outcome> choose_deferrals(std::vector<defer_candidate> const& candidates,
                                            defer_policy const& policy)
{
  std::vector<defer_outcome> outcomes;
  outcomes.reserve(candidates.size());
  for (auto const& c : candidates) {
    defer_outcome out;
    out.slot            = c.slot;
    out.boundaries      = c.boundaries;
    out.net_value_bytes = c.net_value_bytes(policy.rowid_bytes);
    if (c.columns.empty()) {
      out.refusal = defer_refusal::no_columns;
    } else if (c.boundaries < policy.min_boundaries) {
      out.refusal = defer_refusal::too_short_a_ride;
    } else if (out.net_value_bytes < policy.min_value_bytes) {
      out.refusal = defer_refusal::too_little_value;
    } else if (out.net_value_bytes * static_cast<std::int64_t>(out.boundaries) <
               policy.min_value_x_boundaries) {
      // Two independent floors cannot say that a thin ride over many crossings
      // repays where a fat one over few does not.
      out.refusal = defer_refusal::below_value_x_boundaries;
    }
    outcomes.push_back(std::move(out));
  }

  // One bundle per slot, widest wins. Evaluated after the thresholds so a
  // candidate that could never have installed does not evict one that could.
  std::unordered_map<std::string, std::size_t> holder;
  for (std::size_t i = 0; i < outcomes.size(); ++i) {
    if (!outcomes[i].installed()) { continue; }
    auto const it = holder.find(outcomes[i].slot);
    if (it == holder.end()) {
      holder.emplace(outcomes[i].slot, i);
      continue;
    }
    auto& sitting = outcomes[it->second];
    if (outcomes[i].net_value_bytes > sitting.net_value_bytes) {
      sitting.refusal = defer_refusal::evicted;
      it->second      = i;
    } else {
      outcomes[i].refusal = defer_refusal::evicted;
    }
  }
  return outcomes;
}

}  // namespace sirius::late_mat
