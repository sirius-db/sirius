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

#include "op/dynamic_filter/static_filter_selection.hpp"

#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <algorithm>
#include <ranges>
#include <utility>

namespace sirius::op {

static_filter_facts facts_of(sirius_dynamic_filter const& filter)
{
  static_filter_facts facts{.estimated_keep = filter.estimated_selectivity()};
  if (auto const* small = dynamic_cast<sirius_dynamic_small_in_list_filter const*>(&filter)) {
    facts.kind = membership_filter_kind::small_in_list;
    facts.keys = small->size();
  } else if (auto const* set = dynamic_cast<sirius_dynamic_in_list_filter const*>(&filter)) {
    facts.kind = membership_filter_kind::hash_in_list;
    facts.keys = set->size();
  } else if (filter.kind() == sirius_dynamic_filter_kind::BLOOM) {
    facts.kind = membership_filter_kind::bloom;
  }
  return facts;
}

bool statically_selected(static_filter_facts const& candidate,
                         std::span<static_filter_facts const> published,
                         std::size_t decode_cap) noexcept
{
  auto const preferred = std::ranges::count_if(published, [&](static_filter_facts const& other) {
    return decode_preference(other, candidate) < 0;
  });
  if (std::cmp_less(preferred, decode_cap)) { return true; }
  return !negligible_after_stronger(
    candidate.estimated_keep,
    published | std::views::transform(&static_filter_facts::estimated_keep));
}

}  // namespace sirius::op
