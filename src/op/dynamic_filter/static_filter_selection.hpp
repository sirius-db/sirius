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

/**
 * @file static_filter_selection.hpp
 * @brief Static Selection: how a Consumer chooses among a Channel's Membership Filters before it
 * has measured them.
 *
 * Two rules make up the choice, and both read only `static_filter_facts`. Filtered Decode carries
 * the filters that come first in `decode_preference`, up to its cap. Every other filter is applied
 * after the decode unless the Negligible-Share Rule (`negligible_after_stronger`) skips it.
 * `statically_selected` combines the two for a filter that does not exist yet.
 *
 * `sirius::op::scan::dynamic_filter_consumer` applies these rules to every batch
 * (`src/op/scan/dynamic_filter_merge.cpp`). They are defined here, below the Consumer, so that code
 * deciding whether a filter is worth building judges it exactly as the Consumer will. Every
 * function is pure and takes no lock.
 */

#pragma once

#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>

namespace sirius::op {

class sirius_dynamic_filter;

/**
 * @brief The share of a batch below which a filter's removed rows do not repay probing every row
 */
inline constexpr double negligible_removed_share = 0.10;

/**
 * @brief What is known about a Membership Filter before any batch measured it
 *
 * Equality compares every fact; `decode_preference` ignores the estimate.
 */
struct static_filter_facts {
  membership_filter_kind kind = membership_filter_kind::none;  ///< `none` for any other filter
  std::uint64_t keys =
    std::numeric_limits<std::uint64_t>::max();  ///< IN-list key count; the maximum when unknown
  std::optional<double> estimated_keep;         ///< The Producing Join's estimate of the kept share
  bool estimate_proven = false;                 ///< Whether `estimated_keep` is a Proven Estimate

  [[nodiscard]] bool operator==(static_filter_facts const&) const = default;
};

/**
 * @brief The facts of a published filter
 */
[[nodiscard]] static_filter_facts facts_of(sirius_dynamic_filter const& filter);

namespace detail {

/**
 * @brief Position of `kind` in Filtered Decode's preference: exact forms keep fewer rows than a
 * Bloom Filter over the same keys, and anything that is not a Membership Filter comes last
 */
[[nodiscard]] constexpr std::uint8_t decode_rank(membership_filter_kind kind) noexcept
{
  switch (kind) {
    case membership_filter_kind::small_in_list: return 0;
    case membership_filter_kind::hash_in_list: return 1;
    case membership_filter_kind::bloom: return 2;
    case membership_filter_kind::none: break;
  }
  return std::numeric_limits<std::uint8_t>::max();
}

}  // namespace detail

/**
 * @brief Filtered Decode's preference between two unmeasured filters
 *
 * Compares the kind (Small IN-list, Hash IN-list, Bloom Filter, anything else) and then the key
 * count: fewer build keys keep fewer rows. The Consumer breaks a tie by the filters' positions in
 * its Channel snapshot, which follow the Channel's columns and not the order of publication.
 *
 * @return `less` when @p a is preferred
 */
[[nodiscard]] constexpr std::strong_ordering decode_preference(
  static_filter_facts const& a, static_filter_facts const& b) noexcept
{
  if (auto const by_kind = detail::decode_rank(a.kind) <=> detail::decode_rank(b.kind);
      by_kind != 0) {
    return by_kind;
  }
  return a.keys <=> b.keys;
}

/**
 * @brief A range of estimated kept shares, each empty when its filter has no estimate
 */
template <typename Range>
concept keep_estimate_range =
  std::ranges::input_range<Range> &&
  std::convertible_to<std::ranges::range_reference_t<Range>, std::optional<double>>;

/**
 * @brief The share of a batch that a filter removes once every filter estimated strictly stronger
 * has been applied
 *
 * A probe that runs on every row is paid for by the rows it removes from the whole batch, however
 * selective the filter is on the rows the stronger filters kept.
 *
 * @param own The filter's estimated kept share
 * @param all The estimates of the filters applied to the same batch; may include @p own
 * @return The removed share, or empty when @p own is empty or no estimate in @p all is stronger
 */
template <keep_estimate_range Estimates>
[[nodiscard]] constexpr std::optional<double> removed_share_after_stronger(
  std::optional<double> own, Estimates&& all)
{
  if (!own) { return std::nullopt; }
  double survival = 1.0;
  for (std::optional<double> const other : all) {
    if (other && *other < *own) { survival *= *other; }
  }
  if (!(survival < 1.0)) { return std::nullopt; }
  return survival * (1.0 - *own);
}

/**
 * @brief The Negligible-Share Rule: whether `removed_share_after_stronger` is below
 * `negligible_removed_share`
 */
template <keep_estimate_range Estimates>
[[nodiscard]] constexpr bool negligible_after_stronger(std::optional<double> own, Estimates&& all)
{
  auto const removed = removed_share_after_stronger(own, all);
  return removed && *removed < negligible_removed_share;
}

/**
 * @brief Whether a Consumer could select a filter that is not on its Channel yet
 *
 * The filter is selected when Filtered Decode would carry it, or when the Negligible-Share Rule
 * would not skip it. Filtered Decode is certain to pass it over only when at least @p decode_cap of
 * @p published are strictly preferred in `decode_preference`; a tie does not count, because nothing
 * here can tell which side the Consumer's snapshot order favors.
 *
 * @param candidate The facts the filter would be published with
 * @param published The Membership Filters already on the Channel
 * @param decode_cap The Membership Filters one Filtered Decode carries
 */
[[nodiscard]] bool statically_selected(static_filter_facts const& candidate,
                                       std::span<static_filter_facts const> published,
                                       std::size_t decode_cap) noexcept;

}  // namespace sirius::op
