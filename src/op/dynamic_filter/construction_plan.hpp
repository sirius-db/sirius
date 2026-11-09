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
 * @file construction_plan.hpp
 * @brief The policy that chooses each admitted key's dynamic filters from the facts of a certified
 * build input: `plan_construction`.
 *
 * The policy is a pure function of plain values: the `build_input_facts` of a
 * `complete_build_inventory`, the plan's admitted keys, what each key's bindings can read, and
 * `construction_limits`. It reuses the whole-build publisher's rules (`choose_membership_filter`,
 * the domain gate, valid-row sizing, and the carrier rule `can_restore_to` /
 * `membership_same_family`), so a key gets the filter the whole-build path would choose for the
 * same rows. Building the planned filters is the mechanism's job; the policy touches no GPU, memory
 * space, or channel.
 */

#pragma once

#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/dynamic_filter_key_outcome.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace sirius::op {

/**
 * @brief The limits `plan_construction` applies
 */
struct construction_limits {
  std::size_t l2_bytes;                   ///< Smallest replica GPU's L2 size; 0 means unknown
  double inlist_max_l2_fraction;          ///< Largest share of L2 a hash IN-list may take
  bool emit_zone_map_filters;             ///< Whether zone maps are enabled
  double domain_coverage_threshold;       ///< The domain gate's coverage threshold
  std::uint64_t max_bloom_bytes_per_gpu;  ///< Cap on the sum of every Bloom key's aligned array
};

/**
 * @brief What one binding of an admitted key can receive
 */
struct binding_facts {
  cudf::data_type probe_type;  ///< The binding's recorded probe type; EMPTY when unknown
  bool accepts_zone_map;       ///< Whether the binding's target accepts zone maps
};

/**
 * @brief The facts of every binding of every admitted key of @p plan, by admitted-key index
 */
[[nodiscard]] std::vector<std::vector<binding_facts>> binding_facts_of(
  dynamic_filter_publish_plan const& plan);

/// An exact IN-list over the key's valid rows.
struct exact_membership {
  membership_filter_kind kind;  ///< `small_in_list` or `hash_in_list`
  std::size_t valid_rows;       ///< Keys the list stages
};

/// A Bloom filter.
struct bloom_membership {
  std::size_t sized_rows;  ///< Rows the filter is sized for: valid rows, or every row when unknown
  std::size_t blocks;      ///< `bloom_blocks_for(sized_rows)`
};

/// One key's membership filter, if any.
using membership_plan = std::variant<std::monostate, exact_membership, bloom_membership>;

/**
 * @brief The filters planned for one admitted key
 */
struct key_plan {
  /** @brief The key's index in the plan's admitted keys */
  std::size_t admitted_key_index;
  /** @brief The type the membership filter is built over: the one carrier when it lands in the
   * recorded type's family, otherwise the recorded type */
  cudf::data_type construction_type;
  /** @brief The key's membership filter */
  membership_plan membership;
  /** @brief Whether a zone map is planned */
  bool zone_map;
  /** @brief Whether some binding accepts the planned zone map, so the key's line reports it */
  bool zone_map_deliverable;
  /** @brief A DECIMAL128 membership filter that publishes only if the merged bounds fit int64 */
  bool fit_pending;

  /** @brief Whether the key's bounds must be accumulated */
  [[nodiscard]] bool needs_bounds() const noexcept { return zone_map || fit_pending; }
};

/**
 * @brief The plan for every admitted key with a binding
 */
struct construction_plan {
  /** @brief Keys with a planned membership filter or bounds, in admitted-key order */
  std::vector<key_plan> keys;
  /** @brief One line per admitted key with a binding, in admitted-key order. A planned key's line
   * is `not_constructed` until its filters reach the fan-out; any other line carries its final
   * reason. */
  std::vector<dynamic_filter_key_line> outcomes;
  /** @brief The selection counters the plan adds to `dynamic_filter_stats` */
  dynamic_filter_stats_snapshot selection;
};

/**
 * @brief Plans each admitted key's filters
 *
 * For each admitted key with at least one binding, in this order (which fixes counter attribution):
 * 1. A column absent from some batch is a `keys_skipped_type_mismatch`.
 * 2. Every carrier must equal or restore to the recorded type (`can_restore_to`), else
 * `keys_skipped_type_mismatch`. One carrier in the recorded type's family is the construction type;
 * any other mix builds at the recorded type.
 * 3. Membership is possible when the construction type classifies and some binding can read it (an
 * EMPTY probe type counts as readable); a classifiable key no binding can read adds its bindings to
 * `bindings_skipped_incompatible_probe`. A zone map is possible when enabled, supported for the
 * recorded type, and the key does not preserve NULLs.
 * 4. The domain gate judges the key on the total rows when it passed the carrier rule and some
 * filter is possible.
 * 5. Membership follows `choose_membership_filter` over the valid rows. With an unknown null count
 * only a Bloom filter sized on every row is planned (`keys_sized_on_total_rows`). DECIMAL128 plans
 * as if it fits, with `fit_pending`.
 * 6. When the Bloom keys' aligned arrays exceed `construction_limits::max_bloom_bytes_per_gpu`,
 * every Bloom key drops its membership (`keys_skipped_bloom_size_gate`).
 *
 * Keys left with neither membership nor bounds are omitted from `construction_plan::keys` but keep
 * their line. Reasons follow the `[dynamic_filter_key]` rules: `unsupported_type` is reported only
 * when no zone map would reach a binding either.
 *
 * @param facts The inventory's facts
 * @param admitted_keys The plan's admitted keys
 * @param bindings Each admitted key's bindings; a key with none is not planned
 * @param limits The limits to apply
 * @throw std::logic_error if an admitted key's build ordinal lies outside every batch's schema, or
 * the spans' sizes differ
 */
[[nodiscard]] construction_plan plan_construction(
  build_input_facts const& facts,
  std::span<dynamic_filter_publish_plan::admitted_key const> admitted_keys,
  std::span<std::vector<binding_facts> const> bindings,
  construction_limits const& limits);

}  // namespace sirius::op
