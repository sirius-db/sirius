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
 *
 * The policy also plans no membership filter for a Shadowed Key (`is_shadowed`): a key whose filter
 * no Consumer would select over the Membership Filters already published on its Channels. It judges
 * with the Consumer's own Static Selection (`static_filter_selection.hpp`). What is already
 * published reaches the policy as plain values too: `read_published_filters` is the one function
 * that reads Channels, and a caller that does not call it gets the plan without the rule.
 */

#pragma once

#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/dynamic_filter_key_outcome.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/static_filter_selection.hpp"

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
  bool bloom_only              = false;   ///< Membership filters are Bloom filters only
  std::size_t fixed_bloom_keys = 0;       ///< Non-zero: every Bloom filter's key count, uncapped
  std::size_t decode_membership_cap = 1;  ///< Membership Filters one Filtered Decode carries
};

/**
 * @brief What one binding of an admitted key can receive
 */
struct binding_facts {
  cudf::data_type probe_type;  ///< The binding's recorded probe type; EMPTY when unknown
  bool accepts_zone_map;       ///< Whether the binding's target accepts zone maps
  std::vector<static_filter_facts>
    published;  ///< Membership Filters already on the binding's Channel, in no particular order
};

/**
 * @brief The facts of every binding of every admitted key of `plan`, by admitted-key index
 *
 * `binding_facts::published` is left empty; `read_published_filters` fills it.
 */
[[nodiscard]] std::vector<std::vector<binding_facts>> binding_facts_of(
  dynamic_filter_publish_plan const& plan);

/**
 * @brief Fills each binding's `binding_facts::published` from one snapshot of its Channel
 *
 * A filter counts when a Consumer on any of the plan's replica GPUs is certain to hold it, so that
 * a key shadowed by it is never left with neither filter:
 * - only Membership Filters count, each filter object once per Channel;
 * - only filters with a Replica on every GPU of `plan.replica_spaces()`: on a GPU that lacks the
 * published filter a Consumer would use the new one;
 * - an estimate that is not a Proven Estimate is dropped, so its filter can take a Filtered Decode
 * slot but cannot make another filter negligible.
 *
 * A filter that is only planned or announced never counts: it may still fail, whereas a Channel is
 * append-only and a published filter stays. Under a cross-rank merge nothing counts, because this
 * process's rows are not the merged filter's keys and the ranks must plan the same filters.
 *
 * @throw std::logic_error if @p bindings was not taken from `binding_facts_of(plan)`
 * @param plan The plan whose Channels are read
 * @param bindings The plan's `binding_facts_of`, updated in place
 */
void read_published_filters(dynamic_filter_publish_plan const& plan,
                            std::span<std::vector<binding_facts>> bindings);

/**
 * @brief An exact IN-list over the key's valid rows.
 */
struct exact_membership {
  membership_filter_kind kind;  ///< `small_in_list` or `hash_in_list`
  std::size_t valid_rows;       ///< Keys the list stages
};

/**
 * @brief A Bloom filter.
 */
struct bloom_membership {
  std::size_t rows;        ///< The build's keys: valid rows, or every row when unknown
  std::size_t sized_rows;  ///< Rows the filter is sized for: `rows`, or a fixed key count
  std::size_t blocks;      ///< `bloom_blocks_for(sized_rows)`
};

/**
 * @brief One key's membership filter, if any.
 */
using membership_plan = std::variant<std::monostate, exact_membership, bloom_membership>;

/**
 * @brief The filters planned for one admitted key
 *
 * `construction_type` is the one carrier when it lands in the recorded type's family, otherwise the
 * recorded type.
 */
struct key_plan {
  std::size_t admitted_key_index;     ///< The key's index in the plan's admitted keys
  cudf::size_type build_key_ordinal;  ///< The key's column in every build batch
  cudf::data_type recorded_type;      ///< The Recorded Type, which the zone-map bounds carry
  cudf::data_type construction_type;  ///< The type the membership filter is built over
  membership_plan membership;         ///< The key's membership filter
  bool zone_map;                      ///< Whether a zone map is planned
  bool zone_map_deliverable;          ///< Whether some binding accepts the planned zone map
  bool fit_pending;                   ///< DECIMAL128 membership publishes only if bounds fit int64
  static_filter_facts candidate{};    ///< The planned membership filter as Consumers will see it

  /**
   * @brief Whether the key's bounds must be accumulated
   */
  [[nodiscard]] bool needs_bounds() const noexcept { return zone_map || fit_pending; }
};

/**
 * @brief Whether @p key is a Shadowed Key: no Consumer would select its planned membership filter
 * on any of its bindings
 *
 * Only a key whose `key_plan::candidate` carries a Proven Estimate is judged, and it is shadowed
 * only when `statically_selected` is false on every binding. One binding whose Consumer could
 * select the filter builds it for all.
 *
 * @param key The key, with its planned membership filter
 * @param bindings The key's bindings, with `binding_facts::published` filled
 * @param decode_membership_cap `construction_limits::decode_membership_cap`
 */
[[nodiscard]] bool is_shadowed(key_plan const& key,
                               std::span<binding_facts const> bindings,
                               std::size_t decode_membership_cap) noexcept;

/**
 * @brief The plan for every admitted key with a binding
 *
 * `keys` holds the keys with a planned membership filter or bounds. `outcomes` is in admitted-key
 * order: a planned key's line is `NOT_CONSTRUCTED` until its filters reach the fan-out; any other
 * line carries its final reason.
 */
struct construction_plan {
  std::vector<key_plan> keys;                     ///< Keys with a planned filter, by admitted key
  std::vector<dynamic_filter_key_line> outcomes;  ///< One line per admitted key with a binding
  dynamic_filter_stats_snapshot selection;        ///< Selection counts for `dynamic_filter_stats`
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
 * 6. A Shadowed Key (`is_shadowed`) drops its membership (`keys_skipped_shadowed`) and keeps its
 * zone map. It is judged before the Bloom cap, so its array never pushes the other keys over the
 * cap.
 * 7. When the Bloom keys' aligned arrays exceed `construction_limits::max_bloom_bytes_per_gpu`,
 * every Bloom key drops its membership (`keys_skipped_bloom_size_gate`).
 *
 * Keys left with neither membership nor bounds are omitted from `construction_plan::keys` but keep
 * their line. Reasons follow the `[dynamic_filter_key]` rules: `UNSUPPORTED_TYPE` is reported only
 * when no zone map would reach a binding either.
 *
 * @throw std::logic_error if an admitted key's build ordinal lies outside every batch's schema, or
 * the spans' sizes differ
 * @param facts The inventory's facts
 * @param admitted_keys The plan's admitted keys
 * @param bindings Each admitted key's bindings; a key with none is not planned
 * @param limits The limits to apply
 */
[[nodiscard]] construction_plan plan_construction(
  build_input_facts const& facts,
  std::span<dynamic_filter_publish_plan::admitted_key const> admitted_keys,
  std::span<std::vector<binding_facts> const> bindings,
  construction_limits const& limits);

}  // namespace sirius::op
