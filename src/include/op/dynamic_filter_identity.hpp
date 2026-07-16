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

/**
 * @file
 * @brief The dynamic-filter identity vocabulary: strong entity IDs, strong ordinals, and the
 * planning-time allocator that mints the IDs.
 *
 * Dynamic-filter planning uses several numbers that look alike but are not interchangeable. This
 * file makes accidental mixing a compile error by giving every number a distinct type.
 *
 * There are two deliberately different validity contracts:
 *
 * - Entity IDs: zero is invalid; allocator-minted values start at one:
 *   - `dynamic_filter_publication_plan_id`
 *   - `dynamic_filter_target_id`
 *   - `dynamic_filter_channel_id`
 *   - `dynamic_filter_id` (declared for the future lifecycle; not minted in C1a-2a)
 * - Ordinals: zero is a valid position:
 *   - `duckdb_filter_ordinal`
 *   - `join_condition_index`
 *   - `sirius_key_ordinal`
 *   - `duckdb_column_ids_index`
 *   - `probe_schema_ordinal`
 *
 * The three planning ID categories have independent counters. Equal numeric values across
 * categories are type-incompatible:
 *
 *   planning allocator
 *      |
 *      +-- publication plan IDs:  1, 2, 3, ...
 *      +-- target IDs:            1, 2, 3, ...
 *      +-- channel IDs:           1, 2, 3, ...
 *
 * Current integration status (C1a-2a): the plan generator (`sirius_physical_plan_generator`) owns
 * one allocator per planning transaction and mints publication-plan, target, and channel IDs into
 * the sidecar publish-plan builder. `dynamic_filter_id` is declared here so the identity
 * vocabulary is complete, but nothing mints one yet. Filter IDs are per-execution values; their
 * counter, execution generation, and prepared-topology lease arrive with C1a-2d's fresh-execution
 * lifecycle wiring.
 */

// cudf
#include <cudf/types.hpp>

// standard library
#include <compare>  // IWYU pragma: keep — required for the defaulted <=> (std::strong_ordering)
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::op {

class dynamic_filter_identity_allocator;

/**
 * @brief Strong entity identity. Zero means "not assigned"; real IDs start at one.
 */
template <class Tag, class Rep = std::uint32_t>
  requires std::unsigned_integral<Rep>
class dynamic_filter_entity_id {
 public:
  using rep = Rep;

  constexpr dynamic_filter_entity_id() noexcept = default;

  [[nodiscard]] constexpr bool is_valid() const noexcept { return _value != Rep{0}; }
  [[nodiscard]] constexpr Rep value() const noexcept { return _value; }

  constexpr auto operator<=>(dynamic_filter_entity_id const&) const noexcept = default;

 private:
  friend class dynamic_filter_identity_allocator;

  explicit constexpr dynamic_filter_entity_id(Rep value) noexcept : _value(value) {}

  Rep _value = 0;
};

/**
 * @brief Strong ordinal: a position in one specific index space.
 *
 * Zero is a valid position; there is deliberately no `is_valid()` — absence is expressed with
 * `std::optional`, never a sentinel value.
 */
template <class Tag, class Rep = std::size_t>
  requires std::integral<Rep>  // signed-capable: cudf::size_type ordinals
struct dynamic_filter_ordinal {
  using rep = Rep;

  Rep value = 0;

  constexpr auto operator<=>(dynamic_filter_ordinal const&) const noexcept = default;
};

//===-----------------------------------------------------------------------------------------===//
// Entity IDs
//===-----------------------------------------------------------------------------------------===//

/// One canonical publication sidecar for a producing join. Minted before physical key resolution,
/// so the resolved sidecar can ultimately contain no admitted key.
using dynamic_filter_publication_plan_id =
  dynamic_filter_entity_id<struct dynamic_filter_publication_plan_id_tag>;

/// One planned producer-to-consumer edge: a scan target today or a future SIP target. Its
/// existence does not imply that runtime publication occurs.
using dynamic_filter_target_id = dynamic_filter_entity_id<struct dynamic_filter_target_id_tag>;

/**
 * Identity assigned to one logical delivery channel for a consumer endpoint.
 *
 * In C1a-2a, the plan generator owns the channel-object-to-ID association and assigns the ID only
 * when producer wiring finds an existing channel. `sirius_dynamic_filter_set` does not yet store
 * this identity. C1b needs the ID to move into a persistent channel/topology record so it is
 * available independently of producer discovery and filter count.
 */
using dynamic_filter_channel_id = dynamic_filter_entity_id<struct dynamic_filter_channel_id_tag>;

/**
 * One constructed immutable filter. Declared for vocabulary completeness in C1a-2a; the
 * execution-scoped counter that mints filter IDs arrives in C1a-2d.
 */
using dynamic_filter_id = dynamic_filter_entity_id<struct dynamic_filter_id_tag>;

//===-----------------------------------------------------------------------------------------===//
// Ordinals (each names a DIFFERENT index space)
//===-----------------------------------------------------------------------------------------===//

/**
 * Position `j` in DuckDB's recorded vectors: `join_condition[j]` and `probe_info[t].columns[j]`.
 *
 * Example: if `join_condition == [2, 0]`, duckdb_filter_ordinal{0} selects the first recorded
 * filter and its first probe column. Its stored @ref join_condition_index is 2, not 0.
 */
using duckdb_filter_ordinal = dynamic_filter_ordinal<struct duckdb_filter_ordinal_tag>;

/// Index into the join-condition vector after DuckDB/Sirius equality-first reordering.
using join_condition_index = dynamic_filter_ordinal<struct join_condition_index_tag>;

/**
 * Compact position after Sirius narrows candidates to admitted keys.
 *
 * For example, if DuckDB filter ordinals [0, 1, 2] contain [equality, range, equality], admitted
 * keys may map filter ordinals [0, 2] to Sirius key ordinals [0, 1]. This is a third index space;
 * it is neither a DuckDB filter ordinal nor a join-condition index.
 */
using sirius_key_ordinal = dynamic_filter_ordinal<struct sirius_key_ordinal_tag>;

/// Scan-consumer column position in DuckDB `column_ids` space (Phase 1 scan targets). Declared
/// for C1b's compact scan-target vocabulary; C1a-2a keeps raw size_t probe positions.
using duckdb_column_ids_index = dynamic_filter_ordinal<struct duckdb_column_ids_index_tag>;

/**
 * Zero-based column index in the probe-side table batch entering a hash join.
 *
 * For a future SIP target, route discovery traces a producer key to an intermediate consuming join
 * and records which column of that join's probe batch carries the key. The consumer applies the
 * published filter to this column after taking the batch from its probe repository and before
 * preparing or hashing its probe keys.
 */
using probe_schema_ordinal =
  dynamic_filter_ordinal<struct probe_schema_ordinal_tag, cudf::size_type>;

//===-----------------------------------------------------------------------------------------===//
// Allocation
//===-----------------------------------------------------------------------------------------===//

/**
 * @brief Planning-time minting authority for publication-plan, target, and channel IDs.
 *
 * One instance is owned by the current plan generator (`sirius_physical_plan_generator`). It is
 * not thread-safe because planning is single-threaded. Future C3 route discovery must receive this
 * same instance rather than create a second counter domain. Generator-owned memos decide when an
 * existing producer or channel must reuse an ID; this allocator only supplies the next value in
 * each category.
 */
class dynamic_filter_identity_allocator {
 public:
  dynamic_filter_identity_allocator()  = default;
  ~dynamic_filter_identity_allocator() = default;

  // Copying or moving would fork or transfer the counter domain independently of the generator that
  // owns it. Consumers hold a reference to that single authority instead.
  dynamic_filter_identity_allocator(dynamic_filter_identity_allocator const&)            = delete;
  dynamic_filter_identity_allocator& operator=(dynamic_filter_identity_allocator const&) = delete;
  dynamic_filter_identity_allocator(dynamic_filter_identity_allocator&&)                 = delete;
  dynamic_filter_identity_allocator& operator=(dynamic_filter_identity_allocator&&)      = delete;

  [[nodiscard]] dynamic_filter_publication_plan_id mint_publication_plan_id()
  {
    return mint<dynamic_filter_publication_plan_id>(_next_publication_plan_id, "publication-plan");
  }
  [[nodiscard]] dynamic_filter_target_id mint_target_id()
  {
    return mint<dynamic_filter_target_id>(_next_target_id, "target");
  }
  [[nodiscard]] dynamic_filter_channel_id mint_channel_id()
  {
    return mint<dynamic_filter_channel_id>(_next_channel_id, "channel");
  }

 private:
  template <class Id>
  [[nodiscard]] static Id mint(std::uint64_t& next, char const* category)
  {
    using rep = typename Id::rep;
    if (next > static_cast<std::uint64_t>(std::numeric_limits<rep>::max())) {
      throw std::overflow_error(std::string{"dynamic-filter "} + category +
                                " identity space exhausted");
    }
    auto const value = static_cast<rep>(next);
    ++next;
    return Id{value};
  }

  std::uint64_t _next_publication_plan_id = 1;
  std::uint64_t _next_target_id           = 1;
  std::uint64_t _next_channel_id          = 1;
};

}  // namespace sirius::op
