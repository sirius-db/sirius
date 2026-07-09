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

//===-----------------------------------------------------------------------------------------===//
// Dynamic-filter identity vocabulary (C1a-2)
//
// One strong-value template per category, distinct tag per space. Two deliberately different
// validity contracts:
//
//   * ENTITY IDS (publication plan, target, channel, filter, execution generation): zero is
//     invalid; allocators mint from one. `is_valid()` exists only on these.
//   * ORDINALS (DuckDB filter ordinal, condition index, Sirius key ordinal, column_ids index,
//     probe-schema ordinal): zero is a valid position, so they do NOT share the entity validity
//     API — asking an ordinal `is_valid()` is a compile error by design.
//
// Exactly one dynamic_filter_identity_allocator exists per plan generator; C3's route registry
// receives it through the generator handoff and never starts a second counter domain. Object
// addresses are never event identity (program convention) — these values are.
//
// Formatting: log sites format the public `.value` member directly. The zero-dependency fmt
// `format_as()` hook is deliberately absent: the environment's spdlog (1.8.x) bundles fmt 7,
// which predates it, and op headers must not include spdlog/fmt (log/logging.hpp is the single
// spdlog entry point, and nvcc cannot compile fmt's chrono headers).
//===-----------------------------------------------------------------------------------------===//

#include <cudf/types.hpp>

#include <atomic>
#include <compare>  // IWYU pragma: keep — required for the defaulted <=> (std::strong_ordering)
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace sirius::op {

/// @brief Strong entity identity. Zero is invalid; allocators mint from one.
template <class Tag, class Rep = std::uint32_t>
  requires std::unsigned_integral<Rep>
struct dynamic_filter_entity_id {
  using rep = Rep;

  Rep value = 0;

  [[nodiscard]] constexpr bool is_valid() const noexcept { return value != Rep{0}; }

  constexpr auto operator<=>(dynamic_filter_entity_id const&) const noexcept = default;
};

/// @brief Strong ordinal (a position in one specific index space). Zero is a valid position;
/// there is deliberately no `is_valid()` — absence is expressed with `std::optional`, never a
/// sentinel value (design "value semantics, strong index types, and std::optional for absence").
template <class Tag, class Rep = std::size_t>
  requires std::integral<Rep>  // signed-capable: cudf::size_type ordinals
struct dynamic_filter_ordinal {
  using rep = Rep;

  Rep value = 0;

  constexpr auto operator<=>(dynamic_filter_ordinal const&) const noexcept = default;
};

//===-----------------------------------------------------------------------------------------===//
// Entity IDs (query-relative, monotonic, never cached across replans)
//===-----------------------------------------------------------------------------------------===//

/// One producing join plus its admitted key plan; filter construction is attempted once per plan.
using dynamic_filter_publication_plan_id =
  dynamic_filter_entity_id<struct dynamic_filter_publication_plan_id_tag>;

/// One scan or SIP consumer endpoint receiving a publication.
using dynamic_filter_target_id = dynamic_filter_entity_id<struct dynamic_filter_target_id_tag>;

/// One append-only delivery channel owned by one logical consumer endpoint.
using dynamic_filter_channel_id = dynamic_filter_entity_id<struct dynamic_filter_channel_id_tag>;

/// One constructed immutable filter; assigned before fan-out and identical in every target
/// channel receiving the object. Minted per execution (reset by the execution boundary).
using dynamic_filter_id = dynamic_filter_entity_id<struct dynamic_filter_id_tag>;

/// The execution-reset generation, derived from the query execution ID by the engine's central
/// begin pass — a reset epoch, not a timestamp (the query-relative event epoch is a separate
/// monotonic-clock time point).
using dynamic_filter_execution_generation =
  dynamic_filter_entity_id<struct dynamic_filter_execution_generation_tag, std::uint64_t>;

//===-----------------------------------------------------------------------------------------===//
// Ordinals (each names a DIFFERENT index space — never convert one into another implicitly)
//===-----------------------------------------------------------------------------------------===//

/// Position `j` in DuckDB's recorded vectors: `join_condition[j]` and `probe_info[t].columns[j]`.
/// Distinct from @ref join_condition_index: the element *stored at* `join_condition[j]` is itself
/// a `join_condition_index` value.
using duckdb_filter_ordinal = dynamic_filter_ordinal<struct duckdb_filter_ordinal_tag>;

/// Index into the (identically DuckDB- and Sirius-reordered) join-condition vector.
using join_condition_index = dynamic_filter_ordinal<struct join_condition_index_tag>;

/// Compact ordinal after Sirius narrowing: admitted keys only, unique and contiguous from zero.
using sirius_key_ordinal = dynamic_filter_ordinal<struct sirius_key_ordinal_tag>;

/// Scan-consumer column position in DuckDB `column_ids` space (Phase 1 scan targets).
using duckdb_column_ids_index = dynamic_filter_ordinal<struct duckdb_column_ids_index_tag>;

/// Join-probe-consumer column position in the consumer's runtime probe schema (SIP targets).
using probe_schema_ordinal =
  dynamic_filter_ordinal<struct probe_schema_ordinal_tag, cudf::size_type>;

//===-----------------------------------------------------------------------------------------===//
// Allocation
//===-----------------------------------------------------------------------------------------===//

/// @brief The single planning-time minting authority for publication-plan/target/channel IDs.
///
/// Owned by the plan generator; C3 receives it through the generator/route-registry handoff. NOT
/// thread-safe by design: planning is single-threaded, and the one-allocator rule is what makes
/// IDs unique per executable plan. The producer-node→publication-ID memo (one ID per producer no
/// matter how many targets are added) is generator-owned, beside its channel map — it keys on
/// DuckDB nodes, which this op-layer header must not name.
class dynamic_filter_identity_allocator {
 public:
  dynamic_filter_identity_allocator() = default;
  // Non-copyable (and thereby non-movable): copying would silently fork the counter domain and
  // mint duplicate IDs. Consumers (C3's registry) hold a reference, never a value.
  dynamic_filter_identity_allocator(dynamic_filter_identity_allocator const&)            = delete;
  dynamic_filter_identity_allocator& operator=(dynamic_filter_identity_allocator const&) = delete;

  [[nodiscard]] dynamic_filter_publication_plan_id mint_publication_plan_id() noexcept
  {
    return {next_publication_plan_id_++};
  }
  [[nodiscard]] dynamic_filter_target_id mint_target_id() noexcept { return {next_target_id_++}; }
  [[nodiscard]] dynamic_filter_channel_id mint_channel_id() noexcept
  {
    return {next_channel_id_++};
  }

 private:
  // Wrap at 2^32 would mint the invalid zero ID; unreachable in practice — counts are bounded by
  // producing joins/targets in one executable plan, not by rows or batches.
  std::uint32_t next_publication_plan_id_ = 1;
  std::uint32_t next_target_id_           = 1;
  std::uint32_t next_channel_id_          = 1;
};

/// @brief Execution-scoped identity state retained by the accepted executable plan.
///
/// The engine's single execution-boundary pass calls @ref begin_execution exactly once per
/// execution, after ALL tasks of the prior execution have completed (quiescence — a straggler
/// minting concurrently with the reset would produce duplicate IDs across generations) and before
/// any task of the new execution runs; publishers then mint filter IDs concurrently. Filter IDs
/// restart at one each execution — they are execution-relative, and events disambiguate
/// executions with the generation.
class dynamic_filter_execution_identity {
 public:
  /// Reset for a new execution: restart filter IDs at one, then release-publish the generation —
  /// an observer that acquires the new generation is thereby guaranteed to see the reset counter.
  /// (Belt-and-braces: the quiescent single-threaded boundary above is the primary guarantee.)
  void begin_execution(dynamic_filter_execution_generation generation) noexcept
  {
    next_filter_id_.store(1, std::memory_order_relaxed);
    generation_.store(generation.value, std::memory_order_release);
  }

  [[nodiscard]] dynamic_filter_execution_generation generation() const noexcept
  {
    return {generation_.load(std::memory_order_acquire)};
  }

  /// Concurrent-safe; relaxed order suffices — uniqueness is the only requirement, and event
  /// ordering comes from the query-relative clock, not from ID allocation order.
  [[nodiscard]] dynamic_filter_id mint_filter_id() noexcept
  {
    return {next_filter_id_.fetch_add(1, std::memory_order_relaxed)};
  }

 private:
  std::atomic<std::uint64_t> generation_{0};  ///< Zero until the first begin_execution.
  std::atomic<std::uint32_t> next_filter_id_{1};
};

}  // namespace sirius::op
