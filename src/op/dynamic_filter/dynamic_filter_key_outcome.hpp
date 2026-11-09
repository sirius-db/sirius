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
 * @file dynamic_filter_key_outcome.hpp
 * @brief The per-key outcome that `dynamic_filter_publication_session` logs once for every admitted
 * key with a binding, and the structural join identity that lets
 * `scripts/dynamic_filter_key_parity.py` match those lines across two builds of the same query.
 *
 * Each enabled session emits one `[dynamic_filter_key]` DEBUG line per admitted key that has at
 * least one binding, when the session becomes terminal, including keys that ended without a filter.
 * The line reports what the probe side could receive: `kind` and `zone_map` describe a filter that
 * reached the session's fan-out and that at least one of the key's bindings accepts by type and
 * route; a filter no binding could ever accept is reported as absent, with `reason` naming why.
 * `pushed` counts the bindings that actually received a filter, which also depends on whether their
 * channels still accepted filters.
 */

#pragma once

#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::op {

/**
 * @brief Why a key ended with the filters its line reports
 *
 * The three `attempt_*` reasons mean the whole publication attempt ended before this key's filters
 * reached the fan-out; `dynamic_filter_key_outcome::attempt_counter` names the counter that
 * recorded the attempt's end.
 */
enum class dynamic_filter_key_reason : std::uint8_t {
  published,         ///< Printed `-`: the key's filters reached the fan-out
  domain_gate,       ///< The domain-coverage gate skipped the key
  type_mismatch,     ///< The build column's type cannot carry the recorded key type
  unsupported_type,  ///< No filter kind supports the key type
  unreadable_probe,  ///< No binding can read the key's membership filter
  bloom_cap,  ///< The accumulated Bloom arrays exceed `max_dynamic_filter_bloom_bytes_per_gpu`
  decimal128_unfit,  ///< A DECIMAL128 build value lies outside the int64 membership rep
  attempt_skip,      ///< The attempt ended without a filter
  attempt_fail,      ///< The attempt failed
  attempt_cancel,    ///< The attempt was cancelled
  not_constructed    ///< No attempt constructed this key's filters, for example an empty build
};

/**
 * @brief What one admitted key's publication produced
 */
struct dynamic_filter_key_outcome {
  /** @brief Build rows the key was judged on; 0 when it was never judged */
  std::size_t rows = 0;
  /** @brief Non-null build rows, when the path counted them */
  std::optional<std::size_t> valid_rows;
  /** @brief The membership filter some binding can read */
  membership_filter_kind kind = membership_filter_kind::none;
  /** @brief IN-list key count or Bloom array bytes; 0 without a membership filter */
  std::size_t size = 0;
  /** @brief Whether a zone map some binding accepts reached the fan-out */
  bool zone_map = false;
  /** @brief Bindings that received this key's membership filter or zone map */
  std::size_t pushed = 0;
  /** @brief Why the key ended this way */
  dynamic_filter_key_reason reason = dynamic_filter_key_reason::not_constructed;
  /** @brief For an `attempt_*` reason, the `dynamic_filter_stats` counter that ended the attempt */
  std::string_view attempt_counter{};
};

/**
 * @brief Identifies one hash join across plans of the same query, independently of operator IDs
 */
struct dynamic_filter_join_identity {
  /** @brief `dynamic_filter_join_signature` of the join's type and conditions */
  std::uint64_t signature = 0;
  /** @brief The join's position among the query's joins with the same signature, by operator ID */
  std::size_t occurrence = 0;
  /** @brief The join's operator ID, for humans only; empty before the plan is numbered */
  std::optional<std::size_t> operator_id;
};

/**
 * @brief One `[dynamic_filter_key]` log line
 */
struct dynamic_filter_key_line {
  /** @brief The join that owns the key */
  dynamic_filter_join_identity join;
  /** @brief The key's index in its plan's admitted keys; not printed */
  std::size_t admitted_key_index = 0;
  /** @brief The key's original join-condition index */
  std::size_t planner_condition_index = 0;
  /** @brief The key's column in the build table */
  cudf::size_type build_key_ordinal = 0;
  /** @brief What the key's publication produced */
  dynamic_filter_key_outcome outcome;
};

/**
 * @brief Spells @p line as its log message
 *
 * The message is `[dynamic_filter_key] v=1 sig=<16 hex> occ=<n> join=<id|-> condition=<n>
 * build_ordinal=<n> rows=<n> valid=<n|unknown> kind=<none|bloom|small_in_list|hash_in_list>
 * size=<n> zone_map=<0|1> pushed=<n> reason=<token>`, where an `attempt_*` reason is followed by
 * `_<counter>`. `scripts/dynamic_filter_key_parity.py` parses this form, so a change to it must
 * bump `v`.
 */
[[nodiscard]] std::string to_string(dynamic_filter_key_line const& line);

/**
 * @brief The 64-bit FNV-1a hash of @p join_type followed by each of @p conditions, each condition
 * preceded by a unit-separator byte so that no two condition lists share a byte stream
 *
 * The planner passes `duckdb::JoinTypeToString` and each condition spelled `lhs <operator> rhs`, so
 * equal joins in two plans of one query hash equally whatever their operator IDs.
 */
[[nodiscard]] std::uint64_t dynamic_filter_join_signature(
  std::string_view join_type, std::span<std::string const> conditions) noexcept;

/**
 * @brief The occurrence of each join among the joins sharing its signature
 *
 * @param signatures One signature per join, in operator-ID order
 * @return For each join, how many earlier joins have the same signature
 */
[[nodiscard]] std::vector<std::size_t> dynamic_filter_join_occurrences(
  std::span<std::uint64_t const> signatures);

}  // namespace sirius::op

/**
 * @brief Formats a `sirius::op::dynamic_filter_key_line` as `sirius::op::to_string` spells it, so a
 * log macro formats it only when its level is enabled
 */
template <>
struct std::formatter<sirius::op::dynamic_filter_key_line> : std::formatter<std::string_view> {
  auto format(sirius::op::dynamic_filter_key_line const& line, std::format_context& context) const
  {
    return std::formatter<std::string_view>::format(sirius::op::to_string(line), context);
  }
};
