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

#include "op/dynamic_filter/dynamic_filter_key_outcome.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sirius::op {

namespace {

constexpr std::uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t k_fnv_prime        = 1099511628211ULL;
constexpr unsigned char k_unit_separator   = 0x1F;

constexpr void fnv1a(std::uint64_t& hash, unsigned char byte) noexcept
{
  hash ^= byte;
  hash *= k_fnv_prime;
}

constexpr void fnv1a(std::uint64_t& hash, std::string_view bytes) noexcept
{
  for (char const byte : bytes) {
    fnv1a(hash, static_cast<unsigned char>(byte));
  }
}

[[nodiscard]] constexpr std::string_view to_token(membership_filter_kind kind) noexcept
{
  switch (kind) {
    case membership_filter_kind::none: return "none";
    case membership_filter_kind::small_in_list: return "small_in_list";
    case membership_filter_kind::hash_in_list: return "hash_in_list";
    case membership_filter_kind::bloom: return "bloom";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view to_token(dynamic_filter_key_reason reason) noexcept
{
  switch (reason) {
    case dynamic_filter_key_reason::published: return "-";
    case dynamic_filter_key_reason::domain_gate: return "domain_gate";
    case dynamic_filter_key_reason::type_mismatch: return "type_mismatch";
    case dynamic_filter_key_reason::unsupported_type: return "unsupported_type";
    case dynamic_filter_key_reason::unreadable_probe: return "unreadable_probe";
    case dynamic_filter_key_reason::bloom_cap: return "bloom_cap";
    case dynamic_filter_key_reason::decimal128_unfit: return "decimal128_unfit";
    case dynamic_filter_key_reason::attempt_skip: return "attempt_skip_";
    case dynamic_filter_key_reason::attempt_fail: return "attempt_fail_";
    case dynamic_filter_key_reason::attempt_cancel: return "attempt_cancel_";
    case dynamic_filter_key_reason::not_constructed: return "not_constructed";
  }
  return "invalid";
}

[[nodiscard]] constexpr bool ends_attempt(dynamic_filter_key_reason reason) noexcept
{
  return reason == dynamic_filter_key_reason::attempt_skip ||
         reason == dynamic_filter_key_reason::attempt_fail ||
         reason == dynamic_filter_key_reason::attempt_cancel;
}

}  // namespace

std::string to_string(dynamic_filter_key_line const& line)
{
  auto const& outcome = line.outcome;
  std::string text    = std::format(
    "[dynamic_filter_key] v=1 sig={:016x} occ={} join=", line.join.signature, line.join.occurrence);
  auto out = std::back_inserter(text);
  if (line.join.operator_id) {
    std::format_to(out, "{}", *line.join.operator_id);
  } else {
    text += '-';
  }
  std::format_to(out,
                 " condition={} build_ordinal={} rows={} valid=",
                 line.planner_condition_index,
                 line.build_key_ordinal,
                 outcome.rows);
  if (outcome.valid_rows) {
    std::format_to(out, "{}", *outcome.valid_rows);
  } else {
    text += "unknown";
  }
  std::format_to(out,
                 " kind={} size={} zone_map={} pushed={} reason={}{}",
                 to_token(outcome.kind),
                 outcome.size,
                 outcome.zone_map ? 1 : 0,
                 outcome.pushed,
                 to_token(outcome.reason),
                 ends_attempt(outcome.reason) ? outcome.attempt_counter : std::string_view{});
  return text;
}

std::uint64_t dynamic_filter_join_signature(std::string_view join_type,
                                            std::span<std::string const> conditions) noexcept
{
  std::uint64_t hash = k_fnv_offset_basis;
  fnv1a(hash, join_type);
  for (auto const& condition : conditions) {
    fnv1a(hash, k_unit_separator);
    fnv1a(hash, condition);
  }
  return hash;
}

std::vector<std::size_t> dynamic_filter_join_occurrences(std::span<std::uint64_t const> signatures)
{
  std::vector<std::size_t> occurrences;
  occurrences.reserve(signatures.size());
  std::unordered_map<std::uint64_t, std::size_t> seen;
  for (auto const signature : signatures) {
    occurrences.push_back(seen[signature]++);
  }
  return occurrences;
}

}  // namespace sirius::op
