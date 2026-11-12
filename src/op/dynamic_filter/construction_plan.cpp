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

#include "op/dynamic_filter/construction_plan.hpp"

#include "helper/numeric_narrowing.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/bloom_sizing.hpp"
#include "op/dynamic_filter/detail/domain_gate.hpp"
#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <rmm/aligned.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op {

std::vector<std::vector<binding_facts>> binding_facts_of(dynamic_filter_publish_plan const& plan)
{
  std::vector<std::vector<binding_facts>> facts(plan.admitted_keys().size());
  for (auto const& target : plan.probe_targets()) {
    for (auto const& binding : target.key_bindings) {
      facts[binding.admitted_key_index].push_back(
        {.probe_type       = binding.probe_storage_type,
         .accepts_zone_map = target.accepts_zone_map_filters});
    }
  }
  return facts;
}

namespace {

/// Bytes one Bloom array of @p blocks blocks occupies, rounded up to the allocation alignment.
[[nodiscard]] std::optional<std::uint64_t> aligned_bloom_bytes(std::size_t blocks) noexcept
{
  constexpr auto alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
  constexpr auto maximum   = std::numeric_limits<std::size_t>::max();
  if (blocks > maximum / bloom_bytes_per_block) { return std::nullopt; }
  auto const raw = blocks * bloom_bytes_per_block;
  if (raw > maximum - (alignment - 1)) { return std::nullopt; }
  return rmm::align_up(raw, alignment);
}

/// The membership filter `choose_membership_filter` selects for @p valid_rows valid keys.
[[nodiscard]] membership_plan choose_membership(cudf::data_type type,
                                                std::size_t valid_rows,
                                                construction_limits const& limits)
{
  auto const kind = choose_membership_filter(
    {.build_rows     = valid_rows,
     .l2_cache_bytes = limits.l2_bytes,
     .estimated_hash_set_bytes =
       sirius_dynamic_in_list_filter::estimated_set_bytes(valid_rows, type),
     .inlist_max_l2_fraction = limits.inlist_max_l2_fraction,
     .supports_small_in_list = sirius_dynamic_small_in_list_filter::supports(type, valid_rows),
     .supports_hash_in_list  = sirius_dynamic_in_list_filter::supports(type, valid_rows),
     .supports_bloom         = sirius_dynamic_bloom_filter::supports(type)});
  switch (kind) {
    case membership_filter_kind::small_in_list:
    case membership_filter_kind::hash_in_list:
      return exact_membership{.kind = kind, .valid_rows = valid_rows};
    case membership_filter_kind::bloom:
      return bloom_membership{.sized_rows = valid_rows, .blocks = bloom_blocks_for(valid_rows)};
    case membership_filter_kind::none: break;
  }
  return std::monostate{};
}

}  // namespace

construction_plan plan_construction(
  build_input_facts const& facts,
  std::span<dynamic_filter_publish_plan::admitted_key const> admitted_keys,
  std::span<std::vector<binding_facts> const> bindings,
  construction_limits const& limits)
{
  if (bindings.size() != admitted_keys.size()) {
    throw std::logic_error(
      "[plan_construction] every admitted key needs its (possibly empty) binding list");
  }
  construction_plan plan;
  auto& counts = plan.selection;
  for (std::size_t index = 0; index < admitted_keys.size(); ++index) {
    auto const& key_bindings = bindings[index];
    if (key_bindings.empty()) { continue; }
    auto const& key = admitted_keys[index];
    if (key.build_key_ordinal < 0 ||
        std::cmp_greater_equal(key.build_key_ordinal, facts.columns.size())) {
      throw std::logic_error(
        "[plan_construction] an admitted key's build ordinal lies outside every build batch");
    }
    auto& line = plan.outcomes.emplace_back(
      dynamic_filter_key_line{.admitted_key_index      = index,
                              .planner_condition_index = key.planner_condition_index,
                              .build_key_ordinal       = key.build_key_ordinal,
                              .outcome                 = {.rows = facts.total_rows}});
    auto& outcome = line.outcome;

    // Carrier rule: equality first, since `can_restore_to` requires a strictly narrower carrier.
    auto const& column    = facts.columns[static_cast<std::size_t>(key.build_key_ordinal)];
    auto const recorded   = key.storage_type;
    bool const restorable = !column.carriers.empty() &&
                            std::ranges::all_of(column.carriers, [&](cudf::data_type carrier) {
                              return carrier == recorded || can_restore_to(carrier, recorded);
                            });
    auto const construction_type =
      column.carriers.size() == 1 && membership_same_family(recorded, column.carriers.front())
        ? column.carriers.front()
        : recorded;

    // Eligibility, judged before the domain gate.
    auto const domain   = restorable ? classify_membership_key(construction_type) : std::nullopt;
    bool const readable = domain && std::ranges::any_of(key_bindings, [&](binding_facts const& b) {
                            return b.probe_type.id() == cudf::type_id::EMPTY ||
                                   membership_probe_compatible(*domain, b.probe_type);
                          });
    bool const zone_possible = restorable && limits.emit_zone_map_filters &&
                               sirius_dynamic_zone_map_filter::supports(recorded) &&
                               !key.preserve_nulls;
    bool const zone_deliverable =
      zone_possible && std::ranges::any_of(key_bindings, [&](binding_facts const& b) {
        return b.accepts_zone_map && b.probe_type == recorded;
      });
    bool const gated = detail::consider_bound_key(counts,
                                                  key,
                                                  facts.total_rows,
                                                  limits.domain_coverage_threshold,
                                                  restorable && (readable || zone_possible));
    if (!restorable) {
      ++counts.keys_skipped_type_mismatch;
      outcome.reason = dynamic_filter_key_reason::type_mismatch;
      continue;
    }
    if (domain && !readable) { counts.bindings_skipped_incompatible_probe += key_bindings.size(); }
    if (gated) {
      ++counts.keys_skipped_domain_gate;
      outcome.reason = dynamic_filter_key_reason::domain_gate;
      continue;
    }

    key_plan planned{.admitted_key_index   = index,
                     .build_key_ordinal    = key.build_key_ordinal,
                     .recorded_type        = recorded,
                     .construction_type    = construction_type,
                     .membership           = std::monostate{},
                     .zone_map             = zone_possible,
                     .zone_map_deliverable = zone_deliverable,
                     .fit_pending          = false};
    if (column.null_rows) {
      auto const valid_rows = facts.total_rows - std::min(*column.null_rows, facts.total_rows);
      outcome.valid_rows    = valid_rows;
      if (readable) {
        planned.membership = choose_membership(construction_type, valid_rows, limits);
      }
    } else if (readable && sirius_dynamic_bloom_filter::supports(construction_type)) {
      // Without a null count the valid rows are unknown: an exact list could not be sized.
      planned.membership = bloom_membership{.sized_rows = facts.total_rows,
                                            .blocks     = bloom_blocks_for(facts.total_rows)};
      ++counts.keys_sized_on_total_rows;
    }
    bool const has_membership = !std::holds_alternative<std::monostate>(planned.membership);
    planned.fit_pending = has_membership && construction_type.id() == cudf::type_id::DECIMAL128;
    if (domain && !readable) {
      outcome.reason = dynamic_filter_key_reason::unreadable_probe;
    } else if (!has_membership && !zone_deliverable) {
      outcome.reason = dynamic_filter_key_reason::unsupported_type;
    }
    if (has_membership || planned.needs_bounds()) { plan.keys.push_back(planned); }
  }

  // The cap applies to every Bloom key's aligned array together; exact and bound storage is
  // bounded by the IN-list policy instead.
  std::uint64_t required = 0;
  std::size_t bloom_keys = 0;
  for (auto const& planned : plan.keys) {
    auto const* bloom = std::get_if<bloom_membership>(&planned.membership);
    if (bloom == nullptr) { continue; }
    ++bloom_keys;
    auto const bytes = aligned_bloom_bytes(bloom->blocks);
    required         = bytes && *bytes <= std::numeric_limits<std::uint64_t>::max() - required
                         ? required + *bytes
                         : std::numeric_limits<std::uint64_t>::max();
  }
  if (bloom_keys != 0 && required > limits.max_bloom_bytes_per_gpu) {
    for (auto& planned : plan.keys) {
      if (!std::holds_alternative<bloom_membership>(planned.membership)) { continue; }
      planned.membership  = std::monostate{};
      planned.fit_pending = false;
      auto const line     = std::ranges::find(
        plan.outcomes, planned.admitted_key_index, &dynamic_filter_key_line::admitted_key_index);
      line->outcome.reason = dynamic_filter_key_reason::bloom_cap;
    }
    counts.keys_skipped_bloom_size_gate += bloom_keys;
    log::log_noexcept([&] {
      SIRIUS_LOG_INFO(
        "[plan_construction] Bloom filters refused: per-GPU cap, keys={}, required_bytes={}, "
        "cap={}",
        bloom_keys,
        required == std::numeric_limits<std::uint64_t>::max() ? std::string{"overflow"}
                                                              : std::to_string(required),
        limits.max_bloom_bytes_per_gpu);
    });
    std::erase_if(plan.keys, [](key_plan const& planned) {
      return std::holds_alternative<std::monostate>(planned.membership) && !planned.needs_bounds();
    });
  }
  return plan;
}

}  // namespace sirius::op
