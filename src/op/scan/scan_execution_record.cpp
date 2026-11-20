/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/scan_execution_record.hpp"

#include "op/scan/gpu_ingestible_types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace sirius::op::scan {
visibility_bindings remaining_visibility(std::span<applied_visibility const> required,
                                         std::span<applied_visibility const> applied,
                                         bool complete)
{
  std::unordered_map<std::string_view, void const*> pending;
  for (auto const& binding : required) {
    if (binding.file.empty() || !binding.input ||
        !pending.emplace(binding.file, binding.input.get()).second)
      throw std::invalid_argument("invalid required visibility binding");
  }
  for (auto const& binding : applied) {
    auto it = pending.find(binding.file);
    if (it == pending.end() || it->second != binding.input.get())
      throw std::invalid_argument("unknown, duplicate or misbound applied visibility");
    pending.erase(it);
  }
  if (complete && !pending.empty())
    throw std::invalid_argument("visibility reported complete with missing bindings");
  visibility_bindings remaining;
  remaining.reserve(pending.size());
  for (auto const& binding : required)
    if (pending.contains(binding.file)) remaining.push_back(binding);
  return remaining;
}

void finish_visibility(std::span<applied_visibility const> required,
                       visibility_bindings& applied,
                       bool& complete,
                       std::function<void(std::span<applied_visibility const>)> const& apply)
{
  auto pending = remaining_visibility(required, applied, complete);
  if (!pending.empty()) {
    if (!apply) throw std::invalid_argument("missing visibility step");
    apply(pending);
    applied.insert(applied.end(), pending.begin(), pending.end());
  }
  complete = true;
}

scan_execution_record translate_legacy_execution(legacy_execution_facts const& facts,
                                                 residual_filter const& catalogue,
                                                 uint64_t contract,
                                                 uint64_t generation)
{
  scan_execution_record record;
  record.contract                = contract;
  record.generation              = generation;
  record.logical_schema_complete = true;
  switch (facts.state) {
    case filter_state::UNFILTERED:
    case filter_state::ROWGROUP_FILTERED: break;
    case filter_state::ROW_FILTERED: record.whole_static_complete = true; break;
    case filter_state::ROW_FILTERED_AND_PROJECTED:
      record.whole_static_complete = true;
      record.projection_complete   = true;
      break;
    default: throw std::invalid_argument("unknown legacy filter state");
  }
  record.whole_static_complete |= facts.row_filtered;
  record.whole_static_complete |= catalogue.empty();
  record.predicates = catalogue.progress_from_answers(facts.answer_columns, facts.answers_enforced);
  return record;
}

void validate_execution_record(scan_execution_record const& record,
                               residual_filter const& catalogue,
                               uint64_t contract,
                               uint64_t generation,
                               execution_requirements const* required)
{
  if (!contract || !generation || record.contract != contract || record.generation != generation)
    throw std::invalid_argument("scan execution record binding mismatch");
  catalogue.validate_progress(record.predicates);
  if (required) {
    remaining_visibility(
      required->visibility, record.visibility_applied, record.visibility_complete);
    if (!required->layout || record.layout != required->layout)
      throw std::invalid_argument("scan execution record ordered layout mismatch");
  }
  if (record.layout_width)
    for (auto const& p : record.predicates)
      if (p.answer_slot && *p.answer_slot >= *record.layout_width)
        throw std::invalid_argument("predicate answer is outside its reported layout");
  if (record.projection_complete && !record.whole_static_complete && !catalogue.empty())
    throw std::invalid_argument("scan output assembled with an incomplete static predicate");
  std::unordered_set<std::string_view> files;
  for (auto const& binding : record.visibility_applied) {
    if (binding.file.empty() || !binding.input)
      throw std::invalid_argument("scan visibility has an incomplete binding");
    if (record.visibility_applied.size() > 1 && !files.insert(binding.file).second)
      throw std::invalid_argument("duplicate scan visibility binding");
  }
}

execution_record_selection select_execution_record(scan_execution_record const* producer,
                                                   legacy_execution_facts const& legacy,
                                                   residual_filter const& catalogue,
                                                   uint64_t contract,
                                                   uint64_t generation,
                                                   execution_requirements const* required)
{
  if (producer && producer->authoritative) {
    validate_execution_record(*producer, catalogue, contract, generation, required);
    if (producer->producer == execution_record_producer::legacy ||
        !producer->logical_schema_complete || !producer->visibility_complete)
      throw std::invalid_argument("authoritative scan result has unfinished adapter work");
    return {*producer, true, false};
  }
  execution_record_selection selected;
  try {
    selected.facts = translate_legacy_execution(legacy, catalogue, contract, generation);
  } catch (std::invalid_argument const&) {
    // Old consumers ignored duplicate/unoffered answers. Report the failed
    // translation without changing their established decision or error path.
    selected.facts = translate_legacy_execution(
      {legacy.state, {}, false, legacy.row_filtered}, catalogue, contract, generation);
    selected.facts.predicates =
      catalogue.progress_from_legacy_answers(legacy.answer_columns, legacy.answers_enforced);
    selected.shadow_mismatch = true;
  }
  if (!producer) return selected;
  try {
    validate_execution_record(*producer, catalogue, contract, generation, required);
    auto const& old = selected.facts;
    selected.shadow_mismatch |= old.whole_static_complete != producer->whole_static_complete ||
                                old.projection_complete != producer->projection_complete;
    if (!old.whole_static_complete) {
      auto same = [](auto const& a, auto const& b) {
        return a.conjunct == b.conjunct && a.stage == b.stage && a.answer_slot == b.answer_slot;
      };
      for (auto const& prior : old.predicates)
        if (std::none_of(producer->predicates.begin(),
                         producer->predicates.end(),
                         [&](auto const& p) { return same(prior, p); }))
          selected.shadow_mismatch = true;
      for (auto const& p : producer->predicates) {
        auto prior = std::find_if(old.predicates.begin(), old.predicates.end(), [&](auto const& q) {
          return q.conjunct == p.conjunct;
        });
        if (prior != old.predicates.end()) continue;
        // Legacy UNFILTERED cannot express partial reader coverage. Additional
        // enforced IDs refine that fact; missing/rebound answers are not refinements.
        if (p.stage == predicate_stage::enforced && !p.answer_slot)
          selected.shadow_coverage_added = true;
        else
          selected.shadow_mismatch = true;
      }
    }
  } catch (std::invalid_argument const&) {
    selected.shadow_mismatch = true;
  }
  if (selected.shadow_mismatch) selected.shadow_coverage_added = false;
  return selected;
}
}  // namespace sirius::op::scan
