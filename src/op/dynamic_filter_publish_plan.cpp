/*
 * Copyright 2025, Sirius Contributors.
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

// sirius
#include <op/dynamic_filter_publish_plan.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <sirius/exception.hpp>

// cucascade
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sirius::op {

//===----------------------------------------------------------------------===//
// dynamic_filter_publish_plan
//===----------------------------------------------------------------------===//

dynamic_filter_publish_plan::dynamic_filter_publish_plan(
  dynamic_filter_publication_plan_id publication_plan_id,
  std::vector<dynamic_filter_planning_ordinal_view> ordinals,
  std::vector<probe_target> probe_targets,
  bool emit_zone_map_filters,
  std::vector<std::size_t> build_key_domain_cardinalities,
  std::vector<dynamic_filter_replica_space> replica_spaces,
  double domain_coverage_threshold)
  : _publication_plan_id(publication_plan_id),
    _ordinals(std::move(ordinals)),
    _probe_targets(std::move(probe_targets)),
    _emit_zone_map_filters(emit_zone_map_filters),
    _build_key_domain_cardinalities(std::move(build_key_domain_cardinalities)),
    _domain_coverage_threshold(domain_coverage_threshold),
    _replica_spaces(std::move(replica_spaces))
{
  if (!_probe_targets.empty() && _replica_spaces.empty()) {
    throw std::invalid_argument(
      "[dynamic_filter_publish_plan] An enabled dynamic-filter publish plan requires at least one "
      "GPU memory space");
  }
  for (auto const& target : _replica_spaces) {
    if (target.get_gpu_space().get_tier() != cucascade::memory::Tier::GPU) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] A dynamic-filter replica target must be a GPU memory space");
    }
    if (target.get_host_staging_space().get_tier() != cucascade::memory::Tier::HOST) {
      throw std::invalid_argument(
        "[dynamic_filter_publish_plan] Dynamic-filter staging requires a HOST memory space");
    }
  }
  auto const device_less = [](auto const& lhs, auto const& rhs) {
    return lhs.get_gpu_space().get_device_id() < rhs.get_gpu_space().get_device_id();
  };
  auto const same_device = [](auto const& lhs, auto const& rhs) {
    return lhs.get_gpu_space().get_device_id() == rhs.get_gpu_space().get_device_id();
  };
  // Ensure no duplicated device ids
  std::sort(_replica_spaces.begin(), _replica_spaces.end(), device_less);
  _replica_spaces.erase(std::unique(_replica_spaces.begin(), _replica_spaces.end(), same_device),
                        _replica_spaces.end());
}

//===----------------------------------------------------------------------===//
// dynamic_filter_publish_plan_builder
//===----------------------------------------------------------------------===//

dynamic_filter_publish_plan_builder::dynamic_filter_publish_plan_builder(
  dynamic_filter_publication_plan_id publication_plan_id,
  bool wired,
  std::vector<scan_target_draft> scan_targets,
  bool emit_zone_map_filters,
  double domain_coverage_threshold,
  std::vector<dynamic_filter_replica_space> replica_spaces,
  std::vector<dynamic_filter_key_candidate> key_candidates)
  : _publication_plan_id(publication_plan_id),
    _wired(wired),
    _scan_targets(std::move(scan_targets)),
    _emit_zone_map_filters(emit_zone_map_filters),
    _domain_coverage_threshold(domain_coverage_threshold),
    _replica_spaces(std::move(replica_spaces)),
    _key_candidates(std::move(key_candidates))
{
}

void dynamic_filter_publish_plan_builder::resolve_keys(
  std::vector<dynamic_filter_key_decision> decisions,
  std::vector<dynamic_filter_key_plan> resolved_keys,
  std::size_t build_input_column_count)
{
  if (_keys_resolved) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] resolve_keys called twice on one builder");
  }
  _decisions                = std::move(decisions);
  _resolved_keys            = std::move(resolved_keys);
  _build_input_column_count = build_input_column_count;

  // Build the per-candidate records once, right here: planning_view() (read before the freeze)
  // and finalize() (the freeze) hand out THE SAME records, so what C3 read at bind time and what
  // the runtime plan carries cannot disagree. The bounds guards tolerate mismatched decision/key
  // counts because finalize() is where those mismatches are REPORTED — a record built from bad
  // input never reaches a runtime plan.
  _ordinal_records.reserve(_key_candidates.size());
  std::size_t next_key = 0;
  for (std::size_t j = 0; j < _key_candidates.size(); ++j) {
    dynamic_filter_planning_ordinal_view v{};
    v.duckdb_ordinal  = _key_candidates[j].duckdb_ordinal;
    v.condition_index = _key_candidates[j].condition_index;
    v.decision = j < _decisions.size() ? _decisions[j] : dynamic_filter_key_decision::unresolved;
    if (v.decision == dynamic_filter_key_decision::admitted && next_key < _resolved_keys.size()) {
      v.admitted_key = _resolved_keys[next_key];
      v.build_type   = _resolved_keys[next_key].build_type;
      ++next_key;
    }
    _ordinal_records.push_back(v);
  }

  _keys_resolved = true;
}

dynamic_filter_planning_view dynamic_filter_publish_plan_builder::planning_view() const
{
  if (!_keys_resolved) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] planning_view before resolve_keys");
  }
  dynamic_filter_planning_view view{};
  view.publication_plan_id = _publication_plan_id;
  view.wired               = _wired;
  // Live targets alone make the plan enabled: a producer whose every key was rejected still
  // attempts publication and reports the terminal "Pushed 0 ..." line, exactly like the old
  // on-the-fly checks did (zero admitted keys is a publish outcome, not a disabled plan).
  view.enabled           = !_scan_targets.empty();
  view.by_duckdb_ordinal = ordinal_records();
  return view;
}

std::shared_ptr<dynamic_filter_publish_plan const> dynamic_filter_publish_plan_builder::finalize()
  const
{
  auto const fail = [this](char const* what) {
    throw sirius::internal_exception(
      "[dynamic_filter_publish_plan_builder] publication plan {} failed final validation: {}",
      _publication_plan_id.value,
      what);
  };

  if (!_keys_resolved) { fail("keys were never resolved"); }

  // 1. decision count equals candidate count (identity match is positional by construction).
  if (_decisions.size() != _key_candidates.size()) { fail("decision count != candidate count"); }
  // 2./4. every admitted decision has exactly one key; key count == admitted-decision count.
  auto const admitted_count = static_cast<std::size_t>(
    std::count(_decisions.begin(), _decisions.end(), dynamic_filter_key_decision::admitted));
  if (_resolved_keys.size() != admitted_count) {
    fail("resolved key count != admitted decision count");
  }
  // 3. rejected decisions have no Sirius ordinal — enforced structurally: keys exist only in
  //    _resolved_keys, and (5.) their compact ordinals must be unique and contiguous from zero,
  //    in candidate order.
  {
    auto next_key = std::size_t{0};
    for (std::size_t j = 0; j < _decisions.size(); ++j) {
      if (_decisions[j] != dynamic_filter_key_decision::admitted) { continue; }
      auto const& key = _resolved_keys[next_key];
      if (key.ordinal.value != next_key) {
        fail("Sirius key ordinals are not contiguous in candidate order");
      }
      // 6./7. provenance bijection: the key names exactly its candidate's ordinal spaces.
      if (key.duckdb_ordinal != _key_candidates[j].duckdb_ordinal ||
          key.condition_index != _key_candidates[j].condition_index) {
        fail("admitted key does not match its candidate's ordinals");
      }
      // 8. each build column is inside the captured build input width.
      if (key.build_column_index >= _build_input_column_count) {
        fail("admitted key's build column exceeds the build input width");
      }
      ++next_key;
    }
  }
  // 6. DuckDB ordinals unique and below key count (candidate side).
  {
    std::vector<bool> seen(_key_candidates.size(), false);
    for (auto const& cand : _key_candidates) {
      auto const j = static_cast<std::size_t>(cand.duckdb_ordinal.value);
      if (j >= _key_candidates.size() || seen[j]) {
        fail("duplicate or out-of-range DuckDB ordinal");
      }
      seen[j] = true;
    }
  }
  // 9. full-arity scan target column/type vectors equal DuckDB key count.
  for (auto const& target : _scan_targets) {
    if (target.probe_col_idx.size() != _key_candidates.size() ||
        target.probe_col_type.size() != _key_candidates.size()) {
      fail("scan target arity != DuckDB key count");
    }
  }
  // 10. enabled plans: nonzero IDs, unique target IDs, non-null channels.
  bool const enabled = !_scan_targets.empty();  // zero admitted keys still publishes ("Pushed 0")
  if (!_publication_plan_id.is_valid()) { fail("publication plan ID is zero"); }
  {
    std::vector<dynamic_filter_target_id> ids;
    for (auto const& target : _scan_targets) {
      if (!target.target_id.is_valid() || !target.channel_id.is_valid()) {
        fail("target or channel ID is zero");
      }
      if (target.channel == nullptr) { fail("scan target has a null channel"); }
      ids.push_back(target.target_id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) { fail("duplicate target IDs"); }
  }
  // 11. replica spaces: reuse the existing constructor validation (GPU/HOST tier + unique
  //     devices) by construction below — dynamic_filter_publish_plan's constructor enforces it.
  // 12. disabled plans contain no live target but are still built and installed.

  std::vector<dynamic_filter_publish_plan::probe_target> targets;
  if (enabled) {
    targets.reserve(_scan_targets.size());
    for (auto const& draft : _scan_targets) {
      targets.push_back(dynamic_filter_publish_plan::probe_target{draft.channel,
                                                                  draft.probe_col_idx,
                                                                  draft.probe_col_type,
                                                                  draft.target_id,
                                                                  draft.channel_id});
    }
  }
  // Domain evidence is null in C1a-2: all-zero cardinalities keep the coverage gates exactly off,
  // byte-for-byte matching the dead pre-C1a-2 walk's runtime effect.
  return std::make_shared<dynamic_filter_publish_plan const>(
    _publication_plan_id,
    ordinal_records(),
    std::move(targets),
    _emit_zone_map_filters,
    std::vector<std::size_t>(_key_candidates.size(), 0),
    _replica_spaces,
    _domain_coverage_threshold);
}

//===----------------------------------------------------------------------===//
// The freeze seam
//
// How the pieces cooperate: the engine (src/sirius_engine.cpp) collects every hash
// join in the physical plan after pipelines are built, and calls
// freeze_or_verify_dynamic_filter_plans exactly once per execution — BEFORE any
// task can run, so no runtime code can ever observe a half-frozen topology.
//
//   * First execution: prepare_dynamic_filter_plans does all the work that can
//     fail (validating each builder, building each immutable plan, minting one
//     write-once-slot token per join). Only when every producer prepared
//     successfully does commit_dynamic_filter_plans publish them — a loop of
//     non-throwing moves. If preparation fails half-way, the already-minted
//     tokens roll their slots back on destruction: nothing changed.
//
//   * Re-execution of a cached plan: the slots are already filled and must not be
//     written twice. Instead we check that the topology the builders describe
//     still matches the topology that was frozen (it always should — a cached
//     plan cannot change shape without replanning) and reuse it.
//===----------------------------------------------------------------------===//

namespace {

/// The canonical plan for a join that is not a dynamic-filter producer (no builder):
/// valid, installable, publishes nothing.
std::shared_ptr<dynamic_filter_publish_plan const> make_disabled_plan()
{
  return std::make_shared<dynamic_filter_publish_plan const>();
}

/// One descriptor row from a builder (what a freeze of this join WOULD produce).
dynamic_filter_frozen_descriptor::producer_record describe_builder(
  dynamic_filter_publish_plan_builder const* builder)
{
  dynamic_filter_frozen_descriptor::producer_record record{};
  if (builder == nullptr) { return record; }  // non-producer join: all-default row
  // planning_view() throws for a builder whose keys were never resolved. That state cannot reach
  // the freeze (the hash-join constructor resolves keys immediately after construction), and if
  // it ever does, failing loudly here is the right outcome.
  auto const view            = builder->planning_view();
  record.publication_plan_id = builder->publication_plan_id();
  record.enabled             = view.enabled;
  for (auto const& ordinal : view.by_duckdb_ordinal) {
    record.decisions.push_back(static_cast<std::uint8_t>(ordinal.decision));
  }
  for (auto const& draft : builder->scan_targets()) {
    record.target_ids.push_back(draft.target_id);
    record.channel_ids.push_back(draft.channel_id);
  }
  return record;
}

/// One descriptor row from a frozen plan (what the freeze actually published).
dynamic_filter_frozen_descriptor::producer_record describe_frozen_plan(
  dynamic_filter_publish_plan const& plan)
{
  dynamic_filter_frozen_descriptor::producer_record record{};
  record.publication_plan_id = plan.publication_plan_id();
  record.enabled             = plan.enabled();
  for (auto const& ordinal : plan.ordinals()) {
    record.decisions.push_back(static_cast<std::uint8_t>(ordinal.decision));
  }
  for (auto const& target : plan.probe_targets()) {
    record.target_ids.push_back(target.target_id);
    record.channel_ids.push_back(target.channel_id);
  }
  return record;
}

}  // namespace

std::uint64_t dynamic_filter_frozen_descriptor::digest() const noexcept
{
  // FNV-1a over a canonical encoding of every record. Only a fast reject: equality of the full
  // descriptor is the real verification.
  std::uint64_t hash = 14695981039346656037ULL;
  auto const mix     = [&hash](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & 0xFFULL;
      hash *= 1099511628211ULL;
    }
  };
  mix(producers.size());
  for (auto const& record : producers) {
    mix(record.publication_plan_id.value);
    mix(record.enabled ? 1 : 0);
    mix(record.decisions.size());
    for (auto const decision : record.decisions) {
      mix(decision);
    }
    mix(record.target_ids.size());
    for (auto const& id : record.target_ids) {
      mix(id.value);
    }
    mix(record.channel_ids.size());
    for (auto const& id : record.channel_ids) {
      mix(id.value);
    }
  }
  return hash;
}

prepared_dynamic_filter_plans prepare_dynamic_filter_plans(
  std::span<sirius_physical_hash_join* const> producers,
  std::span<dynamic_filter_target_addition const> grouped_additions)
{
  // C3b will hand validated SIP targets through grouped_additions; until C1b defines the target
  // payload there is nothing legal to fold in.
  for (auto const& addition : grouped_additions) {
    if (!addition.targets.empty()) {
      throw sirius::internal_exception(
        "[prepare_dynamic_filter_plans] SIP target additions are not supported before C1b/C3");
    }
  }

  prepared_dynamic_filter_plans prepared;
  prepared._producers.reserve(producers.size());
  prepared._descriptor.producers.reserve(producers.size());

  for (auto* join : producers) {
    if (join == nullptr) {
      throw sirius::internal_exception("[prepare_dynamic_filter_plans] null producer");
    }
    if (join->has_frozen_dynamic_filter_plan()) {
      // Freezing twice is the verify path's job, and a half-frozen enumeration means two callers
      // disagree about this plan's state — both are engine bugs worth failing loudly on.
      throw sirius::internal_exception(
        "[prepare_dynamic_filter_plans] producer already has a frozen plan; cached re-execution "
        "must verify, not re-freeze");
    }
    auto const* builder = join->dynamic_filter_builder();
    auto plan           = builder != nullptr ? builder->finalize() : make_disabled_plan();
    prepared._descriptor.producers.push_back(describe_builder(builder));
    prepared._producers.push_back(prepared_dynamic_filter_plans::prepared_producer{
      join, join->prepare_dynamic_filter_plan_assignment(std::move(plan))});
  }
  return prepared;
}

void commit_dynamic_filter_plans(prepared_dynamic_filter_plans&& prepared) noexcept
{
  // Nothing here can fail: every fallible step already happened in prepare. This loop is just
  // non-throwing pointer moves, so a topology is either fully published or not published at all.
  for (auto& producer : prepared._producers) {
    producer.join->commit_dynamic_filter_plan_assignment(std::move(producer.token));
  }
  prepared._producers.clear();
}

dynamic_filter_frozen_descriptor describe_planned_dynamic_filter_topology(
  std::span<sirius_physical_hash_join* const> producers)
{
  dynamic_filter_frozen_descriptor descriptor;
  descriptor.producers.reserve(producers.size());
  for (auto const* join : producers) {
    descriptor.producers.push_back(describe_builder(join->dynamic_filter_builder()));
  }
  return descriptor;
}

dynamic_filter_frozen_descriptor describe_frozen_dynamic_filter_topology(
  std::span<sirius_physical_hash_join* const> producers)
{
  dynamic_filter_frozen_descriptor descriptor;
  descriptor.producers.reserve(producers.size());
  for (auto const* join : producers) {
    descriptor.producers.push_back(describe_frozen_plan(*join->dynamic_filter_plan()));
  }
  return descriptor;
}

void verify_frozen_dynamic_filter_topology(dynamic_filter_frozen_descriptor const& cached,
                                           dynamic_filter_frozen_descriptor const& current)
{
  // The digest is a cheap first look; only full value equality accepts.
  if (cached.digest() != current.digest() || !(cached == current)) {
    throw sirius::internal_exception(
      "[verify_frozen_dynamic_filter_topology] cached frozen dynamic-filter topology does not "
      "match the current plan (cached digest {:x}, current digest {:x}); a cached physical plan "
      "cannot change shape without replanning",
      cached.digest(),
      current.digest());
  }
}

void freeze_or_verify_dynamic_filter_plans(std::span<sirius_physical_hash_join* const> producers)
{
  std::size_t frozen_count = 0;
  for (auto const* join : producers) {
    if (join == nullptr) {
      throw sirius::internal_exception("[freeze_or_verify_dynamic_filter_plans] null producer");
    }
    if (join->has_frozen_dynamic_filter_plan()) { ++frozen_count; }
  }

  if (frozen_count == 0) {
    // First execution: the two-phase freeze. All fallible work first, then a commit that
    // cannot fail.
    commit_dynamic_filter_plans(prepare_dynamic_filter_plans(producers, {}));
    return;
  }
  if (frozen_count == producers.size()) {
    // Cached plan re-executed: never assign twice; check the frozen topology still matches what
    // the builders describe and reuse it.
    verify_frozen_dynamic_filter_topology(describe_frozen_dynamic_filter_topology(producers),
                                          describe_planned_dynamic_filter_topology(producers));
    return;
  }
  throw sirius::internal_exception(
    "[freeze_or_verify_dynamic_filter_plans] {} of {} producers are frozen; a plan must be "
    "entirely unfrozen (first execution) or entirely frozen (cached re-execution)",
    frozen_count,
    producers.size());
}

}  // namespace sirius::op
