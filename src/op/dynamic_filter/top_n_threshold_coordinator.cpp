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

#include <log/logging.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/dynamic_filter/top_n_boundary_filter.hpp>
#include <op/dynamic_filter/top_n_threshold_coordinator.hpp>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op {

top_n_threshold_coordinator::top_n_threshold_coordinator(std::size_t k,
                                                         std::vector<top_n_key_semantics> keys,
                                                         bool lex_admitted,
                                                         dynamic_filter_stats* stats,
                                                         top_n_producer_kind kind)
  : _k(k), _keys(std::move(keys)), _lex_admitted(lex_admitted), _stats(stats), _kind(kind)
{
}

threshold_offer_result top_n_threshold_coordinator::offer(top_n_distinct_key_witness witness)
{
  if (_kind != top_n_producer_kind::GROUP_KEY) {
    throw std::logic_error(
      "[top_n_threshold_coordinator] distinct-key offers require a GROUP_KEY coordinator");
  }
  bump(&dynamic_filter_stats::top_n_group_offers);

  std::unique_lock lock{_mu};
  if (_state != state::open) { return threshold_offer_result::REJECTED_STATE; }

  // Union by value into the bounded set: merge, drop duplicates, keep the K best. Sorting by the
  // ORDER BY semantics makes "best" and "duplicate" the same comparison the boundary uses.
  auto const better = [this](exact_host_key_tuple const& a, exact_host_key_tuple const& b) {
    return a.lex_compare(b, _keys) < 0;
  };
  auto const equal = [this](exact_host_key_tuple const& a, exact_host_key_tuple const& b) {
    return a.lex_compare(b, _keys) == 0;
  };
  for (auto& candidate : witness.best_keys) {
    if (candidate.size() != _keys.size()) { continue; }
    auto const at =
      std::lower_bound(_distinct_keys.begin(), _distinct_keys.end(), candidate, better);
    if (at != _distinct_keys.end() && equal(*at, candidate)) { continue; }  // already witnessed
    _distinct_keys.insert(at, std::move(candidate));
    if (_distinct_keys.size() > _k) {
      _distinct_keys.erase(_distinct_keys.begin() + static_cast<std::ptrdiff_t>(_k),
                           _distinct_keys.end());
    }
  }

  // The boundary exists only once K distinct groups are proven; it is then the Kth best.
  if (_distinct_keys.size() < _k) { return threshold_offer_result::NO_ACCEPTING_TARGET; }
  if (!_witness_set_full) {
    _witness_set_full = true;
    bump(&dynamic_filter_stats::top_n_group_witness_set_full);
  }

  auto candidate = _distinct_keys.back();
  // A null first component publishes nothing (main doc, "Predicate layers and null derivations"):
  // neither layer can express it, and every published RANGE reads component 0 unconditionally. The
  // null-keyed group stays in the witness set -- group-by treats null as a group, so it is a real
  // witnessed value -- and a later, better key can still make the Kth element publishable.
  if (!candidate.component(0).has_value()) {
    bump(&dynamic_filter_stats::top_n_offers_unsupported);
    return threshold_offer_result::UNSUPPORTED_BOUNDARY;
  }
  if (_tightest_seen && candidate.lex_compare(*_tightest_seen, _keys) >= 0) {
    bump(&dynamic_filter_stats::top_n_offers_not_tighter);
    return threshold_offer_result::NOT_TIGHTER;
  }
  _tightest_seen = candidate;
  _boundary_updates.fetch_add(1, std::memory_order_relaxed);

  if (_plan.targets.empty()) { return threshold_offer_result::NO_ACCEPTING_TARGET; }

  _pending = std::move(candidate);
  if (_publisher_active) { return threshold_offer_result::COALESCED; }
  _publisher_active = true;
  lock.unlock();

  publisher_loop();
  return threshold_offer_result::ACCEPTED_FOR_PUBLICATION;
}

void top_n_threshold_coordinator::bump(
  std::atomic<std::uint64_t> dynamic_filter_stats::* counter) const noexcept
{
  if (_stats) { (_stats->*counter).fetch_add(1, std::memory_order_relaxed); }
}

threshold_offer_result top_n_threshold_coordinator::offer(top_n_threshold_witness witness)
{
  bump(&dynamic_filter_stats::top_n_offers);

  if (witness.boundary.size() != _keys.size() || !witness.boundary.component(0).has_value()) {
    bump(&dynamic_filter_stats::top_n_offers_unsupported);
    return threshold_offer_result::UNSUPPORTED_BOUNDARY;
  }

  std::unique_lock lock{_mu};
  if (_state != state::open) { return threshold_offer_result::REJECTED_STATE; }
  if (_tightest_seen && witness.boundary.lex_compare(*_tightest_seen, _keys) >= 0) {
    bump(&dynamic_filter_stats::top_n_offers_not_tighter);
    return threshold_offer_result::NOT_TIGHTER;
  }
  _tightest_seen = witness.boundary;
  _boundary_updates.fetch_add(1, std::memory_order_relaxed);

  // A producer whose discovery bound nothing tightens the shared boundary for the sink prefilter
  // and stops here -- no pending candidate, no publisher loop.
  if (_plan.targets.empty()) { return threshold_offer_result::NO_ACCEPTING_TARGET; }

  _pending = std::move(witness.boundary);
  if (_publisher_active) { return threshold_offer_result::COALESCED; }
  _publisher_active = true;
  lock.unlock();

  publisher_loop();
  return threshold_offer_result::ACCEPTED_FOR_PUBLICATION;
}

void top_n_threshold_coordinator::set_publish_plan(top_n_dynamic_filter_publish_plan plan)
{
  std::scoped_lock lock{_mu};
  // Plan-time only. Enforced rather than merely documented, because `publish_revision` reads the
  // plan without the lock: that is safe exactly while the plan is immutable once any offer can
  // occur, and this check is what makes it so.
  if (_state != state::open || _boundary_updates.load(std::memory_order_relaxed) != 0) {
    throw std::logic_error(
      "[top_n_threshold_coordinator] the publication plan is frozen at plan time and cannot be "
      "replaced once offers have begun");
  }
  for (auto const& target : plan.targets) {
    if (target.layer == top_n_filter_layer::LEX &&
        target.component_ordinals.size() != _keys.size()) {
      throw std::logic_error(
        "[top_n_threshold_coordinator] a LEX target must declare one component ordinal per "
        "ORDER BY key");
    }
    if (target.component_ordinals.empty()) {
      throw std::logic_error(
        "[top_n_threshold_coordinator] every target must declare its primary ordinal");
    }
  }
  _plan = std::move(plan);
  _target_closed.assign(_plan.targets.size(), false);
}

void top_n_threshold_coordinator::publisher_loop()
{
  while (true) {
    std::optional<exact_host_key_tuple> candidate;
    std::uint64_t revision = 0;
    {
      std::scoped_lock lock{_mu};
      if (!_pending) {
        // The pending-empty check and the handoff share this critical section, so an offer that
        // arrives after it sees no active publisher and takes ownership itself.
        _publisher_active = false;
        _publisher_idle.notify_all();
        return;
      }
      candidate = std::move(_pending);
      _pending.reset();
      revision = _next_revision++;
    }
    publish_revision(*candidate, revision);
  }
}

std::optional<exact_host_key_tuple> top_n_threshold_coordinator::tightest_boundary() const
{
  std::scoped_lock lock{_mu};
  return _tightest_seen;
}

std::size_t top_n_threshold_coordinator::boundary_update_count() const noexcept
{
  return _boundary_updates.load(std::memory_order_relaxed);
}

void top_n_threshold_coordinator::record_prefilter(std::size_t rows_in,
                                                   std::size_t rows_out) noexcept
{
  if (!_stats) { return; }
  if (_kind == top_n_producer_kind::GROUP_KEY) {
    _stats->top_n_group_prefilter_rows_in.fetch_add(rows_in, std::memory_order_relaxed);
    _stats->top_n_group_prefilter_rows_out.fetch_add(rows_out, std::memory_order_relaxed);
    return;
  }
  _stats->top_n_prefilter_rows_in.fetch_add(rows_in, std::memory_order_relaxed);
  _stats->top_n_prefilter_rows_out.fetch_add(rows_out, std::memory_order_relaxed);
}

void top_n_threshold_coordinator::record_prefilter_disabled() noexcept
{
  // Deliberately not split by kind, unlike `record_prefilter`: "a sink prefilter stopped paying
  // for itself" is one lifecycle event whichever discipline owns the sink, and the counter block
  // defines no group-key equivalent.
  bump(&dynamic_filter_stats::top_n_prefilter_disabled);
}

void top_n_threshold_coordinator::publish_revision(exact_host_key_tuple const& boundary,
                                                   std::uint64_t revision)
{
  // Once every consumer has drained there is nothing left to publish into, and the sink outlives
  // its scans, so late offers are normal. Skip construction entirely rather than allocating
  // device scalars for a revision that can only be rejected (main doc failure table, "All
  // channels closed"). The boundary still tightens for the sink prefilter.
  if (!_target_closed.empty() &&
      std::all_of(
        _target_closed.begin(), _target_closed.end(), [](bool closed) { return closed; })) {
    return;
  }

  // Build every planned layer and replicate every component scalar to every planned device before
  // installing anything: a partially replicated revision would be missing on some consumer GPU.
  // Construction and replication are therefore all-or-nothing.
  std::vector<std::pair<std::size_t, std::shared_ptr<sirius_dynamic_filter const>>> ready;
  try {
    ready.reserve(_plan.targets.size());
    // RANGE carries no ordinal, so one object genuinely fans out to every first-key target -- the
    // join path's channel_push_ordinal precedent. LEX owns its component ordinals, and each
    // target's trace remaps them, so one filter is built per accepting LEX target (R7).
    std::shared_ptr<sirius_dynamic_filter const> first_key_filter;
    for (std::size_t i = 0; i < _plan.targets.size(); ++i) {
      auto const& target = _plan.targets[i];
      if (target.layer == top_n_filter_layer::LEX) {
        std::vector<lex_component_semantics> components;
        components.reserve(_keys.size());
        for (std::size_t c = 0; c < _keys.size(); ++c) {
          components.push_back({.consumer_ordinal = target.component_ordinals[c], .key = _keys[c]});
        }
        // Strict for a row producer (full-tuple peers are interchangeable once K witnesses
        // exist); inclusive for a group-key producer, whose boundary is the Kth-best grouping key
        // and whose ties therefore belong to a group that is in the answer.
        ready.emplace_back(i,
                           std::make_shared<sirius_dynamic_lex_range_filter const>(
                             boundary, std::move(components), lex_inclusive()));
      } else {
        if (!first_key_filter) {
          // The first-key layer bounds key zero: k0 <= b0 ascending, k0 >= b0 descending.
          // Strictness is a per-kind policy -- see first_key_inclusive(). Nulls that sort before
          // every value are still better than the boundary and must be admitted; otherwise they
          // are worse and are rejected.
          auto const& key0  = _keys.front();
          auto const side   = key0.order == cudf::order::DESCENDING ? range_bound_side::LOWER
                                                                    : range_bound_side::UPPER;
          auto const policy = detail::nulls_first_in_output(key0)
                                ? dynamic_filter_null_policy::ADMIT
                                : dynamic_filter_null_policy::REJECT;
          first_key_filter  = std::make_shared<sirius_dynamic_range_filter const>(
            *boundary.component(0), side, first_key_inclusive(), policy);
        }
        ready.emplace_back(i, first_key_filter);
      }
    }
    for (auto const& [index, filter] : ready) {
      // const_cast is confined here: replication is the one producer-side mutation before the
      // filter becomes visible, and every consumer only ever sees the immutable handle.
      auto* replicable =
        dynamic_cast<sirius_device_replicable*>(const_cast<sirius_dynamic_filter*>(filter.get()));
      if (replicable != nullptr) { replicable->replicate_to_devices(_plan.replica_spaces); }
    }
  } catch (std::exception const& e) {
    // Any construction or replica failure installs nothing; the previous revision stays visible
    // on every device and every target.
    bump(&dynamic_filter_stats::top_n_revisions_failed);
    SIRIUS_LOG_WARN(
      "[top_n_threshold_coordinator] revision {} not installed: {}. Retaining the previous "
      "revision on every target.",
      revision,
      e.what());
    return;
  }

  // Closure is a per-target condition, not a failure: a drained consumer returns CLOSED and is
  // skipped while every still-open target receives this revision.
  bool installed_any = false;
  for (auto& [index, filter] : ready) {
    auto& target      = _plan.targets[index];
    auto const result = target.publisher.publish(revision, filter);
    switch (result) {
      case refinement_publish_result::ACCEPTED:
        installed_any = true;
        bump(target.layer == top_n_filter_layer::LEX
               ? &dynamic_filter_stats::top_n_lex_filters_pushed
               : &dynamic_filter_stats::top_n_first_key_filters_pushed);
        break;
      case refinement_publish_result::STALE:
        bump(&dynamic_filter_stats::top_n_revisions_stale);
        break;
      case refinement_publish_result::CLOSED:
        // Latch it: a drained consumer never reopens, so later revisions skip this target and,
        // once every target is closed, skip construction altogether.
        _target_closed[index] = true;
        break;
      case refinement_publish_result::IGNORED:
        // A slot whose primary or referenced ordinal is ignored (hive partition) can never
        // publish; count it so it is not invisible in the delivery counters.
        bump(&dynamic_filter_stats::top_n_revisions_ignored);
        break;
    }
  }
  if (installed_any) { bump(&dynamic_filter_stats::top_n_revisions_published); }
}

void top_n_threshold_coordinator::finish()
{
  bool caller_becomes_publisher = false;
  {
    std::scoped_lock lock{_mu};
    if (_state == state::finished || _state == state::cancelled) { return; }
    _state = state::finishing;  // later offers are rejected from here on
    if (_pending && !_publisher_active) {
      _publisher_active        = true;
      caller_becomes_publisher = true;
    }
  }

  // Producer-side drain: this blocks the merge/finalization task only, never a scan consumer.
  // Under pipeline fusion the merge finalizes at parent-pipeline finish, so the drain runs in
  // that finish path.
  if (caller_becomes_publisher) { publisher_loop(); }

  std::unique_lock lock{_mu};
  _publisher_idle.wait(lock, [this] { return !_publisher_active && !_pending; });
  // A cancellation racing this drain stays cancelled: both reject offers, but collapsing the two
  // would lose the distinction for anything that later inspects the terminal state.
  if (_state != state::cancelled) { _state = state::finished; }
}

void top_n_threshold_coordinator::cancel() noexcept
{
  std::scoped_lock lock{_mu};
  if (_state == state::finished) { return; }
  _state = state::cancelled;
  // Abandon work that has not started; an in-flight publisher finishes its current revision and
  // then observes the empty pending slot and goes quiescent.
  _pending.reset();
  _publisher_idle.notify_all();
}

}  // namespace sirius::op
