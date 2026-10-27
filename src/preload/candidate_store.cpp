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

#include "preload/candidate_store.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>

namespace sirius::preload {

namespace {

using cucascade::memory::Tier;

constexpr std::uint64_t no_epoch = ~std::uint64_t{0};

/// Whether epoch @p a is later than epoch @p b, robust to wrap-around.
[[nodiscard]] constexpr bool epoch_after(std::uint64_t a, std::uint64_t b) noexcept
{
  return static_cast<std::int64_t>(a - b) > 0;
}

[[nodiscard]] bool lower_tier(Tier tier) noexcept
{
  return tier == Tier::HOST || tier == Tier::DISK;
}

/// Counter of a worker outcome other than @c promoted.
[[nodiscard]] counter outcome_counter(promotion_outcome o) noexcept
{
  switch (o) {
    case promotion_outcome::retired: return counter::retired;
    case promotion_outcome::expired: return counter::expired;
    case promotion_outcome::contended: return counter::contended;
    case promotion_outcome::stale: return counter::stale;
    case promotion_outcome::already_resident: return counter::already_resident;
    case promotion_outcome::unsupported: return counter::unsupported;
    case promotion_outcome::too_big: return counter::too_big;
    case promotion_outcome::no_memory: return counter::no_memory;
    case promotion_outcome::gated: return counter::gated;
    case promotion_outcome::held_off: return counter::held_off;
    case promotion_outcome::oom: return counter::oom;
    case promotion_outcome::promoted:
    case promotion_outcome::failed: break;
  }
  return counter::failed;
}

/// Promotion and byte counters of a promotion kind.
[[nodiscard]] std::pair<counter, counter> kind_counters(promotion_kind kind) noexcept
{
  switch (kind) {
    case promotion_kind::h2g: return {counter::promoted_h2g, counter::bytes_h2g};
    case promotion_kind::d2g: return {counter::promoted_d2g, counter::bytes_d2g};
    case promotion_kind::d2h: return {counter::promoted_d2h, counter::bytes_d2h};
    case promotion_kind::relocate_d2h:
      return {counter::promoted_relocate_d2h, counter::bytes_relocate_d2h};
    case promotion_kind::none: break;
  }
  return {counter::promoted_h2g, counter::bytes_h2g};
}

/// The kind of a promotion whose result did not say, from its source and target tiers.
[[nodiscard]] promotion_kind infer_kind(promotion_job const& job) noexcept
{
  auto const from_disk = job.expected_source.tier == Tier::DISK;
  if (job.t.tier == Tier::GPU) { return from_disk ? promotion_kind::d2g : promotion_kind::h2g; }
  return promotion_kind::d2h;
}

[[nodiscard]] std::optional<std::size_t> index_of(std::vector<int> const& devices,
                                                  int device) noexcept
{
  auto const it = std::find(devices.begin(), devices.end(), device);
  if (it == devices.end()) { return std::nullopt; }
  return static_cast<std::size_t>(it - devices.begin());
}

}  // namespace

candidate_store::candidate_store(preload_executor_config cfg,
                                 std::vector<int> gpu_devices,
                                 std::vector<int> host_devices,
                                 preload_counters& counters)
  : _cfg(std::move(cfg)),
    _gpu_devices(std::move(gpu_devices)),
    _host_devices(std::move(host_devices)),
    _counters(counters),
    _mem_epoch(_gpu_devices.size() + _host_devices.size(), 0),
    _gate_epoch(_gpu_devices.size() + _host_devices.size(), no_epoch),
    _inflight(_gpu_devices.size() + _host_devices.size(), 0),
    _h2g_tokens(_gpu_devices.size(), 0),
    _outstanding(_gpu_devices.size(), 0),
    _outstanding_b(_gpu_devices.size(), 0)
{
}

// ---- spaces ---------------------------------------------------------------------------------

std::optional<std::size_t> candidate_store::gpu_index_of(int device) const noexcept
{
  return index_of(_gpu_devices, device);
}

std::optional<std::size_t> candidate_store::host_index_of(int device) const noexcept
{
  return index_of(_host_devices, device);
}

std::optional<std::size_t> candidate_store::slot(target t) const noexcept
{
  if (t.tier == Tier::GPU && t.index < _gpu_devices.size()) { return t.index; }
  if (t.tier == Tier::HOST && t.index < _host_devices.size()) {
    return _gpu_devices.size() + t.index;
  }
  return std::nullopt;
}

// ---- epochs ---------------------------------------------------------------------------------

std::uint64_t candidate_store::mem_epoch(target t) const noexcept
{
  auto const s = slot(t);
  return s ? _mem_epoch[*s] : 0;
}

std::uint64_t candidate_store::gate_epoch(target t) const noexcept
{
  auto const s = slot(t);
  return s ? _gate_epoch[*s] : no_epoch;
}

bool candidate_store::gated(target t) const noexcept
{
  auto const s = slot(t);
  return s && _gate_epoch[*s] == _mem_epoch[*s];
}

void candidate_store::advance_mem_epoch(target t) noexcept
{
  if (auto const s = slot(t)) { ++_mem_epoch[*s]; }
}

void candidate_store::advance_host_mem_epochs() noexcept
{
  for (std::size_t i = _gpu_devices.size(); i < _mem_epoch.size(); ++i) {
    ++_mem_epoch[i];
  }
}

void candidate_store::advance_all_mem_epochs() noexcept
{
  for (auto& e : _mem_epoch) {
    ++e;
  }
}

// ---- in-flight accounting -------------------------------------------------------------------

std::size_t candidate_store::inflight_bytes(target t) const noexcept
{
  auto const s = slot(t);
  return s ? _inflight[*s] : 0;
}

void candidate_store::add_inflight(target t, std::size_t bytes) noexcept
{
  if (auto const s = slot(t)) { _inflight[*s] = sat_add(_inflight[*s], bytes); }
}

void candidate_store::remove_inflight(target t, std::size_t bytes) noexcept
{
  if (auto const s = slot(t)) { _inflight[*s] = sat_sub(_inflight[*s], bytes); }
}

std::uint32_t candidate_store::h2g_tokens(std::size_t gpu) const noexcept
{
  return gpu < _h2g_tokens.size() ? _h2g_tokens[gpu] : 0;
}

void candidate_store::acquire_h2g_token(std::size_t gpu) noexcept
{
  if (gpu < _h2g_tokens.size()) { ++_h2g_tokens[gpu]; }
}

void candidate_store::release_h2g_token(std::size_t gpu) noexcept
{
  if (gpu < _h2g_tokens.size() && _h2g_tokens[gpu] > 0) { --_h2g_tokens[gpu]; }
}

// ---- statistics -----------------------------------------------------------------------------

void candidate_store::count(query_id_t q, counter c, std::uint64_t n) noexcept
{
  _counters.add(c, n);
  if (auto const it = _queries.find(q); it != _queries.end()) {
    it->second.stats.add(c, n);
  } else if (auto const rit = _stats_by_query.find(q); rit != _stats_by_query.end()) {
    rit->second.add(c, n);
  }
}

void candidate_store::count_outcome(promotion_job const& job,
                                    promotion_result const& result) noexcept
{
  if (result.outcome != promotion_outcome::promoted) {
    count(job.query, outcome_counter(result.outcome));
    return;
  }
  auto const kind = result.kind != promotion_kind::none ? result.kind : infer_kind(job);
  auto const [promoted, bytes] = kind_counters(kind);
  count(job.query, promoted);
  count(job.query, bytes, result.bytes);
}

// ---- queries --------------------------------------------------------------------------------

bool candidate_store::add_query(query_id_t q,
                                std::vector<int> active_gpus,
                                std::vector<repo_info> repos)
{
  if (_queries.contains(q)) { return false; }
  query_state qs;
  qs.active_gpus = std::move(active_gpus);
  // Deduplicate in input order first, so the first entry of a token wins whatever its priority.
  qs.repos.reserve(repos.size());
  for (auto& r : repos) {
    if (qs.repo_index.emplace(r.token, 0).second) { qs.repos.push_back(std::move(r)); }
  }
  std::stable_sort(qs.repos.begin(), qs.repos.end(), [](repo_info const& a, repo_info const& b) {
    return a.dest_priority < b.dest_priority;
  });
  for (std::size_t i = 0; i < qs.repos.size(); ++i) {
    qs.repo_index[qs.repos[i].token] = i;
  }
  _queries.emplace(q, std::move(qs));
  return true;
}

query_state const* candidate_store::find_query(query_id_t q) const noexcept
{
  auto const it = _queries.find(q);
  return it == _queries.end() ? nullptr : &it->second;
}

void candidate_store::mark_readiness_dirty(query_id_t q) noexcept
{
  if (auto const it = _queries.find(q); it != _queries.end()) { it->second.readiness_dirty = true; }
}

void candidate_store::retire_query(query_id_t q) noexcept
{
  // Candidates and ledger entries exist only for known queries, and each is indexed in its
  // query's state, so an unknown query has nothing to purge and a known one needs no full scan.
  auto const it = _queries.find(q);
  if (it == _queries.end()) { return; }
  auto& qs = it->second;
  // Erasing a candidate or a ledger entry also drops its index entry; an index entry that no longer
  // resolves to one of q is dropped directly, so both loops always make progress.
  while (!qs.by_seq.empty()) {
    auto const first = qs.by_seq.begin();
    auto const cit   = _by_id.find(first->second);
    if (cit != _by_id.end() && cit->second.query == q) {
      erase_candidate(cit);
    } else {
      qs.by_seq.erase(first);
    }
  }
  while (!qs.ledger_ids.empty()) {
    auto const first = qs.ledger_ids.begin();
    auto const lit   = _ledger.find(*first);
    if (lit != _ledger.end() && lit->second.query == q) {
      count(q, counter::wasted_unconsumed);
      erase_ledger(lit);
    } else {
      qs.ledger_ids.erase(first);
    }
  }
  try {
    _stats_by_query.insert_or_assign(q, it->second.stats);
  } catch (...) {
    // Losing a retired query's summary is acceptable; its state must still go.
  }
  _queries.erase(it);
}

std::optional<per_query_stats> candidate_store::forget_query(query_id_t q) noexcept
{
  // A query the store never knew (or already forgot) freed nothing preload could have been
  // refused by: leave the epochs alone rather than ungating every space.
  if (!_queries.contains(q) && !_stats_by_query.contains(q)) { return std::nullopt; }
  retire_query(q);
  advance_all_mem_epochs();
  auto const it = _stats_by_query.find(q);
  if (it == _stats_by_query.end()) { return std::nullopt; }
  auto stats = it->second;
  _stats_by_query.erase(it);
  return stats;
}

// ---- intake ---------------------------------------------------------------------------------

intake_status candidate_store::intake(query_id_t q, event::downgraded_batch const& rec)
{
  _counters.add(counter::events_received);
  auto const qit = _queries.find(q);
  if (qit == _queries.end()) {
    _counters.add(counter::dropped_unknown_query);
    return intake_status::dropped_unknown_query;
  }
  auto& qs      = qit->second;
  auto const id = rec.batch_id;

  // A batch still in the ledger was moved down again before its consumer took it: whatever
  // happens to the record, the ledger no longer describes where the batch is. A record of a move
  // from before the promotion (delivered late, or out of order across reporters) says nothing
  // about where the batch is now.
  bool bounced = false;
  if (auto const lit = _ledger.find(id); lit != _ledger.end()) {
    if (rec.conversion_seq <= lit->second.seq) {
      count(q, counter::dropped_stale_record);
      return intake_status::dropped_stale_record;
    }
    erase_ledger(lit);
    count(q, counter::bounce);
    bounced = true;
  }

  auto const pit        = qs.promotions.find(id);
  auto const promotions = pit == qs.promotions.end() ? 0U : pit->second;
  if (promotions >= static_cast<std::uint32_t>(std::max(_cfg.max_promotions_per_batch, 0))) {
    if (auto const cit = _by_id.find(id); cit != _by_id.end() && !cit->second.in_flight) {
      erase_candidate(cit);
    }
    count(q, counter::dropped_promotion_cap);
    return intake_status::dropped_promotion_cap;
  }

  if (auto const cit = _by_id.find(id); cit != _by_id.end()) {
    merge(cit->second, qs, rec);
    if (bounced) { cit->second.speculative_banned = true; }
    count(q, counter::merged);
    return intake_status::merged;
  }

  if (rec.batch.expired()) {
    count(q, counter::dropped_expired);
    return intake_status::dropped_expired;
  }
  if (!lower_tier(rec.to.tier)) {
    count(q, counter::dropped_unsupported);
    return intake_status::dropped_unsupported;
  }

  candidate c;
  c.batch              = rec.batch;
  c.batch_id           = id;
  c.query              = q;
  c.last_seq           = rec.conversion_seq;
  c.location           = rec.to;
  c.est_bytes          = rec.bytes;
  c.source             = rec.source;
  c.repo_token         = rec.repository;
  c.partition          = rec.partition;
  c.task_priority      = rec.task_priority;
  c.preferred_device   = rec.preferred_device;
  c.speculative_banned = bounced;
  if (rec.from.tier == Tier::GPU) { c.origin_gpu = rec.from.device_id; }
  if (rec.from.tier == Tier::HOST) {
    c.origin_host = rec.from.device_id;
  } else if (rec.to.tier == Tier::HOST) {
    c.origin_host = rec.to.device_id;
  }

  auto const [it, inserted] = _by_id.emplace(id, std::move(c));
  try {
    qs.by_seq.emplace(rec.conversion_seq, id);
    if (rec.source == event::downgrade_source::repository) {
      qs.candidates_by_repo[rec.repository].insert(id);
    }
  } catch (...) {
    erase_candidate(it);
    throw;
  }

  if (_cfg.max_candidates > 0 && _by_id.size() > _cfg.max_candidates) {
    if (auto const victim = evict_one(); victim && *victim == id) {
      return intake_status::dropped_overflow;
    }
  }
  return bounced ? intake_status::bounced : intake_status::inserted;
}

void candidate_store::merge(candidate& c, query_state& qs, event::downgraded_batch const& rec)
{
  if (rec.conversion_seq > c.last_seq) {
    // Index first: if it throws, the candidate is unchanged.
    qs.by_seq.emplace(rec.conversion_seq, c.batch_id);
    bool const was_filed = c.source == event::downgrade_source::repository;
    bool const filed     = rec.source == event::downgrade_source::repository;
    if (filed && (!was_filed || rec.repository != c.repo_token)) {
      try {
        qs.candidates_by_repo[rec.repository].insert(c.batch_id);
      } catch (...) {
        qs.by_seq.erase({rec.conversion_seq, c.batch_id});
        throw;
      }
    }
    if (was_filed && (!filed || rec.repository != c.repo_token)) {
      if (auto const rit = qs.candidates_by_repo.find(c.repo_token);
          rit != qs.candidates_by_repo.end()) {
        rit->second.erase(c.batch_id);
        if (rit->second.empty()) { qs.candidates_by_repo.erase(rit); }
      }
    }
    qs.by_seq.erase({c.last_seq, c.batch_id});
    c.last_seq   = rec.conversion_seq;
    c.location   = rec.to;
    c.source     = rec.source;
    c.repo_token = rec.repository;
    c.partition  = rec.partition;
  }
  c.est_bytes = std::max(c.est_bytes, rec.bytes);
  if (!c.origin_gpu && rec.from.tier == Tier::GPU) { c.origin_gpu = rec.from.device_id; }
  if (!c.origin_host) {
    if (rec.from.tier == Tier::HOST) {
      c.origin_host = rec.from.device_id;
    } else if (rec.to.tier == Tier::HOST) {
      c.origin_host = rec.to.device_id;
    }
  }
  if (!c.task_priority && rec.task_priority) { c.task_priority = rec.task_priority; }
  if (!c.preferred_device && rec.preferred_device) { c.preferred_device = rec.preferred_device; }
}

std::optional<std::uint64_t> candidate_store::evict_one() noexcept
{
  for (auto qit = _queries.rbegin(); qit != _queries.rend(); ++qit) {
    auto& seqs = qit->second.by_seq;
    for (auto sit = seqs.rbegin(); sit != seqs.rend(); ++sit) {
      auto const id = sit->second;
      auto const it = _by_id.find(id);
      if (it == _by_id.end() || it->second.in_flight) { continue; }
      count(qit->first, counter::dropped_overflow);
      erase_candidate(it);
      return id;
    }
  }
  return std::nullopt;
}

// ---- candidates -----------------------------------------------------------------------------

candidate const* candidate_store::find(std::uint64_t batch_id) const noexcept
{
  auto const it = _by_id.find(batch_id);
  return it == _by_id.end() ? nullptr : &it->second;
}

bool candidate_store::plannable(candidate const& c) const noexcept
{
  return !c.in_flight && !c.unsupported && c.blocked_queue_epoch != _queue_epoch;
}

void candidate_store::erase_candidate(candidate_map::iterator it) noexcept
{
  auto const& c = it->second;
  if (auto const qit = _queries.find(c.query); qit != _queries.end()) {
    auto& qs = qit->second;
    qs.by_seq.erase({c.last_seq, c.batch_id});
    if (c.source == event::downgrade_source::repository) {
      if (auto const rit = qs.candidates_by_repo.find(c.repo_token);
          rit != qs.candidates_by_repo.end()) {
        rit->second.erase(c.batch_id);
        if (rit->second.empty()) { qs.candidates_by_repo.erase(rit); }
      }
    }
  }
  _by_id.erase(it);
}

void candidate_store::drop_candidate_noexcept(std::uint64_t batch_id) noexcept
{
  if (auto const it = _by_id.find(batch_id); it != _by_id.end()) { erase_candidate(it); }
}

std::size_t candidate_store::prune_expired_candidates(std::size_t budget) noexcept
{
  budget              = std::min(budget, _by_id.size());
  std::size_t visited = 0;
  std::size_t erased  = 0;
  // Resume after the cursor, then wrap around to the start once.
  for (int sweep = 0; sweep < 2 && visited < budget; ++sweep) {
    auto qit = sweep == 0 ? _queries.lower_bound(_prune_query) : _queries.begin();
    for (; qit != _queries.end() && visited < budget; ++qit) {
      auto& seqs = qit->second.by_seq;
      auto sit =
        sweep == 0 && qit->first == _prune_query ? seqs.upper_bound(_prune_key) : seqs.begin();
      while (sit != seqs.end() && visited < budget) {
        auto const key = *sit++;
        ++visited;
        _prune_query   = qit->first;
        _prune_key     = key;
        auto const cit = _by_id.find(key.second);
        if (cit != _by_id.end() && !cit->second.in_flight && cit->second.batch.expired()) {
          count(qit->first, counter::dropped_expired);
          erase_candidate(cit);
          ++erased;
        }
      }
    }
  }
  return erased;
}

// ---- planning and jobs ----------------------------------------------------------------------

pass_snapshot candidate_store::snapshot_phase0()
{
  pass_snapshot s;
  s.has_candidates = !_by_id.empty();
  s.queue_epoch    = _queue_epoch;
  s.mem_epoch      = _mem_epoch;
  s.gate_epoch     = _gate_epoch;
  s.inflight_bytes = _inflight;
  s.outstanding    = _outstanding;
  s.outstanding_b  = _outstanding_b;
  s.queries.reserve(_queries.size());
  for (auto const& [q, qs] : _queries) {
    query_view v;
    v.query           = q;
    v.active_gpus     = qs.active_gpus;
    v.readiness_dirty = qs.readiness_dirty;
    for (auto const& r : qs.repos) {
      auto const cit = qs.candidates_by_repo.find(r.token);
      if (cit == qs.candidates_by_repo.end() || cit->second.empty()) { continue; }
      repo_view rv;
      rv.info = r;
      rv.candidate_ids.assign(cit->second.begin(), cit->second.end());
      std::sort(rv.candidate_ids.begin(), rv.candidate_ids.end());
      v.repos.push_back(std::move(rv));
    }
    s.queries.push_back(std::move(v));
  }
  s.ledger.reserve(_ledger.size());
  for (auto const& [id, e] : _ledger) {
    if (!e.batch.expired()) { s.ledger.push_back({id, e.query, e.batch, e.k}); }
  }
  for (auto& [q, qs] : _queries) {
    qs.readiness_dirty = false;
  }
  return s;
}

std::optional<promotion_job> candidate_store::take(ready_item const& item)
{
  auto const s = slot(item.t);
  if (!s) { return std::nullopt; }
  auto const it = _by_id.find(item.batch_id);
  if (it == _by_id.end()) { return std::nullopt; }
  auto& c = it->second;
  if (!plannable(c) || !_queries.contains(c.query)) { return std::nullopt; }

  promotion_job job;
  job.batch               = c.batch;
  job.batch_id            = c.batch_id;
  job.query               = c.query;
  job.expected_source     = c.location;
  job.t                   = item.t;
  job.k                   = item.k;
  job.oom_multiplier      = item.t.tier == Tier::GPU ? oom_multiplier(c.oom_count) : 1;
  job.est_reserve         = reserve_estimate(c.est_bytes, job.oom_multiplier);
  job.ceiling             = item.ceiling;
  job.mem_epoch_at_take   = _mem_epoch[*s];
  job.queue_epoch_at_take = _queue_epoch;
  job.seq_at_take         = c.last_seq;
  c.in_flight             = true;
  return job;
}

outcome_effect candidate_store::apply_outcome(promotion_job const& job,
                                              promotion_result const& result)
{
  outcome_effect effect;
  count_outcome(job, result);

  // Space-level effects hold whatever happened to the query.
  auto const o = result.outcome;
  if (o == promotion_outcome::no_memory || o == promotion_outcome::gated) {
    // Outcomes of concurrent jobs can be applied out of order: a gate only moves forward, so an
    // older refusal never reopens a space a newer one closed.
    if (auto const s = slot(job.t)) {
      auto& gate = _gate_epoch[*s];
      if (gate == no_epoch || epoch_after(job.mem_epoch_at_take, gate)) {
        gate = job.mem_epoch_at_take;
      }
    }
    effect.purge = purge_scope::target;
  } else if (o == promotion_outcome::held_off) {
    effect.purge = purge_scope::held_off;
  } else if (o == promotion_outcome::promoted && job.expected_source.tier == Tier::HOST) {
    if (auto const h = host_index_of(job.expected_source.device_id)) {
      advance_mem_epoch({Tier::HOST, *h});
    }
  }

  auto const qit = _queries.find(job.query);
  auto const it  = _by_id.find(job.batch_id);
  if (qit == _queries.end()) {
    count(job.query, counter::late_outcome_after_retire);
    effect.late = true;
    return effect;
  }
  if (it == _by_id.end() || it->second.query != job.query) { return effect; }

  auto& qs    = qit->second;
  auto& c     = it->second;
  c.in_flight = false;

  // A lost race on the batch: wait for the queue to move past the dispatch epoch, and give up
  // after max_lock_attempts.
  auto const lost_race = [&] {
    ++c.lock_misses;
    if (c.lock_misses >= static_cast<std::uint32_t>(std::max(_cfg.max_lock_attempts, 1))) {
      count(job.query, counter::dropped_lock_attempts);
      erase_candidate(it);
    } else {
      c.blocked_queue_epoch = job.queue_epoch_at_take;
    }
  };

  switch (o) {
    case promotion_outcome::promoted: {
      auto& promotions = qs.promotions[c.batch_id];
      promotions =
        promotions == std::numeric_limits<std::uint32_t>::max() ? promotions : promotions + 1;
      auto const cap = static_cast<std::uint32_t>(std::max(_cfg.max_promotions_per_batch, 0));
      if (c.last_seq > job.seq_at_take) {
        // Downgraded again while the copy ran: the merge already moved the candidate.
        count(job.query, counter::bounce);
        c.speculative_banned = true;
        if (promotions >= cap) {
          count(job.query, counter::dropped_promotion_cap);
          erase_candidate(it);
        }
        break;
      }
      auto const seq   = std::max(job.seq_at_take, result.conversion_seq);
      auto const bytes = result.bytes != 0 ? result.bytes : c.est_bytes;
      add_ledger(c, qs, job, bytes, seq);
      if (job.t.tier != Tier::HOST || job.t.index >= _host_devices.size() || promotions >= cap) {
        erase_candidate(it);
        break;
      }
      // Staged on HOST: keep the candidate there, so a later pass can take it on to its GPU.
      reindex_seq(c, qs, seq);
      c.location = cucascade::memory::memory_space_id{Tier::HOST, _host_devices[job.t.index]};
      break;
    }
    case promotion_outcome::retired:
    case promotion_outcome::expired:
    case promotion_outcome::already_resident: erase_candidate(it); break;
    case promotion_outcome::stale:
      // A record merged during the attempt is newer than what the worker saw.
      if (c.last_seq != job.seq_at_take) { break; }
      if (result.observed.tier != Tier::SIZE && result.observed != job.expected_source) {
        c.location = result.observed;
        break;
      }
      // Nothing moved (a task claimed a class-B batch): a lost race like a lock refusal. Without
      // this bound it would stay plannable and could be re-planned with no event in between.
      lost_race();
      break;
    case promotion_outcome::unsupported: c.unsupported = true; break;
    case promotion_outcome::too_big:
      if (!c.gpu_banned) {
        c.gpu_banned = true;
        count(job.query, counter::gpu_banned);
      }
      break;
    case promotion_outcome::no_memory:
    case promotion_outcome::gated:
    case promotion_outcome::held_off: break;
    case promotion_outcome::contended: lost_race(); break;
    case promotion_outcome::oom:
      if (c.oom_count < std::numeric_limits<std::uint32_t>::max()) { ++c.oom_count; }
      if (!c.gpu_banned &&
          c.oom_count > static_cast<std::uint32_t>(std::max(_cfg.max_oom_retries, 0))) {
        c.gpu_banned = true;
        count(job.query, counter::gpu_banned);
      }
      break;
    case promotion_outcome::failed:
      effect.first_failure_in_query = !qs.reported_failure;
      qs.reported_failure           = true;
      erase_candidate(it);
      break;
  }
  return effect;
}

// ---- ledger ---------------------------------------------------------------------------------

ledger_entry const* candidate_store::find_ledger(std::uint64_t batch_id) const noexcept
{
  auto const it = _ledger.find(batch_id);
  return it == _ledger.end() ? nullptr : &it->second;
}

std::size_t candidate_store::outstanding_bytes(std::size_t gpu) const noexcept
{
  return gpu < _outstanding.size() ? _outstanding[gpu] : 0;
}

std::size_t candidate_store::outstanding_b_bytes(std::size_t gpu) const noexcept
{
  return gpu < _outstanding_b.size() ? _outstanding_b[gpu] : 0;
}

void candidate_store::add_ledger(candidate const& c,
                                 query_state& qs,
                                 promotion_job const& job,
                                 std::size_t bytes,
                                 std::uint64_t seq)
{
  int const gpu = job.t.tier == Tier::GPU && job.t.index < _gpu_devices.size()
                    ? static_cast<int>(job.t.index)
                    : -1;
  if (auto const old = _ledger.find(c.batch_id); old != _ledger.end()) { erase_ledger(old); }
  // Index first: if the ledger insert throws, the index entry is rolled back.
  qs.ledger_ids.insert(c.batch_id);
  try {
    _ledger.emplace(c.batch_id, ledger_entry{c.batch, c.query, gpu, job.k, bytes, seq, false});
  } catch (...) {
    qs.ledger_ids.erase(c.batch_id);
    throw;
  }
  if (gpu >= 0) {
    auto const g    = static_cast<std::size_t>(gpu);
    _outstanding[g] = sat_add(_outstanding[g], bytes);
    if (job.k == klass::B) { _outstanding_b[g] = sat_add(_outstanding_b[g], bytes); }
  }
}

void candidate_store::reindex_seq(candidate& c, query_state& qs, std::uint64_t seq)
{
  if (seq <= c.last_seq) { return; }
  qs.by_seq.emplace(seq, c.batch_id);  // may throw; the candidate is unchanged then
  qs.by_seq.erase({c.last_seq, c.batch_id});
  c.last_seq = seq;
}

void candidate_store::erase_ledger(
  std::unordered_map<std::uint64_t, ledger_entry>::iterator it) noexcept
{
  auto const& e = it->second;
  if (auto const qit = _queries.find(e.query); qit != _queries.end()) {
    qit->second.ledger_ids.erase(it->first);
  }
  if (e.gpu >= 0) {
    auto const g    = static_cast<std::size_t>(e.gpu);
    _outstanding[g] = sat_sub(_outstanding[g], e.bytes);
    if (e.k == klass::B) { _outstanding_b[g] = sat_sub(_outstanding_b[g], e.bytes); }
  }
  _ledger.erase(it);
}

std::size_t candidate_store::sweep_expired_ledger() noexcept
{
  std::size_t erased = 0;
  for (auto it = _ledger.begin(); it != _ledger.end();) {
    if (it->second.batch.expired()) {
      count(it->second.query, counter::hit);
      auto const next = std::next(it);
      erase_ledger(it);
      it = next;
      ++erased;
    } else {
      ++it;
    }
  }
  return erased;
}

void candidate_store::apply_ledger_verdicts(
  std::span<std::pair<std::uint64_t, ledger_verdict> const> verdicts) noexcept
{
  for (auto const& [id, verdict] : verdicts) {
    auto const it = _ledger.find(id);
    if (it == _ledger.end()) { continue; }
    auto& e = it->second;
    switch (verdict) {
      case ledger_verdict::outstanding: break;
      case ledger_verdict::hit:
        count(e.query, counter::hit);
        erase_ledger(it);
        break;
      case ledger_verdict::reclassify_a:
        if (e.k == klass::B) {
          if (e.gpu >= 0) {
            auto const g      = static_cast<std::size_t>(e.gpu);
            _outstanding_b[g] = sat_sub(_outstanding_b[g], e.bytes);
          }
          e.k = klass::A;
        }
        break;
    }
  }
}

void candidate_store::note_consumer_device(std::uint64_t batch_id, std::size_t gpu) noexcept
{
  auto const it = _ledger.find(batch_id);
  if (it == _ledger.end() || it->second.gpu < 0 || it->second.mispredicted) { return; }
  if (static_cast<std::size_t>(it->second.gpu) != gpu) {
    it->second.mispredicted = true;
    count(it->second.query, counter::mispredicted_device);
    if (it->second.k == klass::B) { count(it->second.query, counter::mispredicted_device_b); }
  }
}

void candidate_store::clear() noexcept
{
  _by_id.clear();
  _queries.clear();
  _stats_by_query.clear();
  _ledger.clear();
  std::fill(_outstanding.begin(), _outstanding.end(), 0);
  std::fill(_outstanding_b.begin(), _outstanding_b.end(), 0);
}

}  // namespace sirius::preload
