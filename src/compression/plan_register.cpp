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

#include "plan_register.hpp"

#include <api/simpatico_codegen.hpp>

#include <algorithm>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace sirius::compression {

plan_register& plan_register::global()
{
  static plan_register instance;
  return instance;
}

void plan_register::set_table_plan(const std::string& table_name, std::string full_plan_dsl)
{
  std::unique_lock lock(_mutex);
  _table_plans[table_name] = std::move(full_plan_dsl);
}

void plan_register::clear_table_plan(const std::string& table_name)
{
  std::unique_lock lock(_mutex);
  _table_plans.erase(table_name);
}

std::optional<std::string> plan_register::resolve_table_plan(const std::string& table_name) const
{
  std::shared_lock lock(_mutex);
  auto it = _table_plans.find(table_name);
  if (it != _table_plans.end() && !it->second.empty()) { return it->second; }
  return std::nullopt;
}

void plan_register::set_plan(const std::string& table_name,
                             const std::string& column_name,
                             std::string plan_dsl)
{
  std::unique_lock lock(_mutex);
  _col_plans[table_name + "::" + column_name] = std::move(plan_dsl);
}

void plan_register::clear_plan(const std::string& table_name, const std::string& column_name)
{
  std::unique_lock lock(_mutex);
  _col_plans.erase(table_name + "::" + column_name);
}

plan_register::spill_plan_decision plan_register::decide_spill_plan(
  const cucascade::shared_data_repository* repo, std::uint64_t replan_after_uses) const
{
  std::shared_lock lock(_mutex);
  auto it = _spill_plans.find(repo);
  if (it == _spill_plans.end()) { return {spill_plan_verdict::explore, {}}; }

  if (it->second.columns.empty()) {
    // No plans yet — only a record of failed explorations. Keep asking for one
    // until they prove durable, then stop: re-running a beam search that has
    // repeatedly failed costs far more than spilling uncompressed. The entry
    // still ages, so the edge is retried on the normal replan schedule.
    const auto& state = it->second;
    const std::uint64_t period =
      state.replan_interval != 0 ? state.replan_interval : replan_after_uses;
    const bool expired = period > 0 && state.uses >= period;
    if (!expired && state.explore_exhausted) { return {spill_plan_verdict::skip, {}}; }
    return {spill_plan_verdict::explore, {}};
  }

  // An expired entry is re-explored whatever its verdict was, so a stale plan or
  // a premature "not worth it" ruling does not stick for the rest of the query.
  // Adaptive backoff (see conclude_spill_attempt) overrides the configured
  // schedule once it has stretched this edge's own interval.
  const auto& state = it->second;
  const std::uint64_t period =
    state.replan_interval != 0 ? state.replan_interval : replan_after_uses;
  if (period > 0 && state.uses >= period) { return {spill_plan_verdict::explore, {}}; }

  // Nothing here compresses, so there is no point paying to find out again.
  // A partially viable edge still proceeds: its viable columns are compressed and
  // the rest stored raw.
  if (state.viable_count() == 0) { return {spill_plan_verdict::skip, {}}; }
  return {spill_plan_verdict::use, state.columns};
}

namespace {

/// True when @p a and @p b differ by more than @p threshold, relative to the
/// larger of the two. Using the larger as the denominator keeps the test
/// symmetric and handles a previously unmeasured (zero) value: anything against
/// zero reads as a full change, while zero against zero reads as none.
bool differs_materially(double a, double b, double threshold)
{
  const double scale = std::max(std::abs(a), std::abs(b));
  if (scale <= 0.0) { return false; }
  return std::abs(a - b) / scale > threshold;
}

/// True when @p candidate performs materially differently from @p cached.
bool worth_adopting(const plan_register::column_plan_state& cached,
                    const plan_register::column_plan_candidate& candidate,
                    double threshold)
{
  return differs_materially(cached.compression_ratio, candidate.compression_ratio, threshold) ||
         differs_materially(cached.compress_gbps, candidate.compress_gbps, threshold) ||
         differs_materially(cached.decompress_gbps, candidate.decompress_gbps, threshold);
}

plan_register::column_plan_state adopt(plan_register::column_plan_candidate&& candidate)
{
  plan_register::column_plan_state state;
  state.dsl                = std::move(candidate.dsl);
  state.viable             = true;
  state.consecutive_errors = 0;
  state.compression_ratio  = candidate.compression_ratio;
  state.compress_gbps      = candidate.compress_gbps;
  state.decompress_gbps    = candidate.decompress_gbps;
  return state;
}

}  // namespace

void plan_register::set_spill_plan(const cucascade::shared_data_repository* repo,
                                   std::vector<column_plan_candidate> candidates,
                                   double change_threshold)
{
  std::unique_lock lock(_mutex);

  auto it = _spill_plans.find(repo);
  // An entry holding only a failed-exploration streak is not a previous plan, so
  // this is a first success rather than a replan — and installing it clears the
  // streak, since `fresh` starts with the counters at zero.
  const bool replan = it != _spill_plans.end() && !it->second.columns.empty();

  spill_plan_state fresh;
  fresh.columns.reserve(candidates.size());

  if (!replan) {
    for (auto& candidate : candidates) {
      fresh.columns.push_back(adopt(std::move(candidate)));
    }
    _spill_plans[repo] = std::move(fresh);
    return;
  }

  // Re-explore: decide each column on its own. A candidate that performs like the
  // cached plan is dropped — the explorer readily returns a differently spelled
  // plan with the same characteristics, and adopting those would churn the cache
  // and register as a change, resetting the backoff and re-exploring forever.
  const auto& prev = it->second;
  bool adopted_any = prev.columns.size() != candidates.size();

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (i < prev.columns.size() &&
        !worth_adopting(prev.columns[i], candidates[i], change_threshold)) {
      // Keep the cached plan *and* its verdict: an equivalent plan will not
      // compress any better than the one already judged.
      fresh.columns.push_back(prev.columns[i]);
      continue;
    }
    fresh.columns.push_back(adopt(std::move(candidates[i])));
    adopted_any = true;
  }

  fresh.from_replan       = true;
  fresh.plan_changed      = adopted_any;
  fresh.prev_viable_count = prev.viable_count();
  fresh.replan_interval   = prev.replan_interval;

  _spill_plans[repo] = std::move(fresh);
}

void plan_register::conclude_spill_attempt(const cucascade::shared_data_repository* repo,
                                           std::span<const spill_attempt_outcome> per_column,
                                           std::uint64_t base_interval,
                                           std::uint32_t error_tolerance)
{
  // Saturate rather than wrap when doubling; at this point the edge is
  // effectively never re-explored again, which is the intent.
  constexpr std::uint64_t max_interval = std::numeric_limits<std::uint64_t>::max() / 2;
  const std::uint32_t tolerance        = std::max<std::uint32_t>(error_tolerance, 1);

  std::unique_lock lock(_mutex);
  auto it = _spill_plans.find(repo);
  if (it == _spill_plans.end()) { return; }
  auto& state = it->second;

  // An empty span means the attempt died before any column could be judged;
  // treat every column as having errored.
  bool any_measured = false;
  for (std::size_t i = 0; i < state.columns.size(); ++i) {
    auto& col          = state.columns[i];
    const auto outcome = i < per_column.size() ? per_column[i] : spill_attempt_outcome::failed;

    if (outcome == spill_attempt_outcome::failed) {
      // Not evidence about this column's data — absorb it until it proves durable.
      ++col.consecutive_errors;
      if (col.consecutive_errors < tolerance) { continue; }
      col.viable = false;
    } else {
      col.viable   = outcome == spill_attempt_outcome::compressed;
      any_measured = true;
    }
    col.consecutive_errors = 0;
  }

  // Only conclude a pending re-explore once something was actually measured;
  // an all-errors attempt has not judged the new plans yet.
  if (state.from_replan && any_measured) {
    const std::uint64_t current =
      state.replan_interval != 0 ? state.replan_interval : base_interval;
    const std::size_t now_viable = state.viable_count();

    // Only a change that actually compresses something is worth staying on
    // schedule for. The same plans, or plans that still compress nothing, teach
    // us nothing — back off so we stop paying for fruitless explores.
    const bool changed = state.plan_changed || now_viable != state.prev_viable_count;
    if (changed && now_viable > 0) {
      state.replan_interval = base_interval;
    } else if (current > 0) {
      state.replan_interval = current > max_interval / 2 ? max_interval : current * 2;
    }

    state.from_replan  = false;
    state.plan_changed = false;
  }
}

void plan_register::set_spill_column_origins(const cucascade::shared_data_repository* repo,
                                             spill_column_origins origins)
{
  std::unique_lock lock(_mutex);
  _spill_origins[repo] = std::move(origins);
}

std::optional<plan_register::spill_column_origins> plan_register::resolve_spill_column_origins(
  const cucascade::shared_data_repository* repo) const
{
  std::shared_lock lock(_mutex);
  auto it = _spill_origins.find(repo);
  if (it == _spill_origins.end() || it->second.empty()) { return std::nullopt; }
  return it->second;
}

std::optional<std::vector<std::optional<std::string>>> plan_register::seed_plans_from_lineage(
  const cucascade::shared_data_repository* repo, std::size_t expected_columns) const
{
  std::shared_lock lock(_mutex);
  auto it = _spill_origins.find(repo);
  if (it == _spill_origins.end() || it->second.size() != expected_columns) { return std::nullopt; }

  std::vector<std::optional<std::string>> seeds(expected_columns);
  bool any = false;
  for (std::size_t i = 0; i < expected_columns; ++i) {
    auto const& origin = it->second[i];
    if (!origin.has_value()) { continue; }  // computed column: no base plan

    auto plan_it = _table_plans.find(origin->table_name);
    if (plan_it == _table_plans.end() || plan_it->second.empty()) { continue; }

    // The table plan has one block per full-table column in schema order, which is
    // exactly the index space table_column_index lives in.
    auto block = select_plan_blocks(plan_it->second, {origin->table_column_index});
    if (!block.has_value()) { continue; }
    seeds[i] = std::move(*block);
    any      = true;
  }
  if (!any) { return std::nullopt; }
  return seeds;
}

void plan_register::note_spill_explore_failure(const cucascade::shared_data_repository* repo,
                                               std::uint32_t error_tolerance)
{
  std::unique_lock lock(_mutex);
  // Creates the entry when absent: exploration fails before any per-column state
  // exists, so without this there is nothing to record the streak against and
  // every later spill from this edge re-runs the whole beam search.
  auto& state = _spill_plans[repo];
  ++state.explore_failures;
  if (state.explore_failures >= std::max<std::uint32_t>(error_tolerance, 1)) {
    state.explore_exhausted = true;
  }
}

void plan_register::note_spill_plan_use(const cucascade::shared_data_repository* repo)
{
  std::unique_lock lock(_mutex);
  auto it = _spill_plans.find(repo);
  if (it != _spill_plans.end()) { ++it->second.uses; }
}

void plan_register::clear_spill_plan(const cucascade::shared_data_repository* repo)
{
  std::unique_lock lock(_mutex);
  _spill_plans.erase(repo);
}

std::optional<plan_register::spill_plan_state> plan_register::resolve_spill_plan(
  const cucascade::shared_data_repository* repo) const
{
  std::shared_lock lock(_mutex);
  auto it = _spill_plans.find(repo);
  if (it != _spill_plans.end() && !it->second.columns.empty()) { return it->second; }
  return std::nullopt;
}

void plan_register::clear_spill_state()
{
  std::unique_lock lock(_mutex);
  _spill_plans.clear();
  _spill_origins.clear();
}

void plan_register::clear_all()
{
  std::unique_lock lock(_mutex);
  _table_plans.clear();
  _col_plans.clear();
  _spill_plans.clear();
  _spill_origins.clear();
}

std::optional<std::string> select_plan_blocks(const std::string& full_plan_dsl,
                                              const std::vector<std::size_t>& column_indices)
{
  auto blocks = simpatico::split_plan_dsl(full_plan_dsl);
  std::string out;
  for (std::size_t k = 0; k < column_indices.size(); ++k) {
    std::size_t const idx = column_indices[k];
    if (idx >= blocks.size()) { return std::nullopt; }
    if (k != 0) { out += "\n---\n"; }
    out += blocks[idx];
  }
  return out;
}

}  // namespace sirius::compression
