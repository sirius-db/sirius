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
  if (it == _spill_plans.end() || it->second.dsl.empty()) {
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
  if (!state.viable) { return {spill_plan_verdict::skip, {}}; }
  return {spill_plan_verdict::use, state.dsl};
}

void plan_register::set_spill_plan(const cucascade::shared_data_repository* repo,
                                   std::string plan_dsl)
{
  std::unique_lock lock(_mutex);

  spill_plan_state fresh;
  fresh.dsl = std::move(plan_dsl);

  // Replacing an entry means this is a re-explore: carry the current backoff
  // interval and remember what we are replacing, so conclude_spill_attempt can
  // tell whether the explore actually changed anything.
  if (auto it = _spill_plans.find(repo); it != _spill_plans.end()) {
    fresh.from_replan     = true;
    fresh.plan_changed    = it->second.dsl != fresh.dsl;
    fresh.prev_viable     = it->second.viable;
    fresh.replan_interval = it->second.replan_interval;
  }

  _spill_plans[repo] = std::move(fresh);
}

void plan_register::conclude_spill_attempt(const cucascade::shared_data_repository* repo,
                                           bool compressed_ok,
                                           std::uint64_t base_interval)
{
  // Saturate rather than wrap when doubling; at this point the edge is
  // effectively never re-explored again, which is the intent.
  constexpr std::uint64_t max_interval = std::numeric_limits<std::uint64_t>::max() / 2;

  std::unique_lock lock(_mutex);
  auto it = _spill_plans.find(repo);
  if (it == _spill_plans.end()) { return; }
  auto& state = it->second;

  if (state.from_replan) {
    const std::uint64_t current =
      state.replan_interval != 0 ? state.replan_interval : base_interval;

    // Only a change that actually compresses is worth staying on schedule for.
    // Same plan, or a new plan that still misses the threshold, teaches us
    // nothing — back off so we stop paying for fruitless explores.
    const bool changed = state.plan_changed || (compressed_ok != state.prev_viable);
    if (changed && compressed_ok) {
      state.replan_interval = base_interval;
    } else if (current > 0) {
      state.replan_interval = current > max_interval / 2 ? max_interval : current * 2;
    }

    state.from_replan  = false;
    state.plan_changed = false;
  }

  state.viable = compressed_ok;
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
  if (it != _spill_plans.end() && !it->second.dsl.empty()) { return it->second; }
  return std::nullopt;
}

void plan_register::clear_all()
{
  std::unique_lock lock(_mutex);
  _table_plans.clear();
  _col_plans.clear();
  _spill_plans.clear();
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
