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
#include <cctype>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <utility>

namespace sirius::compression {

plan_register& plan_register::global()
{
  static plan_register instance;
  return instance;
}

void plan_register::set_table_plan(const std::string& table_name, std::string full_plan_dsl)
{
  // Parse the measurement comments before the string is moved from, and before
  // any consumer runs it through split_plan_dsl — which strips every `#` line.
  auto metrics = parse_plan_metrics(full_plan_dsl);

  std::unique_lock lock(_mutex);
  _table_plans[table_name]        = std::move(full_plan_dsl);
  _table_plan_metrics[table_name] = std::move(metrics);
}

void plan_register::clear_table_plan(const std::string& table_name)
{
  std::unique_lock lock(_mutex);
  _table_plans.erase(table_name);
  _table_plan_metrics.erase(table_name);
}

std::optional<plan_register::plan_metrics> plan_register::resolve_plan_metrics(
  const std::string& table_name, std::size_t column_index) const
{
  std::shared_lock lock(_mutex);
  auto it = _table_plan_metrics.find(table_name);
  if (it == _table_plan_metrics.end() || column_index >= it->second.size()) { return std::nullopt; }
  return it->second[column_index];
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

/// True when @p candidate is materially BETTER than @p cached, and so worth
/// swapping the cached plan for.
///
/// This used to ask only whether the two performed *differently*, in either
/// direction, which meant a re-explore traded a good plan for a worse one as
/// readily as the reverse — and `adopt()` then marked the replacement viable
/// regardless. Combined with cached plans carrying placeholder metrics, the
/// explorer won every comparison: q3/SF1000 decayed from 5.43x on the seeded
/// plans to 3.83x after the first re-explore.
///
/// Ratio decides. Throughput only breaks ties, and can veto: the spill path has
/// no other speed check anywhere (plan_quality_gate, which does test throughput,
/// is reached only from the task-output path), so if a materially slower plan is
/// not refused here it is not refused at all.
bool worth_adopting(const plan_register::column_plan_state& cached,
                    const plan_register::column_plan_candidate& candidate,
                    double threshold)
{
  // Nothing measured for the plan in use, so there is no claim to defend.
  if (cached.compression_ratio <= 0.0) { return true; }

  if (differs_materially(cached.compression_ratio, candidate.compression_ratio, threshold)) {
    // Only ever trade up. Note the asymmetry in how the two numbers were
    // obtained: the cached ratio is measured on whole spilled batches, while the
    // candidate's comes from the explorer's row-prefix sample, which the explorer
    // itself warns is optimistic on ordered columns. Requiring a material
    // improvement is what keeps a flattering sample from displacing a plan with a
    // real track record; if the candidate is genuinely better, the next attempt
    // measures it and the record corrects itself.
    return candidate.compression_ratio > cached.compression_ratio;
  }

  // Comparable ratios: take a materially faster plan, refuse a materially slower
  // one, and otherwise leave the incumbent alone so the cache stops churning.
  const bool slower =
    (differs_materially(cached.compress_gbps, candidate.compress_gbps, threshold) &&
     candidate.compress_gbps < cached.compress_gbps) ||
    (differs_materially(cached.decompress_gbps, candidate.decompress_gbps, threshold) &&
     candidate.decompress_gbps < cached.decompress_gbps);
  if (slower) { return false; }

  return (differs_materially(cached.compress_gbps, candidate.compress_gbps, threshold) &&
          candidate.compress_gbps > cached.compress_gbps) ||
         (differs_materially(cached.decompress_gbps, candidate.decompress_gbps, threshold) &&
          candidate.decompress_gbps > cached.decompress_gbps);
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
                                           std::span<const spill_column_result> per_column,
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
    auto& col = state.columns[i];
    const spill_column_result result =
      i < per_column.size() ? per_column[i] : spill_column_result{};

    if (result.outcome == spill_attempt_outcome::failed) {
      // Not evidence about this column's data — absorb it until it proves durable.
      ++col.consecutive_errors;
      if (col.consecutive_errors < tolerance) { continue; }
      col.viable = false;
    } else {
      col.viable   = result.outcome == spill_attempt_outcome::compressed;
      any_measured = true;
      // Replace the plan's recorded ratio with what it actually delivered on a
      // whole batch. Until this existed, a seeded plan kept the placeholder 1.0
      // it was installed with and lost every replan comparison by default.
      if (result.achieved_ratio > 0.0) { col.compression_ratio = result.achieved_ratio; }
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
  // Same lifetime and the same recycled-address hazard: keyed by a repository
  // pointer that QueryEnd is about to free.
  _output_plans.clear();
}

void plan_register::clear_all()
{
  std::unique_lock lock(_mutex);
  _table_plans.clear();
  _table_plan_metrics.clear();
  _col_plans.clear();
  _spill_plans.clear();
  _spill_origins.clear();
  _output_plans.clear();
}

namespace {

/// Find `key:` in @p line as a whole token — preceded by start-of-string or a
/// space — and parse the number after it. Without the boundary check, looking for
/// "comp:" would match inside "decomp:", which sits on the same line.
std::optional<double> metric_after(std::string_view line, std::string_view key)
{
  for (std::size_t pos = line.find(key); pos != std::string_view::npos;
       pos             = line.find(key, pos + 1)) {
    if (pos != 0 && line[pos - 1] != ' ' && line[pos - 1] != '#') { continue; }
    std::size_t at = pos + key.size();
    if (at >= line.size() || line[at] != ':') { continue; }
    ++at;
    while (at < line.size() && line[at] == ' ') {
      ++at;
    }
    const std::size_t start = at;
    while (at < line.size() && (std::isdigit(static_cast<unsigned char>(line[at])) != 0 ||
                                line[at] == '.' || line[at] == '-')) {
      ++at;
    }
    if (at == start) { return std::nullopt; }
    // std::stod over a copy: the token is short and this runs once per plan at
    // startup, so from_chars' extra machinery buys nothing here.
    try {
      return std::stod(std::string(line.substr(start, at - start)));
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace

std::vector<std::optional<plan_register::plan_metrics>> parse_plan_metrics(
  std::string_view full_plan_dsl)
{
  std::vector<std::optional<plan_register::plan_metrics>> out;

  bool has_dsl = false;
  std::optional<double> ratio, comp, decomp;

  // Close the block under construction, mirroring split_plan_dsl: a block with no
  // DSL lines is not emitted at all, so indices stay aligned with its output.
  auto flush = [&]() {
    if (has_dsl) {
      if (ratio && comp && decomp) {
        out.push_back(plan_register::plan_metrics{*ratio, *comp, *decomp});
      } else {
        out.emplace_back();
      }
    }
    has_dsl = false;
    ratio = comp = decomp = std::nullopt;
  };

  std::size_t i = 0;
  while (i < full_plan_dsl.size()) {
    std::size_t line_end = full_plan_dsl.find('\n', i);
    if (line_end == std::string_view::npos) { line_end = full_plan_dsl.size(); }
    std::string_view line = full_plan_dsl.substr(i, line_end - i);
    if (!line.empty() && line.back() == '\r') { line.remove_suffix(1); }
    while (!line.empty() && line.front() == ' ') {
      line.remove_prefix(1);
    }
    while (!line.empty() && line.back() == ' ') {
      line.remove_suffix(1);
    }

    if (line == "---") {
      flush();
    } else if (!line.empty() && line.front() == '#') {
      if (auto v = metric_after(line, "ratio")) { ratio = v; }
      if (auto v = metric_after(line, "decomp")) { decomp = v; }
      if (auto v = metric_after(line, "comp")) { comp = v; }
    } else if (!line.empty()) {
      has_dsl = true;
    }

    i = (line_end == full_plan_dsl.size()) ? full_plan_dsl.size() : line_end + 1;
  }
  flush();
  return out;
}

std::vector<plan_register::output_plan_selection> plan_register::select_output_plans(
  const cucascade::shared_data_repository* repo,
  std::size_t expected_columns,
  const plan_quality_gate& gate) const
{
  std::shared_lock lock(_mutex);
  std::vector<output_plan_selection> selected;

  auto origins = _spill_origins.find(repo);
  if (origins == _spill_origins.end() || origins->second.size() != expected_columns) {
    return selected;
  }

  for (std::size_t i = 0; i < expected_columns; ++i) {
    auto const& origin = origins->second[i];
    if (!origin.has_value()) { continue; }  // computed column: no base plan to judge

    auto metrics_it = _table_plan_metrics.find(origin->table_name);
    if (metrics_it == _table_plan_metrics.end() ||
        origin->table_column_index >= metrics_it->second.size()) {
      continue;
    }
    auto const& measured = metrics_it->second[origin->table_column_index];
    if (!measured.has_value() || !gate.admits(*measured)) { continue; }

    auto plan_it = _table_plans.find(origin->table_name);
    if (plan_it == _table_plans.end() || plan_it->second.empty()) { continue; }
    auto block = select_plan_blocks(plan_it->second, {origin->table_column_index});
    if (!block.has_value()) { continue; }

    const bool order_dependent = block->find("delta") != std::string::npos;
    selected.push_back({i, std::move(*block), *measured, order_dependent});
  }
  return selected;
}

std::optional<plan_register::output_column_plans> plan_register::decide_output_plan(
  const cucascade::shared_data_repository* repo,
  std::size_t expected_columns,
  const plan_quality_gate& gate)
{
  {
    std::shared_lock lock(_mutex);
    auto it = _output_plans.find(repo);
    if (it != _output_plans.end()) {
      if (!it->second.any_viable) { return std::nullopt; }
      return it->second.columns;
    }
  }

  // select_output_plans takes the lock itself, so decide outside our own.
  auto picked = select_output_plans(repo, expected_columns, gate);

  output_edge_state fresh;
  fresh.columns.assign(expected_columns, std::nullopt);
  for (auto& p : picked) {
    if (p.column_index < expected_columns) {
      fresh.columns[p.column_index] = std::move(p.dsl);
      fresh.any_viable              = true;
    }
  }

  std::unique_lock lock(_mutex);
  // Another thread may have decided this edge while we were selecting; its
  // decision is equivalent (same inputs), so keep whichever landed first rather
  // than clobbering verdicts it may already have concluded.
  auto [it, inserted] = _output_plans.try_emplace(repo, std::move(fresh));
  if (!it->second.any_viable) { return std::nullopt; }
  return it->second.columns;
}

void plan_register::conclude_output_attempt(const cucascade::shared_data_repository* repo,
                                            std::span<const double> achieved_ratios,
                                            const plan_quality_gate& gate)
{
  std::unique_lock lock(_mutex);
  auto it = _output_plans.find(repo);
  if (it == _output_plans.end()) { return; }
  auto& state = it->second;

  bool any = false;
  for (std::size_t i = 0; i < state.columns.size(); ++i) {
    if (!state.columns[i].has_value()) { continue; }
    // A column with no measurement this time keeps its plan: silence is not
    // evidence that the plan failed.
    if (i < achieved_ratios.size() && achieved_ratios[i] > 0.0 &&
        achieved_ratios[i] <= gate.min_ratio) {
      state.columns[i] = std::nullopt;
      continue;
    }
    any = true;
  }
  state.any_viable = any;
}

std::optional<plan_register::output_column_plans> plan_register::resolve_output_plan(
  const cucascade::shared_data_repository* repo) const
{
  std::shared_lock lock(_mutex);
  auto it = _output_plans.find(repo);
  if (it == _output_plans.end()) { return std::nullopt; }
  return it->second.columns;
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
