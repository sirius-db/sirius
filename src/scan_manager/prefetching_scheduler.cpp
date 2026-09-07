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

#include "scan_manager/prefetching_scheduler.hpp"

#include "op/sirius_physical_operator.hpp"

#include <algorithm>
#include <limits>

namespace sirius::scan_manager {

namespace {

using planner::scheduling_mode;

/// How many consecutive picks a step gets before the rotation moves on.
/// Derived from the mode rather than trusted from the step so the policy lives
/// in one place: barrier_all holds the rotation until it is depleted, pipeline
/// is always a single split, and only barrier_serial is actually rationed.
std::size_t quantum_of(const planner::prefetch_step& step) noexcept
{
  switch (step.mode) {
    case scheduling_mode::barrier_all: return std::numeric_limits<std::size_t>::max();
    case scheduling_mode::pipeline: return 1;
    case scheduling_mode::barrier_serial: return std::max<std::size_t>(step.count, 1);
  }
  return 1;
}

/// Whether @p cur continues the group @p prev belongs to.  See the grouping
/// rules on @ref prefetching_scheduler.
bool continues_group(const planner::prefetch_step& prev, const planner::prefetch_step& cur) noexcept
{
  switch (cur.mode) {
    // Feeds a FULL port: nothing may interleave with it, on either side.
    case scheduling_mode::barrier_all: return false;
    // Only scans blocking the SAME barrier are advanced together.
    case scheduling_mode::barrier_serial:
      return prev.mode == scheduling_mode::barrier_serial && prev.branch_id == cur.branch_id;
    // Nothing gates these, so there is no barrier to group by -- adjacency alone
    // decides, and the branch id is deliberately ignored.
    case scheduling_mode::pipeline: return prev.mode == scheduling_mode::pipeline;
  }
  return false;
}

}  // namespace

void prefetching_scheduler::reset(std::span<const planner::prefetch_step> order)
{
  clear();

  planner::prefetch_step const* prev = nullptr;
  for (auto const& step : order) {
    if (step.scan == nullptr || !step.scan->has_operator_id()) { continue; }

    auto const op_id = step.scan->get_operator_id();
    // A scan appearing twice would give one operator two cursors and two
    // depletion flags; keep the first position, which is the one the traversal
    // actually wants it prefetched at.
    if (_by_operator.contains(op_id)) { continue; }

    if (prev == nullptr || !continues_group(*prev, step)) { _groups.emplace_back(); }

    _by_operator.emplace(op_id, _entries.size());
    _groups.back().push_back(_entries.size());
    _entries.push_back(entry{.scan        = step.scan,
                             .operator_id = op_id,
                             .branch_id   = step.branch_id,
                             .mode        = step.mode,
                             .quantum     = quantum_of(step),
                             .stage       = io::cache::scan_stage::none,
                             .depleted    = false});
    prev = &step;
  }

  advance();
}

void prefetching_scheduler::clear()
{
  _entries.clear();
  _groups.clear();
  _by_operator.clear();
  _group   = 0;
  _member  = 0;
  _emitted = 0;
}

void prefetching_scheduler::update(std::size_t operator_id, io::cache::scan_stage stage)
{
  auto it = _by_operator.find(operator_id);
  if (it == _by_operator.end()) { return; }

  auto& e = _entries[it->second];
  e.stage = stage;
  // `disposed` is the caller's assertion that this operator is finished, not
  // merely that one of its splits is -- see the note on the class.
  if (stage == io::cache::scan_stage::disposed) { e.depleted = true; }

  advance();
}

bool prefetching_scheduler::group_depleted(const group& g) const
{
  return std::ranges::all_of(g, [this](std::size_t idx) { return _entries[idx].depleted; });
}

void prefetching_scheduler::advance()
{
  while (_group < _groups.size()) {
    auto const& g = _groups[_group];

    // Nothing left in this group: groups run strictly in order, so move on.
    if (g.empty() || group_depleted(g)) {
      ++_group;
      _member  = 0;
      _emitted = 0;
      continue;
    }

    auto const& e = _entries[g[_member]];
    if (!e.depleted && _emitted < e.quantum) { return; }

    // This member is done for now -- either retired or its quantum is spent.
    // The group has a live member (checked above) and every quantum is at least
    // one, so rotating is guaranteed to terminate on a servable member.
    _member  = (_member + 1) % g.size();
    _emitted = 0;
  }
}

op::sirius_physical_operator* prefetching_scheduler::get_next_prefetching_operator()
{
  advance();
  if (_group >= _groups.size()) { return nullptr; }

  auto& e = _entries[_groups[_group][_member]];
  ++_emitted;
  return e.scan;
}

std::optional<std::size_t> prefetching_scheduler::peek_next_operator_id() const
{
  // advance() is not const, but the cursor is already parked on a servable
  // member after every reset/update/get_next, so a plain read is enough here.
  if (_group >= _groups.size()) { return std::nullopt; }
  auto const& e = _entries[_groups[_group][_member]];
  if (e.depleted) { return std::nullopt; }
  return e.operator_id;
}

std::vector<std::size_t> prefetching_scheduler::peek_group_operator_ids() const
{
  std::vector<std::size_t> ids;
  if (_group >= _groups.size()) { return ids; }
  auto const& g = _groups[_group];
  ids.reserve(g.size());
  for (std::size_t i = 0; i < g.size(); ++i) {
    auto const& e = _entries[g[(_member + i) % g.size()]];
    if (!e.depleted) { ids.push_back(e.operator_id); }
  }
  return ids;
}

std::vector<std::size_t> prefetching_scheduler::peek_lookahead_operator_ids() const
{
  std::vector<std::size_t> ids;
  for (std::size_t g = _group + 1; g < _groups.size(); ++g) {
    for (auto const idx : _groups[g]) {
      auto const& e = _entries[idx];
      if (!e.depleted) { ids.push_back(e.operator_id); }
    }
  }
  return ids;
}

bool prefetching_scheduler::focus_member(std::size_t operator_id)
{
  if (_group >= _groups.size()) { return false; }
  auto const& g = _groups[_group];
  for (std::size_t i = 0; i < g.size(); ++i) {
    auto const& e = _entries[g[i]];
    if (e.operator_id != operator_id || e.depleted) { continue; }
    if (_member != i) {
      _member  = i;
      _emitted = 0;
    }
    return true;
  }
  return false;
}

io::cache::scan_stage prefetching_scheduler::stage_of(std::size_t operator_id) const
{
  auto it = _by_operator.find(operator_id);
  if (it == _by_operator.end()) { return io::cache::scan_stage::none; }
  return _entries[it->second].stage;
}

}  // namespace sirius::scan_manager
