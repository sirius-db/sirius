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

#include "op/join_cross_schedule.hpp"

#include <algorithm>

namespace sirius::op {

void merge_cross_schedule_batch_ids(partition_cross_schedule& cross,
                                    std::vector<uint64_t> const& probe_ids,
                                    std::vector<uint64_t> const& build_ids)
{
  for (uint64_t id : probe_ids) {
    if (cross.probe_id_seen.insert(id).second) {
      cross.probe_ids.push_back(id);
      cross.probe_paired_count.push_back(0);
    }
  }
  for (uint64_t id : build_ids) {
    if (cross.build_id_seen.insert(id).second) {
      cross.build_ids.push_back(id);
      cross.build_paired_count.push_back(0);
    }
  }
}

std::vector<cross_schedule_discard> collect_cross_schedule_discards(
  std::vector<partition_cross_schedule>& cross, bool probe_finished, bool build_finished)
{
  std::vector<cross_schedule_discard> discards;
  for (std::size_t p = 0; p < cross.size(); ++p) {
    auto& c = cross[p];
    // A probe batch is done once no more build batches can arrive (build finished) AND it has been
    // paired with every known build batch. Requires >= 1 build batch: the empty-opposite case is
    // skipped, matching the legacy grid (which likewise never popped a batch it could not pair).
    if (build_finished && !c.build_ids.empty()) {
      for (std::size_t i = 0; i < c.probe_ids.size(); ++i) {
        uint64_t const id = c.probe_ids[i];
        if (c.probe_popped.count(id)) { continue; }
        if (c.probe_paired_count[i] == c.build_ids.size()) {
          c.probe_popped.insert(id);
          discards.push_back({p, /*is_build=*/false, id});
        }
      }
    }
    if (probe_finished && !c.probe_ids.empty()) {
      for (std::size_t j = 0; j < c.build_ids.size(); ++j) {
        uint64_t const id = c.build_ids[j];
        if (c.build_popped.count(id)) { continue; }
        if (c.build_paired_count[j] == c.probe_ids.size()) {
          c.build_popped.insert(id);
          discards.push_back({p, /*is_build=*/true, id});
        }
      }
    }
  }
  return discards;
}

std::size_t pairing_weighted_probe_bytes(
  std::vector<partition_cross_schedule> const& cross,
  std::unordered_map<uint64_t, std::size_t> const& probe_bytes)
{
  std::size_t weighted = 0;
  for (auto const& c : cross) {
    auto const builds = c.build_ids.size();
    if (builds == 0) { continue; }
    auto const n = std::min(c.probe_ids.size(), c.probe_paired_count.size());
    for (std::size_t i = 0; i < n; ++i) {
      auto const it = probe_bytes.find(c.probe_ids[i]);
      // A failed non-blocking size read is retried on a later pairing.
      if (it == probe_bytes.end()) { continue; }
      // Multiply first to minimize truncation.
      weighted += it->second * c.probe_paired_count[i] / builds;
    }
  }
  return weighted;
}

cross_schedule_pair next_cross_schedule_pair(std::vector<partition_cross_schedule>& cross,
                                             bool probe_finished,
                                             bool build_finished)
{
  for (std::size_t p = 0; p < cross.size(); ++p) {
    auto& c                 = cross[p];
    std::size_t const total = c.probe_ids.size() * c.build_ids.size();
    if (c.scheduled_pairs.size() >= total) { continue; }  // fully scheduled (or a side is empty)
    for (std::size_t i = 0; i < c.probe_ids.size(); ++i) {
      for (std::size_t j = 0; j < c.build_ids.size(); ++j) {
        if (c.scheduled_pairs.insert(encode_cross_pair(i, j)).second) {
          ++c.probe_paired_count[i];
          ++c.build_paired_count[j];
          return {cross_schedule_kind::emit_pair, p, i, j};
        }
      }
    }
  }
  // No schedulable pair now. If a producer can still deliver batches that would create new pairs,
  // wait on it; otherwise the operator is done.
  if (!build_finished) { return {cross_schedule_kind::wait_build, 0, 0, 0}; }
  if (!probe_finished) { return {cross_schedule_kind::wait_probe, 0, 0, 0}; }
  return {cross_schedule_kind::done, 0, 0, 0};
}

bool has_pending_cross_orphan(std::vector<partition_cross_schedule> const& cross,
                              bool probe_finished,
                              bool build_finished)
{
  // Only a terminal state: an empty opposite side is only known-final once both producers finished.
  if (!(probe_finished && build_finished)) { return false; }
  for (auto const& c : cross) {
    // Surviving probe batches whose build side is empty.
    if (c.build_ids.empty()) {
      for (uint64_t id : c.probe_ids) {
        if (!c.probe_popped.count(id)) { return true; }
      }
    }
    // Surviving build batches whose probe side is empty.
    if (c.probe_ids.empty()) {
      for (uint64_t id : c.build_ids) {
        if (!c.build_popped.count(id)) { return true; }
      }
    }
  }
  return false;
}

cross_schedule_orphan next_cross_schedule_orphan(std::vector<partition_cross_schedule>& cross,
                                                 bool probe_finished,
                                                 bool build_finished)
{
  if (!(probe_finished && build_finished)) { return {}; }
  for (std::size_t p = 0; p < cross.size(); ++p) {
    auto& c = cross[p];
    if (c.build_ids.empty()) {  // empty build → each surviving probe batch is an orphan
      for (uint64_t id : c.probe_ids) {
        if (c.probe_popped.insert(id).second) {
          return {/*found=*/true, p, /*present_is_build=*/false, id};
        }
      }
    }
    if (c.probe_ids.empty()) {  // empty probe → each surviving build batch is an orphan
      for (uint64_t id : c.build_ids) {
        if (c.build_popped.insert(id).second) {
          return {/*found=*/true, p, /*present_is_build=*/true, id};
        }
      }
    }
  }
  return {};
}

cross_schedule_kind peek_cross_schedule_kind(std::vector<partition_cross_schedule> const& cross,
                                             bool probe_finished,
                                             bool build_finished)
{
  for (auto const& c : cross) {
    if (c.scheduled_pairs.size() < c.probe_ids.size() * c.build_ids.size()) {
      return cross_schedule_kind::emit_pair;
    }
  }
  // Terminal empty-opposite batches still need a single-side task before the operator is done.
  if (has_pending_cross_orphan(cross, probe_finished, build_finished)) {
    return cross_schedule_kind::emit_pair;
  }
  if (!build_finished) { return cross_schedule_kind::wait_build; }
  if (!probe_finished) { return cross_schedule_kind::wait_probe; }
  return cross_schedule_kind::done;
}

}  // namespace sirius::op
