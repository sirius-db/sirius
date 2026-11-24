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

#include "op/merge/merge_sort_stats_log.hpp"

#include <format>

namespace sirius::op::merge {

std::string format_merge_sort_stats(std::size_t operator_id, const merge_sort_counters& counters)
{
  auto const& rounds = counters.rounds;
  return std::format(
    "{} op={} windowed_partitions={} one_shot_partitions={} windowed_tasks={} "
    "intermediate_tasks={} mergers_built={} rounds={} rows_out={} round_rows_min={} "
    "round_rows_max={} refill_rows={} refill_bytes={} refill_bytes_host={} refill_bytes_disk={} "
    "refill_bytes_compressed={} runs_read_compressed={} window_growths={} compacted_rows={} "
    "fallback_conversions={} shrinks={}",
    kMergeSortStatsTag,
    operator_id,
    counters.windowed_partitions,
    counters.one_shot_partitions,
    counters.windowed_tasks,
    counters.intermediate_tasks,
    counters.mergers_built,
    rounds.rounds,
    rounds.rows_out,
    rounds.round_rows_min,
    rounds.round_rows_max,
    rounds.refill_rows,
    rounds.refill_bytes,
    rounds.refill_bytes_host,
    rounds.refill_bytes_disk,
    rounds.refill_bytes_compressed,
    rounds.runs_read_compressed,
    rounds.window_growths,
    rounds.compacted_rows,
    rounds.fallback_conversions,
    counters.shrinks);
}

}  // namespace sirius::op::merge
