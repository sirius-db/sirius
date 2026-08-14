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

/**
 * @file test_reader_pruning_gate.cpp
 * @brief Device-free tests for `sirius::op::scan::reader_pruning_gate` -- the WI-0b state machine
 *        that decides per scan whether merging dynamic filters into the parquet reader AST is
 *        due, from the reader's own row-group accounting.
 *
 * The gate is pure decision mechanism (no reader, no device), so these tests drive
 * `applicable`/`record_sample`/`current_state` directly and assert counters exactly -- everything
 * here is single-threaded, so exactness is legitimate where the delivery-time counter contract
 * would otherwise demand deltas or directions.
 */

#include <catch.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/scan/reader_pruning_gate.hpp>

#include <cstdint>

using sirius::op::dynamic_filter_stats;
using sirius::op::scan::reader_pruning_gate;

namespace {

/// Drive a fresh gate to `disabled` with barren samples, all tagged @p generation.
void disable_with_barren_samples(reader_pruning_gate& gate,
                                 dynamic_filter_stats* stats,
                                 std::uint64_t generation)
{
  for (std::uint64_t i = 0; i < reader_pruning_gate::k_disable_after_barren_splits; ++i) {
    gate.record_sample(/*row_groups_considered=*/8, /*row_groups_remaining=*/8, generation, stats);
  }
  REQUIRE(gate.current_state() == reader_pruning_gate::state::disabled);
}

}  // namespace

TEST_CASE("reader_pruning_gate starts measuring and merges", "[scan][dynamic_filter][reader_gate]")
{
  reader_pruning_gate gate;
  dynamic_filter_stats stats;

  REQUIRE(gate.applicable(0));
  REQUIRE(gate.applicable(999));
  REQUIRE(gate.current_state() == reader_pruning_gate::state::measuring);

  auto const snap = stats.snapshot();
  REQUIRE(snap.reader_gate_row_groups_considered == 0);
  REQUIRE(snap.reader_gate_row_groups_pruned == 0);
  REQUIRE(snap.reader_gate_measurements == 0);
  REQUIRE(snap.reader_gate_disabled == 0);
  REQUIRE(snap.reader_gate_rearmed == 0);
  REQUIRE(snap.reader_gate_merges_skipped == 0);
}

TEST_CASE("reader_pruning_gate: any pruned row group activates terminally",
          "[scan][dynamic_filter][reader_gate]")
{
  reader_pruning_gate gate;
  dynamic_filter_stats stats;

  gate.record_sample(8, 7, /*observed_generation=*/1, &stats);
  REQUIRE(gate.current_state() == reader_pruning_gate::state::active);

  for (int i = 0; i < 10; ++i) {
    gate.record_sample(8, 8, /*observed_generation=*/1, &stats);
    REQUIRE(gate.current_state() == reader_pruning_gate::state::active);
    REQUIRE(gate.applicable(0));
    REQUIRE(gate.applicable(1));
  }

  auto const snap = stats.snapshot();
  REQUIRE(snap.reader_gate_measurements == 11);
  REQUIRE(snap.reader_gate_row_groups_pruned == 1);
  REQUIRE(snap.reader_gate_disabled == 0);
}

TEST_CASE("reader_pruning_gate disables after exactly k_disable_after_barren_splits barren samples",
          "[scan][dynamic_filter][reader_gate]")
{
  reader_pruning_gate gate;
  dynamic_filter_stats stats;

  for (int i = 0; i < 3; ++i) {
    gate.record_sample(8, 8, /*observed_generation=*/5, &stats);
  }
  REQUIRE(gate.current_state() == reader_pruning_gate::state::measuring);
  REQUIRE(gate.applicable(5));

  gate.record_sample(8, 8, /*observed_generation=*/5, &stats);
  REQUIRE(gate.current_state() == reader_pruning_gate::state::disabled);
  REQUIRE(stats.snapshot().reader_gate_disabled == 1);
  REQUIRE_FALSE(gate.applicable(5));
  REQUIRE(gate.applicable(6));
}

TEST_CASE("reader_pruning_gate backoff schedule 1, 2, 4", "[scan][dynamic_filter][reader_gate]")
{
  reader_pruning_gate gate;
  dynamic_filter_stats stats;
  disable_with_barren_samples(gate, &stats, /*generation=*/5);  // re-arm at 6, gap 1

  // First permitted re-measurement, barren: gap doubles to 2, re-arm at 6 + 2 = 8.
  gate.record_sample(8, 8, /*observed_generation=*/6, &stats);
  REQUIRE(stats.snapshot().reader_gate_rearmed == 1);
  REQUIRE(stats.snapshot().reader_gate_disabled == 2);
  REQUIRE_FALSE(gate.applicable(7));
  REQUIRE(gate.applicable(8));

  // Second permitted re-measurement, barren: gap doubles to 4, re-arm at 8 + 4 = 12.
  gate.record_sample(8, 8, /*observed_generation=*/8, &stats);
  REQUIRE(stats.snapshot().reader_gate_rearmed == 2);
  REQUIRE(stats.snapshot().reader_gate_disabled == 3);
  REQUIRE_FALSE(gate.applicable(9));
  REQUIRE_FALSE(gate.applicable(10));
  REQUIRE_FALSE(gate.applicable(11));
  REQUIRE(gate.applicable(12));
}

TEST_CASE("reader_pruning_gate: a re-armed measurement that prunes activates terminally",
          "[scan][dynamic_filter][reader_gate]")
{
  constexpr std::uint64_t g = 7;
  reader_pruning_gate gate;
  dynamic_filter_stats stats;
  disable_with_barren_samples(gate, &stats, g);  // re-arm at g + 1

  gate.record_sample(8, 4, /*observed_generation=*/g + 1, &stats);
  REQUIRE(gate.current_state() == reader_pruning_gate::state::active);
  REQUIRE(stats.snapshot().reader_gate_rearmed == 1);
  REQUIRE(gate.applicable(g + 1));
  REQUIRE(gate.applicable(g + 2));
  REQUIRE(gate.applicable(g + 100));
}

TEST_CASE(
  "reader_pruning_gate: pre-decision stragglers consume no budget; a pruning straggler activates",
  "[scan][dynamic_filter][reader_gate]")
{
  constexpr std::uint64_t g = 9;
  reader_pruning_gate gate;
  dynamic_filter_stats stats;
  disable_with_barren_samples(gate, &stats, g);  // re-arm at g + 1
  auto const at_disable = stats.snapshot();

  // Barren sample tagged older than the re-arm point: discarded whole -- no counter moves, the
  // state stays disabled, and the single re-arm budget at g + 1 is not consumed.
  gate.record_sample(8, 8, /*observed_generation=*/g, &stats);
  REQUIRE(gate.current_state() == reader_pruning_gate::state::disabled);
  auto const after_straggler = stats.snapshot();
  REQUIRE(after_straggler.reader_gate_measurements == at_disable.reader_gate_measurements);
  REQUIRE(after_straggler.reader_gate_rearmed == at_disable.reader_gate_rearmed);
  REQUIRE(after_straggler.reader_gate_disabled == at_disable.reader_gate_disabled);
  REQUIRE(gate.applicable(g + 1));

  // A straggler that observed pruning is never discarded: activation is sound from any
  // generation because tightening is monotone.
  gate.record_sample(8, 2, /*observed_generation=*/g, &stats);
  REQUIRE(gate.current_state() == reader_pruning_gate::state::active);
}

TEST_CASE("reader_pruning_gate: non-evidence inputs are no-ops",
          "[scan][dynamic_filter][reader_gate]")
{
  auto require_no_op = [](reader_pruning_gate& gate, dynamic_filter_stats& stats) {
    auto const before       = stats.snapshot();
    auto const state_before = gate.current_state();
    gate.record_sample(0, 0, /*observed_generation=*/3, &stats);
    gate.record_sample(4, 5, /*observed_generation=*/3, &stats);
    REQUIRE(gate.current_state() == state_before);
    auto const after = stats.snapshot();
    REQUIRE(after.reader_gate_row_groups_considered == before.reader_gate_row_groups_considered);
    REQUIRE(after.reader_gate_row_groups_pruned == before.reader_gate_row_groups_pruned);
    REQUIRE(after.reader_gate_measurements == before.reader_gate_measurements);
    REQUIRE(after.reader_gate_disabled == before.reader_gate_disabled);
    REQUIRE(after.reader_gate_rearmed == before.reader_gate_rearmed);
  };

  SECTION("measuring")
  {
    reader_pruning_gate gate;
    dynamic_filter_stats stats;
    require_no_op(gate, stats);
  }
  SECTION("active")
  {
    reader_pruning_gate gate;
    dynamic_filter_stats stats;
    gate.record_sample(8, 7, 1, &stats);
    REQUIRE(gate.current_state() == reader_pruning_gate::state::active);
    require_no_op(gate, stats);
  }
  SECTION("disabled")
  {
    reader_pruning_gate gate;
    dynamic_filter_stats stats;
    disable_with_barren_samples(gate, &stats, 1);
    require_no_op(gate, stats);
  }
}

TEST_CASE("reader_pruning_gate: null stats is safe", "[scan][dynamic_filter][reader_gate]")
{
  reader_pruning_gate gate;

  // Measure to disable.
  for (std::uint64_t i = 0; i < reader_pruning_gate::k_disable_after_barren_splits; ++i) {
    gate.record_sample(8, 8, /*observed_generation=*/1, nullptr);
  }
  REQUIRE(gate.current_state() == reader_pruning_gate::state::disabled);
  REQUIRE_FALSE(gate.applicable(1));

  // Re-arm barren, re-disable on the doubled gap.
  gate.record_sample(8, 8, /*observed_generation=*/2, nullptr);
  REQUIRE(gate.current_state() == reader_pruning_gate::state::disabled);
  REQUIRE_FALSE(gate.applicable(3));
  REQUIRE(gate.applicable(4));

  // Re-armed measurement with pruning activates.
  gate.record_sample(8, 3, /*observed_generation=*/4, nullptr);
  REQUIRE(gate.current_state() == reader_pruning_gate::state::active);
  REQUIRE(gate.applicable(0));
}
