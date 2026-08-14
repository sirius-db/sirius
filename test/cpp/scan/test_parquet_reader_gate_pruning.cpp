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
 * @file test_parquet_reader_gate_pruning.cpp
 * @brief WI-0b at the real merge site: `parquet_gpu_ingestible::materialize_metadata_to_table`
 *        driven split-by-split with controlled refinement-slot publications, so the reader
 *        pruning gate's samples come from `cudf::io::read_parquet`'s own row-group accounting.
 *
 * The fixture is one parquet file of BIGINT `(id, v)` whose row groups hold disjoint ascending
 * `v` bands, split with `approximate_batch_size = 1` so every row group seals its own split.
 * Publication ordering is fully under test control here, so sections assert exact counter
 * deltas on a local `dynamic_filter_stats` -- unlike the integration layer, where publication
 * races split arrival. The first section pins the cuDF accounting premise the whole signal
 * rests on (`num_input_row_groups` under `set_row_groups`); if it fails, the gate's design must
 * be revisited rather than patched (work order, Risks R1).
 */

#include "operator/operator_test_utils.hpp"

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/reader_pruning_gate.hpp>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace scan = sirius::op::scan;
using scan::reader_pruning_gate;
using sirius::op::dynamic_filter_stats;
using sirius::op::refinement_publish_result;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_dynamic_zone_map_filter;
using sirius::op::zone_map_entry;

/// Rows per row group in the fixture, and the number of row groups (= splits) it produces.
constexpr int64_t ROWS_PER_GROUP = 2048;
constexpr int64_t GROUP_COUNT    = 16;
/// Row group `g` holds `v` in `[g * BAND_STRIDE, g * BAND_STRIDE + ROWS_PER_GROUP - 1]`.
constexpr int64_t BAND_STRIDE = 10000;

struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0))
  {
  }
};

test_env& env()
{
  static test_env e;
  return e;
}

/// Write the banded fixture: 16 row groups of 2048 rows, disjoint ascending `v` bands.
std::filesystem::path write_banded_parquet(std::filesystem::path const& dir)
{
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto const row_count = ROWS_PER_GROUP * GROUP_COUNT;
  auto result = con.Query("CREATE TABLE banded AS SELECT range::BIGINT AS id, ((range // " +
                          std::to_string(ROWS_PER_GROUP) + ") * " + std::to_string(BAND_STRIDE) +
                          " + (range % " + std::to_string(ROWS_PER_GROUP) +
                          "))::BIGINT AS v FROM range(0, " + std::to_string(row_count) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto const path = dir / "banded.parquet";
  result = con.Query("COPY banded TO '" + path.string() + "' (FORMAT PARQUET, ROW_GROUP_SIZE " +
                     std::to_string(ROWS_PER_GROUP) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  return path;
}

std::unique_ptr<cudf::scalar> make_int64_scalar(int64_t v)
{
  return std::make_unique<cudf::numeric_scalar<int64_t>>(
    v, true, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

/// Inclusive `[lo, hi]` zone map -- AST-lowerable, so it reaches the reader merge.
std::shared_ptr<sirius_dynamic_zone_map_filter const> make_zone_map(int64_t lo, int64_t hi)
{
  std::vector<zone_map_entry> zones;
  zones.push_back({make_int64_scalar(lo), make_int64_scalar(hi)});
  return std::make_shared<sirius_dynamic_zone_map_filter>(std::move(zones));
}

/// Build `v < upper_bound` as a static DuckDB table filter on the second column.
duckdb::unique_ptr<duckdb::TableFilterSet> make_static_v_filter(int64_t upper_bound)
{
  auto filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  filters->PushFilter(
    duckdb::ColumnIndex(1),
    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                              duckdb::Value::BIGINT(upper_bound)));
  return filters;
}

/// One ingestible over the fixture, wired to a fresh channel, a refinement-slot publisher on
/// `v`'s output ordinal, and a local stats sink; splits pre-built one row group each.
struct gate_fixture {
  dynamic_filter_stats stats;
  std::shared_ptr<sirius_dynamic_filter_set> channel;
  sirius::op::dynamic_filter_refinement_publisher publisher;
  std::shared_ptr<scan::parquet_gpu_ingestible> ingestible;
  std::vector<std::unique_ptr<scan::scan_info>> splits;

  gate_fixture(std::filesystem::path const& path,
               duckdb::unique_ptr<duckdb::TableFilterSet> static_filters)
    : channel(std::make_shared<sirius_dynamic_filter_set>()),
      publisher(channel->register_refinement_slot(/*primary_ordinal=*/1))
  {
    auto info            = std::make_unique<scan::parquet_ingestible_table_info>();
    info->returned_types = {
      sirius::logical_type::make(sirius::type_id::BIGINT),
      sirius::logical_type::make(sirius::type_id::BIGINT),
    };
    info->resolved_file_paths = {path.string()};
    info->column_ids          = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
    info->names               = {"id", "v"};
    info->table_filters       = std::move(static_filters);
    info->scan_output_arity   = info->returned_types.size();
    // One row group per split: the coalescer's byte cap seals the running slice as soon as a
    // second row group arrives, so every split is exactly one row group.
    info->approximate_batch_size = 1;
    info->sirius_dynamic_filters = channel;
    info->stats                  = &stats;

    ingestible = scan::make_ingestible(std::move(info));

    auto ioctx     = std::make_shared<sirius::io::kvikio_context>();
    auto coalescer = ingestible->create_batch_coalescer();
    while (!ingestible->has_processed_all_metadata()) {
      auto task = ingestible->next_split_provider(
        [ioctx](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return ioctx; });
      REQUIRE(task);
      auto file = task();
      REQUIRE(file);
      for (auto& split : coalescer->push(std::move(file))) {
        splits.push_back(std::move(split));
      }
    }
    for (auto& split : coalescer->flush()) {
      splits.push_back(std::move(split));
    }
  }

  void publish(std::uint64_t revision, int64_t lo, int64_t hi)
  {
    REQUIRE(publisher.publish(revision, make_zone_map(lo, hi)) ==
            refinement_publish_result::ACCEPTED);
  }

  /// Materialize one split; returns the decoded row count.
  int64_t materialize(std::size_t split_idx)
  {
    auto result = ingestible->materialize_metadata_to_table(
      *splits.at(split_idx), *env().gpu_space, sirius::test::operator_utils::default_stream());
    return result.table.view().num_rows();
  }

  [[nodiscard]] std::size_t split_row_group_count(std::size_t split_idx) const
  {
    auto const& split = static_cast<scan::parquet_split_info const&>(*splits.at(split_idx));
    std::size_t count = 0;
    for (auto const& slice : split.rg_slices) {
      count += slice.row_group_indices.size();
    }
    return count;
  }

  [[nodiscard]] reader_pruning_gate::state gate_state() const
  {
    return ingestible->reader_gate().current_state();
  }
};

/// A bound every band satisfies, so merging it can never prune a row group.
constexpr int64_t LOOSE_LO = -1;
constexpr int64_t LOOSE_HI = GROUP_COUNT * BAND_STRIDE;

std::filesystem::path fixture_dir()
{
  return std::filesystem::temp_directory_path() /
         ("pgi_reader_gate_pruning." + std::to_string(::getpid()));
}

}  // namespace

TEST_CASE("reader gate premise: the reader's considered count equals the split's row groups",
          "[scan][parquet][dynamic_filter][reader_gate]")
{
  auto const path = write_banded_parquet(fixture_dir());
  gate_fixture f(path, /*static_filters=*/nullptr);
  REQUIRE(f.splits.size() == static_cast<std::size_t>(GROUP_COUNT));

  f.publish(1, LOOSE_LO, LOOSE_HI);
  auto const before = f.stats.snapshot();
  auto const rows   = f.materialize(0);
  auto const after  = f.stats.snapshot();

  // The whole WI-0b signal rests on this accounting: under set_row_groups the reader reports
  // the split's own row groups as its input count, and engages the stats filter when a filter
  // ran. If either half fails, stop and report -- do not substitute a different signal.
  REQUIRE(after.reader_gate_measurements - before.reader_gate_measurements == 1);
  REQUIRE(after.reader_gate_row_groups_considered - before.reader_gate_row_groups_considered ==
          f.split_row_group_count(0));
  REQUIRE(rows == ROWS_PER_GROUP);

  std::filesystem::remove_all(fixture_dir());
}

TEST_CASE("reader gate: a prunable split activates the gate",
          "[scan][parquet][dynamic_filter][reader_gate]")
{
  auto const path = write_banded_parquet(fixture_dir());
  gate_fixture f(path, /*static_filters=*/nullptr);

  // [0, 5000] keeps band 0 only; band 3 ([30000, 32047]) lies entirely above it.
  f.publish(1, 0, 5000);
  auto const before = f.stats.snapshot();
  auto const rows   = f.materialize(3);
  auto const after  = f.stats.snapshot();

  REQUIRE(after.reader_gate_row_groups_pruned - before.reader_gate_row_groups_pruned ==
          f.split_row_group_count(3));
  REQUIRE(rows == 0);
  REQUIRE(f.gate_state() == reader_pruning_gate::state::active);

  // Success is terminal: a later split the bound keeps whole still merges.
  auto const kept_rows = f.materialize(0);
  auto const last      = f.stats.snapshot();
  REQUIRE(kept_rows == ROWS_PER_GROUP);
  REQUIRE(last.reader_gate_measurements - after.reader_gate_measurements == 1);
  REQUIRE(last.reader_gate_merges_skipped == 0);

  std::filesystem::remove_all(fixture_dir());
}

TEST_CASE("reader gate: barren splits disable the merge and later splits skip it",
          "[scan][parquet][dynamic_filter][reader_gate]")
{
  auto const path = write_banded_parquet(fixture_dir());
  gate_fixture f(path, /*static_filters=*/nullptr);

  f.publish(1, LOOSE_LO, LOOSE_HI);
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(f.materialize(i) == ROWS_PER_GROUP);
  }
  REQUIRE(f.stats.snapshot().reader_gate_disabled == 0);
  REQUIRE(f.materialize(3) == ROWS_PER_GROUP);
  REQUIRE(f.stats.snapshot().reader_gate_disabled == 1);
  REQUIRE(f.gate_state() == reader_pruning_gate::state::disabled);

  // The fifth split skips the merge entirely -- and skipping is behavior-preserving: with no
  // static filter and a gated-off dynamic merge, the split decodes every row it holds.
  auto const before = f.stats.snapshot();
  REQUIRE(f.materialize(4) == ROWS_PER_GROUP);
  auto const after = f.stats.snapshot();
  REQUIRE(after.reader_gate_merges_skipped - before.reader_gate_merges_skipped == 1);
  REQUIRE(after.reader_gate_measurements == before.reader_gate_measurements);

  std::filesystem::remove_all(fixture_dir());
}

TEST_CASE("reader gate: tightened revisions re-arm on the 1, 2, 4 schedule",
          "[scan][parquet][dynamic_filter][reader_gate]")
{
  auto const path = write_banded_parquet(fixture_dir());
  gate_fixture f(path, /*static_filters=*/nullptr);

  // Disable at generation 1 (four barren merged splits), so the re-arm point is generation 2.
  f.publish(1, LOOSE_LO, LOOSE_HI);
  for (std::size_t i = 0; i < 4; ++i) {
    f.materialize(i);
  }
  REQUIRE(f.gate_state() == reader_pruning_gate::state::disabled);

  // Gap 1: one tighter-but-still-loose revision reaches generation 2 -> one re-measurement,
  // barren, so the gate re-disables with the gap doubled to 2 (re-arm at generation 4).
  f.publish(2, LOOSE_LO, LOOSE_HI - 1);
  auto s = f.stats.snapshot();
  f.materialize(4);
  auto after_rearm = f.stats.snapshot();
  REQUIRE(after_rearm.reader_gate_rearmed - s.reader_gate_rearmed == 1);
  REQUIRE(after_rearm.reader_gate_disabled - s.reader_gate_disabled == 1);
  REQUIRE(f.gate_state() == reader_pruning_gate::state::disabled);

  // Still generation 2 < 4: the next split skips, measuring nothing.
  f.materialize(5);
  auto after_skip = f.stats.snapshot();
  REQUIRE(after_skip.reader_gate_merges_skipped - after_rearm.reader_gate_merges_skipped == 1);
  REQUIRE(after_skip.reader_gate_measurements == after_rearm.reader_gate_measurements);

  // Gap 2, first half: one more revision reaches only generation 3 < 4, so the split still
  // skips -- this is what separates the doubled gap from a stuck gap of 1, which would already
  // re-measure here.
  f.publish(3, LOOSE_LO, LOOSE_HI - 2);
  f.materialize(6);
  auto after_gen3 = f.stats.snapshot();
  REQUIRE(after_gen3.reader_gate_merges_skipped - after_skip.reader_gate_merges_skipped == 1);
  REQUIRE(after_gen3.reader_gate_measurements == after_skip.reader_gate_measurements);

  // Gap 2, second half: the next revision reaches generation 4 -> the second re-measurement.
  f.publish(4, LOOSE_LO, LOOSE_HI - 3);
  f.materialize(7);
  auto after_second = f.stats.snapshot();
  REQUIRE(after_second.reader_gate_rearmed - after_gen3.reader_gate_rearmed == 1);
  REQUIRE(after_second.reader_gate_rearmed == 2);

  std::filesystem::remove_all(fixture_dir());
}

TEST_CASE("reader gate: a re-armed measurement that prunes re-activates",
          "[scan][parquet][dynamic_filter][reader_gate]")
{
  auto const path = write_banded_parquet(fixture_dir());
  gate_fixture f(path, /*static_filters=*/nullptr);

  f.publish(1, LOOSE_LO, LOOSE_HI);
  for (std::size_t i = 0; i < 4; ++i) {
    f.materialize(i);
  }
  REQUIRE(f.gate_state() == reader_pruning_gate::state::disabled);

  // A tight replacement reaches the re-arm generation, and its re-measurement prunes: band 4
  // ([40000, 42047]) lies entirely outside [0, 100].
  f.publish(2, 0, 100);
  REQUIRE(f.materialize(4) == 0);
  REQUIRE(f.gate_state() == reader_pruning_gate::state::active);
  REQUIRE(f.stats.snapshot().reader_gate_rearmed == 1);

  // Active is terminal: every later split merges, none skips.
  auto const before = f.stats.snapshot();
  f.materialize(5);
  f.materialize(6);
  auto const after = f.stats.snapshot();
  REQUIRE(after.reader_gate_measurements - before.reader_gate_measurements == 2);
  REQUIRE(after.reader_gate_merges_skipped == before.reader_gate_merges_skipped);

  std::filesystem::remove_all(fixture_dir());
}

TEST_CASE("reader gate: static WHERE pruning does not train the gate",
          "[scan][parquet][dynamic_filter][reader_gate]")
{
  auto const path = write_banded_parquet(fixture_dir());
  // `v < 60000` stats-prunes bands 6..15 at metadata time, so only 6 splits survive to
  // materialize; the merged static+dynamic reader AST re-evaluates over exactly those.
  gate_fixture f(path, make_static_v_filter(6 * BAND_STRIDE));
  REQUIRE(f.splits.size() == 6);

  f.publish(1, LOOSE_LO, LOOSE_HI);
  for (std::size_t i = 0; i < 4; ++i) {
    // Surviving bands satisfy the static predicate whole, so no rows are dropped either.
    REQUIRE(f.materialize(i) == ROWS_PER_GROUP);
  }

  // Deterministic static re-pruning contributes zero to the signal: the survivors were already
  // pruned against the static conjuncts at metadata time, so four merged splits are four barren
  // samples and the gate disables -- static behavior can neither keep it measuring forever nor
  // disable it spuriously before the evidence budget is spent.
  auto const snap = f.stats.snapshot();
  REQUIRE(snap.reader_gate_measurements == 4);
  REQUIRE(snap.reader_gate_row_groups_pruned == 0);
  REQUIRE(snap.reader_gate_disabled == 1);
  REQUIRE(f.gate_state() == reader_pruning_gate::state::disabled);

  std::filesystem::remove_all(fixture_dir());
}

TEST_CASE("reader gate: the zero-row fallback split is not evidence and not a skip",
          "[scan][parquet][dynamic_filter][reader_gate]")
{
  auto const path = write_banded_parquet(fixture_dir());
  // `v < -1` prunes every row group at metadata time; the coalescer's flush fallback emits one
  // zero-row-group split so the scan still creates a task.
  gate_fixture f(path, make_static_v_filter(-1));
  REQUIRE(f.splits.size() == 1);
  REQUIRE(f.split_row_group_count(0) == 0);

  f.publish(1, LOOSE_LO, LOOSE_HI);
  auto const before = f.stats.snapshot();
  auto result       = f.ingestible->materialize_metadata_to_table(
    *f.splits[0], *env().gpu_space, sirius::test::operator_utils::default_stream());

  // Schema-correct empty table, and the dynamic block was never entered: zero rows need no
  // reader filter, and a zero-row read is no pruning evidence.
  REQUIRE(result.table.view().num_rows() == 0);
  REQUIRE(result.table.view().num_columns() == 2);
  auto const after = f.stats.snapshot();
  REQUIRE(after.reader_gate_row_groups_considered == before.reader_gate_row_groups_considered);
  REQUIRE(after.reader_gate_row_groups_pruned == before.reader_gate_row_groups_pruned);
  REQUIRE(after.reader_gate_measurements == before.reader_gate_measurements);
  REQUIRE(after.reader_gate_disabled == before.reader_gate_disabled);
  REQUIRE(after.reader_gate_rearmed == before.reader_gate_rearmed);
  REQUIRE(after.reader_gate_merges_skipped == before.reader_gate_merges_skipped);
  REQUIRE(f.gate_state() == reader_pruning_gate::state::measuring);

  std::filesystem::remove_all(fixture_dir());
}
