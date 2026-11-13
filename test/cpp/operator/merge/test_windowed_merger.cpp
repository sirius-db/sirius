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

#include "catch.hpp"
#include "op/merge/windowed_merge.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/row_range_read.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op::merge;
using namespace sirius::test::merge_utils;
using sirius::test::operator_utils::get_default_gpu_space;
using sirius::test::operator_utils::get_resource_ref;

namespace {

constexpr std::size_t kUnlimitedOutput = std::numeric_limits<std::size_t>::max();

merge_key_spec ascending_key(cudf::size_type col = 0)
{
  return {.key_cols        = {col},
          .column_order    = {cudf::order::ASCENDING},
          .null_precedence = {cudf::null_order::AFTER}};
}

/// An unsorted (key INT64 in [0, key_range), id INT64) table; ids are unique from @p first_id on.
std::unique_ptr<cudf::table> random_rows(std::size_t rows,
                                         std::int64_t key_range,
                                         std::int64_t first_id,
                                         std::uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_int_distribution<std::int64_t> dist(0, key_range - 1);
  std::vector<std::int64_t> keys(rows);
  std::vector<std::int64_t> ids(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    keys[i] = dist(gen);
    ids[i]  = first_id + static_cast<std::int64_t>(i);
  }
  return make_table(make_column(keys, cudf::type_id::INT64),
                    make_column(ids, cudf::type_id::INT64));
}

const std::vector<std::size_t> kTwoCols{0, 1};
const std::vector<cudf::data_type> kTwoInt64{cudf::data_type{cudf::type_id::INT64},
                                             cudf::data_type{cudf::type_id::INT64}};

/// Everything a windowed_merger over GPU runs of (key, id) rows needs to outlive it.
struct merge_fixture {
  cucascade::memory::memory_space& space = *get_default_gpu_space();
  cucascade::shared_data_repository repository;
  cucascade::row_range_reader_registry readers;  // empty: GPU-tier inputs need no reader
  std::vector<test_run> runs;
  std::vector<merge_input_cursor> cursors;
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  rmm::device_async_resource_ref mr() { return get_resource_ref(space); }

  void add_run(const cudf::table_view& unsorted,
               const merge_key_spec& keys,
               std::size_t num_batches = 1)
  {
    runs.push_back(make_sorted_gpu_run(repository, space, unsorted, keys, num_batches));
  }

  windowed_merger make_merger(const merge_key_spec& keys,
                              window_config cfg,
                              bounded_merge_fn merge              = reference_bounded_merge,
                              std::vector<std::size_t> fetch_cols = kTwoCols,
                              std::vector<cudf::data_type> types  = kTwoInt64)
  {
    if (cursors.empty()) { cursors = make_cursors(runs); }
    return windowed_merger(run_batches(runs),
                           cursors,
                           std::move(fetch_cols),
                           std::move(types),
                           keys,
                           cfg,
                           readers,
                           space,
                           std::move(merge));
  }

  /// True when every batch can be locked exclusively, i.e. no read lock is left behind.
  bool all_batches_unlocked() const
  {
    for (const auto& run : runs) {
      for (const auto& batch : run.batches) {
        if (!batch->try_to_mutable().has_value()) { return false; }
      }
    }
    return true;
  }
};

}  // namespace

TEST_CASE("windowed_merger merges 3 GPU runs over several run() calls", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(5000, 100000, 0, 1)->view(), keys);
  f.add_run(random_rows(3000, 100000, 10000, 2)->view(), keys);
  f.add_run(random_rows(4200, 100000, 20000, 3)->view(), keys);

  // 16 bytes per row: a 1000-row round is 16000 bytes, so each run() is about 3 rounds.
  auto merger = f.make_merger(keys, {.window_rows = 1000, .output_target_bytes = 40000});
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);

  CHECK(collected.chunks.size() > 1);
  for (const auto& chunk : collected.chunks) {
    CHECK(chunk->num_rows() > 0);
    // A call stops with the round that reaches 2500 rows, and a round is at most 1000 rows.
    CHECK(chunk->num_rows() < 3500);
  }
  auto expected = expected_full_sort(f.runs, keys);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(all_exhausted(f.cursors));
  CHECK(stats.rows_out == 12200);
  CHECK(stats.rounds >= 13);
  CHECK(stats.refill_rows == 0);
  CHECK(stats.refill_bytes == 0);
  CHECK(stats.fallback_conversions == 0);
  CHECK(f.all_batches_unlocked());

  // Every input is exhausted: another call returns an empty table with the fetched schema.
  auto empty = merger.run(f.stream, f.mr(), &stats);
  CHECK(empty->num_rows() == 0);
  CHECK(empty->num_columns() == 2);
  CHECK(stats.rows_out == 12200);
}

TEST_CASE("windowed_merger windows never span the batches of a run", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  // Run 0 is three batches of 700, 1500 and 300 rows (with an empty batch in between).
  {
    auto sorted = stable_sort(random_rows(2500, 5000, 0, 11)->view(), keys);
    f.runs.push_back(make_gpu_run(f.repository, f.space, sorted->view(), {700, 0, 1500, 300}));
  }
  f.add_run(random_rows(2000, 5000, 10000, 12)->view(), keys, 3);
  f.add_run(random_rows(900, 5000, 20000, 13)->view(), keys);

  std::vector<std::vector<cudf::size_type>> window_rows;  // per round, the rows of each window
  std::vector<std::vector<std::uint8_t>> window_truncated;
  auto recording_merge = [&](std::span<const cudf::table_view> windows,
                             const merge_key_spec& key_spec,
                             std::span<const std::uint8_t> truncated,
                             cudf::size_type max_rows,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) {
    auto& rows = window_rows.emplace_back();
    for (const auto& window : windows) {
      rows.push_back(window.num_rows());
    }
    window_truncated.emplace_back(truncated.begin(), truncated.end());
    return reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };

  auto merger = f.make_merger(
    keys, {.window_rows = 1000, .output_target_bytes = kUnlimitedOutput}, recording_merge);
  merge_round_stats stats;
  auto out = merger.run(f.stream, f.mr(), &stats);

  // Round 1: run 0's window stops at its first batch (700 rows) although n = 1000; run 1's at its
  // first batch (666 rows); run 2 fits whole and is not truncated.
  REQUIRE(window_rows.size() >= 2);
  CHECK(window_rows[0] == std::vector<cudf::size_type>{700, 666, 900});
  CHECK(window_truncated[0] == std::vector<std::uint8_t>{1, 1, 0});
  for (const auto& rows : window_rows) {
    for (const auto window : rows) {
      CHECK(window > 0);
      CHECK(window <= 1000);
    }
  }

  auto expected = expected_full_sort(f.runs, keys);
  CHECK(tables_equal(out->view(), expected->view()));
  CHECK(all_exhausted(f.cursors));
  CHECK(f.cursors[0].batch == 4);
  CHECK(f.cursors[0].row == 0);
  CHECK(stats.rows_out == 5400);
  CHECK(stats.rounds == window_rows.size());
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger leaves cursors and locks untouched when the merge throws",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(3000, 100000, 0, 21)->view(), keys, 2);
  f.add_run(random_rows(2500, 100000, 10000, 22)->view(), keys);
  f.add_run(random_rows(2000, 100000, 20000, 23)->view(), keys);

  int calls          = 0;
  auto failing_merge = [&](std::span<const cudf::table_view> windows,
                           const merge_key_spec& key_spec,
                           std::span<const std::uint8_t> truncated,
                           cudf::size_type max_rows,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) {
    if (++calls == 2) { throw std::runtime_error("injected merge failure"); }
    return reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };

  const window_config cfg{.window_rows = 1000, .output_target_bytes = kUnlimitedOutput};
  merge_round_stats stats;
  {
    auto merger = f.make_merger(keys, cfg, failing_merge);
    REQUIRE_THROWS_WITH(merger.run(f.stream, f.mr(), &stats), "injected merge failure");
    CHECK(calls == 2);

    // Round 1 succeeded, but nothing of it is committed.
    for (const auto& cursor : f.cursors) {
      CHECK(cursor.batch == 0);
      CHECK(cursor.row == 0);
    }
    CHECK(stats.rounds == 0);
    CHECK(stats.rows_out == 0);
    // The read locks are gone although the merger is still alive.
    CHECK(f.all_batches_unlocked());

    // The same merger replays from the unchanged cursors (the injected failure is one-shot).
    auto out      = merger.run(f.stream, f.mr(), &stats);
    auto expected = expected_full_sort(f.runs, keys);
    CHECK(tables_equal(out->view(), expected->view()));
    CHECK(all_exhausted(f.cursors));
    CHECK(stats.rows_out == 7500);
  }
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger stops at output_target_bytes and continues exactly", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(2600, 1000, 0, 31)->view(), keys, 2);
  f.add_run(random_rows(1800, 1000, 10000, 32)->view(), keys);
  f.add_run(random_rows(2200, 1000, 20000, 33)->view(), keys, 3);
  auto expected = expected_full_sort(f.runs, keys);

  SECTION("a target below one round's bytes gives one round per run()")
  {
    auto merger = f.make_merger(keys, {.window_rows = 1000, .output_target_bytes = 1});
    merge_round_stats stats;
    auto first = merger.run(f.stream, f.mr(), &stats);
    CHECK(stats.rounds == 1);
    CHECK(first->num_rows() > 0);
    CHECK(first->num_rows() <= 1000);
    CHECK_FALSE(all_exhausted(f.cursors));

    // The cursors moved by exactly the rows emitted.
    std::size_t remaining = 0;
    for (const auto& cursor : f.cursors) {
      remaining += cursor.remaining();
    }
    CHECK(remaining + static_cast<std::size_t>(first->num_rows()) == 6600);
    CHECK(f.all_batches_unlocked());

    auto second = merger.run(f.stream, f.mr(), &stats);
    CHECK(stats.rounds == 2);
    const auto head_rows = first->num_rows() + second->num_rows();
    std::vector<cudf::table_view> head{first->view(), second->view()};
    auto head_table = cudf::concatenate(head);
    CHECK(tables_equal(head_table->view(), cudf::slice(expected->view(), {0, head_rows}).front()));

    // A new merger over the same cursors (the next task) picks up where this one stopped.
    auto next = f.make_merger(keys, {.window_rows = 500, .output_target_bytes = 20000});
    auto rest = collect_full_sort(next, f.cursors, f.stream, f.mr(), &stats);
    CHECK(rest.chunks.size() > 1);
    CHECK(tables_equal(rest.table->view(),
                       cudf::slice(expected->view(), {head_rows, expected->num_rows()}).front()));
    CHECK(stats.rows_out == 6600);
  }

  SECTION("a run() stops with the round that reaches the target")
  {
    // 16 bytes per row: 32000 bytes are 2000 rows, and a round is at most 1000 rows.
    auto merger = f.make_merger(keys, {.window_rows = 1000, .output_target_bytes = 32000});
    merge_round_stats stats;
    auto first = merger.run(f.stream, f.mr(), &stats);
    CHECK(stats.rounds >= 2);
    CHECK(first->num_rows() >= 2000);
    CHECK(first->num_rows() < 3000);
    CHECK(
      tables_equal(first->view(), cudf::slice(expected->view(), {0, first->num_rows()}).front()));
  }

  SECTION("an unlimited target merges everything in one run()")
  {
    auto merger =
      f.make_merger(keys, {.window_rows = 1000, .output_target_bytes = kUnlimitedOutput});
    auto out = merger.run(f.stream, f.mr());
    CHECK(tables_equal(out->view(), expected->view()));
    CHECK(all_exhausted(f.cursors));
  }
}

TEST_CASE("windowed_merger handles heavy ties", "[windowed_merger]")
{
  // Descending keys with NULLs first, only 3 distinct key values over 4 runs: almost every row
  // ties across runs and across windows, so the run-index tie-break decides the order.
  const merge_key_spec keys{.key_cols        = {0},
                            .column_order    = {cudf::order::DESCENDING},
                            .null_precedence = {cudf::null_order::AFTER}};
  merge_fixture f;
  std::mt19937 gen(41);
  std::int64_t next_id = 0;
  for (const std::size_t rows : {1500UL, 2300UL, 800UL, 1900UL}) {
    std::vector<std::int64_t> key_values(rows);
    std::vector<bool> valids(rows);
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      key_values[i] = static_cast<std::int64_t>(gen() % 3);
      valids[i]     = gen() % 10 != 0;
      ids[i]        = next_id++;
    }
    auto table = make_table(make_column(key_values, cudf::type_id::INT64, valids),
                            make_column(ids, cudf::type_id::INT64));
    f.add_run(table->view(), keys, 2);
  }
  // One run of a single repeated key, longer than several windows.
  {
    auto table = make_table(make_column(std::vector<std::int64_t>(1200, 1), cudf::type_id::INT64),
                            make_column(std::vector<std::int64_t>(1200, -1), cudf::type_id::INT64));
    f.add_run(table->view(), keys);
  }

  auto merger = f.make_merger(keys, {.window_rows = 256, .output_target_bytes = 10000});
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);

  auto expected = expected_full_sort(f.runs, keys);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(stats.rows_out == 7700);
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger merges on STRING keys", "[windowed_merger]")
{
  // (STRING key, INT64 id, STRING payload); the merger fetches (key, payload, id).
  const merge_key_spec keys{.key_cols        = {0, 2},
                            .column_order    = {cudf::order::ASCENDING, cudf::order::DESCENDING},
                            .null_precedence = {cudf::null_order::AFTER, cudf::null_order::AFTER}};
  const std::vector<std::size_t> fetch_cols{0, 2, 1};
  const std::vector<cudf::data_type> fetch_types{cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::INT64}};
  const std::vector<std::string> words{"", "a", "ab", "abc", "b", "kiwi", "kiwi fruit", "zebra"};

  merge_fixture f;
  std::mt19937 gen(51);
  std::int64_t next_id = 0;
  std::vector<std::unique_ptr<cudf::table>> fetched_runs;
  for (const std::size_t rows : {1300UL, 700UL, 1100UL}) {
    std::vector<std::string> key_values(rows);
    std::vector<std::string> payload(rows);
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      key_values[i] = words[gen() % words.size()];
      ids[i]        = static_cast<std::int64_t>(gen() % 50);
      payload[i]    = "row-" + std::to_string(next_id++);
    }
    auto table = make_table(make_string_column(key_values),
                            make_column(ids, cudf::type_id::INT64),
                            make_string_column(payload));
    // The runs are sorted on the source columns (key, id): source column 1 is fetched column 2.
    const merge_key_spec source_keys{.key_cols        = {0, 1},
                                     .column_order    = keys.column_order,
                                     .null_precedence = keys.null_precedence};
    f.add_run(table->view(), source_keys, 2);
    fetched_runs.push_back(
      std::make_unique<cudf::table>(f.runs.back().rows->view().select({0, 2, 1})));
  }

  auto merger = f.make_merger(keys,
                              {.window_rows = 200, .output_target_bytes = 8000},
                              reference_bounded_merge,
                              fetch_cols,
                              fetch_types);
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);
  CHECK(collected.chunks.size() > 1);

  std::vector<cudf::table_view> fetched_views;
  for (const auto& run : fetched_runs) {
    fetched_views.push_back(run->view());
  }
  auto expected = expected_full_sort(std::span<const cudf::table_view>(fetched_views), keys);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(stats.rows_out == 3100);
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger rejects inconsistent inputs", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(100, 10, 0, 61)->view(), keys);
  f.cursors = make_cursors(f.runs);
  const window_config cfg{.window_rows = 10, .output_target_bytes = 1};

  SECTION("runs and cursors differ in size")
  {
    std::vector<merge_input_cursor> none;
    CHECK_THROWS_AS(windowed_merger(run_batches(f.runs),
                                    none,
                                    kTwoCols,
                                    kTwoInt64,
                                    keys,
                                    cfg,
                                    f.readers,
                                    f.space,
                                    reference_bounded_merge),
                    std::invalid_argument);
  }
  SECTION("window_rows is zero")
  {
    CHECK_THROWS_AS(f.make_merger(keys, {.window_rows = 0, .output_target_bytes = 1}),
                    std::invalid_argument);
  }
  SECTION("a merge that makes no progress is an internal error, not a hang")
  {
    auto stuck_merge = [](std::span<const cudf::table_view> windows,
                          const merge_key_spec&,
                          std::span<const std::uint8_t>,
                          cudf::size_type,
                          rmm::cuda_stream_view,
                          rmm::device_async_resource_ref) {
      bounded_merge_result result;
      result.output = cudf::empty_like(windows.front());
      result.consumed.assign(windows.size(), 0);
      return result;
    };
    auto merger = f.make_merger(keys, cfg, stuck_merge);
    CHECK_THROWS_AS(merger.run(f.stream, f.mr()), sirius::internal_exception);
    CHECK(f.cursors[0].row == 0);
    CHECK(f.all_batches_unlocked());
  }
}

TEST_CASE("windowed_merger starts from cursors left by an earlier task", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(3000, 50000, 0, 71)->view(), keys, 3);      // batches of 1000 rows
  f.add_run(random_rows(1200, 50000, 10000, 72)->view(), keys, 2);  // batches of 600 rows
  f.add_run(random_rows(800, 50000, 20000, 73)->view(), keys);
  f.add_run(random_rows(500, 50000, 30000, 74)->view(), keys);
  f.cursors = make_cursors(f.runs);
  // Run 0 starts mid-way through its second batch; run 1 rests at the end of its first batch (not
  // normalized); run 2 is already exhausted; run 3 starts at its last row.
  f.cursors[0].batch = 1;
  f.cursors[0].row   = 250;
  f.cursors[1].batch = 0;
  f.cursors[1].row   = 600;
  f.cursors[2].batch = 1;
  f.cursors[2].row   = 0;
  f.cursors[3].batch = 0;
  f.cursors[3].row   = 499;

  std::vector<std::unique_ptr<cudf::table>> tails;
  std::vector<cudf::table_view> tail_views;
  const std::vector<cudf::size_type> starts{1250, 600, 800, 499};
  for (std::size_t i = 0; i < f.runs.size(); ++i) {
    tails.push_back(std::make_unique<cudf::table>(
      cudf::slice(f.runs[i].rows->view(), {starts[i], f.runs[i].rows->num_rows()}).front()));
    tail_views.push_back(tails.back()->view());
  }
  auto expected = expected_full_sort(std::span<const cudf::table_view>(tail_views), keys);
  REQUIRE(expected->num_rows() == 1750 + 600 + 1);

  std::size_t windows_seen = 0;
  auto checking_merge      = [&](std::span<const cudf::table_view> windows,
                            const merge_key_spec& key_spec,
                            std::span<const std::uint8_t> truncated,
                            cudf::size_type max_rows,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) {
    // The exhausted run never contributes a window, and no window is empty.
    CHECK(windows.size() <= 3);
    for (const auto& window : windows) {
      CHECK(window.num_rows() > 0);
    }
    windows_seen = std::max(windows_seen, windows.size());
    return reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };
  auto merger = f.make_merger(
    keys, {.window_rows = 300, .output_target_bytes = kUnlimitedOutput}, checking_merge);
  merge_round_stats stats;
  auto out = merger.run(f.stream, f.mr(), &stats);

  CHECK(windows_seen == 3);
  CHECK(tables_equal(out->view(), expected->view()));
  CHECK(all_exhausted(f.cursors));
  CHECK(stats.rows_out == 2351);
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger merges a single run", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(2345, 1000, 0, 81)->view(), keys, 3);
  auto merger = f.make_merger(keys, {.window_rows = 400, .output_target_bytes = 9000});
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);
  CHECK(tables_equal(collected.table->view(), f.runs[0].rows->view()));
  CHECK(stats.rows_out == 2345);
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger holds the read locks of GPU batches during a round", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(1000, 1000, 0, 91)->view(), keys, 2);
  f.add_run(random_rows(600, 1000, 10000, 92)->view(), keys);

  int rounds              = 0;
  auto lock_probing_merge = [&](std::span<const cudf::table_view> windows,
                                const merge_key_spec& key_spec,
                                std::span<const std::uint8_t> truncated,
                                cudf::size_type max_rows,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr) {
    if (rounds++ == 0) {
      // The first batch of each run is windowed, so it is read-locked; run 0's second batch is
      // not touched yet.
      CHECK_FALSE(f.runs[0].batches[0]->try_to_mutable().has_value());
      CHECK_FALSE(f.runs[1].batches[0]->try_to_mutable().has_value());
      CHECK(f.runs[0].batches[1]->try_to_mutable().has_value());
    }
    return reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };
  auto merger = f.make_merger(
    keys, {.window_rows = 128, .output_target_bytes = kUnlimitedOutput}, lock_probing_merge);
  auto out = merger.run(f.stream, f.mr());
  CHECK(rounds > 1);
  CHECK(tables_equal(out->view(), expected_full_sort(f.runs, keys)->view()));
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger is movable and accepts null stats", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(1500, 300, 0, 101)->view(), keys);
  f.add_run(random_rows(1700, 300, 10000, 102)->view(), keys, 2);
  auto expected = expected_full_sort(f.runs, keys);

  auto original = f.make_merger(keys, {.window_rows = 256, .output_target_bytes = 1});
  auto first    = original.run(f.stream, f.mr(), nullptr);
  REQUIRE(first->num_rows() > 0);

  windowed_merger moved(std::move(original));
  windowed_merger assigned = f.make_merger(keys, {.window_rows = 1, .output_target_bytes = 1});
  assigned                 = std::move(moved);
  // The new merger was built over the same cursors; the moved-in state uses 256-row windows.
  auto rest = assigned.run(f.stream, f.mr());
  CHECK(rest->num_rows() > 1);
  while (!all_exhausted(f.cursors)) {
    auto more = assigned.run(f.stream, f.mr());
    std::vector<cudf::table_view> views{rest->view(), more->view()};
    rest = cudf::concatenate(views);
  }
  std::vector<cudf::table_view> all{first->view(), rest->view()};
  CHECK(tables_equal(cudf::concatenate(all)->view(), expected->view()));
}

TEST_CASE("windowed_merger rejects a bounded merge that breaks its contract", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(500, 100, 0, 111)->view(), keys);
  f.add_run(random_rows(400, 100, 1000, 112)->view(), keys);
  const window_config cfg{.window_rows = 100, .output_target_bytes = kUnlimitedOutput};

  // Calls the reference merge, then lets @p tamper corrupt its result.
  auto tampering = [](std::function<void(bounded_merge_result&)> tamper) {
    return [tamper = std::move(tamper)](std::span<const cudf::table_view> windows,
                                        const merge_key_spec& key_spec,
                                        std::span<const std::uint8_t> truncated,
                                        cudf::size_type max_rows,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr) {
      auto result = reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
      tamper(result);
      return result;
    };
  };

  bounded_merge_fn bad;
  SECTION("too few consumed counts")
  {
    bad = tampering([](bounded_merge_result& r) { r.consumed.pop_back(); });
  }
  SECTION("more consumed than the window holds")
  {
    bad = tampering([](bounded_merge_result& r) { r.consumed[0] = 101; });
  }
  SECTION("a negative consumed count")
  {
    bad = tampering([](bounded_merge_result& r) {
      r.consumed[1] += r.consumed[0] + 1;
      r.consumed[0] = -1;
    });
  }
  SECTION("consumed rows differ from the emitted rows")
  {
    bad = tampering([](bounded_merge_result& r) {
      if (r.consumed[0] > 0) {
        --r.consumed[0];
      } else {
        --r.consumed[1];
      }
    });
  }
  SECTION("no output table")
  {
    bad = tampering([](bounded_merge_result& r) { r.output.reset(); });
  }

  auto merger = f.make_merger(keys, cfg, bad);
  merge_round_stats stats;
  CHECK_THROWS_AS(merger.run(f.stream, f.mr(), &stats), sirius::internal_exception);
  for (const auto& cursor : f.cursors) {
    CHECK(cursor.batch == 0);
    CHECK(cursor.row == 0);
  }
  CHECK(stats.rounds == 0);
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger rejects a batch whose rows differ from its run", "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(300, 100, 0, 121)->view(), keys);
  f.add_run(random_rows(200, 100, 1000, 122)->view(), keys);
  f.cursors = make_cursors(f.runs);
  // The cursor believes run 1 has 250 rows in its only batch; the batch holds 200.
  f.cursors[1].run.batch_rows[0] = 250;
  auto merger = f.make_merger(keys, {.window_rows = 64, .output_target_bytes = kUnlimitedOutput});
  CHECK_THROWS_AS(merger.run(f.stream, f.mr()), sirius::internal_exception);
  CHECK(f.cursors[0].row == 0);
  CHECK(f.cursors[1].row == 0);
  CHECK(f.all_batches_unlocked());
}
