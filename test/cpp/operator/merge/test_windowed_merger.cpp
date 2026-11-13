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
#include "compression/compressed_representation.hpp"
#include "compression/device_compressed_blob.hpp"
#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/detail/run_fallback.hpp"
#include "op/merge/windowed_merge.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_stream.hpp>

#include <api/simpatico_codegen.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <data/sirius_converter_registry.hpp>
#include <data/sirius_row_range_reader_registry.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <set>
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
    // try_to_mutable also fails while a reader event is outstanding: let the stream drain, so
    // only a read lock that is still held makes this false.
    stream.synchronize();
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

// ---------------------------------------------------------------------------
// HOST-tier runs: window buffers
// ---------------------------------------------------------------------------

namespace {

/// Runs whose batches live in the HOST tier, so the merger reads them through window buffers.
/// `memory` is declared first, so the batches are destroyed before the memory manager.
struct host_merge_fixture {
  sirius::test::operator_utils::tiered_memory_fixture memory;
  cucascade::shared_data_repository repository;
  std::vector<test_run> runs;
  std::vector<merge_input_cursor> cursors;
  rmm::cuda_stream owned_stream;  // the converters need a real stream, not the default one
  rmm::cuda_stream_view stream{owned_stream.value()};

  host_merge_fixture()
  {
    REQUIRE(memory.setup());
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
  }

  rmm::device_async_resource_ref mr() { return get_resource_ref(*memory.gpu0); }

  void add_host_run(const cudf::table_view& unsorted,
                    const merge_key_spec& keys,
                    std::size_t num_batches = 1)
  {
    runs.push_back(make_sorted_gpu_run(repository, *memory.gpu0, unsorted, keys, num_batches));
    // The batches were copied on the default stream; the converter runs on this one.
    cudf::get_default_stream().sync();
    for (const auto& batch : runs.back().batches) {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(
        sirius::converter_registry::get(), memory.host0, stream);
      stream.synchronize();
      REQUIRE(mut.get_current_tier() == cucascade::memory::Tier::HOST);
    }
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
                           sirius::row_range_reader_registry::get(),
                           *memory.gpu0,
                           std::move(merge));
  }
};

}  // namespace

TEST_CASE("windowed_merger reads every HOST row once across run() calls", "[windowed_merger]")
{
  const auto keys = ascending_key();
  host_merge_fixture f;
  f.add_host_run(random_rows(2000, 100000, 0, 201)->view(), keys, 2);
  f.add_host_run(random_rows(1500, 100000, 10000, 202)->view(), keys);
  f.add_host_run(random_rows(1700, 100000, 20000, 203)->view(), keys, 3);
  auto expected = expected_full_sort(f.runs, keys);

  // One round per run(): the windows are kept by the merger from one call to the next.
  std::size_t rounds        = 0;
  std::size_t short_windows = 0;
  auto counting_merge       = [&](std::span<const cudf::table_view> windows,
                            const merge_key_spec& key_spec,
                            std::span<const std::uint8_t> truncated,
                            cudf::size_type max_rows,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) {
    ++rounds;
    for (std::size_t w = 0; w < windows.size(); ++w) {
      REQUIRE(windows[w].num_rows() > 0);
      REQUIRE(windows[w].num_rows() <= 512);
      // A window below half the window size is only passed when its batch has no more rows.
      if (windows[w].num_rows() < 256) { ++short_windows; }
    }
    return reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };
  auto merger = f.make_merger(keys, {.window_rows = 512, .output_target_bytes = 1}, counting_merge);
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);

  CHECK(collected.chunks.size() == rounds);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(all_exhausted(f.cursors));
  CHECK(stats.rows_out == 5200);
  // No row is read twice, neither between rounds nor between run() calls.
  CHECK(stats.refill_rows == 5200);
  CHECK(stats.refill_bytes == 5200 * 16);
  CHECK(stats.fallback_conversions == 0);
  // Short windows only appear at the end of the 6 batches, never every round.
  CHECK(short_windows < rounds);
}

TEST_CASE("windowed_merger drops its HOST windows when a run() fails or the cursors move",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  host_merge_fixture f;
  f.add_host_run(random_rows(1800, 1000, 0, 211)->view(), keys, 2);
  f.add_host_run(random_rows(1300, 1000, 10000, 212)->view(), keys);
  auto expected = expected_full_sort(f.runs, keys);

  int calls          = 0;
  int fail_at        = 3;
  auto failing_merge = [&](std::span<const cudf::table_view> windows,
                           const merge_key_spec& key_spec,
                           std::span<const std::uint8_t> truncated,
                           cudf::size_type max_rows,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) {
    if (++calls == fail_at) { throw std::runtime_error("injected merge failure"); }
    return reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };
  // 300-row rounds of 16-byte rows: each run() is two rounds.
  auto merger =
    f.make_merger(keys, {.window_rows = 300, .output_target_bytes = 9000}, failing_merge);
  merge_round_stats stats;

  auto first             = merger.run(f.stream, f.mr(), &stats);
  const auto first_rows  = first->num_rows();
  const auto after_first = f.cursors;

  // The second run() fails in its first round: the cursors stay where the first run() left them
  // and the replay continues exactly, although the buffers had moved on.
  REQUIRE_THROWS_WITH(merger.run(f.stream, f.mr(), &stats), "injected merge failure");
  for (std::size_t i = 0; i < f.cursors.size(); ++i) {
    CHECK(f.cursors[i].batch == after_first[i].batch);
    CHECK(f.cursors[i].row == after_first[i].row);
  }
  auto second = merger.run(f.stream, f.mr(), &stats);
  std::vector<cudf::table_view> head{first->view(), second->view()};
  auto head_table = cudf::concatenate(head);
  CHECK(tables_equal(head_table->view(),
                     cudf::slice(expected->view(), {0, first_rows + second->num_rows()}).front()));

  // The caller rewinds the cursors: the merger must not serve the rows it had buffered for the
  // old positions.
  for (auto& cursor : f.cursors) {
    cursor.batch = 0;
    cursor.row   = 0;
  }
  fail_at        = -1;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(all_exhausted(f.cursors));
}

TEST_CASE("windowed_merger merges HOST runs with several STRING columns", "[windowed_merger]")
{
  // (STRING key, STRING long payload, INT64 id): the two STRING columns differ a lot in width.
  const merge_key_spec keys{.key_cols        = {0, 2},
                            .column_order    = {cudf::order::ASCENDING, cudf::order::ASCENDING},
                            .null_precedence = {cudf::null_order::AFTER, cudf::null_order::AFTER}};
  const std::vector<std::size_t> fetch_cols{0, 1, 2};
  const std::vector<cudf::data_type> fetch_types{cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::INT64}};
  host_merge_fixture f;
  std::mt19937 gen(221);
  std::int64_t next_id = 0;
  for (const std::size_t rows : {900UL, 1400UL, 600UL}) {
    std::vector<std::string> key_values(rows);
    std::vector<std::string> payload(rows);
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      key_values[i] = "k" + std::to_string(gen() % 40);
      payload[i]    = std::string(gen() % 200, 'p') + std::to_string(next_id);
      ids[i]        = next_id++;
    }
    auto table = make_table(make_string_column(key_values),
                            make_string_column(payload),
                            make_column(ids, cudf::type_id::INT64));
    f.add_host_run(table->view(), keys, 2);
  }

  auto merger = f.make_merger(keys,
                              {.window_rows = 128, .output_target_bytes = 30000},
                              reference_bounded_merge,
                              fetch_cols,
                              fetch_types);
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);
  CHECK(collected.chunks.size() > 1);
  auto expected = expected_full_sort(f.runs, keys);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(stats.rows_out == 2900);
  CHECK(stats.refill_rows == 2900);
}

// ---------------------------------------------------------------------------
// One merger kept across tasks: rebase_consumed and a stream per run()
// ---------------------------------------------------------------------------

namespace {

/// Calls run() and rebase_consumed() until every cursor is exhausted, on the @p streams in turn,
/// and checks after each call that the merger let go of exactly the batches read to the end.
/// Returns the concatenated output.
std::unique_ptr<cudf::table> merge_with_rebase(windowed_merger& merger,
                                               const std::vector<test_run>& runs,
                                               std::vector<merge_input_cursor>& cursors,
                                               const std::vector<rmm::cuda_stream_view>& streams,
                                               rmm::device_async_resource_ref mr,
                                               merge_round_stats& stats)
{
  // References held while the merger has all of them: the run, the repository and the merger.
  std::vector<std::vector<long>> refs;
  for (const auto& run : runs) {
    auto& counts = refs.emplace_back();
    for (const auto& batch : run.batches) {
      counts.push_back(batch.use_count());
    }
  }
  std::vector<std::size_t> released(runs.size(), 0);
  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::table_view> views;
  for (std::size_t call = 0; !all_exhausted(cursors); ++call) {
    REQUIRE(call < 100000);
    const auto stream = streams[call % streams.size()];
    chunks.push_back(merger.run(stream, mr, &stats));
    REQUIRE(chunks.back()->num_rows() > 0);
    views.push_back(chunks.back()->view());
    stream.synchronize();

    const auto before = cursors;
    merger.rebase_consumed();
    for (std::size_t i = 0; i < runs.size(); ++i) {
      CAPTURE(call, i);
      released[i] += before[i].batch;
      CHECK(cursors[i].batch == 0);
      CHECK(cursors[i].row == before[i].row);
      CHECK(cursors[i].remaining() == before[i].remaining());
      // The cursor's run is what is left of the original one.
      const auto& ids  = runs[i].run.batch_ids;
      const auto& rows = runs[i].run.batch_rows;
      const auto first = static_cast<std::ptrdiff_t>(released[i]);
      CHECK(cursors[i].run.batch_ids == std::vector<std::uint64_t>(ids.begin() + first, ids.end()));
      CHECK(cursors[i].run.batch_rows ==
            std::vector<std::size_t>(rows.begin() + first, rows.end()));
      for (std::size_t b = 0; b < runs[i].batches.size(); ++b) {
        CHECK(runs[i].batches[b].use_count() == refs[i][b] - (b < released[i] ? 1 : 0));
      }
    }
  }
  for (std::size_t i = 0; i < runs.size(); ++i) {
    CHECK(released[i] == runs[i].batches.size());
    CHECK(cursors[i].run.batch_ids.empty());
  }
  // Exhausted runs stay harmless: nothing to merge, nothing to release.
  auto empty = merger.run(streams.front(), mr, &stats);
  CHECK(empty->num_rows() == 0);
  merger.rebase_consumed();
  CHECK(all_exhausted(cursors));
  return cudf::concatenate(views, streams.front(), mr);
}

}  // namespace

TEST_CASE("windowed_merger rebase_consumed releases the HOST batches read to the end",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  host_merge_fixture f;
  f.add_host_run(random_rows(2400, 100000, 0, 301)->view(), keys, 4);
  f.add_host_run(random_rows(1500, 100000, 10000, 302)->view(), keys);
  f.add_host_run(random_rows(1800, 100000, 20000, 303)->view(), keys, 3);
  // A short run with small keys: exhausted long before the others.
  f.add_host_run(random_rows(300, 20000, 30000, 304)->view(), keys);
  auto expected = expected_full_sort(f.runs, keys);

  // One round per run(), so every run() leaves rows in the window buffers for the next one. Each
  // task has its own stream.
  rmm::cuda_stream second;
  rmm::cuda_stream third;
  const std::vector<rmm::cuda_stream_view> streams{
    f.stream, rmm::cuda_stream_view{second.value()}, rmm::cuda_stream_view{third.value()}};
  auto merger = f.make_merger(keys, {.window_rows = 256, .output_target_bytes = 1});
  merge_round_stats stats;
  auto out = merge_with_rebase(merger, f.runs, f.cursors, streams, f.mr(), stats);

  CHECK(tables_equal(out->view(), expected->view()));
  CHECK(stats.rows_out == 6000);
  // No row is read twice although the cursors were rebased between the calls.
  CHECK(stats.refill_rows == 6000);
  CHECK(stats.compacted_rows <= stats.refill_rows);
  CHECK(stats.fallback_conversions == 0);
  CHECK(stats.rounds > 20);
}

TEST_CASE("windowed_merger rebase_consumed releases the GPU batches read to the end",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  merge_fixture f;
  f.add_run(random_rows(2000, 50000, 0, 311)->view(), keys, 5);
  f.add_run(random_rows(900, 50000, 10000, 312)->view(), keys);
  {
    // An empty batch in the middle of a run is released with the batch before it.
    auto sorted = stable_sort(random_rows(1500, 50000, 20000, 313)->view(), keys);
    f.runs.push_back(make_gpu_run(f.repository, f.space, sorted->view(), {600, 0, 900}));
  }
  auto expected = expected_full_sort(f.runs, keys);

  auto merger = f.make_merger(keys, {.window_rows = 200, .output_target_bytes = 5000});
  merge_round_stats stats;
  auto out = merge_with_rebase(merger, f.runs, f.cursors, {f.stream}, f.mr(), stats);
  CHECK(tables_equal(out->view(), expected->view()));
  CHECK(stats.rows_out == 4400);
  CHECK(stats.refill_rows == 0);
  CHECK(f.all_batches_unlocked());
}

TEST_CASE("windowed_merger rebase_consumed after a failed run() replays exactly",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  host_merge_fixture f;
  f.add_host_run(random_rows(1600, 100000, 0, 321)->view(), keys, 4);
  f.add_host_run(random_rows(1400, 100000, 10000, 322)->view(), keys, 2);
  auto expected = expected_full_sort(f.runs, keys);

  int calls          = 0;
  auto failing_merge = [&](std::span<const cudf::table_view> windows,
                           const merge_key_spec& key_spec,
                           std::span<const std::uint8_t> truncated,
                           cudf::size_type max_rows,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) {
    if (++calls == 4) { throw std::runtime_error("injected merge failure"); }
    return reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };
  // Two rounds per run(): the fourth merge is the second round of the second run().
  auto merger =
    f.make_merger(keys, {.window_rows = 300, .output_target_bytes = 9000}, failing_merge);
  merge_round_stats stats;
  auto first = merger.run(f.stream, f.mr(), &stats);
  merger.rebase_consumed();
  const auto after_first = f.cursors;

  REQUIRE_THROWS_WITH(merger.run(f.stream, f.mr(), &stats), "injected merge failure");
  // The failed run() dropped the buffers; rebasing again changes nothing.
  merger.rebase_consumed();
  for (std::size_t i = 0; i < f.cursors.size(); ++i) {
    CHECK(f.cursors[i].batch == after_first[i].batch);
    CHECK(f.cursors[i].row == after_first[i].row);
    CHECK(f.cursors[i].run.batch_ids == after_first[i].run.batch_ids);
  }

  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::table_view> views{first->view()};
  while (!all_exhausted(f.cursors)) {
    chunks.push_back(merger.run(f.stream, f.mr(), &stats));
    views.push_back(chunks.back()->view());
    merger.rebase_consumed();
  }
  CHECK(tables_equal(cudf::concatenate(views)->view(), expected->view()));
  CHECK(stats.rows_out == 3000);
}

namespace {

const std::vector<cudf::data_type> kKeyAndFourStrings{cudf::data_type{cudf::type_id::INT64},
                                                      cudf::data_type{cudf::type_id::STRING},
                                                      cudf::data_type{cudf::type_id::STRING},
                                                      cudf::data_type{cudf::type_id::STRING},
                                                      cudf::data_type{cudf::type_id::STRING}};

/// An unsorted (INT64 key, STRING wide, 3 x STRING narrow) table. The wide value of a row has
/// wide_bytes(key) bytes, give or take 10%; the narrow values have 3 to 5 bytes.
template <typename WideBytes>
std::unique_ptr<cudf::table> uneven_string_rows(std::size_t rows,
                                                std::int64_t key_range,
                                                std::uint32_t seed,
                                                WideBytes&& wide_bytes)
{
  std::mt19937 gen(seed);
  std::vector<std::int64_t> keys(rows);
  std::vector<std::string> wide(rows);
  std::vector<std::vector<std::string>> narrow(3, std::vector<std::string>(rows));
  for (std::size_t i = 0; i < rows; ++i) {
    keys[i]          = static_cast<std::int64_t>(gen() % static_cast<std::uint64_t>(key_range));
    const auto bytes = static_cast<std::size_t>(wide_bytes(keys[i]));
    wide[i] =
      std::string(bytes - bytes / 10 + gen() % (bytes / 5 + 1), static_cast<char>('a' + i % 26));
    for (auto& column : narrow) {
      column[i] = std::string(3 + gen() % 3, static_cast<char>('A' + gen() % 26));
    }
  }
  return make_table(make_column(keys, cudf::type_id::INT64),
                    make_string_column(wide),
                    make_string_column(narrow[0]),
                    make_string_column(narrow[1]),
                    make_string_column(narrow[2]));
}

}  // namespace

TEST_CASE("windowed_merger reads full windows when the STRING columns differ in width",
          "[windowed_merger]")
{
  // k = 8 HOST runs of (INT64 key, one ~100-byte STRING, three ~4-byte STRINGs). Sized from the
  // average of the four columns, the wide column would fill at about a third of a window, every
  // refill would stop short and the buffers would be compacted every round.
  const auto keys = ascending_key();
  host_merge_fixture f;
  for (std::uint32_t run = 0; run < 8; ++run) {
    f.add_host_run(
      uneven_string_rows(1500, 1000000, 400 + run, [](std::int64_t) { return 100; })->view(),
      keys,
      2);
  }
  auto expected = expected_full_sort(f.runs, keys);

  auto merger = f.make_merger(keys,
                              {.window_rows = 256, .output_target_bytes = 200000},
                              bounded_kway_merge,
                              {0, 1, 2, 3, 4},
                              kKeyAndFourStrings);
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);
  CHECK(collected.chunks.size() > 1);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(stats.rows_out == 12000);
  // Every row is read once, every refill fills its window, and a row is carried to the front of
  // its buffer at most once.
  CHECK(stats.refill_rows == 12000);
  CHECK(stats.refill_truncated == 0);
  CHECK(stats.window_growths == 0);
  CHECK(stats.compacted_rows < stats.refill_rows);
  CHECK(stats.fallback_conversions == 0);
}

TEST_CASE("windowed_merger carries a row at most once when a STRING column is skewed",
          "[windowed_merger]")
{
  // The wide column is 8 bytes for the low half of the keys and 300 bytes for the high half, so
  // the per-column estimate (the batch average) is too small for the windows of the high half:
  // their refills stop short of the window on bytes.
  const auto keys = ascending_key();
  host_merge_fixture f;
  for (std::uint32_t run = 0; run < 6; ++run) {
    f.add_host_run(
      uneven_string_rows(
        2000, 1000000, 500 + run, [](std::int64_t key) { return key < 500000 ? 8 : 300; })
        ->view(),
      keys);
  }
  auto expected = expected_full_sort(f.runs, keys);

  auto merger = f.make_merger(keys,
                              {.window_rows = 256, .output_target_bytes = 200000},
                              bounded_kway_merge,
                              {0, 1, 2, 3, 4},
                              kKeyAndFourStrings);
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(stats.rows_out == 12000);
  CHECK(stats.refill_rows == 12000);
  CHECK(stats.refill_truncated > 0);
  // The low-water mark follows what the buffer held after its last refill, so short refills do
  // not bring back a compaction per round.
  CHECK(stats.compacted_rows <= stats.refill_rows);
}

TEST_CASE("windowed_merger keeps its windows across run() calls on different streams",
          "[windowed_merger]")
{
  // (STRING key, STRING payload, INT64 id) HOST runs; every run() uses another stream, as the
  // tasks of one partition do, with the same memory resource.
  const merge_key_spec keys{.key_cols        = {0, 2},
                            .column_order    = {cudf::order::ASCENDING, cudf::order::ASCENDING},
                            .null_precedence = {cudf::null_order::AFTER, cudf::null_order::AFTER}};
  const std::vector<cudf::data_type> fetch_types{cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::INT64}};
  host_merge_fixture f;
  std::mt19937 gen(331);
  std::int64_t next_id = 0;
  for (const std::size_t rows : {1100UL, 1600UL, 700UL, 1300UL}) {
    std::vector<std::string> key_values(rows);
    std::vector<std::string> payload(rows);
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      key_values[i] = "key-" + std::to_string(gen() % 300);
      payload[i]    = std::string(gen() % 60, 'p') + std::to_string(next_id);
      ids[i]        = next_id++;
    }
    auto table = make_table(make_string_column(key_values),
                            make_string_column(payload),
                            make_column(ids, cudf::type_id::INT64));
    f.add_host_run(table->view(), keys, 2);
  }
  auto expected = expected_full_sort(f.runs, keys);

  std::vector<rmm::cuda_stream> owned(3);
  auto merger = f.make_merger(keys,
                              {.window_rows = 128, .output_target_bytes = 1},
                              bounded_kway_merge,
                              {0, 1, 2},
                              fetch_types);
  merge_round_stats stats;
  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::table_view> views;
  for (std::size_t call = 0; !all_exhausted(f.cursors); ++call) {
    REQUIRE(call < 100000);
    // No host synchronization between the calls: the merger orders the streams itself.
    const rmm::cuda_stream_view stream{owned[call % owned.size()].value()};
    chunks.push_back(merger.run(stream, f.mr(), &stats));
    views.push_back(chunks.back()->view());
  }
  for (const auto& stream : owned) {
    stream.synchronize();
  }
  CHECK(chunks.size() == stats.rounds);
  CHECK(tables_equal(cudf::concatenate(views)->view(), expected->view()));
  CHECK(stats.rows_out == 4700);
  // The rows buffered on one stream are served to the next one, not read again.
  CHECK(stats.refill_rows == 4700);
}

// ---------------------------------------------------------------------------
// Output target, failures and representations
// ---------------------------------------------------------------------------

namespace {

/// GPU runs of (STRING key, INT64 id, STRING payload) rows, fetched as (key, payload, id) and
/// merged on (key, id).
struct string_key_runs {
  const merge_key_spec keys{.key_cols        = {0, 2},
                            .column_order    = {cudf::order::ASCENDING, cudf::order::ASCENDING},
                            .null_precedence = {cudf::null_order::AFTER, cudf::null_order::AFTER}};
  const std::vector<std::size_t> fetch_cols{0, 2, 1};
  const std::vector<cudf::data_type> fetch_types{cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::STRING},
                                                 cudf::data_type{cudf::type_id::INT64}};
  merge_fixture f;
  std::unique_ptr<cudf::table> expected;

  explicit string_key_runs(std::uint32_t seed)
  {
    std::mt19937 gen(seed);
    std::int64_t next_id = 0;
    std::vector<std::unique_ptr<cudf::table>> fetched;
    std::vector<cudf::table_view> fetched_views;
    for (const std::size_t rows : {1200UL, 800UL, 1000UL}) {
      std::vector<std::string> key_values(rows);
      std::vector<std::string> payload(rows);
      std::vector<std::int64_t> ids(rows);
      for (std::size_t i = 0; i < rows; ++i) {
        key_values[i] = "k" + std::to_string(gen() % 500);
        payload[i]    = std::string(gen() % 120, 'x');
        ids[i]        = next_id++;
      }
      auto table = make_table(make_string_column(key_values),
                              make_column(ids, cudf::type_id::INT64),
                              make_string_column(payload));
      // The runs are sorted on the source columns (key, id): source column 1 is fetched column 2.
      const merge_key_spec source_keys{.key_cols        = {0, 1},
                                       .column_order    = keys.column_order,
                                       .null_precedence = keys.null_precedence};
      f.add_run(table->view(), source_keys, 2);
      fetched.push_back(
        std::make_unique<cudf::table>(f.runs.back().rows->view().select({0, 2, 1})));
      fetched_views.push_back(fetched.back()->view());
    }
    expected = expected_full_sort(std::span<const cudf::table_view>(fetched_views), keys);
  }

  windowed_merger make_merger(window_config cfg, bounded_merge_fn merge)
  {
    return f.make_merger(keys, cfg, std::move(merge), fetch_cols, fetch_types);
  }
};

}  // namespace

TEST_CASE("windowed_merger stops with the round whose output bytes reach the target",
          "[windowed_merger]")
{
  string_key_runs runs(341);
  std::vector<std::size_t> round_bytes;  // alloc_size() of each round's output
  auto recording_merge = [&](std::span<const cudf::table_view> windows,
                             const merge_key_spec& key_spec,
                             std::span<const std::uint8_t> truncated,
                             cudf::size_type max_rows,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) {
    auto result = reference_bounded_merge(windows, key_spec, truncated, max_rows, stream, mr);
    round_bytes.push_back(result.output->alloc_size());
    return result;
  };
  // A 100-row round is about 100 * (60 + 6 + 8 + 8) bytes: the target takes several rounds.
  constexpr std::size_t target = 40000;
  auto merger =
    runs.make_merger({.window_rows = 100, .output_target_bytes = target}, recording_merge);

  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::table_view> views;
  while (!all_exhausted(runs.f.cursors)) {
    round_bytes.clear();
    merge_round_stats stats;
    chunks.push_back(merger.run(runs.f.stream, runs.f.mr(), &stats));
    views.push_back(chunks.back()->view());
    REQUIRE(stats.rounds == round_bytes.size());
    REQUIRE(!round_bytes.empty());
    // The exact bytes of the rounds: the last round is the first to reach the target, no round
    // more and none less (unless the inputs ran out first).
    std::size_t total = 0;
    for (const auto bytes : round_bytes) {
      total += bytes;
    }
    CHECK(total - round_bytes.back() < target);
    if (!all_exhausted(runs.f.cursors)) {
      CHECK(total >= target);
      CHECK(round_bytes.size() > 2);
    }
  }
  CHECK(chunks.size() > 2);
  CHECK(tables_equal(cudf::concatenate(views)->view(), runs.expected->view()));
}

TEST_CASE("windowed_merger with a STRING key replays after the merge throws in round 2",
          "[windowed_merger]")
{
  string_key_runs runs(351);
  int calls          = 0;
  auto failing_merge = [&](std::span<const cudf::table_view> windows,
                           const merge_key_spec& key_spec,
                           std::span<const std::uint8_t> truncated,
                           cudf::size_type max_rows,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) {
    if (++calls == 2) { throw std::runtime_error("injected merge failure"); }
    return bounded_kway_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };
  auto merger =
    runs.make_merger({.window_rows = 150, .output_target_bytes = kUnlimitedOutput}, failing_merge);
  merge_round_stats stats;
  // Round 1 produced a STRING output that is dropped with the failure.
  REQUIRE_THROWS_WITH(merger.run(runs.f.stream, runs.f.mr(), &stats), "injected merge failure");
  CHECK(calls == 2);
  for (const auto& cursor : runs.f.cursors) {
    CHECK(cursor.batch == 0);
    CHECK(cursor.row == 0);
  }
  CHECK(stats.rounds == 0);
  CHECK(runs.f.all_batches_unlocked());

  // The following run() uses the real merge for every round.
  auto out = merger.run(runs.f.stream, runs.f.mr(), &stats);
  CHECK(tables_equal(out->view(), runs.expected->view()));
  CHECK(all_exhausted(runs.f.cursors));
  CHECK(stats.rows_out == 3000);
  CHECK(runs.f.all_batches_unlocked());
}

TEST_CASE("windowed_merger converts a compressed device batch instead of slicing it",
          "[windowed_merger]")
{
  // A GPU-tier batch in the task's GPU space that is not an uncompressed table: it has no views
  // to slice and no row-range reader, so it is converted once.
  const auto keys = ascending_key();
  merge_fixture f;
  sirius::test::operator_utils::ensure_converter_registry();
  f.add_run(random_rows(1200, 100000, 0, 361)->view(), keys);
  {
    auto sorted = stable_sort(random_rows(900, 100000, 10000, 362)->view(), keys);
    auto blob   = std::make_shared<sirius::compressed_device_blob>();
    blob->table = simpatico::compress_with_plan(sorted->view(),
                                                "input -> identity\n---\ninput -> identity\n",
                                                sirius::test::operator_utils::default_stream(),
                                                f.mr());
    f.stream.synchronize();
    auto rep = std::make_unique<sirius::compressed_device_representation>(
      f.space,
      std::move(blob),
      std::vector<std::string>{"key", "id"},
      sorted->alloc_size(),
      sorted->alloc_size(),
      sorted->num_rows());
    auto batch = cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(rep));
    f.repository.add_data_batch(batch);
    test_run run;
    run.run.batch_ids.push_back(batch->get_batch_id());
    run.run.batch_rows.push_back(900);
    run.batches.push_back(std::move(batch));
    run.rows = std::move(sorted);
    f.runs.push_back(std::move(run));
  }
  auto expected = expected_full_sort(f.runs, keys);

  auto merger = f.make_merger(keys, {.window_rows = 256, .output_target_bytes = 1});
  merge_round_stats stats;
  auto collected = collect_full_sort(merger, f.cursors, f.stream, f.mr(), &stats);
  CHECK(collected.chunks.size() > 2);
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(stats.fallback_conversions == 1);
  CHECK(stats.refill_rows == 0);
  {
    auto lock = f.runs[1].batches[0]->to_read_only();
    CHECK(dynamic_cast<const cucascade::gpu_table_representation*>(lock.get_data()) != nullptr);
  }
  CHECK(f.all_batches_unlocked());
}

// ---------------------------------------------------------------------------
// A batch that changes tier under the merger; the whole-batch fallback
// ---------------------------------------------------------------------------

TEST_CASE("windowed_merger follows a batch that changes tier while its rows are buffered",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  tiers_fixture fx;  // first: the batches are destroyed before the memory manager
  cucascade::shared_data_repository repository;
  std::vector<test_run> runs;
  const auto stream = fx.stream_view();
  const std::size_t run_rows[]{2000, 1800, 1600};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto first_id = static_cast<std::int64_t>(10000 * i);
    runs.push_back(make_sorted_gpu_run(
      repository,
      *fx.gpu,
      random_rows(run_rows[i], 100000, first_id, 401 + static_cast<std::uint32_t>(i))->view(),
      keys));
    cudf::get_default_stream().sync();
    auto mut = runs.back().batches[0]->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), fx.host, fx.stream);
    fx.stream.synchronize();
  }
  auto expected = expected_full_sort(runs, keys);
  auto cursors  = make_cursors(runs);

  // One round per run(): after the first, every run has rows of its HOST batch in its buffer.
  windowed_merger merger(run_batches(runs),
                         cursors,
                         kTwoCols,
                         kTwoInt64,
                         keys,
                         {.window_rows = 256, .output_target_bytes = 1},
                         sirius::row_range_reader_registry::get(),
                         *fx.gpu,
                         reference_bounded_merge);
  merge_round_stats stats;
  std::vector<std::unique_ptr<cudf::table>> chunks;
  chunks.push_back(merger.run(stream, fx.mr(), &stats));
  REQUIRE(stats.rounds == 1);
  REQUIRE(stats.refill_rows == 3 * 256);
  REQUIRE(stats.refill_bytes_host == 3 * 256 * 16);
  REQUIRE(chunks.back()->num_rows() < 3 * 256);
  REQUIRE_FALSE(all_exhausted(cursors));

  // What a downgrade (run 0: HOST -> DISK) and another task's upgrade (run 1: HOST -> GPU) do
  // between two tasks of the merge. Run 2 stays on the HOST tier.
  stream.synchronize();
  {
    auto mut = runs[0].batches[0]->to_mutable();
    mut.convert_to<cucascade::disk_data_representation>(
      sirius::converter_registry::get(), fx.disk, fx.stream);
    fx.stream.synchronize();
    REQUIRE(mut.get_current_tier() == cucascade::memory::Tier::DISK);
  }
  {
    auto mut = runs[1].batches[0]->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(
      sirius::converter_registry::get(), fx.gpu, fx.stream);
    fx.stream.synchronize();
    REQUIRE(mut.get_current_tier() == cucascade::memory::Tier::GPU);
  }

  std::vector<cudf::table_view> views{chunks.back()->view()};
  while (!all_exhausted(cursors)) {
    REQUIRE(chunks.size() < 100000);
    chunks.push_back(merger.run(stream, fx.mr(), &stats));
    views.push_back(chunks.back()->view());
  }
  auto merged = cudf::concatenate(views, stream, fx.mr());
  stream.synchronize();
  CHECK(tables_equal(merged->view(), expected->view()));
  CHECK(stats.rows_out == 2000 + 1800 + 1600);
  // Run 0 keeps the rows it had buffered and reads the rest from DISK: every row once. Run 1
  // drops its buffer (the 256 rows of the first refill) and is sliced on the GPU from its cursor
  // on. Run 2 is read from HOST throughout.
  CHECK(stats.refill_rows == 2000 + 256 + 1600);
  CHECK(stats.refill_bytes_host == (256 + 256 + 1600) * 16);
  CHECK(stats.refill_bytes_disk == (2000 - 256) * 16);
  CHECK(stats.fallback_conversions == 0);
}

TEST_CASE("windowed_merger converts a batch again when it left the GPU before its window",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  host_merge_fixture f;
  f.add_host_run(random_rows(900, 100000, 0, 411)->view(), keys);
  f.add_host_run(random_rows(700, 100000, 10000, 412)->view(), keys);
  auto expected = expected_full_sort(f.runs, keys);
  f.cursors     = make_cursors(f.runs);

  // No reader: every batch takes the whole-batch fallback. The first conversion of each batch is
  // undone before the merger locks the batch again, as a downgrade that wins that race does.
  cucascade::row_range_reader_registry no_readers;
  std::set<std::uint64_t> lost;
  int calls            = 0;
  auto losing_the_race = [&](cucascade::data_batch& batch,
                             cucascade::memory::memory_space& gpu_space,
                             rmm::cuda_stream_view stream) {
    ++calls;
    const auto converted =
      sirius::op::merge::detail::convert_run_batch_to_gpu(batch, gpu_space, stream);
    if (lost.insert(batch.get_batch_id()).second) {
      auto mut = batch.to_mutable();
      mut.convert_to<cucascade::host_data_representation>(
        sirius::converter_registry::get(), f.memory.host0, stream);
      stream.synchronize();
    }
    return converted;
  };
  windowed_merger merger(run_batches(f.runs),
                         f.cursors,
                         kTwoCols,
                         kTwoInt64,
                         keys,
                         {.window_rows = 256, .output_target_bytes = kUnlimitedOutput},
                         no_readers,
                         *f.memory.gpu0,
                         reference_bounded_merge,
                         losing_the_race);
  merge_round_stats stats;
  auto merged = merger.run(f.stream, f.mr(), &stats);
  f.stream.synchronize();
  CHECK(tables_equal(merged->view(), expected->view()));
  CHECK(all_exhausted(f.cursors));
  CHECK(calls == 4);
  CHECK(stats.fallback_conversions == 4);
  CHECK(stats.refill_rows == 0);
}

TEST_CASE("windowed_merger gives up on a batch that never stays on the GPU", "[windowed_merger]")
{
  const auto keys = ascending_key();
  host_merge_fixture f;
  f.add_host_run(random_rows(900, 100000, 0, 421)->view(), keys);
  f.add_host_run(random_rows(700, 100000, 10000, 422)->view(), keys);
  auto expected = expected_full_sort(f.runs, keys);
  f.cursors     = make_cursors(f.runs);

  // No reader, and a fallback that never leaves the batch on the GPU.
  cucascade::row_range_reader_registry no_readers;
  int calls           = 0;
  bool convert        = false;
  auto never_resident = [&](cucascade::data_batch& batch,
                            cucascade::memory::memory_space& gpu_space,
                            rmm::cuda_stream_view stream) {
    ++calls;
    return convert && sirius::op::merge::detail::convert_run_batch_to_gpu(batch, gpu_space, stream);
  };
  windowed_merger merger(run_batches(f.runs),
                         f.cursors,
                         kTwoCols,
                         kTwoInt64,
                         keys,
                         {.window_rows = 256, .output_target_bytes = kUnlimitedOutput},
                         no_readers,
                         *f.memory.gpu0,
                         reference_bounded_merge,
                         never_resident);
  merge_round_stats stats;
  REQUIRE_THROWS_WITH(merger.run(f.stream, f.mr(), &stats),
                      Catch::Matchers::ContainsSubstring("after 3 attempts"));
  // One conversion per attempt on the first run's batch, then the run() fails: nothing is
  // committed.
  CHECK(calls == 3);
  for (const auto& cursor : f.cursors) {
    CHECK(cursor.batch == 0);
    CHECK(cursor.row == 0);
  }
  CHECK(stats.rounds == 0);
  CHECK(stats.fallback_conversions == 0);

  // The same merger replays from the unchanged cursors once the conversion works.
  convert     = true;
  auto merged = merger.run(f.stream, f.mr(), &stats);
  f.stream.synchronize();
  CHECK(tables_equal(merged->view(), expected->view()));
  CHECK(stats.fallback_conversions == 2);
}

TEST_CASE("windowed_merger rejects a HOST batch whose rows differ from its run",
          "[windowed_merger]")
{
  const auto keys = ascending_key();
  // The only batch of the run holds 200 rows; windows of 64 rows are read through its reader.
  const auto claimed = GENERATE(std::size_t{150}, std::size_t{250});
  CAPTURE(claimed);
  host_merge_fixture f;
  f.add_host_run(random_rows(200, 100, 0, 431)->view(), keys);
  f.cursors                      = make_cursors(f.runs);
  f.cursors[0].run.batch_rows[0] = claimed;
  auto merger = f.make_merger(keys, {.window_rows = 64, .output_target_bytes = kUnlimitedOutput});
  // 150: the third refill delivers rows past the cursor's count. 250: the reader ends at row 200.
  CHECK_THROWS_AS(merger.run(f.stream, f.mr()), sirius::internal_exception);
  CHECK(f.cursors[0].batch == 0);
  CHECK(f.cursors[0].row == 0);
}
