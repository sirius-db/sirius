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

// The counters of merge_round_stats that split the refills by what they read (HOST, DISK,
// compressed) and bound the rows of a round, and accumulate().

#include "catch.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/sirius_converter_registry.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/windowed_merge.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_payload_sources.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/device_compressed_blob.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/disk_data_representation.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::merge_utils;

namespace {

namespace utils = sirius::test::operator_utils;

bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <typename Representation>
bool holds(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return dynamic_cast<Representation const*>(ro.get_data()) != nullptr;
}

/// What a run's batches are turned into after they are built on the GPU.
enum class placement { gpu, host, disk, compressed_device, compressed_host, compressed_disk };

bool is_compressed(placement where)
{
  return where == placement::compressed_device || where == placement::compressed_host ||
         where == placement::compressed_disk;
}

/// Moves the uncompressed GPU batch @p batch to the HOST or the DISK tier, uncompressed.
void move_batch(tiers_fixture& fx, cucascade::data_batch& batch, placement where)
{
  auto mut = batch.to_mutable();
  mut.convert_to<cucascade::host_data_representation>(
    sirius::converter_registry::get(), fx.host, fx.stream);
  fx.stream.synchronize();
  if (where == placement::disk) {
    mut.convert_to<cucascade::disk_data_representation>(
      sirius::converter_registry::get(), fx.disk, fx.stream);
    fx.stream.synchronize();
  }
}

/// Spills the GPU batch @p batch compressed, as a downgrade does, and leaves it in @p where.
void compress_batch(tiers_fixture& fx,
                    const std::shared_ptr<cucascade::data_batch>& batch,
                    placement where)
{
  scoped_spill_compression compression;
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({fx.host}, fx.stream_ref(), *fx.manager, true).has_value());
  fx.stream.synchronize();
  REQUIRE(holds<sirius::compressed_host_representation>(*batch));
  if (where == placement::compressed_host) { return; }

  if (where == placement::compressed_disk) {
    REQUIRE(convertible.convert({fx.disk}, fx.stream_ref(), *fx.manager, true).has_value());
    fx.stream.synchronize();
    REQUIRE(holds<sirius::compressed_disk_representation>(*batch));
    return;
  }

  // The device tier: the compressed table rebuilt on the device from the pinned payload.
  auto mut           = batch->to_mutable();
  auto const& host   = mut.get_data()->cast<sirius::compressed_host_representation>();
  auto const& blocks = host.payload();
  auto blob          = std::make_shared<sirius::compressed_device_blob>();
  std::string error;
  simpatico::payload_fetch_fn fetch =
    [&blocks](std::uint64_t offset, std::size_t size, void* dst, ::cuda::stream_ref stream) {
      sirius::copy_pinned_blocks_to_device(blocks, offset, dst, size, stream);
    };
  blob->table = simpatico::read_compressed_table_from_memory(
    host.header(), fetch, fx.stream_ref(), fx.mr(), &error);
  REQUIRE(error.empty());
  fx.stream.synchronize();
  auto device = std::make_unique<sirius::compressed_device_representation>(
    *fx.gpu,
    std::move(blob),
    host.column_names(),
    host.get_size_in_bytes(),
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());
  mut.set_data(std::move(device));
}

merge::merge_key_spec ascending_key()
{
  return {.key_cols        = {0},
          .column_order    = {cudf::order::ASCENDING},
          .null_precedence = {cudf::null_order::AFTER}};
}

/// An unsorted (key INT64 in [0, 1000), id INT64) table; ids are unique from @p first_id on.
std::unique_ptr<cudf::table> random_rows(std::size_t rows,
                                         std::int64_t first_id,
                                         std::uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_int_distribution<std::int64_t> dist(0, 999);
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
/// Data bytes of one (INT64, INT64) row.
constexpr std::size_t kRowBytes = 16;

/// Sorted (key, id) runs, each placed whole in one tier and representation.
/// Declared after the fixture: the batches must go before the memory manager does.
struct placed_runs {
  tiers_fixture& fx;
  cucascade::shared_data_repository repository;
  std::vector<test_run> runs;
  std::vector<merge::merge_input_cursor> cursors;
  std::size_t total_rows = 0;

  explicit placed_runs(tiers_fixture& fixture) : fx(fixture) {}

  /// Adds a run of @p rows rows in @p num_batches batches, all turned into @p where.
  void add(placement where, std::size_t rows, std::size_t num_batches = 1)
  {
    auto const index = runs.size();
    auto const table = random_rows(
      rows, static_cast<std::int64_t>(index) * 1'000'000, static_cast<std::uint32_t>(7 + index));
    runs.push_back(
      make_sorted_gpu_run(repository, *fx.gpu, table->view(), ascending_key(), num_batches));
    // The batches were copied on the default stream; the converters run on the fixture's.
    cudf::get_default_stream().sync();
    for (auto const& batch : runs.back().batches) {
      if (where == placement::host || where == placement::disk) {
        move_batch(fx, *batch, where);
      } else if (is_compressed(where)) {
        compress_batch(fx, batch, where);
      }
    }
    total_rows += rows;
  }

  merge::windowed_merger make_merger(merge::window_config cfg)
  {
    cursors = make_cursors(runs);
    return merge::windowed_merger(run_batches(runs),
                                  cursors,
                                  kTwoCols,
                                  kTwoInt64,
                                  ascending_key(),
                                  cfg,
                                  sirius::row_range_reader_registry::get(),
                                  *fx.gpu);
  }
};

constexpr std::size_t kNoTarget = std::numeric_limits<std::size_t>::max();

}  // namespace

// ---------------------------------------------------------------------------
// 1. GPU-only runs
// ---------------------------------------------------------------------------

TEST_CASE("windowed_merger reads no refill bytes from GPU runs and bounds the rows of a round",
          "[merger_stats]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  tiers_fixture fx;
  placed_runs runs(fx);
  constexpr std::size_t kWindowRows = 1000;

  SECTION("three runs over several run() calls")
  {
    runs.add(placement::gpu, 5000, 2);
    runs.add(placement::gpu, 3000);
    runs.add(placement::gpu, 4200, 3);
    auto const expected = expected_full_sort(runs.runs, ascending_key());
    auto merger = runs.make_merger({.window_rows = kWindowRows, .output_target_bytes = 48 * 1024});

    merge::merge_round_stats stats;
    auto const collected =
      collect_full_sort(merger, runs.cursors, fx.stream_view(), fx.mr(), &stats);

    CHECK(collected.chunks.size() > 1);
    CHECK(tables_equal(collected.table->view(), expected->view()));
    CHECK(stats.rows_out == runs.total_rows);
    CHECK(stats.refill_rows == 0);
    CHECK(stats.refill_bytes == 0);
    CHECK(stats.refill_bytes_host == 0);
    CHECK(stats.refill_bytes_disk == 0);
    CHECK(stats.refill_bytes_compressed == 0);
    CHECK(stats.runs_read_compressed == 0);
    CHECK(stats.rounds > 1);
    CHECK(stats.round_rows_min >= 1);
    CHECK(stats.round_rows_min <= stats.round_rows_max);
    CHECK(stats.round_rows_max <= kWindowRows);
    CHECK(stats.rounds * stats.round_rows_min <= stats.rows_out);
    CHECK(stats.rows_out <= stats.rounds * stats.round_rows_max);
  }

  SECTION("one run: every round emits its window")
  {
    runs.add(placement::gpu, 2500);
    auto merger = runs.make_merger({.window_rows = kWindowRows, .output_target_bytes = kNoTarget});

    merge::merge_round_stats stats;
    auto const output = merger.run(fx.stream_view(), fx.mr(), &stats);

    CHECK(output->num_rows() == 2500);
    CHECK(all_exhausted(runs.cursors));
    CHECK(stats.rounds == 3);
    CHECK(stats.round_rows_min == 500);
    CHECK(stats.round_rows_max == 1000);
  }

  SECTION("no round: the bounds stay 0")
  {
    runs.add(placement::gpu, 100);
    auto merger = runs.make_merger({.window_rows = kWindowRows, .output_target_bytes = kNoTarget});
    merge::merge_round_stats stats;
    (void)merger.run(fx.stream_view(), fx.mr(), &stats);
    CHECK(stats.rounds == 1);
    CHECK(stats.round_rows_min == 100);
    CHECK(stats.round_rows_max == 100);

    // Every cursor is exhausted: this run() merges nothing and leaves the bounds as they were.
    merge::merge_round_stats idle;
    auto const empty = merger.run(fx.stream_view(), fx.mr(), &idle);
    CHECK(empty->num_rows() == 0);
    CHECK(idle.rounds == 0);
    CHECK(idle.round_rows_min == 0);
    CHECK(idle.round_rows_max == 0);
    (void)merger.run(fx.stream_view(), fx.mr(), &stats);
    CHECK(stats.round_rows_min == 100);
    CHECK(stats.round_rows_max == 100);
  }
}

// ---------------------------------------------------------------------------
// 2. HOST, DISK and compressed runs
// ---------------------------------------------------------------------------

TEST_CASE("windowed_merger splits its refill bytes by the tier and representation read",
          "[merger_stats]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  tiers_fixture fx;
  placed_runs runs(fx);
  constexpr std::size_t kWindowRows = 1000;
  constexpr std::size_t kHostRows   = 5000;
  constexpr std::size_t kDiskRows   = 3500;
  runs.add(placement::host, kHostRows, 2);
  runs.add(placement::disk, kDiskRows);
  runs.add(placement::gpu, 4000);
  runs.add(placement::compressed_host, 6000, 2);
  runs.add(placement::compressed_disk, 5000);
  runs.add(placement::compressed_device, 4500);
  constexpr std::size_t kCompressedRuns = 3;
  constexpr std::size_t kCompressedRows = 6000 + 5000 + 4500;
  auto const expected                   = expected_full_sort(runs.runs, ascending_key());

  SECTION("one run() call counts each compressed run once")
  {
    auto merger = runs.make_merger({.window_rows = kWindowRows, .output_target_bytes = kNoTarget});
    merge::merge_round_stats stats;
    auto const output = merger.run(fx.stream_view(), fx.mr(), &stats);

    CHECK(all_exhausted(runs.cursors));
    CHECK(tables_equal(output->view(), expected->view()));
    CHECK(stats.fallback_conversions == 0);
    // Every row of the five runs that are not GPU tables came through a window buffer, once.
    CHECK(stats.refill_rows == kHostRows + kDiskRows + kCompressedRows);
    CHECK(stats.refill_bytes_host == kHostRows * kRowBytes);
    CHECK(stats.refill_bytes_disk == kDiskRows * kRowBytes);
    CHECK(stats.refill_bytes_compressed > 0);
    CHECK(stats.refill_bytes_host + stats.refill_bytes_disk + stats.refill_bytes_compressed ==
          stats.refill_bytes);
    // Each compressed run was refilled several times in this call and counts once.
    CHECK(stats.rounds > kCompressedRuns);
    CHECK(stats.runs_read_compressed == kCompressedRuns);
    CHECK(stats.round_rows_min >= 1);
    CHECK(stats.round_rows_max <= kWindowRows);
  }

  SECTION("several run() calls add up like accumulate")
  {
    auto merger = runs.make_merger({.window_rows = kWindowRows, .output_target_bytes = 64 * 1024});
    merge::merge_round_stats summed;  // accumulate() of the stats of each call
    std::size_t calls = 0;
    std::size_t rows  = 0;
    while (!all_exhausted(runs.cursors)) {
      REQUIRE(calls < 1000);
      merge::merge_round_stats call;
      auto const output = merger.run(fx.stream_view(), fx.mr(), &call);
      rows += static_cast<std::size_t>(output->num_rows());
      CHECK(call.runs_read_compressed <= kCompressedRuns);
      // The first call fills the window of every run.
      if (calls == 0) { CHECK(call.runs_read_compressed == kCompressedRuns); }
      CHECK(call.refill_bytes_host + call.refill_bytes_disk + call.refill_bytes_compressed ==
            call.refill_bytes);
      merge::accumulate(summed, call);
      ++calls;
    }
    CHECK(calls > 2);
    CHECK(rows == runs.total_rows);
    CHECK(summed.rows_out == runs.total_rows);
    CHECK(summed.refill_bytes_host == kHostRows * kRowBytes);
    CHECK(summed.refill_bytes_disk == kDiskRows * kRowBytes);
    CHECK(summed.refill_bytes_compressed > 0);
    CHECK(summed.refill_bytes_host + summed.refill_bytes_disk + summed.refill_bytes_compressed ==
          summed.refill_bytes);
    CHECK(summed.runs_read_compressed >= kCompressedRuns);
    CHECK(summed.round_rows_min >= 1);
    CHECK(summed.round_rows_max <= kWindowRows);
  }

  SECTION("stats passed to every run() are added to, not reset")
  {
    auto merger = runs.make_merger({.window_rows = kWindowRows, .output_target_bytes = 64 * 1024});
    merge::merge_round_stats stats;
    auto const collected =
      collect_full_sort(merger, runs.cursors, fx.stream_view(), fx.mr(), &stats);

    CHECK(collected.chunks.size() > 2);
    CHECK(tables_equal(collected.table->view(), expected->view()));
    CHECK(stats.refill_rows == kHostRows + kDiskRows + kCompressedRows);
    CHECK(stats.refill_bytes_host == kHostRows * kRowBytes);
    CHECK(stats.refill_bytes_disk == kDiskRows * kRowBytes);
    CHECK(stats.refill_bytes_host + stats.refill_bytes_disk + stats.refill_bytes_compressed ==
          stats.refill_bytes);
    CHECK(stats.runs_read_compressed >= kCompressedRuns);
    CHECK(stats.round_rows_min >= 1);
    CHECK(stats.round_rows_min <= stats.round_rows_max);
    CHECK(stats.round_rows_max <= kWindowRows);
  }
}

TEST_CASE(
  "runs_read_compressed is call-scoped: one compressed run refilled in two run() calls "
  "accumulates to 2, not 1",
  "[merger_stats]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  tiers_fixture fx;
  placed_runs runs(fx);
  constexpr std::size_t kWindowRows = 1000;
  // The only run, so each round emits its whole window and the next call has to refill it.
  runs.add(placement::compressed_host, 6000);
  auto merger =
    runs.make_merger({.window_rows = kWindowRows, .output_target_bytes = kWindowRows * kRowBytes});

  merge::merge_round_stats first;
  auto const first_output = merger.run(fx.stream_view(), fx.mr(), &first);
  merge::merge_round_stats second;
  auto const second_output = merger.run(fx.stream_view(), fx.mr(), &second);

  REQUIRE(first_output->num_rows() > 0);
  REQUIRE(second_output->num_rows() > 0);
  REQUIRE_FALSE(runs.cursors[0].exhausted());
  REQUIRE(first.refill_bytes_compressed > 0);
  REQUIRE(second.refill_bytes_compressed > 0);
  CHECK(first.runs_read_compressed == 1);
  CHECK(second.runs_read_compressed == 1);

  merge::merge_round_stats summed;
  merge::accumulate(summed, first);
  merge::accumulate(summed, second);
  CHECK(summed.runs_read_compressed == 2);

  // The same sum when one stats object is passed to both calls.
  placed_runs again(fx);
  again.add(placement::compressed_host, 6000);
  auto second_merger =
    again.make_merger({.window_rows = kWindowRows, .output_target_bytes = kWindowRows * kRowBytes});
  merge::merge_round_stats shared;
  (void)second_merger.run(fx.stream_view(), fx.mr(), &shared);
  (void)second_merger.run(fx.stream_view(), fx.mr(), &shared);
  CHECK(shared.runs_read_compressed == 2);
}

TEST_CASE("windowed_merger leaves the new counters untouched when run() throws", "[merger_stats]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  tiers_fixture fx;
  placed_runs runs(fx);
  runs.add(placement::compressed_host, 3000);
  runs.add(placement::host, 3000);
  runs.cursors = make_cursors(runs.runs);
  merge::windowed_merger merger(
    run_batches(runs.runs),
    runs.cursors,
    kTwoCols,
    kTwoInt64,
    ascending_key(),
    {.window_rows = 1000, .output_target_bytes = kNoTarget},
    sirius::row_range_reader_registry::get(),
    *fx.gpu,
    [calls = 0](
      auto windows, const auto& keys, auto truncated, auto max_rows, auto stream, auto mr) mutable {
      if (++calls == 2) { throw std::runtime_error("injected merge failure"); }
      return merge::bounded_kway_merge(windows, keys, truncated, max_rows, stream, mr);
    });

  merge::merge_round_stats stats;
  CHECK_THROWS_AS(merger.run(fx.stream_view(), fx.mr(), &stats), std::runtime_error);
  CHECK(stats.refill_bytes == 0);
  CHECK(stats.refill_bytes_host == 0);
  CHECK(stats.refill_bytes_compressed == 0);
  CHECK(stats.runs_read_compressed == 0);
  CHECK(stats.round_rows_min == 0);
  CHECK(stats.round_rows_max == 0);

  // The retry reads the runs again from the unchanged cursors.
  auto const output = merger.run(fx.stream_view(), fx.mr(), &stats);
  CHECK(output->num_rows() == 6000);
  CHECK(stats.refill_bytes_host == 3000 * kRowBytes);
  CHECK(stats.refill_bytes_host + stats.refill_bytes_compressed == stats.refill_bytes);
  CHECK(stats.runs_read_compressed == 1);
}

// ---------------------------------------------------------------------------
// 3. accumulate
// ---------------------------------------------------------------------------

TEST_CASE("accumulate sums the counters and keeps the round bounds", "[merger_stats]")
{
  merge::merge_round_stats a{.rounds                  = 1,
                             .rows_out                = 2,
                             .refill_rows             = 3,
                             .refill_bytes            = 4,
                             .refill_truncated        = 5,
                             .window_growths          = 6,
                             .compacted_rows          = 7,
                             .fallback_conversions    = 8,
                             .refill_bytes_host       = 9,
                             .refill_bytes_disk       = 10,
                             .refill_bytes_compressed = 11,
                             .runs_read_compressed    = 12,
                             .round_rows_min          = 40,
                             .round_rows_max          = 90};
  const merge::merge_round_stats b{.rounds                  = 100,
                                   .rows_out                = 200,
                                   .refill_rows             = 300,
                                   .refill_bytes            = 400,
                                   .refill_truncated        = 500,
                                   .window_growths          = 600,
                                   .compacted_rows          = 700,
                                   .fallback_conversions    = 800,
                                   .refill_bytes_host       = 900,
                                   .refill_bytes_disk       = 1000,
                                   .refill_bytes_compressed = 1100,
                                   .runs_read_compressed    = 1200,
                                   .round_rows_min          = 25,
                                   .round_rows_max          = 70};

  SECTION("every counter is summed; min and max are kept")
  {
    merge::accumulate(a, b);
    CHECK(a.rounds == 101);
    CHECK(a.rows_out == 202);
    CHECK(a.refill_rows == 303);
    CHECK(a.refill_bytes == 404);
    CHECK(a.refill_truncated == 505);
    CHECK(a.window_growths == 606);
    CHECK(a.compacted_rows == 707);
    CHECK(a.fallback_conversions == 808);
    CHECK(a.refill_bytes_host == 909);
    CHECK(a.refill_bytes_disk == 1010);
    CHECK(a.refill_bytes_compressed == 1111);
    CHECK(a.runs_read_compressed == 1212);
    CHECK(a.round_rows_min == 25);
    CHECK(a.round_rows_max == 90);
  }

  SECTION("a larger min and a larger max")
  {
    merge::merge_round_stats into{.round_rows_min = 10, .round_rows_max = 20};
    merge::accumulate(into, b);
    CHECK(into.round_rows_min == 10);
    CHECK(into.round_rows_max == 70);
  }

  SECTION("a round_rows_min of 0 means no round yet and never wins")
  {
    merge::merge_round_stats empty;
    merge::accumulate(empty, b);
    CHECK(empty.round_rows_min == 25);
    CHECK(empty.round_rows_max == 70);

    merge::accumulate(a, merge::merge_round_stats{});
    CHECK(a.round_rows_min == 40);
    CHECK(a.round_rows_max == 90);

    merge::merge_round_stats both;
    merge::accumulate(both, merge::merge_round_stats{});
    CHECK(both.round_rows_min == 0);
    CHECK(both.round_rows_max == 0);
  }
}
