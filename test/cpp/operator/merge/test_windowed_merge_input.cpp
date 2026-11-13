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
#include "op/merge/merge_chunk_position.hpp"
#include "op/merge/merge_config.hpp"
#include "op/merge/windowed_merge_input.hpp"
#include "operator/operator_test_utils.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_stream.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace sirius::op::merge;
using sirius::op::operator_data;
using sirius::op::operator_data_type;
using sirius::test::operator_utils::tiered_memory_fixture;

namespace {

constexpr std::size_t kMiB              = std::size_t{1} << 20;
constexpr std::size_t kRowWidth         = 32;
constexpr std::size_t kKeyRowWidth      = 8;
constexpr std::size_t kGpuRuns          = 1;
constexpr std::size_t kLargestFallback  = 3 * kMiB;
constexpr std::size_t kOutputBytesFloor = 16 * kMiB;

/// A cursor at (@p batch, @p row) of a run made of batches @p ids with @p rows rows each.
merge_input_cursor make_cursor(std::vector<std::uint64_t> ids,
                               std::vector<std::size_t> rows,
                               std::size_t batch = 0,
                               std::size_t row   = 0)
{
  return {.run   = {.batch_ids = std::move(ids), .batch_rows = std::move(rows)},
          .batch = batch,
          .row   = row};
}

std::vector<merge_input_cursor> make_cursors()
{
  std::vector<merge_input_cursor> cursors;
  cursors.push_back(make_cursor({11, 12}, {100, 200}, 1, 7));
  cursors.push_back(make_cursor({21}, {300}));
  cursors.push_back(make_cursor({31, 32, 33}, {10, 20, 30}, 2, 0));
  return cursors;
}

windowed_merge_input make_input(window_config cfg)
{
  return windowed_merge_input(
    5, make_cursors(), false, 9, cfg, kRowWidth, kKeyRowWidth, kGpuRuns, kLargestFallback);
}

std::size_t expected_estimate(const window_config& cfg)
{
  return estimate_task_bytes(make_cursors().size(),
                             kGpuRuns,
                             cfg.window_rows,
                             kRowWidth,
                             kKeyRowWidth,
                             cfg.output_target_bytes,
                             kLargestFallback);
}

}  // namespace

TEST_CASE("windowed_merge_input exposes its fields and a window-sized estimate",
          "[windowed_merge_input]")
{
  const window_config cfg{.window_rows = 65536, .output_target_bytes = 64 * kMiB};
  auto input = make_input(cfg);

  const operator_data& base = input;
  REQUIRE(base.get_type() == operator_data_type::WINDOWED_MERGE);

  REQUIRE(input.partition() == 5);
  REQUIRE_FALSE(input.final_pass());
  REQUIRE(input.first_chunk() == 9);
  REQUIRE(input.config().window_rows == cfg.window_rows);
  REQUIRE(input.config().output_target_bytes == cfg.output_target_bytes);

  const auto& cursors = input.cursors();
  REQUIRE(cursors.size() == 3);
  REQUIRE(cursors[0].run.batch_ids == std::vector<std::uint64_t>{11, 12});
  REQUIRE(cursors[0].run.batch_rows == std::vector<std::size_t>{100, 200});
  REQUIRE(cursors[0].batch == 1);
  REQUIRE(cursors[0].row == 7);
  REQUIRE(cursors[1].run.batch_ids == std::vector<std::uint64_t>{21});
  REQUIRE(cursors[1].batch == 0);
  REQUIRE(cursors[1].row == 0);
  REQUIRE(cursors[2].run.batch_ids == std::vector<std::uint64_t>{31, 32, 33});
  REQUIRE(cursors[2].batch == 2);

  REQUIRE(windowed_merge_input(0, make_cursors(), true, 0, cfg, 1, 1, 0, 0).final_pass());

  REQUIRE_FALSE(input.get_preferred_device_id().has_value());
  input.set_preferred_device_id(1);
  REQUIRE(input.get_preferred_device_id() == 1);

  const auto expected = expected_estimate(cfg);
  REQUIRE(input.get_estimated_size_in_bytes() == expected);
  REQUIRE(base.get_estimated_working_set_size_in_bytes() == expected);
}

TEST_CASE("windowed_merge_input shrinks by halving down to both floors", "[windowed_merge_input]")
{
  // window_rows: 5000 -> 2500 -> 1250 -> 1024; output: 100 -> 50 -> 25 -> 16 MiB.
  auto input          = make_input({.window_rows = 5000, .output_target_bytes = 100 * kMiB});
  operator_data& base = input;

  const std::vector<std::pair<std::size_t, std::size_t>> expected_steps = {
    {2500, 50 * kMiB}, {1250, 25 * kMiB}, {kShrinkWindowRowsFloor, kOutputBytesFloor}};

  auto previous_estimate = input.get_estimated_size_in_bytes();
  for (const auto& [window_rows, output_target_bytes] : expected_steps) {
    REQUIRE(base.shrink_for_retry());
    REQUIRE(input.config().window_rows == window_rows);
    REQUIRE(input.config().output_target_bytes == output_target_bytes);

    const auto estimate = input.get_estimated_size_in_bytes();
    REQUIRE(estimate == expected_estimate(input.config()));
    REQUIRE(estimate < previous_estimate);
    previous_estimate = estimate;
  }

  REQUIRE_FALSE(base.shrink_for_retry());
  REQUIRE(input.config().window_rows == kShrinkWindowRowsFloor);
  REQUIRE(input.config().output_target_bytes == kOutputBytesFloor);
  REQUIRE(input.get_estimated_size_in_bytes() == previous_estimate);
}

TEST_CASE("windowed_merge_input keeps shrinking while either size is above its floor",
          "[windowed_merge_input]")
{
  SECTION("only the window is above its floor")
  {
    auto input = make_input({.window_rows = 4096, .output_target_bytes = kOutputBytesFloor});
    REQUIRE(input.shrink_for_retry());
    REQUIRE(input.config().window_rows == 2048);
    REQUIRE(input.config().output_target_bytes == kOutputBytesFloor);
    REQUIRE(input.shrink_for_retry());
    REQUIRE(input.config().window_rows == kShrinkWindowRowsFloor);
    REQUIRE_FALSE(input.shrink_for_retry());
  }
  SECTION("only the output target is above its floor")
  {
    auto input =
      make_input({.window_rows = kShrinkWindowRowsFloor, .output_target_bytes = 24 * kMiB});
    REQUIRE(input.shrink_for_retry());
    REQUIRE(input.config().window_rows == kShrinkWindowRowsFloor);
    REQUIRE(input.config().output_target_bytes == kOutputBytesFloor);
    REQUIRE_FALSE(input.shrink_for_retry());
  }
  SECTION("sizes already below their floors are not raised")
  {
    auto input = make_input({.window_rows = 100, .output_target_bytes = kMiB});
    REQUIRE_FALSE(input.shrink_for_retry());
    REQUIRE(input.config().window_rows == 100);
    REQUIRE(input.config().output_target_bytes == kMiB);
  }
}

TEST_CASE("windowed_merge_input captures the GPU space it is prepared in", "[windowed_merge_input]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;

  auto input = make_input({.window_rows = 4096, .output_target_bytes = 32 * kMiB});
  REQUIRE_THROWS_AS(input.gpu_memory_space(), std::runtime_error);

  const auto gpu_available_before = f.gpu0->get_available_memory();
  input.prepare_for_processing(f.gpu0, stream);
  REQUIRE(&input.gpu_memory_space() == f.gpu0);
  REQUIRE(f.gpu0->get_available_memory() == gpu_available_before);

  input.prepare_for_processing(f.host0, stream);
  REQUIRE_THROWS_AS(input.gpu_memory_space(), std::runtime_error);

  input.prepare_for_processing(nullptr, stream);
  REQUIRE_THROWS_AS(input.gpu_memory_space(), std::runtime_error);
}

TEST_CASE("merge chunk positions round-trip and reject values of 2^32 or more",
          "[windowed_merge_input]")
{
  constexpr std::size_t limit = std::size_t{1} << 32;
  constexpr std::size_t last  = limit - 1;

  REQUIRE(make_merge_chunk_position(0, 0) == 0);
  REQUIRE(make_merge_chunk_position(0, 7) == 7);
  REQUIRE(make_merge_chunk_position(1, 0) == limit);
  REQUIRE(make_merge_chunk_position(3, 5) == ((std::size_t{3} << 32) | 5));

  const std::vector<std::pair<std::size_t, std::size_t>> cases = {
    {0, 0}, {0, 1}, {1, 0}, {3, 5}, {last, 0}, {0, last}, {last, last}, {12345, 67890}};
  for (const auto& [partition, chunk] : cases) {
    const auto [split_partition, split_chunk] =
      split_merge_chunk_position(make_merge_chunk_position(partition, chunk));
    REQUIRE(split_partition == partition);
    REQUIRE(split_chunk == chunk);
  }

  // Positions order by partition first, then by chunk.
  REQUIRE(make_merge_chunk_position(1, last) < make_merge_chunk_position(2, 0));
  REQUIRE(make_merge_chunk_position(2, 0) < make_merge_chunk_position(2, 1));

  REQUIRE_THROWS_AS(make_merge_chunk_position(limit, 0), sirius::internal_exception);
  REQUIRE_THROWS_AS(make_merge_chunk_position(0, limit), sirius::internal_exception);
  REQUIRE_THROWS_AS(make_merge_chunk_position(limit, limit), sirius::internal_exception);
}
