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
#include "helper/type_conversions.hpp"
#include "op/merge/detail/window_buffer.hpp"
#include "op/merge/merge_chunk_position.hpp"
#include "op/merge/merge_config.hpp"
#include "op/merge/windowed_merge_input.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_memory_history.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "sirius/exception.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_stream.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace sirius::op::merge;
using sirius::op::operator_data;
using sirius::op::operator_data_type;
using sirius::test::operator_utils::tiered_memory_fixture;

namespace {

constexpr std::size_t kMiB         = std::size_t{1} << 20;
constexpr std::size_t kRowWidth    = 32;
constexpr std::size_t kKeyRowWidth = 8;
// No run sliced in place: every run is charged its window buffers (the in-place cases build
// their inputs themselves).
constexpr std::size_t kGpuRuns          = 0;
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

/// The source of a pipeline that only has to exist.
class source_operator : public sirius::op::sirius_physical_operator {
 public:
  source_operator()
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::MERGE_SORT,
                               sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
                               0)
  {
  }

  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }
};

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

/// @p unbuffered_runs: the runs charged no window buffer. @p fetch_columns and @p string_columns:
/// the column counts the input was built with.
std::size_t expected_estimate(const window_config& cfg,
                              std::size_t unbuffered_runs = kGpuRuns,
                              std::size_t fetch_columns   = 0,
                              std::size_t string_columns  = 0)
{
  return estimate_task_bytes(make_cursors().size(),
                             unbuffered_runs,
                             cfg.window_rows,
                             kRowWidth,
                             kKeyRowWidth,
                             cfg.output_target_bytes,
                             kLargestFallback,
                             fetch_columns,
                             string_columns);
}

/// An input over make_cursors() with no compressed run, so its estimate has no reader term.
windowed_merge_input make_input_with_columns(window_config cfg,
                                             std::size_t fetch_columns,
                                             std::size_t string_columns)
{
  return windowed_merge_input(5,
                              make_cursors(),
                              false,
                              9,
                              cfg,
                              kRowWidth,
                              kKeyRowWidth,
                              kGpuRuns,
                              kLargestFallback,
                              0,
                              fetch_columns,
                              string_columns);
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

TEST_CASE("windowed_merge_input adds the reader state of its compressed runs to the estimate",
          "[windowed_merge_input]")
{
  using sirius::op::merge::estimate_reader_state_bytes;
  constexpr std::size_t compressed_runs = 2, fetch_columns = 5, string_columns = 1;
  windowed_merge_input input(5,
                             make_cursors(),
                             false,
                             9,
                             {.window_rows = 65536, .output_target_bytes = 64 * kMiB},
                             kRowWidth,
                             kKeyRowWidth,
                             kGpuRuns,
                             kLargestFallback,
                             compressed_runs,
                             fetch_columns,
                             string_columns);
  const auto reader = [&](std::size_t window_rows) {
    return estimate_reader_state_bytes(compressed_runs, fetch_columns, string_columns, window_rows);
  };
  const auto task = [&] {
    return expected_estimate(input.config(), kGpuRuns, fetch_columns, string_columns);
  };
  REQUIRE(reader(65536) > 0);
  REQUIRE(input.get_estimated_size_in_bytes() == task() + reader(65536));

  // The reader term follows the window when the task shrinks.
  REQUIRE(input.shrink_for_retry());
  REQUIRE(input.config().window_rows == 32768);
  REQUIRE(input.get_estimated_size_in_bytes() == task() + reader(32768));
}

TEST_CASE("windowed_merge_input reserves the masks and STRING slots of its window buffers",
          "[windowed_merge_input]")
{
  constexpr std::size_t columns = 3;
  const window_config cfg{.window_rows = 65536, .output_target_bytes = 64 * kMiB};
  // Two slots per run that is not sliced in place.
  const std::size_t slots = 2 * (make_cursors().size() - kGpuRuns);

  // The same task without its column counts: the rows of the windows only.
  const auto rows_only = make_input(cfg).get_estimated_size_in_bytes();
  REQUIRE(rows_only == expected_estimate(cfg));

  // Fixed-width columns: every column of a slot has a validity mask.
  const auto fixed_width = make_input_with_columns(cfg, columns, 0).get_estimated_size_in_bytes();
  REQUIRE(fixed_width == expected_estimate(cfg, kGpuRuns, columns, 0));
  REQUIRE(fixed_width >= rows_only + slots * columns * (cfg.window_rows / 8));

  // One of the columns is STRING: its offsets and the slack of its chars are added.
  const auto with_string = make_input_with_columns(cfg, columns, 1).get_estimated_size_in_bytes();
  REQUIRE(with_string == expected_estimate(cfg, kGpuRuns, columns, 1));
  REQUIRE(with_string >= fixed_width + slots * (cfg.window_rows * 8));

  // Each further STRING column adds its own offsets.
  REQUIRE(make_input_with_columns(cfg, columns, 2).get_estimated_size_in_bytes() > with_string);

  SECTION("the terms follow the window when the task shrinks")
  {
    auto input = make_input_with_columns(cfg, columns, 1);
    REQUIRE(input.shrink_for_retry());
    REQUIRE(input.get_estimated_size_in_bytes() ==
            expected_estimate(input.config(), kGpuRuns, columns, 1));
    REQUIRE(input.get_estimated_size_in_bytes() > expected_estimate(input.config()));
  }

  SECTION("a run whose buffers a kept merger holds is charged neither")
  {
    auto input = make_input_with_columns(cfg, columns, 1);
    input.set_kept_buffer_runs(make_cursors().size());
    REQUIRE(input.get_estimated_size_in_bytes() ==
            expected_estimate(cfg, make_cursors().size(), columns, 1));
    REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(cfg, make_cursors().size()));
  }
}

TEST_CASE("windowed_merge_input leaves out the window buffers a kept merger holds",
          "[windowed_merge_input]")
{
  const window_config cfg{.window_rows = 65536, .output_target_bytes = 64 * kMiB};
  auto input = make_input(cfg);
  REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(cfg));

  input.set_kept_buffer_runs(1);
  REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(cfg, kGpuRuns + 1));
  REQUIRE(input.get_estimated_size_in_bytes() < expected_estimate(cfg));
  REQUIRE(input.get_estimated_working_set_size_in_bytes() == input.get_estimated_size_in_bytes());

  // More kept runs than the task has buffered ones charge no buffer, and nothing below that.
  input.set_kept_buffer_runs(100);
  REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(cfg, make_cursors().size()));

  // The retry of a failed task builds a new merger: every buffer is charged again.
  REQUIRE(input.shrink_for_retry());
  REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(input.config()));

  SECTION("also when the sizes are at their floors")
  {
    auto at_floor =
      make_input({.window_rows = kShrinkWindowRowsFloor, .output_target_bytes = kOutputBytesFloor});
    at_floor.set_kept_buffer_runs(2);
    REQUIRE(at_floor.get_estimated_size_in_bytes() < expected_estimate(at_floor.config()));
    REQUIRE_FALSE(at_floor.shrink_for_retry());
    REQUIRE(at_floor.get_estimated_size_in_bytes() == expected_estimate(at_floor.config()));
  }
}

TEST_CASE("a windowed merge task reserves at least its working set when the history says less",
          "[windowed_merge_input]")
{
  const sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(build_ctx);
  source_operator source;
  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipeline, source);
  build_state.add_pipeline_operator(*pipeline, source);
  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());

  auto input = std::make_unique<windowed_merge_input>(
    make_input({.window_rows = 65536, .output_target_bytes = 64 * kMiB}));
  const auto working_set = input->get_estimated_working_set_size_in_bytes();
  // An earlier task of the pipeline reused a kept merger's windows: its peak was a hundredth of
  // its estimate.
  global_state->get_memory_history().record({working_set, working_set / 100, working_set / 100});

  auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    1,
    std::vector<cucascade::shared_data_repository*>{},
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input)),
    std::move(global_state));
  const auto estimate = task->get_estimated_reservation_size_info(nullptr);
  REQUIRE(estimate.had_history);
  REQUIRE(estimate.peak_memory_estimate == working_set);
  REQUIRE(estimate.reservation_size == working_set);
}

TEST_CASE("windowed_merge_input charges no window buffer for the runs sliced in place but one",
          "[windowed_merge_input]")
{
  const window_config cfg{.window_rows = 65536, .output_target_bytes = 64 * kMiB};
  const auto num_runs   = make_cursors().size();
  const auto input_with = [&](std::size_t gpu_runs) {
    return windowed_merge_input(
      5, make_cursors(), false, 9, cfg, kRowWidth, kKeyRowWidth, gpu_runs, kLargestFallback);
  };

  // One run in place: it is the spare, so every run is still charged.
  REQUIRE(input_with(1).get_estimated_size_in_bytes() == expected_estimate(cfg, 0));
  REQUIRE(input_with(1).gpu_runs() == 1);
  // Two and three in place: all but one of them are left out.
  REQUIRE(input_with(2).get_estimated_size_in_bytes() == expected_estimate(cfg, 1));
  REQUIRE(input_with(num_runs).get_estimated_size_in_bytes() ==
          expected_estimate(cfg, num_runs - 1));
  REQUIRE(input_with(num_runs).get_estimated_size_in_bytes() < expected_estimate(cfg, 0));
  // What is left with every run in place: the spare's two slots, the output and the scratch.
  REQUIRE(input_with(num_runs).get_estimated_size_in_bytes() ==
          expected_estimate(cfg, num_runs) + 2 * cfg.window_rows * kRowWidth);

  SECTION("a kept merger's runs are left out on top of them")
  {
    auto input = input_with(2);
    input.set_kept_buffer_runs(1);
    REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(cfg, 2));
  }

  SECTION("the first retry charges every run at unchanged sizes, later ones shrink")
  {
    auto input = input_with(num_runs);
    input.set_kept_buffer_runs(1);
    int shrinks = 0, recharges = 0;
    input.set_shrink_observer([&shrinks] { ++shrinks; });
    input.set_recharge_observer([&recharges] { ++recharges; });

    REQUIRE(input.shrink_for_retry());
    REQUIRE(recharges == 1);
    REQUIRE(shrinks == 0);
    REQUIRE(input.gpu_runs() == 0);
    REQUIRE(input.config().window_rows == cfg.window_rows);
    REQUIRE(input.config().output_target_bytes == cfg.output_target_bytes);
    // Every run, the kept merger's too (the failed task destroyed it).
    REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(cfg, 0));

    REQUIRE(input.shrink_for_retry());
    REQUIRE(recharges == 1);
    REQUIRE(shrinks == 1);
    REQUIRE(input.config().window_rows == cfg.window_rows / 2);
    REQUIRE(input.config().output_target_bytes == cfg.output_target_bytes / 2);
    REQUIRE(input.get_estimated_size_in_bytes() == expected_estimate(input.config(), 0));
  }

  SECTION("the recharge happens at the shrink floors too")
  {
    windowed_merge_input at_floor(
      5,
      make_cursors(),
      false,
      9,
      {.window_rows = kShrinkWindowRowsFloor, .output_target_bytes = kOutputBytesFloor},
      kRowWidth,
      kKeyRowWidth,
      2,
      kLargestFallback);
    const auto before = at_floor.get_estimated_size_in_bytes();
    REQUIRE(at_floor.shrink_for_retry());
    REQUIRE(at_floor.get_estimated_size_in_bytes() > before);
    REQUIRE_FALSE(at_floor.shrink_for_retry());
  }
}

TEST_CASE("windowed_merge_input sizes its output and scratch from the round",
          "[windowed_merge_input]")
{
  // 64 MiB of 32-byte rows: a round of 2,097,152 rows over windows of 65,536.
  window_config cfg{.window_rows = 65536, .output_target_bytes = 64 * kMiB};
  cfg.round_rows = round_rows_for(cfg.window_rows, cfg.output_target_bytes, kRowWidth);
  REQUIRE(cfg.round_rows == 64 * kMiB / kRowWidth);
  REQUIRE(cfg.effective_round_rows() == cfg.round_rows);
  REQUIRE(window_config{.window_rows = 7}.effective_round_rows() == 7);
  REQUIRE(cfg.same_sizes(cfg));
  REQUIRE_FALSE(cfg.same_sizes({.window_rows = 65536, .output_target_bytes = 64 * kMiB}));
  REQUIRE(window_config{.window_rows = 7, .output_target_bytes = 9}.same_sizes(
    {.window_rows = 7, .output_target_bytes = 9, .round_rows = 7}));

  const auto estimate_of = [](const windowed_merge_input& input) {
    return estimate_task_bytes(make_cursors().size(),
                               0,
                               input.config().window_rows,
                               kRowWidth,
                               kKeyRowWidth,
                               input.config().output_target_bytes,
                               kLargestFallback,
                               0,
                               0,
                               input.config().round_rows);
  };
  windowed_merge_input input(5,
                             make_cursors(),
                             false,
                             9,
                             cfg,
                             kRowWidth,
                             kKeyRowWidth,
                             0,
                             kLargestFallback,
                             0,
                             0,
                             0,
                             0,
                             true);
  REQUIRE(input.get_estimated_size_in_bytes() == estimate_of(input));
  REQUIRE(input.get_estimated_size_in_bytes() >
          make_input({.window_rows = 65536, .output_target_bytes = 64 * kMiB})
            .get_estimated_size_in_bytes());

  // A shrink halves the window and the target, and the round follows the target.
  REQUIRE(input.shrink_for_retry());
  REQUIRE(input.config().window_rows == 32768);
  REQUIRE(input.config().output_target_bytes == 32 * kMiB);
  REQUIRE(input.config().round_rows == 32 * kMiB / kRowWidth);
  REQUIRE(input.config().round_rows == round_rows_for(input.config().window_rows,
                                                      input.config().output_target_bytes,
                                                      kRowWidth));
  REQUIRE(input.get_estimated_size_in_bytes() == estimate_of(input));

  SECTION("a round that does not follow the output is halved, down to the window")
  {
    windowed_merge_input fixed(
      5,
      make_cursors(),
      false,
      9,
      {.window_rows = 4096, .output_target_bytes = 64 * kMiB, .round_rows = 6000},
      kRowWidth,
      kKeyRowWidth,
      0,
      kLargestFallback);
    REQUIRE(fixed.shrink_for_retry());
    REQUIRE(fixed.config().window_rows == 2048);
    REQUIRE(fixed.config().round_rows == 3000);
    REQUIRE(fixed.shrink_for_retry());
    REQUIRE(fixed.config().window_rows == 1024);
    REQUIRE(fixed.config().round_rows == 1500);
  }

  SECTION("a config without round rows keeps none")
  {
    auto plain = make_input({.window_rows = 4096, .output_target_bytes = 64 * kMiB});
    REQUIRE(plain.shrink_for_retry());
    REQUIRE(plain.config().round_rows == 0);
    REQUIRE(plain.config().effective_round_rows() == 2048);
  }
}

TEST_CASE("the STRING slot estimate covers what a window buffer allocates",
          "[windowed_merge_input][window_buffer]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;
  const rmm::cuda_stream_view stream_view{stream.value()};
  const auto mr = sirius::test::operator_utils::get_resource_ref(*f.gpu0);
  const cudf::data_type int64{cudf::type_id::INT64};
  const cudf::data_type int32{cudf::type_id::INT32};
  const cudf::data_type string{cudf::type_id::STRING};

  struct schema {
    std::vector<cudf::data_type> types;
    std::vector<std::size_t> string_widths;  ///< row width of each STRING column, offsets included
  };
  // The widths are what estimate_column_widths gives a STRING column: its chars per row plus
  // its 4-byte offset. few_distinct-strings of the benchmark is the second schema.
  const auto s    = GENERATE_COPY(schema{{string}, {24}},
                               schema{{string, int64, int64}, {20}},
                               schema{{int64, string, int32, string}, {12, 40}},
                               schema{{string, string, string}, {5, 9, 300}});
  const auto rows = GENERATE(std::size_t{1000}, std::size_t{65536}, std::size_t{200'000});
  CAPTURE(s.types.size(), rows);

  std::size_t row_width = 0, string_row_width = 0, strings = 0;
  std::vector<std::size_t> chars_estimates;
  for (const auto& type : s.types) {
    if (type.id() == cudf::type_id::STRING) {
      const auto width = s.string_widths[strings++];
      row_width += width;
      string_row_width += width;
      // What windowed_merger asks for when a batch has no column metadata: the column's share
      // of the row plus one.
      chars_estimates.push_back(width + 1);
    } else {
      row_width += cudf::size_of(type);
      chars_estimates.push_back(0);
    }
  }
  const sirius::op::merge::detail::window_buffer buffer(
    s.types, rows, chars_estimates, stream_view, mr);
  const auto allocated = buffer.device_bytes();

  const auto estimate =
    2 * estimate_window_slot_bytes(rows, row_width, s.types.size(), strings, string_row_width);
  // Covers the two slots, and by no more than the padding of the masks and of the minimum chars.
  REQUIRE(estimate >= allocated);
  REQUIRE(estimate - allocated <= 2 * (s.types.size() + strings) * 64);

  // The estimate without the STRING width is the over-reservation the rule removes.
  const auto legacy = 2 * estimate_window_slot_bytes(rows, row_width, s.types.size(), strings);
  REQUIRE(legacy > estimate);
  if (strings == 1 && rows >= 65536) {
    REQUIRE(static_cast<double>(legacy) > 1.1 * static_cast<double>(allocated));
  }
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
