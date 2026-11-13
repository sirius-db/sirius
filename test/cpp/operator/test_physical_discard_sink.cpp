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
#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_discard_sink.hpp"
#include "operator_test_utils.hpp"
#include "sirius/exception.hpp"
#include "utils/log_test_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/data/data_repository.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
using cucascade::data_batch;

namespace {

using batch_vector = std::vector<std::shared_ptr<data_batch>>;

sirius_physical_discard_sink make_sink(duckdb::LogicalType type, discard_sink_options options = {})
{
  return sirius_physical_discard_sink(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{std::move(type)}),
    std::move(options),
    1000);
}

/// Order check on column 0.
discard_sink_options check_first_column(cudf::order order           = cudf::order::ASCENDING,
                                        cudf::null_order null_order = cudf::null_order::BEFORE)
{
  return {.check_order     = true,
          .key_columns     = {0},
          .column_order    = {order},
          .null_precedence = {null_order}};
}

std::shared_ptr<data_batch> int64_batch(const std::vector<std::int64_t>& values)
{
  return make_numeric_batch<std::int64_t>(*get_default_gpu_space(), values, cudf::type_id::INT64);
}

/// INT64 batch where a `std::nullopt` entry is NULL.
std::shared_ptr<data_batch> nullable_int64_batch(
  const std::vector<std::optional<std::int64_t>>& values)
{
  std::vector<std::int64_t> data;
  std::vector<bool> valid;
  for (auto const& value : values) {
    data.push_back(value.value_or(0));
    valid.push_back(value.has_value());
  }
  return make_numeric_batch_with_nulls<std::int64_t>(
    *get_default_gpu_space(), data, valid, cudf::type_id::INT64);
}

/// Runs one task over @p batches and returns the single count the sink emits for it.
std::int64_t discard(sirius_physical_discard_sink& sink,
                     batch_vector batches,
                     std::optional<std::size_t> position)
{
  pipelineable_operator_data input(std::move(batches), position);
  auto output         = sink.execute(input, default_stream());
  auto const& counted = dynamic_cast<const pipelineable_operator_data&>(*output);
  REQUIRE(counted.get_position() == position);
  REQUIRE(counted.get_data_batches().size() == 1);
  auto const view = sirius::get_cudf_table_view(*counted.get_data_batches().front());
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.num_rows() == 1);
  REQUIRE(view.column(0).type().id() == cudf::type_id::INT64);
  return copy_column_to_host<std::int64_t>(view.column(0)).front();
}

}  // namespace

TEST_CASE("sirius_physical_discard_sink counts rows and bytes and emits one count row per task",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT);
  CHECK(sink.get_name() == "DISCARD_SINK");
  CHECK(sink.output_order_preservation() == sirius::OrderPreservationType::INSERTION_ORDER);
  CHECK(sink.no_history_peak_memory_estimate({.num_batches = 4, .bytes = std::size_t{1} << 30}) <=
        (std::size_t{1} << 20));

  CHECK(discard(sink, {int64_batch({1, 2, 3})}, 7) == 3);
  CHECK(sink.rows_discarded() == 3);
  CHECK(sink.bytes_discarded() == 3 * sizeof(std::int64_t));

  // A task with several batches emits their total, and one without a position keeps none.
  CHECK(discard(sink,
                {int64_batch({4, 5}), int64_batch({}), int64_batch({6, 7, 8, 9})},
                std::nullopt) == 6);
  CHECK(sink.rows_discarded() == 9);
  CHECK(sink.bytes_discarded() == 9 * sizeof(std::int64_t));
  CHECK(sink.order_violations() == 0);

  // A task without batches emits nothing.
  pipelineable_operator_data empty(batch_vector{}, 3);
  auto output          = sink.execute(empty, default_stream());
  auto const& no_count = dynamic_cast<const pipelineable_operator_data&>(*output);
  CHECK(no_count.get_data_batches().empty());
  CHECK(no_count.get_position() == std::optional<std::size_t>{3});
  CHECK(sink.rows_discarded() == 9);
}

TEST_CASE("sirius_physical_discard_sink keeps no reference to its input", "[physical_discard_sink]")
{
  for (bool check_order : {false, true}) {
    INFO("check_order=" << check_order);
    auto sink  = make_sink(duckdb::LogicalType::BIGINT,
                          check_order ? check_first_column() : discard_sink_options{});
    auto batch = int64_batch({1, 2, 3});
    REQUIRE(batch.use_count() == 1);
    {
      pipelineable_operator_data input(batch_vector{batch}, 0);
      input.prepare_for_processing(get_default_gpu_space(), default_stream());
      // The test's reference plus the ones the prepared input holds.
      auto const held_by_input = batch.use_count();
      auto output              = sink.execute(input, default_stream());
      // The output is a new table, not the input.
      auto const& counted = dynamic_cast<const pipelineable_operator_data&>(*output);
      REQUIRE(counted.get_data_batches().size() == 1);
      CHECK(counted.get_data_batches().front().get() != batch.get());
      CHECK(batch.use_count() == held_by_input);
    }
    CHECK(batch.use_count() == 1);
    sink.finish();
    CHECK(batch.use_count() == 1);
  }
}

TEST_CASE("sirius_physical_discard_sink order check accepts sorted input",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  // Tasks finish in any order; the positions say where their rows belong.
  discard(sink, {int64_batch({20, 20, 30})}, 2);
  discard(sink, {int64_batch({1, 2, 3})}, 0);
  discard(sink, {int64_batch({3, 10}), int64_batch({}), int64_batch({10, 20})}, 1);
  CHECK(sink.order_violations() == 0);
  sink.finish();
  CHECK(sink.order_violations() == 0);
  CHECK(sink.rows_discarded() == 10);
}

TEST_CASE("sirius_physical_discard_sink order check counts a shuffled batch",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  discard(sink, {int64_batch({1, 2, 3})}, 0);
  CHECK(sink.order_violations() == 0);
  // Its first and last rows fit between its neighbours, so only the batch itself is wrong.
  discard(sink, {int64_batch({4, 9, 5, 7, 10})}, 1);
  CHECK(sink.order_violations() == 1);
  discard(sink, {int64_batch({10, 11})}, 2);
  sink.finish();
  CHECK(sink.order_violations() == 1);
}

TEST_CASE("sirius_physical_discard_sink order check counts batches out of order across positions",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  discard(sink, {int64_batch({10, 20})}, 0);
  discard(sink, {int64_batch({5, 6})}, 1);
  discard(sink, {int64_batch({6, 7})}, 2);
  // Each batch is sorted: the fault only shows once the positions are walked.
  CHECK(sink.order_violations() == 0);

  sirius::test::scoped_recording_log_sink log("info");
  sink.finish();
  CHECK(sink.order_violations() == 1);
  // A second call neither counts nor logs again.
  sink.finish();
  CHECK(sink.order_violations() == 1);

  auto const records = log.records();
  auto const summary = std::count_if(records.begin(), records.end(), [](auto const& record) {
    return record.message == "[discard_sink] rows=6 bytes=48 order_violations=1";
  });
  CHECK(summary == 1);
}

TEST_CASE("sirius_physical_discard_sink order check covers the seam between batches of one task",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  discard(sink, {int64_batch({1, 5}), int64_batch({4, 9})}, 0);
  CHECK(sink.order_violations() == 1);
  // The task's range runs from its first batch's first row to its last batch's last row.
  discard(sink, {int64_batch({8, 12})}, 1);
  sink.finish();
  CHECK(sink.order_violations() == 2);
}

TEST_CASE("sirius_physical_discard_sink without order check ignores unsorted input",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT);
  discard(sink, {int64_batch({3, 1, 2})}, 1);
  discard(sink, {int64_batch({0})}, 0);
  sink.finish();
  CHECK(sink.order_violations() == 0);
  CHECK(sink.rows_discarded() == 4);

  // `check_order` without key columns checks nothing either.
  auto keyless = make_sink(duckdb::LogicalType::BIGINT, {.check_order = true});
  discard(keyless, {int64_batch({3, 1, 2})}, 0);
  keyless.finish();
  CHECK(keyless.order_violations() == 0);
}

TEST_CASE("sirius_physical_discard_sink order check honours descending keys",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column(cudf::order::DESCENDING));
  discard(sink, {int64_batch({9, 7, 7})}, 0);
  discard(sink, {int64_batch({7, 2})}, 1);
  sink.finish();
  CHECK(sink.order_violations() == 0);

  auto ascending_input =
    make_sink(duckdb::LogicalType::BIGINT, check_first_column(cudf::order::DESCENDING));
  discard(ascending_input, {int64_batch({1, 2})}, 0);  // within the batch
  discard(ascending_input, {int64_batch({5, 4})}, 1);  // across the two batches
  ascending_input.finish();
  CHECK(ascending_input.order_violations() == 2);
}

TEST_CASE("sirius_physical_discard_sink order check honours the null placement",
          "[physical_discard_sink]")
{
  using cudf::null_order;
  using cudf::order;
  const std::optional<std::int64_t> null;

  SECTION("ascending, nulls first")
  {
    auto sink = make_sink(duckdb::LogicalType::BIGINT,
                          check_first_column(order::ASCENDING, null_order::BEFORE));
    discard(sink, {nullable_int64_batch({null, null})}, 0);
    discard(sink, {nullable_int64_batch({null, 1, 2})}, 1);
    sink.finish();
    CHECK(sink.order_violations() == 0);
  }
  SECTION("ascending, nulls last")
  {
    auto sink = make_sink(duckdb::LogicalType::BIGINT,
                          check_first_column(order::ASCENDING, null_order::AFTER));
    discard(sink, {nullable_int64_batch({1, 2})}, 0);
    discard(sink, {nullable_int64_batch({2, null})}, 1);
    discard(sink, {nullable_int64_batch({null, null})}, 2);
    sink.finish();
    CHECK(sink.order_violations() == 0);
  }
  SECTION("nulls first input fails a nulls last check")
  {
    auto sink = make_sink(duckdb::LogicalType::BIGINT,
                          check_first_column(order::ASCENDING, null_order::AFTER));
    discard(sink, {nullable_int64_batch({null, 1})}, 0);  // within the batch
    CHECK(sink.order_violations() == 1);
    discard(sink, {nullable_int64_batch({2, null})}, 1);
    discard(sink, {nullable_int64_batch({3, 4})}, 2);  // 3 belongs before the null
    sink.finish();
    CHECK(sink.order_violations() == 2);
  }
  SECTION("descending reverses the null placement, as cudf's sort does")
  {
    // cudf applies the null placement before it reverses a descending column, so BEFORE puts the
    // nulls last there.
    auto sink = make_sink(duckdb::LogicalType::BIGINT,
                          check_first_column(order::DESCENDING, null_order::BEFORE));
    discard(sink, {nullable_int64_batch({5, 4})}, 0);
    discard(sink, {nullable_int64_batch({3, null})}, 1);
    discard(sink, {nullable_int64_batch({null})}, 2);
    sink.finish();
    CHECK(sink.order_violations() == 0);

    auto nulls_first = make_sink(duckdb::LogicalType::BIGINT,
                                 check_first_column(order::DESCENDING, null_order::BEFORE));
    discard(nulls_first, {nullable_int64_batch({null, 5})}, 0);  // within the batch
    discard(nulls_first, {nullable_int64_batch({4, 3})}, 1);
    nulls_first.finish();
    CHECK(nulls_first.order_violations() == 1);

    auto null_then_value = make_sink(duckdb::LogicalType::BIGINT,
                                     check_first_column(order::DESCENDING, null_order::BEFORE));
    discard(null_then_value, {nullable_int64_batch({5, null})}, 0);
    discard(null_then_value, {nullable_int64_batch({3})}, 1);  // across the two batches
    null_then_value.finish();
    CHECK(null_then_value.order_violations() == 1);
  }
}

TEST_CASE("sirius_physical_discard_sink order check compares string keys",
          "[physical_discard_sink]")
{
  auto string_batch = [](const std::vector<std::string>& values) {
    return make_string_batch(*get_default_gpu_space(), values);
  };

  auto sink = make_sink(duckdb::LogicalType::VARCHAR, check_first_column());
  discard(sink, {string_batch({"", "apple", "apple pie"})}, 0);
  discard(sink, {string_batch({"apple pie", "banana"})}, 1);
  // Bytes compare unsigned: a multi-byte UTF-8 character sorts after ASCII.
  discard(sink, {string_batch({"zebra", "\xC3\xA9t\xC3\xA9"})}, 2);
  sink.finish();
  CHECK(sink.order_violations() == 0);
  CHECK(sink.rows_discarded() == 7);

  auto unsorted = make_sink(duckdb::LogicalType::VARCHAR, check_first_column());
  discard(unsorted, {string_batch({"cherry", "banana", "date"})}, 0);  // within the batch
  CHECK(unsorted.order_violations() == 1);
  discard(unsorted, {string_batch({"dat", "fig"})}, 1);  // "dat" belongs before "date"
  unsorted.finish();
  CHECK(unsorted.order_violations() == 2);
}

TEST_CASE("sirius_physical_discard_sink order check compares several keys in turn",
          "[physical_discard_sink]")
{
  auto two_keys = [](const std::vector<std::int64_t>& first,
                     const std::vector<std::int32_t>& second) {
    return make_two_column_batch<std::int64_t, std::int32_t>(
      *get_default_gpu_space(), first, second, cudf::type_id::INT32);
  };
  auto make = [](cudf::order second_order) {
    return sirius_physical_discard_sink(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT,
                                                                  duckdb::LogicalType::INTEGER}),
      {.check_order     = true,
       .key_columns     = {0, 1},
       .column_order    = {cudf::order::ASCENDING, second_order},
       .null_precedence = {cudf::null_order::BEFORE, cudf::null_order::BEFORE}},
      1000);
  };

  // The first key ties across the two batches, so the second one decides.
  auto ascending = make(cudf::order::ASCENDING);
  discard(ascending, {two_keys({1, 1}, {5, 7})}, 0);
  discard(ascending, {two_keys({1, 2}, {6, 0})}, 1);
  ascending.finish();
  CHECK(ascending.order_violations() == 1);

  auto descending = make(cudf::order::DESCENDING);
  discard(descending, {two_keys({1, 1}, {9, 7})}, 0);
  discard(descending, {two_keys({1, 2}, {6, 8})}, 1);
  descending.finish();
  CHECK(descending.order_violations() == 0);
}

TEST_CASE("sirius_physical_discard_sink rejects order options that do not match the keys",
          "[physical_discard_sink]")
{
  CHECK_THROWS_AS(make_sink(duckdb::LogicalType::BIGINT,
                            {.check_order  = true,
                             .key_columns  = {0},
                             .column_order = {cudf::order::ASCENDING, cudf::order::ASCENDING}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
    make_sink(duckdb::LogicalType::BIGINT,
              {.check_order     = true,
               .key_columns     = {0},
               .null_precedence = {cudf::null_order::BEFORE, cudf::null_order::AFTER}}),
    std::invalid_argument);
}

TEST_CASE("sirius_physical_discard_sink counts a HOST batch without converting it",
          "[physical_discard_sink]")
{
  tiered_memory_fixture fixture;
  REQUIRE(fixture.setup());
  rmm::cuda_stream stream;  // the GPU -> HOST conversion needs a stream of its own
  auto const rows = tiered_memory_fixture::default_num_rows;

  auto gpu_batch  = fixture.make_gpu_batch(stream);
  auto host_batch = fixture.make_host_batch(stream);
  REQUIRE(host_batch->to_read_only().get_current_tier() == cucascade::memory::Tier::HOST);

  auto sink = make_sink(duckdb::LogicalType::BIGINT);
  CHECK(discard(sink, {gpu_batch, host_batch}, 4) == static_cast<std::int64_t>(2 * rows));
  CHECK(sink.rows_discarded() == 2 * rows);
  CHECK(sink.bytes_discarded() == 2 * rows * sizeof(std::int64_t));
  CHECK(host_batch->to_read_only().get_current_tier() == cucascade::memory::Tier::HOST);
  CHECK(host_batch.use_count() == 1);

  // An input that did not come from the sink's port and was never prepared names no GPU memory
  // space for the count row. Nothing is counted for the failed task.
  pipelineable_operator_data host_only(batch_vector{host_batch}, 5);
  CHECK_THROWS_AS(sink.execute(host_only, stream), sirius::internal_exception);
  CHECK(sink.rows_discarded() == 2 * rows);

  // The order check reads the keys on the GPU, so it refuses a batch that was not converted.
  auto checking = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  pipelineable_operator_data mixed(batch_vector{gpu_batch, host_batch}, 0);
  CHECK_THROWS_AS(checking.execute(mixed, stream), sirius::internal_exception);
  CHECK(checking.rows_discarded() == 0);
  CHECK(host_batch->to_read_only().get_current_tier() == cucascade::memory::Tier::HOST);
}

TEST_CASE("sirius_physical_discard_sink compares tasks that share a position in arrival order",
          "[physical_discard_sink]")
{
  auto in_order = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  discard(in_order, {int64_batch({1, 2})}, 0);
  discard(in_order, {int64_batch({3, 4})}, 0);
  discard(in_order, {int64_batch({5})}, 1);
  in_order.finish();
  CHECK(in_order.order_violations() == 0);

  auto reversed = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  discard(reversed, {int64_batch({3, 4})}, 0);
  discard(reversed, {int64_batch({1, 2})}, 0);  // arrived second, so it is checked after {3, 4}
  reversed.finish();
  CHECK(reversed.order_violations() == 1);
}

TEST_CASE(
  "sirius_physical_discard_sink leaves tasks without a position out of the cross-task check",
  "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  discard(sink, {int64_batch({10, 20})}, 0);
  // Sorted on its own, and with no position nothing places it relative to the other tasks.
  discard(sink, {int64_batch({1, 2})}, std::nullopt);
  discard(sink, {int64_batch({30, 40})}, 1);
  sink.finish();
  CHECK(sink.order_violations() == 0);
  CHECK(sink.rows_discarded() == 6);

  // The seam between batches of a positionless task is still checked.
  auto seams = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  discard(seams, {int64_batch({5, 6}), int64_batch({1, 2})}, std::nullopt);
  CHECK(seams.order_violations() == 1);
  seams.finish();
  CHECK(seams.order_violations() == 1);
}

TEST_CASE("sirius_physical_discard_sink order check sorts NaN after every float, as cudf does",
          "[physical_discard_sink]")
{
  auto double_batch = [](const std::vector<double>& values) {
    return make_numeric_batch<double>(*get_default_gpu_space(), values, cudf::type_id::FLOAT64);
  };
  auto const nan = std::numeric_limits<double>::quiet_NaN();

  auto sink = make_sink(duckdb::LogicalType::DOUBLE, check_first_column());
  discard(sink, {double_batch({-1.5, 0.0, 2.25})}, 0);
  discard(sink, {double_batch({1e300, nan})}, 1);
  discard(sink, {double_batch({nan, nan})}, 2);
  sink.finish();
  CHECK(sink.order_violations() == 0);

  auto nan_first = make_sink(duckdb::LogicalType::DOUBLE, check_first_column());
  discard(nan_first, {double_batch({1.0, nan})}, 0);
  discard(nan_first, {double_batch({2.0, 3.0})}, 1);  // 2.0 belongs before the NaN
  nan_first.finish();
  CHECK(nan_first.order_violations() == 1);
}

TEST_CASE("sirius_physical_discard_sink order check compares timestamp and decimal keys",
          "[physical_discard_sink]")
{
  SECTION("timestamps")
  {
    auto ts_batch = [](const std::vector<std::int64_t>& values) {
      return make_timestamp_batch<std::int64_t>(
        *get_default_gpu_space(), values, cudf::type_id::TIMESTAMP_MICROSECONDS);
    };
    auto sink = make_sink(duckdb::LogicalType::TIMESTAMP, check_first_column());
    discard(sink, {ts_batch({-5, 100})}, 0);
    discard(sink, {ts_batch({100, 200})}, 1);
    discard(sink, {ts_batch({150, 300})}, 2);  // 150 belongs before 200
    sink.finish();
    CHECK(sink.order_violations() == 1);
  }
  SECTION("decimals")
  {
    auto dec_batch = [](const std::vector<std::int64_t>& values) {
      return make_decimal64_batch(*get_default_gpu_space(), values, -2);
    };
    auto sink = make_sink(duckdb::LogicalType::DECIMAL(18, 2), check_first_column());
    discard(sink, {dec_batch({-300, 5})}, 0);
    discard(sink, {dec_batch({5, 1000})}, 1);
    sink.finish();
    CHECK(sink.order_violations() == 0);

    auto unsorted = make_sink(duckdb::LogicalType::DECIMAL(18, 2), check_first_column());
    discard(unsorted, {dec_batch({100, 200})}, 0);
    discard(unsorted, {dec_batch({-100, 300})}, 1);
    unsorted.finish();
    CHECK(unsorted.order_violations() == 1);
  }
}

TEST_CASE("sirius_physical_discard_sink logs its summary once even without an order check",
          "[physical_discard_sink]")
{
  auto sink = make_sink(duckdb::LogicalType::BIGINT);
  discard(sink, {int64_batch({3, 1, 2})}, 0);
  discard(sink, {int64_batch({7})}, std::nullopt);

  sirius::test::scoped_recording_log_sink log("info");
  sink.finish();
  sink.finish();
  auto const records = log.records();
  auto const summary = std::count_if(records.begin(), records.end(), [](auto const& record) {
    return record.message == "[discard_sink] rows=4 bytes=32 order_violations=0";
  });
  CHECK(summary == 1);
}

namespace {

/// Wires a "default" input port to @p sink the way the repository wiring does, policy included.
void attach_input_port(sirius_physical_discard_sink& sink, cucascade::shared_data_repository& repo)
{
  auto port           = std::make_unique<sirius_physical_operator::port>();
  port->type          = MemoryBarrierType::PIPELINE;
  port->repo          = &repo;
  port->src_pipeline  = nullptr;
  port->dest_pipeline = nullptr;
  port->residency     = sink.input_residency_policy("default");
  repo.set_downgrade_priority(port->residency.downgrade_priority);
  sink.add_port("default", std::move(port));
}

}  // namespace

TEST_CASE(
  "sirius_physical_discard_sink input port skips GPU materialization unless it checks order",
  "[physical_discard_sink]")
{
  auto const counting = make_sink(duckdb::LogicalType::BIGINT).input_residency_policy("default");
  CHECK_FALSE(counting.materialize_inputs_on_gpu);
  // Spilled last: below the default (0) and the merge sort's inputs (100).
  CHECK(counting.downgrade_priority < 0);

  auto const checking =
    make_sink(duckdb::LogicalType::BIGINT, check_first_column()).input_residency_policy("default");
  CHECK(checking.materialize_inputs_on_gpu);
  CHECK(checking.downgrade_priority == counting.downgrade_priority);

  // `check_order` without key columns checks nothing, so it needs nothing on the GPU.
  auto const keyless =
    make_sink(duckdb::LogicalType::BIGINT, {.check_order = true}).input_residency_policy("default");
  CHECK_FALSE(keyless.materialize_inputs_on_gpu);
}

TEST_CASE("sirius_physical_discard_sink counts a task whose batches all spilled, without an upload",
          "[physical_discard_sink]")
{
  tiered_memory_fixture fixture;
  REQUIRE(fixture.setup());
  rmm::cuda_stream stream;
  auto const rows = tiered_memory_fixture::default_num_rows;

  auto sink = make_sink(duckdb::LogicalType::BIGINT);
  cucascade::shared_data_repository repo;
  attach_input_port(sink, repo);

  auto host_batch = fixture.make_host_batch(stream);
  sink.push_data_batch_at_position("default", host_batch, 6);

  auto input = sink.get_next_task_input_data();
  REQUIRE(input != nullptr);
  CHECK(sink.get_next_task_input_data() == nullptr);
  auto const& task_input = dynamic_cast<const pipelineable_operator_data&>(*input);
  CHECK_FALSE(task_input.materializes_on_gpu());
  CHECK(task_input.get_position() == std::optional<std::size_t>{6});

  // What the task does: prepare for its GPU memory space, then run.
  input->prepare_for_processing(fixture.gpu0, stream);
  CHECK(host_batch->to_read_only().get_current_tier() == cucascade::memory::Tier::HOST);
  auto output = sink.execute(*input, stream);
  CHECK(host_batch->to_read_only().get_current_tier() == cucascade::memory::Tier::HOST);
  CHECK(sink.rows_discarded() == rows);
  CHECK(sink.bytes_discarded() == rows * sizeof(std::int64_t));

  // The count row is in the task's memory space, not derived from an input batch.
  auto const& counted = dynamic_cast<const pipelineable_operator_data&>(*output);
  CHECK(counted.get_position() == std::optional<std::size_t>{6});
  REQUIRE(counted.get_data_batches().size() == 1);
  auto const count_row = counted.get_data_batches().front()->to_read_only();
  CHECK(count_row.get_current_tier() == cucascade::memory::Tier::GPU);
  CHECK(count_row.get_memory_space() == fixture.gpu0);
  auto const view = sirius::get_cudf_table_view(count_row);
  REQUIRE(view.num_rows() == 1);
  CHECK(copy_column_to_host<std::int64_t>(view.column(0)).front() ==
        static_cast<std::int64_t>(rows));
}

TEST_CASE("sirius_physical_discard_sink order check has its spilled input brought to the GPU",
          "[physical_discard_sink]")
{
  tiered_memory_fixture fixture;
  REQUIRE(fixture.setup());
  rmm::cuda_stream stream;
  auto const rows = tiered_memory_fixture::default_num_rows;

  auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
  cucascade::shared_data_repository repo;
  attach_input_port(sink, repo);

  auto host_batch = fixture.make_host_batch(stream);
  sink.push_data_batch_at_position("default", host_batch, 0);

  auto input = sink.get_next_task_input_data();
  REQUIRE(input != nullptr);
  CHECK(dynamic_cast<const pipelineable_operator_data&>(*input).materializes_on_gpu());
  input->prepare_for_processing(fixture.gpu0, stream);
  auto output = sink.execute(*input, stream);
  CHECK(sink.rows_discarded() == rows);
  // The fixture's batch holds random values, so the check ran and found it unsorted.
  CHECK(sink.order_violations() == 1);
}

TEST_CASE("sirius_physical_discard_sink declares the count it emits, not the schema it drops",
          "[physical_discard_sink]")
{
  auto const options = check_first_column(cudf::order::DESCENDING, cudf::null_order::AFTER);
  sirius_physical_discard_sink sink(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
      duckdb::LogicalType::BIGINT, duckdb::LogicalType::VARCHAR, duckdb::LogicalType::DOUBLE}),
    options,
    1000);
  // The task-level schema diagnostic compares get_types() with what execute() returns.
  CHECK(sink.get_types() ==
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}));
  CHECK(sink.input_types().size() == 3);
  CHECK(sink.options().check_order);
  CHECK(sink.options().key_columns == options.key_columns);
  CHECK(sink.options().column_order == options.column_order);
  CHECK(sink.options().null_precedence == options.null_precedence);
  // Its input streams whatever produces it.
  CHECK(sink.input_barrier_for(sink) == MemoryBarrierType::PIPELINE);
}

TEST_CASE("sirius_physical_discard_sink order check agrees with cudf::is_sorted on the whole input",
          "[physical_discard_sink]")
{
  // The check across batches compares key rows on the host; the check within a batch is cudf's.
  // Both must order nulls, NaN and descending keys the same way.
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<std::optional<double>> domain{std::nullopt, nan, -1.0, 0.0, 1.0, 2.5};
  auto& space = *get_default_gpu_space();
  std::mt19937 rng(20261113);

  for (auto const order : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    for (auto const null_order : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
      std::size_t sorted_inputs = 0;
      for (int trial = 0; trial < 12; ++trial) {
        std::vector<double> data;
        std::vector<bool> valid;
        for (int i = 0; i < 24; ++i) {
          auto const& value = domain[rng() % domain.size()];
          data.push_back(value.value_or(0.0));
          valid.push_back(value.has_value());
        }
        auto const source =
          make_numeric_batch_with_nulls<double>(space, data, valid, cudf::type_id::FLOAT64);
        // Two trials in three feed sorted rows, the others the rows as drawn.
        std::unique_ptr<cudf::table> rows;
        if (trial % 3 != 0) {
          rows = cudf::sort(sirius::get_cudf_table_view(*source), {order}, {null_order});
        } else {
          rows = std::make_unique<cudf::table>(sirius::get_cudf_table_view(*source));
        }
        auto const expected = cudf::is_sorted(rows->view(), {order}, {null_order});
        sorted_inputs += expected ? 1 : 0;

        // Cut the rows into tasks of 1 to 6 rows, at increasing positions.
        std::vector<cudf::size_type> cuts;
        for (cudf::size_type begin = 0; begin < rows->num_rows();) {
          auto const end =
            std::min<cudf::size_type>(rows->num_rows(), begin + 1 + static_cast<int>(rng() % 6));
          cuts.push_back(begin);
          cuts.push_back(end);
          begin = end;
        }
        auto const slices = cudf::slice(rows->view(), cuts);
        std::vector<std::size_t> arrival(slices.size());
        std::iota(arrival.begin(), arrival.end(), std::size_t{0});
        std::shuffle(arrival.begin(), arrival.end(), rng);

        auto sink = make_sink(duckdb::LogicalType::DOUBLE, check_first_column(order, null_order));
        for (auto const position : arrival) {
          auto batch = sirius::make_data_batch(
            std::make_unique<cudf::table>(slices[position]), space, default_stream(), {});
          discard(sink, {std::move(batch)}, position);
        }
        sink.finish();
        INFO("order " << static_cast<int>(order) << " null_order " << static_cast<int>(null_order)
                      << " trial " << trial);
        CHECK((sink.order_violations() == 0) == expected);
      }
      CHECK(sorted_inputs >= 8);
    }
  }
}

namespace {

/// Runs one task per entry of @p slices on its own thread and stream, in the order of @p arrival.
/// Returns the number of tasks that threw.
std::size_t discard_concurrently(sirius_physical_discard_sink& sink,
                                 const std::vector<std::shared_ptr<data_batch>>& slices,
                                 const std::vector<std::size_t>& positions,
                                 const std::vector<std::size_t>& arrival)
{
  std::atomic<std::size_t> failures{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (auto const index : arrival) {
    threads.emplace_back([&, index] {
      rmm::cuda_stream stream;
      while (!go.load()) {
        std::this_thread::yield();
      }
      try {
        pipelineable_operator_data input(batch_vector{slices[index]}, positions[index]);
        sink.execute(input, stream);
      } catch (...) {
        failures.fetch_add(1);
      }
    });
  }
  go.store(true);
  for (auto& thread : threads) {
    thread.join();
  }
  return failures.load();
}

}  // namespace

TEST_CASE("sirius_physical_discard_sink counts and checks order with tasks running concurrently",
          "[physical_discard_sink]")
{
  constexpr std::size_t num_tasks      = 8;
  constexpr std::int64_t rows_per_task = 512;
  std::vector<std::shared_ptr<data_batch>> slices;
  for (std::size_t task = 0; task < num_tasks; ++task) {
    std::vector<std::int64_t> values(rows_per_task);
    std::iota(values.begin(), values.end(), static_cast<std::int64_t>(task) * rows_per_task);
    slices.push_back(int64_batch(values));
  }
  std::vector<std::size_t> in_order(num_tasks);
  std::iota(in_order.begin(), in_order.end(), std::size_t{0});
  std::mt19937 rng(7);

  SECTION("sorted input: every row counted once, no violation whatever the interleaving")
  {
    for (int round = 0; round < 20; ++round) {
      auto arrival = in_order;
      std::shuffle(arrival.begin(), arrival.end(), rng);
      auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
      REQUIRE(discard_concurrently(sink, slices, in_order, arrival) == 0);
      sink.finish();
      CHECK(sink.rows_discarded() == num_tasks * rows_per_task);
      CHECK(sink.bytes_discarded() == num_tasks * rows_per_task * sizeof(std::int64_t));
      CHECK(sink.order_violations() == 0);
    }
  }

  SECTION("two neighbouring tasks at each other's position: one violation at finish()")
  {
    // Slices 3 and 4 swapped: ... 2 | 4 | 3 | 5 ... has exactly one descent, between 4 and 3.
    auto positions = in_order;
    std::swap(positions[3], positions[4]);
    for (int round = 0; round < 20; ++round) {
      auto arrival = in_order;
      std::shuffle(arrival.begin(), arrival.end(), rng);
      auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
      REQUIRE(discard_concurrently(sink, slices, positions, arrival) == 0);
      CHECK(sink.order_violations() == 0);
      sink.finish();
      CHECK(sink.order_violations() == 1);
    }
  }

  SECTION("finish() from two threads walks the positions and logs once")
  {
    auto positions = in_order;
    std::swap(positions[3], positions[4]);
    for (int round = 0; round < 20; ++round) {
      auto sink = make_sink(duckdb::LogicalType::BIGINT, check_first_column());
      REQUIRE(discard_concurrently(sink, slices, positions, in_order) == 0);

      sirius::test::scoped_recording_log_sink log("info");
      std::thread first([&] { sink.finish(); });
      std::thread second([&] { sink.finish(); });
      first.join();
      second.join();
      CHECK(sink.order_violations() == 1);
      auto const records = log.records();
      auto const summary = std::count_if(records.begin(), records.end(), [](auto const& record) {
        return record.message.rfind("[discard_sink] ", 0) == 0;
      });
      CHECK(summary == 1);
    }
  }
}
