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

#include "catch.hpp"
#include "helper/logical_type.hpp"
#include "op/input_free_operator_data.hpp"
#include "op/scan/generator/generator_spec.hpp"
#include "op/sirius_physical_generator.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "query_id.hpp"
#include "sirius/exception.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using sirius::logical_type;
using sirius::op::input_free_operator_data;
using sirius::op::sirius_physical_generator;
using sirius::op::scan::generator::column_spec;
using sirius::op::scan::generator::distribution;
using sirius::op::scan::generator::estimated_batch_bytes;
using sirius::op::scan::generator::generator_spec;
using sirius::op::scan::generator::value_type;
using sirius::test::operator_utils::copy_column_to_host;

/// INTEGER, BIGINT and DOUBLE columns (VARCHAR is covered by the string kernel's own tests).
generator_spec numeric_spec(std::uint64_t num_batches, std::uint64_t rows_per_batch)
{
  generator_spec spec;
  spec.columns        = {{"i", value_type::int32, distribution::uniform, 0, 1000},
                         {"b", value_type::int64, distribution::uniform, 0, 1e12},
                         {"d", value_type::float64, distribution::normal, 0, 1}};
  spec.num_batches    = num_batches;
  spec.rows_per_batch = rows_per_batch;
  spec.seed           = 42;
  return spec;
}

duckdb::vector<logical_type> numeric_types()
{
  return {logical_type::make(sirius::type_id::INTEGER),
          logical_type::make(sirius::type_id::BIGINT),
          logical_type::make(sirius::type_id::DOUBLE)};
}

std::unique_ptr<sirius_physical_generator> make_generator(std::uint64_t num_batches,
                                                          std::uint64_t rows_per_batch)
{
  return std::make_unique<sirius_physical_generator>(
    numeric_types(), numeric_spec(num_batches, rows_per_batch), num_batches * rows_per_batch);
}

std::uint64_t sequence_of(const sirius::op::operator_data& input)
{
  auto const* generator_input = dynamic_cast<const input_free_operator_data*>(&input);
  REQUIRE(generator_input);
  REQUIRE(generator_input->sequence().has_value());
  return *generator_input->sequence();
}

struct host_batch {
  std::vector<std::int32_t> i;
  std::vector<std::int64_t> b;
  std::vector<double> d;
  friend bool operator==(const host_batch&, const host_batch&) = default;
};

/// Runs one task of @p source on @p input and copies the generated batch to the host.
host_batch execute_to_host(sirius_physical_generator& source,
                           const sirius::op::operator_data& input,
                           std::uint64_t rows_per_batch,
                           ::cuda::stream_ref stream)
{
  auto output   = source.execute(input, stream);
  auto& batches = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto ro    = batches[0]->to_read_only();
  auto table = sirius::get_cudf_table_view(ro);
  REQUIRE(table.num_rows() == static_cast<cudf::size_type>(rows_per_batch));
  REQUIRE(table.num_columns() == 3);
  REQUIRE(table.column(0).type().id() == cudf::type_id::INT32);
  REQUIRE(table.column(1).type().id() == cudf::type_id::INT64);
  REQUIRE(table.column(2).type().id() == cudf::type_id::FLOAT64);
  return {copy_column_to_host<std::int32_t>(table.column(0), stream),
          copy_column_to_host<std::int64_t>(table.column(1), stream),
          copy_column_to_host<double>(table.column(2), stream)};
}

}  // namespace

TEST_CASE("generator hands out one input per batch, then none", "[operator][physical_generator]")
{
  auto source = make_generator(5, 100);
  CHECK(source->is_source());
  CHECK(source->output_order_preservation() == sirius::OrderPreservationType::NO_ORDER);

  for (std::uint64_t batch = 0; batch < 5; ++batch) {
    CHECK_FALSE(source->all_ports_empty());
    auto hint = source->get_next_task_hint();
    REQUIRE(hint.has_value());
    CHECK(hint->hint == sirius::op::TaskCreationHint::READY);
    CHECK(hint->producer == source.get());
    auto input = source->get_next_task_input_data();
    REQUIRE(input);
    CHECK(sequence_of(*input) == batch);
  }
  CHECK(source->all_ports_empty());
  CHECK_FALSE(source->get_next_task_hint().has_value());
  CHECK_FALSE(source->get_next_task_input_data());
  CHECK_FALSE(source->get_next_task_input_data());
  CHECK(source->all_ports_empty());
}

TEST_CASE("generator with no batches is exhausted from the start", "[operator][physical_generator]")
{
  auto source = make_generator(0, 100);
  CHECK(source->all_ports_empty());
  CHECK_FALSE(source->get_next_task_hint().has_value());
  CHECK_FALSE(source->get_next_task_input_data());
}

TEST_CASE("generator gives each batch index once under concurrent claims",
          "[operator][physical_generator]")
{
  constexpr std::uint64_t num_batches = 1000;
  auto source                         = make_generator(num_batches, 100);

  std::mutex mutex;
  std::vector<std::uint64_t> claimed;
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&] {
      std::vector<std::uint64_t> mine;
      while (auto input = source->get_next_task_input_data()) {
        auto const* generator_input = dynamic_cast<const input_free_operator_data*>(input.get());
        if (generator_input && generator_input->sequence()) {
          mine.push_back(*generator_input->sequence());
        }
      }
      std::lock_guard lock(mutex);
      claimed.insert(claimed.end(), mine.begin(), mine.end());
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(claimed.size() == num_batches);
  std::sort(claimed.begin(), claimed.end());
  for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
    REQUIRE(claimed[batch] == batch);
  }
  CHECK(source->all_ports_empty());
  CHECK_FALSE(source->get_next_task_input_data());
}

TEST_CASE("generator execute produces the batch named by its input",
          "[operator][physical_generator]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(gpu);
  rmm::cuda_stream stream;

  constexpr std::uint64_t rows = 1000;
  auto source                  = make_generator(3, rows);
  auto first                   = source->get_next_task_input_data();
  auto second                  = source->get_next_task_input_data();
  REQUIRE(first);
  REQUIRE(second);
  first->prepare_for_processing(gpu, stream);
  second->prepare_for_processing(gpu, stream);

  auto const batch0 = execute_to_host(*source, *first, rows, stream);
  auto const batch1 = execute_to_host(*source, *second, rows, stream);

  // An OOM retry runs the same input again and must regenerate the same batch.
  CHECK(execute_to_host(*source, *first, rows, stream) == batch0);
  CHECK(batch0.i != batch1.i);
  CHECK(batch0.b != batch1.b);
  CHECK(batch0.d != batch1.d);

  // The batch is a function of (spec, batch index) only: a second operator with the same spec,
  // claiming in the same order, reproduces it.
  auto other       = make_generator(3, rows);
  auto other_first = other->get_next_task_input_data();
  REQUIRE(other_first);
  other_first->prepare_for_processing(gpu, stream);
  CHECK(execute_to_host(*other, *other_first, rows, stream) == batch0);
}

TEST_CASE("generator execute rejects inputs it did not hand out", "[operator][physical_generator]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(gpu);
  rmm::cuda_stream stream;
  auto source = make_generator(2, 10);

  sirius::op::pipelineable_operator_data wrong_type(
    std::vector<std::shared_ptr<cucascade::data_batch>>{});
  CHECK_THROWS_AS(source->execute(wrong_type, stream), sirius::internal_exception);

  input_free_operator_data no_sequence(64);
  no_sequence.prepare_for_processing(gpu, stream);
  CHECK_THROWS_AS(source->execute(no_sequence, stream), sirius::internal_exception);
}

TEST_CASE("generator rejects output types that disagree with its spec",
          "[operator][physical_generator]")
{
  auto swapped = numeric_types();
  std::swap(swapped[0], swapped[1]);
  CHECK_THROWS_AS(sirius_physical_generator(swapped, numeric_spec(1, 10), 10),
                  sirius::internal_exception);

  auto too_few = numeric_types();
  too_few.pop_back();
  CHECK_THROWS_AS(sirius_physical_generator(too_few, numeric_spec(1, 10), 10),
                  sirius::internal_exception);
}

TEST_CASE("generator memory estimates follow the spec", "[operator][physical_generator]")
{
  constexpr std::uint64_t num_batches = 7;
  constexpr std::uint64_t rows        = 1000;
  auto const spec                     = numeric_spec(num_batches, rows);
  auto const batch_bytes              = estimated_batch_bytes(spec);
  REQUIRE(batch_bytes == rows * (4 + 8 + 8));

  auto source = make_generator(num_batches, rows);
  REQUIRE(source->total_source_input_bytes().has_value());
  CHECK(*source->total_source_input_bytes() == num_batches * batch_bytes);

  auto input = source->get_next_task_input_data();
  REQUIRE(input);
  CHECK(input->get_estimated_size_in_bytes() == batch_bytes);
  CHECK(input->get_type() == sirius::op::operator_data_type::INPUT_FREE);

  sirius::op::input_stats stats;
  stats.num_batches = 1;
  stats.bytes       = batch_bytes;
  stats.type        = sirius::op::operator_data_type::INPUT_FREE;
  CHECK(source->no_history_peak_memory_estimate(stats) == 2 * batch_bytes);

  // A spec too large for std::size_t saturates and does not wrap.
  auto huge           = numeric_spec(std::numeric_limits<std::uint64_t>::max(), rows);
  huge.rows_per_batch = std::numeric_limits<std::uint64_t>::max() / 2;
  sirius_physical_generator huge_source(numeric_types(), huge, 0);
  CHECK(*huge_source.total_source_input_bytes() == std::numeric_limits<std::size_t>::max());
  CHECK(huge_source.no_history_peak_memory_estimate(stats) ==
        std::numeric_limits<std::size_t>::max());
}

TEST_CASE("generator produces and reserves only its output columns",
          "[operator][physical_generator]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(gpu);
  rmm::cuda_stream stream;

  constexpr std::uint64_t num_batches = 3;
  constexpr std::uint64_t rows        = 1000;
  auto const spec                     = numeric_spec(num_batches, rows);

  // DOUBLE then INTEGER: columns 2 and 0 of the spec, out of order; BIGINT is never drawn.
  sirius_physical_generator subset(
    {logical_type::make(sirius::type_id::DOUBLE), logical_type::make(sirius::type_id::INTEGER)},
    spec,
    {2, 0},
    num_batches * rows);
  CHECK(subset.spec() == spec);
  CHECK(subset.output_columns() == std::vector<std::size_t>{2, 0});

  auto const subset_bytes = rows * (8 + 4);
  REQUIRE(estimated_batch_bytes(spec, subset.output_columns()) == subset_bytes);
  CHECK(*subset.total_source_input_bytes() == num_batches * subset_bytes);
  sirius::op::input_stats stats;
  CHECK(subset.no_history_peak_memory_estimate(stats) == 2 * subset_bytes);

  auto full = make_generator(num_batches, rows);
  for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
    auto subset_input = subset.get_next_task_input_data();
    auto full_input   = full->get_next_task_input_data();
    REQUIRE(subset_input);
    REQUIRE(full_input);
    CHECK(subset_input->get_estimated_size_in_bytes() == subset_bytes);
    subset_input->prepare_for_processing(gpu, stream);
    full_input->prepare_for_processing(gpu, stream);

    auto output = subset.execute(*subset_input, stream);
    auto& batches =
      dynamic_cast<sirius::op::pipelineable_operator_data&>(*output).get_data_batches();
    REQUIRE(batches.size() == 1);
    auto ro    = batches[0]->to_read_only();
    auto table = sirius::get_cudf_table_view(ro);
    REQUIRE(table.num_columns() == 2);
    REQUIRE(table.num_rows() == static_cast<cudf::size_type>(rows));
    REQUIRE(table.column(0).type().id() == cudf::type_id::FLOAT64);
    REQUIRE(table.column(1).type().id() == cudf::type_id::INT32);

    // Each column has the values it has in the full batch.
    auto const expected = execute_to_host(*full, *full_input, rows, stream);
    CHECK(copy_column_to_host<double>(table.column(0), stream) == expected.d);
    CHECK(copy_column_to_host<std::int32_t>(table.column(1), stream) == expected.i);
  }
  CHECK_FALSE(subset.get_next_task_input_data());
}

TEST_CASE("generator rejects output columns that disagree with its spec",
          "[operator][physical_generator]")
{
  auto const spec    = numeric_spec(1, 10);
  auto const integer = logical_type::make(sirius::type_id::INTEGER);
  auto const bigint  = logical_type::make(sirius::type_id::BIGINT);
  // No columns, a column outside the spec, a type for another column, a missing type.
  CHECK_THROWS_AS(sirius_physical_generator({}, spec, std::vector<std::size_t>{}, 10),
                  sirius::internal_exception);
  CHECK_THROWS_AS(sirius_physical_generator({integer}, spec, {3}, 10), sirius::internal_exception);
  CHECK_THROWS_AS(sirius_physical_generator({integer}, spec, {1}, 10), sirius::internal_exception);
  CHECK_THROWS_AS(sirius_physical_generator({bigint}, spec, {1, 0}, 10),
                  sirius::internal_exception);
  CHECK_NOTHROW(sirius_physical_generator({bigint, integer}, spec, {1, 0}, 10));
}

TEST_CASE("query::build_indices lists a GENERATOR source among the scan operators",
          "[operator][physical_generator]")
{
  auto telemetry = sirius::test::make_test_telemetry_context();
  sirius::pipeline::pipeline_build_context ctx{nullptr, true};
  sirius::pipeline::sirius_pipeline_build_state build;
  auto source         = make_generator(2, 10);
  source->operator_id = 1;

  auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(ctx);
  build.set_pipeline_source(*pipeline, *source);
  auto const id = sirius::make_query_id(1);
  sirius::planner::query query(
    std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{pipeline},
    telemetry->context(),
    id,
    sirius::telemetry::query_telemetry_info{quent::now_v7(), telemetry->worker_id().raw(), id});

  auto const scans = query.get_scan_operators();
  REQUIRE(scans.size() == 1);
  CHECK(scans[0] == source.get());
  CHECK(query.get_pipeline(source.get()) == pipeline);
}

TEST_CASE("generator batches do not depend on the order tasks run in",
          "[operator][physical_generator]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(gpu);
  rmm::cuda_stream stream;

  constexpr std::uint64_t num_batches = 4;
  constexpr std::uint64_t rows        = 500;
  auto in_order                       = make_generator(num_batches, rows);
  auto reversed                       = make_generator(num_batches, rows);

  std::vector<std::unique_ptr<sirius::op::operator_data>> forward_inputs;
  std::vector<std::unique_ptr<sirius::op::operator_data>> reverse_inputs;
  for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
    forward_inputs.push_back(in_order->get_next_task_input_data());
    reverse_inputs.push_back(reversed->get_next_task_input_data());
    REQUIRE(forward_inputs.back());
    REQUIRE(reverse_inputs.back());
    forward_inputs.back()->prepare_for_processing(gpu, stream);
    reverse_inputs.back()->prepare_for_processing(gpu, stream);
  }

  std::vector<host_batch> forward;
  for (auto& input : forward_inputs) {
    forward.push_back(execute_to_host(*in_order, *input, rows, stream));
  }
  std::vector<host_batch> backward(num_batches);
  for (std::size_t k = num_batches; k-- > 0;) {
    backward[k] = execute_to_host(*reversed, *reverse_inputs[k], rows, stream);
  }

  for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
    CHECK(forward[batch] == backward[batch]);
    // The spec's uniform INTEGER column draws from the inclusive range [0, 1000].
    for (auto value : forward[batch].i) {
      REQUIRE(value >= 0);
      REQUIRE(value <= 1000);
    }
    for (auto value : forward[batch].b) {
      REQUIRE(value >= 0);
      REQUIRE(value <= static_cast<std::int64_t>(1e12));
    }
  }
}

TEST_CASE("generator peak estimate for a narrow INTEGER-only spec is twice the batch",
          "[operator][physical_generator]")
{
  constexpr std::uint64_t rows = 4096;
  generator_spec spec;
  spec.columns           = {{"i", value_type::int32, distribution::uniform, 0, 10}};
  spec.num_batches       = 3;
  spec.rows_per_batch    = rows;
  spec.seed              = 7;
  auto const batch_bytes = estimated_batch_bytes(spec);
  REQUIRE(batch_bytes == rows * 4);

  sirius_physical_generator source(
    {logical_type::make(sirius::type_id::INTEGER)}, spec, spec.num_batches * rows);
  CHECK(*source.total_source_input_bytes() == spec.num_batches * batch_bytes);

  sirius::op::input_stats stats;
  stats.num_batches = 1;
  stats.bytes       = batch_bytes;
  stats.type        = sirius::op::operator_data_type::INPUT_FREE;
  // An INTEGER column is drawn at its own width: the finished column plus one draw buffer.
  CHECK(source.no_history_peak_memory_estimate(stats) == 2 * batch_bytes);
}

namespace {

struct mixed_host_batch {
  std::vector<std::int32_t> i;
  std::vector<std::int64_t> b;
  std::vector<bool> b_valid;
  std::vector<double> d;
  std::vector<std::string> s;
  friend bool operator==(const mixed_host_batch&, const mixed_host_batch&) = default;
};

/// Copies @p count elements at @p device to the host on @p stream and waits for that stream
/// alone. The blocking cudaMemcpy would wait for every stream of the device.
template <typename T>
std::vector<T> copy_on_stream(const T* device, std::size_t count, ::cuda::stream_ref stream)
{
  std::vector<T> host(count);
  if (count != 0 &&
      cudaMemcpyAsync(
        host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost, stream.get()) !=
        cudaSuccess) {
    throw std::runtime_error("cudaMemcpyAsync failed");
  }
  stream.sync();
  return host;
}

/// Runs one task of @p source on @p stream and copies the batch to the host, waiting only on the
/// batch's writer event and @p stream.
mixed_host_batch execute_mixed_to_host(sirius_physical_generator& source,
                                       const sirius::op::operator_data& input,
                                       ::cuda::stream_ref stream)
{
  auto output   = source.execute(input, stream);
  auto& batches = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output).get_data_batches();
  if (batches.size() != 1) { throw std::runtime_error("expected one generated batch"); }
  auto ro                 = batches[0]->to_read_only();
  auto const writer_event = ro.get_writer_event();
  if (writer_event == nullptr) { throw std::runtime_error("generated batch has no writer event"); }
  if (cudaStreamWaitEvent(stream.get(), writer_event, 0) != cudaSuccess) {
    throw std::runtime_error("cudaStreamWaitEvent failed");
  }
  auto table = sirius::get_cudf_table_view(ro);
  if (table.num_columns() != 4) { throw std::runtime_error("expected four generated columns"); }
  auto const rows = static_cast<std::size_t>(table.num_rows());
  mixed_host_batch host;
  host.i = copy_column_to_host<std::int32_t>(table.column(0), stream);
  host.b = copy_column_to_host<std::int64_t>(table.column(1), stream);
  host.d = copy_column_to_host<double>(table.column(2), stream);

  host.b_valid.assign(rows, true);
  if (auto const* mask = table.column(1).null_mask()) {
    auto const words = copy_on_stream(mask, (rows + 31) / 32, stream);
    for (std::size_t row = 0; row < rows; ++row) {
      host.b_valid[row] = (words[row / 32] >> (row % 32)) & 1u;
    }
  }

  cudf::strings_column_view strings(table.column(3));
  auto const offsets = copy_on_stream(strings.offsets().data<cudf::size_type>(), rows + 1, stream);
  auto const chars   = copy_on_stream(
    strings.chars_begin(rmm::cuda_stream_view{stream.get()}), offsets.back(), stream);
  host.s.reserve(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    host.s.emplace_back(chars.data() + offsets[row], chars.data() + offsets[row + 1]);
  }
  return host;
}

}  // namespace

TEST_CASE("generator execute is stream-safe under concurrent tasks",
          "[operator][physical_generator]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(gpu);

  constexpr std::uint64_t num_batches = 16;
  constexpr std::uint64_t rows        = 20000;
  constexpr std::size_t num_threads   = 4;
  generator_spec spec;
  spec.columns        = {{"i", value_type::int32, distribution::uniform, 0, 1000},
                         {"b", value_type::int64, distribution::normal, 0, 1e6, 0, 0.2},
                         {"d", value_type::float64, distribution::uniform, 0, 1},
                         {"s", value_type::varchar, distribution::uniform, 8, 16}};
  spec.num_batches    = num_batches;
  spec.rows_per_batch = rows;
  spec.seed           = 11;
  auto const types    = [] {
    return duckdb::vector<logical_type>{logical_type::make(sirius::type_id::INTEGER),
                                        logical_type::make(sirius::type_id::BIGINT),
                                        logical_type::make(sirius::type_id::DOUBLE),
                                        logical_type::make(sirius::type_id::VARCHAR)};
  };

  // Reference: every batch, one after another on one stream.
  std::vector<mixed_host_batch> serial;
  {
    rmm::cuda_stream stream;
    sirius_physical_generator source(types(), spec, num_batches * rows);
    for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
      auto input = source.get_next_task_input_data();
      REQUIRE(input);
      input->prepare_for_processing(gpu, stream);
      serial.push_back(execute_mixed_to_host(source, *input, stream));
    }
  }
  REQUIRE(serial[0].i.size() == rows);
  REQUIRE(serial[0].s.size() == rows);
  REQUIRE(std::count(serial[0].b_valid.begin(), serial[0].b_valid.end(), false) > 0);
  REQUIRE(serial[0] != serial[1]);

  sirius_physical_generator source(types(), spec, num_batches * rows);
  std::vector<std::unique_ptr<sirius::op::operator_data>> inputs;
  for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
    inputs.push_back(source.get_next_task_input_data());
    REQUIRE(inputs.back());
    REQUIRE(sequence_of(*inputs.back()) == batch);
  }

  // Thread t runs batches t, t + 4, ... on its own stream; nothing synchronizes the device
  // between the threads' launches.
  std::vector<mixed_host_batch> concurrent(num_batches);
  std::vector<std::string> errors(num_threads);
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      try {
        rmm::cuda_stream stream;
        for (std::size_t batch = t; batch < num_batches; batch += num_threads) {
          inputs[batch]->prepare_for_processing(gpu, stream);
          concurrent[batch] = execute_mixed_to_host(source, *inputs[batch], stream);
        }
      } catch (const std::exception& e) {
        errors[t] = e.what();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (std::size_t t = 0; t < num_threads; ++t) {
    CAPTURE(t);
    REQUIRE(errors[t].empty());
  }
  for (std::uint64_t batch = 0; batch < num_batches; ++batch) {
    CAPTURE(batch);
    CHECK(concurrent[batch] == serial[batch]);
  }
}
