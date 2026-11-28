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

#include "data/convertible_data_batch.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <rmm/error.hpp>

#include <catch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_limit.hpp>

#include <future>
#include <map>
#include <numeric>
#include <type_traits>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

// A DISK space without an io context: its batches carry row metadata only, and any disk I/O
// would first have to install a context (see memory_space::get_or_init_io_context).
struct limit_metadata_inputs {
  cucascade::memory::memory_space space{
    cucascade::memory::disk_memory_space_config{0, 1ull << 20, "/tmp"}};
  data_repository repo;
  uint64_t next_id = 0;

  std::weak_ptr<data_batch> add(std::size_t rows,
                                std::optional<std::size_t> position = std::nullopt)
  {
    auto table = std::make_unique<cucascade::memory::disk_table_allocation>(
      "", std::vector<cucascade::memory::column_metadata>{}, 0);
    table->columns.emplace_back();
    table->columns[0].num_rows = rows;
    auto batch                 = data_batch::make(
      next_id++, std::make_unique<cucascade::disk_data_representation>(std::move(table), space));
    if (position) {
      repo.add_data_batch_at_position(batch, *position);
    } else {
      repo.add_data_batch(batch);
    }
    return batch;
  }

  void attach_to(sirius_physical_streaming_limit& limit)
  {
    limit.add_port("default",
                   std::make_unique<sirius_physical_operator::port>(
                     MemoryBarrierType::FULL, &repo, nullptr, nullptr));
  }
};

// Exercise the real task-creation contract, including claims made before execution.
std::unique_ptr<operator_data> run_limit(sirius_physical_streaming_limit& limiter,
                                         std::vector<std::shared_ptr<data_batch>> const& batches)
{
  data_repository repo;
  for (auto const& batch : batches) {
    repo.add_data_batch(batch);
  }
  auto* port = limiter.try_get_port("default");
  if (!port) {
    limiter.add_port("default",
                     std::make_unique<sirius_physical_operator::port>(
                       MemoryBarrierType::FULL, &repo, nullptr, nullptr));
    port = limiter.get_port("default");
  }
  port->repo = &repo;
  std::vector<std::shared_ptr<data_batch>> output;
  while (auto task = limiter.get_next_task_input_data()) {
    auto result          = limiter.execute(*task, cudf::get_default_stream());
    auto const& produced = dynamic_cast<pipelineable_operator_data&>(*result).get_data_batches();
    output.insert(output.end(), produced.begin(), produced.end());
  }
  port->repo = nullptr;
  return std::make_unique<pipelineable_operator_data>(std::move(output));
}

}  // namespace

TEMPLATE_TEST_CASE("sirius_physical_streaming_limit limits rows in data_batch",
                   "[physical_limit]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   bool,
                   decimal64_tag,
                   string_tag,
                   timestamp_us_tag,
                   date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  std::vector<typename Traits::type> values(10);
  if constexpr (Traits::is_string) {
    values = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};
  } else if constexpr (Traits::is_decimal) {
    for (int i = 0; i < 10; ++i) {
      values[i] = static_cast<typename Traits::type>(i * 100);
    }
  } else if constexpr (Traits::is_ts) {
    for (int i = 0; i < 10; ++i) {
      values[i] = static_cast<typename Traits::type>(i * 1'000'000);
    }
  } else if constexpr (std::is_same_v<typename Traits::type, int32_t> ||
                       std::is_same_v<typename Traits::type, int64_t> ||
                       std::is_same_v<typename Traits::type, int16_t>) {
    std::iota(values.begin(), values.end(), static_cast<typename Traits::type>(0));
  } else if constexpr (std::is_same_v<typename Traits::type, float> ||
                       std::is_same_v<typename Traits::type, double>) {
    for (int i = 0; i < 10; ++i) {
      values[i] = static_cast<typename Traits::type>(i);
    }
  } else if constexpr (std::is_same_v<typename Traits::type, bool>) {
    for (int i = 0; i < 10; ++i) {
      values[i] = (i % 2 == 0);
    }
  }

  std::shared_ptr<data_batch> input_batch;
  if constexpr (Traits::is_string) {
    input_batch = make_string_batch(*space, values);
  } else if constexpr (Traits::is_decimal) {
    input_batch = make_decimal64_batch(*space, values, Traits::scale);
  } else if constexpr (Traits::is_ts) {
    input_batch = make_timestamp_batch(*space, values, Traits::cudf_type);
  } else {
    input_batch = make_numeric_batch<typename Traits::type>(*space, values, Traits::cudf_type);
  }

  auto limit_node  = duckdb::BoundLimitNode::ConstantValue(3);
  auto offset_node = duckdb::BoundLimitNode::ConstantValue(2);

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(Traits::logical_type());

  sirius_physical_streaming_limit limiter(sirius::from_duckdb_vec(std::move(types)),
                                          std::move(limit_node),
                                          std::move(offset_node),
                                          values.size(),
                                          false);

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{input_batch};
  auto outputs = run_limit(limiter, inputs);
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);
  auto output_table = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  auto host_vals = copy_column_to_host<typename Traits::type>(output_table.column(0));

  std::vector<typename Traits::type> expected = {values[2], values[3], values[4]};
  REQUIRE(host_vals == expected);
}

// Helper to build a single-column int64 batch with sequential values [start, start+count)
static std::shared_ptr<data_batch> make_range_batch(memory_space& space,
                                                    int64_t start,
                                                    int64_t count)
{
  std::vector<int64_t> values(count);
  std::iota(values.begin(), values.end(), start);
  return sirius::test::operator_utils::make_numeric_batch<int64_t>(
    space, values, cudf::type_id::INT64);
}

// Collect all int64 values from multiple output batches into a single host vector
static std::vector<int64_t> collect_all_rows(
  const std::vector<std::shared_ptr<data_batch>>& batches)
{
  std::vector<int64_t> all_rows;
  for (auto const& b : batches) {
    auto table = sirius::get_cudf_table_view(*b);
    auto col   = sirius::test::operator_utils::copy_column_to_host<int64_t>(table.column(0));
    all_rows.insert(all_rows.end(), col.begin(), col.end());
  }
  return all_rows;
}

TEST_CASE("streaming_limit caps total rows across multiple batches",
          "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // 3 batches of 10 rows each: [0..9], [10..19], [20..29]
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 10));
  batches.push_back(make_range_batch(*space, 10, 10));
  batches.push_back(make_range_batch(*space, 20, 10));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  sirius_physical_streaming_limit limiter(sirius::from_duckdb_vec(std::move(types)),
                                          duckdb::BoundLimitNode::ConstantValue(5),
                                          duckdb::BoundLimitNode(),
                                          30,
                                          true);

  auto outputs = run_limit(limiter, batches);
  auto rows =
    collect_all_rows(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches());

  // Should return exactly 5 rows: [0, 1, 2, 3, 4]
  std::vector<int64_t> expected{0, 1, 2, 3, 4};
  REQUIRE(rows == expected);
}

TEST_CASE("streaming_limit spans across two batches returning correct data",
          "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // 2 batches of 150 rows each: [0..149], [150..299]
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 150));
  batches.push_back(make_range_batch(*space, 150, 150));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  sirius_physical_streaming_limit limiter(sirius::from_duckdb_vec(std::move(types)),
                                          duckdb::BoundLimitNode::ConstantValue(200),
                                          duckdb::BoundLimitNode(),
                                          300,
                                          true);

  auto outputs = run_limit(limiter, batches);
  auto rows =
    collect_all_rows(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches());

  // Should return exactly 200 rows: all 150 from batch 1, plus first 50 from batch 2
  REQUIRE(rows.size() == 200);

  std::vector<int64_t> expected(200);
  std::iota(expected.begin(), expected.end(), 0);  // [0, 1, ..., 199]
  REQUIRE(rows == expected);
}

TEST_CASE("streaming_limit offset spans across multiple batches", "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // 3 batches of 10 rows each: [0..9], [10..19], [20..29]
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 10));
  batches.push_back(make_range_batch(*space, 10, 10));
  batches.push_back(make_range_batch(*space, 20, 10));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  // offset=15 skips all of batch 1 (10 rows) + 5 rows of batch 2, limit=5
  sirius_physical_streaming_limit limiter(sirius::from_duckdb_vec(std::move(types)),
                                          duckdb::BoundLimitNode::ConstantValue(5),
                                          duckdb::BoundLimitNode::ConstantValue(15),
                                          30,
                                          true);

  auto outputs = run_limit(limiter, batches);
  auto rows =
    collect_all_rows(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches());

  // Should return [15, 16, 17, 18, 19]
  std::vector<int64_t> expected{15, 16, 17, 18, 19};
  REQUIRE(rows == expected);
}

TEST_CASE("streaming_limit with separate execute calls enforces global limit",
          "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  // limit=5 shared across multiple execute() calls (simulating concurrent tasks)
  sirius_physical_streaming_limit limiter(sirius::from_duckdb_vec(std::move(types)),
                                          duckdb::BoundLimitNode::ConstantValue(5),
                                          duckdb::BoundLimitNode(),
                                          30,
                                          true);

  // First call with batch [0..9]
  std::vector<std::shared_ptr<data_batch>> batch1{make_range_batch(*space, 0, 10)};
  auto out1 = run_limit(limiter, batch1);
  auto rows1 =
    collect_all_rows(dynamic_cast<const pipelineable_operator_data&>(*out1).get_data_batches());

  // Second call with batch [10..19] — limit should already be exhausted
  std::vector<std::shared_ptr<data_batch>> batch2{make_range_batch(*space, 10, 10)};
  auto out2 = run_limit(limiter, batch2);
  auto rows2 =
    collect_all_rows(dynamic_cast<const pipelineable_operator_data&>(*out2).get_data_batches());

  // Total across both calls should be exactly 5
  REQUIRE(rows1.size() + rows2.size() == 5);

  // First call should have taken all 5
  std::vector<int64_t> expected{0, 1, 2, 3, 4};
  REQUIRE(rows1 == expected);
  REQUIRE(rows2.empty());
}

TEST_CASE("ordered limit claims sparse positions before reverse execution",
          "[physical_limit][ordering]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);
  data_repository repo;
  repo.add_data_batch_at_position(make_range_batch(*space, 30, 10), 9);
  repo.add_data_batch_at_position(make_range_batch(*space, 10, 10), 2);
  repo.add_data_batch_at_position(make_range_batch(*space, 20, 10), 5);
  sirius_physical_streaming_limit limiter(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT}),
    duckdb::BoundLimitNode::ConstantValue(12),
    duckdb::BoundLimitNode::ConstantValue(8),
    30,
    true);
  limiter.add_port("default",
                   std::make_unique<sirius_physical_operator::port>(
                     MemoryBarrierType::FULL, &repo, nullptr, nullptr));
  std::vector<std::unique_ptr<operator_data>> tasks;
  while (auto task = limiter.get_next_task_input_data()) {
    tasks.push_back(std::move(task));
  }
  REQUIRE(tasks.size() == 2);
  REQUIRE(repo.all_empty());
  std::map<size_t, std::vector<int64_t>> results;
  for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) {
    auto result      = limiter.execute(**it, cudf::get_default_stream());
    auto const& data = dynamic_cast<pipelineable_operator_data&>(*result);
    REQUIRE(data.get_position().has_value());
    results[*data.get_position()] = collect_all_rows(data.get_data_batches());
  }
  REQUIRE(results[2] == std::vector<int64_t>{18, 19});
  REQUIRE(results[5] == std::vector<int64_t>{20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
  REQUIRE(results.count(9) == 0);
  // Reconstructing input (late materialization) and retrying must retain the same claim.
  auto& first = dynamic_cast<limiting_operator_data&>(*tasks.front());
  auto retry  = first.with_batches_preserving_rows(first.get_data_batches());
  REQUIRE(dynamic_cast<limiting_operator_data&>(*retry).slice_begin == 8);
  auto result = limiter.execute(*retry, cudf::get_default_stream());
  REQUIRE(collect_all_rows(dynamic_cast<pipelineable_operator_data&>(*result).get_data_batches()) ==
          results[2]);
}

TEST_CASE("operator input position describes a multi-batch destination",
          "[ordering][operator_data]")
{
  std::vector<std::shared_ptr<cucascade::data_batch>> batches(2);
  pipelineable_operator_data input(batches, 7);
  REQUIRE(input.get_position() == 7);
  auto replacement = input.with_batches_preserving_rows(batches);
  REQUIRE(replacement->get_position() == 7);
  partitioned_operator_data partitioned(batches, std::optional<std::size_t>{11});
  REQUIRE(partitioned.with_batches_preserving_rows(batches)->get_position() == 11);
  REQUIRE_FALSE(pipelineable_operator_data(batches).get_position());
}

TEST_CASE("unordered limit execution claims offset and limit and drops positions",
          "[physical_limit][unordered_limit]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  sirius_physical_unordered_streaming_limit limit(
    {sirius::from_duckdb(duckdb::LogicalType::BIGINT)},
    BoundLimitNode::ConstantValue(7),
    BoundLimitNode::ConstantValue(12),
    100,
    true);
  std::vector<int64_t> rows;
  for (int begin : {0, 10, 20}) {
    pipelineable_operator_data input(std::vector{make_range_batch(*space, begin, 10)}, 99);
    auto result  = limit.execute(input, cudf::get_default_stream());
    auto& output = dynamic_cast<pipelineable_operator_data&>(*result);
    CHECK_FALSE(output.get_position());
    auto batch_rows = collect_all_rows(output.get_data_batches());
    rows.insert(rows.end(), batch_rows.begin(), batch_rows.end());
  }
  CHECK(rows == std::vector<int64_t>{12, 13, 14, 15, 16, 17, 18});
  CHECK(limit.is_limit_exhausted());
}

TEST_CASE("forwarded payload retains read protection across replacement and cleanup",
          "[operator_data][ordered_limit_cpu]")
{
  limit_metadata_inputs inputs;
  auto explicit_cleanup = GENERATE(false, true);
  auto batch            = inputs.add(10).lock();
  REQUIRE(batch);
  inputs.repo.pop_next_data_batch();
  auto input = std::make_unique<pipelineable_operator_data>(std::vector{batch}, 7);
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto output = input->forward_with_read_only_locks(7);
    CHECK(output->get_data_batches().front() == batch);
    CHECK(output->get_position() == 7);
    input.reset();
    CHECK_FALSE(batch->try_to_mutable());
    // Forwarding a protected output again must also retain protection across replacement.
    auto next = output->forward_with_read_only_locks(std::nullopt);
    output.reset();
    CHECK_FALSE(next->get_position());
    CHECK_FALSE(batch->try_to_mutable());
    if (explicit_cleanup) {
      next->remove_read_only_lock();
      CHECK(batch->try_to_mutable().has_value());
    }
    next.reset();
    CHECK(batch->try_to_mutable().has_value());
    // A retry can prepare and forward the original batch again without leaked locks.
    input = std::make_unique<pipelineable_operator_data>(std::vector{batch}, 7);
  }
}

TEST_CASE("whole-batch LIMIT protects GPU input against a previously selected spill",
          "[physical_limit][limit_input_protection]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(Tier::GPU, 0);
  auto* host   = manager->get_memory_space(Tier::HOST, 0);
  REQUIRE(gpu);
  REQUIRE(host);
  rmm::cuda_stream stream;
  auto batch = make_range_batch(*gpu, 0, 10);
  // Save the cleanup request before the task subscribes/prepares its input.
  sirius::convertible_data_batch saved_spill(batch);
  std::unique_ptr<operator_data> input;
  std::unique_ptr<sirius_physical_operator> limit;
  SECTION("ordered LIMIT retains position")
  {
    limit = std::make_unique<sirius_physical_streaming_limit>(
      duckdb::vector<sirius::logical_type>{sirius::from_duckdb(LogicalType::BIGINT)},
      BoundLimitNode::ConstantValue(10),
      BoundLimitNode(),
      10,
      true);
    data_repository repo;
    repo.add_data_batch_at_position(batch, 7);
    limit->add_port("default",
                    std::make_unique<sirius_physical_operator::port>(
                      MemoryBarrierType::FULL, &repo, nullptr, nullptr));
    input                            = limit->get_next_task_input_data();
    limit->get_port("default")->repo = nullptr;
  }
  SECTION("unordered LIMIT discards position")
  {
    limit = std::make_unique<sirius_physical_unordered_streaming_limit>(
      duckdb::vector<sirius::logical_type>{sirius::from_duckdb(LogicalType::BIGINT)},
      BoundLimitNode::ConstantValue(10),
      BoundLimitNode(),
      10,
      true);
    input = std::make_unique<pipelineable_operator_data>(std::vector{batch}, 7);
  }
  REQUIRE(input);
  auto position = limit->type == SiriusPhysicalOperatorType::STREAMING_LIMIT
                    ? std::optional<std::size_t>{7}
                    : std::nullopt;
  input->prepare_for_processing(gpu, stream);
  input        = limit->execute(*input, stream);
  auto& output = dynamic_cast<pipelineable_operator_data&>(*input);
  CHECK(output.get_data_batches().front() == batch);
  CHECK(output.get_position() == position);
  // Resume the saved request in the window before the next inline GPU read.
  CHECK_FALSE(saved_spill.convert({host}, stream, *manager, false));
  CHECK(collect_all_rows(output.get_data_batches()) ==
        std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  stream.synchronize();
  input.reset();
  REQUIRE(saved_spill.convert({host}, stream, *manager, false).has_value());
  CHECK(batch->to_read_only().get_memory_space() == host);
}

TEST_CASE("unordered limit zero produces an empty GPU batch", "[physical_limit][unordered_limit]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(Tier::GPU, 0);
  auto batch   = make_range_batch(*space, 0, 10);
  sirius_physical_unordered_streaming_limit zero({sirius::from_duckdb(duckdb::LogicalType::BIGINT)},
                                                 BoundLimitNode::ConstantValue(0),
                                                 BoundLimitNode::ConstantValue(0),
                                                 100,
                                                 true);
  CHECK_FALSE(zero.is_limit_exhausted());
  std::unique_ptr<operator_data> input =
    std::make_unique<pipelineable_operator_data>(std::vector{batch});
  auto result = zero.execute(*input, cudf::get_default_stream());
  CHECK(zero.is_limit_exhausted());
  auto& output = dynamic_cast<pipelineable_operator_data&>(*result);
  REQUIRE(output.get_data_batches().size() == 1);
  CHECK(collect_all_rows(output.get_data_batches()).empty());
}

TEST_CASE("unordered limit execute refunds its claim when output construction fails",
          "[unordered_limit_cpu]")
{
  // Metadata-only disk batches: no file is created or read and no GPU allocation is needed.
  cucascade::memory::disk_memory_space_config config;
  config.mount_paths     = "/tmp";
  config.memory_capacity = 1024;
  cucascade::memory::memory_space space(config);
  auto make_input = [&](std::size_t rows) -> std::unique_ptr<operator_data> {
    auto table = std::make_unique<cucascade::memory::disk_table_allocation>(
      "", std::vector<cucascade::memory::column_metadata>{}, 0);
    table->columns.emplace_back();
    table->columns[0].num_rows = rows;
    auto batch                 = data_batch::make(
      0, std::make_unique<cucascade::disk_data_representation>(std::move(table), space));
    return std::make_unique<pipelineable_operator_data>(std::vector{std::move(batch)});
  };
  sirius_physical_unordered_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(27), BoundLimitNode::ConstantValue(13), 100, true);
  auto input = make_input(20);
  // A disk representation supplies valid row metadata but cannot supply a GPU table view.
  // This exception occurs inside execute after quota reservation, before any CUDA operation.
  CHECK_THROWS_AS(limit.execute(*input, cudf::get_default_stream()), std::bad_cast);
  CHECK_FALSE(limit.is_limit_exhausted());
  sirius_physical_unordered_streaming_limit::claim_request retry(limit, 40);
  CHECK(retry.slice_begin() == 13);
  CHECK(retry.slice_count() == 27);
  retry.approve();
  CHECK(limit.is_limit_exhausted());
}

using unordered_claim = sirius_physical_unordered_streaming_limit::claim_request;
static_assert(!std::is_copy_constructible_v<unordered_claim>);
static_assert(!std::is_move_constructible_v<unordered_claim>);

TEST_CASE("unordered limit refunds quota during OOM unwinding", "[unordered_limit_cpu]")
{
  sirius_physical_unordered_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(7), BoundLimitNode::ConstantValue(3), 100, true);
  auto fail = [&] {
    unordered_claim request(limit, 10);
    CHECK_FALSE(limit.is_limit_exhausted());
    throw rmm::out_of_memory("injected claim failure");
  };
  CHECK_THROWS_AS(fail(), rmm::out_of_memory);
  CHECK_FALSE(limit.is_limit_exhausted());
  unordered_claim retry(limit, 10);
  CHECK(retry.slice_begin() == 3);
  CHECK(retry.slice_count() == 7);
  retry.approve();
  CHECK(limit.is_limit_exhausted());
}

TEST_CASE("unordered limit rollback preserves concurrent approvals", "[unordered_limit_cpu]")
{
  sirius_physical_unordered_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(10), BoundLimitNode::ConstantValue(5), 100, true);
  {
    unordered_claim failed(limit, 8);
    CHECK(failed.slice_begin() == 5);
    CHECK(failed.slice_count() == 3);
    unordered_claim success(limit, 10);
    CHECK(success.slice_count() == 7);
    success.approve();
    CHECK_FALSE(limit.is_limit_exhausted());
    // Destruction refunds only the first claim, retaining the second task's seven rows.
  }
  CHECK_FALSE(limit.is_limit_exhausted());
  unordered_claim retry(limit, 20);
  CHECK(retry.slice_begin() == 5);
  CHECK(retry.slice_count() == 3);
  retry.approve();
  retry.approve();  // Approval is idempotent.
  CHECK(limit.is_limit_exhausted());
}

TEST_CASE("unordered limit exhaustion does not wait for overlapping zero-row claims",
          "[unordered_limit_cpu]")
{
  sirius_physical_unordered_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(5), BoundLimitNode(), 100, true);
  unordered_claim final_rows(limit, 5);
  CHECK_FALSE(limit.is_limit_exhausted());
  {
    unordered_claim empty(limit, 5);
    CHECK(empty.slice_count() == 0);
    final_rows.approve();
    CHECK(limit.is_limit_exhausted());
    // An open stream can keep admitting zero-row requests while earlier ones are still running.
    // They cannot refund the successful result quota and must not keep the source open.
    for (int i = 0; i < 8; ++i) {
      unordered_claim overlapping(limit, 5);
      CHECK(overlapping.slice_count() == 0);
      CHECK(limit.is_limit_exhausted());
      if (i % 2 == 0) { overlapping.approve(); }
    }
  }
  CHECK(limit.is_limit_exhausted());
}

TEST_CASE("unordered limit exhaustion still waits for refundable offset quota",
          "[unordered_limit_cpu]")
{
  sirius_physical_unordered_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(5), BoundLimitNode::ConstantValue(3), 100, true);
  unordered_claim offset(limit, 3);
  CHECK(offset.slice_begin() == 3);
  CHECK(offset.slice_count() == 0);
  unordered_claim final_rows(limit, 5);
  final_rows.approve();
  CHECK_FALSE(limit.is_limit_exhausted());
  offset.approve();
  CHECK(limit.is_limit_exhausted());
}

TEST_CASE("unordered limit zero requires a successful output", "[unordered_limit_cpu]")
{
  sirius_physical_unordered_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(0), BoundLimitNode(), 100, true);
  {
    unordered_claim failed(limit, 0);
    CHECK_FALSE(limit.is_limit_exhausted());
  }
  CHECK_FALSE(limit.is_limit_exhausted());
  unordered_claim overlapping(limit, 0);
  unordered_claim success(limit, 0);
  success.approve();
  CHECK(limit.is_limit_exhausted());
}

TEST_CASE("unordered limit concurrent failures return quota for retry", "[unordered_limit_cpu]")
{
  sirius_physical_unordered_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(127), BoundLimitNode::ConstantValue(23), 1000, true);
  std::vector<std::future<std::pair<std::size_t, std::size_t>>> tasks;
  for (int i = 0; i < 32; ++i) {
    tasks.push_back(std::async(std::launch::async, [&, i] {
      unordered_claim claim(limit, 10);
      if (i % 2 == 0) { return std::pair<std::size_t, std::size_t>{0, 0}; }
      auto result = std::pair{claim.slice_begin(), claim.slice_count()};
      claim.approve();
      return result;
    }));
  }
  std::size_t skipped = 0, taken = 0;
  for (auto& task : tasks) {
    auto [skip, take] = task.get();
    skipped += skip;
    taken += take;
  }
  unordered_claim retry(limit, 1000);
  skipped += retry.slice_begin();
  taken += retry.slice_count();
  retry.approve();
  CHECK(skipped == 23);
  CHECK(taken == 127);
  CHECK(limit.is_limit_exhausted());
}

TEST_CASE("full-barrier limit drains trailing spilled batches without scheduling GPU work",
          "[ordered_limit_cpu]")
{
  limit_metadata_inputs inputs;
  // Publish in reverse logical order to exercise the sparse position snapshot.
  std::vector<std::weak_ptr<data_batch>> unused;
  for (std::size_t position : {30, 20, 9, 4}) {
    unused.push_back(inputs.add(100, position));
  }
  auto selected = inputs.add(100, 1);
  sirius_physical_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(10), BoundLimitNode(), 500, true);
  inputs.attach_to(limit);
  auto task = limit.get_next_task_input_data();
  REQUIRE(task);
  auto& slice = dynamic_cast<limiting_operator_data&>(*task);
  CHECK(slice.get_position() == 1);
  CHECK(slice.slice_begin == 0);
  CHECK(slice.slice_count == 10);
  REQUIRE(slice.get_data_batches().size() == 1);
  CHECK(slice.get_data_batches().front() == selected.lock());
  CHECK(inputs.repo.total_size() == 4);
  // The next scheduling call drains all four disk batches, never creating tasks that could
  // restore them in prepare_for_processing(). The first task has not executed yet.
  CHECK_FALSE(limit.get_next_task_input_data());
  CHECK(inputs.repo.all_empty());
  for (const auto& batch : unused) {
    CHECK(batch.expired());
  }
  CHECK(inputs.space.get_io_context() == nullptr);
  CHECK_FALSE(limit.is_limit_exhausted());
  // A retry of the selected task retains its immutable claim even after the repository drains.
  auto retry              = slice.with_batches_preserving_rows(slice.get_data_batches());
  const auto& retry_slice = dynamic_cast<const limiting_operator_data&>(*retry);
  CHECK(retry_slice.slice_begin == 0);
  CHECK(retry_slice.slice_count == 10);
  CHECK(retry_slice.get_position() == 1);
}

TEST_CASE("full-barrier limit skips offset-only inputs without GPU tasks", "[ordered_limit_cpu]")
{
  const bool positioned = GENERATE(false, true);
  limit_metadata_inputs inputs;
  auto prefix   = inputs.add(100, positioned ? std::optional<std::size_t>{2} : std::nullopt);
  auto selected = inputs.add(100, positioned ? std::optional<std::size_t>{5} : std::nullopt);
  auto trailing = inputs.add(100, positioned ? std::optional<std::size_t>{9} : std::nullopt);
  sirius_physical_streaming_limit limit(
    {}, BoundLimitNode::ConstantValue(50), BoundLimitNode::ConstantValue(150), 300, true);
  inputs.attach_to(limit);
  auto task = limit.get_next_task_input_data();
  REQUIRE(task);
  const auto& slice = dynamic_cast<const limiting_operator_data&>(*task);
  CHECK(slice.slice_begin == 50);
  CHECK(slice.slice_count == 50);
  CHECK(slice.get_data_batches().front() == selected.lock());
  CHECK(prefix.expired());
  CHECK_FALSE(limit.get_next_task_input_data());
  CHECK(trailing.expired());
  CHECK(inputs.repo.all_empty());
  CHECK(inputs.space.get_io_context() == nullptr);
}

TEST_CASE("full-barrier limit keeps one empty-output task when no rows are selected",
          "[ordered_limit_cpu]")
{
  const auto limit_rows = GENERATE(std::size_t{0}, std::size_t{5});
  const bool positioned = GENERATE(false, true);
  limit_metadata_inputs inputs;
  auto first  = inputs.add(100, positioned ? std::optional<std::size_t>{2} : std::nullopt);
  auto second = inputs.add(100, positioned ? std::optional<std::size_t>{9} : std::nullopt);
  sirius_physical_streaming_limit limit({},
                                        BoundLimitNode::ConstantValue(limit_rows),
                                        BoundLimitNode::ConstantValue(limit_rows == 0 ? 0 : 250),
                                        200,
                                        true);
  inputs.attach_to(limit);
  auto task = limit.get_next_task_input_data();
  REQUIRE(task);
  const auto& slice = dynamic_cast<const limiting_operator_data&>(*task);
  CHECK(slice.slice_count == 0);
  CHECK(slice.get_data_batches().front() == first.lock());
  CHECK(slice.get_position() == (positioned ? std::optional<std::size_t>{2} : std::nullopt));
  CHECK(second.expired());
  CHECK(inputs.repo.all_empty());
  CHECK_FALSE(limit.get_next_task_input_data());
  CHECK(inputs.space.get_io_context() == nullptr);
}

TEST_CASE("full-barrier limit zero and offset past input produce one empty GPU batch",
          "[physical_limit][ordering]")
{
  const auto limit_rows = GENERATE(std::size_t{0}, std::size_t{5});
  auto manager          = sirius::test::operator_utils::initialize_memory_manager();
  auto* space           = manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);
  sirius_physical_streaming_limit limit({sirius::from_duckdb(duckdb::LogicalType::BIGINT)},
                                        BoundLimitNode::ConstantValue(limit_rows),
                                        BoundLimitNode::ConstantValue(limit_rows == 0 ? 0 : 250),
                                        200,
                                        true);
  auto result =
    run_limit(limit, {make_range_batch(*space, 0, 100), make_range_batch(*space, 100, 100)});
  const auto& batches = dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches();
  REQUIRE(batches.size() == 1);
  CHECK(collect_all_rows(batches).empty());
}
