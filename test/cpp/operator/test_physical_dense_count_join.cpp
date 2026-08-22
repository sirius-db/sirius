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

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <op/aggregate/dense_count_join_impl.hpp>
#include <op/sirius_physical_dense_count_join.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace duckdb;
using namespace sirius::op;
using namespace sirius::test::operator_utils;

namespace {

// nullopt denotes the SQL NULL group.
using group_row = std::pair<std::optional<int64_t>, int64_t>;

template <typename KeyT>
std::vector<group_row> run_dense_count_join(
  cucascade::memory::memory_space& space,
  duckdb::LogicalTypeId key_logical_type,
  const std::vector<std::shared_ptr<cucascade::data_batch>>& preserved_batches,
  const std::vector<std::shared_ptr<cucascade::data_batch>>& counted_batches,
  std::optional<std::size_t> counted_value_idx,
  uint64_t max_bins_bytes,
  sirius_physical_dense_count_join::strategy expected_strategy,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(key_logical_type));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  sirius_physical_dense_count_join op(sirius::from_duckdb_vec(types),
                                      /*estimated_cardinality=*/16,
                                      /*preserved_key_idx=*/0,
                                      /*counted_key_idx=*/0,
                                      counted_value_idx,
                                      max_bins_bytes);

  dense_count_join_input input(preserved_batches, counted_batches);

  auto output = op.execute(input, stream);
  stream.synchronize();
  REQUIRE(op.last_strategy() == expected_strategy);

  auto const& out_batches =
    dynamic_cast<const pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(out_batches.size() == 1);
  auto const view = sirius::get_cudf_table_view(*out_batches[0]);
  REQUIRE(view.num_columns() == 2);
  REQUIRE(view.column(1).type().id() == cudf::type_id::INT64);

  auto const keys     = copy_column_to_host<KeyT>(view.column(0));
  auto const validity = copy_validity_to_host(view.column(0));
  auto const counts   = copy_column_to_host<int64_t>(view.column(1));

  std::vector<group_row> rows;
  rows.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    rows.emplace_back(
      validity[i] ? std::optional<int64_t>(static_cast<int64_t>(keys[i])) : std::nullopt,
      counts[i]);
  }
  // Sparse output order is unspecified.
  std::sort(rows.begin(), rows.end());
  return rows;
}

constexpr uint64_t k_default_max_bytes = 2ULL * 1024 * 1024 * 1024;
// Eight bytes admit one u32 presence/count slot and force these tests through the sparse path.
constexpr uint64_t k_tiny_max_bytes = 8;

std::shared_ptr<cucascade::data_batch> make_counted_batch(cucascade::memory::memory_space& space,
                                                          const std::vector<int32_t>& keys,
                                                          const std::vector<int64_t>& values,
                                                          const std::vector<bool>& value_valids)
{
  auto key_batch = make_numeric_batch<int32_t>(space, keys, cudf::type_id::INT32);
  auto value_batch =
    make_numeric_batch_with_nulls<int64_t>(space, values, value_valids, cudf::type_id::INT64);
  return concatenate_batches_horizontal({key_batch, value_batch}, space);
}

}  // namespace

TEST_CASE("dense_count_join: zero-count outer groups, duplicates, out-of-range counted keys",
          "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int32_t>(*space, {1, 2, 2, 3}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {4, 5, 6}, cudf::type_id::INT32)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_counted_batch(*space,
                       {2, 2, 3, 5, 5, 5, 99},
                       {10, 10, 10, 10, 10, 10, 10},
                       {true, true, true, true, true, true, true})};

  const std::vector<group_row> expected{{1, 0}, {2, 4}, {3, 1}, {4, 0}, {5, 3}, {6, 0}};

  SECTION("dense strategy")
  {
    auto rows = run_dense_count_join<int32_t>(*space,
                                              duckdb::LogicalTypeId::INTEGER,
                                              preserved,
                                              counted,
                                              std::size_t{1},
                                              k_default_max_bytes,
                                              sirius_physical_dense_count_join::strategy::DENSE);
    REQUIRE(rows == expected);
  }
  SECTION("sparse strategy is byte-equivalent (dense gate negative)")
  {
    auto rows = run_dense_count_join<int32_t>(*space,
                                              duckdb::LogicalTypeId::INTEGER,
                                              preserved,
                                              counted,
                                              std::size_t{1},
                                              k_tiny_max_bytes,
                                              sirius_physical_dense_count_join::strategy::SPARSE);
    REQUIRE(rows == expected);
  }
}

TEST_CASE("dense_count_join: NULL keys and COUNT(col) NULL semantics", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch_with_nulls<int32_t>(
      *space, {1, 2, 2, 0, 0}, {true, true, true, false, false}, cudf::type_id::INT32)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{make_counted_batch(
    *space, {1, 1, 2, 2, 0}, {10, 0, 10, 10, 10}, {true, false, true, true, true})};

  SECTION("COUNT(col): NULL values excluded, NULL preserved keys form the 0-count NULL group")
  {
    const std::vector<group_row> expected{{std::nullopt, 0}, {1, 1}, {2, 4}};
    for (auto [max_bytes, strategy] :
         {std::pair{k_default_max_bytes, sirius_physical_dense_count_join::strategy::DENSE},
          std::pair{k_tiny_max_bytes, sirius_physical_dense_count_join::strategy::SPARSE}}) {
      auto rows = run_dense_count_join<int32_t>(*space,
                                                duckdb::LogicalTypeId::INTEGER,
                                                preserved,
                                                counted,
                                                std::size_t{1},
                                                max_bytes,
                                                strategy);
      REQUIRE(rows == expected);
    }
  }
  SECTION("COUNT(*): unmatched rows count 1 each; NULL group counts its own rows")
  {
    const std::vector<group_row> expected{{std::nullopt, 2}, {1, 2}, {2, 4}};
    for (auto [max_bytes, strategy] :
         {std::pair{k_default_max_bytes, sirius_physical_dense_count_join::strategy::DENSE},
          std::pair{k_tiny_max_bytes, sirius_physical_dense_count_join::strategy::SPARSE}}) {
      auto rows = run_dense_count_join<int32_t>(*space,
                                                duckdb::LogicalTypeId::INTEGER,
                                                preserved,
                                                counted,
                                                std::nullopt,
                                                max_bytes,
                                                strategy);
      REQUIRE(rows == expected);
    }
  }
}

TEST_CASE("dense_count_join: NULL counted keys never match", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int32_t>(*space, {0, 1}, cudf::type_id::INT32)};
  auto counted_keys = make_numeric_batch_with_nulls<int32_t>(
    *space, {0, 0, 0}, {true, false, false}, cudf::type_id::INT32);

  const std::vector<group_row> expected{{0, 1}, {1, 0}};
  for (auto [max_bytes, strategy] :
       {std::pair{k_default_max_bytes, sirius_physical_dense_count_join::strategy::DENSE},
        std::pair{k_tiny_max_bytes, sirius_physical_dense_count_join::strategy::SPARSE}}) {
    auto rows = run_dense_count_join<int32_t>(*space,
                                              duckdb::LogicalTypeId::INTEGER,
                                              preserved,
                                              {counted_keys},
                                              std::size_t{0},  // COUNT(key col) itself
                                              max_bytes,
                                              strategy);
    REQUIRE(rows == expected);
  }
}

TEST_CASE("dense_count_join: offset BIGINT key range", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{make_numeric_batch<int64_t>(
    *space, {1000000007LL, 1000000009LL, 1000000010LL}, cudf::type_id::INT64)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_numeric_batch<int64_t>(*space, {1000000009LL, 1000000009LL}, cudf::type_id::INT64)};

  const std::vector<group_row> expected{{1000000007LL, 0}, {1000000009LL, 2}, {1000000010LL, 0}};
  auto rows = run_dense_count_join<int64_t>(*space,
                                            duckdb::LogicalTypeId::BIGINT,
                                            preserved,
                                            counted,
                                            std::size_t{0},
                                            k_default_max_bytes,
                                            sirius_physical_dense_count_join::strategy::DENSE);
  REQUIRE(rows == expected);
}

TEST_CASE("dense_count_join: empty counted side emits all-zero counts", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int32_t>(*space, {7, 8, 9}, cudf::type_id::INT32)};

  const std::vector<group_row> expected{{7, 0}, {8, 0}, {9, 0}};
  auto rows = run_dense_count_join<int32_t>(*space,
                                            duckdb::LogicalTypeId::INTEGER,
                                            preserved,
                                            /*counted_batches=*/{},
                                            std::size_t{1},
                                            k_default_max_bytes,
                                            sirius_physical_dense_count_join::strategy::DENSE);
  REQUIRE(rows == expected);
}

TEST_CASE("dense_count_join: empty preserved side emits no groups", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_counted_batch(*space, {1, 2, 3}, {10, 10, 10}, {true, true, true})};

  auto rows = run_dense_count_join<int32_t>(*space,
                                            duckdb::LogicalTypeId::INTEGER,
                                            /*preserved_batches=*/{},
                                            counted,
                                            std::size_t{1},
                                            k_default_max_bytes,
                                            sirius_physical_dense_count_join::strategy::DENSE);
  REQUIRE(rows.empty());
}

TEST_CASE("dense_count_join: negative and zero keys address the offset histogram exactly",
          "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int32_t>(*space, {-3, -2, 0, 1}, cudf::type_id::INT32)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_numeric_batch<int32_t>(*space, {-2, -2, 1, 5}, cudf::type_id::INT32)};

  const std::vector<group_row> expected{{-3, 0}, {-2, 2}, {0, 0}, {1, 1}};
  for (auto [max_bytes, strategy] :
       {std::pair{k_default_max_bytes, sirius_physical_dense_count_join::strategy::DENSE},
        std::pair{k_tiny_max_bytes, sirius_physical_dense_count_join::strategy::SPARSE}}) {
    auto rows = run_dense_count_join<int32_t>(*space,
                                              duckdb::LogicalTypeId::INTEGER,
                                              preserved,
                                              counted,
                                              std::size_t{0},  // COUNT(key col)
                                              max_bytes,
                                              strategy);
    REQUIRE(rows == expected);
  }
}

TEST_CASE("dense_count_join: duplicate keys across batches accumulate on both sides",
          "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int32_t>(*space, {1, 2}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {2}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {3}, cudf::type_id::INT32)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_numeric_batch<int32_t>(*space, {2}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {2}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {3}, cudf::type_id::INT32)};

  const std::vector<group_row> expected{{1, 0}, {2, 4}, {3, 1}};
  for (auto [max_bytes, strategy] :
       {std::pair{k_default_max_bytes, sirius_physical_dense_count_join::strategy::DENSE},
        std::pair{k_tiny_max_bytes, sirius_physical_dense_count_join::strategy::SPARSE}}) {
    auto rows = run_dense_count_join<int32_t>(*space,
                                              duckdb::LogicalTypeId::INTEGER,
                                              preserved,
                                              counted,
                                              std::size_t{0},
                                              max_bytes,
                                              strategy);
    REQUIRE(rows == expected);
  }
}

TEST_CASE("dense_count_join: wide (u64) histogram slots match the u32 result", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  auto preserved = make_numeric_batch<int32_t>(*space, {5, 6, 6, 8}, cudf::type_id::INT32);
  auto counted   = make_numeric_batch<int32_t>(*space, {6, 6, 6, 9, 4}, cudf::type_id::INT32);
  auto const preserved_keys = sirius::get_cudf_table_view(*preserved).column(0);
  auto const counted_keys   = sirius::get_cudf_table_view(*counted).column(0);

  for (bool wide : {false, true}) {
    dense_count_state state(/*min_key=*/5, /*range=*/4, wide, stream, mr);
    REQUIRE(state.wide() == wide);
    state.accumulate_preserved(preserved_keys, stream);
    state.accumulate_counted(counted_keys, nullptr, stream);
    auto table        = state.emit(cudf::data_type{cudf::type_id::INT32},
                            /*count_star=*/false,
                            /*null_group_rows=*/0,
                            stream,
                            mr,
                            /*check_product_overflow=*/false);
    auto const keys   = copy_column_to_host<int32_t>(table->view().column(0));
    auto const counts = copy_column_to_host<int64_t>(table->view().column(1));
    REQUIRE(keys == std::vector<int32_t>{5, 6, 8});
    REQUIRE(counts == std::vector<int64_t>{0, 6, 0});
  }
}

TEST_CASE("dense_count_join: runtime density and input-cost gates", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  SECTION("tiny contiguous domain remains dense")
  {
    std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
      make_numeric_batch<int32_t>(*space, {3, 4}, cudf::type_id::INT32)};
    auto rows = run_dense_count_join<int32_t>(*space,
                                              duckdb::LogicalTypeId::INTEGER,
                                              preserved,
                                              {},
                                              std::size_t{0},
                                              k_default_max_bytes,
                                              sirius_physical_dense_count_join::strategy::DENSE);
    REQUIRE((rows == std::vector<group_row>{{3, 0}, {4, 0}}));
  }

  SECTION("within-budget but sparse domain avoids a disproportionate histogram")
  {
    std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
      make_numeric_batch<int32_t>(*space, {0, 100}, cudf::type_id::INT32)};
    std::vector<std::shared_ptr<cucascade::data_batch>> counted{
      make_numeric_batch<int32_t>(*space, {0}, cudf::type_id::INT32)};
    auto rows = run_dense_count_join<int32_t>(*space,
                                              duckdb::LogicalTypeId::INTEGER,
                                              preserved,
                                              counted,
                                              std::size_t{0},
                                              k_default_max_bytes,
                                              sirius_physical_dense_count_join::strategy::SPARSE);
    REQUIRE((rows == std::vector<group_row>{{0, 1}, {100, 0}}));
  }
}

TEST_CASE("dense_count_join: retained multi-batch extrema merge on a non-default stream",
          "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch_with_nulls<int32_t>(*space, {0, 0}, {false, false}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {4}, cudf::type_id::INT32),
    make_numeric_batch_with_nulls<int32_t>(
      *space, {0, 0, 0}, {false, false, false}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {5}, cudf::type_id::INT32),
    make_numeric_batch<int32_t>(*space, {4}, cudf::type_id::INT32)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_numeric_batch<int32_t>(*space, {4, 4}, cudf::type_id::INT32)};

  // The direct operator-test path does not run task input preparation, so order batch creation
  // before deliberately executing the operator on another stream.
  cudf::get_default_stream().synchronize();
  rmm::cuda_stream stream;
  auto rows = run_dense_count_join<int32_t>(*space,
                                            duckdb::LogicalTypeId::INTEGER,
                                            preserved,
                                            counted,
                                            std::size_t{0},
                                            k_default_max_bytes,
                                            sirius_physical_dense_count_join::strategy::DENSE,
                                            stream.view());
  REQUIRE((rows == std::vector<group_row>{{std::nullopt, 0}, {4, 4}, {5, 0}}));
}

TEST_CASE("dense_count_join: extreme INT64 domain takes exact sparse path", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  auto const min = std::numeric_limits<int64_t>::min();
  auto const max = std::numeric_limits<int64_t>::max();
  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int64_t>(*space, {min, max}, cudf::type_id::INT64)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_numeric_batch<int64_t>(*space, {min}, cudf::type_id::INT64)};

  auto rows = run_dense_count_join<int64_t>(*space,
                                            duckdb::LogicalTypeId::BIGINT,
                                            preserved,
                                            counted,
                                            std::size_t{0},
                                            k_default_max_bytes,
                                            sirius_physical_dense_count_join::strategy::SPARSE);
  REQUIRE((rows == std::vector<group_row>{{min, 1}, {max, 0}}));
}

TEST_CASE("dense_count_join rejects malformed batch metadata with diagnostics",
          "[dense_count_join][validation]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto batch = make_numeric_batch<int32_t>(*space, {1, 2}, cudf::type_id::INT32);

  auto make_operator = [](std::size_t preserved_key_idx,
                          std::size_t counted_key_idx,
                          std::optional<std::size_t> counted_value_idx) {
    duckdb::vector<duckdb::LogicalType> types;
    types.push_back(duckdb::LogicalType::INTEGER);
    types.push_back(duckdb::LogicalType::BIGINT);
    return std::make_unique<sirius_physical_dense_count_join>(sirius::from_duckdb_vec(types),
                                                              /*estimated_cardinality=*/2,
                                                              preserved_key_idx,
                                                              counted_key_idx,
                                                              counted_value_idx,
                                                              k_default_max_bytes);
  };

  SECTION("null batches are rejected before generic materialization can drop them")
  {
    std::shared_ptr<cucascade::data_batch> null_batch;
    REQUIRE_THROWS_WITH(dense_count_join_input({batch, null_batch}, {}),
                        Catch::Contains("null preserved batch at index 1"));
    REQUIRE_THROWS_WITH(dense_count_join_input({}, {null_batch}),
                        Catch::Contains("null counted batch at index 0"));
  }

  SECTION("batch side identity is explicit and aligned")
  {
    dense_count_join_input input({batch}, {batch});
    REQUIRE(input.input_sides() == std::vector<dense_count_join_input::input_side>{
                                     dense_count_join_input::input_side::PRESERVED,
                                     dense_count_join_input::input_side::COUNTED});
  }

  SECTION("preserved key index is checked before narrowing")
  {
    auto op = make_operator(1, 0, std::nullopt);
    dense_count_join_input input({batch}, {});
    REQUIRE_THROWS_WITH(op->execute(input, default_stream()),
                        Catch::Contains("preserved key column index 1 is out of range"));
  }

  SECTION("COUNT argument index is checked before narrowing")
  {
    auto op = make_operator(0, 0, std::size_t{1});
    dense_count_join_input input({}, {batch});
    REQUIRE_THROWS_WITH(op->execute(input, default_stream()),
                        Catch::Contains("COUNT argument column index 1 is out of range"));
  }
}
TEST_CASE("dense_count_join owns direct child port and barrier wiring",
          "[dense_count_join][pipeline]")
{
  duckdb::vector<duckdb::LogicalType> output_types;
  output_types.push_back(duckdb::LogicalType::INTEGER);
  output_types.push_back(duckdb::LogicalType::BIGINT);
  sirius_physical_dense_count_join op(sirius::from_duckdb_vec(output_types),
                                      /*estimated_cardinality=*/1,
                                      /*preserved_key_idx=*/0,
                                      /*counted_key_idx=*/0,
                                      /*counted_value_idx=*/std::nullopt,
                                      k_default_max_bytes);

  duckdb::vector<sirius::logical_type> child_types;
  child_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  auto preserved =
    duckdb::make_uniq<sirius_physical_operator>(SiriusPhysicalOperatorType::CONCAT, child_types, 1);
  auto* preserved_ptr = preserved.get();
  auto counted =
    duckdb::make_uniq<sirius_physical_operator>(SiriusPhysicalOperatorType::FILTER, child_types, 1);
  auto* counted_ptr = counted.get();
  op.children.push_back(std::move(preserved));
  op.children.push_back(std::move(counted));

  CHECK(op.input_port_for(*preserved_ptr) == sirius_physical_dense_count_join::PRESERVED_PORT);
  CHECK(op.input_port_for(*counted_ptr) == sirius_physical_dense_count_join::COUNTED_PORT);
  CHECK(op.input_barrier_for(*preserved_ptr) == MemoryBarrierType::FULL);
  CHECK(op.input_barrier_for(*counted_ptr) == MemoryBarrierType::FULL);
}

TEST_CASE("dense_count_join first-run estimate is proportional and saturates",
          "[dense_count_join][no_history_peak_memory_estimate]")
{
  constexpr std::size_t allocation_floor = 1024 * 1024;
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType::INTEGER);
  types.push_back(duckdb::LogicalType::BIGINT);

  constexpr uint64_t histogram_budget = 2ULL * 1024 * 1024 * 1024;
  sirius_physical_dense_count_join op(sirius::from_duckdb_vec(types),
                                      /*estimated_cardinality=*/2,
                                      /*preserved_key_idx=*/0,
                                      /*counted_key_idx=*/0,
                                      /*counted_value_idx=*/std::nullopt,
                                      histogram_budget);
  CHECK(op.max_bins_bytes() == histogram_budget);
  auto const tiny_estimate = op.no_history_peak_memory_estimate({1, 8});
  CHECK(tiny_estimate >= allocation_floor);
  CHECK(tiny_estimate < 2 * allocation_floor);
  CHECK(tiny_estimate < histogram_budget);

  input_stats gate_stats{4, 1024 * 1024};
  auto const low_cardinality_estimate = op.no_history_peak_memory_estimate(gate_stats);

  duckdb::vector<duckdb::LogicalType> high_cardinality_types;
  high_cardinality_types.push_back(duckdb::LogicalType::INTEGER);
  high_cardinality_types.push_back(duckdb::LogicalType::BIGINT);
  sirius_physical_dense_count_join high_cardinality(
    sirius::from_duckdb_vec(high_cardinality_types),
    /*estimated_cardinality=*/std::numeric_limits<std::size_t>::max(),
    /*preserved_key_idx=*/0,
    /*counted_key_idx=*/0,
    /*counted_value_idx=*/std::nullopt,
    histogram_budget);
  CHECK(high_cardinality.no_history_peak_memory_estimate(gate_stats) == low_cardinality_estimate);
  auto const max_admitted_histogram = std::min<std::size_t>(histogram_budget, 4 * gate_stats.bytes);
  CHECK(low_cardinality_estimate >= allocation_floor + max_admitted_histogram);

  duckdb::vector<duckdb::LogicalType> sparse_types;
  sparse_types.push_back(duckdb::LogicalType::INTEGER);
  sparse_types.push_back(duckdb::LogicalType::BIGINT);
  sirius_physical_dense_count_join sparse(sirius::from_duckdb_vec(sparse_types),
                                          /*estimated_cardinality=*/100,
                                          /*preserved_key_idx=*/0,
                                          /*counted_key_idx=*/0,
                                          /*counted_value_idx=*/std::nullopt,
                                          /*max_bins_bytes=*/8);
  CHECK(sparse.no_history_peak_memory_estimate({2, 100}) >= allocation_floor + 16 * 100);
  CHECK(sparse.no_history_peak_memory_estimate({2, std::numeric_limits<std::size_t>::max()}) ==
        std::numeric_limits<std::size_t>::max());
}

TEST_CASE("dense_count_join rejects histogram allocation arithmetic overflow",
          "[dense_count_join][validation]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  REQUIRE_THROWS_WITH(dense_count_state(/*min_key=*/0,
                                        std::numeric_limits<int64_t>::max(),
                                        /*wide=*/true,
                                        default_stream(),
                                        get_resource_ref(*space)),
                      Catch::Contains("exceeds size_t allocation capacity"));
}

TEST_CASE("dense_count_join exact rare-path BIGINT product validation",
          "[dense_count_join][validation]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto const stream = default_stream();
  auto const mr     = get_resource_ref(*space);

  auto lhs = make_numeric_batch<int64_t>(
    *space, {std::numeric_limits<int64_t>::max(), 4}, cudf::type_id::INT64);
  auto safe_rhs     = make_numeric_batch<int64_t>(*space, {1, 2}, cudf::type_id::INT64);
  auto overflow_rhs = make_numeric_batch<int64_t>(*space, {2, 2}, cudf::type_id::INT64);

  auto const lhs_view          = sirius::get_cudf_table_view(*lhs).column(0);
  auto const safe_rhs_view     = sirius::get_cudf_table_view(*safe_rhs).column(0);
  auto const overflow_rhs_view = sirius::get_cudf_table_view(*overflow_rhs).column(0);

  REQUIRE_NOTHROW(throw_if_count_product_overflows(lhs_view, safe_rhs_view, stream, mr));
  REQUIRE_THROWS_WITH(throw_if_count_product_overflows(lhs_view, overflow_rhs_view, stream, mr),
                      Catch::Contains("COUNT result exceeds BIGINT max"));
}
