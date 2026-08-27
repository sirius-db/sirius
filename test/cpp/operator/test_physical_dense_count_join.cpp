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

// Host reference for one COUNT(col) dense shape: every preserved key contributes its multiplicity
// and every in-domain counted key one match.
std::vector<std::pair<int32_t, int64_t>> expected_dense_groups(
  const std::vector<int32_t>& preserved,
  const std::vector<int32_t>& counted,
  int32_t min_key,
  std::size_t slots)
{
  std::vector<int64_t> presence(slots, 0);
  std::vector<int64_t> matches(slots, 0);
  for (auto key : preserved) {
    ++presence[static_cast<std::size_t>(key - min_key)];
  }
  for (auto key : counted) {
    auto const offset = static_cast<int64_t>(key) - min_key;
    if (offset >= 0 && offset < static_cast<int64_t>(slots)) {
      ++matches[static_cast<std::size_t>(offset)];
    }
  }
  std::vector<std::pair<int32_t, int64_t>> rows;
  for (std::size_t k = 0; k < slots; ++k) {
    if (presence[k] > 0) {
      rows.emplace_back(min_key + static_cast<int32_t>(k), presence[k] * matches[k]);
    }
  }
  return rows;
}

// Drives dense_count_state directly, so a test can choose the emit and accumulate paths by shape.
// A layout may be wider than the data needs, so an inflated preserved-row bound is how
// forced_slot_bytes reaches the 64-bit slots without changing the keys under test.
std::vector<std::pair<int32_t, int64_t>> run_dense_count_state(
  cucascade::memory::memory_space& space,
  const std::vector<int32_t>& preserved,
  const std::vector<int32_t>& counted,
  int32_t min_key,
  int32_t max_key,
  std::size_t forced_slot_bytes = sizeof(uint32_t))
{
  auto mr     = get_resource_ref(space);
  auto stream = default_stream();

  auto const preserved_bound = forced_slot_bytes == sizeof(uint64_t)
                                 ? int64_t{1} << 32
                                 : static_cast<int64_t>(preserved.size());
  auto const layout          = dense_count_layout::plan(
    min_key, max_key, preserved_bound, static_cast<int64_t>(counted.size()));
  REQUIRE(layout);
  REQUIRE(layout->slot_bytes() == forced_slot_bytes);
  dense_count_state state(*layout, stream, mr);

  auto preserved_batch = make_numeric_batch<int32_t>(space, preserved, cudf::type_id::INT32);
  state.accumulate_preserved(sirius::get_cudf_table_view(*preserved_batch).column(0), stream);
  std::shared_ptr<cucascade::data_batch> counted_batch;
  if (!counted.empty()) {
    counted_batch = make_numeric_batch<int32_t>(space, counted, cudf::type_id::INT32);
    state.accumulate_counted(
      sirius::get_cudf_table_view(*counted_batch).column(0), std::nullopt, stream);
  }

  auto table        = state.emit(cudf::data_type{cudf::type_id::INT32},
                          dense_count_semantics::for_count_star(false),
                          /*null_group_rows=*/0,
                          dense_count_bounds{static_cast<int64_t>(preserved.size()),
                                             static_cast<int64_t>(counted.size())},
                          stream,
                          mr);
  auto const keys   = copy_column_to_host<int32_t>(table->view().column(0));
  auto const counts = copy_column_to_host<int64_t>(table->view().column(1));
  std::vector<std::pair<int32_t, int64_t>> rows;
  rows.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    rows.emplace_back(keys[i], counts[i]);
  }
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

TEST_CASE("dense_count_join: a matched key whose COUNT(col) values are all NULL counts zero",
          "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  // Key 2 is present on both sides, so it matches, but every counted argument for it is NULL.
  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int32_t>(*space, {1, 2, 2}, cudf::type_id::INT32)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{
    make_counted_batch(*space, {1, 2, 2}, {10, 0, 0}, {true, false, false})};

  SECTION("COUNT(col) keeps the matched all-NULL group at zero")
  {
    const std::vector<group_row> expected{{1, 1}, {2, 0}};
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
  SECTION("COUNT(*) counts the matched rows regardless of argument NULLs")
  {
    const std::vector<group_row> expected{{1, 1}, {2, 4}};
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

  const std::vector<int32_t> preserved{5, 6, 6, 8};
  const std::vector<int32_t> counted{6, 6, 6, 9, 4};
  const std::vector<std::pair<int32_t, int64_t>> expected{{5, 0}, {6, 6}, {8, 0}};

  for (auto slot_bytes : {sizeof(uint32_t), sizeof(uint64_t)}) {
    CHECK(run_dense_count_state(*space, preserved, counted, 5, 8, slot_bytes) == expected);
  }
}

TEST_CASE("dense_count_join emit covers the identity and gathered shapes", "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  SECTION("a gap-free preserved domain needs no gather map")
  {
    const std::vector<int32_t> preserved{0, 1, 1, 2, 3};
    const std::vector<int32_t> counted{1, 2, 2, 7};
    CHECK(run_dense_count_state(*space, preserved, counted, 0, 3) ==
          expected_dense_groups(preserved, counted, 0, 4));
  }

  SECTION("a gapped preserved domain counts before gathering")
  {
    const std::vector<int32_t> preserved{5, 6, 6, 8};
    const std::vector<int32_t> counted{6, 6, 6, 9, 4};
    CHECK(run_dense_count_state(*space, preserved, counted, 5, 8) ==
          expected_dense_groups(preserved, counted, 5, 4));
  }

  SECTION("a domain far wider than the preserved rows gathers the occupied slots")
  {
    const std::vector<int32_t> preserved{0, 9};
    const std::vector<int32_t> counted{0, 0, 9};
    CHECK(run_dense_count_state(*space, preserved, counted, 0, 9) ==
          expected_dense_groups(preserved, counted, 0, 10));
  }
}

TEST_CASE("dense_count_join accumulation agrees across the shared-memory gate",
          "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  // Accumulation is privatized into shared memory when the domain fits the 48 KiB budget and the
  // batch carries at least eight rows per slot.
  constexpr int32_t gate_slots = 48 * 1024 / static_cast<int32_t>(sizeof(uint32_t));

  auto keys_covering = [](int32_t slots, int64_t rows) {
    std::vector<int32_t> keys;
    keys.reserve(static_cast<std::size_t>(rows));
    for (int64_t i = 0; i < rows; ++i) {
      keys.push_back(static_cast<int32_t>(i % slots));
    }
    return keys;
  };
  auto check_domain = [&](int32_t slots, int64_t rows, std::size_t slot_bytes = sizeof(uint32_t)) {
    auto const keys = keys_covering(slots, rows);
    CHECK(run_dense_count_state(*space, keys, keys, 0, slots - 1, slot_bytes) ==
          expected_dense_groups(keys, keys, 0, static_cast<std::size_t>(slots)));
  };

  SECTION("a single slot") { check_domain(1, 4096); }
  SECTION("a low-cardinality domain") { check_domain(25, 4096); }
  SECTION("a low-cardinality domain in wide slots") { check_domain(25, 4096, sizeof(uint64_t)); }
  SECTION("exactly at the gate") { check_domain(gate_slots, int64_t{8} * gate_slots); }
  SECTION("just above the gate") { check_domain(gate_slots + 1, int64_t{8} * (gate_slots + 1)); }
  SECTION("too few rows per slot to privatize") { check_domain(1024, 8 * 1024 - 1); }

  // Out-of-domain counted keys must be rejected by the privatized kernel too. Its bounds check
  // guards a shared-memory write, so a miss corrupts a live block rather than faulting.
  SECTION("out-of-domain counted keys while privatizing")
  {
    constexpr int32_t slots = 25;
    auto const preserved    = keys_covering(slots, 4096);
    auto counted            = keys_covering(slots, 4096);
    for (int32_t stray : {-1, -1000, slots, slots + 1000}) {
      counted.push_back(stray);
    }
    CHECK(run_dense_count_state(*space, preserved, counted, 0, slots - 1) ==
          expected_dense_groups(preserved, counted, 0, static_cast<std::size_t>(slots)));
  }
}

TEST_CASE("dense_count_join admits a histogram at exactly the configured byte budget",
          "[dense_count_join]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch<int32_t>(*space, {0, 1, 2, 3, 4, 5, 6, 7}, cudf::type_id::INT32)};
  auto const layout = dense_count_layout::plan(/*min_key=*/0,
                                               /*max_key=*/7,
                                               /*preserved_rows=*/8,
                                               /*counted_rows=*/0);
  REQUIRE(layout);

  const std::vector<group_row> expected{
    {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};
  for (auto [max_bytes, strategy] :
       {std::pair{layout->total_bytes(), sirius_physical_dense_count_join::strategy::DENSE},
        std::pair{layout->total_bytes() - 1, sirius_physical_dense_count_join::strategy::SPARSE}}) {
    auto rows = run_dense_count_join<int32_t>(
      *space, duckdb::LogicalTypeId::INTEGER, preserved, {}, std::size_t{0}, max_bytes, strategy);
    REQUIRE(rows == expected);
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

  SECTION("the batch split index records both sides")
  {
    dense_count_join_input input({batch}, {batch});
    REQUIRE(input.preserved_count() == 1);
    REQUIRE(input.counted_count() == 1);
    REQUIRE(input.get_data_batches().size() == 2);
  }

  SECTION("an out-of-range preserved key index fails instead of reading past the batch")
  {
    auto op = make_operator(1, 0, std::nullopt);
    dense_count_join_input input({batch}, {});
    REQUIRE_THROWS_AS(op->execute(input, default_stream()), std::out_of_range);
  }

  SECTION("an out-of-range COUNT argument index fails instead of reading past the batch")
  {
    auto op = make_operator(0, 0, std::size_t{1});
    dense_count_join_input input({}, {batch});
    REQUIRE_THROWS_AS(op->execute(input, default_stream()), std::out_of_range);
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

TEST_CASE("dense_count_join rejects an unrepresentable histogram layout",
          "[dense_count_join][validation]")
{
  CHECK_FALSE(dense_count_layout::plan(
                0, std::numeric_limits<int64_t>::max(), int64_t{1} << 32, int64_t{1} << 32)
                .has_value());
  CHECK_FALSE(dense_count_layout::plan(
                std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(), 1, 1)
                .has_value());
  CHECK_FALSE(dense_count_layout::plan(5, 4, 0, 0).has_value());
  // Representable as an int64_t range, so only the 2 * range * slot_bytes guard rejects this one.
  CHECK_FALSE(dense_count_layout::plan(0, int64_t{1} << 62, 1, 1).has_value());
}

TEST_CASE("dense_count_layout sizes slots from the domain and slot width from the row counts",
          "[dense_count_join]")
{
  constexpr auto uint32_max = std::numeric_limits<uint32_t>::max();

  auto const narrow = dense_count_layout::plan(-4, 5, int64_t{uint32_max} - 1, 0);
  REQUIRE(narrow);
  CHECK(narrow->min_key() == -4);
  CHECK(narrow->slots() == 10);
  CHECK(narrow->slot_bytes() == sizeof(uint32_t));
  CHECK(narrow->total_bytes() == 2 * narrow->slots() * narrow->slot_bytes());

  auto const wide = dense_count_layout::plan(-4, 5, int64_t{uint32_max}, 0);
  REQUIRE(wide);
  CHECK(wide->slot_bytes() == sizeof(uint64_t));
  CHECK(wide->total_bytes() == 2 * wide->slots() * wide->slot_bytes());

  // Either row count widens the slots: the counted side is what a count slot can wrap on.
  auto const narrow_counted = dense_count_layout::plan(-4, 5, 0, int64_t{uint32_max} - 1);
  REQUIRE(narrow_counted);
  CHECK(narrow_counted->slot_bytes() == sizeof(uint32_t));
  auto const wide_counted = dense_count_layout::plan(-4, 5, 0, int64_t{uint32_max});
  REQUIRE(wide_counted);
  CHECK(wide_counted->slot_bytes() == sizeof(uint64_t));
}

TEST_CASE("dense_count_bounds gates BIGINT overflow detection", "[dense_count_join][validation]")
{
  constexpr auto int64_max = std::numeric_limits<int64_t>::max();

  CHECK_FALSE(dense_count_bounds{int64_max, 0}.may_exceed_bigint());
  CHECK_FALSE(dense_count_bounds{int64_max, 1}.may_exceed_bigint());
  CHECK_FALSE(dense_count_bounds{int64_max / 5, 5}.may_exceed_bigint());
  CHECK(dense_count_bounds{int64_max / 5 + 1, 5}.may_exceed_bigint());
  CHECK(dense_count_bounds{int64_max / 2 + 1, 2}.may_exceed_bigint());

  // COUNT(*) floors an unmatched group at one match, so its match bound is never zero and its
  // products are never exempt from validation.
  CHECK(dense_count_semantics::for_count_star(true).max_matched(0) == 1);
  CHECK(dense_count_semantics::for_count_star(false).max_matched(0) == 0);
}

TEST_CASE("dense_count_join emit validates products when the bounds allow overflow",
          "[dense_count_join][validation]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto mr     = get_resource_ref(*space);
  auto stream = default_stream();

  auto preserved = make_numeric_batch<int32_t>(*space, {5, 6, 6, 8}, cudf::type_id::INT32);
  auto counted   = make_numeric_batch<int32_t>(*space, {6, 6, 6, 9, 4}, cudf::type_id::INT32);

  auto const layout = dense_count_layout::plan(/*min_key=*/5, /*max_key=*/8, 4, 5);
  REQUIRE(layout);
  dense_count_state state(*layout, stream, mr);
  state.accumulate_preserved(sirius::get_cudf_table_view(*preserved).column(0), stream);
  state.accumulate_counted(sirius::get_cudf_table_view(*counted).column(0), std::nullopt, stream);

  // Bounds this coarse cannot rule the product out, so emit arms the device overflow flag and reads
  // it back; the actual products are tiny, so it must report clean and emit the unvalidated result.
  dense_count_bounds const bounds{std::numeric_limits<int64_t>::max(), 2};
  REQUIRE(bounds.may_exceed_bigint());
  auto table = state.emit(cudf::data_type{cudf::type_id::INT32},
                          dense_count_semantics::for_count_star(false),
                          /*null_group_rows=*/0,
                          bounds,
                          stream,
                          mr);
  CHECK(copy_column_to_host<int32_t>(table->view().column(0)) == std::vector<int32_t>{5, 6, 8});
  CHECK(copy_column_to_host<int64_t>(table->view().column(1)) == std::vector<int64_t>{0, 6, 0});
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

TEST_CASE("dense_count_join: a retried task re-executes on the same input",
          "[dense_count_join][validation]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto const stream = default_stream();

  // NULL preserved keys, so the NULL-group accounting is exercised on every run.
  std::vector<std::shared_ptr<cucascade::data_batch>> preserved{
    make_numeric_batch_with_nulls<int32_t>(
      *space, {1, 2, 2, 0, 0}, {true, true, true, false, false}, cudf::type_id::INT32)};
  std::vector<std::shared_ptr<cucascade::data_batch>> counted{make_counted_batch(
    *space, {1, 1, 2, 2, 0}, {10, 0, 10, 10, 10}, {true, false, true, true, true})};

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  // An OOM'd task is rescheduled carrying the same input_data and re-enters execute() from the
  // start, so a second execute() on one operator instance must behave exactly like the first.
  auto const require_repeatable = [&](uint64_t max_bins_bytes,
                                      sirius_physical_dense_count_join::strategy expected) {
    sirius_physical_dense_count_join op(sirius::from_duckdb_vec(types),
                                        /*estimated_cardinality=*/16,
                                        /*preserved_key_idx=*/0,
                                        /*counted_key_idx=*/0,
                                        std::size_t{1},
                                        max_bins_bytes);
    dense_count_join_input input(preserved, counted);

    auto first = op.execute(input, stream);
    stream.synchronize();
    REQUIRE(op.last_strategy() == expected);

    std::unique_ptr<sirius::op::operator_data> second;
    REQUIRE_NOTHROW(second = op.execute(input, stream));
    stream.synchronize();
    REQUIRE(op.last_strategy() == expected);

    auto const rows_of = [](sirius::op::operator_data const& data) {
      auto const& batches =
        dynamic_cast<const pipelineable_operator_data&>(data).get_data_batches();
      REQUIRE(batches.size() == 1);
      auto const view = sirius::get_cudf_table_view(*batches[0]);
      return std::pair{copy_column_to_host<int32_t>(view.column(0)),
                       copy_column_to_host<int64_t>(view.column(1))};
    };
    // The input is read through column views only, so the retry sees identical bytes.
    CHECK(rows_of(*first) == rows_of(*second));
  };

  SECTION("dense path")
  {
    require_repeatable(k_default_max_bytes, sirius_physical_dense_count_join::strategy::DENSE);
  }
  SECTION("sparse path")
  {
    require_repeatable(k_tiny_max_bytes, sirius_physical_dense_count_join::strategy::SPARSE);
  }
}
