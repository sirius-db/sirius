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

// Stage-1 Top-N sink self-consumption: the local operator prefilters batches against the shared
// coordinator boundary and offers each exact-K result's row K-1 tuple. Every equivalence case
// runs the same batches through a coordinator-armed pipeline and a plain pipeline and requires
// identical merged results -- the prefilter may only drop rows the final Top-N would drop anyway.

#include "operator_test_utils.hpp"

#include <cudf/null_mask.hpp>

#include <catch.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <helper/type_conversions.hpp>
#include <op/cudf_sort_order.hpp>
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
#include <op/dynamic_filter/top_n_threshold_coordinator.hpp>
#include <op/sirius_physical_top_n.hpp>
#include <op/sirius_physical_top_n_merge.hpp>
#include <sirius/exception.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace sirius::op;
using namespace cucascade;
using namespace sirius::test::operator_utils;
using sirius::op::pipelineable_operator_data;

namespace {

duckdb::BoundOrderByNode order_on(duckdb::idx_t col_idx,
                                  duckdb::OrderType dir,
                                  duckdb::OrderByNullType nulls,
                                  duckdb::LogicalType type = duckdb::LogicalType::BIGINT)
{
  return duckdb::BoundOrderByNode(
    dir, nulls, duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, col_idx));
}

/// Coordinator with semantics derived exactly as the planner derives them.
std::shared_ptr<top_n_threshold_coordinator> make_test_coordinator(
  duckdb::vector<duckdb::BoundOrderByNode> const& orders,
  std::vector<cudf::data_type> const& storage_types,
  std::size_t k,
  bool lex_admitted,
  dynamic_filter_stats* stats = nullptr)
{
  std::vector<top_n_key_semantics> keys;
  keys.reserve(orders.size());
  for (std::size_t i = 0; i < orders.size(); ++i) {
    keys.push_back({.storage_type = storage_types[i],
                    .order        = to_cudf_order(orders[i].type),
                    .null_order   = to_cudf_null_order(orders[i].type, orders[i].null_order)});
  }
  return std::make_shared<top_n_threshold_coordinator>(k, std::move(keys), lex_admitted, stats);
}

/// Host copy of an INT64 column honoring its null mask (columns here have zero offset).
std::vector<std::optional<std::int64_t>> copy_int64_with_nulls(cudf::column_view const& col)
{
  REQUIRE(col.offset() == 0);
  auto const values = copy_column_to_host<std::int64_t>(col);
  std::vector<std::optional<std::int64_t>> out(values.size());
  if (!col.nullable()) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      out[i] = values[i];
    }
    return out;
  }
  std::vector<cudf::bitmask_type> mask(cudf::num_bitmask_words(col.size()));
  cudaMemcpy(
    mask.data(), col.null_mask(), mask.size() * sizeof(cudf::bitmask_type), cudaMemcpyDeviceToHost);
  for (std::size_t i = 0; i < values.size(); ++i) {
    bool const valid = (mask[i / 32] >> (i % 32)) & 1U;
    if (valid) { out[i] = values[i]; }
  }
  return out;
}

/// Run the local operator batch-by-batch (one batch per execution, as the pipeline does) and
/// collect every output batch.
std::vector<std::shared_ptr<data_batch>> run_local(
  sirius_physical_top_n& op, std::vector<std::shared_ptr<data_batch>> const& batches)
{
  std::vector<std::shared_ptr<data_batch>> collected;
  for (auto const& batch : batches) {
    auto out = op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
    for (auto const& produced :
         dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches()) {
      collected.push_back(produced);
    }
  }
  return collected;
}

/// Full local-then-merge pipeline; returns every merged column as host values with validity.
std::vector<std::vector<std::optional<std::int64_t>>> run_pipeline(
  sirius_physical_top_n& local,
  sirius_physical_top_n_merge& merge,
  std::vector<std::shared_ptr<data_batch>> const& batches)
{
  auto const local_outputs = run_local(local, batches);
  auto out = merge.execute(pipelineable_operator_data(local_outputs), cudf::get_default_stream());
  auto const& merged = dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches();
  REQUIRE(merged.size() == 1);
  auto const view = sirius::get_cudf_table_view(*merged[0]);
  std::vector<std::vector<std::optional<std::int64_t>>> columns;
  for (cudf::size_type c = 0; c < view.num_columns(); ++c) {
    columns.push_back(copy_int64_with_nulls(view.column(c)));
  }
  return columns;
}

duckdb::vector<duckdb::LogicalType> bigint_types(std::size_t n)
{
  duckdb::vector<duckdb::LogicalType> types;
  for (std::size_t i = 0; i < n; ++i) {
    types.push_back(duckdb::LogicalType::BIGINT);
  }
  return types;
}

constexpr auto k_int64 = cudf::data_type{cudf::type_id::INT64};

/// One DECIMAL32 key column at @p scale; the shared utilities carry only the DECIMAL64 form.
std::shared_ptr<data_batch> make_decimal32_batch(cucascade::memory::memory_space& space,
                                                 std::vector<std::int32_t> const& values,
                                                 std::int32_t scale)
{
  auto const stream = default_stream();
  auto const mr     = get_resource_ref(space);
  auto column = cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL32, scale},
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
  cudaMemcpy(column->mutable_view().data<std::int32_t>(),
             values.data(),
             values.size() * sizeof(std::int32_t),
             cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  auto repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(table), space, stream);
  return data_batch::make(::sirius::get_next_batch_id(), std::move(repr));
}

}  // namespace

TEST_CASE("top_n prefilter single-key equivalence across direction and null order",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  int const dir_case  = GENERATE(0, 1);
  int const null_case = GENERATE(0, 1);
  auto const dir = dir_case == 0 ? duckdb::OrderType::ASCENDING : duckdb::OrderType::DESCENDING;
  auto const nulls =
    null_case == 0 ? duckdb::OrderByNullType::NULLS_FIRST : duckdb::OrderByNullType::NULLS_LAST;
  CAPTURE(dir_case, null_case);

  // One key column with duplicates and nulls; batch 1 establishes a boundary that prunes batch 2
  // (and a stale boundary for batch 3 remains safe).
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(
    make_numeric_batch_with_nulls<std::int64_t>(*space,
                                                {5, 0, 9, 3, 0, 7, 4, 8},
                                                {true, false, true, true, false, true, true, true},
                                                cudf::type_id::INT64));
  batches.push_back(make_numeric_batch_with_nulls<std::int64_t>(
    *space, {2, 0, 6, 1, 10, 3}, {true, false, true, true, true, true}, cudf::type_id::INT64));
  batches.push_back(make_numeric_batch<std::int64_t>(*space, {12, 11, 2, 6}, cudf::type_id::INT64));

  auto const make_orders = [&] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(order_on(0, dir, nulls));
    return orders;
  };
  std::size_t const limit  = 4;
  std::size_t const offset = 1;

  sirius_physical_top_n plain(
    sirius::from_duckdb_vec(bigint_types(1)), make_orders(), limit, offset, nullptr, 0);
  sirius_physical_top_n_merge plain_merge(&plain);

  dynamic_filter_stats stats;
  sirius_physical_top_n armed(
    sirius::from_duckdb_vec(bigint_types(1)), make_orders(), limit, offset, nullptr, 0);
  armed.threshold_coordinator =
    make_test_coordinator(armed.orders, {k_int64}, limit + offset, true, &stats);
  sirius_physical_top_n_merge armed_merge(&armed);

  auto const expected = run_pipeline(plain, plain_merge, batches);
  auto const actual   = run_pipeline(armed, armed_merge, batches);
  REQUIRE(actual == expected);

  // Every batch holds at least K rows before pruning, so offers flowed and only pruning varies.
  REQUIRE(stats.top_n_offers.load() >= 1);
  REQUIRE(stats.top_n_prefilter_rows_out.load() <= stats.top_n_prefilter_rows_in.load());
}

TEST_CASE("top_n prefilter equivalence per allowlisted key storage type",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // One deterministic boundary -> prefilter -> equivalence round trip per allowlisted storage
  // type, so every literal-construction branch (including the DATE day-count round trip) is
  // known to evaluate the predicate on the second batch.
  int const type_case = GENERATE(0, 1, 2, 3, 4, 5);
  auto const type_id  = std::array{cudf::type_id::INT8,
                                  cudf::type_id::INT16,
                                  cudf::type_id::INT32,
                                  cudf::type_id::TIMESTAMP_DAYS,
                                  cudf::type_id::DECIMAL32,
                                  cudf::type_id::DECIMAL64}[type_case];
  auto const key_type = std::array<duckdb::LogicalType, 6>{
    duckdb::LogicalType::TINYINT,
    duckdb::LogicalType::SMALLINT,
    duckdb::LogicalType::INTEGER,
    duckdb::LogicalType::DATE,
    duckdb::LogicalType::DECIMAL(9, 2),
    duckdb::LogicalType::DECIMAL(18, 4)}[static_cast<std::size_t>(type_case)];
  // The fixed-point cases carry a non-zero scale deliberately: the values below are the *scaled*
  // integers, so a path that rescaled anywhere -- extraction, the device literal, or the kernel's
  // widening -- would order them differently and the equivalence below would fail.
  auto const key_scale =
    std::array<std::int32_t, 6>{0, 0, 0, 0, -2, -4}[static_cast<std::size_t>(type_case)];
  auto const storage_type = cudf::is_fixed_point(cudf::data_type{type_id})
                              ? cudf::data_type{type_id, key_scale}
                              : cudf::data_type{type_id};
  CAPTURE(type_case);

  // Values fit every tested width; batch 1 sets boundary 33 (K = 5), so batch 2 must keep exactly
  // {1, 3, 4, 7, -6} when the typed literal compares correctly. The negative sits in batch 2 only,
  // which is the batch the boundary is applied to, so what it pins is the signedness of
  // `load_widened`'s typed loads: read unsigned, -6 becomes 4294967290, the row is dropped as
  // worse than the boundary, and both result assertions below fail. Both boundaries in play here
  // are positive (33, then batch 2's own 7), so this says nothing about the marshaller's widening,
  // and publication is not on this test's path at all.
  std::vector<std::int32_t> const batch1_values{5, 60, 9, 33, 2, 71, 40, 8};
  std::vector<std::int32_t> const batch2_values{1, 90, 3, 55, 4, 100, 7, -6};

  auto const narrowed = [](std::vector<std::int32_t> const& wide, auto tag) {
    std::vector<decltype(tag)> out;
    out.reserve(wide.size());
    for (auto const v : wide) {
      out.push_back(static_cast<decltype(tag)>(v));
    }
    return out;
  };
  auto const make_key_batch = [&](std::vector<std::int32_t> const& values) {
    switch (type_id) {
      case cudf::type_id::INT8:
        return make_numeric_batch<std::int8_t>(
          *space, narrowed(values, std::int8_t{}), cudf::type_id::INT8);
      case cudf::type_id::INT16:
        return make_numeric_batch<std::int16_t>(
          *space, narrowed(values, std::int16_t{}), cudf::type_id::INT16);
      case cudf::type_id::INT32:
        return make_numeric_batch<std::int32_t>(*space, values, cudf::type_id::INT32);
      case cudf::type_id::DECIMAL32: return make_decimal32_batch(*space, values, key_scale);
      case cudf::type_id::DECIMAL64:
        return make_decimal64_batch(
          *space, std::vector<std::int64_t>(values.begin(), values.end()), key_scale);
      default:
        return make_timestamp_batch<std::int32_t>(*space, values, cudf::type_id::TIMESTAMP_DAYS);
    }
  };
  auto const read_key_column = [&](cudf::column_view const& col) -> std::vector<std::int64_t> {
    auto const widened = [](auto host) {
      return std::vector<std::int64_t>(host.begin(), host.end());
    };
    switch (type_id) {
      case cudf::type_id::INT8: return widened(copy_column_to_host<std::int8_t>(col));
      case cudf::type_id::INT16: return widened(copy_column_to_host<std::int16_t>(col));
      // A fixed-point column's storage is its scaled integer, so reading the representation is
      // reading the value; DECIMAL64 is eight bytes wide, unlike every other case here.
      case cudf::type_id::DECIMAL64: return copy_column_to_host<std::int64_t>(col);
      default: return widened(copy_column_to_host<std::int32_t>(col));
    }
  };

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_key_batch(batch1_values));
  batches.push_back(make_key_batch(batch2_values));

  auto const make_orders = [&] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(
      order_on(0, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST, key_type));
    return orders;
  };
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(key_type);
  std::size_t const limit  = 4;
  std::size_t const offset = 1;

  sirius_physical_top_n plain(
    sirius::from_duckdb_vec(types), make_orders(), limit, offset, nullptr, 0);
  sirius_physical_top_n_merge plain_merge(&plain);

  dynamic_filter_stats stats;
  sirius_physical_top_n armed(
    sirius::from_duckdb_vec(types), make_orders(), limit, offset, nullptr, 0);
  armed.threshold_coordinator =
    make_test_coordinator(armed.orders, {storage_type}, limit + offset, true, &stats);
  sirius_physical_top_n_merge armed_merge(&armed);

  auto const run = [&](sirius_physical_top_n& local, sirius_physical_top_n_merge& merge) {
    auto const local_outputs = run_local(local, batches);
    auto out = merge.execute(pipelineable_operator_data(local_outputs), cudf::get_default_stream());
    auto const& merged = dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches();
    REQUIRE(merged.size() == 1);
    return read_key_column(sirius::get_cudf_table_view(*merged[0]).column(0));
  };

  auto const expected = run(plain, plain_merge);
  auto const actual   = run(armed, armed_merge);
  REQUIRE(actual == expected);
  // -6 is the best row overall and the offset drops it, so its presence is what makes the rest of
  // the window shift down by one from {2,3,4,5}: the negative provably ordered correctly.
  REQUIRE(actual == std::vector<std::int64_t>{1, 2, 3, 4});

  // Batch 2 was measured against boundary 33: the typed literal provably evaluated. It offers as
  // well as batch 1 -- its five survivors are exactly K, so its local result is itself a witness.
  REQUIRE(stats.top_n_offers.load() == 2);
  REQUIRE(stats.top_n_prefilter_rows_in.load() == batch2_values.size());
  REQUIRE(stats.top_n_prefilter_rows_out.load() == 5);
}

TEST_CASE("top_n extraction refuses a column whose type does not match the key's",
          "[physical_top_n][dynamic_filter][top_n]")
{
  // The boundary's representation is labelled with the key's declared storage type everywhere
  // downstream -- the published fixed-point literal, the kernel's component width, and the sink
  // prefilter. Reading a rep out of a column of a *different* type would attach the wrong label.
  // Before fixed-point keys that needed differing ids, where the extraction's static_cast is
  // already undefined; decimals add a scale axis on which the two can disagree while the ids
  // match, which is well-defined and silently wrong. Extraction must produce nothing instead.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // The column is DECIMAL64 at scale -4; the coordinator's key declares scale -2.
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_decimal64_batch(*space, {5, 60, 9, 33, 2, 71, 40, 8}, -4));
  batches.push_back(make_decimal64_batch(*space, {1, 90, 3, 55, 4, 100, 7}, -4));

  auto const make_orders = [] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(order_on(0,
                              duckdb::OrderType::ASCENDING,
                              duckdb::OrderByNullType::NULLS_LAST,
                              duckdb::LogicalType::DECIMAL(18, 2)));
    return orders;
  };
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType::DECIMAL(18, 2));
  std::size_t const limit  = 4;
  std::size_t const offset = 1;

  sirius_physical_top_n plain(
    sirius::from_duckdb_vec(types), make_orders(), limit, offset, nullptr, 0);
  sirius_physical_top_n_merge plain_merge(&plain);

  dynamic_filter_stats stats;
  sirius_physical_top_n armed(
    sirius::from_duckdb_vec(types), make_orders(), limit, offset, nullptr, 0);
  armed.threshold_coordinator = make_test_coordinator(
    armed.orders, {cudf::data_type{cudf::type_id::DECIMAL64, -2}}, limit + offset, true, &stats);
  sirius_physical_top_n_merge armed_merge(&armed);

  auto const run = [&](sirius_physical_top_n& local, sirius_physical_top_n_merge& merge) {
    auto const local_outputs = run_local(local, batches);
    auto out = merge.execute(pipelineable_operator_data(local_outputs), cudf::get_default_stream());
    auto const& merged = dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches();
    REQUIRE(merged.size() == 1);
    return copy_column_to_host<std::int64_t>(sirius::get_cudf_table_view(*merged[0]).column(0));
  };

  // Results are unaffected either way -- a refused component only loosens the boundary.
  REQUIRE(run(armed, armed_merge) == run(plain, plain_merge));
  // Nothing was witnessed: the boundary's only component was refused, so the offer carried a null
  // head and was rejected as unpublishable rather than acted on at the wrong scale.
  REQUIRE(stats.top_n_offers_unsupported.load() == stats.top_n_offers.load());
  REQUIRE(stats.top_n_prefilter_rows_in.load() == 0);
}

TEST_CASE("top_n prefilter beyond the boundary component limit keeps prefix-tied rows",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // Nine ORDER BY keys exceed boundary_filter_params::k_max_components (8), so the marshalled
  // prefilter must degrade to the inclusive 8-prefix form: a row tying the boundary on the first
  // 8 keys but better on key 9 is strictly better under the true predicate and must survive. A
  // strict 8-prefix would drop it and change the merged result.
  constexpr std::size_t k_num_keys = 9;
  auto const make_batch            = [&](std::vector<std::vector<std::int64_t>> const& rows) {
    std::vector<std::shared_ptr<data_batch>> key_batches;
    for (std::size_t key = 0; key < k_num_keys; ++key) {
      std::vector<std::int64_t> column;
      column.reserve(rows.size());
      for (auto const& row : rows) {
        column.push_back(row[key]);
      }
      key_batches.push_back(make_numeric_batch<std::int64_t>(*space, column, cudf::type_id::INT64));
    }
    return concatenate_batches_horizontal(key_batches, *space);
  };

  std::vector<std::shared_ptr<data_batch>> batches;
  // Batch 1: exactly K rows, so the offered boundary is (1 x8, 9).
  batches.push_back(make_batch(
    {{1, 1, 1, 1, 1, 1, 1, 1, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 7}, {1, 1, 1, 1, 1, 1, 1, 1, 9}}));
  // Batch 2: one 8-prefix tie that wins on key 9, one row worse on key 0, one better on key 0.
  batches.push_back(make_batch(
    {{1, 1, 1, 1, 1, 1, 1, 1, 2}, {2, 1, 1, 1, 1, 1, 1, 1, 0}, {0, 9, 9, 9, 9, 9, 9, 9, 9}}));

  auto const make_orders = [&] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    for (std::size_t key = 0; key < k_num_keys; ++key) {
      orders.push_back(
        order_on(key, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST));
    }
    return orders;
  };
  std::size_t const limit = 3;

  sirius_physical_top_n plain(
    sirius::from_duckdb_vec(bigint_types(k_num_keys)), make_orders(), limit, 0, nullptr, 0);
  sirius_physical_top_n_merge plain_merge(&plain);

  dynamic_filter_stats stats;
  sirius_physical_top_n armed(
    sirius::from_duckdb_vec(bigint_types(k_num_keys)), make_orders(), limit, 0, nullptr, 0);
  armed.threshold_coordinator = make_test_coordinator(
    armed.orders, std::vector<cudf::data_type>(k_num_keys, k_int64), limit, true, &stats);
  sirius_physical_top_n_merge armed_merge(&armed);

  auto const expected = run_pipeline(plain, plain_merge, batches);
  auto const actual   = run_pipeline(armed, armed_merge, batches);
  REQUIRE(actual == expected);

  // Batch 2 was measured: the prefix tie (kept) and the key-0 winner survive, the key-0 loser
  // does not -- a strict 8-prefix clamp would keep only one row and fail both asserts.
  REQUIRE(stats.top_n_prefilter_rows_in.load() == 3);
  REQUIRE(stats.top_n_prefilter_rows_out.load() == 2);
}

TEST_CASE("top_n prefilter multi-key equivalence with null tails",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  int const dir_case  = GENERATE(0, 1);
  int const null_case = GENERATE(0, 1);
  auto const tail_dir =
    dir_case == 0 ? duckdb::OrderType::ASCENDING : duckdb::OrderType::DESCENDING;
  auto const tail_nulls =
    null_case == 0 ? duckdb::OrderByNullType::NULLS_FIRST : duckdb::OrderByNullType::NULLS_LAST;
  CAPTURE(dir_case, null_case);

  // Two key columns; duplicate heads make the tail decisive, and the tail carries nulls.
  auto const make_batch = [&](std::vector<std::int64_t> const& k0,
                              std::vector<std::int64_t> const& k1,
                              std::vector<bool> const& k1_valid) {
    auto head = make_numeric_batch<std::int64_t>(*space, k0, cudf::type_id::INT64);
    auto tail =
      make_numeric_batch_with_nulls<std::int64_t>(*space, k1, k1_valid, cudf::type_id::INT64);
    return concatenate_batches_horizontal({head, tail}, *space);
  };
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_batch({3, 3, 3, 5, 5, 9, 1, 1},
                               {4, 0, 7, 2, 0, 1, 6, 0},
                               {true, false, true, true, false, true, true, false}));
  batches.push_back(
    make_batch({2, 3, 3, 7, 1, 1}, {5, 1, 0, 3, 2, 9}, {true, true, false, true, true, true}));

  auto const make_orders = [&] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(
      order_on(0, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST));
    orders.push_back(order_on(1, tail_dir, tail_nulls));
    return orders;
  };
  std::size_t const limit = 5;

  sirius_physical_top_n plain(
    sirius::from_duckdb_vec(bigint_types(2)), make_orders(), limit, 0, nullptr, 0);
  sirius_physical_top_n_merge plain_merge(&plain);

  sirius_physical_top_n armed(
    sirius::from_duckdb_vec(bigint_types(2)), make_orders(), limit, 0, nullptr, 0);
  armed.threshold_coordinator =
    make_test_coordinator(armed.orders, {k_int64, k_int64}, limit, true);
  sirius_physical_top_n_merge armed_merge(&armed);

  auto const expected = run_pipeline(plain, plain_merge, batches);
  auto const actual   = run_pipeline(armed, armed_merge, batches);
  REQUIRE(actual == expected);
}

TEST_CASE("top_n witness offers row K-1's tuple only at exact K",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto const make_orders = [&] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(
      order_on(0, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST));
    return orders;
  };
  std::size_t const limit  = 3;
  std::size_t const offset = 2;  // K = 5: the boundary is the 5th row's key

  dynamic_filter_stats stats;
  sirius_physical_top_n op(
    sirius::from_duckdb_vec(bigint_types(1)), make_orders(), limit, offset, nullptr, 0);
  op.threshold_coordinator =
    make_test_coordinator(op.orders, {k_int64}, limit + offset, true, &stats);

  SECTION("a sub-K local result publishes nothing")
  {
    auto batch = make_numeric_batch<std::int64_t>(*space, {30, 10, 20}, cudf::type_id::INT64);
    (void)op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
    REQUIRE(stats.top_n_offers.load() == 0);
    REQUIRE_FALSE(op.threshold_coordinator->tightest_boundary().has_value());
  }
  SECTION("an exact-K result offers the Kth row's key")
  {
    auto batch = make_numeric_batch<std::int64_t>(
      *space, {70, 10, 50, 30, 90, 20, 80, 40, 60, 100}, cudf::type_id::INT64);
    (void)op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
    REQUIRE(stats.top_n_offers.load() == 1);
    auto const boundary = op.threshold_coordinator->tightest_boundary();
    REQUIRE(boundary.has_value());
    REQUIRE(std::get<std::int64_t>(boundary->component(0)->value()) == 50);
  }
}

TEST_CASE("top_n prefilter keep-ratio disable stops measuring until re-armed",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto const make_orders = [&] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(
      order_on(0, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST));
    return orders;
  };
  std::size_t const limit = 3;

  dynamic_filter_stats stats;
  // keep_threshold 0.0: any measured batch that keeps a row disables the prefilter.
  sirius_physical_top_n op(sirius::from_duckdb_vec(bigint_types(1)),
                           make_orders(),
                           limit,
                           0,
                           nullptr,
                           0,
                           /*gate_keep_threshold=*/0.0);
  op.threshold_coordinator = make_test_coordinator(op.orders, {k_int64}, limit, true, &stats);

  auto const execute = [&](std::vector<std::int64_t> const& values) {
    auto batch = make_numeric_batch<std::int64_t>(*space, values, cudf::type_id::INT64);
    (void)op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  };

  // Batch 1: no boundary yet, so nothing is measured; its K-row output arms the boundary (3).
  execute({1, 2, 3, 4, 5});
  REQUIRE(stats.top_n_prefilter_rows_in.load() == 0);
  REQUIRE(stats.top_n_offers.load() == 1);

  // Batch 2: measured; one of three rows survives (2 < 3), so the ratio disables the gate. The
  // sub-K survivor set offers nothing, leaving the boundary count unchanged.
  execute({2, 300, 400});
  REQUIRE(stats.top_n_prefilter_rows_in.load() == 3);
  REQUIRE(stats.top_n_prefilter_rows_out.load() == 1);
  REQUIRE(stats.top_n_prefilter_disabled.load() == 1);

  // Batch 3: disabled with an unchanged boundary count -- no further measurement.
  execute({500, 600});
  REQUIRE(stats.top_n_prefilter_rows_in.load() == 3);
  REQUIRE(stats.top_n_prefilter_disabled.load() == 1);
}

TEST_CASE("top_n refuses an out-of-range ORDER BY column index on the prefilter path",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // A bound reference past the input's column count models a planner ordinal-remapping
  // regression. key_column_indices consumes the indices before compute_top_n_table's own checks
  // (the prefilter runs first), so the refusal must fire there rather than let the fused kernel
  // read a nonexistent column.
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(order_on(5, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST));
  std::size_t const limit = 3;

  dynamic_filter_stats stats;
  sirius_physical_top_n op(
    sirius::from_duckdb_vec(bigint_types(1)), std::move(orders), limit, 0, nullptr, 0);
  op.threshold_coordinator = make_test_coordinator(op.orders, {k_int64}, limit, true, &stats);

  auto batch = make_numeric_batch<std::int64_t>(*space, {1, 2, 3, 4}, cudf::type_id::INT64);
  REQUIRE_THROWS_AS(op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream()),
                    sirius::internal_exception);
}

TEST_CASE("top_n degraded first-key prefilter with an unsupported tail type",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // INT64 head, VARCHAR tail: the tail type is outside the allowlist, so the coordinator runs
  // first-key-only (lex_admitted = false) and the prefilter degrades to the inclusive head
  // comparison, keeping head ties for the tail to order.
  auto const make_batch = [&](std::vector<std::int64_t> const& k0,
                              std::vector<std::string> const& k1) {
    return make_two_column_batch<std::int64_t, std::string>(*space, k0, k1, cudf::type_id::STRING);
  };
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_batch({5, 3, 3, 9, 7, 3}, {"e", "c", "a", "x", "q", "b"}));
  batches.push_back(make_batch({3, 2, 8, 3, 1}, {"d", "m", "y", "aa", "z"}));

  auto const make_orders = [&] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(
      order_on(0, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST));
    orders.push_back(order_on(1,
                              duckdb::OrderType::ASCENDING,
                              duckdb::OrderByNullType::NULLS_LAST,
                              duckdb::LogicalType::VARCHAR));
    return orders;
  };
  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType::BIGINT);
  types.push_back(duckdb::LogicalType::VARCHAR);
  std::size_t const limit = 4;

  sirius_physical_top_n plain(sirius::from_duckdb_vec(types), make_orders(), limit, 0, nullptr, 0);
  sirius_physical_top_n_merge plain_merge(&plain);

  dynamic_filter_stats stats;
  sirius_physical_top_n armed(sirius::from_duckdb_vec(types), make_orders(), limit, 0, nullptr, 0);
  armed.threshold_coordinator =
    make_test_coordinator(armed.orders,
                          {k_int64, cudf::data_type{cudf::type_id::EMPTY}},
                          limit,
                          /*lex_admitted=*/false,
                          &stats);
  sirius_physical_top_n_merge armed_merge(&armed);

  auto const run_strings = [&](sirius_physical_top_n& local, sirius_physical_top_n_merge& merge) {
    auto const local_outputs = run_local(local, batches);
    auto out = merge.execute(pipelineable_operator_data(local_outputs), cudf::get_default_stream());
    auto const& merged = dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches();
    REQUIRE(merged.size() == 1);
    auto const view = sirius::get_cudf_table_view(*merged[0]);
    return std::pair{copy_column_to_host<std::int64_t>(view.column(0)),
                     copy_column_to_host<std::string>(view.column(1))};
  };

  auto const expected = run_strings(plain, plain_merge);
  auto const actual   = run_strings(armed, armed_merge);
  REQUIRE(actual == expected);

  // The witness engaged only the head; the unsupported tail is recorded disengaged.
  auto const boundary = armed.threshold_coordinator->tightest_boundary();
  REQUIRE(boundary.has_value());
  REQUIRE(boundary->component(0).has_value());
  REQUIRE_FALSE(boundary->component(1).has_value());
  REQUIRE(stats.top_n_offers.load() >= 1);
}

TEST_CASE("top_n merge finalize drains the shared coordinator",
          "[physical_top_n][dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(order_on(0, duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_LAST));
  sirius_physical_top_n local(
    sirius::from_duckdb_vec(bigint_types(1)), std::move(orders), 2, 0, nullptr, 0);
  local.threshold_coordinator = make_test_coordinator(local.orders, {k_int64}, 2, true);

  sirius_physical_top_n_merge merge(&local);
  REQUIRE(merge.threshold_coordinator == local.threshold_coordinator);

  auto batch = make_numeric_batch<std::int64_t>(*space, {4, 1, 3, 2}, cudf::type_id::INT64);
  (void)local.execute(pipelineable_operator_data({batch}), cudf::get_default_stream());
  REQUIRE(local.threshold_coordinator->tightest_boundary().has_value());

  // The pipeline invokes finalize_operator when the merge's pipeline completes; afterwards the
  // coordinator rejects further offers.
  merge.finalize_operator();
  auto rejected = local.threshold_coordinator->offer(
    {exact_host_key_tuple{{exact_host_scalar{std::int64_t{0}, k_int64}}}, {}});
  REQUIRE(rejected == threshold_offer_result::REJECTED_STATE);
}
