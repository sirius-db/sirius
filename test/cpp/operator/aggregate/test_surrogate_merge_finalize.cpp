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

/**
 * @file test_surrogate_merge_finalize.cpp
 * @brief Functional tests for the MERGE_GROUP_BY surrogate-key finalization paths through the
 *        production clone-from-aggregate constructor: the distinct-proof fast path (with the
 *        knob-off variant that must re-group to identical output), the conservative full-tuple
 *        re-group over duplicate tuples, the floating-point NaN gate, multi-source base-offset
 *        addressing, and the store release hook.
 */

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "op/groupby_surrogate_deferral.hpp"
#include "op/groupby_surrogate_store.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/table/table.hpp>

#include <catch.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
using sirius::test::vector_to_cudf_column;

namespace {

sirius::logical_type bigint_type() { return sirius::logical_type::make(sirius::type_id::BIGINT); }
sirius::logical_type varchar_type() { return sirius::logical_type::make(sirius::type_id::VARCHAR); }
sirius::logical_type double_type() { return sirius::logical_type::make(sirius::type_id::DOUBLE); }

/// Shared setup and tails for every finalize case: memory manager / GPU space / stream, a
/// deferral store, string-source registration, a merge built through the production
/// clone-from-aggregate constructor, and the execute + expected-table comparison tails.
struct surrogate_finalize_fixture {
  using batches = std::vector<std::shared_ptr<cucascade::data_batch>>;

  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager =
    sirius::test::operator_utils::initialize_memory_manager();
  cucascade::memory::memory_space* space =
    memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream_view stream                    = default_stream();
  std::shared_ptr<surrogate_deferral_store> store = std::make_shared<surrogate_deferral_store>();

  /// One committed string source batch on the deferral join's left side; returns its base.
  int64_t add_string_source(std::vector<std::string> const& values)
  {
    auto col =
      vector_to_cudf_column<gpu_type_traits<string_tag>>(values, stream, get_resource_ref(*space));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));
    auto batch = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                         *space,
                                         stream,
                                         sirius::telemetry::batch_telemetry_info{});
    auto res   = store->reserve(
      join_side::left, batch->get_batch_id(), static_cast<cudf::size_type>(values.size()));
    auto const base = res.base();
    std::move(res).commit(batch->to_read_only());
    return base;
  }

  /// A merge in the shape produced by the rewrite for keys [key, rowid] + SUM(BIGINT): a REAL
  /// grouped aggregate over the rewritten carriers gets the restore plan installed, and the
  /// production clone-from-aggregate constructor wires the merge exactly as the generator does
  /// (declaring the original restored schema).
  sirius_physical_grouped_aggregate_merge& make_merge(sirius::logical_type key_type,
                                                      bool allow_unique_fastpath = true)
  {
    duckdb::vector<sirius::logical_type> carrier_types{key_type, bigint_type(), bigint_type()};
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups;
    {
      auto ref0 =
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(sirius::to_duckdb(key_type), 0ULL);
      auto ref1 =
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 1ULL);
      groups.push_back(sirius::ast::from_duckdb(*ref0));
      groups.push_back(sirius::ast::from_duckdb(*ref1));
    }
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_children;
    agg_children.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 2ULL));
    auto agg_function = sirius::test::MakeDummyAggregate(
      "sum", {duckdb::LogicalType::BIGINT}, duckdb::LogicalType::BIGINT);
    auto agg_expr = duckdb::make_uniq<duckdb::BoundAggregateExpression>(
      agg_function, std::move(agg_children), nullptr, nullptr, duckdb::AggregateType::NON_DISTINCT);
    duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
    aggregates.push_back(sirius::ast::from_duckdb(*agg_expr));

    _aggregate = duckdb::make_uniq<sirius_physical_grouped_aggregate>(
      std::move(carrier_types), std::move(aggregates), std::move(groups), /*estimated=*/10);
    _aggregate->install_surrogate_restore(make_restore_plan(key_type, allow_unique_fastpath));
    _merge = std::make_unique<sirius_physical_grouped_aggregate_merge>(_aggregate.get());
    return *_merge;
  }

  /// A merged-shape input batch [key, rowid BIGINT, sum BIGINT].
  template <typename KeyTraits>
  std::shared_ptr<cucascade::data_batch> make_merged_batch(
    std::vector<typename KeyTraits::type> const& keys,
    std::vector<int64_t> const& rowids,
    std::vector<int64_t> const& sums)
  {
    auto mr = get_resource_ref(*space);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<KeyTraits>(keys, stream, mr));
    cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(rowids, stream, mr));
    cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(sums, stream, mr));
    return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                   *space,
                                   stream,
                                   sirius::telemetry::batch_telemetry_info{});
  }

  /// Execute the fixture's merge over the inputs and return its output batches.
  batches run(batches inputs)
  {
    auto outputs = _merge->execute(pipelineable_operator_data(std::move(inputs)), stream);
    return dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches();
  }

  /// Compare the single output batch against the expected [BIGINT, VARCHAR, BIGINT] rows.
  void expect(batches const& outputs,
              std::vector<int64_t> const& keys,
              std::vector<std::string> const& strings,
              std::vector<int64_t> const& sums)
  {
    auto mr = get_resource_ref(*space);
    std::vector<std::unique_ptr<cudf::column>> expected_cols;
    expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(keys, stream, mr));
    expected_cols.push_back(
      vector_to_cudf_column<gpu_type_traits<string_tag>>(strings, stream, mr));
    expected_cols.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(sums, stream, mr));
    cudf::table expected(std::move(expected_cols));
    REQUIRE(outputs.size() == 1);
    REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(outputs[0],
                                                                expected.view(),
                                                                /*sort=*/true));
  }

 private:
  /// The restore plan the planner would build for this shape: one left-side group restoring
  /// key slot 1 (the rowid slot) from source column 0, real key slot 0.
  std::shared_ptr<surrogate_restore_plan const> make_restore_plan(sirius::logical_type key_type,
                                                                  bool allow_unique_fastpath)
  {
    std::vector<surrogate_restore_plan::restored_key> keys;
    keys.push_back(
      surrogate_restore_plan::restored_key{/*key_slot=*/1, /*source_col=*/0, varchar_type()});
    std::vector<surrogate_restore_plan::restore_group> groups;
    groups.emplace_back(join_side::left, /*rowid_key_slot=*/1, std::move(keys));
    duckdb::vector<sirius::logical_type> original_types{
      std::move(key_type), varchar_type(), bigint_type()};
    return std::make_shared<surrogate_restore_plan const>(store,
                                                          std::move(groups),
                                                          std::vector<int>{0},
                                                          std::move(original_types),
                                                          allow_unique_fastpath);
  }

  duckdb::unique_ptr<sirius_physical_grouped_aggregate> _aggregate;
  std::unique_ptr<sirius_physical_grouped_aggregate_merge> _merge;
};

}  // namespace

TEST_CASE("surrogate merge finalize fast path restores strings without a re-group",
          "[surrogate_groupby]")
{
  surrogate_finalize_fixture fx;
  auto const run_case = [&](bool allow_unique_fastpath) -> auto& {
    fx.add_string_source({"alpha", "beta", "gamma", "delta"});
    auto& merge = fx.make_merge(bigint_type(), allow_unique_fastpath);
    surrogate_finalize_fixture::batches inputs;
    inputs.push_back(fx.make_merged_batch<gpu_type_traits<int64_t>>({1, 2}, {0, 1}, {10, 20}));
    inputs.push_back(fx.make_merged_batch<gpu_type_traits<int64_t>>({1, 3}, {0, 2}, {5, 7}));
    auto const outputs = fx.run(std::move(inputs));
    fx.expect(outputs, {1, 2, 3}, {"alpha", "beta", "gamma"}, {15, 20, 7});
    return merge;
  };

  SECTION("unique tuples take the fast path")
  {
    auto& merge = run_case(/*allow_unique_fastpath=*/true);
    // The finalize hook releases the retained sources exactly once.
    merge.on_finalize_operator();
    auto const stats = fx.store->release();
    REQUIRE(stats.sources == 0);
  }
  SECTION("fast-path knob off: the conservative re-group produces identical output")
  {
    run_case(/*allow_unique_fastpath=*/false);
  }
}

TEST_CASE("surrogate merge finalize re-groups duplicate full tuples (conservative path)",
          "[surrogate_groupby]")
{
  surrogate_finalize_fixture fx;
  // Two DISTINCT source rows carrying an IDENTICAL string: the wrong-results class the
  // conservative path exists for.
  fx.add_string_source({"x", "x"});
  fx.make_merge(bigint_type());

  surrogate_finalize_fixture::batches inputs;
  inputs.push_back(fx.make_merged_batch<gpu_type_traits<int64_t>>({1}, {0}, {10}));
  inputs.push_back(fx.make_merged_batch<gpu_type_traits<int64_t>>({1}, {1}, {5}));
  auto const outputs = fx.run(std::move(inputs));
  fx.expect(outputs, {1}, {"x"}, {15});
}

TEST_CASE("surrogate merge finalize declines the fast path on floating-point real keys",
          "[surrogate_groupby]")
{
  surrogate_finalize_fixture fx;
  fx.add_string_source({"x", "x"});
  fx.make_merge(double_type());

  // Two NaN-keyed rows with distinct rowids but identical strings: SQL grouping semantics
  // (all NaNs are one group) require ONE output row, which only the conservative re-group can
  // produce — the distinct-count proof must not be consulted for floating-point keys.
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  surrogate_finalize_fixture::batches inputs;
  inputs.push_back(fx.make_merged_batch<gpu_type_traits<double>>({nan}, {0}, {10}));
  inputs.push_back(fx.make_merged_batch<gpu_type_traits<double>>({nan}, {1}, {5}));
  auto const outputs = fx.run(std::move(inputs));

  REQUIRE(outputs.size() == 1);
  auto const view = sirius::get_cudf_table_view(*outputs[0]);
  REQUIRE(view.num_rows() == 1);
  REQUIRE(view.num_columns() == 3);
}

TEST_CASE("surrogate merge finalize gathers across multiple source base ranges",
          "[surrogate_groupby]")
{
  surrogate_finalize_fixture fx;
  REQUIRE(fx.add_string_source({"a", "b"}) == 0);
  REQUIRE(fx.add_string_source({"c", "d"}) == 2);
  fx.make_merge(bigint_type());

  // Single-batch merge input (exercises the clone -> finalize path); rowids straddle the two
  // source ranges: 1 -> "b" (first source), 2 -> "c" (second source).
  surrogate_finalize_fixture::batches inputs;
  inputs.push_back(fx.make_merged_batch<gpu_type_traits<int64_t>>({1, 2}, {1, 2}, {10, 20}));
  auto const outputs = fx.run(std::move(inputs));
  fx.expect(outputs, {1, 2}, {"b", "c"}, {10, 20});
}
