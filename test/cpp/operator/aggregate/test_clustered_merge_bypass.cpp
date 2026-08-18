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

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "op/aggregate/clustered_merge_bypass.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <catch.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
using I64Traits = gpu_type_traits<int64_t>;
using sirius::test::vector_to_cudf_column;

/// Build a (key, value) INT64 partial-aggregate-shaped batch.
std::shared_ptr<data_batch> make_partial_batch(const std::vector<int64_t>& keys,
                                               const std::vector<int64_t>& values,
                                               cucascade::memory::memory_space& space)
{
  auto stream = default_stream();
  auto mr     = get_resource_ref(space);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<I64Traits>(keys, stream, mr));
  cols.push_back(vector_to_cudf_column<I64Traits>(values, stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(cols));
  return sirius::make_data_batch(
    std::move(table), space, stream, sirius::telemetry::batch_telemetry_info{});
}

/// The HAVING predicate: column @p column_index > threshold (BIGINT).
std::unique_ptr<sirius::ast::node> make_having_expression(int64_t threshold, idx_t column_index = 1)
{
  auto ref =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, column_index);
  auto cst = duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BIGINT(threshold));
  duckdb::BoundComparisonExpression cmp(
    duckdb::ExpressionType::COMPARE_GREATERTHAN, std::move(ref), std::move(cst));
  return sirius::ast::from_duckdb(cmp);
}

/// A fresh merge operator over the given groups/aggregations (all INT64).
std::unique_ptr<sirius_physical_grouped_aggregate_merge> make_merge(
  const std::vector<std::size_t>& group_indexes,
  const std::vector<std::string>& aggregations,
  const std::vector<std::size_t>& agg_indexes)
{
  auto agg =
    sirius::test::create_aggregate_expressions<I64Traits>(group_indexes, aggregations, agg_indexes);
  return std::make_unique<sirius_physical_grouped_aggregate_merge>(
    std::move(agg.output_types), std::move(agg.aggregates), std::move(agg.groups), 100);
}

/// A fresh SUM(col1) GROUP BY col0 merge operator.
std::unique_ptr<sirius_physical_grouped_aggregate_merge> make_sum_merge()
{
  return make_merge({0}, {"sum"}, {1});
}

/// Concatenate every output batch of @p outputs into one owned table.
std::unique_ptr<cudf::table> concat_output_batches(const operator_data& outputs)
{
  const auto& batches =
    dynamic_cast<const pipelineable_operator_data&>(outputs).get_read_only_batches();
  REQUIRE(!batches.empty());
  std::vector<cudf::table_view> views;
  views.reserve(batches.size());
  for (const auto& ro : batches) {
    views.push_back(sirius::get_cudf_table_view(ro));
  }
  return cudf::concatenate(views, default_stream(), cudf::get_current_device_resource_ref());
}

/// Apply @p predicate to @p input (the downstream FILTER's job in the real pipeline).
std::unique_ptr<cudf::table> apply_predicate(const sirius::ast::node& predicate,
                                             cudf::table_view input)
{
  sirius::expression_evaluator evaluator(
    predicate, cudf::get_current_device_resource_ref(), default_stream());
  return evaluator.select(input);
}

/// Sorted table equivalence.
bool tables_equal_sorted(cudf::table_view lhs, cudf::table_view rhs)
{
  auto mr     = cudf::get_current_device_resource_ref();
  auto stream = default_stream();
  std::vector<cudf::order> orders(static_cast<std::size_t>(lhs.num_columns()),
                                  cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_orders(static_cast<std::size_t>(lhs.num_columns()),
                                            cudf::null_order::AFTER);
  auto sorted_lhs = cudf::sort(lhs, orders, null_orders, stream, mr);
  auto sorted_rhs = cudf::sort(rhs, orders, null_orders, stream, mr);
  return sirius::test::expect_tables_equivalent_impl(sorted_lhs->view(), sorted_rhs->view());
}

/// filter(bypass-or-merge output) must equal filter(reference merge output): the downstream
/// FILTER runs in both worlds, so this is exactly the pipeline-visible contract.
/// @p reference must be a bypass-unconfigured merge with the same aggregate definitions.
void require_filtered_equivalence(sirius_physical_grouped_aggregate_merge& merge_under_test,
                                  sirius_physical_grouped_aggregate_merge& reference,
                                  const sirius::ast::node& predicate,
                                  const std::vector<std::shared_ptr<data_batch>>& partials)
{
  auto outputs = merge_under_test.execute(pipelineable_operator_data(partials), default_stream());
  auto actual_table    = concat_output_batches(*outputs);
  auto actual_filtered = apply_predicate(predicate, actual_table->view());

  auto reference_outputs =
    reference.execute(pipelineable_operator_data(partials), default_stream());
  auto reference_table    = concat_output_batches(*reference_outputs);
  auto reference_filtered = apply_predicate(predicate, reference_table->view());

  REQUIRE(tables_equal_sorted(actual_filtered->view(), reference_filtered->view()));
}

void require_filtered_equivalence(sirius_physical_grouped_aggregate_merge& merge_under_test,
                                  const sirius::ast::node& predicate,
                                  const std::vector<std::shared_ptr<data_batch>>& partials)
{
  auto reference = make_sum_merge();  // bypass unconfigured: normal merge path
  require_filtered_equivalence(merge_under_test, *reference, predicate, partials);
}

}  // namespace

TEST_CASE("clustered merge bypass arms on boundary-overlapping partials and is exact",
          "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);

  // Three clustered partials; adjacent batches share exactly one boundary key. Boundary keys
  // 999 and 1999 only clear the HAVING threshold once their fragments are combined, so a naive
  // per-partial filter WITHOUT the boundary fix-up would drop them — this is the case that
  // makes the test meaningful.
  constexpr int64_t threshold = 100;
  auto keys_of                = [](int64_t lo, int64_t hi) {
    std::vector<int64_t> keys(static_cast<std::size_t>(hi - lo + 1));
    std::iota(keys.begin(), keys.end(), lo);
    return keys;
  };
  auto values_for = [](const std::vector<int64_t>& keys,
                       std::vector<std::pair<int64_t, int64_t>> specials) {
    std::vector<int64_t> values(keys.size(), 10);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      for (auto [key, value] : specials) {
        if (keys[i] == key) { values[i] = value; }
      }
    }
    return values;
  };

  auto keys0 = keys_of(0, 999);
  auto keys1 = keys_of(999, 1999);
  auto keys2 = keys_of(1999, 2999);
  std::vector<std::shared_ptr<data_batch>> partials{
    make_partial_batch(keys0, values_for(keys0, {{999, 60}}), *space),
    make_partial_batch(keys1, values_for(keys1, {{999, 50}, {1500, 200}, {1999, 70}}), *space),
    make_partial_batch(keys2, values_for(keys2, {{1999, 60}, {2500, 150}}), *space)};

  auto predicate = make_having_expression(threshold);
  auto merge     = make_sum_merge();
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());

  REQUIRE(merge->try_plan_clustered_bypass(partials));
  REQUIRE(merge->clustered_bypass_armed());

  require_filtered_equivalence(*merge, *predicate, partials);
}

TEST_CASE("clustered merge bypass with fully disjoint partials emits filtered partials directly",
          "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);

  std::vector<int64_t> keys0(1000), keys1(1000);
  std::iota(keys0.begin(), keys0.end(), 0);
  std::iota(keys1.begin(), keys1.end(), 5000);
  std::vector<int64_t> vals0(1000, 10);
  std::vector<int64_t> vals1(1000, 10);
  vals0[7]  = 500;  // passes the filter
  vals1[13] = 600;
  std::vector<std::shared_ptr<data_batch>> partials{make_partial_batch(keys0, vals0, *space),
                                                    make_partial_batch(keys1, vals1, *space)};

  auto predicate = make_having_expression(100);
  auto merge     = make_sum_merge();
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());

  REQUIRE(merge->try_plan_clustered_bypass(partials));
  require_filtered_equivalence(*merge, *predicate, partials);
}

TEST_CASE("clustered merge bypass refuses interleaved (overlapping-range) partials",
          "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);

  // Adversarial: the batches interleave over the same wide key range (even keys vs odd keys),
  // so nearly every "partial" row would be wrong to treat as final. The range proof must
  // refuse, and the operator must fall through to the exact merge path.
  std::vector<int64_t> even_keys, odd_keys;
  for (int64_t k = 0; k < 100000; k += 2) {
    even_keys.push_back(k);
    odd_keys.push_back(k + 1);
  }
  std::vector<int64_t> values(even_keys.size(), 10);
  std::vector<std::shared_ptr<data_batch>> partials{make_partial_batch(even_keys, values, *space),
                                                    make_partial_batch(odd_keys, values, *space)};

  auto predicate = make_having_expression(5);
  auto merge     = make_sum_merge();
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());

  REQUIRE_FALSE(merge->try_plan_clustered_bypass(partials));
  REQUIRE_FALSE(merge->clustered_bypass_armed());

  // Un-armed execute must be the untouched merge path.
  require_filtered_equivalence(*merge, *predicate, partials);
}

TEST_CASE("clustered merge bypass refuses non-adjacent range overlap", "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);

  // Sorted by min: [0, 1000000], [999999, 2000000], [1000000, 3000000]. Both ADJACENT overlaps
  // are tiny (would pass the width gate), but ranges 1 and 3 also intersect at key 1000000 —
  // a key could hide in a non-adjacent batch, so the containment proof must refuse.
  std::vector<std::shared_ptr<data_batch>> partials{
    make_partial_batch({0, 999999, 1000000}, {10, 10, 10}, *space),
    make_partial_batch({999999, 1000000, 2000000}, {10, 10, 10}, *space),
    make_partial_batch({1000000, 2000000, 3000000}, {10, 10, 10}, *space)};

  auto predicate = make_having_expression(5);
  auto merge     = make_sum_merge();
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());

  REQUIRE_FALSE(merge->try_plan_clustered_bypass(partials));
}

TEST_CASE("clustered merge bypass refuses unsupported key types and stays off when unconfigured",
          "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);

  std::vector<std::shared_ptr<data_batch>> partials{
    make_partial_batch({0, 1, 2}, {10, 10, 10}, *space),
    make_partial_batch({100, 101, 102}, {10, 10, 10}, *space)};

  auto predicate = make_having_expression(5);

  SECTION("no downstream filter stamped")
  {
    auto merge = make_sum_merge();
    merge->set_clustered_bypass_params(true, 0.05);
    REQUIRE_FALSE(merge->clustered_bypass_wanted());
    REQUIRE_FALSE(merge->try_plan_clustered_bypass(partials));
  }

  SECTION("knob disabled")
  {
    auto merge = make_sum_merge();
    merge->set_clustered_bypass_params(false, 0.05);
    merge->set_clustered_bypass_filter(predicate.get());
    REQUIRE_FALSE(merge->clustered_bypass_wanted());
    REQUIRE_FALSE(merge->try_plan_clustered_bypass(partials));
  }

  SECTION("unsupported (floating-point) leading key type")
  {
    using F64Traits        = gpu_type_traits<double>;
    auto stream            = default_stream();
    auto mr                = get_resource_ref(*space);
    auto make_double_batch = [&](std::vector<double> keys, std::vector<int64_t> values) {
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(vector_to_cudf_column<F64Traits>(keys, stream, mr));
      cols.push_back(vector_to_cudf_column<I64Traits>(values, stream, mr));
      auto table = std::make_unique<cudf::table>(std::move(cols));
      return sirius::make_data_batch(
        std::move(table), *space, stream, sirius::telemetry::batch_telemetry_info{});
    };
    std::vector<std::shared_ptr<data_batch>> double_partials{
      make_double_batch({0.0, 1.0}, {10, 10}), make_double_batch({100.0, 101.0}, {10, 10})};
    auto merge = make_sum_merge();
    merge->set_clustered_bypass_params(true, 0.05);
    merge->set_clustered_bypass_filter(predicate.get());
    REQUIRE_FALSE(merge->try_plan_clustered_bypass(double_partials));
  }
}

TEST_CASE("clustered merge bypass refuses NULL group keys", "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  // A NULL group key can occur in ANY batch regardless of key ranges (min/max skip nulls), so
  // the containment proof cannot see it — the gate must refuse.
  auto keys = vector_to_cudf_column<I64Traits>({1, 2, 3, 4}, stream, mr);
  auto mask = cudf::create_null_mask(4, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, 1, false, stream);
  keys->set_null_mask(std::move(mask), 1);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(keys));
  cols.push_back(vector_to_cudf_column<I64Traits>({10, 10, 10, 10}, stream, mr));
  auto null_key_batch = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                                *space,
                                                stream,
                                                sirius::telemetry::batch_telemetry_info{});

  std::vector<std::shared_ptr<data_batch>> partials{
    std::move(null_key_batch), make_partial_batch({100, 101, 102}, {10, 10, 10}, *space)};

  auto predicate = make_having_expression(5);
  auto merge     = make_sum_merge();
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());
  REQUIRE_FALSE(merge->try_plan_clustered_bypass(partials));
  REQUIRE_FALSE(merge->clustered_bypass_armed());
}

TEST_CASE("clustered merge bypass unknown-batch-id fallback re-groups exactly",
          "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);

  // Arm the plan over {A, B}, then execute with an extra batch C the range proof never saw.
  // The defensive fallback must re-group EVERY input row through the exact merge combine.
  auto batch_a = make_partial_batch({0, 1, 2, 3}, {10, 200, 10, 10}, *space);
  auto batch_b = make_partial_batch({5000, 5001}, {10, 300}, *space);
  auto batch_c = make_partial_batch({8000, 8001}, {400, 10}, *space);

  auto predicate = make_having_expression(100);
  auto merge     = make_sum_merge();
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());

  REQUIRE(merge->try_plan_clustered_bypass({batch_a, batch_b}));
  REQUIRE(merge->clustered_bypass_armed());

  std::vector<std::shared_ptr<data_batch>> all_partials{batch_a, batch_b, batch_c};
  require_filtered_equivalence(*merge, *predicate, all_partials);
}

TEST_CASE("clustered merge bypass handles multi-key group-bys via the leading key",
          "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  // GROUP BY (k1, k2), SUM(col2). Leading-key ranges share boundary k1=49; the composite key
  // (49, 0) only clears the threshold once its fragments combine (60 + 50), while (49, 1)
  // stays below it in both fragments — the fix-up must combine per COMPOSITE key.
  auto make_partial3 = [&](const std::vector<int64_t>& k1,
                           const std::vector<int64_t>& k2,
                           const std::vector<int64_t>& vals) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<I64Traits>(k1, stream, mr));
    cols.push_back(vector_to_cudf_column<I64Traits>(k2, stream, mr));
    cols.push_back(vector_to_cudf_column<I64Traits>(vals, stream, mr));
    return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                   *space,
                                   stream,
                                   sirius::telemetry::batch_telemetry_info{});
  };
  std::vector<int64_t> k1a, k2a, va, k1b, k2b, vb;
  for (int64_t k = 0; k <= 49; ++k) {
    for (int64_t sub = 0; sub < 2; ++sub) {
      k1a.push_back(k);
      k2a.push_back(sub);
      va.push_back(k == 49 && sub == 0 ? 60 : 10);
    }
  }
  for (int64_t k = 49; k <= 99; ++k) {
    for (int64_t sub = 0; sub < 2; ++sub) {
      k1b.push_back(k);
      k2b.push_back(sub);
      vb.push_back(k == 49 && sub == 0 ? 50 : (k == 75 && sub == 1 ? 200 : 10));
    }
  }
  std::vector<std::shared_ptr<data_batch>> partials{make_partial3(k1a, k2a, va),
                                                    make_partial3(k1b, k2b, vb)};

  auto predicate = make_having_expression(100, /*column_index=*/2);
  auto merge     = make_merge({0, 1}, {"sum"}, {2});
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());

  REQUIRE(merge->try_plan_clustered_bypass(partials));
  auto reference = make_merge({0, 1}, {"sum"}, {2});
  require_filtered_equivalence(*merge, *reference, *predicate, partials);
}

TEST_CASE("clustered merge bypass combines COUNT kinds through the partitioned regroup",
          "[clustered_merge_bypass]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto mr     = get_resource_ref(*space);

  // Partials shaped [key, sum, count]; COUNT partials merge via SUM. Many survivors + a tiny
  // hash_partition_bytes force the fix-up regroup through its hash-partitioned branch.
  auto make_partial3 = [&](const std::vector<int64_t>& keys,
                           const std::vector<int64_t>& sums,
                           const std::vector<int64_t>& counts) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(vector_to_cudf_column<I64Traits>(keys, stream, mr));
    cols.push_back(vector_to_cudf_column<I64Traits>(sums, stream, mr));
    cols.push_back(vector_to_cudf_column<I64Traits>(counts, stream, mr));
    return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                   *space,
                                   stream,
                                   sirius::telemetry::batch_telemetry_info{});
  };
  std::vector<int64_t> keys0, sums0, cnts0, keys1, sums1, cnts1;
  for (int64_t k = 0; k <= 199; ++k) {
    keys0.push_back(k);
    sums0.push_back(k == 199 ? 60 : (k % 2 == 0 ? 200 : 10));  // ~100 survivors
    cnts0.push_back(k == 199 ? 3 : 1);
  }
  for (int64_t k = 199; k <= 399; ++k) {
    keys1.push_back(k);
    sums1.push_back(k == 199 ? 50 : (k % 2 == 1 ? 200 : 10));
    cnts1.push_back(k == 199 ? 4 : 1);  // combined boundary count must be 7
  }
  std::vector<std::shared_ptr<data_batch>> partials{make_partial3(keys0, sums0, cnts0),
                                                    make_partial3(keys1, sums1, cnts1)};

  auto make_sum_count_merge = [&](uint64_t hash_partition_bytes) {
    auto agg = sirius::test::create_aggregate_expressions<I64Traits>({0}, {"sum", "count"}, {1, 1});
    auto gagg = std::make_unique<sirius_physical_grouped_aggregate>(
      std::move(agg.output_types), std::move(agg.aggregates), std::move(agg.groups), 100);
    auto merge =
      std::make_unique<sirius_physical_grouped_aggregate_merge>(gagg.get(), hash_partition_bytes);
    return std::pair{std::move(gagg), std::move(merge)};
  };

  auto predicate                = make_having_expression(100);
  auto [gagg_under_test, merge] = make_sum_count_merge(/*hash_partition_bytes=*/256);
  merge->set_clustered_bypass_params(true, 0.05);
  merge->set_clustered_bypass_filter(predicate.get());
  REQUIRE(merge->try_plan_clustered_bypass(partials));

  auto [gagg_reference, reference] =
    make_sum_count_merge(sirius::config::DEFAULT_HASH_PARTITION_BYTES);
  require_filtered_equivalence(*merge, *reference, *predicate, partials);
}
