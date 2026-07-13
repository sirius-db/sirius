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
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition.hpp"

#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

using namespace sirius::op;

namespace {

// Minimal concrete subclass for testing the base-class default.
class stub_operator : public sirius_physical_operator {
 public:
  stub_operator() : sirius_physical_operator(SiriusPhysicalOperatorType::FILTER, {}, 0) {}
  std::string get_name() const override { return "stub"; }
};

// Builds a minimal hash join (INNER) used as a parent_op for concat and partition.
struct hash_join_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;
};

hash_join_fixture make_hash_join()
{
  hash_join_fixture f;
  f.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  f.logical_join->types = {duckdb::LogicalType::INTEGER};

  auto left = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);
  auto right = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);

  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition cond;
  cond.left  = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.right = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(cond));

  f.hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
    *f.logical_join,
    std::move(left),
    std::move(right),
    sirius::wrap_join_conditions(std::move(conditions)),
    duckdb::JoinType::INNER,
    duckdb::vector<duckdb::idx_t>{},
    duckdb::vector<duckdb::idx_t>{},
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
    0,
    nullptr);
  return f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Base class — default 2× bytes
// ---------------------------------------------------------------------------

TEST_CASE("default no_history_peak_memory_estimate returns 2× bytes",
          "[no_history_peak_memory_estimate][base]")
{
  stub_operator op;
  REQUIRE(op.no_history_peak_memory_estimate({0, 0}) == 0);
  REQUIRE(op.no_history_peak_memory_estimate({1, 100}) == 200);
  REQUIRE(op.no_history_peak_memory_estimate({3, 1000}) == 2000);
}

// ---------------------------------------------------------------------------
// sirius_physical_concat
// ---------------------------------------------------------------------------

TEST_CASE("concat no_history_peak_memory_estimate: 0 batches returns 0",
          "[no_history_peak_memory_estimate][concat]")
{
  auto f = make_hash_join();
  sirius_physical_concat concat{
    sirius::from_duckdb_vec(f.logical_join->types), 0, f.hash_join.get(), /*is_build=*/false};
  REQUIRE(concat.no_history_peak_memory_estimate({0, 500}) == 0);
}

TEST_CASE("concat no_history_peak_memory_estimate: 1 batch returns 0",
          "[no_history_peak_memory_estimate][concat]")
{
  auto f = make_hash_join();
  sirius_physical_concat concat{
    sirius::from_duckdb_vec(f.logical_join->types), 0, f.hash_join.get(), false};
  REQUIRE(concat.no_history_peak_memory_estimate({1, 500}) == 0);
}

TEST_CASE("concat no_history_peak_memory_estimate: 2 batches returns bytes",
          "[no_history_peak_memory_estimate][concat]")
{
  auto f = make_hash_join();
  sirius_physical_concat concat{
    sirius::from_duckdb_vec(f.logical_join->types), 0, f.hash_join.get(), false};
  REQUIRE(concat.no_history_peak_memory_estimate({2, 500}) == 500);
}

TEST_CASE("concat no_history_peak_memory_estimate: many batches returns bytes",
          "[no_history_peak_memory_estimate][concat]")
{
  auto f = make_hash_join();
  sirius_physical_concat concat{
    sirius::from_duckdb_vec(f.logical_join->types), 0, f.hash_join.get(), false};
  REQUIRE(concat.no_history_peak_memory_estimate({10, 1024}) == 1024);
}

// ---------------------------------------------------------------------------
// sirius_physical_partition
// ---------------------------------------------------------------------------

TEST_CASE("partition no_history_peak_memory_estimate: num_partitions unset returns bytes",
          "[no_history_peak_memory_estimate][partition]")
{
  // _num_partitions is nullopt after construction with a hash join parent.
  auto f = make_hash_join();
  sirius_physical_partition part{
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    f.hash_join.get()};
  REQUIRE(part.no_history_peak_memory_estimate({1, 1000}) == 2000);
}

TEST_CASE("partition no_history_peak_memory_estimate: 1 partition returns 0",
          "[no_history_peak_memory_estimate][partition]")
{
  auto f = make_hash_join();
  sirius_physical_partition part{
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    f.hash_join.get()};
  part.set_num_partitions(1);
  REQUIRE(part.no_history_peak_memory_estimate({3, 2000}) == 0);
}

TEST_CASE("partition no_history_peak_memory_estimate: 2 partitions returns bytes",
          "[no_history_peak_memory_estimate][partition]")
{
  auto f = make_hash_join();
  sirius_physical_partition part{
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    f.hash_join.get()};
  part.set_num_partitions(2);
  REQUIRE(part.no_history_peak_memory_estimate({3, 2000}) == 4000);
}

TEST_CASE("partition no_history_peak_memory_estimate: many partitions returns bytes",
          "[no_history_peak_memory_estimate][partition]")
{
  auto f = make_hash_join();
  sirius_physical_partition part{
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    f.hash_join.get()};
  part.set_num_partitions(8);
  REQUIRE(part.no_history_peak_memory_estimate({5, 4096}) == 8192);
}

TEST_CASE("GPU scan adds filter-only decode bytes to its no-history estimate",
          "[no_history_peak_memory_estimate][gpu_scan]")
{
  scan::sirius_gpu_scan_operator scan{/*types=*/{}, /*estimated_cardinality=*/0, /*ingestible=*/{}};

  CHECK(scan.no_history_peak_memory_estimate({1, 100, operator_data_type::GPU_SCAN, false, 100}) ==
        800);
  CHECK(scan.no_history_peak_memory_estimate({1, 100, operator_data_type::GPU_SCAN, false, 700}) ==
        1400);
  CHECK(scan.no_history_peak_memory_estimate({1, 0, operator_data_type::GPU_SCAN, false, 600}) ==
        600);
  CHECK(scan.no_history_peak_memory_estimate({1, 100, operator_data_type::GPU_SCAN, true, 700}) ==
        100);
}
