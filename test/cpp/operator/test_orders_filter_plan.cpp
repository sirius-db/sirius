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

// ---------------------------------------------------------------------------
// Course example: a hand-built physical plan for `SELECT * FROM orders WHERE amount > 100`
//
// This test represents that SQL query using only Sirius physical-plan operators,
// wired and executed directly — no SQL parser, no planner, no task scheduler.
// It is a companion artifact for the Sirius Internals course: it shows the two
// operators a `LOGICAL_GET -> LOGICAL_FILTER` plan lowers to, and how the engine
// runs them.
//
//   SQL:      SELECT * FROM orders WHERE amount > 100
//   Logical:  LOGICAL_FILTER(amount > 100)
//               └── LOGICAL_GET(orders)
//   Physical: sirius_physical_filter        <-- WHERE amount > 100  (LOGICAL_FILTER -> FILTER)
//               └── sirius_physical_table_scan   <-- FROM orders    (LOGICAL_GET   -> TABLE_SCAN)
//
// The operator/query mapping is documented in
//   docs/super-sirius/physical-plan-generation.md   (Operator Mapping Table)
// and the "each operator's execute() is called in sequence" execution model in
//   docs/super-sirius/pipeline-execution.md          (Execution Model).
//
// Instead of going through pipelines and the GPU scheduler (which read Parquet
// splits and stream batches), we drive the two operators by hand: build one
// in-memory `orders` batch on the GPU, run the scan's execute(), then feed its
// output straight into the filter's execute(). This is exactly the direct
// operator pattern used by test_physical_table_scan.cpp and test_physical_filter.cpp.
// ---------------------------------------------------------------------------

#include "helper/type_conversions.hpp"  // sirius::from_duckdb_vec
#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"  // initialize_memory_manager, make_two_column_batch, copy_column_to_host

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_table_scan.hpp>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;

namespace {

using namespace sirius::test::operator_utils;

// Column layout of our in-memory `orders` table (what `SELECT *` returns).
constexpr int kOrderKeyCol = 0;  // o_orderkey : BIGINT
constexpr int kAmountCol   = 1;  // amount     : DOUBLE

// Pull the single output batch out of an operator's execute() result and return
// its cuDF table view for inspection.
cudf::table_view single_output_view(const std::unique_ptr<operator_data>& outputs)
{
  const auto& pod = dynamic_cast<const pipelineable_operator_data&>(*outputs);
  REQUIRE(pod.get_data_batches().size() == 1);
  return sirius::get_cudf_table_view(*pod.get_data_batches()[0]);
}

}  // namespace

TEST_CASE("physical plan for `SELECT * FROM orders WHERE amount > 100` (scan -> filter)",
          "[orders_filter_plan]")
{
  // --- GPU memory: the only "environment" an operator-level test needs. --------
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // --- The `orders` table, built in GPU memory (stands in for FROM orders). ----
  // Two columns: o_orderkey (BIGINT) and amount (DOUBLE). Note the boundary row
  // amount == 100.0, which the strict `> 100` predicate must exclude.
  std::vector<int64_t> orderkeys{1, 2, 3, 4, 5, 6};
  std::vector<double> amounts{50.0, 100.0, 100.01, 250.0, 99.99, 500.0};

  auto orders_batch = make_two_column_batch<int64_t, double>(
    *space, orderkeys, amounts, cudf::type_id::FLOAT64, std::nullopt);

  // --- Operator 1: FROM orders  (LOGICAL_GET -> TABLE_SCAN) --------------------
  // A pass-through scan: project every column, push down no filter. It hands the
  // whole `orders` relation to the next operator, exactly like a LOGICAL_GET with
  // the WHERE clause left for a separate FILTER above it.
  duckdb::vector<duckdb::LogicalType> orders_types;
  orders_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // o_orderkey
  orders_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE));  // amount
  duckdb::vector<duckdb::LogicalType> returned_types = orders_types;

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.push_back(duckdb::ColumnIndex(kOrderKeyCol));
  column_ids.push_back(duckdb::ColumnIndex(kAmountCol));

  duckdb::vector<duckdb::idx_t> projection_ids{kOrderKeyCol, kAmountCol};
  duckdb::vector<std::string> names{"o_orderkey", "amount"};
  duckdb::vector<duckdb::Value> parameters;
  duckdb::virtual_column_map_t virtual_columns;

  // Empty filter set => the scan passes all rows through unchanged.
  auto table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();

  // A minimal TableFunction is required by the constructor but never invoked here:
  // our data is already materialized in the input batch.
  duckdb::TableFunction table_function("orders_scan", {}, nullptr, nullptr);

  sirius_physical_table_scan scan(sirius::from_duckdb_vec(orders_types),
                                  std::move(table_function),
                                  nullptr,  // bind_data
                                  sirius::from_duckdb_vec(returned_types),
                                  std::move(column_ids),
                                  std::move(projection_ids),
                                  std::move(names),
                                  std::move(table_filters),
                                  orderkeys.size(),
                                  duckdb::ExtraOperatorInfo(),
                                  std::move(parameters),
                                  std::move(virtual_columns));

  // --- Operator 2: WHERE amount > 100  (LOGICAL_FILTER -> FILTER) --------------
  // Build the predicate `amount > 100` as a bound comparison, then lower it into
  // Sirius' native expression AST via sirius::ast::from_duckdb (the same call the
  // planner makes in src/planner/sirius_plan_filter.cpp). BoundReferenceExpression's
  // index is the column's position in the input batch — amount lives at column 1.
  auto predicate = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE),
                                                kAmountCol),
    duckdb::make_uniq<BoundConstantExpression>(duckdb::Value::DOUBLE(100)));
  auto filter_expr = sirius::ast::from_duckdb(*predicate);

  sirius_physical_filter filter(
    sirius::from_duckdb_vec(orders_types), std::move(filter_expr), orderkeys.size());

  // --- Run the plan: scan.execute() feeds filter.execute() -------------------
  // This is the whole pipeline by hand — no scheduler. compute_task() would do the
  // same thing: call execute() on each operator in source-to-sink order.
  auto stream = cudf::get_default_stream();

  std::vector<std::shared_ptr<cucascade::data_batch>> scan_input{orders_batch};
  auto scan_output = scan.execute(pipelineable_operator_data(scan_input), stream);

  // The scan yields the full `orders` relation; hand those batches to the filter.
  auto scanned_batches =
    dynamic_cast<const pipelineable_operator_data&>(*scan_output).get_data_batches();
  auto filter_output = filter.execute(pipelineable_operator_data(scanned_batches), stream);

  // --- Verify: only rows with amount > 100 survive, all columns preserved. -----
  auto result_view = single_output_view(filter_output);
  REQUIRE(result_view.num_columns() == 2);  // SELECT * kept both columns

  auto host_orderkeys = copy_column_to_host<int64_t>(result_view.column(kOrderKeyCol));
  auto host_amounts   = copy_column_to_host<double>(result_view.column(kAmountCol));

  std::vector<int64_t> expected_orderkeys;
  std::vector<double> expected_amounts;
  for (size_t i = 0; i < orderkeys.size(); ++i) {
    if (amounts[i] > 100.0) {
      expected_orderkeys.push_back(orderkeys[i]);
      expected_amounts.push_back(amounts[i]);
    }
  }

  // Rows 3 (100.01), 4 (250.0), 6 (500.0) pass; 100.0 boundary and below are dropped.
  REQUIRE(host_orderkeys == expected_orderkeys);
  REQUIRE(host_amounts == expected_amounts);
  REQUIRE(result_view.num_rows() == static_cast<cudf::size_type>(expected_orderkeys.size()));
}
