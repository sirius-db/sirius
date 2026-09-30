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

// What a scan must still evaluate after the decode, given what the decode says
// it did. Three outcomes per conjunct, and getting them wrong is silent:
//   * the decode did nothing with it       -> keep the comparison
//   * the decode ANSWERED it (BOOL8 in the column, rows untouched)
//                                          -> reference the answer; re-running
//                                             the comparison would test a mask
//                                             against the original constant
//   * the decode APPLIED it (folded into the row selection)
//                                          -> drop it; the surviving rows
//                                             already satisfy it
//
// The last case is reachable only when the filtered decode applies with an
// equality source, which the default thresholds decline for the sizes the
// end-to-end tests use — so it is pinned here, where the decision is made,
// rather than left to a configuration that happens to reach it.

#include "op/scan/scan_filter_analysis.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <cstddef>
#include <utility>
#include <vector>

using namespace sirius::op;

namespace {

/// `column[batch_position] = 'x'`, the shape a pure-filter string column's
/// conjunct takes.
table_filter_conjunct string_equality(std::size_t primary_index, std::size_t batch_position)
{
  auto column = duckdb::make_uniq<duckdb::BoundReferenceExpression>(
    duckdb::LogicalType::VARCHAR, static_cast<duckdb::idx_t>(batch_position));
  auto constant =
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value("DELIVER IN PERSON"));
  return {primary_index,
          batch_position,
          duckdb::make_uniq<duckdb::BoundComparisonExpression>(
            duckdb::ExpressionType::COMPARE_EQUAL, std::move(column), std::move(constant))};
}

std::vector<table_filter_conjunct> two_conjuncts()
{
  std::vector<table_filter_conjunct> conjuncts;
  conjuncts.push_back(string_equality(/*primary_index=*/7, /*batch_position=*/1));
  conjuncts.push_back(string_equality(/*primary_index=*/9, /*batch_position=*/2));
  return conjuncts;
}

}  // namespace

TEST_CASE("residual_filter keeps every conjunct when the decode answered nothing", "[scan]")
{
  residual_filter const residual{two_conjuncts(), /*answerable_batch_positions=*/{1, 2}};
  REQUIRE_FALSE(residual.empty());

  auto const predicate = residual.against(/*answered_positions=*/{});
  REQUIRE(predicate != nullptr);
  REQUIRE(predicate->holds<sirius::ast::conjunction>());
  auto const& all = predicate->get<sirius::ast::conjunction>();
  REQUIRE(all.op == sirius::ast::conjunction::kind::op_and);
  REQUIRE(all.children.size() == 2);
  // Both still comparisons — nothing was answered, so nothing may be referenced.
  REQUIRE(all.children[0]->holds<sirius::ast::comparison>());
  REQUIRE(all.children[1]->holds<sirius::ast::comparison>());
}

TEST_CASE("residual_filter references an answered conjunct's column", "[scan]")
{
  residual_filter const residual{two_conjuncts(), {1, 2}};

  auto const predicate = residual.against(/*answered_positions=*/{1});
  REQUIRE(predicate != nullptr);
  auto const& all = predicate->get<sirius::ast::conjunction>();
  REQUIRE(all.children.size() == 2);
  // The answered column IS the conjunct's truth value, read as a boolean.
  REQUIRE(all.children[0]->holds<sirius::ast::reference>());
  auto const& ref = all.children[0]->get<sirius::ast::reference>();
  REQUIRE(ref.column_index == 1);
  REQUIRE(ref.return_type().id() == sirius::type_id::BOOLEAN);
  REQUIRE(all.children[1]->holds<sirius::ast::comparison>());
}

TEST_CASE("residual_filter drops a conjunct the decode applied to the rows", "[scan]")
{
  residual_filter const residual{two_conjuncts(), {1, 2}};

  auto const predicate = residual.against(/*answered_positions=*/{1}, /*answers_enforced=*/true);
  REQUIRE(predicate != nullptr);
  // One conjunct left, so no conjunction wrapper — and it is the OTHER column's
  // comparison, not a reference to the applied one.
  REQUIRE(predicate->holds<sirius::ast::comparison>());
}

TEST_CASE("residual_filter is empty when the decode applied every conjunct", "[scan]")
{
  residual_filter const residual{two_conjuncts(), {1, 2}};

  // Null means "already filtered", NOT "no filtering needed" — the caller must
  // mark the batch row-filtered rather than pass the rows through unfiltered.
  REQUIRE(residual.against({1, 2}, /*answers_enforced=*/true) == nullptr);
  // Without enforcement the same answers only become references: the decode
  // reported the values, not a smaller row set.
  auto const answered_only = residual.against({1, 2}, /*answers_enforced=*/false);
  REQUIRE(answered_only != nullptr);
  REQUIRE(answered_only->get<sirius::ast::conjunction>().children.size() == 2);
}

TEST_CASE("residual_filter leaves unanswerable columns alone", "[scan]")
{
  // Only position 1 was nominated, so position 2 keeps its comparison however
  // the decode reports itself — a column the scan never offered cannot have
  // been answered for it.
  residual_filter const residual{two_conjuncts(), /*answerable_batch_positions=*/{1}};

  auto const predicate = residual.against({1, 2}, /*answers_enforced=*/true);
  REQUIRE(predicate != nullptr);
  REQUIRE(predicate->holds<sirius::ast::comparison>());
}
