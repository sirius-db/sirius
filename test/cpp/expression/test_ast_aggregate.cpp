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

// Tests for the sirius::ast::aggregate node, the sirius::aggregate_id name
// mappers, and the BOUND_AGGREGATE arm of sirius::ast::from_duckdb.
//
// Covers the id mapper round-trips + unsupported-name nullopt, direct
// aggregate-node construction, and translation of a directly-constructed
// duckdb::BoundAggregateExpression (sum, count_star with no children,
// COUNT(DISTINCT struct_pack(a, b)) recursion + distinct flag, and the
// unsupported-name -> nullptr fallback).

#include "catch.hpp"
#include "expression/ast/from_duckdb.hpp"

// sirius — node accessors and per-node struct types
#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"

// duckdb — direct-ctor construction surface
#include <duckdb/common/types.hpp>
#include <duckdb/function/aggregate_function.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// standard library
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using duckdb::AggregateFunction;
using duckdb::AggregateType;
using duckdb::BoundAggregateExpression;
using duckdb::BoundFunctionExpression;
using duckdb::BoundReferenceExpression;
using duckdb::LogicalType;
using duckdb::LogicalTypeId;
using duckdb::ScalarFunction;

using sirius::aggregate_id;
using sirius::from_duckdb_aggregate_name;
using sirius::to_duckdb_aggregate_name;
using sirius::ast::aggregate;
using sirius::ast::function_call;
using sirius::ast::node;
using sirius::ast::reference;

namespace {

// Minimal AggregateFunction stub: only `.name` is read by from_duckdb; the
// callback function pointers are never invoked during translation.
AggregateFunction make_dummy_aggregate(std::string const& name,
                                       duckdb::vector<LogicalType> const& args,
                                       LogicalType const& ret_type)
{
  return AggregateFunction(duckdb::Identifier(name),
                           args,
                           ret_type,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);
}

}  // namespace

// ============================================================================
// Compile-time invariants (locked ABI)
// ============================================================================

static_assert(std::is_enum_v<aggregate_id>, "sirius::aggregate_id must be an enum class.");
static_assert(sizeof(aggregate_id) == 2, "sirius::aggregate_id is uint16_t-backed (locked ABI).");
static_assert(static_cast<uint16_t>(aggregate_id::first) + 1 == 8,
              "sirius::aggregate_id has exactly 8 entries (locked ABI).");

// ============================================================================
// aggregate_id name mapper round-trips
// ============================================================================

TEST_CASE("ast_aggregate - every supported aggregate name maps to its id", "[ast_aggregate]")
{
  struct entry {
    std::string_view name;
    aggregate_id id;
  };
  entry const cases[] = {
    {"sum", aggregate_id::sum},
    {"sum_no_overflow", aggregate_id::sum_no_overflow},
    {"count", aggregate_id::count},
    {"count_star", aggregate_id::count_star},
    {"min", aggregate_id::min},
    {"max", aggregate_id::max},
    {"avg", aggregate_id::avg},
    {"first", aggregate_id::first},
  };
  for (auto const& c : cases) {
    auto const id = from_duckdb_aggregate_name(c.name);
    REQUIRE(id.has_value());
    REQUIRE(*id == c.id);
    // Round-trip back to the canonical DuckDB name.
    REQUIRE(to_duckdb_aggregate_name(*id) == c.name);
  }
}

TEST_CASE("ast_aggregate - unsupported aggregate name returns std::nullopt", "[ast_aggregate]")
{
  REQUIRE_FALSE(from_duckdb_aggregate_name("stddev").has_value());
  REQUIRE_FALSE(from_duckdb_aggregate_name("").has_value());
}

// ============================================================================
// Direct aggregate-node construction
// ============================================================================

TEST_CASE("ast_aggregate - direct node construction reads back fields", "[ast_aggregate]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(std::make_unique<node>(reference{/*column_index=*/3}));

  auto agg_node =
    std::make_unique<node>(aggregate{aggregate_id::sum,
                                     std::move(args),
                                     sirius::logical_type::make(sirius::type_id::BIGINT),
                                     /*distinct=*/true});

  REQUIRE(agg_node->holds<aggregate>());
  auto const& agg = agg_node->get<aggregate>();
  REQUIRE(agg.function() == aggregate_id::sum);
  REQUIRE(agg.arguments().size() == 1);
  REQUIRE(agg.return_type().id() == sirius::type_id::BIGINT);
  REQUIRE(agg.distinct() == true);
}

// ============================================================================
// from_duckdb(BOUND_AGGREGATE)
// ============================================================================

TEST_CASE("ast_aggregate - sum over a single reference translates to aggregate node",
          "[ast_aggregate]")
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
  children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  auto fn = make_dummy_aggregate(
    "sum", {LogicalType{LogicalTypeId::INTEGER}}, LogicalType{LogicalTypeId::BIGINT});
  auto expr = duckdb::make_uniq<BoundAggregateExpression>(duckdb::BoundAggregateFunction(fn),
                                                          std::move(children),
                                                          nullptr,
                                                          nullptr,
                                                          AggregateType::NON_DISTINCT);

  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<aggregate>());
  auto const& agg = out->get<aggregate>();
  REQUIRE(agg.function() == aggregate_id::sum);
  REQUIRE(agg.arguments().size() == 1);
  REQUIRE(agg.arguments()[0]->holds<reference>());
  REQUIRE(agg.distinct() == false);
}

TEST_CASE("ast_aggregate - count_star with no children translates without throwing",
          "[ast_aggregate]")
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;  // empty
  auto fn   = make_dummy_aggregate("count_star", {}, LogicalType{LogicalTypeId::BIGINT});
  auto expr = duckdb::make_uniq<BoundAggregateExpression>(duckdb::BoundAggregateFunction(fn),
                                                          std::move(children),
                                                          nullptr,
                                                          nullptr,
                                                          AggregateType::NON_DISTINCT);

  std::unique_ptr<node> out;
  REQUIRE_NOTHROW(out = sirius::ast::from_duckdb(*expr));
  REQUIRE(out);
  REQUIRE(out->holds<aggregate>());
  auto const& agg = out->get<aggregate>();
  REQUIRE(agg.function() == aggregate_id::count_star);
  REQUIRE(agg.arguments().empty());
}

TEST_CASE("ast_aggregate - COUNT(DISTINCT struct_pack(a, b)) recurses children and sets distinct",
          "[ast_aggregate]")
{
  // Inner struct_pack(a, b): a BoundFunctionExpression over two references.
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> struct_children;
  struct_children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  struct_children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));
  auto struct_return_type = LogicalType::STRUCT(
    {{"a", LogicalType{LogicalTypeId::INTEGER}}, {"b", LogicalType{LogicalTypeId::INTEGER}}});
  ScalarFunction struct_fn(
    "struct_pack", duckdb::vector<LogicalType>{}, struct_return_type, /*function=*/nullptr);
  auto struct_pack_expr = duckdb::make_uniq<BoundFunctionExpression>(
    duckdb::BoundScalarFunction(struct_fn), std::move(struct_children), /*bind_info=*/nullptr);

  // Outer count(DISTINCT struct_pack(a, b)).
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_children;
  agg_children.push_back(std::move(struct_pack_expr));
  auto fn = make_dummy_aggregate("count", {struct_return_type}, LogicalType{LogicalTypeId::BIGINT});
  auto expr = duckdb::make_uniq<BoundAggregateExpression>(duckdb::BoundAggregateFunction(fn),
                                                          std::move(agg_children),
                                                          nullptr,
                                                          nullptr,
                                                          AggregateType::DISTINCT);

  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<aggregate>());
  auto const& agg = out->get<aggregate>();
  REQUIRE(agg.function() == aggregate_id::count);
  REQUIRE(agg.distinct() == true);
  REQUIRE(agg.arguments().size() == 1);
  // The single argument recursed through from_duckdb into a struct_pack function_call.
  REQUIRE(agg.arguments()[0]->holds<function_call>());
  REQUIRE(agg.arguments()[0]->get<function_call>().function() == sirius::function_id::struct_pack);
}

TEST_CASE("ast_aggregate - unsupported aggregate name yields nullptr fallback (no throw)",
          "[ast_aggregate]")
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
  children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::DOUBLE}, 0));
  auto fn = make_dummy_aggregate(
    "stddev", {LogicalType{LogicalTypeId::DOUBLE}}, LogicalType{LogicalTypeId::DOUBLE});
  auto expr = duckdb::make_uniq<BoundAggregateExpression>(duckdb::BoundAggregateFunction(fn),
                                                          std::move(children),
                                                          nullptr,
                                                          nullptr,
                                                          AggregateType::NON_DISTINCT);

  std::unique_ptr<node> out;
  REQUIRE_NOTHROW(out = sirius::ast::from_duckdb(*expr));
  REQUIRE(out == nullptr);
}
