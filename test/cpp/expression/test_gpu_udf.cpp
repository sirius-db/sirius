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

#include "ast_test_builders.hpp"

#include <cudf/column/column.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/gpu_function_call.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/utils.hpp>
#include <sirius/udf.hpp>

#include <memory>
#include <string>
#include <vector>

TEST_CASE("native GPU UDF clone and substitution retain binding and NULL policy", "[gpu_udf][ast]")
{
  using namespace sirius::ast;
  using sirius::ast::test::make_ref;
  auto implementation = std::make_shared<sirius::gpu_scalar_function>();
  auto const type     = sirius::logical_type::make(sirius::type_id::INTEGER);
  std::vector<std::unique_ptr<node>> arguments;
  arguments.push_back(make_ref(0, type));
  arguments.push_back(make_ref(1, type));
  node original{gpu_function_call{implementation, std::move(arguments), type, false}};

  auto copied = clone(original);
  REQUIRE(copied->holds<gpu_function_call>());
  auto const& call = copied->get<gpu_function_call>();
  CHECK(call.implementation() == implementation);
  CHECK(call.return_type() == type);
  CHECK_FALSE(call.propagate_nulls());
  REQUIRE(call.arguments().size() == 2);
  CHECK(call.arguments()[0].get() != original.get<gpu_function_call>().arguments()[0].get());

  std::vector<std::unique_ptr<node>> replacements;
  replacements.push_back(make_ref(4, type));
  replacements.push_back(make_ref(2, type));
  auto substituted = substitute_references(original, replacements);
  REQUIRE(substituted->holds<gpu_function_call>());
  auto const& substituted_call = substituted->get<gpu_function_call>();
  CHECK(substituted_call.implementation() == implementation);
  CHECK(substituted_call.return_type() == type);
  CHECK_FALSE(substituted_call.propagate_nulls());
  std::vector<uint32_t> indices;
  visit_references(*substituted,
                   [&](reference const& ref) { indices.push_back(ref.column_index); });
  CHECK(indices == std::vector<uint32_t>{4, 2});

  // Cloned plans own their binding independently of the temporary used to register it.
  original = node{constant{sirius::value{int32_t{0}}, type}};
  implementation.reset();
  CHECK(call.implementation() != nullptr);
  CHECK(substituted_call.implementation() == call.implementation());
}

TEST_CASE("GPU UDF registration is database local and bound expression copies retain identity",
          "[gpu_udf][ast][registration]")
{
  duckdb::DuckDB first_database{nullptr};
  duckdb::DuckDB second_database{nullptr};
  duckdb::Connection first{first_database};
  duckdb::Connection second{second_database};
  auto make_implementation = [] {
    return std::make_shared<sirius::gpu_scalar_function>(sirius::gpu_scalar_function{
      [](std::span<cudf::column_view const>,
         cudf::size_type,
         ::cuda::stream_ref,
         rmm::device_async_resource_ref) -> std::unique_ptr<cudf::column> { return nullptr; }});
  };
  auto first_implementation  = make_implementation();
  auto second_implementation = make_implementation();
  auto register_function     = [](duckdb::Connection& connection,
                              std::shared_ptr<sirius::gpu_scalar_function const> implementation) {
    duckdb::ExtensionLoader loader(duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                   "synthetic_gpu_udf");
    sirius::register_gpu_scalar_function(
      loader,
      duckdb::ScalarFunction(
        "synthetic_identity",
        {duckdb::LogicalType::INTEGER},
        duckdb::LogicalType::INTEGER,
        [](duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output) {
          for (duckdb::idx_t row = 0; row < args.size(); ++row) {
            output.SetValue(row, args.GetValue(0, row));
          }
        }),
      [implementation](duckdb::BoundFunctionExpression const&) { return implementation; });
    auto disabled = connection.Query("PRAGMA disable_optimizer");
    REQUIRE_FALSE(disabled->HasError());
  };
  register_function(first, first_implementation);
  register_function(second, second_implementation);
  REQUIRE_THROWS(register_function(first, second_implementation));
  auto bind = [](duckdb::Connection& connection,
                 std::string const& query = "SELECT synthetic_identity(2)") {
    auto plan = connection.ExtractPlan(query);
    REQUIRE(plan);
    REQUIRE(plan->expressions.size() == 1);
    auto expression_copy = plan->expressions[0]->Copy();
    auto result          = sirius::ast::from_duckdb(*expression_copy);
    REQUIRE(result);
    REQUIRE(result->holds<sirius::ast::gpu_function_call>());
    return result;
  };
  auto first_call  = bind(first);
  auto second_call = bind(second);
  CHECK(first_call->get<sirius::ast::gpu_function_call>().implementation() == first_implementation);
  CHECK(second_call->get<sirius::ast::gpu_function_call>().implementation() ==
        second_implementation);

  // Keep a DuckDB-bound plan alive across the catalog ALTER that adds an overload.
  auto old_plan = first.ExtractPlan("SELECT synthetic_identity(2)");
  REQUIRE(old_plan);
  REQUIRE(old_plan->expressions.size() == 1);
  auto bigint_implementation = make_implementation();
  duckdb::ExtensionLoader first_loader(duckdb::DatabaseInstance::GetDatabase(*first.context),
                                       "synthetic_gpu_udf");
  sirius::register_gpu_scalar_function(
    first_loader,
    duckdb::ScalarFunction(
      "synthetic_identity",
      {duckdb::LogicalType::BIGINT},
      duckdb::LogicalType::BIGINT,
      [](duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output) {
        for (duckdb::idx_t row = 0; row < args.size(); ++row) {
          output.SetValue(row, args.GetValue(0, row));
        }
      }),
    [bigint_implementation](duckdb::BoundFunctionExpression const&) {
      return bigint_implementation;
    });
  auto old_copy      = old_plan->expressions[0]->Copy();
  auto retained_call = sirius::ast::from_duckdb(*old_copy);
  REQUIRE(retained_call);
  REQUIRE(retained_call->holds<sirius::ast::gpu_function_call>());
  CHECK(retained_call->get<sirius::ast::gpu_function_call>().implementation() ==
        first_implementation);
  CHECK(bind(first)->get<sirius::ast::gpu_function_call>().implementation() ==
        first_implementation);
  auto bigint_call = bind(first, "SELECT synthetic_identity(2::BIGINT)");
  CHECK(bigint_call->get<sirius::ast::gpu_function_call>().implementation() ==
        bigint_implementation);

  // Spelling alone cannot route an unrelated CPU function to a registered GPU implementation.
  auto unrelated = duckdb::make_uniq<duckdb::BoundFunctionExpression>(
    duckdb::BoundScalarFunction(duckdb::ScalarFunction(
      "synthetic_identity", {duckdb::LogicalType::INTEGER}, duckdb::LogicalType::INTEGER, nullptr)),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  unrelated->GetChildrenMutable().push_back(
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(2)));
  CHECK(sirius::ast::from_duckdb(*unrelated) == nullptr);
}

TEST_CASE("GPU UDF explicit schemas keep independent bindings despite loader schema overrides",
          "[gpu_udf][ast][registration]")
{
  duckdb::DuckDB database{nullptr};
  duckdb::Connection connection{database};
  auto disabled = connection.Query("PRAGMA disable_optimizer");
  REQUIRE_FALSE(disabled->HasError());
  duckdb::ExtensionLoader loader(duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                 "synthetic_schema_udf");
  loader.CreateSchema("native_left");
  loader.CreateSchema("native_right");
  loader.UseDedicatedSchemaForExtension("loader_scope");

  auto make_implementation = [] {
    return std::make_shared<sirius::gpu_scalar_function>(sirius::gpu_scalar_function{
      [](std::span<cudf::column_view const>,
         cudf::size_type,
         ::cuda::stream_ref,
         rmm::device_async_resource_ref) -> std::unique_ptr<cudf::column> { return nullptr; }});
  };
  auto left_implementation  = make_implementation();
  auto right_implementation = make_implementation();
  auto register_function    = [&](std::string const& schema,
                               std::shared_ptr<sirius::gpu_scalar_function const> implementation) {
    duckdb::ScalarFunction function(
      "synthetic_identity",
      {duckdb::LogicalType::INTEGER},
      duckdb::LogicalType::INTEGER,
      [](duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output) {
        for (duckdb::idx_t row = 0; row < args.size(); ++row) {
          output.SetValue(row, args.GetValue(0, row));
        }
      });
    if (!schema.empty()) { function.SetSchemaName(duckdb::Identifier(schema)); }
    sirius::register_gpu_scalar_function(
      loader, std::move(function), [implementation](duckdb::BoundFunctionExpression const&) {
        return implementation;
      });
  };
  register_function("native_left", left_implementation);
  register_function("native_right", right_implementation);
  REQUIRE_THROWS(register_function("native_left", right_implementation));

  auto bind = [&](std::string const& schema) {
    auto plan = connection.ExtractPlan("SELECT system." + schema + ".synthetic_identity(2)");
    REQUIRE(plan);
    REQUIRE(plan->expressions.size() == 1);
    auto expression_copy = plan->expressions[0]->Copy();
    CHECK(expression_copy->Equals(*plan->expressions[0]));
    auto result = sirius::ast::from_duckdb(*expression_copy);
    REQUIRE(result);
    REQUIRE(result->holds<sirius::ast::gpu_function_call>());
    return result;
  };
  auto left_call  = bind("native_left");
  auto right_call = bind("native_right");
  CHECK(left_call->get<sirius::ast::gpu_function_call>().implementation() == left_implementation);
  CHECK(right_call->get<sirius::ast::gpu_function_call>().implementation() == right_implementation);

  // An unspecified CPU schema still means main, independently of loader_scope.
  register_function("", left_implementation);
  CHECK(bind("main")->get<sirius::ast::gpu_function_call>().implementation() ==
        left_implementation);
  auto absent = connection.Query("SELECT system.loader_scope.synthetic_identity(2)");
  REQUIRE(absent);
  CHECK(absent->HasError());
}

TEST_CASE("GPU UDF registration rejects unsafe execution and existing CPU binding contracts",
          "[gpu_udf][ast][registration]")
{
  auto const unsupported = GENERATE(0, 1, 2, 3, 4, 5);
  CAPTURE(unsupported);
  duckdb::DuckDB database{nullptr};
  duckdb::Connection connection{database};
  duckdb::ExtensionLoader loader(duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                 "synthetic_invalid_udf");
  duckdb::ScalarFunction function(
    "synthetic_unsafe",
    {duckdb::LogicalType::INTEGER},
    duckdb::LogicalType::INTEGER,
    [](duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output) {
      for (duckdb::idx_t row = 0; row < args.size(); ++row) {
        output.SetValue(row, args.GetValue(0, row));
      }
    });
  switch (unsupported) {
    case 0: function.SetVolatile(); break;
    case 1: function.SetRequiresOrderedExecution(true); break;
    case 2: function.SetStability(duckdb::FunctionStability::CONSISTENT_WITHIN_QUERY); break;
    case 3:
      function.SetBindCallback(
        [](duckdb::BindScalarFunctionInput&) -> duckdb::unique_ptr<duckdb::FunctionData> {
          return nullptr;
        });
      break;
    case 4:
      function.SetExtraFunctionInfo(duckdb::make_shared_ptr<duckdb::ScalarFunctionInfo>());
      break;
    case 5:
      function.SetModifiedDatabasesCallback(
        [](duckdb::ClientContext&, duckdb::FunctionModifiedDatabasesInput&) {});
      break;
  }
  sirius::gpu_scalar_binder decline = [](duckdb::BoundFunctionExpression const&)
    -> std::shared_ptr<sirius::gpu_scalar_function const> { return nullptr; };
  REQUIRE_THROWS_AS(sirius::register_gpu_scalar_function(loader, function, decline),
                    duckdb::InvalidInputException);
  // Rejection must leave no partial registration in the catalog.
  auto absent = connection.Query("SELECT synthetic_unsafe(2)");
  REQUIRE(absent);
  CHECK(absent->HasError());
}

TEST_CASE("declined registered UDF cannot fall through to a builtin with the same name",
          "[gpu_udf][ast][registration]")
{
  duckdb::DuckDB database{nullptr};
  duckdb::Connection connection{database};
  auto disabled = connection.Query("PRAGMA disable_optimizer");
  REQUIRE_FALSE(disabled->HasError());
  duckdb::ExtensionLoader loader(duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                 "synthetic_declined_udf");
  loader.CreateSchema("native_decline");
  duckdb::ScalarFunction function(
    "abs",
    {duckdb::LogicalType::INTEGER},
    duckdb::LogicalType::INTEGER,
    [](duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output) {
      for (duckdb::idx_t row = 0; row < args.size(); ++row) {
        output.SetValue(row, duckdb::Value::INTEGER(17));
      }
    });
  function.SetSchemaName("native_decline");
  auto attempts = std::make_shared<int>(0);
  sirius::register_gpu_scalar_function(loader,
                                       std::move(function),
                                       [attempts](duckdb::BoundFunctionExpression const&)
                                         -> std::shared_ptr<sirius::gpu_scalar_function const> {
                                         ++*attempts;
                                         return nullptr;
                                       });
  auto plan = connection.ExtractPlan("SELECT system.native_decline.abs(-9::INTEGER)");
  REQUIRE(plan);
  REQUIRE(plan->expressions.size() == 1);
  CHECK(sirius::ast::from_duckdb(*plan->expressions[0]) == nullptr);
  CHECK(*attempts == 1);

  auto original = connection.Query("SELECT system.native_decline.abs(-9::INTEGER)");
  REQUIRE(original);
  REQUIRE_FALSE(original->HasError());
  CHECK(original->Collection().GetRows().GetValue(0, 0).GetValue<int32_t>() == 17);
  auto builtin = connection.Query("SELECT system.main.abs(-9::INTEGER)");
  REQUIRE(builtin);
  REQUIRE_FALSE(builtin->HasError());
  CHECK(builtin->Collection().GetRows().GetValue(0, 0).GetValue<int32_t>() == 9);
}
