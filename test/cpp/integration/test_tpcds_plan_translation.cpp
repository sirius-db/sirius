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

/**
 * @file test_tpcds_plan_translation.cpp
 * @brief Tests how many TPC-DS queries can be translated from DuckDB logical
 *        plans to physical plans, for both the new Sirius planner and the
 *        legacy GPU planner.
 *
 * This test does NOT execute the queries on GPU. It only tests plan
 * translation coverage:
 *   - sirius_physical_plan_generator::create_plan  (new Sirius planner)
 *   - GPUPhysicalPlanGenerator dispatch table       (legacy GPU planner)
 *
 * Requirements:
 *   - DuckDB tpcds extension must be installable (requires network on first run)
 *
 * The test registers custom table functions `sirius_plan_check(query)` and
 * `gpu_plan_check(query)` that perform plan extraction and translation,
 * returning the result as a row instead of attempting execution.
 */

#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace duckdb;

// ============================================================================
// Custom table function: sirius_plan_check(query VARCHAR)
// Returns: (success BOOLEAN, error_message VARCHAR)
//
// Performs the same plan extraction and Sirius plan generation as
// gpu_execution's bind phase, but captures the result instead of
// proceeding to execution.
// ============================================================================

struct PlanCheckBindData : public TableFunctionData {
  bool success;
  std::string error_message;
  bool finished = false;
};

unique_ptr<FunctionData> PlanCheckBind(ClientContext& context,
                                       TableFunctionBindInput& input,
                                       vector<LogicalType>& return_types,
                                       vector<string>& names)
{
  auto result  = make_uniq<PlanCheckBindData>();
  string query = input.inputs[0].ToString();

  // Save and modify disabled optimizers (same as Sirius extension)
  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  try {
    // 1. Parse
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);

    if (parser.statements.empty()) { throw std::runtime_error("No statements parsed"); }

    // 2. Plan (creates LogicalOperator tree)
    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));

    if (!planner.plan) { throw std::runtime_error("Planner produced null plan"); }

    auto plan = std::move(planner.plan);

    // 3. Optimize
    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    // 4. Resolve types and column bindings
    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*plan);
    resolver.VisitOperator(*plan);

    // 5. Translate to Sirius physical plan
    sirius::planner::sirius_physical_plan_generator gen(context);
    auto sirius_plan = gen.create_plan(std::move(plan));

    result->success = true;
  } catch (const std::exception& e) {
    result->success       = false;
    result->error_message = e.what();
  }

  // Restore disabled optimizers
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;

  // Output schema
  return_types.push_back(LogicalType::BOOLEAN);
  names.push_back("success");
  return_types.push_back(LogicalType::VARCHAR);
  names.push_back("error_message");

  return std::move(result);
}

// ============================================================================
// Legacy GPU planner operator support check.
//
// We cannot call GPUPhysicalPlanGenerator::CreatePlan directly because GPU
// physical operators allocate CUDA memory in their constructors, which fails
// without an initialized GPU buffer. Instead, we walk the optimized logical
// plan tree and check each operator against the legacy planner's dispatch
// table (GPUPhysicalPlanGenerator::CreatePlan in gpu_physical_plan_generator.cpp).
// ============================================================================

// Returns the error message for the first unsupported operator found,
// or an empty string if all operators in the tree are supported.
std::string find_unsupported_gpu_operator(LogicalOperator& op)
{
  // Check children first (depth-first)
  for (auto& child : op.children) {
    auto err = find_unsupported_gpu_operator(*child);
    if (!err.empty()) return err;
  }

  // Mirror the dispatch table from GPUPhysicalPlanGenerator::CreatePlan(LogicalOperator&)
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET:
    case LogicalOperatorType::LOGICAL_PROJECTION:
    case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
    case LogicalOperatorType::LOGICAL_FILTER:
    case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
    case LogicalOperatorType::LOGICAL_LIMIT:
    case LogicalOperatorType::LOGICAL_ORDER_BY:
    case LogicalOperatorType::LOGICAL_TOP_N:
    case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
    case LogicalOperatorType::LOGICAL_DELIM_JOIN:
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    case LogicalOperatorType::LOGICAL_CHUNK_GET:
    case LogicalOperatorType::LOGICAL_DELIM_GET:
    case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
    case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
    case LogicalOperatorType::LOGICAL_CTE_REF: return "";

    case LogicalOperatorType::LOGICAL_WINDOW: return "Window not supported";
    case LogicalOperatorType::LOGICAL_UNNEST: return "Unnest not supported";
    case LogicalOperatorType::LOGICAL_SAMPLE: return "Sample not supported";
    case LogicalOperatorType::LOGICAL_COPY_TO_FILE: return "Copy to file not supported";
    case LogicalOperatorType::LOGICAL_ANY_JOIN: return "Any join not supported";
    case LogicalOperatorType::LOGICAL_ASOF_JOIN: return "Asof join not supported";
    case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: return "Cross product not supported";
    case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN: return "Positional join not supported";
    case LogicalOperatorType::LOGICAL_UNION:
    case LogicalOperatorType::LOGICAL_EXCEPT:
    case LogicalOperatorType::LOGICAL_INTERSECT: return "Set operation not supported";
    case LogicalOperatorType::LOGICAL_INSERT: return "Insert not supported";
    case LogicalOperatorType::LOGICAL_DELETE: return "Delete not supported";
    case LogicalOperatorType::LOGICAL_UPDATE: return "Update not supported";
    case LogicalOperatorType::LOGICAL_CREATE_TABLE: return "Create table not supported";
    case LogicalOperatorType::LOGICAL_CREATE_INDEX: return "Create index not supported";
    case LogicalOperatorType::LOGICAL_CREATE_SECRET: return "Create secret not supported";
    case LogicalOperatorType::LOGICAL_EXPLAIN: return "Explain not supported";
    case LogicalOperatorType::LOGICAL_DISTINCT: return "Distinct not supported";
    case LogicalOperatorType::LOGICAL_PREPARE: return "Prepare not supported";
    case LogicalOperatorType::LOGICAL_EXECUTE: return "Execute not supported";
    case LogicalOperatorType::LOGICAL_CREATE_VIEW:
    case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
    case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
    case LogicalOperatorType::LOGICAL_CREATE_MACRO:
    case LogicalOperatorType::LOGICAL_CREATE_TYPE: return "Create not supported";
    case LogicalOperatorType::LOGICAL_PRAGMA: return "Pragma not supported";
    case LogicalOperatorType::LOGICAL_VACUUM: return "Vacuum not supported";
    case LogicalOperatorType::LOGICAL_TRANSACTION:
    case LogicalOperatorType::LOGICAL_ALTER:
    case LogicalOperatorType::LOGICAL_DROP:
    case LogicalOperatorType::LOGICAL_LOAD:
    case LogicalOperatorType::LOGICAL_ATTACH:
    case LogicalOperatorType::LOGICAL_DETACH: return "Simple not supported";
    case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: return "Recursive CTE not supported";
    case LogicalOperatorType::LOGICAL_EXPORT: return "Export not supported";
    case LogicalOperatorType::LOGICAL_SET: return "Set not supported";
    case LogicalOperatorType::LOGICAL_RESET: return "Reset not supported";
    case LogicalOperatorType::LOGICAL_PIVOT: return "Pivot not supported";
    case LogicalOperatorType::LOGICAL_COPY_DATABASE: return "Copy database not supported";
    case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS: return "Update extensions not supported";
    case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR: return "Extension operator not supported";
    default: return "Unimplemented logical operator type";
  }
}

// ============================================================================
// Custom table function: gpu_plan_check(query VARCHAR)
// Returns: (success BOOLEAN, error_message VARCHAR)
//
// Checks whether a query's logical plan uses only operators supported by
// the legacy GPUPhysicalPlanGenerator, without constructing GPU operators.
// ============================================================================

unique_ptr<FunctionData> GPUPlanCheckBind(ClientContext& context,
                                          TableFunctionBindInput& input,
                                          vector<LogicalType>& return_types,
                                          vector<string>& names)
{
  auto result  = make_uniq<PlanCheckBindData>();
  string query = input.inputs[0].ToString();

  // Save and modify disabled optimizers (same as Sirius extension)
  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  try {
    // 1. Parse
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);

    if (parser.statements.empty()) { throw std::runtime_error("No statements parsed"); }

    // 2. Plan (creates LogicalOperator tree)
    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));

    if (!planner.plan) { throw std::runtime_error("Planner produced null plan"); }

    auto plan = std::move(planner.plan);

    // 3. Optimize
    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    // 4. Resolve types and column bindings (same as CreatePlan does internally)
    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*plan);
    resolver.VisitOperator(*plan);

    // 5. Check operator support against the legacy GPU planner dispatch table
    auto err = find_unsupported_gpu_operator(*plan);
    if (!err.empty()) { throw NotImplementedException(err); }

    result->success = true;
  } catch (const std::exception& e) {
    result->success       = false;
    result->error_message = e.what();
  }

  // Restore disabled optimizers
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;

  // Output schema
  return_types.push_back(LogicalType::BOOLEAN);
  names.push_back("success");
  return_types.push_back(LogicalType::VARCHAR);
  names.push_back("error_message");

  return std::move(result);
}

void PlanCheckFunction(ClientContext& context, TableFunctionInput& data_p, DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<PlanCheckBindData>();
  if (data.finished) {
    output.SetCardinality(0);
    return;
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(data.success));
  output.SetValue(1, 0, Value(data.error_message));
  data.finished = true;
}

// ============================================================================
// Test fixture
// ============================================================================

class TpcDsPlanTranslationFixture {
 public:
  TpcDsPlanTranslationFixture()
  {
    db  = std::make_unique<DuckDB>(nullptr);
    con = std::make_unique<Connection>(*db);

    auto& catalog    = Catalog::GetSystemCatalog(*db->instance);
    auto transaction = CatalogTransaction::GetSystemTransaction(*db->instance);

    // Register the Sirius (new) plan check function
    TableFunction plan_check(
      "sirius_plan_check", {LogicalType::VARCHAR}, PlanCheckFunction, PlanCheckBind);
    CreateTableFunctionInfo info(plan_check);
    catalog.CreateTableFunction(transaction, info);

    // Register the GPU (legacy) plan check function
    TableFunction gpu_plan_check(
      "gpu_plan_check", {LogicalType::VARCHAR}, PlanCheckFunction, GPUPlanCheckBind);
    CreateTableFunctionInfo gpu_info(gpu_plan_check);
    catalog.CreateTableFunction(transaction, gpu_info);
  }

  bool load_tpcds()
  {
    auto result = con->Query("INSTALL tpcds");
    if (result->HasError()) {
      std::cout << "WARNING: Could not install tpcds extension: " << result->GetError()
                << std::endl;
      return false;
    }
    result = con->Query("LOAD tpcds");
    if (result->HasError()) {
      std::cout << "WARNING: Could not load tpcds extension: " << result->GetError() << std::endl;
      return false;
    }
    result = con->Query("CALL dsdgen(sf=0.01)");
    if (result->HasError()) {
      std::cout << "WARNING: Could not generate TPC-DS schema: " << result->GetError() << std::endl;
      return false;
    }
    return true;
  }

  struct QueryInfo {
    int query_nr;
    std::string query_text;
  };

  std::vector<QueryInfo> get_queries()
  {
    std::vector<QueryInfo> queries;
    auto result = con->Query("SELECT query_nr, query FROM tpcds_queries() ORDER BY query_nr");
    if (result->HasError()) { return queries; }

    while (true) {
      auto chunk = result->Fetch();
      if (!chunk || chunk->size() == 0) break;
      for (idx_t r = 0; r < chunk->size(); r++) {
        QueryInfo qi;
        qi.query_nr   = chunk->GetValue(0, r).GetValue<int32_t>();
        qi.query_text = chunk->GetValue(1, r).ToString();
        queries.push_back(std::move(qi));
      }
    }
    return queries;
  }

  void run_plan_translation_test(const std::string& check_function_name,
                                 const std::string& summary_title)
  {
    auto queries = get_queries();
    REQUIRE(queries.size() > 0);

    int passed = 0;
    int failed = 0;
    std::vector<std::pair<int, std::string>> failures;

    for (const auto& qi : queries) {
      // Escape single quotes for SQL string literal embedding
      std::string escaped = qi.query_text;
      size_t pos          = 0;
      while ((pos = escaped.find('\'', pos)) != std::string::npos) {
        escaped.insert(pos, "'");
        pos += 2;
      }

      auto sql =
        "SELECT success, error_message FROM " + check_function_name + "('" + escaped + "')";
      auto result = con->Query(sql);

      REQUIRE(result);
      if (result->HasError()) {
        failed++;
        failures.push_back({qi.query_nr, result->GetError()});
        std::cout << "  FAIL: Q" << qi.query_nr << " - query error: " << result->GetError()
                  << std::endl;
        continue;
      }

      auto chunk = result->Fetch();
      REQUIRE(chunk);
      REQUIRE(chunk->size() == 1);

      bool success    = chunk->GetValue(0, 0).GetValue<bool>();
      auto error_text = chunk->GetValue(1, 0).ToString();

      if (success) {
        passed++;
        std::cout << "  PASS: Q" << qi.query_nr << std::endl;
      } else {
        failed++;
        // Truncate long error messages
        if (error_text.size() > 150) { error_text = error_text.substr(0, 150) + "..."; }
        failures.push_back({qi.query_nr, error_text});
        std::cout << "  FAIL: Q" << qi.query_nr << " - " << error_text << std::endl;
      }
    }

    // Print summary
    auto total = queries.size();
    std::cout << "\n========================================" << std::endl;
    std::cout << summary_title << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Total:  " << total << std::endl;
    std::cout << "  Passed: " << passed << " (" << (100 * passed / total) << "%)" << std::endl;
    std::cout << "  Failed: " << failed << " (" << (100 * failed / total) << "%)" << std::endl;

    if (!failures.empty()) {
      // Group failures by error type
      std::map<std::string, std::vector<int>> error_groups;
      for (const auto& [nr, err] : failures) {
        // Normalize: take first 80 chars for grouping
        auto key = err.substr(0, std::min(err.size(), size_t(80)));
        error_groups[key].push_back(nr);
      }

      std::cout << "\nFailures by error type:" << std::endl;
      for (const auto& [err, nrs] : error_groups) {
        std::cout << "  [" << nrs.size() << " queries] " << err << std::endl;
        std::cout << "    ";
        for (size_t i = 0; i < nrs.size(); i++) {
          if (i > 0) std::cout << ", ";
          std::cout << "Q" << nrs[i];
        }
        std::cout << std::endl;
      }
    }

    std::cout << "========================================" << std::endl;
  }

  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // anonymous namespace

// ============================================================================
// Test case
// ============================================================================

TEST_CASE_METHOD(TpcDsPlanTranslationFixture,
                 "TPC-DS plan translation coverage",
                 "[tpcds][plan][coverage]")
{
  if (!load_tpcds()) {
    WARN("TPC-DS extension not available — skipping plan translation test");
    return;
  }

  run_plan_translation_test("sirius_plan_check", "TPC-DS Plan Translation Summary (Sirius)");

  // The test passes as long as it runs — the summary shows coverage.
  // If you want to enforce a minimum number of passing queries, uncomment:
  // REQUIRE(passed >= 50);
}

TEST_CASE_METHOD(TpcDsPlanTranslationFixture,
                 "TPC-DS legacy plan translation coverage",
                 "[tpcds][plan][coverage][legacy]")
{
  if (!load_tpcds()) {
    WARN("TPC-DS extension not available — skipping legacy plan translation test");
    return;
  }

  run_plan_translation_test("gpu_plan_check", "TPC-DS Plan Translation Summary (Legacy GPU)");
}
