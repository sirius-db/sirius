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
 * @file test_lance_vector_search_routing.cpp
 * @brief Planner tests for routing DuckDB Lance vector-search table functions through Sirius.
 */

#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/common/types/decimal.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>

using namespace duckdb;

namespace {

duckdb::unique_ptr<sirius::op::sirius_physical_operator> generate_sirius_plan(
  Connection& con, const std::string& query)
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
  try {
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);

    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*plan);
    resolver.VisitOperator(*plan);

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
  } catch (duckdb::InternalException&) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    return nullptr;
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

sirius::op::sirius_physical_table_scan* find_table_scan(sirius::op::sirius_physical_operator* root)
{
  if (!root) { return nullptr; }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    return &root->Cast<sirius::op::sirius_physical_table_scan>();
  }
  for (auto& child : root->children) {
    auto* found = find_table_scan(child.get());
    if (found) { return found; }
  }
  return nullptr;
}

void require_not_implemented_message(Connection& con,
                                     const std::string& query,
                                     std::initializer_list<std::string> expected_fragments)
{
  try {
    auto plan = generate_sirius_plan(con, query);
    (void)plan;
  } catch (const duckdb::NotImplementedException& ex) {
    const std::string message = ex.what();
    for (const auto& fragment : expected_fragments) {
      INFO("exception message: " << message);
      REQUIRE(message.find(fragment) != std::string::npos);
    }
    return;
  }
  FAIL("expected duckdb::NotImplementedException");
}

LogicalType list_float_type() { return LogicalType::LIST(LogicalType::FLOAT); }

LogicalType array_float_type() { return LogicalType::ARRAY(LogicalType::FLOAT, optional_idx(3)); }

LogicalType struct_type()
{
  return LogicalType::STRUCT(
    child_list_t<LogicalType>{{"label", LogicalType::VARCHAR}, {"weight", LogicalType::DOUBLE}});
}

void add_common_candidate_columns(vector<LogicalType>& return_types, vector<string>& names)
{
  return_types.push_back(LogicalType::BIGINT);
  names.push_back("_rowid");
  return_types.push_back(LogicalType::DOUBLE);
  names.push_back("_distance");
  return_types.push_back(LogicalType::DOUBLE);
  names.push_back("_score");
  return_types.push_back(LogicalType::INTEGER);
  names.push_back("doc_id");
}

void add_schema_for_scenario(const std::string& scenario,
                             vector<LogicalType>& return_types,
                             vector<string>& names)
{
  add_common_candidate_columns(return_types, names);

  if (scenario == "scalar") {
    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("title");
    return_types.push_back(LogicalType::DOUBLE);
    names.push_back("rank_feature");
  } else if (scenario == "nested") {
    return_types.push_back(list_float_type());
    names.push_back("embedding");
    return_types.push_back(array_float_type());
    names.push_back("embedding_array");
    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("title");
  } else if (scenario == "struct") {
    return_types.push_back(struct_type());
    names.push_back("attrs");
  } else if (scenario == "wide_numeric") {
    return_types.push_back(LogicalType::HUGEINT);
    names.push_back("huge_value");
    return_types.push_back(LogicalType::UHUGEINT);
    names.push_back("uhuge_value");
    return_types.push_back(LogicalType::DECIMAL(19, 2));
    names.push_back("wide_decimal");
    return_types.push_back(LogicalType::DECIMAL(18, 2));
    names.push_back("ok_decimal");
  } else if (scenario == "mixed") {
    return_types.push_back(list_float_type());
    names.push_back("embedding");
    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("title");
    return_types.push_back(LogicalType::DATE);
    names.push_back("created_date");
    return_types.push_back(LogicalType::TIMESTAMP);
    names.push_back("created_ts");
    return_types.push_back(LogicalType::DECIMAL(18, 2));
    names.push_back("ok_decimal");
  } else if (scenario == "pushdown_nested") {
    return_types.push_back(array_float_type());
    names.push_back("embedding_array");
    return_types.push_back(list_float_type());
    names.push_back("embedding_list");
    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("title");
    return_types.push_back(LogicalType::DOUBLE);
    names.push_back("rank_feature");
  } else {
    throw InvalidInputException("unknown fake lance_vector_search scenario: {}", scenario);
  }
}

struct FakeLanceBindData : TableFunctionData {
  unique_ptr<FunctionData> Copy() const override
  {
    throw InternalException("Copy not supported for TableFunctionData");
  }

  bool Equals(const FunctionData&) const override { return true; }
};

unique_ptr<FunctionData> FakeLanceVectorSearchBind(ClientContext&,
                                                   TableFunctionBindInput& input,
                                                   vector<LogicalType>& return_types,
                                                   vector<string>& names)
{
  const auto scenario =
    input.inputs.empty() ? std::string("scalar") : input.inputs[0].GetValue<std::string>();
  add_schema_for_scenario(scenario, return_types, names);
  return make_uniq<FakeLanceBindData>();
}

void FakeLanceVectorSearchFunction(ClientContext&, TableFunctionInput&, DataChunk& output)
{
  output.SetCardinality(0);
}

void register_fake_vector_search(Catalog& catalog,
                                 CatalogTransaction& transaction,
                                 const std::string& name)
{
  TableFunction function(
    name, {LogicalType::VARCHAR}, FakeLanceVectorSearchFunction, FakeLanceVectorSearchBind);
  function.projection_pushdown = true;
  CreateTableFunctionInfo info(function);
  catalog.CreateTableFunction(transaction, info);
}

struct lance_vector_search_fixture {
  lance_vector_search_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(nullptr);
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    auto& catalog    = Catalog::GetSystemCatalog(*db->instance);
    auto transaction = CatalogTransaction::GetSystemTransaction(*db->instance);
    register_fake_vector_search(catalog, transaction, "lance_vector_search");
    register_fake_vector_search(catalog, transaction, "not_lance_vector_search");
  }

  ~lance_vector_search_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - non-whitelisted function still falls back",
                 "[lance_vector_search][planner][isolated_context]")
{
  require_not_implemented_message(*con,
                                  "SELECT doc_id FROM not_lance_vector_search('scalar')",
                                  {"not_lance_vector_search", "not supported"});
}

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - scalar output plans as table scan",
                 "[lance_vector_search][planner][isolated_context]")
{
  auto plan = generate_sirius_plan(*con,
                                   "SELECT _rowid, _distance, _score, doc_id, title "
                                   "FROM lance_vector_search('scalar') WHERE doc_id > 10");
  REQUIRE(plan);

  auto* scan = find_table_scan(plan.get());
  REQUIRE(scan != nullptr);
  REQUIRE(scan->function.name == "lance_vector_search");
}

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - projected LIST and ARRAY columns are rejected",
                 "[lance_vector_search][planner][isolated_context]")
{
  require_not_implemented_message(
    *con, "SELECT embedding FROM lance_vector_search('nested')", {"embedding", "unsupported type"});
  require_not_implemented_message(*con,
                                  "SELECT embedding_array FROM lance_vector_search('nested')",
                                  {"embedding_array", "unsupported type"});
}

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - projected STRUCT columns are rejected",
                 "[lance_vector_search][planner][isolated_context]")
{
  require_not_implemented_message(
    *con, "SELECT attrs FROM lance_vector_search('struct')", {"attrs", "unsupported type"});
}

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - wide numeric guard keeps DECIMAL boundary",
                 "[lance_vector_search][planner][isolated_context]")
{
  REQUIRE_THROWS_AS(
    generate_sirius_plan(*con, "SELECT huge_value FROM lance_vector_search('wide_numeric')"),
    duckdb::NotImplementedException);
  REQUIRE_THROWS_AS(
    generate_sirius_plan(*con, "SELECT uhuge_value FROM lance_vector_search('wide_numeric')"),
    duckdb::NotImplementedException);
  require_not_implemented_message(*con,
                                  "SELECT wide_decimal FROM lance_vector_search('wide_numeric')",
                                  {"wide_decimal", "unsupported type"});

  auto plan =
    generate_sirius_plan(*con, "SELECT ok_decimal FROM lance_vector_search('wide_numeric')");
  REQUIRE(plan);
  REQUIRE(find_table_scan(plan.get()) != nullptr);
}

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - unsupported unprojected columns do not over-reject",
                 "[lance_vector_search][planner][isolated_context]")
{
  auto plan = generate_sirius_plan(*con,
                                   "SELECT doc_id, title, created_date, created_ts, ok_decimal "
                                   "FROM lance_vector_search('mixed')");
  REQUIRE(plan);

  auto* scan = find_table_scan(plan.get());
  REQUIRE(scan != nullptr);
  REQUIRE(scan->function.name == "lance_vector_search");
}

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - projection pushdown tolerates unprojected vectors",
                 "[lance_vector_search][planner][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con,
                         "SELECT doc_id, title, rank_feature "
                         "FROM lance_vector_search('pushdown_nested') WHERE rank_feature > 0.1");
  REQUIRE(plan);

  auto* scan = find_table_scan(plan.get());
  REQUIRE(scan != nullptr);
  REQUIRE(scan->function.name == "lance_vector_search");
}

TEST_CASE_METHOD(lance_vector_search_fixture,
                 "lance_vector_search routing - pipeline factory uses generic DuckDB scan",
                 "[lance_vector_search][planner][isolated_context]")
{
  auto plan = generate_sirius_plan(*con, "SELECT doc_id, title FROM lance_vector_search('scalar')");
  REQUIRE(plan);

  auto* scan = find_table_scan(plan.get());
  REQUIRE(scan != nullptr);
  REQUIRE(scan->function.name == "lance_vector_search");

  auto routed = sirius::pipeline::construct_sirius_specific_operator(*scan, nullptr);
  REQUIRE(routed);
  REQUIRE(routed->type == sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN);
  REQUIRE(routed->Cast<sirius::op::sirius_physical_duckdb_scan>().function.name ==
          "lance_vector_search");
}
