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

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <sirius/udf.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// All functions and datasets in this file are independently authored synthetic examples.
template <typename T>
void cpu_add(duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output)
{
  for (duckdb::idx_t i = 0; i < args.size(); ++i) {
    auto const left  = args.GetValue(0, i);
    auto const right = args.GetValue(1, i);
    output.SetValue(i,
                    left.IsNull() || right.IsNull()
                      ? duckdb::Value(output.GetType())
                      : duckdb::Value::CreateValue<T>(left.GetValue<T>() + right.GetValue<T>()));
  }
}

void cpu_seventeen(duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output)
{
  for (duckdb::idx_t i = 0; i < args.size(); ++i) {
    output.SetValue(i, duckdb::Value::INTEGER(17));
  }
}

void cpu_ascii_upper(duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output)
{
  for (duckdb::idx_t i = 0; i < args.size(); ++i) {
    auto const value = args.GetValue(0, i);
    if (value.IsNull()) {
      output.SetValue(i, duckdb::Value(duckdb::LogicalType::VARCHAR));
      continue;
    }
    auto text = value.GetValue<std::string>();
    for (auto& ch : text) {
      if (ch >= 'a' && ch <= 'z') { ch = static_cast<char>(ch - 'a' + 'A'); }
    }
    output.SetValue(i, duckdb::Value(text));
  }
}

class GpuUdfFixture : public sirius::test::GpuExecutionFixture {
 public:
  std::string name(std::string const& suffix) const { return attach_alias + "_" + suffix; }

  void initialize(char const* strategy)
  {
    run_ok("SET gpu_execution = false");
    run_ok("SET enable_duckdb_fallback = false");
    run_ok(std::string("SET expression_evaluator_strategy = '") + strategy + "'");
    duckdb::ExtensionLoader loader(duckdb::DatabaseInstance::GetDatabase(*con->context),
                                   name("extension"));
    auto add_binder = [](duckdb::BoundFunctionExpression const& expression) {
      auto const type = expression.GetReturnType().id() == duckdb::LogicalTypeId::INTEGER
                          ? cudf::type_id::INT32
                          : cudf::type_id::INT64;
      return std::make_shared<sirius::gpu_scalar_function>(
        sirius::gpu_scalar_function{[type](std::span<cudf::column_view const> args,
                                           cudf::size_type,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) {
          if (args[0].type() != cudf::data_type{type} || args[1].type() != cudf::data_type{type}) {
            throw std::runtime_error("native callback did not receive its declared input types");
          }
          return cudf::binary_operation(
            args[0], args[1], cudf::binary_operator::ADD, cudf::data_type{type}, stream, mr);
        }});
    };
    sirius::register_gpu_scalar_function(
      loader,
      duckdb::ScalarFunction(duckdb::Identifier(name("add")),
                             {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER},
                             duckdb::LogicalType::INTEGER,
                             cpu_add<int32_t>),
      add_binder);
    sirius::register_gpu_scalar_function(
      loader,
      duckdb::ScalarFunction(duckdb::Identifier(name("add")),
                             {duckdb::LogicalType::BIGINT, duckdb::LogicalType::BIGINT},
                             duckdb::LogicalType::BIGINT,
                             cpu_add<int64_t>),
      add_binder);

    sirius::register_gpu_scalar_function(
      loader,
      duckdb::ScalarFunction(duckdb::Identifier(name("specialized")),
                             {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER},
                             duckdb::LogicalType::INTEGER,
                             cpu_add<int32_t>),
      [](duckdb::BoundFunctionExpression const& expression)
        -> std::shared_ptr<sirius::gpu_scalar_function const> {
        auto const& constant = *expression.GetChildren()[1];
        if (constant.GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) {
          return nullptr;
        }
        auto const& value = constant.Cast<duckdb::BoundConstantExpression>().GetValue();
        if (value.IsNull()) { return nullptr; }
        int32_t const offset = value.GetValue<int32_t>();
        return std::make_shared<sirius::gpu_scalar_function>(
          sirius::gpu_scalar_function{[offset](std::span<cudf::column_view const> args,
                                               cudf::size_type,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr) {
            cudf::numeric_scalar<int32_t> const scalar{offset, true, stream, mr};
            return cudf::binary_operation(args[0],
                                          scalar,
                                          cudf::binary_operator::ADD,
                                          cudf::data_type{cudf::type_id::INT32},
                                          stream,
                                          mr);
          }});
      });

    sirius::register_gpu_scalar_function(
      loader,
      duckdb::ScalarFunction(duckdb::Identifier(name("binder_throws")),
                             {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER},
                             duckdb::LogicalType::INTEGER,
                             cpu_add<int32_t>),
      [](duckdb::BoundFunctionExpression const&)
        -> std::shared_ptr<sirius::gpu_scalar_function const> {
        throw duckdb::InvalidInputException("synthetic GPU binder failure");
      });

    duckdb::ScalarFunction special{duckdb::Identifier(name("seventeen")),
                                   {duckdb::LogicalType::INTEGER},
                                   duckdb::LogicalType::INTEGER,
                                   cpu_seventeen};
    special.SetNullHandling(duckdb::FunctionNullHandling::SPECIAL_HANDLING);
    sirius::register_gpu_scalar_function(
      loader, std::move(special), [](duckdb::BoundFunctionExpression const&) {
        return std::make_shared<sirius::gpu_scalar_function>(
          sirius::gpu_scalar_function{[](std::span<cudf::column_view const>,
                                         cudf::size_type rows,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr) {
            cudf::numeric_scalar<int32_t> const scalar{17, true, stream, mr};
            return cudf::make_column_from_scalar(scalar, rows, stream, mr);
          }});
      });
    sirius::register_gpu_scalar_function(
      loader,
      duckdb::ScalarFunction(duckdb::Identifier(name("ascii_upper")),
                             {duckdb::LogicalType::VARCHAR},
                             duckdb::LogicalType::VARCHAR,
                             cpu_ascii_upper),
      [](duckdb::BoundFunctionExpression const&) {
        return std::make_shared<sirius::gpu_scalar_function>(
          sirius::gpu_scalar_function{[](std::span<cudf::column_view const> args,
                                         cudf::size_type,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr) {
            return cudf::strings::to_upper(cudf::strings_column_view{args[0]}, stream, mr);
          }});
      });

    run_ok("CREATE TABLE udf_inputs(id INTEGER, a INTEGER, b INTEGER, txt VARCHAR)");
    run_ok(
      "INSERT INTO udf_inputs VALUES (1, 3, 7, 'a.B9'), (2, -4, 2, ''),"
      " (3, NULL, 5, NULL), (4, 8, NULL, 'NULL'), (5, 0, 6, 'mixed Case')");
    run_ok("CHECKPOINT");
  }
};
}  // namespace

TEST_CASE_METHOD(GpuUdfFixture,
                 "registered GPU UDFs compose in projections filters and overloads",
                 "[integration][gpu_execution][gpu_udf]")
{
  initialize(GENERATE("materialize", "ast_interpret", "ast_jit"));
  compare_gpu_candidate_to_cpu("SELECT id, (a + b) * 2 + 1, upper(txt) FROM udf_inputs",
                               "SELECT id, " + name("add") + "(a, b) * 2 + 1, " +
                                 name("ascii_upper") + "(txt) FROM udf_inputs");
  compare_gpu_candidate_to_cpu("SELECT id FROM udf_inputs WHERE a + b > 5",
                               "SELECT id FROM udf_inputs WHERE " + name("add") + "(a, b) > 5");
  compare_gpu_vs_cpu("SELECT id, " + name("add") + "(a::BIGINT, b::BIGINT), " + name("add") + "(" +
                     name("add") + "(a, 2), 3) FROM udf_inputs");
  // cuDF's length/year primitives produce INT32/INT16; this callback requires SQL BIGINTs.
  compare_gpu_candidate_to_cpu(
    "SELECT id, length(txt) + year(DATE '2000-01-01' + a) FROM udf_inputs",
    "SELECT id, " + name("add") + "(length(txt), year(DATE '2000-01-01' + a)) FROM udf_inputs");
  compare_gpu_candidate_to_cpu("SELECT id, 17::INTEGER FROM udf_inputs",
                               "SELECT id, " + name("seventeen") + "(a) FROM udf_inputs");
}

TEST_CASE_METHOD(GpuUdfFixture,
                 "GPU UDF constant specialization and prepared plans retain bindings",
                 "[integration][gpu_execution][gpu_udf]")
{
  initialize(GENERATE("materialize", "ast_interpret", "ast_jit"));
  compare_gpu_candidate_to_cpu("SELECT id, a + 9 FROM udf_inputs",
                               "SELECT id, " + name("specialized") + "(a, 9) FROM udf_inputs");
  run_ok("PREPARE prepared_udf AS SELECT id, " + name("add") + "(a, b) FROM udf_inputs");
  compare_gpu_candidate_to_cpu("SELECT id, a + b FROM udf_inputs", "EXECUTE prepared_udf");
  compare_gpu_candidate_to_cpu("SELECT id, a + b FROM udf_inputs", "EXECUTE prepared_udf");
  run_ok("DEALLOCATE prepared_udf");
}

TEST_CASE_METHOD(GpuUdfFixture,
                 "unsupported GPU UDF binding uses CPU plan fallback",
                 "[integration][gpu_execution][gpu_udf]")
{
  initialize("ast_interpret");
  run_ok("SET enable_duckdb_fallback = true");
  expect_plan_fallback_matches_cpu("SELECT id, " + name("specialized") + "(a, b) FROM udf_inputs");
}

TEST_CASE_METHOD(GpuUdfFixture,
                 "SQL macro rewrite uses ordinary GPU expressions",
                 "[integration][gpu_execution][gpu_udf]")
{
  initialize(GENERATE("materialize", "ast_interpret", "ast_jit"));
  run_ok("CREATE MACRO synthetic_affine(x) AS x * 3 + 4");
  compare_gpu_candidate_to_cpu("SELECT id, a * 3 + 4 FROM udf_inputs",
                               "SELECT id, synthetic_affine(a) FROM udf_inputs");
}

TEST_CASE_METHOD(GpuUdfFixture,
                 "GPU UDF binder errors follow fallback policy and keep the connection usable",
                 "[integration][gpu_execution][gpu_udf]")
{
  initialize("ast_interpret");
  auto const query = "SELECT id, " + name("binder_throws") + "(a, b) FROM udf_inputs";
  run_ok("SET enable_duckdb_fallback = true");
  expect_plan_fallback_matches_cpu(query);
  run_ok("SET enable_duckdb_fallback = false");
  auto failed = con->Query(query);
  REQUIRE(failed);
  REQUIRE(failed->HasError());
  CHECK(failed->GetError().find("synthetic GPU binder failure") != std::string::npos);
  compare_gpu_candidate_to_cpu("SELECT id, a + b FROM udf_inputs",
                               "SELECT id, " + name("add") + "(a, b) FROM udf_inputs");
}

TEST_CASE_METHOD(GpuUdfFixture,
                 "SQL execution stats return fresh counters for repeated prepared reads",
                 "[integration][gpu_execution][gpu_udf]")
{
  initialize("ast_interpret");
  auto read_stats_on = [&](duckdb::Connection& source, std::string const& sql) {
    auto result = source.Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->RowCount() == 1);
    REQUIRE(result->GetTypes() ==
            duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::UBIGINT,
                                                duckdb::LogicalType::UBIGINT,
                                                duckdb::LogicalType::UBIGINT});
    auto rows = result->Collection().GetRows();
    return std::array<uint64_t, 3>{rows.GetValue(0, 0).GetValue<uint64_t>(),
                                   rows.GetValue(1, 0).GetValue<uint64_t>(),
                                   rows.GetValue(2, 0).GetValue<uint64_t>()};
  };
  auto read_stats = [&](std::string const& sql) { return read_stats_on(*con, sql); };
  duckdb::Connection observer{duckdb::DatabaseInstance::GetDatabase(*con->context)};
  auto observer_cpu = observer.Query("SET gpu_execution = false");
  REQUIRE(observer_cpu);
  REQUIRE_FALSE(observer_cpu->HasError());
  std::string const stats_sql =
    "SELECT executions, plan_fallbacks, runtime_fallbacks FROM sirius_execution_stats()";
  // CPU reads must neither increment GPU fallback counters nor reset the values.
  auto const before = read_stats(stats_sql);
  REQUIRE(read_stats(stats_sql) == before);
  run_ok("PREPARE prepared_stats AS " + stats_sql);
  REQUIRE(read_stats("EXECUTE prepared_stats") == before);

  run_ok("SET gpu_execution = true");
  run_ok("SELECT id, " + name("add") + "(a, b) FROM udf_inputs");
  run_ok("SET gpu_execution = false");
  auto const after = read_stats("EXECUTE prepared_stats");
  CHECK(after[0] == before[0] + 1);
  CHECK(after[1] == before[1]);
  CHECK(after[2] == before[2]);
  // Connections in this database observe the same Sirius runtime counters.
  REQUIRE(read_stats_on(observer, stats_sql) == after);
  REQUIRE(read_stats(stats_sql) == after);
  REQUIRE(read_stats("EXECUTE prepared_stats") == after);
  run_ok("DEALLOCATE prepared_stats");
}

TEST_CASE_METHOD(GpuUdfFixture,
                 "GPU UDF identity preserves every supported scalar input type",
                 "[integration][gpu_execution][gpu_udf]")
{
  run_ok("SET gpu_execution = false");
  run_ok("SET enable_duckdb_fallback = false");
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok(std::string("SET expression_evaluator_strategy = '") + strategy + "'");
  struct type_case {
    duckdb::LogicalType logical;
    std::string sql;
    cudf::type_id physical;
    std::string rows;
  };
  std::string const timestamps =
    "('1969-12-31 23:59:59.123456789'), ('2001-02-03 04:05:06.987654321'), (NULL)";
  std::vector<type_case> const types{
    {duckdb::LogicalType::BOOLEAN, "BOOLEAN", cudf::type_id::BOOL8, "(true), (false), (NULL)"},
    {duckdb::LogicalType::TINYINT, "TINYINT", cudf::type_id::INT8, "(-128), (127), (NULL)"},
    {duckdb::LogicalType::SMALLINT, "SMALLINT", cudf::type_id::INT16, "(-32768), (32767), (NULL)"},
    {duckdb::LogicalType::INTEGER,
     "INTEGER",
     cudf::type_id::INT32,
     "(-2147483648), (2147483647), (NULL)"},
    {duckdb::LogicalType::BIGINT,
     "BIGINT",
     cudf::type_id::INT64,
     "(-9223372036854775808), (9223372036854775807), (NULL)"},
    {duckdb::LogicalType::UTINYINT, "UTINYINT", cudf::type_id::UINT8, "(0), (255), (NULL)"},
    {duckdb::LogicalType::USMALLINT, "USMALLINT", cudf::type_id::UINT16, "(0), (65535), (NULL)"},
    {duckdb::LogicalType::UINTEGER, "UINTEGER", cudf::type_id::UINT32, "(0), (4294967295), (NULL)"},
    {duckdb::LogicalType::UBIGINT,
     "UBIGINT",
     cudf::type_id::UINT64,
     "(0), (18446744073709551615), (NULL)"},
    {duckdb::LogicalType::FLOAT,
     "FLOAT",
     cudf::type_id::FLOAT32,
     "(-123.25), (0.5), ('-0.0'), ('nan'), (NULL)"},
    {duckdb::LogicalType::DOUBLE,
     "DOUBLE",
     cudf::type_id::FLOAT64,
     "(-123.25), (0.5), ('-0.0'), ('inf'), (NULL)"},
    {duckdb::LogicalType::VARCHAR,
     "VARCHAR",
     cudf::type_id::STRING,
     "(''), ('NULL'), ('mixed Case'), (NULL)"},
    {duckdb::LogicalType::DATE,
     "DATE",
     cudf::type_id::TIMESTAMP_DAYS,
     "('1969-12-31'), ('2000-02-29'), (NULL)"},
    {duckdb::LogicalType::TIMESTAMP_S, "TIMESTAMP_S", cudf::type_id::TIMESTAMP_SECONDS, timestamps},
    {duckdb::LogicalType::TIMESTAMP_MS,
     "TIMESTAMP_MS",
     cudf::type_id::TIMESTAMP_MILLISECONDS,
     timestamps},
    {duckdb::LogicalType::TIMESTAMP,
     "TIMESTAMP",
     cudf::type_id::TIMESTAMP_MICROSECONDS,
     timestamps},
    {duckdb::LogicalType::TIMESTAMP_NS,
     "TIMESTAMP_NS",
     cudf::type_id::TIMESTAMP_NANOSECONDS,
     timestamps}};
  duckdb::ExtensionLoader loader(duckdb::DatabaseInstance::GetDatabase(*con->context),
                                 name("identity_extension"));
  for (auto const& type : types) {
    sirius::register_gpu_scalar_function(
      loader,
      duckdb::ScalarFunction(
        duckdb::Identifier(name("identity")),
        {type.logical},
        type.logical,
        [](duckdb::DataChunk& args, duckdb::ExpressionState&, duckdb::Vector& output) {
          for (duckdb::idx_t row = 0; row < args.size(); ++row) {
            output.SetValue(row, args.GetValue(0, row));
          }
        }),
      [expected = type.physical](duckdb::BoundFunctionExpression const&) {
        return std::make_shared<sirius::gpu_scalar_function>(
          sirius::gpu_scalar_function{[expected](std::span<cudf::column_view const> args,
                                                 cudf::size_type,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr) {
            if (args[0].type() != cudf::data_type{expected}) {
              throw std::runtime_error("identity callback received the wrong physical input type");
            }
            return std::make_unique<cudf::column>(args[0], stream, mr);
          }});
      });
  }
  for (std::size_t index = 0; index < types.size(); ++index) {
    auto const& type = types[index];
    CAPTURE(type.sql);
    auto const table = "identity_input_" + std::to_string(index);
    run_ok("SET gpu_execution = false");
    run_ok("CREATE TABLE " + table + "(v " + type.sql + ")");
    run_ok("INSERT INTO " + table + " VALUES " + type.rows);
    run_ok("CHECKPOINT");
    compare_gpu_candidate_to_cpu("SELECT v FROM " + table,
                                 "SELECT " + name("identity") + "(v) FROM " + table);
  }
}
