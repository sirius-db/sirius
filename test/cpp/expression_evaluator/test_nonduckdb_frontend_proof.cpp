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

// Non-DuckDB frontend proof (PROOF-01, #704 / acceptance proof for #666).
//
// This test is the authoritative evidence that Sirius's expression IR is fully
// decoupled from the DuckDB frontend: it constructs a sirius::ast::node tree BY
// HAND (no DuckDB Bound*Expression, no sirius::ast::from_duckdb), runs it through
// expression_evaluator against a small cuDF table built by hand, and asserts
// the final column values against a hand-computed expected column.
//
// The test deliberately includes NO duckdb/ header in its own body. A self-reading
// grep-guard TEST_CASE below enforces that invariant. (catch.hpp and the AST
// builder helpers pull in a duckdb std-alias header transitively — that is test
// plumbing for the vendored Catch framework, not the expression frontend, and is
// not a direct include of this file.)

// test helpers — sirius::ast::node builders + GPU->host readback (no DuckDB
// expression types); also establishes duckdb_base_std for the vendored catch.hpp.
#include "ast_test_support.hpp"

#include <catch.hpp>

// cudf — build the input table and inspect the output column by hand.
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

// sirius — the native expression IR + executor under test.
#include <expression/function_id.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/expression_evaluator_strategy.hpp>

// cuda / standard library
#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace sirius::expr_test;

namespace {

// Build a single-column INT32 input table from host values. Pure cuDF — no
// cucascade data_batch / memory_space, no DuckDB.
std::unique_ptr<cudf::table> make_int32_table(std::vector<int32_t> const& values)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto n      = static_cast<cudf::size_type>(values.size());

  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

// Build the projection AST by hand: cast( reference(col0) + constant(10), BIGINT ).
// Exercises all four node kinds required by PROOF-01 — reference, constant,
// binary function, cast — with no DuckDB expression types anywhere in the path.
// Returns a fresh owning vector each call: the executor borrows node.get()
// pointers, so the vector must outlive each evaluate(), and it is consumed (moved)
// into a single executor per run.
duckdb::vector<std::unique_ptr<ast_node>> build_cast_add_projection()
{
  std::vector<std::unique_ptr<ast_node>> add_args;
  add_args.push_back(make_ref(0));         // reference: input column 0
  add_args.push_back(make_int_const(10));  // constant: INTEGER 10

  auto add = make_func(  // binary function: col0 + 10
    sirius::function_id::add,
    std::move(add_args),
    sirius::logical_type::make(sirius::type_id::INTEGER));

  auto cast = make_cast(  // cast the INTEGER result up to BIGINT
    std::move(add),
    sirius::logical_type::make(sirius::type_id::BIGINT),
    /*try_cast=*/false);

  duckdb::vector<std::unique_ptr<ast_node>> exprs;
  exprs.push_back(std::move(cast));
  return exprs;
}

}  // namespace

TEST_CASE("nonduckdb proof - hand-built AST projection executes end-to-end",
          "[nonduckdb_frontend_proof]")
{
  std::vector<int32_t> const in_values{1, 7, 42, 100, -5, 0, 123};
  auto input = make_int32_table(in_values);

  std::vector<int64_t> expected;
  expected.reserve(in_values.size());
  for (auto v : in_values) {
    expected.push_back(static_cast<int64_t>(v) + 10);
  }

  // The proof must hold regardless of how the executor evaluates the tree.
  for (auto strategy : {sirius::expression_evaluator_strategy::MATERIALIZE,
                        sirius::expression_evaluator_strategy::AST_INTERPRET}) {
    auto exprs = build_cast_add_projection();
    sirius::expression_evaluator executor(
      exprs, cudf::get_current_device_resource_ref(), cudf::get_default_stream(), strategy);

    auto output = executor.evaluate(input->view());

    REQUIRE(output->num_columns() == 1);
    REQUIRE(output->view().column(0).type().id() == cudf::type_id::INT64);
    REQUIRE(output->num_rows() == static_cast<cudf::size_type>(in_values.size()));

    auto got = copy_column_to_host<int64_t>(output->view().column(0));
    REQUIRE(got == expected);
  }
}

TEST_CASE("nonduckdb proof - test source includes no duckdb headers", "[nonduckdb_frontend_proof]")
{
  // Grep guard: this file must not directly include any duckdb/ header, proving
  // the AST tree above is constructed and executed without the DuckDB frontend.
  std::filesystem::path const self =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) /
    "test/cpp/expression_evaluator/test_nonduckdb_frontend_proof.cpp";

  std::ifstream in(self);
  REQUIRE(in.is_open());

  int direct_duckdb_includes = 0;
  std::string line;
  while (std::getline(in, line)) {
    auto const start = line.find_first_not_of(" \t");
    if (start == std::string::npos) { continue; }
    // An #include line that names a duckdb/ path is the only thing we reject.
    // (The matcher itself lives on non-#include lines, so it never self-triggers.)
    if (line.compare(start, 8, "#include") == 0 && line.find("duckdb/") != std::string::npos) {
      ++direct_duckdb_includes;
    }
  }

  INFO("PROOF-01: this test file must directly include zero duckdb/ headers");
  REQUIRE(direct_duckdb_includes == 0);
}
