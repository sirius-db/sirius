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
 * @file test_fusion_matcher.cpp
 * @brief Expression-level and pipeline-level tests for the intra-pipeline fusion matcher,
 *        plus a TPC-H coverage probe that reports how many adjacent FILTER/PROJECTION runs
 *        the 22 canonical queries actually contain.
 */

#include "expression/ast/node.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/fusion_matcher.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using sirius::pipeline::ast_breaker;
using sirius::pipeline::find_ast_breaker;
using sirius::pipeline::is_ast_fusable;
using sirius::pipeline::match_fusable_chains;
using sirius::pipeline::pipeline_conversion_result;

namespace {

std::unique_ptr<sirius::ast::node> make_node(auto&& alt)
{
  return std::make_unique<sirius::ast::node>(std::forward<decltype(alt)>(alt));
}

std::unique_ptr<sirius::ast::node> col(uint32_t index, sirius::type_id id = sirius::type_id::BIGINT)
{
  return make_node(sirius::ast::reference{index, sirius::logical_type::make(id)});
}

//! `left < right` — the canonical ast-clean predicate shape.
std::unique_ptr<sirius::ast::node> comparison_of(std::unique_ptr<sirius::ast::node> left,
                                                 std::unique_ptr<sirius::ast::node> right)
{
  sirius::ast::comparison cmp;
  cmp.op    = sirius::comparison_type::lt;
  cmp.left  = std::move(left);
  cmp.right = std::move(right);
  return make_node(std::move(cmp));
}

fs::path integration_data_dir()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data";
#else
  return fs::path(__FILE__).parent_path().parent_path() / "integration/data";
#endif
}

}  // namespace

TEST_CASE("fusion matcher: leaves and arithmetic lower wholly into a cuDF AST", "[fusion_matcher]")
{
  CHECK(is_ast_fusable(*col(0)));
  CHECK(is_ast_fusable(*comparison_of(col(0), col(1))));

  sirius::ast::conjunction conj;
  conj.op = sirius::ast::conjunction::kind::op_and;
  conj.children.push_back(comparison_of(col(0), col(1)));
  conj.children.push_back(comparison_of(col(2), col(3)));
  CHECK(is_ast_fusable(sirius::ast::node{std::move(conj)}));

  sirius::ast::between btw;
  btw.input = col(0);
  btw.lower = col(1);
  btw.upper = col(2);
  CHECK(is_ast_fusable(sirius::ast::node{std::move(btw)}));
}

TEST_CASE("fusion matcher: breakers are reported by kind", "[fusion_matcher]")
{
  SECTION("CASE never lowers")
  {
    sirius::ast::case_expr::when_then branch;
    branch.when_ = comparison_of(col(0), col(1));
    branch.then_ = col(2);
    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back(std::move(branch));
    sirius::ast::node n{sirius::ast::case_expr{
      std::move(cases), col(3), sirius::logical_type::make(sirius::type_id::BIGINT)}};
    CHECK_FALSE(is_ast_fusable(n));
    CHECK(find_ast_breaker(n) == ast_breaker::case_expr);
  }

  SECTION("COALESCE never lowers")
  {
    std::vector<std::unique_ptr<sirius::ast::node>> children;
    children.push_back(col(0));
    children.push_back(col(1));
    sirius::ast::node n{sirius::ast::coalesce{std::move(children),
                                              sirius::logical_type::make(sirius::type_id::BIGINT)}};
    CHECK_FALSE(is_ast_fusable(n));
    CHECK(find_ast_breaker(n) == ast_breaker::coalesce);
  }

  SECTION("CAST to a supported target lowers; VARCHAR does not")
  {
    sirius::ast::cast supported;
    supported.child       = col(0, sirius::type_id::INTEGER);
    supported.target_type = sirius::logical_type::make(sirius::type_id::DOUBLE);
    supported.kind        = sirius::ast::cast_kind::semantic;
    CHECK(is_ast_fusable(sirius::ast::node{std::move(supported)}));

    sirius::ast::cast unsupported;
    unsupported.child       = col(0, sirius::type_id::INTEGER);
    unsupported.target_type = sirius::logical_type::make(sirius::type_id::VARCHAR);
    unsupported.kind        = sirius::ast::cast_kind::semantic;
    sirius::ast::node n{std::move(unsupported)};
    CHECK_FALSE(is_ast_fusable(n));
    CHECK(find_ast_breaker(n) == ast_breaker::unsupported_cast);
  }

  SECTION("a carrier restore must reach the materialize path")
  {
    sirius::ast::cast restore;
    restore.child       = col(0, sirius::type_id::INTEGER);
    restore.target_type = sirius::logical_type::make(sirius::type_id::BIGINT);
    restore.kind        = sirius::ast::cast_kind::carrier_restore;
    sirius::ast::node n{std::move(restore)};
    CHECK_FALSE(is_ast_fusable(n));
    CHECK(find_ast_breaker(n) == ast_breaker::carrier_cast);
  }

  SECTION("a breaker nested under a fusable parent still breaks the whole tree")
  {
    std::vector<std::unique_ptr<sirius::ast::node>> children;
    children.push_back(col(0));
    children.push_back(col(1));
    auto nested = make_node(sirius::ast::coalesce{
      std::move(children), sirius::logical_type::make(sirius::type_id::BIGINT)});
    auto n      = comparison_of(std::move(nested), col(2));
    CHECK_FALSE(is_ast_fusable(*n));
    CHECK(find_ast_breaker(*n) == ast_breaker::coalesce);
  }
}

//! Coverage probe: how many adjacent FILTER/PROJECTION runs do the canonical TPC-H queries
//! actually produce? Reports rather than gates — the numbers decide whether a fused
//! execution path is worth building at all.
TEST_CASE("fusion matcher: TPC-H chain coverage", "[integration][pipeline][fusion_matcher]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();

  auto db_path = integration_data_dir() / "duckdb/integration.duckdb";
  REQUIRE(fs::exists(db_path));
  auto r = con.Query("ATTACH IF NOT EXISTS '" + db_path.string() + "' AS tpch (READ_ONLY);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  r = con.Query("USE tpch;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  std::size_t total_chains       = 0;
  std::size_t ast_clean_chains   = 0;
  std::size_t queries_with_chain = 0;
  // Denominators: without these the chain count is uninterpretable — a low count could mean
  // "few FILTER/PROJECTION operators exist" or "they exist but are never adjacent".
  std::size_t total_filters     = 0;
  std::size_t total_projections = 0;

  for (int q = 1; q <= 22; ++q) {
    sirius::test::with_conversion_result(
      con, sirius::test::read_tpch_query_file(q), [&](pipeline_conversion_result& result) {
        auto const reports      = match_fusable_chains(result.scheduled_pipelines);
        std::size_t chains_here = 0;
        for (auto const& pipeline : result.scheduled_pipelines) {
          for (auto const& oper : pipeline->get_operators()) {
            if (oper.get().type == sirius::op::SiriusPhysicalOperatorType::FILTER) {
              total_filters++;
            } else if (oper.get().type == sirius::op::SiriusPhysicalOperatorType::PROJECTION) {
              total_projections++;
            }
          }
        }
        for (auto const& report : reports) {
          for (auto const& chain : report.chains) {
            // Structural invariants of every reported chain.
            INFO("q" << q << " pipeline " << report.pipeline_id);
            CHECK(chain.length() >= 2);
            CHECK(chain.end_index <= report.operator_count);
            CHECK(chain.filter_count + chain.projection_count == chain.length());
            chains_here++;
            if (chain.ast_clean) { ast_clean_chains++; }
          }
        }
        total_chains += chains_here;
        if (chains_here > 0) {
          queries_with_chain++;
          std::cout << "[fusion_matcher] q" << q << "\n"
                    << sirius::pipeline::render_fusion_report(reports);
        }
      });
  }

  std::cout << "[fusion_matcher] TPC-H total: " << total_chains << " candidate chain(s), "
            << ast_clean_chains << " ast-clean, in " << queries_with_chain << "/22 queries; "
            << total_filters << " FILTER + " << total_projections
            << " PROJECTION operator(s) exist in total\n";
}
