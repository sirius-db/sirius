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

#include "pipeline/fusion_matcher.hpp"

#include "expression_evaluator/ast_supported_types.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_projection.hpp"

#include <algorithm>
#include <sstream>
#include <type_traits>
#include <variant>

namespace sirius::pipeline {

namespace {

using op::SiriusPhysicalOperatorType;

/// Pre-order walk mirroring the AST-mode branch conditions in
/// `src/expression_evaluator/specializations/`. Returns the first breaker, or `none`.
ast_breaker walk(sirius::ast::node const& n)
{
  auto const first_breaker_of = [](auto const& children) {
    for (auto const& child : children) {
      auto const breaker = walk(*child);
      if (breaker != ast_breaker::none) { return breaker; }
    }
    return ast_breaker::none;
  };

  return std::visit(
    [&](auto const& alt) -> ast_breaker {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, sirius::ast::reference> ||
                    std::is_same_v<T, sirius::ast::constant>) {
        return ast_breaker::none;
      } else if constexpr (std::is_same_v<T, sirius::ast::comparison>) {
        auto const left = walk(*alt.left);
        return left != ast_breaker::none ? left : walk(*alt.right);
      } else if constexpr (std::is_same_v<T, sirius::ast::conjunction>) {
        if (alt.children.empty()) { return ast_breaker::empty_expression; }
        return first_breaker_of(alt.children);
      } else if constexpr (std::is_same_v<T, sirius::ast::between>) {
        auto breaker = walk(*alt.input);
        if (breaker != ast_breaker::none) { return breaker; }
        breaker = walk(*alt.lower);
        return breaker != ast_breaker::none ? breaker : walk(*alt.upper);
      } else if constexpr (std::is_same_v<T, sirius::ast::cast>) {
        // Only semantic casts to a supported target lower into a CAST_TO_* AST op; a
        // carrier restore is authorized to use the physical-representation tunnel and
        // must reach the materialized branch.
        if (alt.kind != sirius::ast::cast_kind::semantic) { return ast_breaker::carrier_cast; }
        auto const supported =
          std::find(supported_ast_cast_types_native.begin(),
                    supported_ast_cast_types_native.end(),
                    alt.target_type.id()) != supported_ast_cast_types_native.end();
        if (!supported) { return ast_breaker::unsupported_cast; }
        return walk(*alt.child);
      } else if constexpr (std::is_same_v<T, sirius::ast::unary_op>) {
        if (alt.op == sirius::ast::unary_op::kind::op_try) { return ast_breaker::try_operator; }
        return walk(*alt.child);
      } else if constexpr (std::is_same_v<T, sirius::ast::in_list>) {
        if (alt.values.empty()) { return ast_breaker::empty_expression; }
        auto const probe = walk(*alt.probe);
        return probe != ast_breaker::none ? probe : first_breaker_of(alt.values);
      } else if constexpr (std::is_same_v<T, sirius::ast::function_call>) {
        // cuDF AST chokes on intermediate DECIMAL results, and only the arithmetic set
        // lowers at all — mirrors function_call::cudf_ast_op_count().
        if (alt.return_type().id() == sirius::type_id::DECIMAL) {
          return ast_breaker::decimal_function;
        }
        auto const supported = std::find(supported_ast_functions.begin(),
                                         supported_ast_functions.end(),
                                         alt.function()) != supported_ast_functions.end();
        if (!supported) { return ast_breaker::unsupported_function; }
        return first_breaker_of(alt.arguments());
      } else if constexpr (std::is_same_v<T, sirius::ast::case_expr>) {
        return ast_breaker::case_expr;
      } else if constexpr (std::is_same_v<T, sirius::ast::coalesce>) {
        return ast_breaker::coalesce;
      } else {
        // sirius::ast::aggregate — never lowers, and its cudf_ast_op_count() throws.
        return ast_breaker::aggregate;
      }
    },
    n.v);
}

/// Whether `op` is a kind this matcher knows how to fold into a neighbouring operator's
/// expression evaluation. Deliberately narrow: FILTER and PROJECTION are the two operators
/// that do nothing but drive `expression_evaluator` over their input batches.
bool is_fusable_kind(SiriusPhysicalOperatorType type)
{
  return type == SiriusPhysicalOperatorType::FILTER ||
         type == SiriusPhysicalOperatorType::PROJECTION;
}

/// The first breaker across every expression the operator evaluates.
/// The parameter is deliberately not named `op` — that would shadow the `sirius::op`
/// namespace the qualified type names below resolve through.
ast_breaker operator_breaker(op::sirius_physical_operator const& oper)
{
  if (oper.type == SiriusPhysicalOperatorType::FILTER) {
    auto const& filter = oper.Cast<op::sirius_physical_filter>();
    return filter.expression == nullptr ? ast_breaker::empty_expression : walk(*filter.expression);
  }
  auto const& projection = oper.Cast<op::sirius_physical_projection>();
  if (projection.select_list.empty()) { return ast_breaker::empty_expression; }
  for (auto const& expr : projection.select_list) {
    auto const breaker = walk(*expr);
    if (breaker != ast_breaker::none) { return breaker; }
  }
  return ast_breaker::none;
}

}  // namespace

std::string_view to_string(ast_breaker breaker)
{
  switch (breaker) {
    case ast_breaker::none: return "none";
    case ast_breaker::case_expr: return "case";
    case ast_breaker::coalesce: return "coalesce";
    case ast_breaker::unsupported_cast: return "unsupported_cast";
    case ast_breaker::carrier_cast: return "carrier_cast";
    case ast_breaker::unsupported_function: return "unsupported_function";
    case ast_breaker::decimal_function: return "decimal_function";
    case ast_breaker::aggregate: return "aggregate";
    case ast_breaker::try_operator: return "try";
    case ast_breaker::empty_expression: return "empty_expression";
  }
  return "unknown";
}

bool is_ast_fusable(sirius::ast::node const& expr) { return walk(expr) == ast_breaker::none; }

ast_breaker find_ast_breaker(sirius::ast::node const& expr) { return walk(expr); }

pipeline_fusion_report match_fusable_chains(sirius_pipeline const& pipeline)
{
  pipeline_fusion_report report;
  report.pipeline_id = pipeline.get_pipeline_id();

  auto const operators  = pipeline.get_operators();
  report.operator_count = operators.size();

  for (std::size_t i = 0; i < operators.size();) {
    if (!is_fusable_kind(operators[i].get().type)) {
      i++;
      continue;
    }

    fusable_chain chain;
    chain.begin_index = i;
    std::size_t j     = i;
    for (; j < operators.size() && is_fusable_kind(operators[j].get().type); j++) {
      auto const& oper = operators[j].get();
      if (oper.type == SiriusPhysicalOperatorType::FILTER) {
        chain.filter_count++;
      } else {
        chain.projection_count++;
      }
      if (chain.ast_clean) {
        auto const breaker = operator_breaker(oper);
        if (breaker != ast_breaker::none) {
          chain.ast_clean     = false;
          chain.first_breaker = breaker;
        }
      }
    }
    chain.end_index = j;
    if (j < operators.size()) {
      chain.stop_reason = fusion_stop_reason::operator_kind;
      chain.stop_detail = op::SiriusPhysicalOperatorToString(operators[j].get().type);
    } else {
      chain.stop_reason = fusion_stop_reason::end_of_pipeline;
    }

    // A run of one is what already executes today — only report real fusion candidates.
    if (chain.length() >= 2) { report.chains.push_back(std::move(chain)); }
    i = j;  // j > i: the loop consumed at least the operator at i.
  }

  return report;
}

std::vector<pipeline_fusion_report> match_fusable_chains(
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> const& pipelines)
{
  std::vector<pipeline_fusion_report> reports;
  reports.reserve(pipelines.size());
  for (auto const& pipeline : pipelines) {
    if (!pipeline) { continue; }
    reports.push_back(match_fusable_chains(*pipeline));
  }
  return reports;
}

std::string render_fusion_report(std::vector<pipeline_fusion_report> const& reports)
{
  std::size_t total_chains     = 0;
  std::size_t ast_clean_chains = 0;
  std::size_t fused_operators  = 0;

  std::ostringstream out;
  for (auto const& report : reports) {
    for (auto const& chain : report.chains) {
      total_chains++;
      fused_operators += chain.length();
      if (chain.ast_clean) { ast_clean_chains++; }
      out << "  pipeline " << report.pipeline_id << ": ops [" << chain.begin_index << ".."
          << chain.end_index << ") len=" << chain.length() << " filters=" << chain.filter_count
          << " projections=" << chain.projection_count
          << (chain.ast_clean ? " ast_clean" : " breaker=");
      if (!chain.ast_clean) { out << to_string(chain.first_breaker); }
      out << " stopped_by="
          << (chain.stop_reason == fusion_stop_reason::end_of_pipeline ? "end_of_pipeline"
                                                                       : chain.stop_detail)
          << "\n";
    }
  }

  std::ostringstream header;
  header << "fusion matcher: " << total_chains << " candidate chain(s) across " << reports.size()
         << " pipeline(s); " << ast_clean_chains << " ast-clean; " << fused_operators
         << " operator(s) inside a chain\n";
  return header.str() + out.str();
}

}  // namespace sirius::pipeline
