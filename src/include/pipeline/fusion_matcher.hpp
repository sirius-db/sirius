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

#pragma once

#include "expression/ast/node.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <duckdb/common/vector.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

/// Intra-pipeline kernel-fusion matcher.
///
/// Static analysis only: it reports which runs of adjacent operators inside one
/// pipeline's operator list *could* be collapsed into a single expression-evaluator
/// invocation. Nothing here changes the plan or what executes; it measures how much a
/// fused execution path could win. `KERNEL_FUSION_PLAN.md` records the TPC-H numbers.
///
/// Not to be confused with `fuse_merge_pipelines` (task-level merge fusion, see
/// `docs/super-sirius/physical-plan-generation.md`), which folds a MERGE stage into an
/// adjacent pipeline as one more operator. This is about *kernel* launches within one
/// pipeline.
namespace sirius::pipeline {

/// Why an expression tree cannot be lowered wholly into a cuDF AST. `none` means the
/// whole tree lowers; every other value names the first breaker found in a pre-order walk.
enum class ast_breaker {
  none,
  case_expr,             ///< CASE/WHEN — always materializes.
  coalesce,              ///< COALESCE — always materializes (cudf::replace_nulls).
  unsupported_cast,      ///< CAST to a target outside `supported_ast_cast_types_native`.
  carrier_cast,          ///< Compressed-materialization restore cast (physical, not semantic).
  unsupported_function,  ///< Function outside `supported_ast_functions`.
  decimal_function,      ///< Arithmetic returning DECIMAL — cuDF AST cannot hold the
                         ///< intermediate (rapidsai/cudf#21996).
  aggregate,             ///< Aggregate node — never lowers to cuDF AST ops.
  try_operator,          ///< TRY — no defined lowering.
  empty_expression       ///< Malformed / no children.
};

std::string_view to_string(ast_breaker breaker);

/// Why a fusable run stopped growing.
enum class fusion_stop_reason {
  end_of_pipeline,  ///< The run reached the sink.
  operator_kind     ///< The next operator is not a fusable kind (a breaker, join, scan, ...).
};

/// A maximal run of adjacent fusable operators within one pipeline's operator list.
struct fusable_chain {
  std::size_t begin_index      = 0;  ///< First operator index in the run (inclusive).
  std::size_t end_index        = 0;  ///< One past the last operator index (exclusive).
  std::size_t filter_count     = 0;
  std::size_t projection_count = 0;
  /// True when every expression in every operator of the run lowers wholly into a cuDF AST.
  /// A run can still be *structurally* fusable when false — the evaluator materializes at
  /// breakers internally — but only an ast-clean run collapses into one JIT kernel.
  bool ast_clean                 = true;
  ast_breaker first_breaker      = ast_breaker::none;
  fusion_stop_reason stop_reason = fusion_stop_reason::end_of_pipeline;
  /// Operator type name that terminated the run; empty when `end_of_pipeline`.
  std::string stop_detail;

  [[nodiscard]] std::size_t length() const noexcept { return end_index - begin_index; }
};

/// Per-pipeline matcher result.
struct pipeline_fusion_report {
  std::size_t pipeline_id    = 0;
  std::size_t operator_count = 0;
  /// Maximal fusable runs of length >= 2 only — a run of one is what already happens today.
  std::vector<fusable_chain> chains;
};

/// True when `expr` lowers wholly into a cuDF AST (no materialization breaker anywhere in the
/// tree). Mirrors the AST-mode branch conditions in `src/expression_evaluator/specializations/`.
[[nodiscard]] bool is_ast_fusable(sirius::ast::node const& expr);

/// The first AST breaker in a pre-order walk of `expr`, or `ast_breaker::none`.
[[nodiscard]] ast_breaker find_ast_breaker(sirius::ast::node const& expr);

/// Match maximal fusable operator runs in one pipeline.
[[nodiscard]] pipeline_fusion_report match_fusable_chains(sirius_pipeline const& pipeline);

/// Match every scheduled pipeline. Pipelines with no run of length >= 2 are still reported
/// (with an empty `chains`) so coverage denominators stay honest.
[[nodiscard]] std::vector<pipeline_fusion_report> match_fusable_chains(
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> const& pipelines);

/// Human-readable summary of a set of reports, for logging and for the coverage test.
[[nodiscard]] std::string render_fusion_report(std::vector<pipeline_fusion_report> const& reports);

}  // namespace sirius::pipeline
