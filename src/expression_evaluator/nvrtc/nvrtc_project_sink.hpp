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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <expression/ast/node.hpp>
#include <expression_evaluator/like_multiliteral.hpp>
#include <expression_evaluator/nvrtc/boundary_column_map.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/**
 * @file
 * @brief The `NVRTC_JIT` entry points of @c expression_evaluator: decide whether a projection list
 *        or filter predicate compiles to one fused kernel, and run it.
 *
 * @c expression_evaluator first asks @c can_compile_* over its input table; when that fails it
 * pre-evaluates the Materialization Boundaries that @c collect_materialization_boundaries reports
 * into synthetic columns appended to the input, and asks again with the resulting
 * @c boundary_column_map. The @c can_compile_* predicates never throw: any failure to decide is
 * reported as "not compilable", which routes the expression to the materialize path.
 */

namespace sirius::nvrtc {

/**
 * @brief Collects maximal in-tree `Materialization Boundary` sub-trees inside @p expressions that
 *        produce register-supported types (`STRING` or fixed-width) so they can be pre-evaluated
 *        into synthetic input columns before compiling @p expressions.
 *
 * Materialization Boundaries (or JIT 'breakers') exist at the following expression nodes, and at
 * any node whose result type is not fixed-width:
 * - `function_id::concat`
 * - `function_id::concat_operator`
 * - `function_id::upper`
 * - `function_id::lower`
 * - `function_id::regexp_replace`
 * - CAST nodes between non-STRING and STRING types, and non-temporal and temporal types
 *
 * `LIKE`, `NOT LIKE` and `contains` are compiled in-kernel and are not boundaries.
 *
 * @param expressions The expressions to collect boundaries from.
 * @param input The input table view.
 * @param out_boundaries The vector to store the boundaries in.
 */
void collect_materialization_boundaries(std::span<sirius::ast::node const* const> expressions,
                                        cudf::table_view const& input,
                                        std::vector<sirius::ast::node const*>& out_boundaries);

/** @brief Per-evaluation options shared by both @c evaluate_project_sink and @c
 * evaluate_filter_sink. */
struct sink_options {
  boundary_column_map const* boundary_map{
    nullptr};  ///< Pre-evaluated Materialization Boundary nodes mapped to synthetic column
               ///< indices in the input, or null when the input is the original table.
  bool like_swar_fastpath{false};  ///< Whether constant `%lit%...%` patterns may use the SWAR
                                   ///< matcher (the `like_swar_fastpath` setting), and the
                                   ///< query-owned classification cache to consult.
  like_multiliteral_cache const* like_cache{nullptr};  ///< The query-owned classification cache.
};

/** @brief Result of @c evaluate_project_sink. */
struct project_sink_result {
  std::unique_ptr<cudf::table> table;  ///< One column per expression, in projection order.
  std::size_t narrow_domain_comparison_count{0};  ///< Comparisons emitted at a narrowed carrier.
};

/** @brief Result of @c evaluate_filter_sink. */
struct filter_sink_result {
  std::unique_ptr<cudf::table> table;       ///< The surviving rows of the projected input.
  std::unique_ptr<cudf::column> survivors;  ///< Ascending INT32 indices of the surviving rows.
  std::size_t narrow_domain_comparison_count{0};  ///< Comparisons emitted at a narrowed carrier.
};

/**
 * @brief Returns whether all expressions in @p expressions can be compiled and fused by the
 *        `NVRTC_JIT` `ProjectSink` compiler for the given @p input table.
 *
 * @param expressions The expressions to check.
 * @param input The input table.
 * @param boundary_map The boundary column map.
 *
 * @return True if the expressions can be compiled and fused, false otherwise.
 */
[[nodiscard]] bool can_compile_project_sink(
  std::span<sirius::ast::node const* const> expressions,
  cudf::table_view const& input,
  boundary_column_map const* boundary_map = nullptr) noexcept;

/**
 * @brief Compiles (or fetches from `codegen::jit::KernelCache`) and executes a single fused
 *        multi-output `ProjectSink` CUDA kernel for @p expressions over @p input.
 *
 * @param expressions The non-null expression trees to evaluate in projection order.
 * @param input The input table view.
 * @param stream The CUDA stream on which to allocate outputs and launch the kernel.
 * @param mr The device memory resource used for output column allocations.
 * @param options Boundary map and LIKE policy.
 * @return The projected table and the translation statistics.
 */
[[nodiscard]] project_sink_result evaluate_project_sink(
  std::span<sirius::ast::node const* const> expressions,
  cudf::table_view const& input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  sink_options const& options = {});

/**
 * @brief Returns whether @p predicate can be compiled and executed by the `NVRTC_JIT` `FilterSink`
 *        compiler for the given @p input table.
 *
 * @param predicate The boolean filter expression tree to evaluate.
 * @param input The input table view.
 * @param boundary_map Optional map of pre-evaluated `Materialization Boundary` nodes to synthetic
 *                     column indices in @p input.
 *
 * @return True if the predicate can be compiled and executed, false otherwise.
 */
[[nodiscard]] bool can_compile_filter_sink(
  sirius::ast::node const& predicate,
  cudf::table_view const& input,
  boundary_column_map const* boundary_map = nullptr) noexcept;

/**
 * @brief Compiles (or fetches from `codegen::jit::KernelCache`) and executes a `FilterSink` CUDA
 *        kernel for @p predicate over @p input using `Sparse-to-Dense Compaction`.
 *
 * @param predicate The boolean filter expression tree to evaluate.
 * @param input The input table view (possibly augmented with synthetic boundary columns).
 * @param projected_input The table view of output columns to materialize for surviving rows.
 * @param want_survivors Whether to also return the surviving row indices (used by
 *                       `select_with_survivors`).
 * @param stream The CUDA stream on which to allocate outputs and launch the kernel.
 * @param mr The device memory resource used for output column allocations.
 * @param options Boundary map and LIKE policy.
 * @return The surviving rows of @p projected_input, the survivors when requested, and the
 *         translation statistics.
 */
[[nodiscard]] filter_sink_result evaluate_filter_sink(sirius::ast::node const& predicate,
                                                      cudf::table_view const& input,
                                                      cudf::table_view const& projected_input,
                                                      bool want_survivors,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr,
                                                      sink_options const& options = {});

}  // namespace sirius::nvrtc
