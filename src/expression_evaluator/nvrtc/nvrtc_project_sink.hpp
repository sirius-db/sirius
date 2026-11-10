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
#include <expression_evaluator/nvrtc/translation_context.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

/**
 * @file
 * @brief The `NVRTC_JIT` entry points of @c expression_evaluator: plan whether a projection list or
 * filter predicate compiles to one fused kernel, and run it.
 *
 * `expression_evaluator` plans each kernel with `plan_project_sink` or `plan_filter_sink` before it
 * does any GPU work. A `compile_plan` either runs directly over the input table, or names the
 * Materialization Boundaries to evaluate first: the evaluator appends their columns to the input in
 * plan order, checks each with `is_supported_boundary_column`, and binds them through a
 * `boundary_column_map` in `sink_options`. The planners never throw: any failure to decide is
 * reported as "no plan", which routes the expressions to the materialize path.
 */

namespace sirius::nvrtc {

/**
 * @brief How one fused kernel computes its expressions over an input table.
 *
 * A Materialization Boundary is a sub-tree the kernel cannot compute but can read as an input
 * column once it has been evaluated: `concat`, `||`, `upper`, `lower`, `regexp_replace`, and CASTs
 * to a register type that the in-kernel cast gate refuses (between STRING and non-STRING, or
 * between temporal and non-temporal types). `LIKE`, `NOT LIKE` and `contains` are compiled
 * in-kernel and are not boundaries.
 */
struct compile_plan {
  /**
   * @brief The maximal boundaries below the kernel's roots, in the order their columns follow the
   * input table's columns; empty when the kernel reads only the input table.
   */
  std::vector<sirius::ast::node const*> boundaries;

  [[nodiscard]] bool is_direct() const noexcept { return boundaries.empty(); }
};

/** @brief Per-evaluation options shared by both @c evaluate_project_sink and @c
 * evaluate_filter_sink. */
struct sink_options {
  boundary_column_map const* boundary_map{
    nullptr};  ///< Pre-evaluated Materialization Boundary nodes mapped to synthetic column
               ///< indices in the input, or null when the input is the original table.
  bool like_swar_fastpath{false};   ///< Whether constant `%lit%...%` patterns may use the SWAR
                                    ///< matcher (the `like_swar_fastpath` setting).
  like_classifier classify_like{};  ///< The evaluator's memoized pattern classification.
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
 * @brief Plans one `ProjectSink` kernel with one fixed-width output per expression of `expressions`
 * over `input`.
 *
 * @return The plan, or nullopt when no such kernel can compute every expression.
 */
[[nodiscard]] std::optional<compile_plan> plan_project_sink(
  std::span<sirius::ast::node const* const> expressions, cudf::table_view const& input) noexcept;

/**
 * @brief Plans one `FilterSink` kernel for the boolean @p predicate over @p input.
 *
 * @return The plan, or nullopt when no such kernel can compute @p predicate.
 */
[[nodiscard]] std::optional<compile_plan> plan_filter_sink(sirius::ast::node const& predicate,
                                                           cudf::table_view const& input) noexcept;

/**
 * @brief Returns whether a kernel can read `column`, the evaluated column of one of a
 * `compile_plan`'s boundaries.
 *
 * The plan already accepted the boundary's declared type, which its evaluated column has; only the
 * width of a STRING column's offsets is unknown until the column exists.
 */
[[nodiscard]] bool is_supported_boundary_column(cudf::column_view const& column) noexcept;

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
