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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <expression/ast/node.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

/**
 * @file
 * @brief The `NVRTC_JIT` compile planner, which decides before any GPU work which expressions one
 * fused kernel computes, and the type helpers it shares with @c translation_context.
 *
 * `expression_evaluator` plans each kernel with `plan_project_sink` or `plan_filter_sink`. A
 * `compile_plan` names the roots the kernel computes and either runs directly over the input table,
 * or names the Materialization Boundaries to evaluate first: the evaluator appends their columns to
 * the input in plan order, checks each with `is_supported_boundary_column`, and binds them through
 * a `boundary_column_map`. The planners never throw: any failure to decide is reported as "no
 * plan", which routes the expressions to the materialize path.
 *
 * The planner and @c translation_context are two encodings of the same knowledge and must agree:
 * every node kind and type combination the planner accepts has a @c produce_* path, which the
 * GPU-free corpus test in `test/cpp/expression_evaluator/test_nvrtc_translation.cpp` checks by
 * translating every accepted tree.
 */

namespace sirius::nvrtc {

/**
 * @brief Which planned expressions one fused kernel computes, and what it reads besides the input
 * table.
 *
 * A Materialization Boundary is a sub-tree the kernel cannot compute but can read as an input
 * column once it has been evaluated: `concat`, `||`, `upper`, `lower`, `regexp_replace`, and CASTs
 * to a register type that the in-kernel cast gate refuses (between STRING and non-STRING, or
 * between temporal and non-temporal types). `LIKE`, `NOT LIKE` and `contains` are compiled
 * in-kernel and are not boundaries.
 */
struct compile_plan {
  /**
   * @brief The positions, in ascending order, of the planned expressions the kernel computes; the
   * caller evaluates the others on the materialize path.
   */
  std::vector<std::size_t> roots;
  /**
   * @brief The maximal boundaries below the kernel's roots, in the order their columns follow the
   * input table's columns; empty when the kernel reads only the input table.
   */
  std::vector<sirius::ast::node const*> boundaries;

  [[nodiscard]] bool is_direct() const noexcept { return boundaries.empty(); }
};

/**
 * @brief Plans one `ProjectSink` kernel that computes, with one fixed-width output each, every
 * expression of @p expressions that such a kernel can compute over @p input.
 *
 * @return The plan, or nullopt when the kernel can compute none of the expressions.
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

/** @brief Ticks per second of a timestamp type, or 0 for TIMESTAMP_DAYS and non-timestamps. */
[[nodiscard]] std::int64_t timestamp_ticks_per_second(cudf::type_id id) noexcept;

/**
 * @brief Returns the cuDF type of the output column a `ProjectSink` produces for @p expr: the
 *        restored type of a reference, the child's type under `TRY`, otherwise the declared return
 *        type.
 *
 * @throws not_implemented_exception when the declared type has no cuDF equivalent.
 */
[[nodiscard]] cudf::data_type resolve_output_cudf_type(sirius::ast::node const& expr,
                                                       cudf::table_view const& input);

/** @brief The value of an INTEGER or BIGINT constant node, or nullopt for any other node. */
[[nodiscard]] std::optional<std::int64_t> extract_constant_int64(
  sirius::ast::node const& node) noexcept;

}  // namespace sirius::nvrtc
