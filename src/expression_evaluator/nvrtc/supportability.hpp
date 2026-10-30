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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <expression/ast/node.hpp>
#include <expression_evaluator/nvrtc/boundary_column_map.hpp>

#include <cstdint>
#include <optional>
#include <vector>

/**
 * @file
 * @brief The supportability predicate of the `NVRTC_JIT` compiler: decides, without emitting code,
 *        whether `translation_context::produce` can translate an expression tree over a given
 *        input table, and which sub-trees must be pre-evaluated as `Materialization Boundaries`.
 *
 * The predicate and @c translation_context are two encodings of the same knowledge and must agree:
 * every node kind and type combination the predicate accepts has a @c produce_* path, which the
 * GPU-free corpus test in `test/cpp/expression_evaluator/test_nvrtc_translation.cpp` checks by
 * translating every accepted tree. The functions here may throw (they call cuDF and the type
 * helpers); the public @c can_compile_* entry points convert any exception into "not compilable".
 */

namespace sirius::nvrtc {

/** @brief Ticks per second of a timestamp type, or 0 for TIMESTAMP_DAYS and non-timestamps. */
[[nodiscard]] std::int64_t timestamp_ticks_per_second(cudf::type_id id) noexcept;

/**
 * @brief Returns the cuDF type a node evaluates to in registers: a synthetic boundary column's
 *        type, a reference's restored logical type (or its physical carrier when it is not a
 *        narrowed numeric), or the node's declared return type. `EMPTY` when it has none.
 */
[[nodiscard]] cudf::data_type effective_node_cudf_type(
  sirius::ast::node const& expr,
  cudf::table_view const& input,
  boundary_column_map const* boundary_map = nullptr);

/**
 * @brief Returns whether @p expr and every node below it can be translated by
 *        @c translation_context::produce over @p input, treating nodes in @p boundary_map as
 *        loads of their synthetic columns.
 */
[[nodiscard]] bool is_node_supported(sirius::ast::node const& expr,
                                     cudf::table_view const& input,
                                     boundary_column_map const* boundary_map = nullptr);

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

/**
 * @brief Appends to @p out_boundaries every maximal sub-tree of @p expr that is an in-tree
 *        Materialization Boundary (see @c collect_materialization_boundaries), without descending
 *        into a boundary.
 */
void collect_boundaries_from_node(sirius::ast::node const& expr,
                                  cudf::table_view const& input,
                                  std::vector<sirius::ast::node const*>& out_boundaries);

/** @brief Returns whether @p expr itself is an in-tree Materialization Boundary. */
[[nodiscard]] bool is_in_tree_materialization_boundary(sirius::ast::node const& expr,
                                                       cudf::table_view const& input);

}  // namespace sirius::nvrtc
