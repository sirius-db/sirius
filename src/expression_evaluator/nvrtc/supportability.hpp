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

#include <cstdint>
#include <optional>

/**
 * @file
 * @brief Type helpers shared by the `NVRTC_JIT` compile planner (`plan_project_sink` and
 * `plan_filter_sink`, implemented in `supportability.cpp`) and @c translation_context.
 *
 * The planner and @c translation_context are two encodings of the same knowledge and must agree:
 * every node kind and type combination the planner accepts has a @c produce_* path, which the
 * GPU-free corpus test in `test/cpp/expression_evaluator/test_nvrtc_translation.cpp` checks by
 * translating every accepted tree.
 */

namespace sirius::nvrtc {

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
