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

#include <expression/ast/node.hpp>

#include <span>
#include <string>
#include <variant>

/**
 * @file
 * @brief The host-side record that lets an NVRTC Sink name the operation behind an overflow.
 *
 * The generated kernel reports only that some row carried Overflow Poison to the NVRTC Sink, never
 * which operation raised it, so diagnosing an overflow costs the kernel nothing. Instead,
 * `translation_context` records an `overflow_site` for each operation it emits that can raise
 * Overflow Poison by itself, and `read_status_words` passes those sites to `overflow_message` once
 * the status words report an overflow.
 */

namespace sirius::nvrtc {

/**
 * @brief An operation in a fused NVRTC kernel that can raise Overflow Poison by itself, rather than
 * only pass on the poison of its inputs.
 *
 * A site points into the expression tree that `translation_context` translated, so it is valid only
 * while that tree is alive.
 */
using overflow_site = std::variant<sirius::ast::function_call const*, sirius::ast::cast const*>;

/**
 * @brief Returns the SQL error message for an overflow that one of @p sites raised.
 *
 * Each site is named by its operation and its SQL operand and result types, for example
 * `abs(INTEGER) -> INTEGER`, `INTEGER + INTEGER -> INTEGER` or `CAST(BIGINT AS INTEGER)`. The
 * message names the operation exactly when every site reads the same, and otherwise lists the first
 * four distinct candidates in the order of @p sites and counts the rest.
 *
 * @param sites The overflow sites of the kernel whose NVRTC Sink saw Overflow Poison.
 * @return The message, without operation names when @p sites is empty.
 */
[[nodiscard]] std::string overflow_message(std::span<overflow_site const> sites);

}  // namespace sirius::nvrtc
