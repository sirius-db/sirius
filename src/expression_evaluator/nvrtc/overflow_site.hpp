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
 * @brief The host-side record that lets an NVRTC Sink name the operation behind an overflow or a
 *        division by zero.
 *
 * The generated kernel reports only that some row carried Overflow Poison to the NVRTC Sink, never
 * which operation raised it. Instead, `translation_context` records an `overflow_site` for each
 * operation it emits that can raise Overflow Poison by itself, and `read_status_words` passes those
 * operator sites to `overflow_message` once the status words report an overflow.
 */

namespace sirius::nvrtc {

/**
 * @brief The expression tree node of an operation in a fused NVRTC kernel that can raise Overflow
 *        Poison by itself, rather than only pass on the Overflow Poison of its inputs.
 */
using overflow_site = std::variant<sirius::ast::function_call const*, sirius::ast::cast const*>;

/**
 * @brief Formats an overflow message with possible source operations.
 *
 * The kernel reports overflow without identifying the operation that raised it. This function lists
 * the first four distinct operation descriptions in site order and reports how many were omitted.
 * If there are no sites, it returns a generic message.
 *
 * @param sites The overflow sites of the kernel whose NVRTC Sink saw Overflow Poison.
 * @return The message, without operation names when @p sites is empty.
 */
[[nodiscard]] std::string overflow_message(std::span<overflow_site const> sites);

}  // namespace sirius::nvrtc
