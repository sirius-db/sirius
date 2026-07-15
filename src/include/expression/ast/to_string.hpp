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

// standard library
#include <string>

namespace sirius::ast {

struct node;

/**
 * @brief Render an expression tree as compact, single-line SQL-ish text.
 *
 * Column references render as `#<input-column-index>` (names are not available at this
 * level). Intended for display only (operator params_to_string / telemetry / debug
 * printing) — the output is not parseable SQL and must never feed back into planning.
 * Never throws: unrepresentable literals render as `?`.
 */
[[nodiscard]] std::string to_string(node const& n);

}  // namespace sirius::ast
