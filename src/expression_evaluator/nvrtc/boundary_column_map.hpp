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

#include <expression/ast/node.hpp>

#include <cstdint>
#include <unordered_map>

namespace sirius::nvrtc {

/**
 * @brief A map of pre-materialized subexpression nodes to their table column indices.
 *
 * The result of a materialized subexpression that was computed outside the compiling evaluator is
 * stored in input table columns appended to the end of the sequence of original input columns.
 */
using boundary_column_map = std::unordered_map<sirius::ast::node const*, std::uint32_t>;

}  // namespace sirius::nvrtc
