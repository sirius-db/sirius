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

// sirius
#include "expression/ast/node.hpp"  // sirius::ast::node

// standard library
#include <memory>
#include <vector>

namespace sirius::ast {

/**
 * @brief Replace every column reference in @p expr with a deep clone of the
 * corresponding expression from @p inner_select_list.
 *
 * Used when folding adjacent physical projections: the outer projection's
 * references index into the inner projection's output columns, so composition
 * substitutes each reference with the inner expression tree.
 */
std::unique_ptr<node> substitute_references(
  node const& expr, std::vector<std::unique_ptr<node>> const& inner_select_list);

}  // namespace sirius::ast
