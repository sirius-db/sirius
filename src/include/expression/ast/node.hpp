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
#include <memory>
#include <variant>

namespace sirius::ast {

// Forward declarations — each node type is completed by its own header,
// which must be included below before `node` is used by value.
struct reference;
struct constant;
struct comparison;
struct conjunction;
struct between;
struct case_;
struct cast;
struct operator_;
struct function_call;

/**
 * @brief Sum type over every Sirius AST node kind.
 *
 * Recursive children are carried via std::unique_ptr<node> inside each struct,
 * which is why forward declarations above are sufficient at this point.
 *
 * The alternative order is part of the public ABI — std::variant indexes by
 * position and Phase 5's std::visit dispatch depends on this ordering.
 */
using node = std::variant<reference,
                          constant,
                          comparison,
                          conjunction,
                          between,
                          case_,
                          cast,
                          operator_,
                          function_call>;

}  // namespace sirius::ast

// Completing includes — each per-node header defines its struct, which may
// store std::unique_ptr<node>. unique_ptr<T> does not require T to be complete
// at the point of declaration, so circular completion is safe.
#include "expression/ast/between.hpp"
#include "expression/ast/case_.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/comparison.hpp"
#include "expression/ast/conjunction.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/operator_.hpp"
#include "expression/ast/reference.hpp"
