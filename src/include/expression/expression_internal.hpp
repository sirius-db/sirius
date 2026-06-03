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
#include <expression/ast/node.hpp>
#include <expression/expression.hpp>

// standard library
#include <memory>
#include <utility>

namespace sirius {

/**
 * @brief Concrete layout of the sirius::expression PIMPL.
 *
 * Completed only in translation units that include this header. Everywhere else
 * `expression::impl` is an incomplete forward-declared type. The wrapper holds a
 * Sirius-native AST node; the DuckDB->Sirius translation happens at the wrap()
 * boundary (see expression.cpp). See https://github.com/sirius-db/sirius/issues/701.
 */
struct expression::impl {
  std::unique_ptr<sirius::ast::node> node;
};

//===----------------------------------------------------------------------===//
// Read-only unwrap helpers
//===----------------------------------------------------------------------===//

inline sirius::ast::node const* unwrap(expression const& e) noexcept
{
  auto const* p = e.get_impl();
  return p ? p->node.get() : nullptr;
}

inline sirius::ast::node* unwrap(expression& e) noexcept
{
  auto* p = e.get_impl();
  return p ? p->node.get() : nullptr;
}

/**
 * @brief Transfers the underlying Sirius AST node out of the wrapper.
 *
 * After this call, the sirius::expression is null.
 */
inline std::unique_ptr<sirius::ast::node> release(expression& e) noexcept
{
  auto* p = e.get_impl();
  if (p == nullptr) { return nullptr; }
  auto out = std::move(p->node);
  e        = expression{};  // drop the impl so is_null() returns true
  return out;
}

}  // namespace sirius
