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
#include <expression/expression.hpp>

// duckdb
#include <duckdb/planner/expression.hpp>

// standard library
#include <memory>
#include <utility>

namespace sirius {

/**
 * @brief Concrete layout of the sirius::expression PIMPL.
 *
 * Completed only in translation units that include this header. Everywhere else
 * `expression::impl` is an incomplete forward-declared type.
 */
struct expression::impl {
  std::unique_ptr<duckdb::Expression> expr;
};

//===----------------------------------------------------------------------===//
// Read-only unwrap helpers
//===----------------------------------------------------------------------===//

inline duckdb::Expression const* unwrap(expression const& e) noexcept
{
  auto const* p = e.get_impl();
  return p ? p->expr.get() : nullptr;
}

inline duckdb::Expression* unwrap(expression& e) noexcept
{
  auto* p = e.get_impl();
  return p ? p->expr.get() : nullptr;
}

/**
 * @brief Transfers the underlying duckdb::Expression out of the wrapper.
 *
 * After this call, the sirius::expression is null.
 */
inline std::unique_ptr<duckdb::Expression> release(expression& e) noexcept
{
  auto* p = e.get_impl();
  if (p == nullptr) { return nullptr; }
  auto out = std::move(p->expr);
  e        = expression{};  // drop the impl so is_null() returns true
  return out;
}

}  // namespace sirius
