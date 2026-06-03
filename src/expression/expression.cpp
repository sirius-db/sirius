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

// sirius
#include <expression/ast/from_duckdb.hpp>
#include <expression/expression_internal.hpp>

// duckdb — wrap()/wrap_many() own duckdb::Expression unique_ptrs by value, so the
// complete type is needed for their destructors (the internal header no longer pulls it in).
#include <duckdb/planner/expression.hpp>

// standard library
#include <memory>
#include <utility>

namespace sirius {

expression::expression() noexcept = default;

expression::expression(std::unique_ptr<impl> p) noexcept : _impl(std::move(p)) {}

expression::~expression() = default;

expression::expression(expression&&) noexcept = default;

expression& expression::operator=(expression&&) noexcept = default;

expression wrap(std::unique_ptr<duckdb::Expression> expr)
{
  if (!expr) { return expression{}; }
  // wrap() is the DuckDB->Sirius translation boundary. from_duckdb returns nullptr
  // for an unsupported shape (a fallback signal) -> yield a null expression; its
  // own exceptions on a genuinely-unseen ExpressionClass propagate unchanged.
  auto node = sirius::ast::from_duckdb(*expr);
  if (!node) { return expression{}; }
  return expression{std::make_unique<expression::impl>(expression::impl{std::move(node)})};
}

std::vector<expression> wrap_many(std::vector<std::unique_ptr<duckdb::Expression>> exprs)
{
  std::vector<expression> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(wrap(std::move(e)));
  }
  return out;
}

duckdb::vector<expression> wrap_many(duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<expression> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(wrap(std::move(e)));
  }
  return out;
}

}  // namespace sirius
