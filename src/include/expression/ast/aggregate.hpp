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
#include "expression/aggregate_id.hpp"  // sirius::aggregate_id
#include "helper/logical_type.hpp"      // sirius::logical_type

// standard library
#include <memory>
#include <vector>

namespace sirius::ast {

struct node;

/**
 * @brief Sirius-native mirror of duckdb::BoundAggregateExpression.
 *
 * Mirrors function_call (variadic arguments + return_type) and adds a
 * `distinct` flag capturing duckdb::BoundAggregateExpression::IsDistinct().
 * The FILTER clause never reaches this node: the planner rewrites filtered
 * aggregate inputs into mask projections (`CASE WHEN <filter> THEN <input>
 * ELSE NULL END`), lowers `count(*) FILTER` to `count(mask)` at translation,
 * and declines every other filtered shape (CPU fallback). The order-by clause
 * that DuckDB carries on the bound expression is hoisted away upstream by
 * extract_aggregate_expressions and is NOT represented here.
 *
 * No default state: the only way to construct an `aggregate` is via the
 * four-argument constructor. Move-only (the arguments vector holds
 * `unique_ptr<node>`); private fields with const accessors prevent
 * post-construction mutation. Moved-from instances are left in the standard
 * valid-but-unspecified state and must not be read.
 *
 * See https://github.com/sirius-db/sirius/issues/863.
 */
class aggregate {
 public:
  aggregate(sirius::aggregate_id id,
            std::vector<std::unique_ptr<node>> arguments,
            sirius::logical_type return_type,
            bool distinct)
    : function_(id),
      arguments_(std::move(arguments)),
      return_type_(std::move(return_type)),
      distinct_(distinct)
  {
  }

  aggregate()                                = delete;
  aggregate(aggregate const&)                = delete;
  aggregate& operator=(aggregate const&)     = delete;
  aggregate(aggregate&&) noexcept            = default;
  aggregate& operator=(aggregate&&) noexcept = default;
  ~aggregate()                               = default;

  [[nodiscard]] sirius::aggregate_id function() const noexcept { return function_; }
  [[nodiscard]] std::vector<std::unique_ptr<node>> const& arguments() const noexcept
  {
    return arguments_;
  }
  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return return_type_; }
  [[nodiscard]] bool distinct() const noexcept { return distinct_; }

  [[nodiscard]] std::size_t cudf_ast_op_count() const;

 private:
  sirius::aggregate_id function_;
  std::vector<std::unique_ptr<node>> arguments_;
  sirius::logical_type return_type_;
  bool distinct_;
};

}  // namespace sirius::ast
