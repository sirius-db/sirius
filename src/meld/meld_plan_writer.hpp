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

#include "expression/ast/node.hpp"
#include "helper/logical_type.hpp"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace sirius::meld {

/**
 * @brief A run of Sirius operators to hand to Meld: filters and projections over a batch, bottom
 * up, and the ungrouped aggregate they feed.
 */
struct meld_chain {
  /// Keeps its input's columns, of the rows @p condition holds for.
  struct filter {
    sirius::ast::node const* condition;
  };
  /// Replaces its input's columns with @p expressions.
  struct projection {
    std::vector<sirius::ast::node const*> expressions;
  };
  using step = std::variant<filter, projection>;

  std::vector<sirius::logical_type> input_types;
  std::vector<step> steps;
  std::vector<sirius::ast::aggregate const*> aggregates;
};

struct meld_plan {
  /// The plan as Meld's assembly, as its printer writes it.
  std::string text;
  /// The names the plan reads the batch's columns by, in the batch's order.
  std::vector<std::string> column_names;
};

/**
 * @brief The Meld plan computing @p chain's per-batch partial: one row with a column per partial
 * Sirius's ungrouped aggregate makes, AVG as its SUM and its COUNT.
 *
 * Returns std::nullopt and says why in @p why for what Meld does not support in this version:
 * decimals, integer sums, physical carriers, and the expressions it has no counterpart for.
 */
std::optional<meld_plan> write_meld_plan(meld_chain const& chain, std::string& why);

}  // namespace sirius::meld
