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

/**
 * @file expression_evaluator_strategies.hpp
 * @brief The shared list of Expression Evaluator Strategies that SQL integration tests iterate.
 *
 * The list is derived from `sirius::all_expression_evaluator_strategies`, so a new strategy is
 * covered by every test that iterates it. A test either generates one run per strategy with
 * `GENERATE(sirius::test::each_expression_evaluator_strategy())`, or loops over
 * `sirius::test::expression_evaluator_strategy_names()`; in both cases it selects the strategy with
 * a `sirius::test::scoped_expression_evaluator_strategy`, so the choice never outlives the test.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <expression_evaluator/expression_evaluator_strategy.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <algorithm>
#include <array>
#include <string_view>

namespace sirius::test {

using expression_evaluator_strategy_name_list =
  std::array<std::string_view, all_expression_evaluator_strategies.size()>;

/**
 * @brief The `expression_evaluator_strategy` setting value of each entry of
 * `sirius::all_expression_evaluator_strategies`, in the same order.
 */
[[nodiscard]] inline expression_evaluator_strategy_name_list const&
expression_evaluator_strategy_names()
{
  static auto const names = [] {
    expression_evaluator_strategy_name_list spelled{};
    std::ranges::transform(
      all_expression_evaluator_strategies, spelled.begin(), strategy_to_string);
    return spelled;
  }();
  return names;
}

/**
 * @brief A Catch2 generator over `expression_evaluator_strategy_names()`.
 *
 * `GENERATE(each_expression_evaluator_strategy())` runs the enclosing test case once per strategy,
 * each time with a freshly constructed fixture.
 */
[[nodiscard]] inline Catch::Generators::GeneratorWrapper<std::string_view>
each_expression_evaluator_strategy()
{
  auto const& names = expression_evaluator_strategy_names();
  return Catch::Generators::from_range(names.begin(), names.end());
}

/**
 * @brief Selects an Expression Evaluator Strategy on a connection and restores the previous one on
 * destruction.
 */
class scoped_expression_evaluator_strategy final {
 public:
  /**
   * @param connection The connection whose session setting changes; it must outlive the guard.
   * @param strategy One of `expression_evaluator_strategy_names()`.
   */
  scoped_expression_evaluator_strategy(duckdb::Connection& connection, std::string_view strategy)
    : setting_{connection, "expression_evaluator_strategy", strategy}
  {
  }

 private:
  scoped_sirius_setting setting_;
};

}  // namespace sirius::test
