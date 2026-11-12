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
 * Tests iterate `sirius::all_expression_evaluator_strategies`, so a new strategy is covered by
 * every test that iterates it. A test either generates one run per strategy with
 * `GENERATE(sirius::test::each_expression_evaluator_strategy())`, or loops over
 * `sirius::all_expression_evaluator_strategies`; in both cases it selects the strategy with a
 * `sirius::test::scoped_expression_evaluator_strategy`, so the choice never outlives the test.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <expression_evaluator/expression_evaluator_strategy.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <string>

namespace sirius::test {

/**
 * @brief A Catch2 generator over `sirius::all_expression_evaluator_strategies`.
 *
 * `GENERATE(each_expression_evaluator_strategy())` runs the enclosing test case once per strategy,
 * each time with a freshly constructed fixture.
 */
[[nodiscard]] inline Catch::Generators::GeneratorWrapper<expression_evaluator_strategy>
each_expression_evaluator_strategy()
{
  return Catch::Generators::from_range(all_expression_evaluator_strategies.begin(),
                                       all_expression_evaluator_strategies.end());
}

/**
 * @brief Selects an Expression Evaluator Strategy on a connection and restores the previous one on
 * destruction.
 */
class scoped_expression_evaluator_strategy final {
 public:
  /** @param connection The connection whose session setting changes; it must outlive the guard. */
  scoped_expression_evaluator_strategy(duckdb::Connection& connection,
                                       expression_evaluator_strategy strategy)
    : setting_{connection, "expression_evaluator_strategy", strategy_to_string(strategy)}
  {
  }

 private:
  scoped_sirius_setting setting_;
};

}  // namespace sirius::test

/** @brief Prints a strategy in Catch2 messages, such as `CAPTURE`, as its setting value. */
template <>
struct Catch::StringMaker<sirius::expression_evaluator_strategy> {
  static std::string convert(sirius::expression_evaluator_strategy strategy)
  {
    return std::string{sirius::strategy_to_string(strategy)};
  }
};
