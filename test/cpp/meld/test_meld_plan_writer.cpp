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

// sirius::meld::write_meld_plan over Sirius expression trees: plans Meld checks and runs, and what
// the writer does not hand to Meld.

#include "catch.hpp"
#include "meld/meld_plan_writer.hpp"
#include "meld/meld_unit.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using sirius::logical_type;
using sirius::type_id;
namespace ast = sirius::ast;

logical_type const f64 = logical_type::make(type_id::DOUBLE);

std::unique_ptr<ast::node> ref(std::uint32_t column, logical_type type = f64)
{
  return std::make_unique<ast::node>(ast::reference(column, std::move(type)));
}

std::unique_ptr<ast::node> literal(double value)
{
  return std::make_unique<ast::node>(ast::constant(sirius::value{value}, f64));
}

std::unique_ptr<ast::node> less_than(std::unique_ptr<ast::node> left,
                                     std::unique_ptr<ast::node> right)
{
  ast::comparison cmp;
  cmp.op    = sirius::comparison_type::lt;
  cmp.left  = std::move(left);
  cmp.right = std::move(right);
  return std::make_unique<ast::node>(std::move(cmp));
}

ast::aggregate aggregate(sirius::aggregate_id id, std::unique_ptr<ast::node> argument)
{
  auto return_type = argument ? f64 : logical_type::make(type_id::BIGINT);
  std::vector<std::unique_ptr<ast::node>> arguments;
  if (argument) { arguments.push_back(std::move(argument)); }
  return ast::aggregate(id, std::move(arguments), return_type, false);
}

/// SELECT sum(y), count(*), min(x), max(y) WHERE x < 24 OR y < 0.125, over (x, y), with a
/// projection that swaps the columns between.
struct filtered_aggregates {
  std::unique_ptr<ast::node> filter;
  std::unique_ptr<ast::node> first  = ref(1);
  std::unique_ptr<ast::node> second = ref(0);
  ast::aggregate sum                = aggregate(sirius::aggregate_id::sum, ref(1));
  ast::aggregate count              = aggregate(sirius::aggregate_id::count_star, nullptr);
  ast::aggregate min                = aggregate(sirius::aggregate_id::min, ref(1));
  ast::aggregate max                = aggregate(sirius::aggregate_id::max, ref(0));

  filtered_aggregates()
  {
    ast::conjunction either;
    either.op = ast::conjunction::kind::op_or;
    either.children.push_back(less_than(ref(0), literal(24.0)));
    either.children.push_back(less_than(ref(1), literal(0.125)));
    filter = std::make_unique<ast::node>(std::move(either));
  }

  [[nodiscard]] sirius::meld::meld_chain chain() const
  {
    sirius::meld::meld_chain chain;
    chain.input_types = {f64, f64};
    chain.steps.emplace_back(sirius::meld::meld_chain::filter{filter.get()});
    chain.steps.emplace_back(sirius::meld::meld_chain::projection{{first.get(), second.get()}});
    chain.aggregates = {&sum, &count, &min, &max};
    return chain;
  }
};

/// WHERE quantity < 24 AND discount BETWEEN 0.0625 AND 0.125, then price * discount, over
/// (quantity, price, discount).
struct q6 {
  std::unique_ptr<ast::node> filter;
  std::unique_ptr<ast::node> product;
  ast::aggregate sum   = aggregate(sirius::aggregate_id::sum, ref(0));
  ast::aggregate count = aggregate(sirius::aggregate_id::count_star, nullptr);
  ast::aggregate avg   = aggregate(sirius::aggregate_id::avg, ref(0));

  q6()
  {
    ast::between range;
    range.input = ref(2);
    range.lower = literal(0.0625);
    range.upper = literal(0.125);
    ast::conjunction both;
    both.op = ast::conjunction::kind::op_and;
    both.children.push_back(less_than(ref(0), literal(24.0)));
    both.children.push_back(std::make_unique<ast::node>(std::move(range)));
    filter = std::make_unique<ast::node>(std::move(both));

    std::vector<std::unique_ptr<ast::node>> factors;
    factors.push_back(ref(1));
    factors.push_back(ref(2));
    product = std::make_unique<ast::node>(
      ast::function_call(sirius::function_id::mul, std::move(factors), f64));
  }

  [[nodiscard]] sirius::meld::meld_chain chain() const
  {
    sirius::meld::meld_chain chain;
    chain.input_types = {f64, f64, f64};
    chain.steps.emplace_back(sirius::meld::meld_chain::filter{filter.get()});
    chain.steps.emplace_back(sirius::meld::meld_chain::projection{{product.get()}});
    chain.aggregates = {&sum, &count, &avg};
    return chain;
  }
};

std::unique_ptr<cudf::column> doubles(std::vector<double> const& values)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                          static_cast<cudf::size_type>(values.size()));
  cudaMemcpy(column->mutable_view().data<double>(),
             values.data(),
             values.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  return column;
}

template <typename T>
T scalar_at(cudf::column_view const& column)
{
  REQUIRE(column.size() == 1);
  REQUIRE_FALSE(column.has_nulls());
  T value;
  cudaMemcpy(&value, column.data<T>(), sizeof(T), cudaMemcpyDeviceToHost);
  return value;
}

}  // namespace

TEST_CASE("write_meld_plan writes a filter, a projection and aggregates that Meld runs",
          "[meld_plan_writer]")
{
  filtered_aggregates query;
  std::string why;
  auto plan = sirius::meld::write_meld_plan(query.chain(), why);
  INFO(why);
  REQUIRE(plan);
  CHECK(plan->column_names == std::vector<std::string>{"c0", "c1"});

  auto unit = sirius::meld::meld_unit::check(plan->text, plan->column_names, why);
  INFO(plan->text);
  INFO(why);
  REQUIRE(unit);
  auto const f64_type = cudf::data_type{cudf::type_id::FLOAT64};
  CHECK(unit->answer_types() ==
        std::vector<cudf::data_type>{
          f64_type, cudf::data_type{cudf::type_id::INT64}, f64_type, f64_type});

  auto x = doubles({1.0, 30.0, 2.5, 40.0});
  auto y = doubles({16.0, 0.0625, 32.0, 64.0});
  cudf::table_view batch({x->view(), y->view()});
  auto answer =
    unit->run(batch, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  // Rows 0, 1 and 2 pass; row 3 has neither. After the swap, sum and min read x, max reads y.
  CHECK(scalar_at<double>(answer->view().column(0)) == 1.0 + 30.0 + 2.5);
  CHECK(scalar_at<std::int64_t>(answer->view().column(1)) == 3);
  CHECK(scalar_at<double>(answer->view().column(2)) == 1.0);
  CHECK(scalar_at<double>(answer->view().column(3)) == 32.0);
}

TEST_CASE("write_meld_plan writes a Q6 of doubles that Meld runs", "[meld_plan_writer]")
{
  q6 query;
  std::string why;
  auto plan = sirius::meld::write_meld_plan(query.chain(), why);
  INFO(why);
  REQUIRE(plan);
  CHECK(plan->column_names == std::vector<std::string>{"c0", "c1", "c2"});

  auto unit = sirius::meld::meld_unit::check(plan->text, plan->column_names, why);
  INFO(plan->text);
  INFO(why);
  REQUIRE(unit);
  // SUM, COUNT(*), and AVG as its SUM and its COUNT, as Sirius's partials are.
  CHECK(unit->answer_types() ==
        std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::FLOAT64},
                                     cudf::data_type{cudf::type_id::INT64},
                                     cudf::data_type{cudf::type_id::FLOAT64},
                                     cudf::data_type{cudf::type_id::INT64}});

  auto quantity = doubles({1.0, 30.0, 2.5, 4.0, 8.0});
  auto price    = doubles({16.0, 2.0, 32.0, 64.0, 128.0});
  auto discount = doubles({0.0625, 0.125, 0.125, 0.25, 0.09375});
  cudf::table_view batch({quantity->view(), price->view(), discount->view()});
  auto answer =
    unit->run(batch, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  // Rows 0, 2 and 4 pass; row 1 has too many, row 3 too much off.
  double const kept_sum = 16.0 * 0.0625 + 32.0 * 0.125 + 128.0 * 0.09375;
  CHECK(scalar_at<double>(answer->view().column(0)) == kept_sum);
  CHECK(scalar_at<std::int64_t>(answer->view().column(1)) == 3);
  CHECK(scalar_at<double>(answer->view().column(2)) == kept_sum);
  CHECK(scalar_at<std::int64_t>(answer->view().column(3)) == 3);
}

TEST_CASE("write_meld_plan does not hand Meld decimals", "[meld_plan_writer]")
{
  filtered_aggregates query;
  auto chain           = query.chain();
  chain.input_types[0] = logical_type::make_decimal(15, 2);
  std::string why;
  CHECK_FALSE(sirius::meld::write_meld_plan(chain, why));
  CHECK(why.find("DECIMAL") != std::string::npos);
}

TEST_CASE("write_meld_plan does not hand Meld an integer sum", "[meld_plan_writer]")
{
  auto const i32 = logical_type::make(type_id::INTEGER);
  std::vector<std::unique_ptr<ast::node>> arguments;
  arguments.push_back(ref(0, i32));
  ast::aggregate sum(
    sirius::aggregate_id::sum, std::move(arguments), logical_type::make(type_id::BIGINT), false);
  sirius::meld::meld_chain chain;
  chain.input_types = {i32};
  chain.aggregates  = {&sum};
  std::string why;
  CHECK_FALSE(sirius::meld::write_meld_plan(chain, why));
  CHECK(why.find("floating point") != std::string::npos);
}

TEST_CASE("write_meld_plan does not hand Meld a physical carrier", "[meld_plan_writer]")
{
  ast::cast restore;
  restore.child       = ref(0, logical_type::make(type_id::SMALLINT));
  restore.target_type = logical_type::make(type_id::INTEGER);
  restore.kind        = ast::cast_kind::carrier_restore;
  ast::node restored(std::move(restore));
  ast::aggregate count = aggregate(sirius::aggregate_id::count_star, nullptr);
  sirius::meld::meld_chain chain;
  chain.input_types = {logical_type::make(type_id::SMALLINT)};
  chain.steps.emplace_back(sirius::meld::meld_chain::projection{{&restored}});
  chain.aggregates = {&count};
  std::string why;
  CHECK_FALSE(sirius::meld::write_meld_plan(chain, why));
  CHECK(why.find("carrier") != std::string::npos);
}
