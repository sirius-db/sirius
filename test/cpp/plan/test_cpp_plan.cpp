// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <sirius/plan/plan.hpp>

namespace {
sirius::Plan build_plan()
{
  using namespace sirius;
  Field fields[] = {{"id", DataType::int64, true}};
  auto schema    = Schema::create(fields).value();
  auto scan      = PlanNode::scan_parquet("missing.parquet", schema).value();
  auto id        = scan.column(0).value();
  auto ten       = Expression::literal_int64(10).value();
  auto predicate = id.compare(Comparison::greater, ten).value();
  REQUIRE(predicate.type() == DataType::boolean);
  REQUIRE(predicate.nullable());
  return Plan::from_root(scan.filter(predicate).value()).value();
}
}  // namespace

TEST_CASE("A C++ plan and its inspected handles own their inputs", "[cpp_plan]")
{
  using namespace sirius;
  auto description = Plan::from_root(build_plan().root())->describe();
  REQUIRE(description.has_value());
  CHECK(description->contains("missing.parquet"));
  auto root  = build_plan().root();
  auto input = root.input().value();
  REQUIRE(root.kind() == PlanNodeKind::filter);
  root = input;
  REQUIRE(root.kind() == PlanNodeKind::parquet_scan);
  CHECK_FALSE(root.input().has_value());
  auto schema = build_plan().root().schema();
  REQUIRE(schema.size() == 1);
  auto field = schema.field(0).value();
  CHECK(field.name == "id");
  CHECK(field.type == DataType::int64);
  CHECK(field.nullable);
}

TEST_CASE("C++ plan construction propagates invalid argument diagnostics", "[cpp_plan]")
{
  using namespace sirius;
  auto scan  = build_plan().root().input().value();
  auto other = PlanNode::scan_parquet("other.parquet", scan.schema()).value();
  const auto result =
    other.filter(scan.column(0)
                   .value()
                   .compare(Comparison::greater, Expression::literal_int64(10).value())
                   .value());
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == ErrorCode::invalid_argument);
  CHECK_FALSE(result.error().message.empty());
  CHECK_FALSE(scan.schema().field(1).has_value());
  CHECK(scan.filter(Expression::literal_boolean(true).value()).has_value());
}
