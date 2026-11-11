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

// GPU-free tests of the NVRTC_JIT code generator: the generated CUDA text and the supportability
// predicate are checked against structural properties, over fake column views that are never
// dereferenced. No kernel is compiled or launched.

#include "ast_test_support.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <catch.hpp>
#include <expression/ast/node.hpp>
#include <expression/function_id.hpp>
#include <expression_evaluator/nvrtc/kernel_abi.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/overflow_site.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <helper/logical_type.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace sirius::expr_test;
using sirius::logical_type;
using sirius::type_id;

namespace {

constexpr cudf::size_type fake_rows = 64;

// Host storage the fake column views point at; cuDF only checks that the pointers are non-null.
alignas(16) std::array<std::uint8_t, 16 * fake_rows> fake_data{};
alignas(16) std::array<cudf::bitmask_type, 4> fake_mask{};
alignas(16) std::array<std::int32_t, fake_rows + 1> fake_offsets{};

cudf::column_view fake_column(cudf::data_type type, bool nullable)
{
  return cudf::column_view{
    type, fake_rows, fake_data.data(), nullable ? fake_mask.data() : nullptr, nullable ? 3 : 0};
}

cudf::column_view fake_string_column(bool nullable)
{
  cudf::column_view const offsets{
    cudf::data_type{cudf::type_id::INT32}, fake_rows + 1, fake_offsets.data(), nullptr, 0};
  return cudf::column_view{cudf::data_type{cudf::type_id::STRING},
                           fake_rows,
                           fake_data.data(),
                           nullable ? fake_mask.data() : nullptr,
                           nullable ? 3 : 0,
                           0,
                           {offsets}};
}

std::size_t count_occurrences(std::string const& text, std::string_view needle)
{
  std::size_t count = 0;
  for (auto pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + 1)) {
    ++count;
  }
  return count;
}

// Column 0: nullable INT32, 1: INT64, 2: DOUBLE, 3: nullable STRING, 4: TIMESTAMP_US,
// 5: DECIMAL64(18, 2), 6: DATE, 7: INT16 carrier of a BIGINT column, 8: BOOL8.
struct fake_table {
  std::vector<cudf::column_view> columns{
    fake_column(cudf::data_type{cudf::type_id::INT32}, true),
    fake_column(cudf::data_type{cudf::type_id::INT64}, false),
    fake_column(cudf::data_type{cudf::type_id::FLOAT64}, false),
    fake_string_column(true),
    fake_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, false),
    fake_column(cudf::data_type{cudf::type_id::DECIMAL64, -2}, false),
    fake_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, true),
    fake_column(cudf::data_type{cudf::type_id::INT16}, false),
    fake_column(cudf::data_type{cudf::type_id::BOOL8}, false),
  };
  cudf::table_view view{columns};
};

auto const int_t     = logical_type::make(type_id::INTEGER);
auto const bigint_t  = logical_type::make(type_id::BIGINT);
auto const dbl_t     = logical_type::make(type_id::DOUBLE);
auto const varchar_t = logical_type::make(type_id::VARCHAR);
auto const bool_t    = logical_type::make(type_id::BOOLEAN);
auto const ts_us_t   = logical_type::make(type_id::TIMESTAMP);
auto const date_t    = logical_type::make(type_id::DATE);
auto const dec64_2   = logical_type::make_decimal(18, 2);

std::unique_ptr<ast_node> make_like(sirius::function_id fid, std::uint32_t col, std::string pattern)
{
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref_typed(col, varchar_t));
  args.push_back(make_str_const(std::move(pattern)));
  return make_func(fid, std::move(args), bool_t);
}

std::unique_ptr<ast_node> make_binary(sirius::function_id fid,
                                      std::unique_ptr<ast_node> l,
                                      std::unique_ptr<ast_node> r,
                                      logical_type ret)
{
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(std::move(l));
  args.push_back(std::move(r));
  return make_func(fid, std::move(args), ret);
}

std::unique_ptr<ast_node> make_unary_func(sirius::function_id fid,
                                          std::unique_ptr<ast_node> arg,
                                          logical_type ret)
{
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(std::move(arg));
  return make_func(fid, std::move(args), ret);
}

std::string overflow_message_of(ast_node const& expr)
{
  fake_table table;
  sirius::nvrtc::translation_context ctx(table.view);
  (void)ctx.produce(expr);
  return sirius::nvrtc::overflow_message(ctx.overflow_sites());
}

}  // namespace

TEST_CASE("nvrtc_codegen - each conjunction and coalesce operand is produced exactly once",
          "[expression_evaluator][nvrtc_codegen]")
{
  fake_table table;

  SECTION("AND over three children emits two Kleene steps and binds each constant once")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(
      make_cmp(sirius::comparison_type::lt, make_ref_typed(0, int_t), make_int_const(100)));
    children.push_back(make_cmp(
      sirius::comparison_type::gt,
      make_binary(sirius::function_id::add, make_ref_typed(0, int_t), make_int_const(1), int_t),
      make_int_const(0)));
    children.push_back(
      make_unary(sirius::ast::unary_op::kind::op_is_not_null, make_ref_typed(1, bigint_t)));
    auto const pred = std::make_unique<ast_node>(
      sirius::ast::conjunction{sirius::ast::conjunction::kind::op_and, std::move(children)});

    sirius::nvrtc::translation_context ctx(table.view);
    auto const reg  = ctx.produce(*pred);
    auto const body = ctx.body_code();
    CHECK(count_occurrences(body, "eval_kleene_and(") == 2);
    CHECK(count_occurrences(body, "eval_kleene_or(") == 0);
    // The constants 100, 1 and 0 are bound once each: no duplicate `c_N_val` parameter.
    CHECK(ctx.constants().size() == 3);
    CHECK(count_occurrences(body, " < ") == 1);
    CHECK(ctx.input_columns().size() == 2);
    CHECK(reg.device_type.id() == cudf::type_id::BOOL8);
  }

  SECTION("OR dominance guards reference the accumulator with a true test")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref_typed(8, bool_t));
    children.push_back(make_ref_typed(8, bool_t));
    auto const pred = std::make_unique<ast_node>(
      sirius::ast::conjunction{sirius::ast::conjunction::kind::op_or, std::move(children)});
    sirius::nvrtc::translation_context ctx(table.view);
    auto const reg  = ctx.produce(*pred);
    auto const body = ctx.body_code();
    CHECK(count_occurrences(body, "eval_kleene_or(") == 1);
    CHECK(count_occurrences(body,
                            "if (!(" + reg.var_name + ".valid && !" + reg.var_name +
                              ".overflow && " + reg.var_name + ".val)) {") == 1);
    // The same column is loaded once however often it is referenced.
    CHECK(ctx.input_columns().size() == 1);
    CHECK(count_occurrences(body, "in_0_data[row]") == 1);
  }

  SECTION("COALESCE over three children evaluates the fallbacks only while the result is NULL")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref_typed(0, int_t));
    children.push_back(make_cast(make_ref_typed(1, bigint_t), int_t, false));
    children.push_back(make_int_const(7));
    auto const expr = make_coalesce(std::move(children), int_t);
    sirius::nvrtc::translation_context ctx(table.view);
    auto const reg  = ctx.produce(*expr);
    auto const body = ctx.body_code();
    CHECK(count_occurrences(
            body, "if (!" + reg.var_name + ".valid && !" + reg.var_name + ".overflow) {") == 2);
    CHECK(count_occurrences(body, "eval_cast_int_to_int<int32_t>") == 1);
    CHECK(ctx.constants().size() == 1);
  }
}

TEST_CASE("nvrtc_codegen - the LIKE policy selects the matcher and bounds the descriptor count",
          "[expression_evaluator][nvrtc_codegen]")
{
  fake_table table;

  SECTION("SWAR fast path off: every pattern takes the generic matcher as a string constant")
  {
    auto const expr = make_like(sirius::function_id::like, 3, "%alpha%beta%");
    sirius::nvrtc::translation_context ctx(table.view, nullptr, sirius::nvrtc::like_policy{false});
    (void)ctx.produce(*expr);
    CHECK(ctx.like_patterns().empty());
    CHECK(count_occurrences(ctx.body_code(), "eval_like<false>(") == 1);
    CHECK(ctx.constants().size() == 1);
  }

  SECTION("SWAR fast path on: an eligible pattern becomes a by-value descriptor parameter")
  {
    auto const expr = make_like(sirius::function_id::not_like, 3, "%alpha%beta%");
    sirius::nvrtc::translation_context ctx(table.view, nullptr, sirius::nvrtc::like_policy{true});
    (void)ctx.produce(*expr);
    CHECK(ctx.like_patterns().size() == 1);
    CHECK(count_occurrences(ctx.body_code(), "eval_like_multiliteral<true>(") == 1);
    CHECK(ctx.constants().empty());
  }

  SECTION("the shared cache is consulted and classifies a pattern once per query")
  {
    sirius::like_multiliteral_cache cache;
    sirius::nvrtc::like_policy const policy{
      true, [&cache](std::string_view pattern) { return cache.get_or_classify(pattern); }};
    for (int i = 0; i < 3; ++i) {
      auto const expr = make_like(sirius::function_id::like, 3, "%special%requests%");
      sirius::nvrtc::translation_context ctx(table.view, nullptr, policy);
      (void)ctx.produce(*expr);
      CHECK(ctx.like_patterns().size() == 1);
    }
    CHECK(cache.classification_count_for_testing() == 1);
  }

  SECTION("descriptors beyond the parameter budget fall back to the generic matcher")
  {
    std::vector<std::unique_ptr<ast_node>> children;
    auto const total = sirius::nvrtc::max_like_pattern_parameters + 3;
    for (std::size_t i = 0; i < total; ++i) {
      children.push_back(make_like(sirius::function_id::like, 3, "%lit" + std::to_string(i) + "%"));
    }
    auto const pred = std::make_unique<ast_node>(
      sirius::ast::conjunction{sirius::ast::conjunction::kind::op_or, std::move(children)});
    sirius::nvrtc::translation_context ctx(table.view, nullptr, sirius::nvrtc::like_policy{true});
    (void)ctx.produce(*pred);
    CHECK(ctx.like_patterns().size() == sirius::nvrtc::max_like_pattern_parameters);
    CHECK(count_occurrences(ctx.body_code(), "eval_like<false>(") == 3);
    CHECK(ctx.constants().size() == 3);
  }
}

TEST_CASE("nvrtc_codegen - the kernel ABI table drives the signature and the device prelude",
          "[expression_evaluator][nvrtc_codegen]")
{
  fake_table table;
  std::vector<std::unique_ptr<ast_node>> children;
  children.push_back(make_like(sirius::function_id::like, 3, "%alpha%"));
  children.push_back(
    make_cmp(sirius::comparison_type::ge, make_ref_typed(0, int_t), make_int_const(5)));
  children.push_back(
    make_cmp(sirius::comparison_type::equal, make_ref_typed(3, varchar_t), make_str_const("x")));
  auto const pred = std::make_unique<ast_node>(
    sirius::ast::conjunction{sirius::ast::conjunction::kind::op_and, std::move(children)});

  sirius::nvrtc::translation_context ctx(table.view, nullptr, sirius::nvrtc::like_policy{true});
  (void)ctx.produce(*pred);
  sirius::nvrtc::kernel_abi abi(ctx);
  abi.add_sink_parameter("int32_t", "num_rows");
  abi.add_sink_parameter("uint32_t* __restrict__", "status");

  std::vector<std::string> names;
  for (auto const& p : abi.parameters()) {
    names.push_back(p.identifier);
  }
  // String column 3 was bound first (binding 0), then INT32 column 0 (binding 1); the LIKE
  // descriptor precedes the constants, of which the string one has two parameters.
  CHECK(names == std::vector<std::string>{"in_0_offsets",
                                          "in_0_chars",
                                          "in_0_mask",
                                          "in_0_bit_offset",
                                          "in_1_data",
                                          "in_1_mask",
                                          "in_1_bit_offset",
                                          "lp_0",
                                          "c_0_val",
                                          "c_1_ptr",
                                          "c_1_len",
                                          "num_rows",
                                          "status"});
  CHECK(abi.sink_parameter_count() == 2);

  std::string signature;
  abi.emit_parameter_list(signature);
  CHECK(count_occurrences(signature, ",") == names.size() - 1);
  CHECK(signature.find("like_pattern_desc lp_0") != std::string::npos);

  std::string prelude;
  abi.emit_kernel_prelude(prelude);
  CHECK(prelude.find("in_0_chars_end = (in_0_offsets != nullptr) ? in_0_chars + "
                     "in_0_offsets[num_rows] : nullptr;") != std::string::npos);
  // No `chars_end` kernel parameter exists: the bound is derived on device.
  CHECK(signature.find("chars_end") == std::string::npos);
  CHECK(ctx.body_code().find("in_0_chars_end") != std::string::npos);
}

TEST_CASE("nvrtc_codegen - every tree the compile planner accepts directly translates",
          "[expression_evaluator][nvrtc_codegen]")
{
  fake_table table;
  std::vector<std::unique_ptr<ast_node>> corpus;

  corpus.push_back(
    make_cmp(sirius::comparison_type::lt, make_ref_typed(0, int_t), make_int_const(3)));
  corpus.push_back(make_cmp(
    sirius::comparison_type::not_distinct_from, make_ref_typed(0, int_t), make_null_const(int_t)));
  corpus.push_back(make_between(
    make_ref_typed(1, bigint_t), make_bigint_const(1), make_bigint_const(9), true, false));
  corpus.push_back(
    make_cmp(sirius::comparison_type::gt, make_ref_typed(7, bigint_t), make_bigint_const(30)));
  {
    std::vector<std::unique_ptr<ast_node>> values;
    values.push_back(make_int_const(1));
    values.push_back(make_null_const(int_t));
    values.push_back(make_int_const(2));
    corpus.push_back(make_in(make_ref_typed(0, int_t), std::move(values), true));
  }
  {
    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back(sirius::ast::case_expr::when_then{
      make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(0)),
      make_ref_typed(0, int_t)});
    corpus.push_back(std::make_unique<ast_node>(
      sirius::ast::case_expr{std::move(cases), make_int_const(-1), int_t}));
  }
  corpus.push_back(make_cast(make_ref_typed(2, dbl_t), int_t, true));
  corpus.push_back(make_cast(make_ref_typed(2, dbl_t), dec64_2, false));
  corpus.push_back(make_cast(make_ref_typed(5, dec64_2), dbl_t, false));
  corpus.push_back(
    make_cast(make_ref_typed(4, ts_us_t), logical_type::make(type_id::TIMESTAMP_MS), false));
  corpus.push_back(make_cast(make_ref_typed(6, date_t), ts_us_t, false));
  corpus.push_back(make_cast(make_ref_typed(4, ts_us_t), date_t, false));
  corpus.push_back(make_cast(make_ref_typed(8, bool_t), dec64_2, false));
  corpus.push_back(make_binary(sirius::function_id::mul,
                               make_ref_typed(5, dec64_2),
                               make_dec64_const(150, 18, 2),
                               logical_type::make_decimal(18, 4)));
  corpus.push_back(make_binary(sirius::function_id::div,
                               make_ref_typed(5, dec64_2),
                               make_dec64_const(300, 18, 2),
                               logical_type::make_decimal(18, 4)));
  corpus.push_back(make_binary(
    sirius::function_id::mod, make_ref_typed(1, bigint_t), make_bigint_const(7), bigint_t));
  corpus.push_back(make_unary_func(sirius::function_id::abs, make_ref_typed(0, int_t), int_t));
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(2, dbl_t));
    args.push_back(make_int_const(2));
    corpus.push_back(make_func(sirius::function_id::round, std::move(args), dbl_t));
  }
  corpus.push_back(
    make_unary_func(sirius::function_id::year, make_ref_typed(4, ts_us_t), bigint_t));
  corpus.push_back(
    make_unary_func(sirius::function_id::microsecond, make_ref_typed(6, date_t), bigint_t));
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_str_const("day"));
    args.push_back(make_ref_typed(4, ts_us_t));
    corpus.push_back(make_func(sirius::function_id::date_trunc, std::move(args), ts_us_t));
  }
  corpus.push_back(make_like(sirius::function_id::like, 3, "a_c%"));
  corpus.push_back(make_like(sirius::function_id::contains, 3, "needle"));
  corpus.push_back(make_like(sirius::function_id::prefix, 3, "pre"));
  corpus.push_back(
    make_unary_func(sirius::function_id::length, make_ref_typed(3, varchar_t), bigint_t));
  {
    std::vector<std::unique_ptr<ast_node>> args;
    args.push_back(make_ref_typed(3, varchar_t));
    args.push_back(make_bigint_const(2));
    args.push_back(make_bigint_const(3));
    auto sub = make_func(sirius::function_id::substring, std::move(args), varchar_t);
    corpus.push_back(
      make_cmp(sirius::comparison_type::equal, std::move(sub), make_str_const("ab")));
  }
  corpus.push_back(make_unary(
    sirius::ast::unary_op::kind::op_try,
    make_binary(sirius::function_id::add, make_ref_typed(0, int_t), make_int_const(1), int_t)));

  std::size_t accepted = 0;
  for (std::size_t i = 0; i < corpus.size(); ++i) {
    CAPTURE(i);
    std::array<ast_node const*, 1> exprs{corpus[i].get()};
    auto const as_projection = sirius::nvrtc::plan_project_sink(exprs, table.view);
    auto const as_filter     = corpus[i]->return_type().id() == type_id::BOOLEAN
                                 ? sirius::nvrtc::plan_filter_sink(*corpus[i], table.view)
                                 : std::nullopt;
    if (!(as_projection && as_projection->is_direct()) && !(as_filter && as_filter->is_direct())) {
      continue;
    }
    ++accepted;
    sirius::nvrtc::translation_context ctx(table.view, nullptr, sirius::nvrtc::like_policy{true});
    REQUIRE_NOTHROW((void)ctx.produce(*corpus[i]));
  }
  // The corpus is meant to be compilable; a predicate regression would silently shrink it.
  CHECK(accepted == corpus.size());
}

TEST_CASE("nvrtc_codegen - the compile planner refuses malformed and foreign trees",
          "[expression_evaluator][nvrtc_codegen]")
{
  fake_table table;

  SECTION("null children are refused, not dereferenced")
  {
    auto const cmp = std::make_unique<ast_node>(
      sirius::ast::comparison{sirius::comparison_type::lt, nullptr, nullptr});
    std::array<ast_node const*, 1> exprs{cmp.get()};
    CHECK_FALSE(sirius::nvrtc::plan_project_sink(exprs, table.view));
    CHECK_FALSE(sirius::nvrtc::plan_filter_sink(*cmp, table.view));

    std::vector<std::unique_ptr<ast_node>> children;
    children.push_back(make_ref_typed(8, bool_t));
    children.push_back(nullptr);
    auto const conj = std::make_unique<ast_node>(
      sirius::ast::conjunction{sirius::ast::conjunction::kind::op_and, std::move(children)});
    CHECK_FALSE(sirius::nvrtc::plan_filter_sink(*conj, table.view));
  }

  SECTION("a reference past the input, a non-boolean filter, and a STRING projection are refused")
  {
    auto const ref = make_ref_typed(42, int_t);
    std::array<ast_node const*, 1> exprs{ref.get()};
    CHECK_FALSE(sirius::nvrtc::plan_project_sink(exprs, table.view));
    CHECK_FALSE(sirius::nvrtc::plan_filter_sink(*make_ref_typed(0, int_t), table.view));
    auto const str = make_ref_typed(3, varchar_t);
    std::array<ast_node const*, 1> str_exprs{str.get()};
    CHECK_FALSE(sirius::nvrtc::plan_project_sink(str_exprs, table.view));
  }
}

TEST_CASE("nvrtc_codegen - compile planning names the Materialization Boundaries a kernel reads",
          "[expression_evaluator][nvrtc_codegen][compile_planning]")
{
  fake_table table;
  auto upper_s = [] {
    return make_unary_func(sirius::function_id::upper, make_ref_typed(3, varchar_t), varchar_t);
  };

  SECTION("a tree without boundaries plans directly, and LIKE is not a boundary")
  {
    auto const like = make_like(sirius::function_id::like, 3, "%x%");
    auto const plan = sirius::nvrtc::plan_filter_sink(*like, table.view);
    REQUIRE(plan);
    CHECK(plan->is_direct());
  }

  SECTION("upper(s) = 'X' AND a + b > c plans with upper(s) as its one boundary")
  {
    auto upper            = upper_s();
    auto const* upper_ptr = upper.get();
    std::vector<std::unique_ptr<ast_node>> and_args;
    and_args.push_back(
      make_cmp(sirius::comparison_type::equal, std::move(upper), make_str_const("X")));
    and_args.push_back(make_cmp(sirius::comparison_type::gt,
                                make_binary(sirius::function_id::add,
                                            make_ref_typed(1, bigint_t),
                                            make_ref_typed(1, bigint_t),
                                            bigint_t),
                                make_bigint_const(5)));
    auto const pred = make_conj(sirius::ast::conjunction::kind::op_and, std::move(and_args));

    auto const plan = sirius::nvrtc::plan_filter_sink(*pred, table.view);
    REQUIRE(plan);
    REQUIRE(plan->boundaries.size() == 1);
    CHECK(plan->boundaries[0] == upper_ptr);
  }

  SECTION("only the outermost boundary of nested boundaries is named, in root order")
  {
    auto lower_upper            = make_unary_func(sirius::function_id::lower, upper_s(), varchar_t);
    auto const* lower_upper_ptr = lower_upper.get();
    auto upper                  = upper_s();
    auto const* upper_ptr       = upper.get();
    auto const first =
      make_unary_func(sirius::function_id::strlen, std::move(lower_upper), bigint_t);
    auto const second = make_unary_func(sirius::function_id::strlen, std::move(upper), bigint_t);
    std::array<ast_node const*, 2> exprs{first.get(), second.get()};

    auto const plan = sirius::nvrtc::plan_project_sink(exprs, table.view);
    REQUIRE(plan);
    REQUIRE(plan->boundaries.size() == 2);
    CHECK(plan->boundaries[0] == lower_upper_ptr);
    CHECK(plan->boundaries[1] == upper_ptr);
  }

  SECTION("a boundary under an uncompilable ancestor yields no plan")
  {
    // The VARCHAR CASE cannot be a projection output; row() cannot be computed in a kernel.
    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back({make_cmp(sirius::comparison_type::equal, upper_s(), make_str_const("X")),
                     make_ref_typed(3, varchar_t)});
    auto const case_expr = std::make_unique<ast_node>(
      sirius::ast::case_expr{std::move(cases), make_str_const("none"), varchar_t});
    std::array<ast_node const*, 1> case_exprs{case_expr.get()};
    CHECK_FALSE(sirius::nvrtc::plan_project_sink(case_exprs, table.view));

    auto const row_is_null = make_unary(
      sirius::ast::unary_op::kind::op_is_null,
      make_unary_func(sirius::function_id::row, upper_s(), logical_type::make(type_id::STRUCT)));
    CHECK_FALSE(sirius::nvrtc::plan_filter_sink(*row_is_null, table.view));
  }

  SECTION("a root that is itself a boundary yields no plan")
  {
    auto const cast_root = make_cast(make_ref_typed(3, varchar_t), int_t, false);
    std::array<ast_node const*, 1> exprs{cast_root.get()};
    CHECK_FALSE(sirius::nvrtc::plan_project_sink(exprs, table.view));
  }

  SECTION("an evaluated boundary column is readable unless its string offsets are unsupported")
  {
    CHECK(sirius::nvrtc::is_supported_boundary_column(table.view.column(1)));
    CHECK(sirius::nvrtc::is_supported_boundary_column(table.view.column(3)));
    cudf::column_view const int16_offsets{
      cudf::data_type{cudf::type_id::INT16}, fake_rows + 1, fake_offsets.data(), nullptr, 0};
    cudf::column_view const odd_strings{cudf::data_type{cudf::type_id::STRING},
                                        fake_rows,
                                        fake_data.data(),
                                        nullptr,
                                        0,
                                        0,
                                        {int16_offsets}};
    CHECK_FALSE(sirius::nvrtc::is_supported_boundary_column(odd_strings));
  }
}

TEST_CASE("nvrtc_codegen - an overflow message names the operations that can overflow",
          "[expression_evaluator][nvrtc_codegen][overflow_message]")
{
  std::string const generic = "[expression_evaluator]: Overflow in NVRTC evaluation";
  auto const smallint_t     = logical_type::make(type_id::SMALLINT);
  auto const abs_of_col0    = [] {
    return make_unary_func(sirius::function_id::abs, make_ref_typed(0, int_t), int_t);
  };

  SECTION("a single overflow site is named exactly")
  {
    CHECK(overflow_message_of(*abs_of_col0()) == generic + " of abs(INTEGER) -> INTEGER");
    CHECK(overflow_message_of(*make_cast(make_ref_typed(5, dec64_2), int_t, false)) ==
          generic + " of CAST(DECIMAL(18,2) AS INTEGER)");
  }

  SECTION("overflow sites that read the same are named once")
  {
    auto const sum = make_binary(
      sirius::function_id::add,
      make_binary(
        sirius::function_id::add, make_ref_typed(0, int_t), make_ref_typed(0, int_t), int_t),
      make_ref_typed(0, int_t),
      int_t);
    CHECK(overflow_message_of(*sum) == generic + " of INTEGER + INTEGER -> INTEGER");
  }

  SECTION("distinct overflow sites are listed as candidates in evaluation order")
  {
    auto const product =
      make_binary(sirius::function_id::mul, abs_of_col0(), make_int_const(2), int_t);
    CHECK(overflow_message_of(*product) ==
          generic + " of one of: abs(INTEGER) -> INTEGER; INTEGER * INTEGER -> INTEGER");
  }

  SECTION("a long candidate list names the first four and counts the rest")
  {
    // CAST(((-abs(c0) + c0) - c0) * c0 AS SMALLINT) holds six distinct overflow sites.
    auto expr = make_unary_func(sirius::function_id::negate, abs_of_col0(), int_t);
    for (auto const fid :
         {sirius::function_id::add, sirius::function_id::sub, sirius::function_id::mul}) {
      expr = make_binary(fid, std::move(expr), make_ref_typed(0, int_t), int_t);
    }
    expr = make_cast(std::move(expr), smallint_t, false);
    CHECK(overflow_message_of(*expr) ==
          generic +
            " of one of: abs(INTEGER) -> INTEGER; -(INTEGER) -> INTEGER; INTEGER + INTEGER -> "
            "INTEGER; INTEGER - INTEGER -> INTEGER; and 2 more");
  }

  SECTION("operations under TRY or TRY_CAST are not candidates")
  {
    auto const try_cast_product = make_binary(
      sirius::function_id::mul, make_cast(abs_of_col0(), int_t, true), make_int_const(2), int_t);
    CHECK(overflow_message_of(*try_cast_product) == generic + " of INTEGER * INTEGER -> INTEGER");

    auto const try_sum = make_binary(sirius::function_id::add,
                                     make_unary(sirius::ast::unary_op::kind::op_try, abs_of_col0()),
                                     make_ref_typed(0, int_t),
                                     int_t);
    CHECK(overflow_message_of(*try_sum) == generic + " of INTEGER + INTEGER -> INTEGER");

    CHECK(overflow_message_of(*make_unary(sirius::ast::unary_op::kind::op_try, abs_of_col0())) ==
          generic);
  }

  SECTION("operations under CASE remain candidates")
  {
    std::vector<sirius::ast::case_expr::when_then> cases;
    cases.push_back(
      {make_cmp(sirius::comparison_type::gt, make_ref_typed(0, int_t), make_int_const(0)),
       abs_of_col0()});
    auto const case_expr = std::make_unique<ast_node>(
      sirius::ast::case_expr{std::move(cases), make_int_const(0), int_t});
    CHECK(overflow_message_of(*case_expr) == generic + " of abs(INTEGER) -> INTEGER");
  }

  SECTION("operations that cannot overflow are not candidates")
  {
    auto const sum = make_binary(
      sirius::function_id::add, make_ref_typed(2, dbl_t), make_ref_typed(2, dbl_t), dbl_t);
    CHECK(overflow_message_of(*sum) == generic);
  }
}
