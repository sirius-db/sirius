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
#include <expression/ast/node.hpp>
#include <expression/function_id.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/ast_supported_types.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/regex/regex_playground.hpp>
#include <helper/logical_type.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/assert.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>

// standard library
#include <algorithm>
#include <regex>
#include <string>
#include <variant>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

evaluate_result expression_evaluator::evaluate(sirius::ast::function_call const& alt,
                                               evaluation_mode mode)
{
  auto const resolved_id = alt.function();
  auto const& args       = alt.arguments();

  // Disable AST mode when the output type is decimal — cuDF ASTs choke on
  // intermediate decimal results currently.
  // TODO: Remove this special-case once Sirius pins cuDF >= 26.10 (cuDF PR
  // https://github.com/rapidsai/cudf/pull/21996 merged after the 26.08 cut,
  // so the fix will ship in 26.10).
  auto const skip_ast      = alt.return_type().id() == sirius::type_id::DECIMAL;
  auto const ast_supported = !skip_ast && std::find(supported_ast_functions.begin(),
                                                    supported_ast_functions.end(),
                                                    resolved_id) != supported_ast_functions.end();

  auto const ast_op_count = alt.cudf_ast_op_count();

  if (ast_supported && _strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    // Only numeric binary functions are supported in AST currently.
    D_ASSERT(args.size() == 2);

    auto function_type_switch_ast = [](function_id id) -> cudf::ast::ast_operator {
      switch (id) {
        case function_id::add: return cudf::ast::ast_operator::ADD;
        case function_id::sub: return cudf::ast::ast_operator::SUB;
        case function_id::mul: return cudf::ast::ast_operator::MUL;
        case function_id::div:
        case function_id::int_div: return cudf::ast::ast_operator::DIV;
        case function_id::mod: return cudf::ast::ast_operator::MOD;
        default:
          throw invalid_input_exception(
            "[expression_evaluator:function] unsupported AST function id {}", static_cast<int>(id));
      }
    };

    auto left             = evaluate(*args[0], evaluation_mode::AST);
    auto right            = evaluate(*args[1], evaluation_mode::AST);
    auto const& func_expr = _ast_tree.emplace<cudf::ast::operation>(
      function_type_switch_ast(resolved_id), left.get_expr(), right.get_expr());

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return evaluate_result(compose(func_expr, {&left, &right}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(func_expr);
    release_temporaries({&left, &right});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with solo operation----------===//
  if (mode == evaluation_mode::AST) {
    auto result = evaluate(alt, evaluation_mode::MATERIALIZE);
    if (!result.is_owned_column()) {
      throw internal_exception(
        "[expression_evaluator:function]: Expected an owned column after executing function "
        "expression.");
    }
    return materialize_as_ast_column(result.release_column());
  }
  auto const output_type = sirius::get_cudf_type(alt.return_type());

  //----------Numeric Binary Functions----------//
  auto execute_numeric_binary_func = [&](cudf::binary_operator op) -> evaluate_result {
    auto left  = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    auto right = evaluate(*args[1], evaluation_mode::MATERIALIZE);
    D_ASSERT(!left.is_scalar() || !right.is_scalar());  // Both sides cannot be scalars
    if (left.is_scalar()) {
      return evaluate_result(cudf::binary_operation(
        left.get_scalar(), right.get_column_view(), op, output_type, _stream, _mr));
    }
    if (right.is_scalar()) {
      return evaluate_result(cudf::binary_operation(
        left.get_column_view(), right.get_scalar(), op, output_type, _stream, _mr));
    }
    return evaluate_result(cudf::binary_operation(
      left.get_column_view(), right.get_column_view(), op, output_type, _stream, _mr));
  };
  if (resolved_id == function_id::add) {
    return execute_numeric_binary_func(cudf::binary_operator::ADD);
  }
  if (resolved_id == function_id::sub) {
    return execute_numeric_binary_func(cudf::binary_operator::SUB);
  }
  if (resolved_id == function_id::mul) {
    return execute_numeric_binary_func(cudf::binary_operator::MUL);
  }
  if (resolved_id == function_id::div || resolved_id == function_id::int_div) {
    return execute_numeric_binary_func(cudf::binary_operator::DIV);
  }
  if (resolved_id == function_id::mod) {
    return execute_numeric_binary_func(cudf::binary_operator::MOD);
  }

  //----------Substring Function----------//
  if (resolved_id == function_id::substring) {
    auto input = evaluate(*args[0], evaluation_mode::MATERIALIZE);

    // DuckDB binds substring as (VARCHAR, BIGINT, BIGINT), so the start/len
    // children are BIGINT constants — the payload variant holds int64_t.
    D_ASSERT(args[1]->holds<sirius::ast::constant>());
    D_ASSERT(args[2]->holds<sirius::ast::constant>());
    auto const start_raw = std::get<int64_t>(args[1]->get<sirius::ast::constant>().payload);
    auto const len_raw   = std::get<int64_t>(args[2]->get<sirius::ast::constant>().payload);

    // Re-base to 0-indexed and convert <start, len> to <start, stop>. Narrow
    // to cudf::size_type (int32_t) here because cudf::strings::slice_strings
    // only accepts int32 bounds — the narrowing is a cudf API constraint,
    // not a SUBSTRING semantic limit.
    auto const start_val = static_cast<cudf::size_type>(start_raw) - 1;
    auto const stop_val  = static_cast<cudf::size_type>(len_raw) + start_val;

    auto result_column =
      cudf::strings::slice_strings(cudf::strings_column_view(input.get_column_view()),
                                   cudf::numeric_scalar(start_val, true, _stream, _mr),
                                   cudf::numeric_scalar(stop_val, true, _stream, _mr),
                                   cudf::numeric_scalar<cudf::size_type>(1, true, _stream, _mr),
                                   _stream,
                                   _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------String Matching Functions----------//
  auto setup_string_matching = [&]() -> std::pair<evaluate_result, std::string> {
    D_ASSERT(args.size() == 2);
    D_ASSERT(args[1]->holds<sirius::ast::constant>());

    auto input     = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    auto match_str = std::get<std::string>(args[1]->get<sirius::ast::constant>().payload);
    return {evaluate_result(std::move(input)), std::move(match_str)};
  };
  if (resolved_id == function_id::like || resolved_id == function_id::not_like) {
    auto [input, match_str] = setup_string_matching();
    auto result_column = cudf::strings::like(cudf::strings_column_view(input.get_column_view()),
                                             std::string_view(match_str),
                                             std::string_view(),
                                             _stream,
                                             _mr);
    if (resolved_id == function_id::not_like) {
      result_column =
        cudf::unary_operation(result_column->view(), cudf::unary_operator::NOT, _stream, _mr);
    }
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::contains) {
    auto [input, match_str] = setup_string_matching();
    auto result_column = cudf::strings::contains(cudf::strings_column_view(input.get_column_view()),
                                                 cudf::string_scalar(match_str, true, _stream, _mr),
                                                 _stream,
                                                 _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::prefix) {
    auto [input, match_str] = setup_string_matching();
    auto result_column =
      cudf::strings::starts_with(cudf::strings_column_view(input.get_column_view()),
                                 cudf::string_scalar(match_str, true, _stream, _mr),
                                 _stream,
                                 _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::suffix) {
    auto [input, match_str] = setup_string_matching();
    auto result_column =
      cudf::strings::ends_with(cudf::strings_column_view(input.get_column_view()),
                               cudf::string_scalar(match_str, true, _stream, _mr),
                               _stream,
                               _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------DateTime Extraction Functions----------//
  auto execute_datetime_extract_func =
    [&](cudf::datetime::datetime_component component) -> evaluate_result {
    D_ASSERT(args.size() == 1);
    auto input = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    auto result_column =
      cudf::datetime::extract_datetime_component(input.get_column_view(), component, _stream, _mr);
    return evaluate_result(std::move(result_column));
  };
  if (resolved_id == function_id::year) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::YEAR);
  }
  if (resolved_id == function_id::month) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MONTH);
  }
  if (resolved_id == function_id::day) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::DAY);
  }
  if (resolved_id == function_id::hour) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::HOUR);
  }
  if (resolved_id == function_id::minute) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MINUTE);
  }
  if (resolved_id == function_id::second) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::SECOND);
  }
  if (resolved_id == function_id::millisecond) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MILLISECOND);
  }
  if (resolved_id == function_id::microsecond) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MICROSECOND);
  }

  //----------Date Truncation Function----------//
  if (resolved_id == function_id::date_trunc) {
    D_ASSERT(args.size() == 2);
    // The first child is the frequency, which should be a constant string
    D_ASSERT(args[0]->holds<sirius::ast::constant>());
    auto const& freq_str = std::get<std::string>(args[0]->get<sirius::ast::constant>().payload);
    auto input           = evaluate(*args[1], evaluation_mode::MATERIALIZE);
    D_ASSERT(!input.is_scalar());

    auto freq_string_switch =
      [](std::string const& freq_str) -> cudf::datetime::rounding_frequency {
      if (freq_str == "day") {
        return cudf::datetime::rounding_frequency::DAY;
      } else if (freq_str == "hour") {
        return cudf::datetime::rounding_frequency::HOUR;
      } else if (freq_str == "minute") {
        return cudf::datetime::rounding_frequency::MINUTE;
      } else if (freq_str == "second") {
        return cudf::datetime::rounding_frequency::SECOND;
      } else if (freq_str == "millisecond") {
        return cudf::datetime::rounding_frequency::MILLISECOND;
      } else if (freq_str == "microsecond") {
        return cudf::datetime::rounding_frequency::MICROSECOND;
      } else {
        throw invalid_input_exception(
          "[expression_evaluator:function] unrecognized/unsupported date_trunc frequency: {}",
          freq_str);
      }
    };

    auto result_column = cudf::datetime::floor_datetimes(
      input.get_column_view(), freq_string_switch(freq_str), _stream, _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------String Concatenation----------//
  if (resolved_id == function_id::concat) {
    // Evaluate every argument to a materialized column or scalar.
    std::vector<evaluate_result> arg_results;
    arg_results.reserve(args.size());
    for (auto const& arg : args) {
      arg_results.push_back(evaluate(*arg, evaluation_mode::MATERIALIZE));
    }

    // cudf::strings::concatenate takes a table_view — scalars must be expanded
    // to full-length columns first.
    auto const num_rows = _input_table.num_rows();
    std::vector<std::unique_ptr<cudf::column>> scalar_cols;
    std::vector<cudf::column_view> col_views;
    col_views.reserve(arg_results.size());
    for (auto const& res : arg_results) {
      if (res.is_scalar()) {
        scalar_cols.push_back(
          cudf::make_column_from_scalar(res.get_scalar(), num_rows, _stream, _mr));
        col_views.push_back(scalar_cols.back()->view());
      } else {
        col_views.push_back(res.get_column_view());
      }
    }

    // SQL concat: any NULL input produces NULL output.  cuDF achieves this with
    // an invalid (null) narep scalar.
    cudf::table_view concat_table(col_views);
    auto result_column = cudf::strings::concatenate(
      concat_table,
      cudf::string_scalar("", true, _stream, _mr),   // empty separator between parts
      cudf::string_scalar("", false, _stream, _mr),  // null narep → null propagation
      cudf::strings::separator_on_nulls::YES,        // no-op: invalid narep short-circuits before
                                                     // separator logic runs
      _stream,
      _mr);
    return evaluate_result(std::move(result_column));
  }

  //----------Unary Functions----------//
  if (resolved_id == function_id::strlen) {
    D_ASSERT(args.size() == 1);
    auto input = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    auto result_column =
      cudf::strings::count_bytes(cudf::strings_column_view(input.get_column_view()), _stream, _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::length) {
    D_ASSERT(args.size() == 1);
    auto input         = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    auto result_column = cudf::strings::count_characters(
      cudf::strings_column_view(input.get_column_view()), _stream, _mr);
    return evaluate_result(std::move(result_column));
  }
  if (resolved_id == function_id::regexp_replace) {
    // The input should be <input column, pattern string scalar, replace string scalar>
    D_ASSERT(args.size() == 3);
    D_ASSERT(args[1]->holds<sirius::ast::constant>());
    D_ASSERT(args[2]->holds<sirius::ast::constant>());

    auto input = evaluate(*args[0], evaluation_mode::MATERIALIZE);
    D_ASSERT(!input.is_scalar());

    auto const& pattern_str = std::get<std::string>(args[1]->get<sirius::ast::constant>().payload);
    auto const& replace_str = std::get<std::string>(args[2]->get<sirius::ast::constant>().payload);
    auto regex_prog         = cudf::strings::regex_program::create(std::string_view(pattern_str));

    auto const has_backrefs = std::regex_search(replace_str, std::regex(R"(\\[0-9])"));
    if (has_backrefs) {
      if (duckdb::Config::ENABLE_REGEX_JIT_IMPL) {
        if (pattern_str == R"(^https?://(?:www\.)?([^/]+)/.*$)" && replace_str == R"(\1)") {
          return ::sirius::regex::regex_playground::jit_transform_clickbench_q28_regex(
            input.get_column_view());
        }
      }
      return cudf::strings::replace_with_backrefs(
        cudf::strings_column_view(input.get_column_view()),
        *regex_prog,
        std::string_view(replace_str),
        _stream,
        _mr);
    } else {
      return cudf::strings::replace_re(cudf::strings_column_view(input.get_column_view()),
                                       *regex_prog,
                                       cudf::string_scalar(replace_str, true, _stream, _mr),
                                       std::nullopt,
                                       _stream,
                                       _mr);
    }
  }

  //----------Struct Functions----------//
  // row() and struct_pack() both construct a struct column from their child expressions.
  // row() is used by DuckDB for tuple constructors like (col1, col2).
  if (resolved_id == function_id::row || resolved_id == function_id::struct_pack) {
    D_ASSERT(!args.empty());
    std::vector<std::unique_ptr<cudf::column>> child_cols;
    for (const auto& a : args) {
      auto result = evaluate(*a, evaluation_mode::MATERIALIZE);
      if (result.is_scalar()) {
        child_cols.push_back(cudf::make_column_from_scalar(
          result.get_scalar(), _input_table.num_rows(), _stream, _mr));
      } else {
        child_cols.push_back(
          std::make_unique<cudf::column>(result.get_column_view(), _stream, _mr));
      }
    }
    auto const num_rows = child_cols[0]->size();
    return cudf::make_structs_column(
      num_rows, std::move(child_cols), 0, rmm::device_buffer{}, _stream, _mr);
  }

  // `error()` is a runtime-error-raising function. We deliberately do not
  // evaluate it on-device here; the upstream fallback path is expected to
  // handle it on the CPU.
  if (resolved_id == function_id::error) {
    throw not_implemented_exception(
      "[expression_evaluator:function] error() is not dispatched on GPU; "
      "expected to fall back to CPU execution");
  }

  // Invariant violation: alt.function() returned a valid function_id but no
  // dispatch arm above claimed it. This means an entry was added to
  // function_id without a corresponding GPU handler here.
  throw internal_exception(
    "[expression_evaluator:function]: registered function_id has no GPU dispatch arm: id={}",
    static_cast<int>(resolved_id));
}

}  // namespace sirius
