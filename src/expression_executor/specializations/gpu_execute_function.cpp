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
#include <expression_executor/gpu_expression_executor.hpp>
#include <expression_executor/regex/regex_playground.hpp>
#include <operator/gpu_physical_strings_matching.hpp>
#include <operator/strlen_from_offsets.cuh>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/assert.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>

// standard library
#include <regex>
#include <string>

// There has to be a better way to extract the function semantics than string comparison!
//----------Function Strings----------//
#define ADD_FUNC_STR            "+"
#define SUB_FUNC_STR            "-"
#define MUL_FUNC_STR            "*"
#define DIV_FUNC_STR            "/"
#define INT_DIV_FUNC_STR        "//"
#define MOD_FUNC_STR            "%"
#define SUBSTRING_FUNC_STR_1    "substring"
#define SUBSTRING_FUNC_STR_2    "substr"
#define LIKE_FUNC_STR           "~~"
#define NOT_LIKE_FUNC_STR       "!~~"
#define CONTAINS_FUNC_STR       "contains"
#define PREFIX_FUNC_STR         "prefix"
#define SUFFIX_FUNC_STR         "suffix"
#define YEAR_FUNC_STR           "year"
#define MONTH_FUNC_STR          "month"
#define DAY_FUNC_STR            "day"
#define HOUR_FUNC_STR           "hour"
#define MINUTE_FUNC_STR         "minute"
#define SECOND_FUNC_STR         "second"
#define MILLISECOND_FUNC_STR    "millisecond"
#define MICROSECOND_FUNC_STR    "microsecond"
#define DATE_TRUNC_FUNC_STR     "date_trunc"
#define STRLEN_FUNC_STR         "strlen"
#define LENGTH_FUNC_STR         "length"
#define REGEXP_REPLACE_FUNC_STR "regexp_replace"
#define ERROR_FUNC_STR          "error"
#define ROW_FUNC_STR            "row"
#define STRUCT_PACK_FUNC_STR    "struct_pack"

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundFunctionExpression const& expr,
                                                execution_mode mode)
{
  auto const& func_string = expr.function.name;
  /// We disable AST if the output type is decimal, since cuDF ASTs choke on intermediate decimal
  /// results currently.
  /// TODO: Fix when the following bug fix is in:
  /// https://github.com/rapidsai/cudf/pull/21996
  auto const ast_supported =
    (expr.return_type.id() != duckdb::LogicalTypeId::DECIMAL) &&
    (std::find(supported_ast_functions.begin(), supported_ast_functions.end(), func_string) !=
     supported_ast_functions.end());

  if (ast_supported && _strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size)) {
    // Only numeric binary functions are supported in AST currently
    D_ASSERT(expr.children.size() == 2);

    auto function_type_switch_ast =
      [](std::string const& function_name) -> cudf::ast::ast_operator {
      if (function_name == ADD_FUNC_STR) {
        return cudf::ast::ast_operator::ADD;
      } else if (function_name == SUB_FUNC_STR) {
        return cudf::ast::ast_operator::SUB;
      } else if (function_name == MUL_FUNC_STR) {
        return cudf::ast::ast_operator::MUL;
      } else if (function_name == DIV_FUNC_STR || function_name == INT_DIV_FUNC_STR) {
        return cudf::ast::ast_operator::DIV;
      } else if (function_name == MOD_FUNC_STR) {
        return cudf::ast::ast_operator::MOD;
      } else {
        throw invalid_input_exception(
          "[gpu_expression_executor:function] unsupported AST function type {}", function_name);
      }
    };

    auto left             = execute(*expr.children[0], execution_mode::AST);
    auto right            = execute(*expr.children[1], execution_mode::AST);
    auto const& func_expr = _ast_tree.emplace<cudf::ast::operation>(
      function_type_switch_ast(func_string), left.get_expr(), right.get_expr());

    if (mode == execution_mode::AST) {
      //===----------1: AST Mode----------===//
      return execute_result(
        ast_result(func_expr,
                   {left.get_temp_scalar_indices(), right.get_temp_scalar_indices()},
                   {left.get_temp_column_indices(), right.get_temp_column_indices()}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    // Evaluate the AST subtree
    auto result_column = execute_ast(func_expr);

    // Release consumed temporaries
    release_temporaries({left.get_temp_scalar_indices(), right.get_temp_scalar_indices()},
                        {left.get_temp_column_indices(), right.get_temp_column_indices()});
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with solo operation----------===//
  // If the caller requested AST mode but we fell through (unsupported function for AST),
  // materialize the result and wrap it as a temp column for the parent's AST tree.
  if (mode == execution_mode::AST) {
    // Re-enter execute with MATERIALIZE mode to get the result as a column, then add to the AST
    // tree.
    auto result = execute(expr, execution_mode::MATERIALIZE);
    if (!result.is_owned_column()) {
      // Any function execution in MATERIALIZE mode should produce an owned column. Otherwise,
      // something went wrong.
      throw internal_exception(
        "[gpu_expression_executor:function]: Expected an owned column after executing function "
        "expression.");
    }
    return materialize_as_ast_column(std::move(result.release_column()));
  }
  auto const output_type = GetCudfType(expr.return_type);

  //----------Numeric Binary Functions----------//
  auto execute_numeric_binary_func = [&](cudf::binary_operator op) -> execute_result {
    auto left  = execute(*expr.children[0], execution_mode::MATERIALIZE);
    auto right = execute(*expr.children[1], execution_mode::MATERIALIZE);
    D_ASSERT(!left.is_scalar() || !right.is_scalar());  // Both sides cannot be scalars
    if (left.is_scalar()) {
      return execute_result(cudf::binary_operation(
        left.get_scalar(), right.get_column_view(), op, output_type, _stream, _mr));
    }
    if (right.is_scalar()) {
      return execute_result(cudf::binary_operation(
        left.get_column_view(), right.get_scalar(), op, output_type, _stream, _mr));
    }
    return execute_result(cudf::binary_operation(
      left.get_column_view(), right.get_column_view(), op, output_type, _stream, _mr));
  };
  if (func_string == ADD_FUNC_STR) {
    return execute_numeric_binary_func(cudf::binary_operator::ADD);
  }
  if (func_string == SUB_FUNC_STR) {
    return execute_numeric_binary_func(cudf::binary_operator::SUB);
  }
  if (func_string == MUL_FUNC_STR) {
    return execute_numeric_binary_func(cudf::binary_operator::MUL);
  }
  if (func_string == DIV_FUNC_STR || func_string == INT_DIV_FUNC_STR) {
    return execute_numeric_binary_func(cudf::binary_operator::DIV);
  }
  if (func_string == MOD_FUNC_STR) {
    return execute_numeric_binary_func(cudf::binary_operator::MOD);
  }

  //----------Substring Function----------//
  if (func_string == SUBSTRING_FUNC_STR_1 || func_string == SUBSTRING_FUNC_STR_2) {
    auto input = execute(*expr.children[0], execution_mode::MATERIALIZE);

    // The start and len arguments of SUBSTRING are assumed to be constants
    D_ASSERT(expr.children[1]->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT);
    D_ASSERT(expr.children[2]->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT);
    auto const& start_expr = expr.children[1]->Cast<duckdb::BoundConstantExpression>();
    auto const& len_expr   = expr.children[2]->Cast<duckdb::BoundConstantExpression>();

    // The values provided by DuckDB must be modified to be 0-indexed and <start, stop> instead of
    // <start, len>
    auto const start_val = start_expr.value.GetValue<cudf::size_type>() - 1;
    auto const stop_val  = len_expr.value.GetValue<cudf::size_type>() + start_val;

    auto result_column =
      cudf::strings::slice_strings(cudf::strings_column_view(input.get_column_view()),
                                   cudf::numeric_scalar(start_val, true, _stream, _mr),
                                   cudf::numeric_scalar(stop_val, true, _stream, _mr),
                                   1,
                                   _stream,
                                   _mr);
    return execute_result(std::move(result_column));
  }

  //----------String Matching Functions----------//
  auto setup_string_matching = [&]() -> std::pair<execute_result, std::string> {
    // Children should be <input column, comparator scalar>
    D_ASSERT(expr.children.size() == 2);
    D_ASSERT(expr.children[1]->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT);

    auto input                 = execute(*expr.children[0], execution_mode::MATERIALIZE);
    auto const& match_str_expr = expr.children[1]->Cast<duckdb::BoundConstantExpression>();
    auto match_str             = match_str_expr.value.GetValue<std::string>();
    return {execute_result(std::move(input)), std::move(match_str)};
  };
  if (func_string == LIKE_FUNC_STR || func_string == NOT_LIKE_FUNC_STR) {
    auto [input, match_str] = setup_string_matching();
    auto result_column = cudf::strings::like(cudf::strings_column_view(input.get_column_view()),
                                             std::string_view(match_str),
                                             std::string_view(),
                                             _stream,
                                             _mr);
    if (func_string == NOT_LIKE_FUNC_STR) {
      result_column =
        cudf::unary_operation(result_column->view(), cudf::unary_operator::NOT, _stream, _mr);
    }
    return execute_result(std::move(result_column));
  }
  if (func_string == CONTAINS_FUNC_STR) {
    auto [input, match_str] = setup_string_matching();
    auto result_column = cudf::strings::contains(cudf::strings_column_view(input.get_column_view()),
                                                 cudf::string_scalar(match_str, true, _stream, _mr),
                                                 _stream,
                                                 _mr);
    return execute_result(std::move(result_column));
  }
  if (func_string == PREFIX_FUNC_STR) {
    auto [input, match_str] = setup_string_matching();
    auto result_column =
      cudf::strings::starts_with(cudf::strings_column_view(input.get_column_view()),
                                 cudf::string_scalar(match_str, true, _stream, _mr),
                                 _stream,
                                 _mr);
    return execute_result(std::move(result_column));
  }
  if (func_string == SUFFIX_FUNC_STR) {
    auto [input, match_str] = setup_string_matching();
    auto result_column =
      cudf::strings::ends_with(cudf::strings_column_view(input.get_column_view()),
                               cudf::string_scalar(match_str, true, _stream, _mr),
                               _stream,
                               _mr);
    return execute_result(std::move(result_column));
  }

  //----------DateTime Extraction Functions----------//
  auto execute_datetime_extract_func =
    [&](cudf::datetime::datetime_component component) -> execute_result {
    D_ASSERT(expr.children.size() == 1);
    auto input = execute(*expr.children[0], execution_mode::MATERIALIZE);
    auto result_column =
      cudf::datetime::extract_datetime_component(input.get_column_view(), component, _stream, _mr);
    return execute_result(std::move(result_column));
  };
  if (func_string == YEAR_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::YEAR);
  }
  if (func_string == MONTH_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MONTH);
  }
  if (func_string == DAY_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::DAY);
  }
  if (func_string == HOUR_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::HOUR);
  }
  if (func_string == MINUTE_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MINUTE);
  }
  if (func_string == SECOND_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::SECOND);
  }
  if (func_string == MILLISECOND_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MILLISECOND);
  }
  if (func_string == MICROSECOND_FUNC_STR) {
    return execute_datetime_extract_func(cudf::datetime::datetime_component::MICROSECOND);
  }

  //----------Date Truncation Function----------//
  if (func_string == DATE_TRUNC_FUNC_STR) {
    D_ASSERT(expr.children.size() == 2);
    // The first child is the frequency, which should be a constant string
    D_ASSERT(expr.children[0]->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT);
    auto const& freq_expr = expr.children[0]->Cast<duckdb::BoundConstantExpression>();
    auto const& freq_str  = freq_expr.value.GetValue<std::string>();
    auto input            = execute(*expr.children[1], execution_mode::MATERIALIZE);
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
          "[gpu_expression_executor:function] unrecognized/unsupported date_trunc frequency: {}",
          freq_str);
      }
    };

    auto result_column = cudf::datetime::floor_datetimes(
      input.get_column_view(), freq_string_switch(freq_str), _stream, _mr);
    return execute_result(std::move(result_column));
  }

  //----------Unary Functions----------//
  if (func_string == STRLEN_FUNC_STR) {
    D_ASSERT(expr.children.size() == 1);
    auto input = execute(*expr.children[0], execution_mode::MATERIALIZE);
    auto result_column =
      cudf::strings::count_bytes(cudf::strings_column_view(input.get_column_view()), _stream, _mr);
    return execute_result(std::move(result_column));
  }
  if (func_string == LENGTH_FUNC_STR) {
    D_ASSERT(expr.children.size() == 1);
    auto input         = execute(*expr.children[0], execution_mode::MATERIALIZE);
    auto result_column = cudf::strings::count_characters(
      cudf::strings_column_view(input.get_column_view()), _stream, _mr);
    return execute_result(std::move(result_column));
  }
  if (func_string == REGEXP_REPLACE_FUNC_STR) {
    // The input should be <input column, pattern string scalar, replace string scalar>
    D_ASSERT(expr.children.size() == 3);
    D_ASSERT(expr.children[1]->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT);
    D_ASSERT(expr.children[2]->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT);

    auto input = execute(*expr.children[0], execution_mode::MATERIALIZE);
    D_ASSERT(!input.is_scalar());

    auto const& pattern_str =
      expr.children[1]->Cast<duckdb::BoundConstantExpression>().value.GetValue<std::string>();
    auto const& replace_str =
      expr.children[2]->Cast<duckdb::BoundConstantExpression>().value.GetValue<std::string>();
    auto regex_prog = cudf::strings::regex_program::create(std::string_view(pattern_str));

    auto const has_backrefs = std::regex_search(replace_str, std::regex(R"(\\[0-9])"));
    if (has_backrefs) {
      if (duckdb::Config::ENABLE_REGEX_JIT_IMPL) {
        if (pattern_str == R"(^https?://(?:www\.)?([^/]+)/.*$)" && replace_str == R"(\1)") {
          return ::sirius::expression::regex_playground::jit_transform_clickbench_q28_regex(
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
  if (func_string == ROW_FUNC_STR || func_string == STRUCT_PACK_FUNC_STR) {
    D_ASSERT(!expr.children.empty());
    std::vector<std::unique_ptr<cudf::column>> child_cols;
    for (const auto& expr : expr.children) {
      auto result = execute(*expr, execution_mode::MATERIALIZE);
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

  // If we reach here, it means the function is not supported in the expression executor
  throw not_implemented_exception(
    "[gpu_expression_executor:function] execute called on unsupported function: %s", func_string);
}

}  // namespace sirius
