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

// Shared test helpers for the expression_evaluator unit tests.
//
// Two families of helpers were previously copy-pasted across
// test_expression_evaluator.cpp, test_gpu_expression_translator.cpp, and
// test_expression_evaluator_ast_equivalence.cpp:
//   1. sirius::ast::node construction shortcuts (make_ref, make_int_const, ...)
//   2. GPU->host column copy helpers (copy_column_to_host, ...)
// They are gathered here so the three suites share one definition. All entities
// are `inline`/templates so the header can be included into multiple translation
// units that are linked into the same sirius_unittest binary.

// duckdb — defines the duckdb_base_std alias that the vendored catch.hpp relies on;
// consumers that include this header before <catch.hpp> need it set up first.
#include <duckdb/common/unique_ptr.hpp>

// sirius — AST node construction surface
#include <expression/ast/between.hpp>
#include <expression/ast/case_expr.hpp>
#include <expression/ast/cast.hpp>
#include <expression/ast/coalesce.hpp>
#include <expression/ast/comparison.hpp>
#include <expression/ast/conjunction.hpp>
#include <expression/ast/constant.hpp>
#include <expression/ast/function_call.hpp>
#include <expression/ast/in_list.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <expression/ast/unary_op.hpp>
#include <expression/function_id.hpp>
#include <expression/value.hpp>
#include <helper/logical_type.hpp>

// cudf
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sirius::expr_test {

using ast_node = sirius::ast::node;

//===----------------------------------------------------------------------===//
// sirius::ast::node construction helpers
//===----------------------------------------------------------------------===//

inline std::unique_ptr<ast_node> make_ref(uint32_t idx)
{
  return std::make_unique<ast_node>(sirius::ast::reference{idx});
}

inline std::unique_ptr<ast_node> make_ref_typed(uint32_t idx, sirius::logical_type type)
{
  return std::make_unique<ast_node>(sirius::ast::reference{idx, type});
}

inline std::unique_ptr<ast_node> make_int_const(int32_t v)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{v}, sirius::logical_type::make(sirius::type_id::INTEGER)});
}

inline std::unique_ptr<ast_node> make_bigint_const(int64_t v)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{v}, sirius::logical_type::make(sirius::type_id::BIGINT)});
}

inline std::unique_ptr<ast_node> make_double_const(double v)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{v}, sirius::logical_type::make(sirius::type_id::DOUBLE)});
}

inline std::unique_ptr<ast_node> make_str_const(std::string s)
{
  return std::make_unique<ast_node>(sirius::ast::constant{
    sirius::value{std::move(s)}, sirius::logical_type::make(sirius::type_id::VARCHAR)});
}

// A typed NULL constant (no cuDF AST literal representation).
inline std::unique_ptr<ast_node> make_null_const(sirius::logical_type type)
{
  return std::make_unique<ast_node>(sirius::ast::constant{sirius::value{}, type});
}

inline std::unique_ptr<ast_node> make_dec32_const(int32_t unscaled,
                                                  uint8_t precision,
                                                  uint8_t scale)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::decimal32{unscaled, scale}},
                          sirius::logical_type::make_decimal(precision, scale)});
}

inline std::unique_ptr<ast_node> make_dec64_const(int64_t unscaled,
                                                  uint8_t precision,
                                                  uint8_t scale)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::decimal64{unscaled, scale}},
                          sirius::logical_type::make_decimal(precision, scale)});
}

inline std::unique_ptr<ast_node> make_dec128_const(__int128_t unscaled,
                                                   uint8_t precision,
                                                   uint8_t scale)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::decimal128{unscaled, scale}},
                          sirius::logical_type::make_decimal(precision, scale)});
}

inline std::unique_ptr<ast_node> make_cmp(sirius::comparison_type op,
                                          std::unique_ptr<ast_node> left,
                                          std::unique_ptr<ast_node> right)
{
  return std::make_unique<ast_node>(sirius::ast::comparison{op, std::move(left), std::move(right)});
}

inline std::unique_ptr<ast_node> make_func(sirius::function_id id,
                                           std::vector<std::unique_ptr<ast_node>> args,
                                           sirius::logical_type return_type)
{
  return std::make_unique<ast_node>(sirius::ast::function_call{id, std::move(args), return_type});
}

inline std::unique_ptr<ast_node> make_conj(sirius::ast::conjunction::kind kind,
                                           std::vector<std::unique_ptr<ast_node>> children)
{
  return std::make_unique<ast_node>(sirius::ast::conjunction{kind, std::move(children)});
}

inline std::unique_ptr<ast_node> make_between(std::unique_ptr<ast_node> input,
                                              std::unique_ptr<ast_node> lower,
                                              std::unique_ptr<ast_node> upper,
                                              bool lower_inclusive,
                                              bool upper_inclusive)
{
  return std::make_unique<ast_node>(sirius::ast::between{
    std::move(input), std::move(lower), std::move(upper), lower_inclusive, upper_inclusive});
}

inline std::unique_ptr<ast_node> make_cast(std::unique_ptr<ast_node> child,
                                           sirius::logical_type target_type,
                                           bool try_cast)
{
  return std::make_unique<ast_node>(sirius::ast::cast{std::move(child), target_type, try_cast});
}

inline std::unique_ptr<ast_node> make_coalesce(std::vector<std::unique_ptr<ast_node>> children,
                                               sirius::logical_type return_type)
{
  return std::make_unique<ast_node>(sirius::ast::coalesce{std::move(children), return_type});
}

inline std::unique_ptr<ast_node> make_in(std::unique_ptr<ast_node> probe,
                                         std::vector<std::unique_ptr<ast_node>> values,
                                         bool negated)
{
  return std::make_unique<ast_node>(
    sirius::ast::in_list{std::move(probe), std::move(values), negated});
}

inline std::unique_ptr<ast_node> make_unary(sirius::ast::unary_op::kind kind,
                                            std::unique_ptr<ast_node> child)
{
  return std::make_unique<ast_node>(sirius::ast::unary_op{kind, std::move(child)});
}

//===----------------------------------------------------------------------===//
// GPU -> host column copy helpers
//===----------------------------------------------------------------------===//

template <typename T>
std::vector<T> copy_column_to_host(cudf::column_view const& col)
{
  std::vector<T> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.data<T>(), sizeof(T) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

inline std::vector<uint8_t> copy_bool_column_to_host(cudf::column_view const& col)
{
  std::vector<uint8_t> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(host.data(), col.head(), sizeof(uint8_t) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

inline std::vector<std::string> copy_string_column_to_host(cudf::column_view const& col)
{
  std::vector<std::string> host;
  if (col.size() == 0) { return host; }

  cudf::strings_column_view str_col(col);
  std::vector<cudf::size_type> offsets(col.size() + 1);
  cudaMemcpy(offsets.data(),
             str_col.offsets().data<cudf::size_type>(),
             (col.size() + 1) * sizeof(cudf::size_type),
             cudaMemcpyDeviceToHost);

  std::vector<char> chars(offsets.back());
  if (!chars.empty()) {
    cudaMemcpy(chars.data(),
               str_col.chars_begin(cudf::get_default_stream()),
               offsets.back(),
               cudaMemcpyDeviceToHost);
  }

  host.reserve(col.size());
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto start = offsets[i];
    auto end   = offsets[i + 1];
    if (chars.empty()) {
      host.emplace_back();
    } else {
      host.emplace_back(chars.data() + start, chars.data() + end);
    }
  }
  return host;
}

inline std::vector<bool> copy_valids_to_host(cudf::column_view const& col)
{
  std::vector<bool> valids(col.size(), true);
  if (!col.nullable() || col.null_count() == 0) { return valids; }
  auto const num_words = cudf::num_bitmask_words(col.size());
  std::vector<cudf::bitmask_type> host_mask(num_words);
  cudaMemcpy(host_mask.data(),
             col.null_mask(),
             num_words * sizeof(cudf::bitmask_type),
             cudaMemcpyDeviceToHost);
  constexpr auto bits_per_word = sizeof(cudf::bitmask_type) * 8;
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    auto word = host_mask[i / bits_per_word];
    auto bit  = i % bits_per_word;
    valids[i] = ((word >> bit) & 1U) != 0U;
  }
  return valids;
}

}  // namespace sirius::expr_test
