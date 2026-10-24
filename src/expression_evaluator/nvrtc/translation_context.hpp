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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <expression/ast/node.hpp>
#include <expression/function_id.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/like_multiliteral.hpp>
#include <expression_evaluator/nvrtc/register_value.hpp>
#include <helper/logical_type.hpp>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::nvrtc {

using boundary_column_map = std::unordered_map<sirius::ast::node const*, std::uint32_t>;

/**
 * @brief Descriptor for a referenced input column bound to the compiled kernel.
 */
struct input_column_binding {
  /** @brief Index of the column in the input `cudf::table_view`. */
  std::uint32_t column_index{0};
  /** @brief Physical carrier type in the input `cudf::column_view`. */
  cudf::data_type physical_type{};
  /** @brief Offset column type (`INT32` or `INT64`) for `STRING` columns. */
  cudf::data_type offset_type{};
  /** @brief True if the input column has nulls in the current batch. */
  bool is_nullable{false};
  /** @brief Ordinal index among bound input columns (`in_{id}_...`). */
  std::size_t binding_id{0};
};

/**
 * @brief Descriptor for a deduplicated per-row Register Value loaded from an input column.
 */
struct loaded_reference_entry {
  /** @brief Index of the column in the input `cudf::table_view`. */
  std::uint32_t column_index{0};
  /** @brief Target cuDF type after in-register Carrier Restoration. */
  cudf::data_type restored_type{};
  /** @brief Register Value holding the loaded (and restored) attribute. */
  register_value reg{};
};

/**
 * @brief Descriptor for a runtime scalar constant passed as a `(val, valid)` kernel parameter.
 */
struct scalar_constant_binding {
  /** @brief Ordinal index among bound constants (`c_{id}_...`). */
  std::size_t binding_id{0};
  /** @brief Declared SQL logical type of the constant. */
  sirius::logical_type sql_type{};
  /** @brief Physical cuDF type of the constant payload. */
  cudf::data_type device_type{};
  /** @brief Runtime constant value (or `sirius::null_value`). */
  sirius::value payload{};
  /** @brief Runtime validity flag (`!holds_alternative<null_value>`). */
  bool is_valid{false};
  /** @brief Register Value initialized from the kernel parameter. */
  register_value reg{};
};

/**
 * @brief Descriptor for a `%lit1%...%litN%` LIKE (or `contains`) pattern matched by the
 * word-at-a-time multi-literal matcher; passed by value as a kernel parameter (`lp_{id}`).
 */
struct like_pattern_binding {
  /** @brief Ordinal index among bound patterns. */
  std::size_t binding_id{0};
  /** @brief Kernel-side literal descriptors (same layout as the device `like_pattern_desc`). */
  sirius::detail::like_multiliteral_pattern_desc descriptor{};
};

/**
 * @brief Compile-time state passed through `produce`/`consume` tracking input column bindings,
 * loaded per-row Register Values, runtime scalar constant parameters, and emitted CUDA statements.
 */
class translation_context {
 public:
  explicit translation_context(cudf::table_view input,
                               boundary_column_map const* boundary_map = nullptr)
    : _input(input), _boundary_map(boundary_map)
  {
  }

  /**
   * @brief Emits the `produce` code for an Expression Tree node and returns its Register Value.
   */
  [[nodiscard]] register_value produce(sirius::ast::node const& expr);

  /**
   * @brief Allocates a fresh local variable name (`v_0`, `v_1`, ...) in the kernel body.
   */
  [[nodiscard]] std::string next_var_name() { return "v_" + std::to_string(_next_var_id++); }

  /**
   * @brief Appends a statement line to the top-level per-row prelude (before any branching).
   */
  void emit_prelude_line(std::string const& line) { _prelude << "    " << line << "\n"; }

  /**
   * @brief Appends a statement line to the per-row evaluation body.
   */
  void emit_line(std::string const& line) { _body << "    " << line << "\n"; }

  [[nodiscard]] cudf::table_view const& input() const noexcept { return _input; }

  [[nodiscard]] std::vector<input_column_binding> const& input_columns() const noexcept
  {
    return _input_columns;
  }

  [[nodiscard]] std::vector<scalar_constant_binding> const& constants() const noexcept
  {
    return _constants;
  }

  [[nodiscard]] std::vector<like_pattern_binding> const& like_patterns() const noexcept
  {
    return _like_patterns;
  }

  [[nodiscard]] std::size_t narrow_domain_comparison_count() const noexcept
  {
    return _narrow_domain_comparison_count;
  }

  [[nodiscard]] bool has_subquery_error() const noexcept { return _has_subquery_error; }

  [[nodiscard]] std::string body_code() const { return _prelude.str() + _body.str(); }

 private:
  [[nodiscard]] register_value produce_reference(sirius::ast::reference const& expr);
  [[nodiscard]] register_value produce_constant(sirius::ast::constant const& expr);
  [[nodiscard]] register_value produce_comparison(sirius::ast::comparison const& expr);
  [[nodiscard]] register_value produce_conjunction(sirius::ast::conjunction const& expr);
  [[nodiscard]] register_value produce_between(sirius::ast::between const& expr);
  [[nodiscard]] register_value produce_case_expr(sirius::ast::case_expr const& expr);
  [[nodiscard]] register_value produce_coalesce(sirius::ast::coalesce const& expr);
  [[nodiscard]] register_value produce_in_list(sirius::ast::in_list const& expr);
  [[nodiscard]] register_value produce_cast(sirius::ast::cast const& expr);
  [[nodiscard]] register_value produce_unary_op(sirius::ast::unary_op const& expr);
  [[nodiscard]] register_value produce_function_call(sirius::ast::function_call const& expr);

  [[nodiscard]] std::optional<cudf::data_type> narrow_domain_carrier(
    sirius::ast::node const& column_operand,
    std::initializer_list<sirius::ast::node const*> constant_operands) const;

  [[nodiscard]] std::optional<cudf::data_type> narrow_domain_reference_pair_carrier(
    sirius::ast::node const& lhs, sirius::ast::node const& rhs) const;

  [[nodiscard]] register_value produce_narrow_domain_operand(
    sirius::ast::node const& operand, std::optional<cudf::data_type> narrow_carrier);

  [[nodiscard]] register_value produce_constant_in_carrier(sirius::ast::constant const& expr,
                                                           cudf::data_type carrier);

  [[nodiscard]] std::size_t get_or_bind_input_column(std::uint32_t column_index);

  /**
   * @brief Binds @p pattern as a multi-literal LIKE descriptor if it is a constant the
   * word-at-a-time matcher supports; returns its binding id, or nullopt for the generic matcher.
   */
  [[nodiscard]] std::optional<std::size_t> try_bind_like_pattern(sirius::ast::node const& pattern,
                                                                 sirius::function_id fid);

  cudf::table_view _input;
  boundary_column_map const* _boundary_map{nullptr};
  std::vector<input_column_binding> _input_columns;
  std::vector<loaded_reference_entry> _loaded_references;
  std::vector<scalar_constant_binding> _constants;
  std::vector<like_pattern_binding> _like_patterns;
  std::size_t _narrow_domain_comparison_count{0};
  bool _has_subquery_error{false};
  std::ostringstream _prelude;
  std::ostringstream _body;
  std::size_t _next_var_id{0};
};

}  // namespace sirius::nvrtc
