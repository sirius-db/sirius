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
#include <expression/value.hpp>
#include <expression_evaluator/nvrtc/register_value.hpp>
#include <helper/logical_type.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace sirius::nvrtc {

/**
 * @brief Descriptor for a referenced fixed-width input column bound to the compiled kernel.
 */
struct input_column_binding {
  std::uint32_t column_index{0};    ///< Index of the column in the input `cudf::table_view`
  cudf::data_type physical_type{};  ///< Physical carrier type in the input `cudf::column_view`
  bool is_nullable{false};          ///< True if the input column has nulls in the current batch
  std::size_t binding_id{0};        ///< Ordinal index among bound input columns (`in_{id}_...`)
};

/**
 * @brief Descriptor for a deduplicated per-row Register Value loaded from an input column.
 */
struct loaded_reference_entry {
  std::uint32_t column_index{0};    ///< Index of the column in the input `cudf::table_view`
  cudf::data_type restored_type{};  ///< Target cuDF type after in-register Carrier Restoration
  register_value reg{};             ///< Register Value holding the loaded (and restored) attribute
};

/**
 * @brief Descriptor for a runtime scalar constant passed as a `(val, valid)` kernel parameter.
 */
struct scalar_constant_binding {
  std::size_t binding_id{0};        ///< Ordinal index among bound constants (`c_{id}_...`)
  sirius::logical_type sql_type{};  ///< Declared SQL logical type of the constant
  cudf::data_type device_type{};    ///< Physical cuDF type of the constant payload
  sirius::value payload{};          ///< Runtime constant value (or `sirius::null_value`)
  bool is_valid{false};             ///< Runtime validity flag (`!holds_alternative<null_value>`)
  register_value reg{};             ///< Register Value initialized from the kernel parameter
};

/**
 * @brief Compile-time state passed through `produce`/`consume` tracking input column bindings,
 * loaded per-row Register Values, runtime scalar constant parameters, and emitted CUDA statements.
 */
class translation_context {
 public:
  explicit translation_context(cudf::table_view input) : _input(input) {}

  /**
   * @brief Emits the `produce` code for an Expression Tree node and returns its Register Value.
   */
  [[nodiscard]] register_value produce(sirius::ast::node const& expr);

  /**
   * @brief Allocates a fresh local variable name (`v_0`, `v_1`, ...) in the kernel body.
   */
  [[nodiscard]] std::string next_var_name() { return "v_" + std::to_string(_next_var_id++); }

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

  [[nodiscard]] std::string body_code() const { return _body.str(); }

 private:
  [[nodiscard]] register_value produce_reference(sirius::ast::reference const& expr);
  [[nodiscard]] register_value produce_constant(sirius::ast::constant const& expr);

  [[nodiscard]] std::size_t get_or_bind_input_column(std::uint32_t column_index);

  cudf::table_view _input;
  std::vector<input_column_binding> _input_columns;
  std::vector<loaded_reference_entry> _loaded_references;
  std::vector<scalar_constant_binding> _constants;
  std::ostringstream _body;
  std::size_t _next_var_id{0};
};

}  // namespace sirius::nvrtc
