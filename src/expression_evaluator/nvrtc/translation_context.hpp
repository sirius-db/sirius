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
#include <expression_evaluator/nvrtc/boundary_column_map.hpp>
#include <expression_evaluator/nvrtc/overflow_site.hpp>
#include <expression_evaluator/nvrtc/register_value.hpp>
#include <helper/logical_type.hpp>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sirius::nvrtc {

/**
 * @brief Descriptor for a referenced input column bound to the compiled kernel.
 */
struct input_column_binding {
  std::uint32_t column_index{0};    ///< Index of the column in the input @c cudf::table_view.
  cudf::data_type physical_type{};  ///< Physical carrier type in the input @c cudf::column_view.
  cudf::data_type offset_type{};    ///< Offset column type (INT32 or INT64) for STRING columns.
  bool is_nullable{false};          ///< True if the input column has nulls in the current batch.
  std::size_t binding_id{0};        ///< Ordinal index among bound input columns (`in_{id}_...`).
};

/**
 * @brief Descriptor for a deduplicated per-row Register Value loaded from an input column.
 */
struct loaded_reference_entry {
  std::uint32_t column_index{0};    ///< Index of the column in the input @c cudf::table_view.
  cudf::data_type restored_type{};  ///< Target cuDF type after in-register Carrier Restoration.
  register_value reg{};             ///< Register Value holding the loaded (and restored) attribute.
};

/**
 * @brief Descriptor for a runtime scalar constant passed as a `(val, valid)` kernel parameter.
 */
struct scalar_constant_binding {
  std::size_t binding_id{0};      ///< Ordinal index among bound constants (`c_{id}_...`).
  cudf::data_type device_type{};  ///< Physical cuDF type of the constant payload.
  sirius::value payload{};        ///< Runtime constant value (or @c sirius::null_value).
  bool is_valid{false};           ///< Runtime validity flag (@c !holds_alternative<null_value>).
  register_value reg{};           ///< Register Value initialized from the kernel parameter.
};

/**
 * @brief Descriptor for a %lit1%...%litN% LIKE (or CONTAINS) pattern matched by the
 *        word-at-a-time multi-literal matcher; passed by value as a kernel parameter (`lp_{id}`).
 */
struct like_pattern_binding {
  std::size_t binding_id{0};  ///< Ordinal index among bound patterns.
  sirius::detail::like_multiliteral_pattern_desc descriptor{};  ///< Kernel-side literal descriptors
                                                                ///< (same layout as the device
                                                                ///< @c like_pattern_desc).
};

/**
 * @brief Classifies a constant LIKE pattern for the SWAR matcher; the evaluator binds its own
 *        memo over the query-owned @c like_multiliteral_cache so both paths share one lookup.
 */
using like_classifier = std::function<like_multiliteral_cache::entry_ptr(std::string_view)>;

/**
 * @brief How constant LIKE / CONTAINS patterns are matched, shared with the materialize path.
 *
 * @c swar_fastpath false makes every pattern take the generic byte-wise matcher. A non-empty @c
 * classify classifies each pattern through the caller's memo instead of once per evaluation.
 */
struct like_policy {
  bool swar_fastpath{false};   ///< Honour the `like_swar_fastpath` setting.
  like_classifier classify{};  ///< Optional memoized classification shared with the evaluator.
};

/**
 * @brief Upper bound on the SWAR LIKE descriptors a single kernel takes as by-value parameters.
 */
inline constexpr std::size_t max_like_pattern_parameters = 8;

/**
 * @brief Compile-time state passed through produce/consume tracking input column bindings, loaded
 *        per-row Register Values, runtime scalar constant parameters, and emitted CUDA statements.
 *
 * The @c translation_context produces, and the sinks (@c evaluate_project_sink,
 * @c evaluate_filter_sink) consume. The contract between them is the generated text:
 *   - @c body_code() is a sequence of statements that evaluate the produced expressions for one
 *     row. It expects the row number in scope and the kernel parameters that @c kernel_abi
 *     emits for this context (`in_N_*`, `c_N_*`, `lp_N`, and `in_N_chars_end` from the kernel
 *     prelude), plus `bool row_subquery_error` when @c has_subquery_error() is true.
 *   - Every statement produced for a node defines a local `reg_val<T> v_N` named by the returned
 *     @c register_value::var_name; its `.val`, `.valid` and `.overflow` are what a sink reads.
 *   - Input loads and constant initializations go to a prelude that runs before any branch, so a
 *     later `if` arm never introduces a load; node statements follow in the body.
 *
 * The @c overflow_sites() point into the expression trees the context produced, so a context must
 * not outlive those trees, and its sites must never be cached with a compiled kernel in
 * `codegen::jit::KernelCache`, which outlives every tree. `evaluate_project_sink` and
 * `evaluate_filter_sink` create one context per launch.
 */
class translation_context {
 public:
  explicit translation_context(cudf::table_view input,
                               boundary_column_map const* boundary_map = nullptr,
                               like_policy like                        = {})
    : _input(input), _boundary_map(boundary_map), _like(like)
  {
  }

  /**
   * @brief Emits the @c produce code for an expression tree node and returns its @c register_value.
   *
   * Dispatches to the appropriate private specialized @c produce_* method based on the node type.
   *
   * @param expr The expression tree node to produce code for.
   * @return The @c register_value of the produced code.
   */
  [[nodiscard]] register_value produce(sirius::ast::node const& expr);

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
  /**
   * @brief Returns the overflow sites of the produced expressions in the order their statements
   * appear in @c body_code(), leaving out those under a TRY or TRY_CAST, which turns their overflow
   * into NULL before it can reach an NVRTC Sink.
   */
  [[nodiscard]] std::span<overflow_site const> overflow_sites() const noexcept
  {
    return _overflow_sites;
  }
  [[nodiscard]] std::string body_code() const { return _prelude + _body; }

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
  [[nodiscard]] register_value produce_arithmetic(sirius::ast::function_call const& expr);
  [[nodiscard]] register_value produce_extract(sirius::ast::function_call const& expr);
  [[nodiscard]] register_value produce_date_trunc(sirius::ast::function_call const& expr);
  [[nodiscard]] register_value produce_string_match(sirius::ast::function_call const& expr);
  [[nodiscard]] register_value produce_substring(sirius::ast::function_call const& expr);

  /** @brief Allocates a fresh local variable name (v_0, v_1, ...) in the kernel body. */
  [[nodiscard]] std::string next_var_name();
  /** @brief Appends a statement to the per-row prelude (before any branching). */
  void emit_prelude_line(std::string const& line);
  /** @brief Appends a statement to the per-row evaluation body. */
  void emit_line(std::string const& line);
  /** @brief Emits `reg_val<T> name{val, valid, overflow};` in the body. */
  void emit_reg_val(std::string_view type_name,
                    std::string_view name,
                    std::string_view val,
                    std::string_view valid,
                    std::string_view overflow);
  /** @brief Emits `reg_val<T> name = expr;` in the body. */
  void emit_reg_val_assign(std::string_view type_name,
                           std::string_view name,
                           std::string_view expr);

  /**
   * @brief Records @p site when its own operation can raise Overflow Poison, and returns whether
   * the operation's result can carry Overflow Poison, raised by the operation or passed on from its
   * inputs.
   */
  [[nodiscard]] bool record_overflow_site(overflow_site site,
                                          bool raises_overflow,
                                          bool inputs_can_overflow);

  /** @brief Returns the position from which `drop_overflow_sites_since` forgets overflow sites. */
  [[nodiscard]] std::size_t overflow_site_mark() const noexcept { return _overflow_sites.size(); }

  /**
   * @brief Forgets the overflow sites recorded after @p mark, under a TRY or TRY_CAST, which turns
   * their Overflow Poison into NULL before it can reach an NVRTC Sink.
   */
  void drop_overflow_sites_since(std::size_t mark) noexcept;

  /**
   * @brief Returns the narrowed physical carrier type when @p column_operand is a reference whose
   *        physical storage is a strict narrowing of its declared logical type and all @p
   *        constant_operands are exactly representable in that carrier.
   *
   * @param column_operand Candidate reference expression node.
   * @param constant_operands Constant operands compared against @p column_operand.
   * @return The narrowed carrier @c cudf::data_type if eligible, or @c std::nullopt otherwise.
   */
  [[nodiscard]] std::optional<cudf::data_type> narrow_domain_carrier(
    sirius::ast::node const& column_operand,
    std::initializer_list<sirius::ast::node const*> constant_operands) const;

  /**
   * @brief Returns the shared physical carrier type when @p lhs and @p rhs are both references that
   *        share an eligible narrowed carrier and logical domain.
   *
   * @param lhs Left-hand comparison operand.
   * @param rhs Right-hand comparison operand.
   * @return The shared narrowed carrier @c cudf::data_type if eligible, or @c std::nullopt
   * otherwise.
   */
  [[nodiscard]] std::optional<cudf::data_type> narrow_domain_reference_pair_carrier(
    sirius::ast::node const& lhs, sirius::ast::node const& rhs) const;

  /**
   * @brief Emits code for a comparison or BETWEEN operand, preserving the narrowed physical
   *        carrier when @p narrow_carrier is engaged or falling back to standard produce
   *        otherwise.
   *
   * @param operand The operand expression node to evaluate.
   * @param narrow_carrier Optional narrowed carrier type for narrow-domain evaluation.
   * @return The resulting `register_value` in GPU registers.
   */
  [[nodiscard]] register_value produce_narrow_domain_operand(
    sirius::ast::node const& operand, std::optional<cudf::data_type> narrow_carrier);

  /**
   * @brief Binds a numeric or decimal constant parameter narrowed to @p carrier and returns its
   *        @c register_value.
   *
   * @param expr The constant expression node (must be representable in @p carrier).
   * @param carrier The target narrowed physical carrier type.
   * @return The resulting `register_value` initialized from the narrowed constant parameter.
   */
  [[nodiscard]] register_value produce_constant_in_carrier(sirius::ast::constant const& expr,
                                                           cudf::data_type carrier);

  /**
   * @brief Returns the kernel input binding ordinal for @p column_index, creating a new
   *        @c input_column_binding if the column has not been bound yet.
   *
   * @param column_index Index of the column in @c _input.
   * @return Ordinal index into @c _input_columns.
   */
  [[nodiscard]] std::size_t get_or_bind_input_column(std::uint32_t column_index);

  /**
   * @brief Binds @p pattern as a multi-literal LIKE descriptor if the LIKE policy allows the SWAR
   *        matcher, the parameter budget is not exhausted, and the pattern is a constant the
   *        matcher supports; returns its binding id, or nullopt for the generic matcher.
   *
   * @param pattern The pattern to bind.
   * @param fid The function id of the LIKE function.
   * @return The binding id of the pattern, or nullopt for the generic matcher.
   */
  [[nodiscard]] std::optional<std::size_t> try_bind_like_pattern(sirius::ast::node const& pattern,
                                                                 sirius::function_id fid);

  cudf::table_view _input;
  boundary_column_map const* _boundary_map{nullptr};
  like_policy _like{};
  std::vector<input_column_binding> _input_columns;
  std::vector<loaded_reference_entry> _loaded_references;
  std::vector<scalar_constant_binding> _constants;
  std::vector<like_pattern_binding> _like_patterns;
  std::vector<overflow_site> _overflow_sites;
  std::size_t _narrow_domain_comparison_count{0};
  bool _has_subquery_error{false};
  std::string _prelude;
  std::string _body;
  std::size_t _next_var_id{0};
};

}  // namespace sirius::nvrtc
