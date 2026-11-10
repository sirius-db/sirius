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

#include <cudf/cudf_utils.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <expression/date_trunc_unit.hpp>
#include <expression/function_id.hpp>
#include <expression/substring_slice.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/register_value.hpp>
#include <expression_evaluator/nvrtc/supportability.hpp>
#include <helper/numeric_narrowing.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <variant>

namespace sirius::nvrtc {

namespace {

[[nodiscard]] bool has_supported_string_offsets(cudf::column_view const& col_view)
{
  if (col_view.num_children() == 0) { return true; }
  auto const off_id = cudf::strings_column_view(col_view).offsets().type().id();
  return off_id == cudf::type_id::INT32 || off_id == cudf::type_id::INT64;
}

/// @brief STRING only mixes with STRING and fixed-point only with fixed-point; everything else
/// mixes.
[[nodiscard]] bool types_compatible(cudf::data_type a, cudf::data_type b) noexcept
{
  bool const a_str = a.id() == cudf::type_id::STRING;
  bool const b_str = b.id() == cudf::type_id::STRING;
  if (a_str || b_str) { return a_str && b_str; }
  bool const a_dec = cudf::is_fixed_point(a);
  bool const b_dec = cudf::is_fixed_point(b);
  if (a_dec || b_dec) { return a_dec && b_dec; }
  return true;
}

/**
 * @brief A CASE / COALESCE arm must additionally share the result's decimal scale, since the arm is
 *        copied into the result register without rescaling.
 */
[[nodiscard]] bool arm_compatible_with_result(cudf::data_type result, cudf::data_type arm) noexcept
{
  if (!is_supported_register_type(arm) || !types_compatible(result, arm)) { return false; }
  return !cudf::is_fixed_point(result) || result.scale() == arm.scale();
}

/** @brief The CAST gate */
[[nodiscard]] bool is_semantic_cast_supported(cudf::data_type src, cudf::data_type dst) noexcept
{
  if (!is_supported_fixed_width(src) || !is_supported_fixed_width(dst)) { return false; }
  bool const src_temp = cudf::is_timestamp(src);
  bool const dst_temp = cudf::is_timestamp(dst);
  if (src_temp != dst_temp) { return false; }
  if (src_temp) {
    if (src.id() == dst.id()) { return true; }
    if (src.id() == cudf::type_id::TIMESTAMP_DAYS) {
      return timestamp_ticks_per_second(dst.id()) > 0;
    }
    return timestamp_ticks_per_second(src.id()) > 0 &&
           (dst.id() == cudf::type_id::TIMESTAMP_DAYS || timestamp_ticks_per_second(dst.id()) > 0);
  }
  return true;
}

/// @brief The Materialization Boundaries a planning walk has met, in the order it met them.
using boundary_list = std::vector<sirius::ast::node const*>;

/**
 * @brief Returns the cuDF type a node evaluates to in registers: a reference's restored logical
 *        type (or its physical carrier when it is not a narrowed numeric), otherwise the node's
 *        declared return type, which is also the type of a Materialization Boundary's evaluated
 *        column. `EMPTY` when it has none.
 */
[[nodiscard]] cudf::data_type effective_node_cudf_type(sirius::ast::node const& expr,
                                                       cudf::table_view const& input)
{
  if (expr.holds<sirius::ast::reference>()) {
    auto const& ref = expr.get<sirius::ast::reference>();
    if (ref.column_index >= static_cast<std::uint32_t>(input.num_columns())) {
      return cudf::data_type{cudf::type_id::EMPTY};
    }
    auto const col_view = input.column(ref.column_index);
    auto const& logical = ref.return_type();
    if (is_untyped_reference_type(logical)) { return col_view.type(); }
    auto const opt_native = sirius::get_cudf_type(logical);
    if (!opt_native) { return cudf::data_type{cudf::type_id::EMPTY}; }
    if (is_narrowable_numeric_type(logical) && can_restore_to(col_view.type(), *opt_native)) {
      return *opt_native;
    }
    return col_view.type();
  }
  auto const opt_cudf = sirius::get_cudf_type(expr.return_type());
  return opt_cudf.value_or(cudf::data_type{cudf::type_id::EMPTY});
}

/** @brief Returns whether @p expr is a Materialization Boundary, as `compile_plan` defines one. */
[[nodiscard]] bool is_materialization_boundary(sirius::ast::node const& expr,
                                               cudf::table_view const& input)
{
  if (expr.holds<sirius::ast::reference>() || expr.holds<sirius::ast::constant>() ||
      expr.holds<sirius::ast::aggregate>()) {
    return false;
  }
  if (expr.holds<sirius::ast::function_call>()) {
    auto const fid = expr.get<sirius::ast::function_call>().function();
    return fid == sirius::function_id::concat || fid == sirius::function_id::concat_operator ||
           fid == sirius::function_id::upper || fid == sirius::function_id::lower ||
           fid == sirius::function_id::regexp_replace;
  }
  if (expr.holds<sirius::ast::cast>()) {
    auto const& cst    = expr.get<sirius::ast::cast>();
    auto const opt_dst = sirius::get_cudf_type(cst.target_type);
    if (!opt_dst || !is_supported_register_type(*opt_dst)) { return false; }
    if (!cst.child) { return false; }
    auto const src_type = effective_node_cudf_type(*cst.child, input);
    if (cst.kind == sirius::ast::cast_kind::carrier_restore) {
      return !(src_type == *opt_dst || can_restore_to(src_type, *opt_dst));
    }
    return !is_semantic_cast_supported(src_type, *opt_dst);
  }
  return false;
}

/**
 * @brief Records @p boundary in @p boundaries and returns whether a kernel can read its evaluated
 *        column, whose type is the boundary's declared return type.
 */
[[nodiscard]] bool bind_placeholder(sirius::ast::node const& boundary, boundary_list& boundaries)
{
  auto const opt_cudf = sirius::get_cudf_type(boundary.return_type());
  if (!opt_cudf || !is_supported_register_type(*opt_cudf)) { return false; }
  if (std::ranges::find(boundaries, &boundary) == boundaries.end()) {
    boundaries.push_back(&boundary);
  }
  return true;
}

[[nodiscard]] bool is_node_supported(sirius::ast::node const& expr,
                                     cudf::table_view const& input,
                                     boundary_list& boundaries);

[[nodiscard]] bool is_function_call_supported(sirius::ast::function_call const& fc,
                                              cudf::table_view const& input,
                                              boundary_list& boundaries)
{
  auto const& args   = fc.arguments();
  auto const opt_ret = sirius::get_cudf_type(fc.return_type());
  if (!opt_ret || !is_supported_register_type(*opt_ret)) { return false; }
  auto const ret_type = *opt_ret;

  if (fc.function() == sirius::function_id::date_trunc) {
    if (args.size() != 2) { return false; }
    if (!args[0] || !args[0]->holds<sirius::ast::constant>()) { return false; }
    auto const& freq_cst = args[0]->get<sirius::ast::constant>();
    if (!std::holds_alternative<std::string>(freq_cst.payload)) { return false; }
    auto const unit = sirius::parse_gpu_date_trunc_unit(std::get<std::string>(freq_cst.payload));
    if (!unit) { return false; }
    if (!args[1] || !is_node_supported(*args[1], input, boundaries)) { return false; }
    auto const ct = effective_node_cudf_type(*args[1], input);
    return cudf::is_timestamp(ct) && cudf::is_timestamp(ret_type);
  }

  if (fc.function() == sirius::function_id::substring) {
    if ((args.size() != 2 && args.size() != 3) || ret_type.id() != cudf::type_id::STRING) {
      return false;
    }
    if (!args[0] || !is_node_supported(*args[0], input, boundaries)) { return false; }
    if (effective_node_cudf_type(*args[0], input).id() != cudf::type_id::STRING) { return false; }
    auto const opt_offset = args[1] ? extract_constant_int64(*args[1]) : std::nullopt;
    if (!opt_offset) { return false; }
    std::int64_t length = sirius::substring_default_length;
    if (args.size() == 3) {
      auto const opt_len = args[2] ? extract_constant_int64(*args[2]) : std::nullopt;
      if (!opt_len) { return false; }
      length = *opt_len;
    }
    return sirius::gpu_substring_slice(*opt_offset, length).has_value();
  }

  for (auto const& arg : args) {
    if (!arg || !is_node_supported(*arg, input, boundaries)) { return false; }
  }

  switch (fc.function()) {
    case sirius::function_id::add:
    case sirius::function_id::sub:
    case sirius::function_id::mul:
    case sirius::function_id::div:
    case sirius::function_id::int_div:
    case sirius::function_id::mod: {
      if (args.size() != 2) { return false; }
      auto const lt = effective_node_cudf_type(*args[0], input);
      auto const rt = effective_node_cudf_type(*args[1], input);
      if (cudf::is_fixed_point(ret_type)) {
        return cudf::is_fixed_point(lt) && cudf::is_fixed_point(rt);
      }
      return is_integer_or_float(ret_type) && is_integer_or_float(lt) && is_integer_or_float(rt);
    }
    case sirius::function_id::abs: {
      if (args.size() != 1) { return false; }
      auto const ct = effective_node_cudf_type(*args[0], input);
      if (cudf::is_fixed_point(ret_type)) { return cudf::is_fixed_point(ct) && ct == ret_type; }
      return is_integer_or_float(ret_type) && ct == ret_type;
    }
    case sirius::function_id::negate: {
      if (args.size() != 1) { return false; }
      auto const ct = effective_node_cudf_type(*args[0], input);
      if (ct != ret_type) { return false; }
      if (cudf::is_fixed_point(ret_type)) { return true; }
      return is_integer_or_float(ret_type) && !cudf::is_unsigned(ret_type);
    }
    case sirius::function_id::round: {
      if (args.size() != 1 && args.size() != 2) { return false; }
      auto const ct = effective_node_cudf_type(*args[0], input);
      if ((ct.id() != cudf::type_id::FLOAT32 && ct.id() != cudf::type_id::FLOAT64) ||
          ret_type != ct) {
        return false;
      }
      if (args.size() == 2) {
        if (!args[1]->holds<sirius::ast::constant>()) { return false; }
        auto const& prec_cst = args[1]->get<sirius::ast::constant>();
        if (!std::holds_alternative<std::int32_t>(prec_cst.payload)) { return false; }
      }
      return true;
    }
    case sirius::function_id::constant_or_null: {
      if (args.size() < 2) { return false; }
      auto const v0_t = effective_node_cudf_type(*args[0], input);
      if (ret_type.id() == cudf::type_id::STRING || v0_t.id() == cudf::type_id::STRING) {
        return ret_type.id() == cudf::type_id::STRING && v0_t.id() == cudf::type_id::STRING;
      }
      return is_supported_fixed_width(v0_t);
    }
    case sirius::function_id::year:
    case sirius::function_id::month:
    case sirius::function_id::day:
    case sirius::function_id::hour:
    case sirius::function_id::minute:
    case sirius::function_id::second:
    case sirius::function_id::millisecond:
    case sirius::function_id::microsecond: {
      if (args.size() != 1) { return false; }
      auto const ct = effective_node_cudf_type(*args[0], input);
      return cudf::is_timestamp(ct) && cudf::is_integral(ret_type);
    }
    case sirius::function_id::strlen:
    case sirius::function_id::length: {
      if (args.size() != 1) { return false; }
      auto const ct = effective_node_cudf_type(*args[0], input);
      return ct.id() == cudf::type_id::STRING && cudf::is_integral(ret_type);
    }
    case sirius::function_id::prefix:
    case sirius::function_id::suffix:
    case sirius::function_id::contains:
    case sirius::function_id::like:
    case sirius::function_id::not_like: {
      if (args.size() != 2 || ret_type.id() != cudf::type_id::BOOL8) { return false; }
      auto const lt = effective_node_cudf_type(*args[0], input);
      auto const rt = effective_node_cudf_type(*args[1], input);
      return lt.id() == cudf::type_id::STRING && rt.id() == cudf::type_id::STRING;
    }
    default: return false;
  }
}

/**
 * @brief Returns whether `translation_context::produce` can translate @p expr over @p input once
 *        every Materialization Boundary in it is read from an input column, recording those
 *        boundaries in @p boundaries.
 */
bool is_node_supported(sirius::ast::node const& expr,
                       cudf::table_view const& input,
                       boundary_list& boundaries)
{
  //===----------BOUND REFERENCE----------===//
  if (expr.holds<sirius::ast::reference>()) {
    auto const& ref = expr.get<sirius::ast::reference>();
    if (ref.column_index >= static_cast<std::uint32_t>(input.num_columns())) { return false; }
    auto const col_view = input.column(ref.column_index);
    auto const& logical = ref.return_type();
    if (col_view.type().id() == cudf::type_id::STRING) {
      if (!has_supported_string_offsets(col_view)) { return false; }
      return is_untyped_reference_type(logical) || logical.id() == sirius::type_id::VARCHAR;
    }
    if (!is_supported_fixed_width(col_view.type())) { return false; }
    if (!is_untyped_reference_type(logical)) {
      auto const opt_cudf = sirius::get_cudf_type(logical);
      if (!opt_cudf || !is_supported_fixed_width(*opt_cudf)) { return false; }
      if (col_view.type() != *opt_cudf) {
        if (!is_narrowable_numeric_type(logical) || !can_restore_to(col_view.type(), *opt_cudf)) {
          return false;
        }
      }
    }
    return true;
  }

  //===----------CONSTANT----------===//
  if (expr.holds<sirius::ast::constant>()) {
    auto const& cst     = expr.get<sirius::ast::constant>();
    auto const opt_cudf = sirius::get_cudf_type(cst.return_type());
    return opt_cudf.has_value() && is_supported_register_type(*opt_cudf);
  }

  //===----------COMPARISON----------===//
  if (expr.holds<sirius::ast::comparison>()) {
    auto const& cmp = expr.get<sirius::ast::comparison>();
    if (!cmp.left || !cmp.right) { return false; }
    if (!is_node_supported(*cmp.left, input, boundaries) ||
        !is_node_supported(*cmp.right, input, boundaries)) {
      return false;
    }
    auto const lt = effective_node_cudf_type(*cmp.left, input);
    auto const rt = effective_node_cudf_type(*cmp.right, input);
    return is_supported_register_type(lt) && is_supported_register_type(rt) &&
           types_compatible(lt, rt);
  }

  //===----------BETWEEN----------===//
  if (expr.holds<sirius::ast::between>()) {
    auto const& btw = expr.get<sirius::ast::between>();
    if (!btw.input || !btw.lower || !btw.upper) { return false; }
    if (!is_node_supported(*btw.input, input, boundaries) ||
        !is_node_supported(*btw.lower, input, boundaries) ||
        !is_node_supported(*btw.upper, input, boundaries)) {
      return false;
    }
    auto const it = effective_node_cudf_type(*btw.input, input);
    auto const lt = effective_node_cudf_type(*btw.lower, input);
    auto const ut = effective_node_cudf_type(*btw.upper, input);
    return is_supported_register_type(it) && is_supported_register_type(lt) &&
           is_supported_register_type(ut) && types_compatible(it, lt) && types_compatible(it, ut);
  }

  //===----------CONJUNCTION----------===//
  if (expr.holds<sirius::ast::conjunction>()) {
    auto const& conj = expr.get<sirius::ast::conjunction>();
    if (conj.children.empty()) { return false; }
    if (conj.op != sirius::ast::conjunction::kind::op_and &&
        conj.op != sirius::ast::conjunction::kind::op_or) {
      return false;
    }
    return std::ranges::all_of(conj.children, [&](auto const& ch) {
      return ch && is_node_supported(*ch, input, boundaries) &&
             effective_node_cudf_type(*ch, input).id() == cudf::type_id::BOOL8;
    });
  }

  //===----------CASE----------===//
  if (expr.holds<sirius::ast::case_expr>()) {
    auto const& ce = expr.get<sirius::ast::case_expr>();
    if (ce.cases.empty() || !ce.else_) { return false; }
    auto const opt_ret = sirius::get_cudf_type(ce.return_type());
    if (!opt_ret || !is_supported_register_type(*opt_ret)) { return false; }
    auto const ret_type = *opt_ret;
    for (auto const& c : ce.cases) {
      if (!c.when_ || !c.then_) { return false; }
      if (!is_node_supported(*c.when_, input, boundaries)) { return false; }
      if (effective_node_cudf_type(*c.when_, input).id() != cudf::type_id::BOOL8) { return false; }
      if (c.then_->holds<sirius::ast::function_call>() &&
          c.then_->get<sirius::ast::function_call>().function() == sirius::function_id::error) {
        continue;
      }
      if (!is_node_supported(*c.then_, input, boundaries)) { return false; }
      if (!arm_compatible_with_result(ret_type, effective_node_cudf_type(*c.then_, input))) {
        return false;
      }
    }
    if (!is_node_supported(*ce.else_, input, boundaries)) { return false; }
    return arm_compatible_with_result(ret_type, effective_node_cudf_type(*ce.else_, input));
  }

  //===----------COALESCE----------===//
  if (expr.holds<sirius::ast::coalesce>()) {
    auto const& coal = expr.get<sirius::ast::coalesce>();
    if (coal.children.empty()) { return false; }
    auto const opt_ret = sirius::get_cudf_type(coal.return_type());
    if (!opt_ret || !is_supported_register_type(*opt_ret)) { return false; }
    auto const ret_type = *opt_ret;
    return std::ranges::all_of(coal.children, [&](auto const& ch) {
      return ch && is_node_supported(*ch, input, boundaries) &&
             arm_compatible_with_result(ret_type, effective_node_cudf_type(*ch, input));
    });
  }

  //===----------IN_LIST----------===//
  if (expr.holds<sirius::ast::in_list>()) {
    auto const& inl = expr.get<sirius::ast::in_list>();
    if (!inl.probe || inl.values.empty()) { return false; }
    if (!is_node_supported(*inl.probe, input, boundaries)) { return false; }
    auto const pt = effective_node_cudf_type(*inl.probe, input);
    if (!is_supported_register_type(pt)) { return false; }
    return std::ranges::all_of(inl.values, [&](auto const& v) {
      if (!v || !is_node_supported(*v, input, boundaries)) { return false; }
      auto const vt = effective_node_cudf_type(*v, input);
      return is_supported_register_type(vt) && types_compatible(pt, vt);
    });
  }

  //===----------CAST----------===//
  if (expr.holds<sirius::ast::cast>()) {
    if (is_materialization_boundary(expr, input)) { return bind_placeholder(expr, boundaries); }
    auto const& cst = expr.get<sirius::ast::cast>();
    if (!cst.child || !is_node_supported(*cst.child, input, boundaries)) { return false; }
    auto const opt_dst = sirius::get_cudf_type(cst.target_type);
    if (!opt_dst || !is_supported_fixed_width(*opt_dst)) { return false; }
    auto const dst_type = *opt_dst;
    auto const src_type = effective_node_cudf_type(*cst.child, input);
    if (!is_supported_fixed_width(src_type)) { return false; }
    if (cst.kind == sirius::ast::cast_kind::carrier_restore) {
      return src_type == dst_type || can_restore_to(src_type, dst_type);
    }
    // UHUGEINT is carried in UINT64 on the GPU. A TRY_CAST row beyond that carrier is a valid
    // UHUGEINT, so it must request CPU Replay rather than become NULL, which the kernel cannot do.
    if (cst.try_cast && cst.target_type.id() == sirius::type_id::UHUGEINT) { return false; }
    return is_semantic_cast_supported(src_type, dst_type);
  }

  //===----------UNARY_OP----------===//
  if (expr.holds<sirius::ast::unary_op>()) {
    auto const& uo = expr.get<sirius::ast::unary_op>();
    if (!uo.child || !is_node_supported(*uo.child, input, boundaries)) { return false; }
    switch (uo.op) {
      case sirius::ast::unary_op::kind::op_not:
        return effective_node_cudf_type(*uo.child, input).id() == cudf::type_id::BOOL8;
      case sirius::ast::unary_op::kind::op_is_null:
      case sirius::ast::unary_op::kind::op_is_not_null: return true;
      case sirius::ast::unary_op::kind::op_try:
        return is_supported_register_type(effective_node_cudf_type(*uo.child, input));
      default: return false;
    }
  }

  //===----------FUNCTION_CALL----------===//
  if (expr.holds<sirius::ast::function_call>()) {
    if (is_materialization_boundary(expr, input)) { return bind_placeholder(expr, boundaries); }
    return is_function_call_supported(expr.get<sirius::ast::function_call>(), input, boundaries);
  }

  return false;
}

/**
 * @brief Plans one kernel whose outputs are @p roots; @p is_output_type tells which root types
 *        the kernel can write.
 */
template <typename OutputTypePredicate>
[[nodiscard]] std::optional<compile_plan> plan_sink(std::span<sirius::ast::node const* const> roots,
                                                    cudf::table_view const& input,
                                                    OutputTypePredicate is_output_type) noexcept
{
  try {
    if (roots.empty()) { return std::nullopt; }
    compile_plan plan;
    for (auto const* root : roots) {
      if (root == nullptr || !is_output_type(effective_node_cudf_type(*root, input))) {
        return std::nullopt;
      }
      // A root that is itself a boundary stays on the caller's path: a kernel that only copied its
      // evaluated column would gain nothing.
      if (is_materialization_boundary(*root, input) ||
          !is_node_supported(*root, input, plan.boundaries)) {
        return std::nullopt;
      }
    }
    return plan;
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace

std::int64_t timestamp_ticks_per_second(cudf::type_id id) noexcept
{
  switch (id) {
    case cudf::type_id::TIMESTAMP_SECONDS: return 1LL;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return 1'000LL;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return 1'000'000LL;
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return 1'000'000'000LL;
    default: return 0LL;
  }
}

std::optional<std::int64_t> extract_constant_int64(sirius::ast::node const& node) noexcept
{
  if (!node.holds<sirius::ast::constant>()) { return std::nullopt; }
  auto const& payload = node.get<sirius::ast::constant>().payload;
  if (std::holds_alternative<std::int64_t>(payload)) { return std::get<std::int64_t>(payload); }
  if (std::holds_alternative<std::int32_t>(payload)) {
    return static_cast<std::int64_t>(std::get<std::int32_t>(payload));
  }
  return std::nullopt;
}

cudf::data_type resolve_output_cudf_type(sirius::ast::node const& expr,
                                         cudf::table_view const& input)
{
  if (expr.holds<sirius::ast::reference>()) {
    auto const& ref     = expr.get<sirius::ast::reference>();
    auto const col_view = input.column(ref.column_index);
    auto const& logical = ref.return_type();
    if (is_untyped_reference_type(logical)) { return col_view.type(); }
    auto const native = sirius::require_cudf_type(logical);
    if (is_narrowable_numeric_type(logical) && can_restore_to(col_view.type(), native)) {
      return native;
    }
    return col_view.type();
  }
  if (expr.holds<sirius::ast::unary_op>() &&
      expr.get<sirius::ast::unary_op>().op == sirius::ast::unary_op::kind::op_try) {
    return resolve_output_cudf_type(*expr.get<sirius::ast::unary_op>().child, input);
  }
  return sirius::require_cudf_type(expr.return_type());
}

std::optional<compile_plan> plan_project_sink(std::span<sirius::ast::node const* const> expressions,
                                              cudf::table_view const& input) noexcept
{
  return plan_sink(
    expressions, input, [](cudf::data_type t) { return is_supported_fixed_width(t); });
}

std::optional<compile_plan> plan_filter_sink(sirius::ast::node const& predicate,
                                             cudf::table_view const& input) noexcept
{
  std::array<sirius::ast::node const*, 1> const roots{&predicate};
  return plan_sink(roots, input, [](cudf::data_type t) { return t.id() == cudf::type_id::BOOL8; });
}

bool is_supported_boundary_column(cudf::column_view const& column) noexcept
{
  try {
    if (column.type().id() == cudf::type_id::STRING) {
      return has_supported_string_offsets(column);
    }
    return is_supported_fixed_width(column.type());
  } catch (...) {
    return false;
  }
}

}  // namespace sirius::nvrtc
