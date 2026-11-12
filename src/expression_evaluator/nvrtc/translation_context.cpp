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

#include <expression/ast/constant_range.hpp>
#include <expression/date_trunc_unit.hpp>
#include <expression/substring_slice.hpp>
#include <expression_evaluator/nvrtc/device_runtime/abi_constants.cuh>
#include <expression_evaluator/nvrtc/kernel_abi.hpp>
#include <expression_evaluator/nvrtc/supportability.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <iterator>
#include <limits>
#include <variant>

namespace sirius::nvrtc {

namespace {

using std::format;

cudf::data_type const bool8_type{cudf::type_id::BOOL8};

[[nodiscard]] std::int32_t sql_decimal_scale(cudf::data_type type) noexcept
{
  return -type.scale();
}

/** @brief 10^exp as an __int128_t literal, for resolving scale gaps between decimals. */
/**
 * @brief Decimal digits available to @p type: its SQL precision, or the 38 digits of the DECIMAL128
 *        carrier for HUGEINT, which has no SQL precision of its own.
 */
[[nodiscard]] int target_decimal_precision(sirius::logical_type const& type)
{
  if (type.id() == sirius::type_id::HUGEINT) { return 38; }
  return static_cast<int>(type.decimal_precision());
}

[[nodiscard]] std::string format_int128_pow10(int exp)
{
  if (exp <= 0) { return "static_cast<__int128_t>(1)"; }
  __int128_t val = 1;
  for (int i = 0; i < exp; ++i) {
    val *= 10;
  }
  auto const uval = static_cast<unsigned __int128>(val);
  auto const hi   = static_cast<std::uint64_t>(uval >> 64);
  auto const lo   = static_cast<std::uint64_t>(uval);
  if (hi == 0) { return format("static_cast<__int128_t>({}ULL)", lo); }
  return format(
    "((static_cast<__int128_t>({}ULL) << 64) | static_cast<__int128_t>({}ULL))", hi, lo);
}

/** @brief 10^exp as a double literal, for casts that multiply or divide in floating point. */
[[nodiscard]] std::string format_double_pow10(int exp)
{
  double val = 1.0;
  for (int i = 0; i < exp; ++i) {
    val *= 10.0;
  }
  return format("{:.17e}", val);
}

[[nodiscard]] std::int64_t date_trunc_divisor(cudf::type_id id,
                                              sirius::date_trunc_unit unit) noexcept
{
  std::int64_t const tps = timestamp_ticks_per_second(id);
  if (tps <= 0) { return 1LL; }
  switch (unit) {
    case sirius::date_trunc_unit::day: return 86'400LL * tps;
    case sirius::date_trunc_unit::hour: return 3'600LL * tps;
    case sirius::date_trunc_unit::minute: return 60LL * tps;
    case sirius::date_trunc_unit::second: return tps;
    case sirius::date_trunc_unit::millisecond: return (tps >= 1'000LL) ? (tps / 1'000LL) : 1LL;
    case sirius::date_trunc_unit::microsecond:
      return (tps >= 1'000'000LL) ? (tps / 1'000'000LL) : 1LL;
  }
  return 1LL;
}

/**
 * @brief The common type two register payloads are compared in.
 *
 * Decimals of different scales are compared as __int128_t after rescaling; mixed-width integers
 * widen to 64 bits (or 128 when the signedness differs); any float operand makes the comparison
 * floating point.
 */
[[nodiscard]] std::string comparison_type_name(cudf::data_type lt, cudf::data_type rt)
{
  if (cudf::is_fixed_point(lt) || cudf::is_fixed_point(rt)) {
    return lt == rt ? cuda_type_name(lt) : "__int128_t";
  }
  if (lt == rt) { return cuda_type_name(lt); }
  if (lt.id() == cudf::type_id::FLOAT64 || rt.id() == cudf::type_id::FLOAT64) { return "double"; }
  if (lt.id() == cudf::type_id::FLOAT32 || rt.id() == cudf::type_id::FLOAT32) { return "float"; }
  if (cudf::is_unsigned(lt) && cudf::is_unsigned(rt)) { return "uint64_t"; }
  if (!cudf::is_unsigned(lt) && !cudf::is_unsigned(rt)) { return "int64_t"; }
  return "__int128_t";
}

/**
 * @brief static_cast<T>(v_N.val), multiplied by 10^(target_scale - own_scale) for a decimal that
 *        must be brought to a larger scale.
 */
[[nodiscard]] std::string comparison_operand(register_value const& reg,
                                             std::string const& cmp_type,
                                             std::int32_t own_scale,
                                             std::int32_t target_scale)
{
  auto operand = format("static_cast<{}>({}.val)", cmp_type, reg.var_name);
  if (target_scale > own_scale) {
    operand = format("({} * {})", operand, format_int128_pow10(target_scale - own_scale));
  }
  return operand;
}

/**
 * @brief The boolean expression comparing @p lhs with @p rhs, both already rendered as
 *        @p cmp_type. Floating-point comparisons follow DuckDB: NaN equals NaN and orders above
 *        every other value.
 */
[[nodiscard]] std::string comparison_text(sirius::comparison_type op,
                                          std::string const& lhs,
                                          std::string const& rhs,
                                          std::string const& cmp_type)
{
  bool const floating = cmp_type == "float" || cmp_type == "double";
  if (floating) {
    switch (op) {
      case sirius::comparison_type::equal:
      case sirius::comparison_type::not_distinct_from:
        return format("eval_fp_eq({}, {})", lhs, rhs);
      case sirius::comparison_type::not_equal:
      case sirius::comparison_type::distinct_from: return format("!eval_fp_eq({}, {})", lhs, rhs);
      case sirius::comparison_type::lt: return format("eval_fp_lt({}, {})", lhs, rhs);
      case sirius::comparison_type::le: return format("eval_fp_le({}, {})", lhs, rhs);
      case sirius::comparison_type::gt: return format("eval_fp_lt({}, {})", rhs, lhs);
      case sirius::comparison_type::ge: return format("eval_fp_le({}, {})", rhs, lhs);
      default: break;
    }
  } else {
    char const* op_sym = nullptr;
    switch (op) {
      case sirius::comparison_type::equal:
      case sirius::comparison_type::not_distinct_from: op_sym = "=="; break;
      case sirius::comparison_type::not_equal:
      case sirius::comparison_type::distinct_from: op_sym = "!="; break;
      case sirius::comparison_type::lt: op_sym = "<"; break;
      case sirius::comparison_type::le: op_sym = "<="; break;
      case sirius::comparison_type::gt: op_sym = ">"; break;
      case sirius::comparison_type::ge: op_sym = ">="; break;
      default: break;
    }
    if (op_sym != nullptr) { return format("({} {} {})", lhs, op_sym, rhs); }
  }
  throw internal_exception("[nvrtc] Unrecognized comparison operator: {}", static_cast<int>(op));
}

[[nodiscard]] std::string overflow_expression(std::initializer_list<register_value const*> regs)
{
  std::string expr;
  for (auto const* r : regs) {
    if (!r->can_overflow) { continue; }
    if (!expr.empty()) { expr += " || "; }
    expr += r->var_name + ".overflow";
  }
  return expr.empty() ? "false" : expr;
}

[[nodiscard]] bool any_can_overflow(std::initializer_list<register_value const*> regs)
{
  return std::ranges::any_of(regs, [](auto const* r) { return r->can_overflow; });
}

[[nodiscard]] bool all_statically_non_null(std::initializer_list<register_value const*> regs)
{
  return std::ranges::all_of(regs, [](auto const* r) { return r->is_statically_non_null; });
}

[[nodiscard]] register_value make_bool_register(std::string var_name,
                                                bool is_statically_non_null,
                                                bool can_overflow)
{
  return register_value{std::move(var_name), bool8_type, is_statically_non_null, can_overflow};
}

[[nodiscard]] char const* extract_component_name(abi::extract_component comp)
{
  switch (comp) {
    case abi::extract_component::YEAR: return "YEAR";
    case abi::extract_component::MONTH: return "MONTH";
    case abi::extract_component::DAY: return "DAY";
    case abi::extract_component::HOUR: return "HOUR";
    case abi::extract_component::MINUTE: return "MINUTE";
    case abi::extract_component::SECOND: return "SECOND";
    case abi::extract_component::MILLISECOND: return "MILLISECOND";
    case abi::extract_component::MICROSECOND: return "MICROSECOND";
  }
  throw internal_exception("[nvrtc] unknown extract component {}", static_cast<int>(comp));
}

[[nodiscard]] abi::extract_component extract_component_of(sirius::function_id fid)
{
  switch (fid) {
    case sirius::function_id::year: return abi::extract_component::YEAR;
    case sirius::function_id::month: return abi::extract_component::MONTH;
    case sirius::function_id::day: return abi::extract_component::DAY;
    case sirius::function_id::hour: return abi::extract_component::HOUR;
    case sirius::function_id::minute: return abi::extract_component::MINUTE;
    case sirius::function_id::second: return abi::extract_component::SECOND;
    case sirius::function_id::millisecond: return abi::extract_component::MILLISECOND;
    case sirius::function_id::microsecond: return abi::extract_component::MICROSECOND;
    default:
      throw internal_exception("[nvrtc] function {} is not a temporal extraction",
                               static_cast<int>(fid));
  }
}

}  // namespace

std::string translation_context::next_var_name() { return format("v_{}", _next_var_id++); }

void translation_context::emit_prelude_line(std::string const& line)
{
  _prelude.append("    ").append(line).push_back('\n');
}

void translation_context::emit_line(std::string const& line)
{
  _body.append("    ").append(line).push_back('\n');
}

void translation_context::emit_reg_val(std::string_view type_name,
                                       std::string_view name,
                                       std::string_view val,
                                       std::string_view valid,
                                       std::string_view overflow)
{
  emit_line(format("reg_val<{}> {}{{{}, {}, {}}};", type_name, name, val, valid, overflow));
}

void translation_context::emit_reg_val_assign(std::string_view type_name,
                                              std::string_view name,
                                              std::string_view expr)
{
  emit_line(format("reg_val<{}> {} = {};", type_name, name, expr));
}

bool translation_context::record_overflow_site(overflow_site site,
                                               bool raises_overflow,
                                               bool inputs_can_overflow)
{
  if (raises_overflow) { _overflow_sites.push_back(site); }
  return raises_overflow || inputs_can_overflow;
}

void translation_context::drop_overflow_sites_since(std::size_t mark) noexcept
{
  _overflow_sites.resize(mark);
}

std::size_t translation_context::get_or_bind_input_column(std::uint32_t column_index)
{
  auto const it =
    std::ranges::find(_input_columns, column_index, &input_column_binding::column_index);
  if (it != _input_columns.end()) { return it->binding_id; }

  auto const col_view    = _input.column(column_index);
  auto const binding_id  = _input_columns.size();
  bool const is_nullable = col_view.nullable() && col_view.null_count() != 0;
  cudf::data_type offset_type{cudf::type_id::INT32};
  if (col_view.type().id() == cudf::type_id::STRING && col_view.num_children() > 0) {
    offset_type = cudf::strings_column_view(col_view).offsets().type();
  }
  _input_columns.push_back(
    input_column_binding{column_index, col_view.type(), offset_type, is_nullable, binding_id});
  return binding_id;
}

std::optional<std::size_t> translation_context::try_bind_like_pattern(
  sirius::ast::node const& pattern, sirius::function_id fid)
{
  if (!_like.swar_fastpath || _like_patterns.size() >= max_like_pattern_parameters) {
    return std::nullopt;
  }
  if (!pattern.holds<sirius::ast::constant>()) { return std::nullopt; }
  auto const& cst = pattern.get<sirius::ast::constant>();
  if (!std::holds_alternative<std::string>(cst.payload)) { return std::nullopt; }
  auto const& text = std::get<std::string>(cst.payload);
  // `contains(s, lit)` is the one-literal `%lit%` pattern; `%` and `_` are plain bytes there.
  std::string like_text;
  if (fid == sirius::function_id::contains) {
    if (text.empty() || text.find_first_of("%_") != std::string::npos) { return std::nullopt; }
    like_text = "%" + text + "%";
  } else {
    like_text = text;
  }
  sirius::detail::like_multiliteral_pattern_desc descriptor{};
  if (_like.classify) {
    auto const entry = _like.classify(like_text);
    if (!entry || !*entry) { return std::nullopt; }
    descriptor = (*entry)->descriptor();
  } else {
    auto const parsed = sirius::classify_like_multiliteral(like_text, std::string_view{});
    if (!parsed) { return std::nullopt; }
    descriptor = parsed->descriptor();
  }
  auto const binding_id = _like_patterns.size();
  _like_patterns.push_back(like_pattern_binding{binding_id, descriptor});
  return binding_id;
}

//===-------------------------------------------------------------------------===//
// LEAF NODES:
//  - COLUMN REFERENCE
//  - CONSTANT
//===-------------------------------------------------------------------------===//

//===----------COLUMN REFERENCE----------===//
register_value translation_context::produce_reference(sirius::ast::reference const& expr)
{
  auto const col_view           = _input.column(expr.column_index);
  auto const& logical           = expr.return_type();
  cudf::data_type restored_type = col_view.type();

  if (!is_untyped_reference_type(logical)) {
    auto const native = sirius::require_cudf_type(logical);
    if (is_narrowable_numeric_type(logical) && can_restore_to(col_view.type(), native)) {
      restored_type = native;
    }
  }

  auto const existing = std::ranges::find_if(_loaded_references, [&](auto const& entry) {
    return entry.column_index == expr.column_index && entry.restored_type == restored_type;
  });
  if (existing != _loaded_references.end()) { return existing->reg; }

  auto const binding_id   = get_or_bind_input_column(expr.column_index);
  auto const& col_binding = _input_columns.at(binding_id);
  auto const var_name     = next_var_name();
  auto const dst_type_str = cuda_type_name(restored_type);
  auto const param = [&](std::string_view field) { return input_param_name(binding_id, field); };

  std::string valid_expr = "true";
  if (col_binding.is_nullable) {
    emit_prelude_line(format("bool const {}_valid = is_bit_set({}, {} + row);",
                             var_name,
                             param("mask"),
                             param("bit_offset")));
    valid_expr = var_name + "_valid";
  }

  if (restored_type.id() == cudf::type_id::STRING) {
    auto const offsets = param("offsets");
    auto const chars   = param("chars");
    auto const view    = format(
      "str_view{{{0} + {1}[row], static_cast<int32_t>({1}[row + 1] - "
         "{1}[row]), {2}}}",
      chars,
      offsets,
      param("chars_end"));
    auto const guard = col_binding.is_nullable ? format("{} && {} != nullptr", valid_expr, offsets)
                                               : format("{} != nullptr", offsets);
    emit_prelude_line(
      format("reg_val<str_view> {}{{({}) ? {} : str_view{{nullptr, 0, nullptr}}, "
             "{}, false}};",
             var_name,
             guard,
             view,
             valid_expr));
  } else if (!col_binding.is_nullable) {
    emit_prelude_line(format("reg_val<{0}> {1}{{static_cast<{0}>({2}[row]), true, false}};",
                             dst_type_str,
                             var_name,
                             param("data")));
  } else {
    // The payload is loaded independently of the validity bit (a null row's slot still exists) so
    // both loads are in flight together.
    emit_prelude_line(format(
      "{0} const {1}_raw = static_cast<{0}>({2}[row]);", dst_type_str, var_name, param("data")));
    emit_prelude_line(format("reg_val<{0}> {1}{{{1}_valid ? {1}_raw : {0}{{}}, {1}_valid, false}};",
                             dst_type_str,
                             var_name));
  }

  register_value reg{var_name, restored_type, !col_binding.is_nullable, /*can_overflow=*/false};
  _loaded_references.push_back(loaded_reference_entry{expr.column_index, restored_type, reg});
  return reg;
}

//===----------CONSTANT----------===//
register_value translation_context::produce_constant(sirius::ast::constant const& expr)
{
  auto const device_type = sirius::require_cudf_type(expr.return_type());
  bool const is_valid    = !std::holds_alternative<sirius::null_value>(expr.payload);
  auto const binding_id  = _constants.size();
  auto const var_name    = next_var_name();
  auto const type_str    = cuda_type_name(device_type);

  if (!is_valid) {
    emit_prelude_line(
      format("reg_val<{0}> const {1}{{{0}{{}}, false, false}};", type_str, var_name));
  } else if (device_type.id() == cudf::type_id::STRING) {
    auto const ptr = constant_param_name(binding_id, "ptr");
    auto const len = constant_param_name(binding_id, "len");
    emit_prelude_line(
      format("reg_val<str_view> const {0}{{str_view{{{1}, {2}, {1} + {2}}}, true, false}};",
             var_name,
             ptr,
             len));
  } else {
    emit_prelude_line(format("reg_val<{}> const {}{{{}, true, false}};",
                             type_str,
                             var_name,
                             constant_param_name(binding_id, "val")));
  }

  register_value reg{var_name, device_type, /*is_statically_non_null=*/is_valid, false};
  _constants.push_back(
    scalar_constant_binding{binding_id, device_type, expr.payload, is_valid, reg});
  return reg;
}

std::optional<cudf::data_type> translation_context::narrow_domain_carrier(
  sirius::ast::node const& column_operand,
  std::initializer_list<sirius::ast::node const*> constant_operands) const
{
  if (!column_operand.holds<sirius::ast::reference>()) { return std::nullopt; }
  auto const& ref = column_operand.get<sirius::ast::reference>();
  if (ref.column_index >= static_cast<std::uint32_t>(_input.num_columns())) { return std::nullopt; }
  auto const carrier = _input.column(ref.column_index).type();
  if (!sirius::ast::narrow_domain_carrier_eligible(ref.return_type(), carrier, constant_operands)) {
    return std::nullopt;
  }
  return carrier;
}

std::optional<cudf::data_type> translation_context::narrow_domain_reference_pair_carrier(
  sirius::ast::node const& lhs, sirius::ast::node const& rhs) const
{
  if (!lhs.holds<sirius::ast::reference>() || !rhs.holds<sirius::ast::reference>()) {
    return std::nullopt;
  }
  auto const& lhs_ref    = lhs.get<sirius::ast::reference>();
  auto const& rhs_ref    = rhs.get<sirius::ast::reference>();
  auto const num_columns = static_cast<std::uint32_t>(_input.num_columns());
  if (lhs_ref.column_index >= num_columns || rhs_ref.column_index >= num_columns) {
    return std::nullopt;
  }
  auto const lhs_carrier = _input.column(lhs_ref.column_index).type();
  auto const rhs_carrier = _input.column(rhs_ref.column_index).type();
  if (!sirius::ast::narrow_domain_reference_pair_eligible(
        lhs_ref.return_type(), lhs_carrier, rhs_ref.return_type(), rhs_carrier)) {
    return std::nullopt;
  }
  return lhs_carrier;
}

register_value translation_context::produce_constant_in_carrier(sirius::ast::constant const& expr,
                                                                cudf::data_type carrier)
{
  bool const is_valid = !std::holds_alternative<sirius::null_value>(expr.payload);
  __int128_t raw_val  = 0;
  if (is_valid) {
    auto const range = sirius::ast::constant_numeric_range(expr);
    if (!range) {
      throw internal_exception("[nvrtc] narrow-domain constant lost its numeric range");
    }
    raw_val = range->minimum;
  }

  sirius::value narrowed_payload{};
  if (is_valid) {
    auto const scale = static_cast<std::uint8_t>(-carrier.scale());
    switch (carrier.id()) {
      case cudf::type_id::INT8: narrowed_payload = static_cast<std::int8_t>(raw_val); break;
      case cudf::type_id::INT16: narrowed_payload = static_cast<std::int16_t>(raw_val); break;
      case cudf::type_id::INT32: narrowed_payload = static_cast<std::int32_t>(raw_val); break;
      case cudf::type_id::UINT8: narrowed_payload = static_cast<std::uint8_t>(raw_val); break;
      case cudf::type_id::UINT16: narrowed_payload = static_cast<std::uint16_t>(raw_val); break;
      case cudf::type_id::UINT32: narrowed_payload = static_cast<std::uint32_t>(raw_val); break;
      case cudf::type_id::DECIMAL32:
        narrowed_payload = sirius::decimal32{static_cast<std::int32_t>(raw_val), scale};
        break;
      case cudf::type_id::DECIMAL64:
        narrowed_payload = sirius::decimal64{static_cast<std::int64_t>(raw_val), scale};
        break;
      default:
        throw internal_exception("[nvrtc] unsupported narrow constant carrier: {}",
                                 cudf::type_to_name(carrier));
    }
  }

  auto const binding_id = _constants.size();
  auto const var_name   = next_var_name();
  auto const type_str   = cuda_type_name(carrier);
  if (is_valid) {
    emit_prelude_line(format("reg_val<{}> const {}{{{}, true, false}};",
                             type_str,
                             var_name,
                             constant_param_name(binding_id, "val")));
  } else {
    emit_prelude_line(
      format("reg_val<{0}> const {1}{{{0}{{}}, false, false}};", type_str, var_name));
  }

  register_value reg{var_name, carrier, /*is_statically_non_null=*/is_valid, false};
  _constants.push_back(
    scalar_constant_binding{binding_id, carrier, std::move(narrowed_payload), is_valid, reg});
  return reg;
}

register_value translation_context::produce_narrow_domain_operand(
  sirius::ast::node const& operand, std::optional<cudf::data_type> narrow_carrier)
{
  if (!narrow_carrier) { return produce(operand); }
  if (operand.holds<sirius::ast::reference>()) {
    // Load the reference at its raw carrier width without restoring to the wider type.
    auto const& ref = operand.get<sirius::ast::reference>();
    sirius::ast::reference raw_ref{ref.column_index, sirius::logical_type{}};
    return produce_reference(raw_ref);
  }
  return produce_constant_in_carrier(operand.get<sirius::ast::constant>(), *narrow_carrier);
}

//===-------------------------------------------------------------------------===//
// NON-LEAF NODES:
//   - COMPARISON
//   - BETWEEN
//   - CONJUNCTION
//   - CASE
//   - COALESCE
//   - IN LIST
//   - CAST
//   - UNARY OPERATOR
//===-------------------------------------------------------------------------===//

//===----------COMPARISON----------===//
register_value translation_context::produce_comparison(sirius::ast::comparison const& expr)
{
  auto narrow_carrier = narrow_domain_carrier(*expr.left, {expr.right.get()});
  if (!narrow_carrier) { narrow_carrier = narrow_domain_carrier(*expr.right, {expr.left.get()}); }
  if (!narrow_carrier) {
    narrow_carrier = narrow_domain_reference_pair_carrier(*expr.left, *expr.right);
  }
  if (narrow_carrier) { ++_narrow_domain_comparison_count; }

  auto const lhs      = produce_narrow_domain_operand(*expr.left, narrow_carrier);
  auto const rhs      = produce_narrow_domain_operand(*expr.right, narrow_carrier);
  auto const var_name = next_var_name();

  bool const is_dec =
    cudf::is_fixed_point(lhs.device_type) || cudf::is_fixed_point(rhs.device_type);
  std::int32_t const s_l   = is_dec ? sql_decimal_scale(lhs.device_type) : 0;
  std::int32_t const s_r   = is_dec ? sql_decimal_scale(rhs.device_type) : 0;
  std::int32_t const s_max = std::max(s_l, s_r);
  auto const cmp_type      = comparison_type_name(lhs.device_type, rhs.device_type);
  auto const lhs_val       = comparison_operand(lhs, cmp_type, s_l, s_max);
  auto const rhs_val       = comparison_operand(rhs, cmp_type, s_r, s_max);
  bool const can_ovf       = any_can_overflow({&lhs, &rhs});
  auto const ovf_e         = overflow_expression({&lhs, &rhs});

  if (expr.op == sirius::comparison_type::not_distinct_from) {
    emit_reg_val("bool",
                 var_name,
                 format("(!{0}.valid && !{1}.valid) || ({0}.valid && {1}.valid && {2})",
                        lhs.var_name,
                        rhs.var_name,
                        comparison_text(expr.op, lhs_val, rhs_val, cmp_type)),
                 "true",
                 ovf_e);
    return make_bool_register(var_name, /*is_statically_non_null=*/true, can_ovf);
  }
  if (expr.op == sirius::comparison_type::distinct_from) {
    emit_reg_val("bool",
                 var_name,
                 format("({0}.valid != {1}.valid) || ({0}.valid && {1}.valid && {2})",
                        lhs.var_name,
                        rhs.var_name,
                        comparison_text(expr.op, lhs_val, rhs_val, cmp_type)),
                 "true",
                 ovf_e);
    return make_bool_register(var_name, /*is_statically_non_null=*/true, can_ovf);
  }

  auto const compare  = comparison_text(expr.op, lhs_val, rhs_val, cmp_type);
  bool const non_null = all_statically_non_null({&lhs, &rhs});
  if (non_null) {
    emit_reg_val("bool", var_name, compare, "true", ovf_e);
  } else {
    emit_line(
      format("bool const {}_valid = {}.valid && {}.valid;", var_name, lhs.var_name, rhs.var_name));
    emit_reg_val("bool",
                 var_name,
                 format("{}_valid ? {} : false", var_name, compare),
                 var_name + "_valid",
                 ovf_e);
  }
  return make_bool_register(var_name, non_null, can_ovf);
}

//===----------BETWEEN----------===//
register_value translation_context::produce_between(sirius::ast::between const& expr)
{
  auto const narrow_carrier =
    narrow_domain_carrier(*expr.input, {expr.lower.get(), expr.upper.get()});
  if (narrow_carrier) { ++_narrow_domain_comparison_count; }

  auto const in_reg   = produce_narrow_domain_operand(*expr.input, narrow_carrier);
  auto const lo_reg   = produce_narrow_domain_operand(*expr.lower, narrow_carrier);
  auto const hi_reg   = produce_narrow_domain_operand(*expr.upper, narrow_carrier);
  auto const var_name = next_var_name();

  bool const is_dec        = cudf::is_fixed_point(in_reg.device_type);
  std::int32_t const s_i   = is_dec ? sql_decimal_scale(in_reg.device_type) : 0;
  std::int32_t const s_l   = is_dec ? sql_decimal_scale(lo_reg.device_type) : 0;
  std::int32_t const s_h   = is_dec ? sql_decimal_scale(hi_reg.device_type) : 0;
  std::int32_t const s_max = std::max({s_i, s_l, s_h});
  bool const mixed =
    is_dec && (s_i != s_l || s_i != s_h || in_reg.device_type != lo_reg.device_type ||
               in_reg.device_type != hi_reg.device_type);
  auto const cmp_type  = mixed ? std::string("__int128_t") : cuda_type_name(in_reg.device_type);
  auto const in_val    = comparison_operand(in_reg, cmp_type, s_i, s_max);
  auto const lo_val    = comparison_operand(lo_reg, cmp_type, s_l, s_max);
  auto const hi_val    = comparison_operand(hi_reg, cmp_type, s_h, s_max);
  auto const lower_cmp = comparison_text(
    expr.lower_inclusive ? sirius::comparison_type::ge : sirius::comparison_type::gt,
    in_val,
    lo_val,
    cmp_type);
  auto const upper_cmp = comparison_text(
    expr.upper_inclusive ? sirius::comparison_type::le : sirius::comparison_type::lt,
    in_val,
    hi_val,
    cmp_type);

  bool const non_null = all_statically_non_null({&in_reg, &lo_reg, &hi_reg});
  bool const can_ovf  = any_can_overflow({&in_reg, &lo_reg, &hi_reg});
  if (non_null) {
    emit_reg_val("bool",
                 var_name,
                 format("({} && {})", lower_cmp, upper_cmp),
                 "true",
                 overflow_expression({&in_reg, &lo_reg, &hi_reg}));
  } else {
    // SQL BETWEEN is a three-valued conjunction: FALSE AND NULL is FALSE.
    auto const emit_bound = [&](std::string const& cmp, register_value const& bound) {
      auto const bound_var = next_var_name();
      emit_line(format(
        "bool const {}_valid = {}.valid && {}.valid;", bound_var, in_reg.var_name, bound.var_name));
      emit_line(format("reg_val<bool> const {0}{{{0}_valid ? {1} : false, {0}_valid, {2}}};",
                       bound_var,
                       cmp,
                       overflow_expression({&in_reg, &bound})));
      return bound_var;
    };
    auto const lower_var = emit_bound(lower_cmp, lo_reg);
    auto const upper_var = emit_bound(upper_cmp, hi_reg);
    emit_line(
      format("reg_val<bool> const {} = eval_kleene_and({}, {});", var_name, lower_var, upper_var));
  }
  return make_bool_register(var_name, non_null, can_ovf);
}

//===----------CONJUNCTION----------===//
register_value translation_context::produce_conjunction(sirius::ast::conjunction const& expr)
{
  auto const var_name = next_var_name();
  auto const first    = produce(*expr.children.front());
  emit_reg_val_assign("bool", var_name, first.var_name);

  bool non_null       = first.is_statically_non_null;
  bool can_ovf        = first.can_overflow;
  bool const is_and   = (expr.op == sirius::ast::conjunction::kind::op_and);
  char const* fn_name = is_and ? "eval_kleene_and" : "eval_kleene_or";
  // The next operand only runs while the accumulator can still change: a definite FALSE (AND) or
  // TRUE (OR) dominates every later operand, including one that would overflow.
  auto const dom_cond =
    format("!({0}.valid && !{0}.overflow && {1}{0}.val)", var_name, is_and ? "!" : "");

  for (auto it = std::next(expr.children.begin()); it != expr.children.end(); ++it) {
    emit_line(format("if ({}) {{", dom_cond));
    auto const rhs = produce(**it);
    emit_line(format("{0} = {1}({0}, {2});", var_name, fn_name, rhs.var_name));
    emit_line("}");
    non_null = non_null && rhs.is_statically_non_null;
    can_ovf  = can_ovf || rhs.can_overflow;
  }
  return make_bool_register(var_name, non_null, can_ovf);
}

//===----------CASE----------===//
register_value translation_context::produce_case_expr(sirius::ast::case_expr const& expr)
{
  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();
  emit_reg_val(type_str, var_name, type_str + "{}", "false", "false");

  auto const assign_arm = [&](register_value const& arm) {
    emit_line(format("{0} = reg_val<{1}>{{static_cast<{1}>({2}.val), {2}.valid, {2}.overflow}};",
                     var_name,
                     type_str,
                     arm.var_name));
  };

  bool can_ovf      = false;
  bool all_non_null = true;
  for (auto const& c : expr.cases) {
    auto const when_rv = produce(*c.when_);
    can_ovf            = can_ovf || when_rv.can_overflow;
    // A poisoned WHEN poisons the result; a true WHEN selects its arm; otherwise fall through.
    emit_line(format("if ({}.overflow) {{", when_rv.var_name));
    emit_line(format("{0} = reg_val<{1}>{{{1}{{}}, true, true}};", var_name, type_str));
    emit_line(format("}} else if ({0}.valid && {0}.val) {{", when_rv.var_name));
    if (c.then_->holds<sirius::ast::function_call>() &&
        c.then_->get<sirius::ast::function_call>().function() == sirius::function_id::error) {
      _has_subquery_error = true;
      all_non_null        = false;
      emit_line("row_subquery_error = true;");
    } else {
      auto const then_rv = produce(*c.then_);
      can_ovf            = can_ovf || then_rv.can_overflow;
      all_non_null       = all_non_null && then_rv.is_statically_non_null;
      assign_arm(then_rv);
    }
    emit_line("} else {");
  }

  auto const else_rv = produce(*expr.else_);
  can_ovf            = can_ovf || else_rv.can_overflow;
  all_non_null       = all_non_null && else_rv.is_statically_non_null;
  assign_arm(else_rv);
  for (std::size_t i = 0; i < expr.cases.size(); ++i) {
    emit_line("}");
  }
  return register_value{var_name, ret_type, all_non_null, can_ovf};
}

//===----------COALESCE----------===//
register_value translation_context::produce_coalesce(sirius::ast::coalesce const& expr)
{
  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();

  auto const assign = [&](register_value const& child) {
    return format("reg_val<{0}>{{static_cast<{0}>({1}.val), {1}.valid, {1}.overflow}}",
                  type_str,
                  child.var_name);
  };

  auto const first = produce(*expr.children.front());
  emit_reg_val_assign(type_str, var_name, assign(first));
  bool any_non_null = first.is_statically_non_null;
  bool can_ovf      = first.can_overflow;

  for (auto it = std::next(expr.children.begin()); it != expr.children.end(); ++it) {
    emit_line(format("if (!{0}.valid && !{0}.overflow) {{", var_name));
    auto const child = produce(**it);
    emit_line(format("{} = {};", var_name, assign(child)));
    emit_line("}");
    any_non_null = any_non_null || child.is_statically_non_null;
    can_ovf      = can_ovf || child.can_overflow;
  }
  return register_value{var_name, ret_type, any_non_null, can_ovf};
}

//===----------IN LIST----------===//
register_value translation_context::produce_in_list(sirius::ast::in_list const& expr)
{
  auto const probe_reg = produce(*expr.probe);
  auto const var_name  = next_var_name();
  emit_reg_val("bool", var_name, "false", "false", "false");

  bool can_ovf      = probe_reg.can_overflow;
  bool all_non_null = probe_reg.is_statically_non_null;
  for (std::size_t i = 0; i < expr.values.size(); ++i) {
    // A definite TRUE dominates the remaining list values.
    if (i > 0) { emit_line(format("if (!({0}.valid && !{0}.overflow && {0}.val)) {{", var_name)); }
    auto const val_reg = produce(*expr.values[i]);
    can_ovf            = can_ovf || val_reg.can_overflow;
    all_non_null       = all_non_null && val_reg.is_statically_non_null;

    bool const is_dec =
      cudf::is_fixed_point(probe_reg.device_type) || cudf::is_fixed_point(val_reg.device_type);
    std::int32_t const s_l   = is_dec ? sql_decimal_scale(probe_reg.device_type) : 0;
    std::int32_t const s_r   = is_dec ? sql_decimal_scale(val_reg.device_type) : 0;
    std::int32_t const s_max = std::max(s_l, s_r);
    auto const cmp_type      = comparison_type_name(probe_reg.device_type, val_reg.device_type);
    auto const probe_val     = comparison_operand(probe_reg, cmp_type, s_l, s_max);
    auto const rhs_val       = comparison_operand(val_reg, cmp_type, s_r, s_max);

    auto const eq_var = next_var_name();
    emit_line(format(
      "bool const {}_valid = {}.valid && {}.valid;", eq_var, probe_reg.var_name, val_reg.var_name));
    emit_line(format("reg_val<bool> const {0}{{{0}_valid ? {1} : false, {0}_valid, {2}}};",
                     eq_var,
                     comparison_text(sirius::comparison_type::equal, probe_val, rhs_val, cmp_type),
                     overflow_expression({&probe_reg, &val_reg})));
    if (i == 0) {
      emit_line(format("{} = {};", var_name, eq_var));
    } else {
      emit_line(format("{0} = eval_kleene_or({0}, {1});", var_name, eq_var));
      emit_line("}");
    }
  }
  if (expr.negated) { emit_line(format("{0}.val = !{0}.val;", var_name)); }
  return make_bool_register(var_name, all_non_null, can_ovf);
}

//===----------CAST----------===//
register_value translation_context::produce_cast(sirius::ast::cast const& expr)
{
  auto const dst_type = sirius::require_cudf_type(expr.target_type);
  auto const dst_str  = cuda_type_name(dst_type);
  auto const var_name = next_var_name();

  // Carrier restoration: a referenced column is loaded at its raw carrier width and widened inline.
  register_value const child = [&]() {
    if (expr.kind == sirius::ast::cast_kind::carrier_restore &&
        expr.child->holds<sirius::ast::reference>()) {
      auto const& ref = expr.child->get<sirius::ast::reference>();
      sirius::ast::reference raw_ref{ref.column_index, sirius::logical_type{}};
      return produce_reference(raw_ref);
    }
    return produce(*expr.child);
  }();
  auto const& c = child.var_name;

  if (expr.kind == sirius::ast::cast_kind::carrier_restore) {
    emit_reg_val(dst_str,
                 var_name,
                 format("static_cast<{}>({}.val)", dst_str, c),
                 c + ".valid",
                 c + ".overflow");
    return register_value{var_name, dst_type, child.is_statically_non_null, child.can_overflow};
  }

  auto const src_type = child.device_type;
  bool op_can_ovf     = false;
  auto const call     = [&](std::string const& fn_and_args) {
    emit_reg_val_assign(dst_str, var_name, fn_and_args);
  };

  if (cudf::is_timestamp(src_type) && cudf::is_timestamp(dst_type)) {
    if (src_type.id() == dst_type.id()) {
      emit_reg_val_assign(dst_str, var_name, c);
    } else if (src_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
      call(format("eval_cast_date_to_timestamp<{}LL>({})",
                  timestamp_ticks_per_second(dst_type.id()) * 86'400LL,
                  c));
      op_can_ovf = true;
    } else if (dst_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
      call(format("eval_cast_timestamp_to_date<{}LL>({})",
                  timestamp_ticks_per_second(src_type.id()) * 86'400LL,
                  c));
      op_can_ovf = true;
    } else {
      std::int64_t const src_tps = timestamp_ticks_per_second(src_type.id());
      std::int64_t const dst_tps = timestamp_ticks_per_second(dst_type.id());
      if (dst_tps > src_tps) {
        call(format("eval_cast_ts_scale_up<{}LL>({})", dst_tps / src_tps, c));
        op_can_ovf = true;
      } else {
        call(format("eval_cast_ts_round_down<{}LL>({})", src_tps / dst_tps, c));
      }
    }
  } else if (dst_type.id() == cudf::type_id::BOOL8) {
    call(format("eval_cast_to_bool({})", c));
  } else if (src_type.id() == cudf::type_id::BOOL8) {
    if (cudf::is_fixed_point(dst_type)) {
      int const prec       = target_decimal_precision(expr.target_type);
      int const scale      = sql_decimal_scale(dst_type);
      bool const can_fit_1 = (prec > scale);
      call(format("eval_cast_bool_to_dec<{0}, {1}>({2}, static_cast<{0}>({3}))",
                  dst_str,
                  can_fit_1,
                  c,
                  format_int128_pow10(scale)));
      op_can_ovf = !can_fit_1;
    } else {
      emit_reg_val(dst_str,
                   var_name,
                   format("{0}.val ? static_cast<{1}>(1) : static_cast<{1}>(0)", c, dst_str),
                   c + ".valid",
                   c + ".overflow");
    }
  } else if (cudf::is_fixed_point(src_type) && cudf::is_fixed_point(dst_type)) {
    int const s_in       = sql_decimal_scale(src_type);
    int const s_out      = sql_decimal_scale(dst_type);
    int const p_out      = target_decimal_precision(expr.target_type);
    auto const limit_str = format_int128_pow10(p_out);
    if (s_out >= s_in) {
      call(format("eval_cast_dec_scale_up<{}>({}, {}, {})",
                  dst_str,
                  c,
                  format_int128_pow10(s_out - s_in),
                  limit_str));
    } else {
      call(format("eval_cast_dec_scale_down<{}>({}, {}, {})",
                  dst_str,
                  c,
                  format_int128_pow10(s_in - s_out),
                  limit_str));
    }
    op_can_ovf = true;
  } else if (cudf::is_fixed_point(src_type) && cudf::is_integral(dst_type)) {
    call(format("eval_cast_dec_to_int<{}>({}, {})",
                dst_str,
                c,
                format_int128_pow10(sql_decimal_scale(src_type))));
    op_can_ovf = true;
  } else if (cudf::is_fixed_point(src_type) && cudf::is_floating_point(dst_type)) {
    call(format("eval_cast_dec_to_float<{}>({}, {})",
                dst_str,
                c,
                format_double_pow10(sql_decimal_scale(src_type))));
  } else if (cudf::is_integral(src_type) && cudf::is_fixed_point(dst_type)) {
    int const p_out = target_decimal_precision(expr.target_type);
    int const s_out = sql_decimal_scale(dst_type);
    call(format("eval_cast_int_to_dec<{}>({}, {}, {})",
                dst_str,
                c,
                format_int128_pow10(p_out - s_out),
                format_int128_pow10(s_out)));
    op_can_ovf = true;
  } else if (cudf::is_floating_point(src_type) && cudf::is_fixed_point(dst_type)) {
    int const p_out = target_decimal_precision(expr.target_type);
    int const s_out = sql_decimal_scale(dst_type);
    call(format("eval_cast_float_to_dec<{}>({}, {}, {}, {})",
                dst_str,
                c,
                format_double_pow10(s_out),
                format_double_pow10(p_out),
                format_int128_pow10(p_out)));
    op_can_ovf = true;
  } else if (cudf::is_floating_point(src_type) && cudf::is_integral(dst_type)) {
    call(format("eval_cast_float_to_int<{}>({})", dst_str, c));
    op_can_ovf = true;
  } else if (cudf::is_integral(src_type) && cudf::is_floating_point(dst_type)) {
    emit_reg_val(dst_str,
                 var_name,
                 format("static_cast<{}>({}.val)", dst_str, c),
                 c + ".valid",
                 c + ".overflow");
  } else if (cudf::is_floating_point(src_type) && cudf::is_floating_point(dst_type)) {
    call(format("eval_cast_float_to_float<{}>({})", dst_str, c));
    if (src_type.id() == cudf::type_id::FLOAT64 && dst_type.id() == cudf::type_id::FLOAT32) {
      op_can_ovf = true;
    }
  } else if (cudf::is_integral(src_type) && cudf::is_integral(dst_type)) {
    call(format("eval_cast_int_to_int<{}>({})", dst_str, c));
    op_can_ovf = true;
  } else {
    throw not_implemented_exception("[nvrtc] Unsupported cast from {} to {}",
                                    cudf::type_to_name(src_type),
                                    cudf::type_to_name(dst_type));
  }

  if (expr.try_cast && op_can_ovf) {
    // TRY_CAST turns only its own conversion failure into NULL. DuckDB still raises an overflow
    // inside its input, so the child's Overflow Poison and overflow sites stay.
    emit_line(format(
      "if ({0}.overflow && !{1}.overflow) {{ {0} = reg_val<{2}>{{{2}{{}}, false, false}}; }}",
      var_name,
      c,
      dst_str));
    return register_value{var_name, dst_type, /*is_statically_non_null=*/false, child.can_overflow};
  }
  return register_value{var_name,
                        dst_type,
                        child.is_statically_non_null,
                        record_overflow_site(&expr, op_can_ovf, child.can_overflow)};
}

//===----------UNARY OPERATOR----------===//
register_value translation_context::produce_unary_op(sirius::ast::unary_op const& expr)
{
  auto const try_mark = overflow_site_mark();
  auto const child    = produce(*expr.child);
  auto const var_name = next_var_name();
  auto const& c       = child.var_name;

  switch (expr.op) {
    case sirius::ast::unary_op::kind::op_not:
      emit_reg_val("bool", var_name, "!" + c + ".val", c + ".valid", c + ".overflow");
      return make_bool_register(var_name, child.is_statically_non_null, child.can_overflow);
    case sirius::ast::unary_op::kind::op_is_null:
      emit_reg_val("bool", var_name, "!" + c + ".valid", "true", c + ".overflow");
      return make_bool_register(var_name, /*is_statically_non_null=*/true, child.can_overflow);
    case sirius::ast::unary_op::kind::op_is_not_null:
      emit_reg_val("bool", var_name, c + ".valid", "true", c + ".overflow");
      return make_bool_register(var_name, /*is_statically_non_null=*/true, child.can_overflow);
    case sirius::ast::unary_op::kind::op_try: {
      auto const type_str = cuda_type_name(child.device_type);
      emit_reg_val(type_str,
                   var_name,
                   format("{0}.overflow ? {1}{{}} : {0}.val", c, type_str),
                   format("{0}.valid && !{0}.overflow", c),
                   "false");
      drop_overflow_sites_since(try_mark);
      return register_value{var_name, child.device_type, false, false};
    }
    default:
      throw not_implemented_exception("[nvrtc] Unsupported unary_op kind: {}",
                                      static_cast<int>(expr.op));
  }
}

//===----------BINARY ARITHMETIC FUNCTIONS----------===//
register_value translation_context::produce_arithmetic(sirius::ast::function_call const& expr)
{
  auto const fid           = expr.function();
  auto const& args         = expr.arguments();
  auto const ret_type      = sirius::require_cudf_type(expr.return_type());
  auto const type_str      = cuda_type_name(ret_type);
  auto const var_name      = next_var_name();
  auto const lhs           = produce(*args[0]);
  auto const rhs           = produce(*args[1]);
  bool const is_div_or_mod = fid == sirius::function_id::div ||
                             fid == sirius::function_id::int_div || fid == sirius::function_id::mod;

  if (cudf::is_fixed_point(ret_type)) {
    int const s_l   = sql_decimal_scale(lhs.device_type);
    int const s_r   = sql_decimal_scale(rhs.device_type);
    int const s_max = std::max(s_l, s_r);
    if (fid == sirius::function_id::add || fid == sirius::function_id::sub) {
      emit_reg_val_assign(type_str,
                          var_name,
                          format("{}<{}>({}, {}, {}, {})",
                                 fid == sirius::function_id::add ? "eval_dec_add" : "eval_dec_sub",
                                 type_str,
                                 lhs.var_name,
                                 rhs.var_name,
                                 format_int128_pow10(s_max - s_l),
                                 format_int128_pow10(s_max - s_r)));
    } else if (fid == sirius::function_id::mul) {
      emit_reg_val_assign(type_str,
                          var_name,
                          format("eval_dec_mul<{}>({}, {})", type_str, lhs.var_name, rhs.var_name));
    } else if (fid == sirius::function_id::div || fid == sirius::function_id::int_div) {
      int const s_diff = sql_decimal_scale(ret_type) - (s_l - s_r);
      emit_reg_val_assign(type_str,
                          var_name,
                          format("eval_dec_div<{}, {}>({}, {}, {})",
                                 type_str,
                                 s_diff >= 0,
                                 lhs.var_name,
                                 rhs.var_name,
                                 format_int128_pow10(std::abs(s_diff))));
    } else {
      emit_reg_val_assign(type_str,
                          var_name,
                          format("eval_dec_mod<{}>({}, {}, {}, {})",
                                 type_str,
                                 lhs.var_name,
                                 rhs.var_name,
                                 format_int128_pow10(s_max - s_l),
                                 format_int128_pow10(s_max - s_r)));
    }
    // Every decimal operation narrows a 128-bit intermediate to the output representation.
    bool const can_ovf  = record_overflow_site(&expr, true, any_can_overflow({&lhs, &rhs}));
    bool const non_null = all_statically_non_null({&lhs, &rhs}) && !is_div_or_mod;
    return register_value{var_name, ret_type, non_null, can_ovf};
  }

  char const* fn_name = nullptr;
  switch (fid) {
    case sirius::function_id::add: fn_name = "eval_add"; break;
    case sirius::function_id::sub: fn_name = "eval_sub"; break;
    case sirius::function_id::mul: fn_name = "eval_mul"; break;
    case sirius::function_id::div:
    case sirius::function_id::int_div: fn_name = "eval_div"; break;
    case sirius::function_id::mod: fn_name = "eval_mod"; break;
    default:
      throw internal_exception("[nvrtc] function {} is not arithmetic", static_cast<int>(fid));
  }
  emit_reg_val_assign(
    type_str, var_name, format("{}<{}>({}, {})", fn_name, type_str, lhs.var_name, rhs.var_name));

  bool const is_int_type = cudf::is_integral(ret_type);
  bool const op_can_ovf =
    is_int_type && (fid == sirius::function_id::add || fid == sirius::function_id::sub ||
                    fid == sirius::function_id::mul ||
                    ((fid == sirius::function_id::div || fid == sirius::function_id::int_div) &&
                     !cudf::is_unsigned(ret_type)));
  bool const can_ovf  = record_overflow_site(&expr, op_can_ovf, any_can_overflow({&lhs, &rhs}));
  bool const non_null = all_statically_non_null({&lhs, &rhs}) && !(is_div_or_mod && is_int_type);
  return register_value{var_name, ret_type, non_null, can_ovf};
}

//===----------DATE/TIME EXTRACT----------===//
register_value translation_context::produce_extract(sirius::ast::function_call const& expr)
{
  auto const& args    = expr.arguments();
  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();
  auto const child    = produce(*args[0]);
  auto const comp     = extract_component_of(expr.function());
  auto const comp_str = format("extract_component::{}", extract_component_name(comp));
  auto const ct_id    = child.device_type.id();
  if (ct_id == cudf::type_id::TIMESTAMP_DAYS) {
    emit_reg_val_assign(
      type_str,
      var_name,
      format("eval_extract_date<{}, {}>({})", type_str, comp_str, child.var_name));
  } else {
    emit_reg_val_assign(type_str,
                        var_name,
                        format("eval_extract_timestamp<{}, {}LL, {}>({})",
                               type_str,
                               timestamp_ticks_per_second(ct_id),
                               comp_str,
                               child.var_name));
  }
  // Sub-second components of an infinite timestamp are NULL.
  bool const is_subsec =
    comp == abi::extract_component::MILLISECOND || comp == abi::extract_component::MICROSECOND;
  return register_value{
    var_name, ret_type, child.is_statically_non_null && !is_subsec, child.can_overflow};
}

//===----------DATE/TIME TRUNC----------===//
register_value translation_context::produce_date_trunc(sirius::ast::function_call const& expr)
{
  auto const& args    = expr.arguments();
  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();
  auto const& freq    = std::get<std::string>(args[0]->get<sirius::ast::constant>().payload);
  auto const unit     = sirius::parse_gpu_date_trunc_unit(freq);
  if (!unit) {
    throw internal_exception("[nvrtc] date_trunc unit '{}' passed the supportability check", freq);
  }
  auto const child      = produce(*args[1]);
  auto const src_type   = child.device_type;
  std::string trunc_var = child.var_name;
  if (src_type.id() != cudf::type_id::TIMESTAMP_DAYS) {
    std::int64_t const div = date_trunc_divisor(src_type.id(), *unit);
    if (div > 1LL) {
      trunc_var = next_var_name();
      emit_reg_val_assign(
        "int64_t", trunc_var, format("eval_date_trunc<{}LL>({})", div, child.var_name));
    }
  }

  // Reconcile the truncated value with the declared return type; the value is a whole multiple
  // of the coarser unit, so scaling down is exact.
  bool op_can_ovf = false;
  if (ret_type == src_type) {
    emit_reg_val_assign(type_str, var_name, trunc_var);
  } else if (src_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
    emit_reg_val_assign(
      type_str,
      var_name,
      format("eval_cast_ts_scale_up<{0}LL>(reg_val<int64_t>{{static_cast<int64_t>({1}.val), "
             "{1}.valid, {1}.overflow}})",
             timestamp_ticks_per_second(ret_type.id()) * 86'400LL,
             trunc_var));
    op_can_ovf = true;
  } else if (ret_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
    emit_reg_val_assign(type_str,
                        var_name,
                        format("eval_cast_timestamp_to_date<{}LL>({})",
                               timestamp_ticks_per_second(src_type.id()) * 86'400LL,
                               trunc_var));
    op_can_ovf = true;
  } else {
    std::int64_t const src_tps = timestamp_ticks_per_second(src_type.id());
    std::int64_t const dst_tps = timestamp_ticks_per_second(ret_type.id());
    if (dst_tps > src_tps) {
      emit_reg_val_assign(type_str,
                          var_name,
                          format("eval_cast_ts_scale_up<{}LL>({})", dst_tps / src_tps, trunc_var));
      op_can_ovf = true;
    } else {
      emit_reg_val_assign(
        type_str,
        var_name,
        format("eval_cast_ts_scale_down<{}LL>({})", src_tps / dst_tps, trunc_var));
    }
  }
  return register_value{var_name,
                        ret_type,
                        child.is_statically_non_null,
                        record_overflow_site(&expr, op_can_ovf, child.can_overflow)};
}

//===----------STRING MATCHING FUNCTIONS----------===//
register_value translation_context::produce_string_match(sirius::ast::function_call const& expr)
{
  auto const fid      = expr.function();
  auto const& args    = expr.arguments();
  auto const var_name = next_var_name();
  auto const lhs      = produce(*args[0]);
  if (fid != sirius::function_id::prefix && fid != sirius::function_id::suffix) {
    if (auto const pattern_id = try_bind_like_pattern(*args[1], fid)) {
      emit_reg_val_assign("bool",
                          var_name,
                          format("eval_like_multiliteral<{}>({}, {})",
                                 fid == sirius::function_id::not_like,
                                 lhs.var_name,
                                 like_pattern_param_name(*pattern_id)));
      return make_bool_register(var_name, lhs.is_statically_non_null, lhs.can_overflow);
    }
  }
  auto const rhs = produce(*args[1]);
  char const* fn = nullptr;
  switch (fid) {
    case sirius::function_id::prefix: fn = "eval_prefix"; break;
    case sirius::function_id::suffix: fn = "eval_suffix"; break;
    case sirius::function_id::contains: fn = "eval_contains"; break;
    case sirius::function_id::like: fn = "eval_like<false>"; break;
    case sirius::function_id::not_like: fn = "eval_like<true>"; break;
    default:
      throw internal_exception("[nvrtc] function {} is not a string match", static_cast<int>(fid));
  }
  emit_reg_val_assign("bool", var_name, format("{}({}, {})", fn, lhs.var_name, rhs.var_name));
  return make_bool_register(
    var_name, all_statically_non_null({&lhs, &rhs}), any_can_overflow({&lhs, &rhs}));
}

//===----------STRING SUBSTRING----------===//
register_value translation_context::produce_substring(sirius::ast::function_call const& expr)
{
  // substring() yields a view into the input string, so it never allocates a new string column.
  auto const& args          = expr.arguments();
  auto const ret_type       = sirius::require_cudf_type(expr.return_type());
  auto const var_name       = next_var_name();
  std::int64_t const offset = extract_constant_int64(*args[1]).value_or(0);
  std::int64_t const length =
    (args.size() == 3) ? extract_constant_int64(*args[2]).value_or(sirius::substring_default_length)
                       : sirius::substring_default_length;
  auto const slice = sirius::gpu_substring_slice(offset, length);
  if (!slice) {
    throw internal_exception(
      "[nvrtc] substring bounds (offset {}, length {}) passed the supportability check",
      offset,
      length);
  }
  auto const child          = produce(*args[0]);
  auto const int32_sql_type = sirius::logical_type::make(sirius::type_id::INTEGER);
  auto const start_reg      = produce_constant(
    sirius::ast::constant{sirius::value{slice->start.value_or(0)}, int32_sql_type});
  auto const stop_reg = produce_constant(sirius::ast::constant{
    sirius::value{slice->stop.value_or(std::numeric_limits<std::int32_t>::max())}, int32_sql_type});
  emit_reg_val_assign(
    "str_view",
    var_name,
    format(
      "eval_substring({}, {}.val, {}.val)", child.var_name, start_reg.var_name, stop_reg.var_name));
  return register_value{var_name, ret_type, child.is_statically_non_null, child.can_overflow};
}

//===----------FUNCTION----------===//
register_value translation_context::produce_function_call(sirius::ast::function_call const& expr)
{
  auto const fid   = expr.function();
  auto const& args = expr.arguments();
  switch (fid) {
    case sirius::function_id::add:
    case sirius::function_id::sub:
    case sirius::function_id::mul:
    case sirius::function_id::div:
    case sirius::function_id::int_div:
    case sirius::function_id::mod: return produce_arithmetic(expr);
    case sirius::function_id::year:
    case sirius::function_id::month:
    case sirius::function_id::day:
    case sirius::function_id::hour:
    case sirius::function_id::minute:
    case sirius::function_id::second:
    case sirius::function_id::millisecond:
    case sirius::function_id::microsecond: return produce_extract(expr);
    case sirius::function_id::date_trunc: return produce_date_trunc(expr);
    case sirius::function_id::prefix:
    case sirius::function_id::suffix:
    case sirius::function_id::contains:
    case sirius::function_id::like:
    case sirius::function_id::not_like: return produce_string_match(expr);
    case sirius::function_id::substring: return produce_substring(expr);
    default: break;
  }

  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();

  //===----------UNARY ARITHMETIC FUNCTIONS----------===//
  if (fid == sirius::function_id::negate) {
    auto const child = produce(*args[0]);
    emit_reg_val_assign(
      type_str, var_name, format("eval_negate<{}>({})", type_str, child.var_name));
    bool const op_can_ovf = !cudf::is_floating_point(ret_type);
    return register_value{var_name,
                          ret_type,
                          child.is_statically_non_null,
                          record_overflow_site(&expr, op_can_ovf, child.can_overflow)};
  }

  if (fid == sirius::function_id::abs) {
    auto const child = produce(*args[0]);
    emit_reg_val_assign(type_str, var_name, format("eval_abs<{}>({})", type_str, child.var_name));
    bool const op_can_ovf = cudf::is_fixed_point(ret_type) ||
                            (cudf::is_integral(ret_type) && !cudf::is_unsigned(ret_type));
    return register_value{var_name,
                          ret_type,
                          child.is_statically_non_null,
                          record_overflow_site(&expr, op_can_ovf, child.can_overflow)};
  }

  if (fid == sirius::function_id::round) {
    auto const child       = produce(*args[0]);
    std::int32_t precision = 0;
    if (args.size() == 2) {
      precision = std::get<std::int32_t>(args[1]->get<sirius::ast::constant>().payload);
    }
    // DuckDB computes the modifier in double for FLOAT inputs as well.
    bool const neg_prec   = precision < 0;
    double const modifier = std::pow(
      10.0, static_cast<double>(neg_prec ? -static_cast<std::int64_t>(precision) : precision));
    auto const mod_reg = produce_constant(sirius::ast::constant{
      sirius::value{modifier}, sirius::logical_type::make(sirius::type_id::DOUBLE)});
    emit_reg_val_assign(
      type_str,
      var_name,
      format(
        "eval_round<{}, {}>({}, {}.val)", type_str, neg_prec, child.var_name, mod_reg.var_name));
    return register_value{var_name, ret_type, child.is_statically_non_null, child.can_overflow};
  }

  //===----------CONSTANT OR NULL----------===//
  if (fid == sirius::function_id::constant_or_null) {
    auto const val_reg     = produce(*args[0]);
    std::string valid_expr = val_reg.var_name + ".valid";
    std::string ovf_expr   = val_reg.var_name + ".overflow";
    bool any_ovf           = val_reg.can_overflow;
    for (std::size_t i = 1; i < args.size(); ++i) {
      auto const r = produce(*args[i]);
      valid_expr += " && " + r.var_name + ".valid";
      ovf_expr += " || " + r.var_name + ".overflow";
      any_ovf = any_ovf || r.can_overflow;
    }
    emit_line(format("bool const {}_valid = {};", var_name, valid_expr));
    emit_reg_val(
      type_str,
      var_name,
      format(
        "{0}_valid ? static_cast<{1}>({2}.val) : {1}{{}}", var_name, type_str, val_reg.var_name),
      var_name + "_valid",
      any_ovf ? "(" + ovf_expr + ")" : "false");
    return register_value{var_name, ret_type, /*is_statically_non_null=*/false, any_ovf};
  }

  //===----------STRING LENGTH FUNCTIONS----------===//
  if (fid == sirius::function_id::strlen || fid == sirius::function_id::length) {
    auto const child = produce(*args[0]);
    emit_reg_val_assign(
      type_str,
      var_name,
      format("{}<{}>({})",
             fid == sirius::function_id::strlen ? "eval_strlen" : "eval_utf8_length",
             type_str,
             child.var_name));
    return register_value{var_name, ret_type, child.is_statically_non_null, child.can_overflow};
  }

  throw not_implemented_exception("[nvrtc] Unsupported function_id in translation_context: {}",
                                  static_cast<int>(fid));
}

register_value translation_context::produce(sirius::ast::node const& expr)
{
  if (_boundary_map != nullptr) {
    if (auto const it = _boundary_map->find(&expr); it != _boundary_map->end()) {
      sirius::ast::reference syn_ref{it->second, expr.return_type()};
      return produce_reference(syn_ref);
    }
  }
  return std::visit(
    [this](auto const& alt) -> register_value {
      using T = std::decay_t<decltype(alt)>;
      if constexpr (std::is_same_v<T, sirius::ast::reference>) {
        return produce_reference(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::constant>) {
        return produce_constant(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::comparison>) {
        return produce_comparison(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::conjunction>) {
        return produce_conjunction(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::between>) {
        return produce_between(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::case_expr>) {
        return produce_case_expr(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::coalesce>) {
        return produce_coalesce(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::in_list>) {
        return produce_in_list(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::cast>) {
        return produce_cast(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::unary_op>) {
        return produce_unary_op(alt);
      } else if constexpr (std::is_same_v<T, sirius::ast::function_call>) {
        return produce_function_call(alt);
      } else {
        throw not_implemented_exception(
          "[nvrtc] Unsupported AST node in translation_context::produce");
      }
    },
    expr.v);
}

}  // namespace sirius::nvrtc
