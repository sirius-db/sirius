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

#include "codegen/jit/kernel_cache.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/util/cuda_check.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda.h>

#include <expression/ast/constant_range.hpp>
#include <expression/date_trunc_unit.hpp>
#include <expression/function_id.hpp>
#include <expression/join_condition.hpp>
#include <expression/substring_slice.hpp>
#include <expression_evaluator/like_multiliteral.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sirius::nvrtc {

namespace {

[[nodiscard]] bool is_integer_or_float(cudf::data_type type) noexcept
{
  switch (type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64: return true;
    default: return false;
  }
}

[[nodiscard]] bool is_untyped_reference_type(sirius::logical_type const& t) noexcept
{
  return t.id() == sirius::type_id::INVALID || t.id() == sirius::type_id::SQLNULL;
}

[[nodiscard]] std::int32_t sql_decimal_scale(cudf::data_type type) noexcept
{
  return -type.scale();
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
  if (hi == 0) { return "static_cast<__int128_t>(" + std::to_string(lo) + "ULL)"; }
  return "((static_cast<__int128_t>(" + std::to_string(hi) +
         "ULL) << 64) | static_cast<__int128_t>(" + std::to_string(lo) + "ULL))";
}

[[nodiscard]] std::string format_double_pow10(int exp)
{
  double val = 1.0;
  for (int i = 0; i < exp; ++i) {
    val *= 10.0;
  }
  std::ostringstream oss;
  oss.precision(17);
  oss << std::scientific << val;
  return oss.str();
}

[[nodiscard]] std::int64_t timestamp_ticks_per_second(cudf::type_id id) noexcept
{
  switch (id) {
    case cudf::type_id::TIMESTAMP_SECONDS: return 1LL;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return 1'000LL;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return 1'000'000LL;
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return 1'000'000'000LL;
    default: return 0LL;
  }
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

[[nodiscard]] bool is_semantic_cast_supported(cudf::data_type src, cudf::data_type dst) noexcept
{
  if (!is_supported_fixed_width(src) || !is_supported_fixed_width(dst)) { return false; }
  bool const src_temp = cudf::is_timestamp(src);
  bool const dst_temp = cudf::is_timestamp(dst);
  if (src_temp != dst_temp) { return false; }
  if (src_temp) {
    if (src.id() == dst.id()) { return true; }
    if (src.id() == cudf::type_id::TIMESTAMP_DAYS) {
      return dst.id() == cudf::type_id::TIMESTAMP_SECONDS ||
             dst.id() == cudf::type_id::TIMESTAMP_MILLISECONDS ||
             dst.id() == cudf::type_id::TIMESTAMP_MICROSECONDS ||
             dst.id() == cudf::type_id::TIMESTAMP_NANOSECONDS;
    }
    return timestamp_ticks_per_second(src.id()) > 0 &&
           (dst.id() == cudf::type_id::TIMESTAMP_DAYS || timestamp_ticks_per_second(dst.id()) > 0);
  }
  return true;
}

[[nodiscard]] cudf::data_type effective_node_cudf_type(
  sirius::ast::node const& expr,
  cudf::table_view const& input,
  boundary_column_map const* boundary_map = nullptr) noexcept
{
  if (boundary_map != nullptr) {
    if (auto const it = boundary_map->find(&expr); it != boundary_map->end()) {
      if (it->second < static_cast<std::uint32_t>(input.num_columns())) {
        return input.column(it->second).type();
      }
      auto const opt_cudf = sirius::get_cudf_type(expr.return_type());
      return opt_cudf.value_or(cudf::data_type{cudf::type_id::EMPTY});
    }
  }
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

[[nodiscard]] std::optional<std::int64_t> extract_constant_int64(
  sirius::ast::node const& node) noexcept
{
  if (!node.holds<sirius::ast::constant>()) { return std::nullopt; }
  auto const& payload = node.get<sirius::ast::constant>().payload;
  if (std::holds_alternative<std::int64_t>(payload)) { return std::get<std::int64_t>(payload); }
  if (std::holds_alternative<std::int32_t>(payload)) {
    return static_cast<std::int64_t>(std::get<std::int32_t>(payload));
  }
  return std::nullopt;
}

[[nodiscard]] bool is_node_supported(sirius::ast::node const& expr,
                                     cudf::table_view const& input,
                                     boundary_column_map const* boundary_map = nullptr) noexcept
{
  if (boundary_map != nullptr) {
    if (auto const it = boundary_map->find(&expr); it != boundary_map->end()) {
      if (it->second < static_cast<std::uint32_t>(input.num_columns())) {
        auto const col_view = input.column(it->second);
        if (col_view.type().id() == cudf::type_id::STRING) {
          if (col_view.num_children() > 0) {
            auto const off_id = cudf::strings_column_view(col_view).offsets().type().id();
            if (off_id != cudf::type_id::INT32 && off_id != cudf::type_id::INT64) { return false; }
          }
          return true;
        }
        return is_supported_fixed_width(col_view.type());
      }
      auto const opt_cudf = sirius::get_cudf_type(expr.return_type());
      return opt_cudf.has_value() && is_supported_register_type(*opt_cudf);
    }
  }

  if (expr.holds<sirius::ast::reference>()) {
    auto const& ref = expr.get<sirius::ast::reference>();
    if (ref.column_index >= static_cast<std::uint32_t>(input.num_columns())) { return false; }
    auto const col_view = input.column(ref.column_index);
    auto const& logical = ref.return_type();
    if (col_view.type().id() == cudf::type_id::STRING) {
      if (col_view.num_children() > 0) {
        auto const off_id = cudf::strings_column_view(col_view).offsets().type().id();
        if (off_id != cudf::type_id::INT32 && off_id != cudf::type_id::INT64) { return false; }
      }
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

  if (expr.holds<sirius::ast::constant>()) {
    auto const& cst     = expr.get<sirius::ast::constant>();
    auto const opt_cudf = sirius::get_cudf_type(cst.return_type());
    return opt_cudf.has_value() && is_supported_register_type(*opt_cudf);
  }

  if (expr.holds<sirius::ast::comparison>()) {
    auto const& cmp = expr.get<sirius::ast::comparison>();
    if (!cmp.left || !cmp.right) { return false; }
    if (!is_node_supported(*cmp.left, input, boundary_map) ||
        !is_node_supported(*cmp.right, input, boundary_map)) {
      return false;
    }
    auto const lt = effective_node_cudf_type(*cmp.left, input, boundary_map);
    auto const rt = effective_node_cudf_type(*cmp.right, input, boundary_map);
    if (lt.id() == cudf::type_id::STRING || rt.id() == cudf::type_id::STRING) {
      return lt.id() == cudf::type_id::STRING && rt.id() == cudf::type_id::STRING;
    }
    if (!is_supported_fixed_width(lt) || !is_supported_fixed_width(rt)) { return false; }
    if (cudf::is_fixed_point(lt) || cudf::is_fixed_point(rt)) {
      return cudf::is_fixed_point(lt) && cudf::is_fixed_point(rt);
    }
    return true;
  }

  if (expr.holds<sirius::ast::between>()) {
    auto const& btw = expr.get<sirius::ast::between>();
    if (!btw.input || !btw.lower || !btw.upper) { return false; }
    if (!is_node_supported(*btw.input, input, boundary_map) ||
        !is_node_supported(*btw.lower, input, boundary_map) ||
        !is_node_supported(*btw.upper, input, boundary_map)) {
      return false;
    }
    auto const it = effective_node_cudf_type(*btw.input, input, boundary_map);
    auto const lt = effective_node_cudf_type(*btw.lower, input, boundary_map);
    auto const ut = effective_node_cudf_type(*btw.upper, input, boundary_map);
    if (it.id() == cudf::type_id::STRING || lt.id() == cudf::type_id::STRING ||
        ut.id() == cudf::type_id::STRING) {
      return it.id() == cudf::type_id::STRING && lt.id() == cudf::type_id::STRING &&
             ut.id() == cudf::type_id::STRING;
    }
    if (!is_supported_fixed_width(it) || !is_supported_fixed_width(lt) ||
        !is_supported_fixed_width(ut)) {
      return false;
    }
    if (cudf::is_fixed_point(it) || cudf::is_fixed_point(lt) || cudf::is_fixed_point(ut)) {
      return cudf::is_fixed_point(it) && cudf::is_fixed_point(lt) && cudf::is_fixed_point(ut);
    }
    return true;
  }

  if (expr.holds<sirius::ast::conjunction>()) {
    auto const& conj = expr.get<sirius::ast::conjunction>();
    if (conj.children.empty()) { return false; }
    if (conj.op != sirius::ast::conjunction::kind::op_and &&
        conj.op != sirius::ast::conjunction::kind::op_or) {
      return false;
    }
    for (auto const& ch : conj.children) {
      if (!ch || !is_node_supported(*ch, input, boundary_map)) { return false; }
      if (effective_node_cudf_type(*ch, input, boundary_map).id() != cudf::type_id::BOOL8) {
        return false;
      }
    }
    return true;
  }

  if (expr.holds<sirius::ast::case_expr>()) {
    auto const& ce = expr.get<sirius::ast::case_expr>();
    if (ce.cases.empty() || !ce.else_) { return false; }
    auto const opt_ret = sirius::get_cudf_type(ce.return_type());
    if (!opt_ret || !is_supported_register_type(*opt_ret)) { return false; }
    auto const ret_type = *opt_ret;
    for (auto const& c : ce.cases) {
      if (!c.when_ || !c.then_) { return false; }
      if (!is_node_supported(*c.when_, input, boundary_map)) { return false; }
      if (effective_node_cudf_type(*c.when_, input, boundary_map).id() != cudf::type_id::BOOL8) {
        return false;
      }
      if (c.then_->holds<sirius::ast::function_call>() &&
          c.then_->get<sirius::ast::function_call>().function() == sirius::function_id::error) {
        continue;
      }
      if (!is_node_supported(*c.then_, input, boundary_map)) { return false; }
      auto const tt = effective_node_cudf_type(*c.then_, input, boundary_map);
      if (!is_supported_register_type(tt)) { return false; }
      if (ret_type.id() == cudf::type_id::STRING || tt.id() == cudf::type_id::STRING) {
        if (ret_type.id() != cudf::type_id::STRING || tt.id() != cudf::type_id::STRING) {
          return false;
        }
      }
      if (cudf::is_fixed_point(ret_type) || cudf::is_fixed_point(tt)) {
        if (!cudf::is_fixed_point(ret_type) || !cudf::is_fixed_point(tt) ||
            ret_type.scale() != tt.scale()) {
          return false;
        }
      }
    }
    if (!is_node_supported(*ce.else_, input, boundary_map)) { return false; }
    auto const et = effective_node_cudf_type(*ce.else_, input, boundary_map);
    if (!is_supported_register_type(et)) { return false; }
    if (ret_type.id() == cudf::type_id::STRING || et.id() == cudf::type_id::STRING) {
      if (ret_type.id() != cudf::type_id::STRING || et.id() != cudf::type_id::STRING) {
        return false;
      }
    }
    if (cudf::is_fixed_point(ret_type) || cudf::is_fixed_point(et)) {
      if (!cudf::is_fixed_point(ret_type) || !cudf::is_fixed_point(et) ||
          ret_type.scale() != et.scale()) {
        return false;
      }
    }
    return true;
  }

  if (expr.holds<sirius::ast::coalesce>()) {
    auto const& coal = expr.get<sirius::ast::coalesce>();
    if (coal.children.empty()) { return false; }
    auto const opt_ret = sirius::get_cudf_type(coal.return_type());
    if (!opt_ret || !is_supported_register_type(*opt_ret)) { return false; }
    auto const ret_type = *opt_ret;
    for (auto const& ch : coal.children) {
      if (!ch || !is_node_supported(*ch, input, boundary_map)) { return false; }
      auto const ct = effective_node_cudf_type(*ch, input, boundary_map);
      if (!is_supported_register_type(ct)) { return false; }
      if (ret_type.id() == cudf::type_id::STRING || ct.id() == cudf::type_id::STRING) {
        if (ret_type.id() != cudf::type_id::STRING || ct.id() != cudf::type_id::STRING) {
          return false;
        }
      }
      if (cudf::is_fixed_point(ret_type) || cudf::is_fixed_point(ct)) {
        if (!cudf::is_fixed_point(ret_type) || !cudf::is_fixed_point(ct) ||
            ret_type.scale() != ct.scale()) {
          return false;
        }
      }
    }
    return true;
  }

  if (expr.holds<sirius::ast::in_list>()) {
    auto const& inl = expr.get<sirius::ast::in_list>();
    if (!inl.probe || inl.values.empty()) { return false; }
    if (!is_node_supported(*inl.probe, input, boundary_map)) { return false; }
    auto const pt = effective_node_cudf_type(*inl.probe, input, boundary_map);
    if (!is_supported_register_type(pt)) { return false; }
    for (auto const& v : inl.values) {
      if (!v || !is_node_supported(*v, input, boundary_map)) { return false; }
      auto const vt = effective_node_cudf_type(*v, input, boundary_map);
      if (!is_supported_register_type(vt)) { return false; }
      if (pt.id() == cudf::type_id::STRING || vt.id() == cudf::type_id::STRING) {
        if (pt.id() != cudf::type_id::STRING || vt.id() != cudf::type_id::STRING) { return false; }
      }
      if (cudf::is_fixed_point(pt) || cudf::is_fixed_point(vt)) {
        if (!cudf::is_fixed_point(pt) || !cudf::is_fixed_point(vt)) { return false; }
      }
    }
    return true;
  }

  if (expr.holds<sirius::ast::cast>()) {
    auto const& cst = expr.get<sirius::ast::cast>();
    if (!cst.child || !is_node_supported(*cst.child, input, boundary_map)) { return false; }
    auto const opt_dst = sirius::get_cudf_type(cst.target_type);
    if (!opt_dst || !is_supported_fixed_width(*opt_dst)) { return false; }
    auto const dst_type = *opt_dst;
    auto const src_type = effective_node_cudf_type(*cst.child, input, boundary_map);
    if (!is_supported_fixed_width(src_type)) { return false; }
    if (cst.kind == sirius::ast::cast_kind::carrier_restore) {
      return src_type == dst_type || can_restore_to(src_type, dst_type);
    }
    return is_semantic_cast_supported(src_type, dst_type);
  }

  if (expr.holds<sirius::ast::unary_op>()) {
    auto const& uo = expr.get<sirius::ast::unary_op>();
    if (!uo.child || !is_node_supported(*uo.child, input, boundary_map)) { return false; }
    switch (uo.op) {
      case sirius::ast::unary_op::kind::op_not: {
        auto const ct = effective_node_cudf_type(*uo.child, input, boundary_map);
        return ct.id() == cudf::type_id::BOOL8;
      }
      case sirius::ast::unary_op::kind::op_is_null:
      case sirius::ast::unary_op::kind::op_is_not_null: return true;
      case sirius::ast::unary_op::kind::op_try: {
        auto const rt = effective_node_cudf_type(*uo.child, input, boundary_map);
        return is_supported_register_type(rt);
      }
      default: return false;
    }
  }

  if (expr.holds<sirius::ast::function_call>()) {
    auto const& fc     = expr.get<sirius::ast::function_call>();
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
      if (!args[1] || !is_node_supported(*args[1], input, boundary_map)) { return false; }
      auto const ct = effective_node_cudf_type(*args[1], input, boundary_map);
      return cudf::is_timestamp(ct) && cudf::is_timestamp(ret_type);
    }

    if (fc.function() == sirius::function_id::substring) {
      if ((args.size() != 2 && args.size() != 3) || ret_type.id() != cudf::type_id::STRING) {
        return false;
      }
      if (!args[0] || !is_node_supported(*args[0], input, boundary_map)) { return false; }
      if (effective_node_cudf_type(*args[0], input, boundary_map).id() != cudf::type_id::STRING) {
        return false;
      }
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
      if (!arg || !is_node_supported(*arg, input, boundary_map)) { return false; }
    }

    switch (fc.function()) {
      case sirius::function_id::add:
      case sirius::function_id::sub:
      case sirius::function_id::mul:
      case sirius::function_id::div:
      case sirius::function_id::int_div:
      case sirius::function_id::mod: {
        if (args.size() != 2) { return false; }
        auto const lt = effective_node_cudf_type(*args[0], input, boundary_map);
        auto const rt = effective_node_cudf_type(*args[1], input, boundary_map);
        if (cudf::is_fixed_point(ret_type)) {
          return cudf::is_fixed_point(lt) && cudf::is_fixed_point(rt);
        }
        return is_integer_or_float(ret_type) && is_integer_or_float(lt) && is_integer_or_float(rt);
      }
      case sirius::function_id::abs: {
        if (args.size() != 1) { return false; }
        auto const ct = effective_node_cudf_type(*args[0], input, boundary_map);
        if (cudf::is_fixed_point(ret_type)) { return cudf::is_fixed_point(ct) && ct == ret_type; }
        return is_integer_or_float(ret_type) && ct == ret_type;
      }
      case sirius::function_id::round: {
        if (args.size() != 1 && args.size() != 2) { return false; }
        auto const ct = effective_node_cudf_type(*args[0], input, boundary_map);
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
        if (args.size() < 2 || !is_supported_register_type(ret_type)) { return false; }
        auto const v0_t = effective_node_cudf_type(*args[0], input, boundary_map);
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
        auto const ct = effective_node_cudf_type(*args[0], input, boundary_map);
        return cudf::is_timestamp(ct) && cudf::is_integral(ret_type);
      }
      case sirius::function_id::strlen:
      case sirius::function_id::length: {
        if (args.size() != 1) { return false; }
        auto const ct = effective_node_cudf_type(*args[0], input, boundary_map);
        return ct.id() == cudf::type_id::STRING && cudf::is_integral(ret_type);
      }
      case sirius::function_id::prefix:
      case sirius::function_id::suffix:
      case sirius::function_id::contains:
      case sirius::function_id::like:
      case sirius::function_id::not_like: {
        if (args.size() != 2 || ret_type.id() != cudf::type_id::BOOL8) { return false; }
        auto const lt = effective_node_cudf_type(*args[0], input, boundary_map);
        auto const rt = effective_node_cudf_type(*args[1], input, boundary_map);
        return lt.id() == cudf::type_id::STRING && rt.id() == cudf::type_id::STRING;
      }
      default: return false;
    }
  }

  return false;
}

[[nodiscard]] bool is_in_tree_materialization_boundary(sirius::ast::node const& expr,
                                                       cudf::table_view const& input) noexcept
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
    auto const src_type = effective_node_cudf_type(*cst.child, input, nullptr);
    if (cst.kind == sirius::ast::cast_kind::carrier_restore) {
      return !(src_type == *opt_dst || can_restore_to(src_type, *opt_dst));
    }
    return !is_semantic_cast_supported(src_type, *opt_dst);
  }
  return false;
}

template <typename Fn>
void for_each_child_node(sirius::ast::node const& expr, Fn&& fn)
{
  std::visit(
    [&](auto const& alt) {
      using T = std::decay_t<decltype(alt)>;
      if constexpr (std::is_same_v<T, sirius::ast::comparison>) {
        if (alt.left) { fn(*alt.left); }
        if (alt.right) { fn(*alt.right); }
      } else if constexpr (std::is_same_v<T, sirius::ast::between>) {
        if (alt.input) { fn(*alt.input); }
        if (alt.lower) { fn(*alt.lower); }
        if (alt.upper) { fn(*alt.upper); }
      } else if constexpr (std::is_same_v<T, sirius::ast::conjunction> ||
                           std::is_same_v<T, sirius::ast::coalesce>) {
        for (auto const& ch : alt.children) {
          if (ch) { fn(*ch); }
        }
      } else if constexpr (std::is_same_v<T, sirius::ast::case_expr>) {
        for (auto const& c : alt.cases) {
          if (c.when_) { fn(*c.when_); }
          if (c.then_) { fn(*c.then_); }
        }
        if (alt.else_) { fn(*alt.else_); }
      } else if constexpr (std::is_same_v<T, sirius::ast::in_list>) {
        if (alt.probe) { fn(*alt.probe); }
        for (auto const& v : alt.values) {
          if (v) { fn(*v); }
        }
      } else if constexpr (std::is_same_v<T, sirius::ast::cast> ||
                           std::is_same_v<T, sirius::ast::unary_op>) {
        if (alt.child) { fn(*alt.child); }
      } else if constexpr (std::is_same_v<T, sirius::ast::function_call>) {
        for (auto const& arg : alt.arguments()) {
          if (arg) { fn(*arg); }
        }
      }
    },
    expr.v);
}

void collect_boundaries_from_node(sirius::ast::node const& expr,
                                  cudf::table_view const& input,
                                  std::vector<sirius::ast::node const*>& out_boundaries)
{
  if (is_in_tree_materialization_boundary(expr, input)) {
    if (std::find(out_boundaries.begin(), out_boundaries.end(), &expr) == out_boundaries.end()) {
      out_boundaries.push_back(&expr);
    }
    return;
  }

  for_each_child_node(expr, [&](sirius::ast::node const& child) {
    collect_boundaries_from_node(child, input, out_boundaries);
  });
}

[[nodiscard]] cudf::data_type resolve_output_cudf_type(sirius::ast::node const& expr,
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

[[nodiscard]] std::unique_ptr<cudf::column> make_fixed_width_output_column(
  cudf::data_type type,
  cudf::size_type num_rows,
  rmm::device_buffer&& null_mask,
  cudf::size_type null_count,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (cudf::is_numeric(type) || cudf::is_boolean(type)) {
    return cudf::make_numeric_column(type, num_rows, std::move(null_mask), null_count, stream, mr);
  }
  if (cudf::is_timestamp(type)) {
    return cudf::make_timestamp_column(
      type, num_rows, std::move(null_mask), null_count, stream, mr);
  }
  if (cudf::is_fixed_point(type)) {
    return cudf::make_fixed_point_column(
      type, num_rows, std::move(null_mask), null_count, stream, mr);
  }
  throw not_implemented_exception("[nvrtc] Cannot allocate output column for type: {}",
                                  cudf::type_to_name(type));
}

[[nodiscard]] std::unique_ptr<cudf::column> make_unallocated_fixed_width_output_column(
  cudf::data_type type,
  cudf::size_type num_rows,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (cudf::is_numeric(type) || cudf::is_boolean(type)) {
    return cudf::make_numeric_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  if (cudf::is_timestamp(type)) {
    return cudf::make_timestamp_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  if (cudf::is_fixed_point(type)) {
    return cudf::make_fixed_point_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  throw not_implemented_exception("[nvrtc] Cannot allocate output column for type: {}",
                                  cudf::type_to_name(type));
}

// Host-side 16-byte aligned storage for a scalar constant parameter passed to cuLaunchKernel.
struct alignas(16) constant_param_storage {
  std::uint8_t val_bytes[16]{};
  std::uint8_t valid{0};
};

void populate_constant_param(scalar_constant_binding const& binding,
                             constant_param_storage& storage)
{
  std::memset(&storage, 0, sizeof(storage));
  storage.valid = binding.is_valid ? 1U : 0U;
  if (!binding.is_valid) { return; }

  switch (binding.device_type.id()) {
    case cudf::type_id::INT8: {
      auto const v = std::get<std::int8_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::INT16: {
      auto const v = std::get<std::int16_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::INT32: {
      auto const v = std::get<std::int32_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::INT64: {
      auto const v = std::get<std::int64_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::UINT8: {
      auto const v = std::get<std::uint8_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::UINT16: {
      auto const v = std::get<std::uint16_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::UINT32: {
      auto const v = std::get<std::uint32_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::UINT64: {
      auto const v = std::get<std::uint64_t>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::FLOAT32: {
      auto const v = std::get<float>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::FLOAT64: {
      auto const v = std::get<double>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::BOOL8: {
      auto const v = std::get<bool>(binding.payload);
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      auto const v = std::get<sirius::date_value>(binding.payload).days;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      auto const v = std::get<sirius::timestamp_sec_value>(binding.payload).value;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      auto const v = std::get<sirius::timestamp_ms_value>(binding.payload).value;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      auto const v = std::get<sirius::timestamp_us_value>(binding.payload).value;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      auto const v = std::get<sirius::timestamp_ns_value>(binding.payload).value;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::DECIMAL32: {
      auto const v = std::get<sirius::decimal32>(binding.payload).value;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::DECIMAL64: {
      auto const v = std::get<sirius::decimal64>(binding.payload).value;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    case cudf::type_id::DECIMAL128: {
      auto const v = std::get<sirius::decimal128>(binding.payload).value;
      std::memcpy(storage.val_bytes, &v, sizeof(v));
      break;
    }
    default:
      throw not_implemented_exception("[nvrtc] Unsupported constant parameter type: {}",
                                      cudf::type_to_name(binding.device_type));
  }
}

template <typename ColViewT>
[[nodiscard]] auto get_column_data_ptr(ColViewT& col)
  -> std::conditional_t<std::is_const_v<ColViewT>, void const*, void*>
{
  switch (col.type().id()) {
    case cudf::type_id::INT8: return col.template data<std::int8_t>();
    case cudf::type_id::INT16: return col.template data<std::int16_t>();
    case cudf::type_id::INT32: return col.template data<std::int32_t>();
    case cudf::type_id::INT64: return col.template data<std::int64_t>();
    case cudf::type_id::UINT8: return col.template data<std::uint8_t>();
    case cudf::type_id::UINT16: return col.template data<std::uint16_t>();
    case cudf::type_id::UINT32: return col.template data<std::uint32_t>();
    case cudf::type_id::UINT64: return col.template data<std::uint64_t>();
    case cudf::type_id::FLOAT32: return col.template data<float>();
    case cudf::type_id::FLOAT64: return col.template data<double>();
    case cudf::type_id::BOOL8: return col.template data<bool>();
    case cudf::type_id::TIMESTAMP_DAYS: return col.template data<std::int32_t>();
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return col.template data<std::int64_t>();
    case cudf::type_id::DECIMAL32: return col.template data<std::int32_t>();
    case cudf::type_id::DECIMAL64: return col.template data<std::int64_t>();
    case cudf::type_id::DECIMAL128: return col.template data<__int128_t>();
    default:
      throw not_implemented_exception("[nvrtc] Unsupported column type: {}",
                                      cudf::type_to_name(col.type()));
  }
}

[[nodiscard]] void const* get_input_data_ptr(cudf::column_view const& col)
{
  return get_column_data_ptr(col);
}

[[nodiscard]] void* get_output_data_ptr(cudf::mutable_column_view& col)
{
  return get_column_data_ptr(col);
}

}  // namespace

std::size_t translation_context::get_or_bind_input_column(std::uint32_t column_index)
{
  auto const it = std::find_if(
    _input_columns.begin(), _input_columns.end(), [column_index](input_column_binding const& b) {
      return b.column_index == column_index;
    });
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

register_value translation_context::produce_reference(sirius::ast::reference const& expr)
{
  auto const col_view           = _input.column(expr.column_index);
  auto const& logical           = expr.return_type();
  cudf::data_type restored_type = col_view.type();
  sirius::logical_type sql_type = logical;

  if (!is_untyped_reference_type(logical)) {
    auto const native = sirius::require_cudf_type(logical);
    if (is_narrowable_numeric_type(logical) && can_restore_to(col_view.type(), native)) {
      restored_type = native;
    }
  }

  auto const existing = std::find_if(
    _loaded_references.begin(), _loaded_references.end(), [&](loaded_reference_entry const& entry) {
      return entry.column_index == expr.column_index && entry.restored_type == restored_type;
    });
  if (existing != _loaded_references.end()) { return existing->reg; }

  auto const binding_id   = get_or_bind_input_column(expr.column_index);
  auto const& col_binding = _input_columns.at(binding_id);
  auto const var_name     = next_var_name();
  auto const dst_type_str = cuda_type_name(restored_type);
  auto const id_str       = std::to_string(binding_id);

  if (restored_type.id() == cudf::type_id::STRING) {
    if (!col_binding.is_nullable) {
      emit_prelude_line("reg_val<str_view> " + var_name + "{(in_" + id_str +
                        "_offsets != nullptr) ? str_view{in_" + id_str + "_chars + in_" + id_str +
                        "_offsets[row], static_cast<int32_t>(in_" + id_str +
                        "_offsets[row + 1] - in_" + id_str + "_offsets[row]), in_" + id_str +
                        "_chars_end} : str_view{nullptr, 0, nullptr}, true, false};");
    } else {
      // The validity bit and the offsets are loaded independently (a null row still has
      // offsets) so the two loads overlap instead of forming a dependent chain.
      emit_prelude_line("bool const " + var_name + "_valid = is_bit_set(in_" + id_str +
                        "_mask, in_" + id_str + "_bit_offset + row);");
      emit_prelude_line("reg_val<str_view> " + var_name + "{(" + var_name + "_valid && in_" +
                        id_str + "_offsets != nullptr) ? str_view{in_" + id_str + "_chars + in_" +
                        id_str + "_offsets[row], static_cast<int32_t>(in_" + id_str +
                        "_offsets[row + 1] - in_" + id_str + "_offsets[row]), in_" + id_str +
                        "_chars_end} : str_view{nullptr, 0, nullptr}, " + var_name +
                        "_valid, false};");
    }
  } else if (!col_binding.is_nullable) {
    emit_prelude_line("reg_val<" + dst_type_str + "> " + var_name + "{static_cast<" + dst_type_str +
                      ">(in_" + id_str + "_data[row]), true, false};");
  } else {
    // A null row's payload slot exists, so the payload is loaded unconditionally alongside the
    // validity bit (two independent loads) and zeroed afterwards for a deterministic register.
    emit_prelude_line("bool const " + var_name + "_valid = is_bit_set(in_" + id_str + "_mask, in_" +
                      id_str + "_bit_offset + row);");
    emit_prelude_line(dst_type_str + " const " + var_name + "_raw = static_cast<" + dst_type_str +
                      ">(in_" + id_str + "_data[row]);");
    emit_prelude_line("reg_val<" + dst_type_str + "> " + var_name + "{" + var_name + "_valid ? " +
                      var_name + "_raw : " + dst_type_str + "{}, " + var_name + "_valid, false};");
  }

  register_value reg{
    var_name, sql_type, restored_type, !col_binding.is_nullable, /*can_overflow=*/false};
  _loaded_references.push_back(loaded_reference_entry{expr.column_index, restored_type, reg});
  return reg;
}

register_value translation_context::produce_constant(sirius::ast::constant const& expr)
{
  auto const device_type = sirius::require_cudf_type(expr.return_type());
  bool const is_valid    = !std::holds_alternative<sirius::null_value>(expr.payload);
  auto const binding_id  = _constants.size();
  auto const var_name    = next_var_name();
  auto const type_str    = cuda_type_name(device_type);
  auto const id_str      = std::to_string(binding_id);

  // A constant's validity is a property of the expression, not of the batch, so it is baked
  // into the kernel source: valid constants are statically non-null (so consumers elide their
  // validity chains and the Sink skips the null mask) and a NULL constant needs no parameter.
  if (!is_valid) {
    emit_prelude_line("reg_val<" + type_str + "> const " + var_name + "{" + type_str +
                      "{}, false, false};");
  } else if (device_type.id() == cudf::type_id::STRING) {
    emit_prelude_line("reg_val<str_view> const " + var_name + "{str_view{c_" + id_str + "_ptr, c_" +
                      id_str + "_len, c_" + id_str + "_ptr + c_" + id_str + "_len}, true, false};");
  } else {
    emit_prelude_line("reg_val<" + type_str + "> const " + var_name + "{c_" + id_str +
                      "_val, true, false};");
  }

  register_value reg{var_name,
                     expr.return_type(),
                     device_type,
                     /*is_statically_non_null=*/is_valid,
                     /*can_overflow=*/false};
  _constants.push_back(scalar_constant_binding{
    binding_id, expr.return_type(), device_type, expr.payload, is_valid, reg});
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
    switch (carrier.id()) {
      case cudf::type_id::INT8: narrowed_payload = static_cast<std::int8_t>(raw_val); break;
      case cudf::type_id::INT16: narrowed_payload = static_cast<std::int16_t>(raw_val); break;
      case cudf::type_id::INT32: narrowed_payload = static_cast<std::int32_t>(raw_val); break;
      case cudf::type_id::UINT8: narrowed_payload = static_cast<std::uint8_t>(raw_val); break;
      case cudf::type_id::UINT16: narrowed_payload = static_cast<std::uint16_t>(raw_val); break;
      case cudf::type_id::UINT32: narrowed_payload = static_cast<std::uint32_t>(raw_val); break;
      case cudf::type_id::DECIMAL32:
        narrowed_payload = sirius::decimal32{static_cast<std::int32_t>(raw_val),
                                             static_cast<std::uint8_t>(-carrier.scale())};
        break;
      case cudf::type_id::DECIMAL64:
        narrowed_payload = sirius::decimal64{static_cast<std::int64_t>(raw_val),
                                             static_cast<std::uint8_t>(-carrier.scale())};
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
    emit_prelude_line("reg_val<" + type_str + "> const " + var_name + "{c_" +
                      std::to_string(binding_id) + "_val, true, false};");
  } else {
    emit_prelude_line("reg_val<" + type_str + "> const " + var_name + "{" + type_str +
                      "{}, false, false};");
  }

  register_value reg{var_name,
                     expr.return_type(),
                     carrier,
                     /*is_statically_non_null=*/is_valid,
                     /*can_overflow=*/false};
  _constants.push_back(scalar_constant_binding{
    binding_id, expr.return_type(), carrier, std::move(narrowed_payload), is_valid, reg});
  return reg;
}

register_value translation_context::produce_narrow_domain_operand(
  sirius::ast::node const& operand, std::optional<cudf::data_type> narrow_carrier)
{
  if (!narrow_carrier) { return produce(operand); }
  if (operand.holds<sirius::ast::reference>()) {
    auto const& ref = operand.get<sirius::ast::reference>();
    // Load the reference directly at its raw carrier width without restoring to the wider type.
    sirius::ast::reference raw_ref{ref.column_index, sirius::logical_type{}};
    return produce_reference(raw_ref);
  }
  return produce_constant_in_carrier(operand.get<sirius::ast::constant>(), *narrow_carrier);
}

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
  std::int32_t const s_l = is_dec ? sql_decimal_scale(lhs.device_type) : 0;
  std::int32_t const s_r = is_dec ? sql_decimal_scale(rhs.device_type) : 0;

  auto const cmp_type_str = [&]() -> std::string {
    if (is_dec && s_l != s_r) { return "__int128_t"; }
    if (lhs.device_type == rhs.device_type) { return cuda_type_name(lhs.device_type); }
    if (lhs.device_type.id() == cudf::type_id::FLOAT64 ||
        rhs.device_type.id() == cudf::type_id::FLOAT64) {
      return "double";
    }
    if (lhs.device_type.id() == cudf::type_id::FLOAT32 ||
        rhs.device_type.id() == cudf::type_id::FLOAT32) {
      return "float";
    }
    if (cudf::is_unsigned(lhs.device_type) && cudf::is_unsigned(rhs.device_type)) {
      return "uint64_t";
    }
    if (!cudf::is_unsigned(lhs.device_type) && !cudf::is_unsigned(rhs.device_type) && !is_dec) {
      return "int64_t";
    }
    return "__int128_t";
  }();

  std::string lhs_val = "static_cast<" + cmp_type_str + ">(" + lhs.var_name + ".val)";
  std::string rhs_val = "static_cast<" + cmp_type_str + ">(" + rhs.var_name + ".val)";
  if (is_dec && s_l != s_r) {
    auto const s_max = std::max(s_l, s_r);
    if (s_max > s_l) { lhs_val = "(" + lhs_val + " * " + format_int128_pow10(s_max - s_l) + ")"; }
    if (s_max > s_r) { rhs_val = "(" + rhs_val + " * " + format_int128_pow10(s_max - s_r) + ")"; }
  }
  bool const can_ovf = lhs.can_overflow || rhs.can_overflow;
  std::string const ovf_e =
    can_ovf ? (lhs.var_name + ".overflow || " + rhs.var_name + ".overflow") : "false";

  if (expr.op == sirius::comparison_type::not_distinct_from) {
    emit_line("reg_val<bool> " + var_name + "{(!" + lhs.var_name + ".valid && !" + rhs.var_name +
              ".valid) || (" + lhs.var_name + ".valid && " + rhs.var_name + ".valid && (" +
              lhs_val + " == " + rhs_val + ")), true, " + ovf_e + "};");
    return register_value{var_name,
                          sirius::logical_type::make(sirius::type_id::BOOLEAN),
                          cudf::data_type{cudf::type_id::BOOL8},
                          /*is_statically_non_null=*/true,
                          can_ovf};
  }

  if (expr.op == sirius::comparison_type::distinct_from) {
    emit_line("reg_val<bool> " + var_name + "{(" + lhs.var_name + ".valid != " + rhs.var_name +
              ".valid) || (" + lhs.var_name + ".valid && " + rhs.var_name + ".valid && (" +
              lhs_val + " != " + rhs_val + ")), true, " + ovf_e + "};");
    return register_value{var_name,
                          sirius::logical_type::make(sirius::type_id::BOOLEAN),
                          cudf::data_type{cudf::type_id::BOOL8},
                          /*is_statically_non_null=*/true,
                          can_ovf};
  }

  std::string op_sym;
  switch (expr.op) {
    case sirius::comparison_type::equal: op_sym = "=="; break;
    case sirius::comparison_type::not_equal: op_sym = "!="; break;
    case sirius::comparison_type::lt: op_sym = "<"; break;
    case sirius::comparison_type::le: op_sym = "<="; break;
    case sirius::comparison_type::gt: op_sym = ">"; break;
    case sirius::comparison_type::ge: op_sym = ">="; break;
    default:
      throw invalid_input_exception("[nvrtc] Unrecognized comparison operator: {}",
                                    static_cast<int>(expr.op));
  }

  bool const non_null = lhs.is_statically_non_null && rhs.is_statically_non_null;
  if (non_null) {
    emit_line("reg_val<bool> " + var_name + "{(" + lhs_val + " " + op_sym + " " + rhs_val +
              "), true, " + ovf_e + "};");
  } else {
    emit_line("bool const " + var_name + "_valid = " + lhs.var_name + ".valid && " + rhs.var_name +
              ".valid;");
    emit_line("reg_val<bool> " + var_name + "{" + var_name + "_valid ? (" + lhs_val + " " + op_sym +
              " " + rhs_val + ") : false, " + var_name + "_valid, " + ovf_e + "};");
  }

  return register_value{var_name,
                        sirius::logical_type::make(sirius::type_id::BOOLEAN),
                        cudf::data_type{cudf::type_id::BOOL8},
                        non_null,
                        can_ovf};
}

register_value translation_context::produce_between(sirius::ast::between const& expr)
{
  auto const narrow_carrier =
    narrow_domain_carrier(*expr.input, {expr.lower.get(), expr.upper.get()});
  if (narrow_carrier) { ++_narrow_domain_comparison_count; }

  auto const in_reg   = produce_narrow_domain_operand(*expr.input, narrow_carrier);
  auto const lo_reg   = produce_narrow_domain_operand(*expr.lower, narrow_carrier);
  auto const hi_reg   = produce_narrow_domain_operand(*expr.upper, narrow_carrier);
  auto const var_name = next_var_name();

  bool const is_dec      = cudf::is_fixed_point(in_reg.device_type);
  std::int32_t const s_i = is_dec ? sql_decimal_scale(in_reg.device_type) : 0;
  std::int32_t const s_l = is_dec ? sql_decimal_scale(lo_reg.device_type) : 0;
  std::int32_t const s_h = is_dec ? sql_decimal_scale(hi_reg.device_type) : 0;

  auto const cmp_type_str =
    (is_dec && (s_i != s_l || s_i != s_h || in_reg.device_type != lo_reg.device_type ||
                in_reg.device_type != hi_reg.device_type))
      ? std::string("__int128_t")
      : cuda_type_name(in_reg.device_type);

  std::string in_val = "static_cast<" + cmp_type_str + ">(" + in_reg.var_name + ".val)";
  std::string lo_val = "static_cast<" + cmp_type_str + ">(" + lo_reg.var_name + ".val)";
  std::string hi_val = "static_cast<" + cmp_type_str + ">(" + hi_reg.var_name + ".val)";
  if (is_dec && (s_i != s_l || s_i != s_h)) {
    auto const s_max = std::max({s_i, s_l, s_h});
    if (s_max > s_i) { in_val = "(" + in_val + " * " + format_int128_pow10(s_max - s_i) + ")"; }
    if (s_max > s_l) { lo_val = "(" + lo_val + " * " + format_int128_pow10(s_max - s_l) + ")"; }
    if (s_max > s_h) { hi_val = "(" + hi_val + " * " + format_int128_pow10(s_max - s_h) + ")"; }
  }

  std::string const lo_op = expr.lower_inclusive ? ">=" : ">";
  std::string const hi_op = expr.upper_inclusive ? "<=" : "<";

  bool const non_null =
    in_reg.is_statically_non_null && lo_reg.is_statically_non_null && hi_reg.is_statically_non_null;
  bool const can_ovf      = in_reg.can_overflow || lo_reg.can_overflow || hi_reg.can_overflow;
  std::string const ovf_e = can_ovf ? (in_reg.var_name + ".overflow || " + lo_reg.var_name +
                                       ".overflow || " + hi_reg.var_name + ".overflow")
                                    : "false";

  if (non_null) {
    emit_line("reg_val<bool> " + var_name + "{((" + in_val + " " + lo_op + " " + lo_val + ") && (" +
              in_val + " " + hi_op + " " + hi_val + ")), true, " + ovf_e + "};");
  } else {
    emit_line("bool const " + var_name + "_valid = " + in_reg.var_name + ".valid && " +
              lo_reg.var_name + ".valid && " + hi_reg.var_name + ".valid;");
    emit_line("reg_val<bool> " + var_name + "{" + var_name + "_valid ? ((" + in_val + " " + lo_op +
              " " + lo_val + ") && (" + in_val + " " + hi_op + " " + hi_val + ")) : false, " +
              var_name + "_valid, " + ovf_e + "};");
  }

  return register_value{var_name,
                        sirius::logical_type::make(sirius::type_id::BOOLEAN),
                        cudf::data_type{cudf::type_id::BOOL8},
                        non_null,
                        can_ovf};
}

register_value translation_context::produce_conjunction(sirius::ast::conjunction const& expr)
{
  auto const var_name = next_var_name();
  auto const first    = produce(*expr.children[0]);
  emit_line("reg_val<bool> " + var_name + " = " + first.var_name + ";");

  bool non_null             = first.is_statically_non_null;
  bool can_ovf              = first.can_overflow;
  bool const is_and         = (expr.op == sirius::ast::conjunction::kind::op_and);
  std::string const fn_name = is_and ? "eval_kleene_and" : "eval_kleene_or";
  std::string const dom_cond =
    is_and ? ("!(" + var_name + ".valid && !" + var_name + ".overflow && !" + var_name + ".val)")
           : ("!(" + var_name + ".valid && !" + var_name + ".overflow && " + var_name + ".val)");

  for (std::size_t i = 1; i < expr.children.size(); ++i) {
    emit_line("if (" + dom_cond + ") {");
    auto const rhs = produce(*expr.children[i]);
    emit_line(var_name + " = " + fn_name + "(" + var_name + ", " + rhs.var_name + ");");
    emit_line("}");
    non_null = non_null && rhs.is_statically_non_null;
    can_ovf  = can_ovf || rhs.can_overflow;
  }

  return register_value{var_name,
                        sirius::logical_type::make(sirius::type_id::BOOLEAN),
                        cudf::data_type{cudf::type_id::BOOL8},
                        non_null,
                        can_ovf};
}

register_value translation_context::produce_case_expr(sirius::ast::case_expr const& expr)
{
  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();

  emit_line("reg_val<" + type_str + "> " + var_name + "{" + type_str + "{}, false, false};");

  bool can_ovf      = false;
  bool all_non_null = true;

  for (std::size_t i = 0; i < expr.cases.size(); ++i) {
    auto const& c      = expr.cases[i];
    auto const when_rv = produce(*c.when_);
    can_ovf            = can_ovf || when_rv.can_overflow;

    emit_line("if (" + when_rv.var_name + ".overflow) {");
    emit_line(var_name + " = reg_val<" + type_str + ">{" + type_str + "{}, true, true};");
    emit_line("} else if (" + when_rv.var_name + ".valid && " + when_rv.var_name + ".val) {");

    if (c.then_->holds<sirius::ast::function_call>() &&
        c.then_->get<sirius::ast::function_call>().function() == sirius::function_id::error) {
      _has_subquery_error = true;
      all_non_null        = false;
      emit_line("row_subquery_error = true;");
    } else {
      auto const then_rv = produce(*c.then_);
      can_ovf            = can_ovf || then_rv.can_overflow;
      all_non_null       = all_non_null && then_rv.is_statically_non_null;
      emit_line(var_name + " = reg_val<" + type_str + ">{static_cast<" + type_str + ">(" +
                then_rv.var_name + ".val), " + then_rv.var_name + ".valid, " + then_rv.var_name +
                ".overflow};");
    }
    emit_line("} else {");
  }

  auto const else_rv = produce(*expr.else_);
  can_ovf            = can_ovf || else_rv.can_overflow;
  all_non_null       = all_non_null && else_rv.is_statically_non_null;
  emit_line(var_name + " = reg_val<" + type_str + ">{static_cast<" + type_str + ">(" +
            else_rv.var_name + ".val), " + else_rv.var_name + ".valid, " + else_rv.var_name +
            ".overflow};");

  for (std::size_t i = 0; i < expr.cases.size(); ++i) {
    emit_line("}");
  }

  return register_value{var_name, expr.return_type(), ret_type, all_non_null, can_ovf};
}

register_value translation_context::produce_coalesce(sirius::ast::coalesce const& expr)
{
  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();

  auto const first = produce(*expr.children[0]);
  emit_line("reg_val<" + type_str + "> " + var_name + "{static_cast<" + type_str + ">(" +
            first.var_name + ".val), " + first.var_name + ".valid, " + first.var_name +
            ".overflow};");

  bool any_non_null = first.is_statically_non_null;
  bool can_ovf      = first.can_overflow;

  for (std::size_t i = 1; i < expr.children.size(); ++i) {
    emit_line("if (!" + var_name + ".valid && !" + var_name + ".overflow) {");
    auto const child = produce(*expr.children[i]);
    emit_line(var_name + " = reg_val<" + type_str + ">{static_cast<" + type_str + ">(" +
              child.var_name + ".val), " + child.var_name + ".valid, " + child.var_name +
              ".overflow};");
    emit_line("}");
    any_non_null = any_non_null || child.is_statically_non_null;
    can_ovf      = can_ovf || child.can_overflow;
  }

  return register_value{var_name, expr.return_type(), ret_type, any_non_null, can_ovf};
}

register_value translation_context::produce_in_list(sirius::ast::in_list const& expr)
{
  auto const probe_reg = produce(*expr.probe);
  auto const var_name  = next_var_name();

  auto const resolve_cmp_type = [](cudf::data_type lt, cudf::data_type rt) -> std::string {
    if (cudf::is_fixed_point(lt) || cudf::is_fixed_point(rt)) {
      if (lt == rt) { return cuda_type_name(lt); }
      return "__int128_t";
    }
    if (lt == rt) { return cuda_type_name(lt); }
    if (lt.id() == cudf::type_id::FLOAT64 || rt.id() == cudf::type_id::FLOAT64) { return "double"; }
    if (lt.id() == cudf::type_id::FLOAT32 || rt.id() == cudf::type_id::FLOAT32) { return "float"; }
    if (cudf::is_unsigned(lt) && cudf::is_unsigned(rt)) { return "uint64_t"; }
    if (!cudf::is_unsigned(lt) && !cudf::is_unsigned(rt)) { return "int64_t"; }
    return "__int128_t";
  };

  emit_line("reg_val<bool> " + var_name + "{false, false, false};");
  bool can_ovf      = probe_reg.can_overflow;
  bool all_non_null = probe_reg.is_statically_non_null;

  for (std::size_t i = 0; i < expr.values.size(); ++i) {
    if (i > 0) {
      emit_line("if (!(" + var_name + ".valid && !" + var_name + ".overflow && " + var_name +
                ".val)) {");
    }
    auto const val_reg    = produce(*expr.values[i]);
    can_ovf               = can_ovf || val_reg.can_overflow;
    all_non_null          = all_non_null && val_reg.is_statically_non_null;
    auto const cmp_type   = resolve_cmp_type(probe_reg.device_type, val_reg.device_type);
    std::string probe_val = "static_cast<" + cmp_type + ">(" + probe_reg.var_name + ".val)";
    std::string rhs_val   = "static_cast<" + cmp_type + ">(" + val_reg.var_name + ".val)";
    if (cudf::is_fixed_point(probe_reg.device_type) || cudf::is_fixed_point(val_reg.device_type)) {
      auto const s_l = sql_decimal_scale(probe_reg.device_type);
      auto const s_r = sql_decimal_scale(val_reg.device_type);
      if (s_l != s_r) {
        auto const s_max = std::max(s_l, s_r);
        if (s_max > s_l) {
          probe_val = "(" + probe_val + " * " + format_int128_pow10(s_max - s_l) + ")";
        }
        if (s_max > s_r) {
          rhs_val = "(" + rhs_val + " * " + format_int128_pow10(s_max - s_r) + ")";
        }
      }
    }
    auto const eq_var = next_var_name();
    bool const eq_ovf = probe_reg.can_overflow || val_reg.can_overflow;
    std::string const ovf_e =
      eq_ovf ? (probe_reg.var_name + ".overflow || " + val_reg.var_name + ".overflow") : "false";

    emit_line("bool const " + eq_var + "_valid = " + probe_reg.var_name + ".valid && " +
              val_reg.var_name + ".valid;");
    emit_line("reg_val<bool> const " + eq_var + "{" + eq_var + "_valid ? (" + probe_val +
              " == " + rhs_val + ") : false, " + eq_var + "_valid, " + ovf_e + "};");
    if (i == 0) {
      emit_line(var_name + " = " + eq_var + ";");
    } else {
      emit_line(var_name + " = eval_kleene_or(" + var_name + ", " + eq_var + ");");
      emit_line("}");
    }
  }

  if (expr.negated) { emit_line(var_name + ".val = !" + var_name + ".val;"); }

  return register_value{var_name,
                        sirius::logical_type::make(sirius::type_id::BOOLEAN),
                        cudf::data_type{cudf::type_id::BOOL8},
                        all_non_null,
                        can_ovf};
}

register_value translation_context::produce_cast(sirius::ast::cast const& expr)
{
  auto const dst_type = sirius::require_cudf_type(expr.target_type);
  auto const dst_str  = cuda_type_name(dst_type);
  auto const var_name = next_var_name();

  // Carrier restoration: if the child is a reference, load it as raw carrier and widen inline.
  register_value child = [&]() {
    if (expr.kind == sirius::ast::cast_kind::carrier_restore &&
        expr.child->holds<sirius::ast::reference>()) {
      auto const& ref = expr.child->get<sirius::ast::reference>();
      sirius::ast::reference raw_ref{ref.column_index, sirius::logical_type{}};
      return produce_reference(raw_ref);
    }
    return produce(*expr.child);
  }();

  if (expr.kind == sirius::ast::cast_kind::carrier_restore) {
    emit_line("reg_val<" + dst_str + "> " + var_name + "{static_cast<" + dst_str + ">(" +
              child.var_name + ".val), " + child.var_name + ".valid, " + child.var_name +
              ".overflow};");
    return register_value{
      var_name, expr.target_type, dst_type, child.is_statically_non_null, child.can_overflow};
  }

  auto const src_type = child.device_type;
  bool can_ovf        = child.can_overflow;

  // 1. Temporal casts
  if (cudf::is_timestamp(src_type) && cudf::is_timestamp(dst_type)) {
    if (src_type.id() == dst_type.id()) {
      emit_line("reg_val<" + dst_str + "> " + var_name + " = " + child.var_name + ";");
    } else if (src_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
      std::int64_t const scale = timestamp_ticks_per_second(dst_type.id()) * 86'400LL;
      emit_line("reg_val<int64_t> " + var_name + " = eval_cast_date_to_timestamp<" +
                std::to_string(scale) + "LL>(" + child.var_name + ");");
      can_ovf = true;
    } else if (dst_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
      std::int64_t const scale = timestamp_ticks_per_second(src_type.id()) * 86'400LL;
      emit_line("reg_val<int32_t> " + var_name + " = eval_cast_timestamp_to_date<" +
                std::to_string(scale) + "LL>(" + child.var_name + ");");
      can_ovf = true;
    } else if (dst_type.id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
      if (src_type.id() == cudf::type_id::TIMESTAMP_NANOSECONDS) {
        emit_line("reg_val<int64_t> " + var_name + " = eval_cast_ts_ns_to_us(" + child.var_name +
                  ");");
      } else {
        std::int64_t const scale =
          (src_type.id() == cudf::type_id::TIMESTAMP_SECONDS) ? 1'000'000LL : 1'000LL;
        emit_line("reg_val<int64_t> " + var_name + " = eval_cast_ts_scale_up<" +
                  std::to_string(scale) + "LL>(" + child.var_name + ");");
        can_ovf = true;
      }
    } else {
      std::int64_t const src_tps = timestamp_ticks_per_second(src_type.id());
      std::int64_t const dst_tps = timestamp_ticks_per_second(dst_type.id());
      if (dst_tps > src_tps) {
        std::int64_t const scale = dst_tps / src_tps;
        emit_line("reg_val<int64_t> " + var_name + " = eval_cast_ts_scale_up<" +
                  std::to_string(scale) + "LL>(" + child.var_name + ");");
        can_ovf = true;
      } else {
        std::int64_t const scale = src_tps / dst_tps;
        emit_line("reg_val<int64_t> " + var_name + " = eval_cast_ts_scale_down<" +
                  std::to_string(scale) + "LL>(" + child.var_name + ");");
      }
    }
  }
  // 2. Any -> BOOL8
  else if (dst_type.id() == cudf::type_id::BOOL8) {
    emit_line("reg_val<bool> " + var_name + " = eval_cast_to_bool(" + child.var_name + ");");
  }
  // 3. BOOL8 -> Any
  else if (src_type.id() == cudf::type_id::BOOL8) {
    if (cudf::is_fixed_point(dst_type)) {
      int const prec       = static_cast<int>(expr.target_type.decimal_precision());
      int const scale      = sql_decimal_scale(dst_type);
      auto const pow10_s   = format_int128_pow10(scale);
      bool const can_fit_1 = (prec > scale);
      emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_bool_to_dec<" + dst_str +
                ", " + (can_fit_1 ? "true" : "false") + ">(" + child.var_name + ", static_cast<" +
                dst_str + ">(" + pow10_s + "));");
      if (!can_fit_1) { can_ovf = true; }
    } else {
      emit_line("reg_val<" + dst_str + "> " + var_name + "{" + child.var_name +
                ".val ? static_cast<" + dst_str + ">(1) : static_cast<" + dst_str + ">(0), " +
                child.var_name + ".valid, " + child.var_name + ".overflow};");
    }
  }
  // 4. DECIMAL -> DECIMAL
  else if (cudf::is_fixed_point(src_type) && cudf::is_fixed_point(dst_type)) {
    int const s_in       = sql_decimal_scale(src_type);
    int const s_out      = sql_decimal_scale(dst_type);
    int const p_out      = static_cast<int>(expr.target_type.decimal_precision());
    auto const limit_str = format_int128_pow10(p_out);
    if (s_out >= s_in) {
      auto const factor_str = format_int128_pow10(s_out - s_in);
      emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_dec_scale_up<" + dst_str +
                ">(" + child.var_name + ", " + factor_str + ", " + limit_str + ");");
    } else {
      auto const factor_str = format_int128_pow10(s_in - s_out);
      emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_dec_scale_down<" + dst_str +
                ">(" + child.var_name + ", " + factor_str + ", " + limit_str + ");");
    }
    can_ovf = true;
  }
  // 5. DECIMAL -> Integer
  else if (cudf::is_fixed_point(src_type) && cudf::is_integral(dst_type)) {
    int const s_in        = sql_decimal_scale(src_type);
    auto const factor_str = format_int128_pow10(s_in);
    emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_dec_to_int<" + dst_str + ">(" +
              child.var_name + ", " + factor_str + ");");
    can_ovf = true;
  }
  // 6. DECIMAL -> Float/Double
  else if (cudf::is_fixed_point(src_type) && cudf::is_floating_point(dst_type)) {
    int const s_in       = sql_decimal_scale(src_type);
    auto const scale_dbl = format_double_pow10(s_in);
    emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_dec_to_float<" + dst_str +
              ">(" + child.var_name + ", " + scale_dbl + ");");
  }
  // 7. Integer -> DECIMAL
  else if (cudf::is_integral(src_type) && cudf::is_fixed_point(dst_type)) {
    int const p_out          = static_cast<int>(expr.target_type.decimal_precision());
    int const s_out          = sql_decimal_scale(dst_type);
    auto const int_limit_str = format_int128_pow10(p_out - s_out);
    auto const factor_str    = format_int128_pow10(s_out);
    emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_int_to_dec<" + dst_str + ">(" +
              child.var_name + ", " + int_limit_str + ", " + factor_str + ");");
    can_ovf = true;
  }
  // 8. Float/Double -> DECIMAL
  else if (cudf::is_floating_point(src_type) && cudf::is_fixed_point(dst_type)) {
    int const p_out          = static_cast<int>(expr.target_type.decimal_precision());
    int const s_out          = sql_decimal_scale(dst_type);
    auto const scale_dbl_str = format_double_pow10(s_out);
    auto const limit_dbl_str = format_double_pow10(p_out);
    auto const limit_128_str = format_int128_pow10(p_out);
    emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_float_to_dec<" + dst_str +
              ">(" + child.var_name + ", " + scale_dbl_str + ", " + limit_dbl_str + ", " +
              limit_128_str + ");");
    can_ovf = true;
  }
  // 9. Float/Double -> Integer
  else if (cudf::is_floating_point(src_type) && cudf::is_integral(dst_type)) {
    emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_float_to_int<" + dst_str +
              ">(" + child.var_name + ");");
    can_ovf = true;
  }
  // 10. Integer -> Float/Double
  else if (cudf::is_integral(src_type) && cudf::is_floating_point(dst_type)) {
    emit_line("reg_val<" + dst_str + "> " + var_name + "{static_cast<" + dst_str + ">(" +
              child.var_name + ".val), " + child.var_name + ".valid, " + child.var_name +
              ".overflow};");
  }
  // 11. Float/Double -> Float/Double
  else if (cudf::is_floating_point(src_type) && cudf::is_floating_point(dst_type)) {
    emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_float_to_float<" + dst_str +
              ">(" + child.var_name + ");");
    if (src_type.id() == cudf::type_id::FLOAT64 && dst_type.id() == cudf::type_id::FLOAT32) {
      can_ovf = true;
    }
  }
  // 12. Integer -> Integer
  else if (cudf::is_integral(src_type) && cudf::is_integral(dst_type)) {
    emit_line("reg_val<" + dst_str + "> " + var_name + " = eval_cast_int_to_int<" + dst_str + ">(" +
              child.var_name + ");");
    can_ovf = true;
  } else {
    throw not_implemented_exception("[nvrtc] Unsupported cast from {} to {}",
                                    cudf::type_to_name(src_type),
                                    cudf::type_to_name(dst_type));
  }

  bool non_null = child.is_statically_non_null;
  if (expr.try_cast && can_ovf) {
    emit_line("if (" + var_name + ".overflow) { " + var_name + " = reg_val<" + dst_str + ">{" +
              dst_str + "{}, false, false}; }");
    can_ovf  = false;
    non_null = false;
  }

  return register_value{var_name, expr.target_type, dst_type, non_null, can_ovf};
}

register_value translation_context::produce_unary_op(sirius::ast::unary_op const& expr)
{
  auto const child    = produce(*expr.child);
  auto const var_name = next_var_name();

  switch (expr.op) {
    case sirius::ast::unary_op::kind::op_not: {
      emit_line("reg_val<bool> " + var_name + "{!" + child.var_name + ".val, " + child.var_name +
                ".valid, " + child.var_name + ".overflow};");
      return register_value{var_name,
                            sirius::logical_type::make(sirius::type_id::BOOLEAN),
                            cudf::data_type{cudf::type_id::BOOL8},
                            child.is_statically_non_null,
                            child.can_overflow};
    }
    case sirius::ast::unary_op::kind::op_is_null: {
      emit_line("reg_val<bool> " + var_name + "{!" + child.var_name + ".valid, true, " +
                child.var_name + ".overflow};");
      return register_value{var_name,
                            sirius::logical_type::make(sirius::type_id::BOOLEAN),
                            cudf::data_type{cudf::type_id::BOOL8},
                            /*is_statically_non_null=*/true,
                            child.can_overflow};
    }
    case sirius::ast::unary_op::kind::op_is_not_null: {
      emit_line("reg_val<bool> " + var_name + "{" + child.var_name + ".valid, true, " +
                child.var_name + ".overflow};");
      return register_value{var_name,
                            sirius::logical_type::make(sirius::type_id::BOOLEAN),
                            cudf::data_type{cudf::type_id::BOOL8},
                            /*is_statically_non_null=*/true,
                            child.can_overflow};
    }
    case sirius::ast::unary_op::kind::op_try: {
      auto const type_str = cuda_type_name(child.device_type);
      emit_line("reg_val<" + type_str + "> " + var_name + "{" + child.var_name + ".overflow ? " +
                type_str + "{} : " + child.var_name + ".val, " + child.var_name + ".valid && !" +
                child.var_name + ".overflow, false};");
      return register_value{var_name,
                            expr.return_type(),
                            child.device_type,
                            /*is_statically_non_null=*/false,
                            /*can_overflow=*/false};
    }
    default:
      throw not_implemented_exception("[nvrtc] Unsupported unary_op kind: {}",
                                      static_cast<int>(expr.op));
  }
}

register_value translation_context::produce_function_call(sirius::ast::function_call const& expr)
{
  auto const fid      = expr.function();
  auto const& args    = expr.arguments();
  auto const ret_type = sirius::require_cudf_type(expr.return_type());
  auto const type_str = cuda_type_name(ret_type);
  auto const var_name = next_var_name();

  if (fid == sirius::function_id::add || fid == sirius::function_id::sub ||
      fid == sirius::function_id::mul || fid == sirius::function_id::div ||
      fid == sirius::function_id::int_div || fid == sirius::function_id::mod) {
    auto const lhs = produce(*args[0]);
    auto const rhs = produce(*args[1]);

    if (cudf::is_fixed_point(ret_type)) {
      int const s_l = sql_decimal_scale(lhs.device_type);
      int const s_r = sql_decimal_scale(rhs.device_type);
      if (fid == sirius::function_id::add || fid == sirius::function_id::sub) {
        int const s_max      = std::max(s_l, s_r);
        auto const f_l       = format_int128_pow10(s_max - s_l);
        auto const f_r       = format_int128_pow10(s_max - s_r);
        std::string const fn = (fid == sirius::function_id::add) ? "eval_dec_add" : "eval_dec_sub";
        emit_line("reg_val<" + type_str + "> " + var_name + " = " + fn + "<" + type_str + ">(" +
                  lhs.var_name + ", " + rhs.var_name + ", " + f_l + ", " + f_r + ");");
      } else if (fid == sirius::function_id::mul) {
        emit_line("reg_val<" + type_str + "> " + var_name + " = eval_dec_mul<" + type_str + ">(" +
                  lhs.var_name + ", " + rhs.var_name + ");");
      } else if (fid == sirius::function_id::div || fid == sirius::function_id::int_div) {
        int const s_out     = sql_decimal_scale(ret_type);
        int const s_diff    = s_out - (s_l - s_r);
        auto const f_out    = format_int128_pow10(std::abs(s_diff));
        bool const scale_up = (s_diff >= 0);
        emit_line("reg_val<" + type_str + "> " + var_name + " = eval_dec_div<" + type_str + ", " +
                  (scale_up ? "true" : "false") + ">(" + lhs.var_name + ", " + rhs.var_name + ", " +
                  f_out + ");");
      } else {
        int const s_max = std::max(s_l, s_r);
        auto const f_l  = format_int128_pow10(s_max - s_l);
        auto const f_r  = format_int128_pow10(s_max - s_r);
        emit_line("reg_val<" + type_str + "> " + var_name + " = eval_dec_mod<" + type_str + ">(" +
                  lhs.var_name + ", " + rhs.var_name + ", " + f_l + ", " + f_r + ");");
      }
      bool const is_div_or_mod =
        (fid == sirius::function_id::div || fid == sirius::function_id::int_div ||
         fid == sirius::function_id::mod);
      bool const non_null =
        lhs.is_statically_non_null && rhs.is_statically_non_null && !is_div_or_mod;
      bool const can_ovf = lhs.can_overflow || rhs.can_overflow || true;
      return register_value{var_name, expr.return_type(), ret_type, non_null, can_ovf};
    }

    std::string fn_name;
    switch (fid) {
      case sirius::function_id::add: fn_name = "eval_add"; break;
      case sirius::function_id::sub: fn_name = "eval_sub"; break;
      case sirius::function_id::mul: fn_name = "eval_mul"; break;
      case sirius::function_id::div:
      case sirius::function_id::int_div: fn_name = "eval_div"; break;
      case sirius::function_id::mod: fn_name = "eval_mod"; break;
      default: break;
    }

    emit_line("reg_val<" + type_str + "> " + var_name + " = " + fn_name + "<" + type_str + ">(" +
              lhs.var_name + ", " + rhs.var_name + ");");

    bool const is_div_or_mod =
      (fid == sirius::function_id::div || fid == sirius::function_id::int_div ||
       fid == sirius::function_id::mod);
    bool const is_int_type = cudf::is_integral(ret_type);
    bool const can_ovf =
      lhs.can_overflow || rhs.can_overflow ||
      (is_int_type && (fid == sirius::function_id::add || fid == sirius::function_id::sub ||
                       fid == sirius::function_id::mul ||
                       ((fid == sirius::function_id::div || fid == sirius::function_id::int_div) &&
                        !cudf::is_unsigned(ret_type))));
    bool const non_null =
      lhs.is_statically_non_null && rhs.is_statically_non_null && !(is_div_or_mod && is_int_type);

    return register_value{var_name, expr.return_type(), ret_type, non_null, can_ovf};
  }

  if (fid == sirius::function_id::abs) {
    auto const child = produce(*args[0]);
    emit_line("reg_val<" + type_str + "> " + var_name + " = eval_abs<" + type_str + ">(" +
              child.var_name + ");");
    bool const can_ovf = child.can_overflow || cudf::is_fixed_point(ret_type) ||
                         (cudf::is_integral(ret_type) && !cudf::is_unsigned(ret_type));
    return register_value{
      var_name, expr.return_type(), ret_type, child.is_statically_non_null, can_ovf};
  }

  if (fid == sirius::function_id::round) {
    auto const child       = produce(*args[0]);
    std::int32_t precision = 0;
    if (args.size() == 2) {
      precision = std::get<std::int32_t>(args[1]->get<sirius::ast::constant>().payload);
    }
    bool const neg_prec = precision < 0;
    double const modifier =
      (ret_type.id() == cudf::type_id::FLOAT32)
        ? static_cast<double>(std::pow(
            10, static_cast<float>(neg_prec ? -static_cast<std::int64_t>(precision) : precision)))
        : std::pow(
            10, static_cast<double>(neg_prec ? -static_cast<std::int64_t>(precision) : precision));

    sirius::ast::constant mod_cst{sirius::value{modifier},
                                  sirius::logical_type::make(sirius::type_id::DOUBLE)};
    auto const mod_reg = produce_constant(mod_cst);
    emit_line("reg_val<" + type_str + "> " + var_name + " = eval_round<" + type_str + ", " +
              (neg_prec ? "true" : "false") + ">(" + child.var_name + ", " + mod_reg.var_name +
              ".val);");
    return register_value{
      var_name, expr.return_type(), ret_type, child.is_statically_non_null, child.can_overflow};
  }

  if (fid == sirius::function_id::constant_or_null) {
    auto const val_reg = produce(*args[0]);
    std::vector<register_value> arg_regs;
    arg_regs.reserve(args.size() - 1);
    for (std::size_t i = 1; i < args.size(); ++i) {
      arg_regs.push_back(produce(*args[i]));
    }
    std::string valid_expr = val_reg.var_name + ".valid";
    std::string ovf_expr   = val_reg.var_name + ".overflow";
    bool any_ovf           = val_reg.can_overflow;
    for (auto const& r : arg_regs) {
      valid_expr += " && " + r.var_name + ".valid";
      ovf_expr += " || " + r.var_name + ".overflow";
      if (r.can_overflow) { any_ovf = true; }
    }
    emit_line("bool const " + var_name + "_valid = " + valid_expr + ";");
    emit_line("reg_val<" + type_str + "> " + var_name + "{" + var_name + "_valid ? static_cast<" +
              type_str + ">(" + val_reg.var_name + ".val) : " + type_str + "{}, " + var_name +
              "_valid, " + (any_ovf ? ("(" + ovf_expr + ")") : "false") + "};");
    return register_value{
      var_name, expr.return_type(), ret_type, /*is_statically_non_null=*/false, any_ovf};
  }

  if (fid == sirius::function_id::year || fid == sirius::function_id::month ||
      fid == sirius::function_id::day || fid == sirius::function_id::hour ||
      fid == sirius::function_id::minute || fid == sirius::function_id::second ||
      fid == sirius::function_id::millisecond || fid == sirius::function_id::microsecond) {
    auto const child = produce(*args[0]);
    auto const ct_id = child.device_type.id();
    int comp_id      = 0;
    switch (fid) {
      case sirius::function_id::year: comp_id = 0; break;
      case sirius::function_id::month: comp_id = 1; break;
      case sirius::function_id::day: comp_id = 2; break;
      case sirius::function_id::hour: comp_id = 3; break;
      case sirius::function_id::minute: comp_id = 4; break;
      case sirius::function_id::second: comp_id = 5; break;
      case sirius::function_id::millisecond: comp_id = 6; break;
      case sirius::function_id::microsecond: comp_id = 7; break;
      default: break;
    }
    bool const is_subsec =
      (fid == sirius::function_id::millisecond || fid == sirius::function_id::microsecond);
    if (ct_id == cudf::type_id::TIMESTAMP_DAYS) {
      emit_line("reg_val<" + type_str + "> " + var_name + " = eval_extract_date<" + type_str +
                ", " + std::to_string(comp_id) + ">(" + child.var_name + ");");
    } else {
      std::int64_t const tps = timestamp_ticks_per_second(ct_id);
      emit_line("reg_val<" + type_str + "> " + var_name + " = eval_extract_timestamp<" + type_str +
                ", " + std::to_string(tps) + "LL, " + std::to_string(comp_id) + ">(" +
                child.var_name + ");");
    }
    bool const non_null = child.is_statically_non_null && !is_subsec;
    return register_value{var_name, expr.return_type(), ret_type, non_null, child.can_overflow};
  }

  if (fid == sirius::function_id::date_trunc) {
    auto const& freq_str = std::get<std::string>(args[0]->get<sirius::ast::constant>().payload);
    auto const unit      = sirius::parse_gpu_date_trunc_unit(freq_str);
    if (!unit) {
      throw invalid_input_exception(
        "[expression_evaluator:function] unrecognized/unsupported date_trunc frequency: {}",
        freq_str);
    }
    auto const child      = produce(*args[1]);
    auto const src_type   = child.device_type;
    std::string trunc_var = child.var_name;
    if (src_type.id() != cudf::type_id::TIMESTAMP_DAYS) {
      std::int64_t const div = date_trunc_divisor(src_type.id(), *unit);
      if (div > 1LL) {
        trunc_var = next_var_name();
        emit_line("reg_val<int64_t> " + trunc_var + " = eval_date_trunc<" + std::to_string(div) +
                  "LL>(" + child.var_name + ");");
      }
    }
    bool can_ovf = child.can_overflow;
    if (ret_type == src_type) {
      emit_line("reg_val<" + type_str + "> " + var_name + " = " + trunc_var + ";");
    } else if (src_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
      // Reconcile TIMESTAMP_DAYS to wider timestamp return type via cast_through_rep semantics
      emit_line("reg_val<" + type_str + "> " + var_name + " = eval_cast_ts_scale_up<" +
                std::to_string(timestamp_ticks_per_second(ret_type.id()) * 86'400LL) +
                "LL>(reg_val<int64_t>{static_cast<int64_t>(" + trunc_var + ".val), " + trunc_var +
                ".valid, " + trunc_var + ".overflow});");
      can_ovf = true;
    } else if (ret_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
      std::int64_t const scale = timestamp_ticks_per_second(src_type.id()) * 86'400LL;
      emit_line("reg_val<int32_t> " + var_name + " = eval_cast_timestamp_to_date<" +
                std::to_string(scale) + "LL>(" + trunc_var + ");");
      can_ovf = true;
    } else {
      std::int64_t const src_tps = timestamp_ticks_per_second(src_type.id());
      std::int64_t const dst_tps = timestamp_ticks_per_second(ret_type.id());
      if (dst_tps > src_tps) {
        emit_line("reg_val<int64_t> " + var_name + " = eval_cast_ts_scale_up<" +
                  std::to_string(dst_tps / src_tps) + "LL>(" + trunc_var + ");");
        can_ovf = true;
      } else {
        emit_line("reg_val<int64_t> " + var_name + " = eval_cast_ts_scale_down<" +
                  std::to_string(src_tps / dst_tps) + "LL>(" + trunc_var + ");");
      }
    }
    return register_value{
      var_name, expr.return_type(), ret_type, child.is_statically_non_null, can_ovf};
  }

  if (fid == sirius::function_id::strlen || fid == sirius::function_id::length) {
    auto const child = produce(*args[0]);
    std::string const fn =
      (fid == sirius::function_id::strlen) ? "eval_strlen" : "eval_utf8_length";
    emit_line("reg_val<" + type_str + "> " + var_name + " = " + fn + "<" + type_str + ">(" +
              child.var_name + ");");
    return register_value{
      var_name, expr.return_type(), ret_type, child.is_statically_non_null, child.can_overflow};
  }

  if (fid == sirius::function_id::prefix || fid == sirius::function_id::suffix ||
      fid == sirius::function_id::contains || fid == sirius::function_id::like ||
      fid == sirius::function_id::not_like) {
    auto const lhs = produce(*args[0]);
    if (fid != sirius::function_id::prefix && fid != sirius::function_id::suffix) {
      if (auto const pattern_id = try_bind_like_pattern(*args[1], fid)) {
        bool const invert = (fid == sirius::function_id::not_like);
        emit_line("reg_val<bool> " + var_name + " = eval_like_multiliteral<" +
                  (invert ? "true" : "false") + ">(" + lhs.var_name + ", lp_" +
                  std::to_string(*pattern_id) + ");");
        return register_value{
          var_name, expr.return_type(), ret_type, lhs.is_statically_non_null, lhs.can_overflow};
      }
    }
    auto const rhs = produce(*args[1]);
    std::string fn;
    switch (fid) {
      case sirius::function_id::prefix: fn = "eval_prefix"; break;
      case sirius::function_id::suffix: fn = "eval_suffix"; break;
      case sirius::function_id::contains: fn = "eval_contains"; break;
      case sirius::function_id::like: fn = "eval_like<false>"; break;
      case sirius::function_id::not_like: fn = "eval_like<true>"; break;
      default: break;
    }
    emit_line("reg_val<bool> " + var_name + " = " + fn + "(" + lhs.var_name + ", " + rhs.var_name +
              ");");
    return register_value{var_name,
                          expr.return_type(),
                          ret_type,
                          lhs.is_statically_non_null && rhs.is_statically_non_null,
                          lhs.can_overflow || rhs.can_overflow};
  }

  if (fid == sirius::function_id::substring) {
    auto const opt_offset     = extract_constant_int64(*args[1]);
    std::int64_t const offset = opt_offset.value_or(0);
    std::int64_t const length =
      (args.size() == 3)
        ? extract_constant_int64(*args[2]).value_or(sirius::substring_default_length)
        : sirius::substring_default_length;
    auto const slice = sirius::gpu_substring_slice(offset, length);
    if (!slice) {
      throw invalid_input_exception(
        "[expression_evaluator:function] unsupported substring bounds (offset {}, length {})",
        offset,
        length);
    }
    auto const child          = produce(*args[0]);
    std::int32_t const start  = slice->start.value_or(0);
    std::int32_t const stop   = slice->stop.value_or(std::numeric_limits<std::int32_t>::max());
    auto const int32_sql_type = sirius::logical_type::make(sirius::type_id::INTEGER);
    auto const start_reg =
      produce_constant(sirius::ast::constant{sirius::value{start}, int32_sql_type});
    auto const stop_reg =
      produce_constant(sirius::ast::constant{sirius::value{stop}, int32_sql_type});
    emit_line("reg_val<str_view> " + var_name + " = eval_substring(" + child.var_name + ", " +
              start_reg.var_name + ".val, " + stop_reg.var_name + ".val);");
    return register_value{
      var_name, expr.return_type(), ret_type, child.is_statically_non_null, child.can_overflow};
  }

  throw not_implemented_exception("[nvrtc] Unsupported function_id in translation_context: {}",
                                  static_cast<int>(fid));
}

std::optional<std::size_t> translation_context::try_bind_like_pattern(
  sirius::ast::node const& pattern, sirius::function_id fid)
{
  if (!pattern.holds<sirius::ast::constant>()) { return std::nullopt; }
  auto const& cst = pattern.get<sirius::ast::constant>();
  if (!std::holds_alternative<std::string>(cst.payload)) { return std::nullopt; }
  auto const& text = std::get<std::string>(cst.payload);
  // `contains(s, lit)` is the one-literal `%lit%` pattern; `%` and `_` are plain bytes there.
  std::optional<sirius::like_multiliteral_pattern> parsed;
  if (fid == sirius::function_id::contains) {
    if (text.empty() || text.find_first_of("%_") != std::string::npos) { return std::nullopt; }
    parsed = sirius::classify_like_multiliteral("%" + text + "%", std::string_view{});
  } else {
    parsed = sirius::classify_like_multiliteral(text, std::string_view{});
  }
  if (!parsed) { return std::nullopt; }
  auto const binding_id = _like_patterns.size();
  _like_patterns.push_back(like_pattern_binding{binding_id, parsed->descriptor()});
  return binding_id;
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

void collect_materialization_boundaries(std::span<sirius::ast::node const* const> expressions,
                                        cudf::table_view const& input,
                                        std::vector<sirius::ast::node const*>& out_boundaries)
{
  for (auto const* expr : expressions) {
    if (expr == nullptr) { continue; }
    // Only descend into root expressions whose children we might compile around a boundary; do not
    // treat a top-level Materialization Boundary itself as an in-tree synthetic input column.
    if (is_in_tree_materialization_boundary(*expr, input)) { continue; }
    collect_boundaries_from_node(*expr, input, out_boundaries);
  }
}

bool can_compile_project_sink(std::span<sirius::ast::node const* const> expressions,
                              cudf::table_view const& input,
                              boundary_column_map const* boundary_map) noexcept
{
  if (expressions.empty()) { return false; }
  for (auto const* expr : expressions) {
    if (expr == nullptr || !is_node_supported(*expr, input, boundary_map)) { return false; }
    if (!is_supported_fixed_width(effective_node_cudf_type(*expr, input, boundary_map))) {
      return false;
    }
  }
  return true;
}

namespace {

void emit_nvrtc_device_runtime_preamble_impl(std::ostringstream& src)
{
  src
    << "#include \"codegen/stdint_shim.hpp\"\n\n"
    // The preamble is deliberately self-contained (no CCCL includes): parsing <cuda/std/chrono>,
    // <cuda/numeric> and friends dominated every kernel compile (~0.7 s) and even a precompiled
    // header of them took ~0.25 s to load. The handful of traits, limits, checked-arithmetic and
    // civil-calendar helpers the generated code needs are defined here instead.
    << "template <typename T> struct limits;\n"
    << "template <> struct limits<bool> { static constexpr bool lowest() { return false; } static "
       "constexpr bool max() { return true; } };\n"
    << "template <> struct limits<int8_t> { static constexpr int8_t lowest() { return -128; } "
       "static constexpr int8_t max() { return 127; } };\n"
    << "template <> struct limits<int16_t> { static constexpr int16_t lowest() { return -32768; } "
       "static constexpr int16_t max() { return 32767; } };\n"
    << "template <> struct limits<int32_t> { static constexpr int32_t lowest() { return "
       "-2147483647 - 1; } static constexpr int32_t max() { return 2147483647; } };\n"
    << "template <> struct limits<int64_t> { static constexpr int64_t lowest() { return "
       "-9223372036854775807LL - 1; } static constexpr int64_t max() { return "
       "9223372036854775807LL; } };\n"
    << "template <> struct limits<uint8_t> { static constexpr uint8_t lowest() { return 0; } "
       "static "
       "constexpr uint8_t max() { return 255; } };\n"
    << "template <> struct limits<uint16_t> { static constexpr uint16_t lowest() { return 0; } "
       "static constexpr uint16_t max() { return 65535; } };\n"
    << "template <> struct limits<uint32_t> { static constexpr uint32_t lowest() { return 0; } "
       "static constexpr uint32_t max() { return 4294967295u; } };\n"
    << "template <> struct limits<uint64_t> { static constexpr uint64_t lowest() { return 0; } "
       "static constexpr uint64_t max() { return 18446744073709551615ULL; } };\n"
    << "template <> struct limits<__int128_t> {\n"
    << "  static constexpr __int128_t max() { return static_cast<__int128_t>((static_cast<unsigned "
       "__int128>(1) << 127) - 1); }\n"
    << "  static constexpr __int128_t lowest() { return -max() - 1; }\n"
    << "};\n"
    << "template <> struct limits<unsigned __int128> {\n"
    << "  static constexpr unsigned __int128 lowest() { return 0; }\n"
    << "  static constexpr unsigned __int128 max() { return ~static_cast<unsigned __int128>(0); }\n"
    << "};\n"
    << "template <> struct limits<float> { static constexpr float lowest() { return "
       "-3.402823466e+38f; } static constexpr float max() { return 3.402823466e+38f; } };\n"
    << "template <> struct limits<double> { static constexpr double lowest() { return "
       "-1.7976931348623157e+308; } static constexpr double max() { return "
       "1.7976931348623157e+308; } };\n\n"
    << "template <typename T> inline constexpr bool is_floating_point_v = false;\n"
    << "template <> inline constexpr bool is_floating_point_v<float> = true;\n"
    << "template <> inline constexpr bool is_floating_point_v<double> = true;\n"
    << "template <typename T> inline constexpr bool is_signed_v = static_cast<T>(-1) < "
       "static_cast<T>(0);\n"
    << "template <bool B, typename T, typename F> struct conditional { using type = T; };\n"
    << "template <typename T, typename F> struct conditional<false, T, F> { using type = F; };\n"
    << "template <bool B, typename T, typename F> using conditional_t = typename conditional<B, T, "
       "F>::type;\n\n"
    // in_range<Dst>(v): does the integer v fit Dst? Comparisons run in a 128-bit type of the
    // operands' signedness so no implicit conversion changes a value.
    << "template <typename Dst, typename Src>\n"
    << "__device__ __forceinline__ constexpr bool in_range(Src v) {\n"
    << "  if constexpr (is_signed_v<Dst> == is_signed_v<Src>) {\n"
    << "    using W = conditional_t<is_signed_v<Dst>, __int128_t, unsigned __int128>;\n"
    << "    return static_cast<W>(v) >= static_cast<W>(limits<Dst>::lowest()) && static_cast<W>(v) "
       "<= static_cast<W>(limits<Dst>::max());\n"
    << "  } else if constexpr (is_signed_v<Dst>) {\n"
    << "    return static_cast<unsigned __int128>(v) <= static_cast<unsigned "
       "__int128>(limits<Dst>::max());\n"
    << "  } else {\n"
    << "    return v >= Src{0} && static_cast<unsigned __int128>(v) <= static_cast<unsigned "
       "__int128>(limits<Dst>::max());\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T> struct ovf_result { T value; bool overflow; };\n"
    << "template <typename Dst>\n"
    << "__device__ __forceinline__ ovf_result<Dst> overflow_cast(__int128_t v) {\n"
    << "  bool const fits = in_range<Dst>(v);\n"
    << "  return ovf_result<Dst>{fits ? static_cast<Dst>(v) : Dst{}, !fits};\n"
    << "}\n"
    << "__device__ __forceinline__ ovf_result<__int128_t> add_overflow128(__int128_t a, __int128_t "
       "b) {\n"
    << "  __int128_t const r = static_cast<__int128_t>(static_cast<unsigned __int128>(a) + "
       "static_cast<unsigned __int128>(b));\n"
    << "  return ovf_result<__int128_t>{r, ((a ^ r) & (b ^ r)) < 0};\n"
    << "}\n"
    << "__device__ __forceinline__ ovf_result<__int128_t> sub_overflow128(__int128_t a, __int128_t "
       "b) {\n"
    << "  __int128_t const r = static_cast<__int128_t>(static_cast<unsigned __int128>(a) - "
       "static_cast<unsigned __int128>(b));\n"
    << "  return ovf_result<__int128_t>{r, ((a ^ b) & (a ^ r)) < 0};\n"
    << "}\n"
    // Signed 128 x 128 -> 128 with overflow detection via 64-bit limbs of the magnitudes: the
    // product fits iff at most one operand has a high limb, the cross product fits 64 bits, and
    // the assembled magnitude does not exceed 2^127 (negative) or 2^127 - 1 (positive).
    << "__device__ __forceinline__ ovf_result<__int128_t> mul_overflow128(__int128_t a, __int128_t "
       "b) {\n"
    << "  using u128 = unsigned __int128;\n"
    << "  bool const negative = (a < 0) != (b < 0);\n"
    << "  u128 const ua = a < 0 ? u128{0} - static_cast<u128>(a) : static_cast<u128>(a);\n"
    << "  u128 const ub = b < 0 ? u128{0} - static_cast<u128>(b) : static_cast<u128>(b);\n"
    << "  uint64_t const a_lo = static_cast<uint64_t>(ua), a_hi = static_cast<uint64_t>(ua >> "
       "64);\n"
    << "  uint64_t const b_lo = static_cast<uint64_t>(ub), b_hi = static_cast<uint64_t>(ub >> "
       "64);\n"
    << "  if (a_hi != 0u && b_hi != 0u) { return ovf_result<__int128_t>{0, true}; }\n"
    << "  u128 const cross = (a_hi != 0u) ? static_cast<u128>(a_hi) * b_lo : "
       "static_cast<u128>(b_hi) "
       "* a_lo;\n"
    << "  if (static_cast<uint64_t>(cross >> 64) != 0u) { return ovf_result<__int128_t>{0, true}; "
       "}\n"
    << "  u128 const lo = static_cast<u128>(a_lo) * b_lo;\n"
    << "  u128 const hi_sum = (lo >> 64) + cross;\n"
    << "  if (static_cast<uint64_t>(hi_sum >> 64) != 0u) { return ovf_result<__int128_t>{0, true}; "
       "}\n"
    << "  u128 const mag = (hi_sum << 64) | static_cast<uint64_t>(lo);\n"
    << "  u128 const limit = (u128{1} << 127) - (negative ? u128{0} : u128{1});\n"
    << "  if (mag > limit) { return ovf_result<__int128_t>{0, true}; }\n"
    << "  return ovf_result<__int128_t>{static_cast<__int128_t>(negative ? u128{0} - mag : mag), "
       "false};\n"
    << "}\n"
    // Same-width checked product for the decimal representation types.
    << "template <typename T>\n"
    << "__device__ __forceinline__ ovf_result<T> mul_overflow(T a, T b) {\n"
    << "  if constexpr (sizeof(T) == 16) {\n"
    << "    return mul_overflow128(a, b);\n"
    << "  } else if constexpr (sizeof(T) == 8) {\n"
    << "    T const lo = static_cast<T>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));\n"
    << "    bool const ovf = __mul64hi(a, b) != (lo >> 63);\n"
    << "    return ovf_result<T>{ovf ? T{} : lo, ovf};\n"
    << "  } else {\n"
    << "    int64_t const r = static_cast<int64_t>(a) * static_cast<int64_t>(b);\n"
    << "    bool const fits = in_range<T>(r);\n"
    << "    return ovf_result<T>{fits ? static_cast<T>(r) : T{}, !fits};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T>\n"
    << "struct reg_val {\n"
    << "  T val;\n"
    << "  bool valid;\n"
    << "  bool overflow;\n"
    << "};\n\n"
    << "struct str_view {\n"
    << "  char const* ptr;\n"
    << "  int32_t len;\n"
    // End of the readable allocation the string lives in; word-at-a-time matchers may read the
    // aligned words that overlap [ptr, ptr + len) only up to this bound.
    << "  char const* limit;\n"
    << "};\n\n"
    // Mirrors sirius::detail::like_multiliteral_{literal,pattern}_desc (host layout asserted).
    << "struct like_literal_desc {\n"
    << "  uint64_t chunk_val[4];\n"
    << "  uint64_t chunk_mask[4];\n"
    << "  uint64_t b0_bcast;\n"
    << "  uint64_t b1_bcast;\n"
    << "  int32_t len;\n"
    << "  int32_t nchunks;\n"
    << "};\n"
    << "struct like_pattern_desc {\n"
    << "  like_literal_desc literals[4];\n"
    << "  int32_t n;\n"
    << "  int32_t total_len;\n"
    << "};\n\n"
    << "__device__ __forceinline__ uint64_t swar_byte_eq_mask(uint64_t x, uint64_t bcast) {\n"
    << "  uint64_t const v = x ^ bcast;\n"
    << "  return (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;\n"
    << "}\n\n"
    // Load the 8-byte aligned word at `addr` without reading at or past `limit`.
    << "__device__ __forceinline__ uint64_t load_bounded_word(uintptr_t addr, uintptr_t limit) {\n"
    << "  if (addr + 8u <= limit) { return *reinterpret_cast<uint64_t const*>(addr); }\n"
    << "  uint64_t value = 0ULL;\n"
    << "  for (uintptr_t b = addr; b < limit; ++b) {\n"
    << "    value |= static_cast<uint64_t>(*reinterpret_cast<uint8_t const*>(b)) << ((b - addr) "
       "<< 3);\n"
    << "  }\n"
    << "  return value;\n"
    << "}\n\n"
    // Unaligned 8-byte window at byte address `p` assembled from two bounded aligned words.
    << "__device__ __forceinline__ uint64_t read_window(uintptr_t p, uintptr_t limit) {\n"
    << "  uintptr_t const a = p & ~static_cast<uintptr_t>(7u);\n"
    << "  int const bit_off = static_cast<int>(p & 7u) << 3;\n"
    << "  uint64_t const lo = load_bounded_word(a, limit);\n"
    << "  if (bit_off == 0) { return lo; }\n"
    << "  uint64_t const hi = load_bounded_word(a + 8u, limit);\n"
    << "  return (lo >> bit_off) | (hi << (64 - bit_off));\n"
    << "}\n\n"
    // Superset of the byte lanes where the literal's first byte sits at lane k and (len >= 2)
    // its second byte follows; bit 7 of each byte marks a candidate (SWAR false positives are
    // filtered by verification).
    << "__device__ __forceinline__ uint64_t like_candidates(uint64_t cur, uint64_t next, uint64_t "
       "b0, uint64_t b1, int32_t len) {\n"
    << "  uint64_t const c1 = swar_byte_eq_mask(cur, b0);\n"
    << "  if (len == 1) { return c1; }\n"
    << "  uint64_t const c2 = swar_byte_eq_mask(cur, b1);\n"
    << "  uint64_t const c2n = swar_byte_eq_mask(next, b1);\n"
    << "  return c1 & ((c2 >> 8) | (c2n << 56));\n"
    << "}\n\n"
    // Literal fields are selected with a switch so the kernel parameter is never indexed
    // dynamically (which would force a local-memory copy of the whole descriptor).
    << "__device__ __forceinline__ like_literal_desc const& like_literal_at(like_pattern_desc "
       "const& pat, int32_t li) {\n"
    << "  switch (li) {\n"
    << "    case 0: return pat.literals[0];\n"
    << "    case 1: return pat.literals[1];\n"
    << "    case 2: return pat.literals[2];\n"
    << "    default: return pat.literals[3];\n"
    << "  }\n"
    << "}\n\n"
    << "__device__ __forceinline__ bool like_verify(uintptr_t p, uintptr_t limit, "
       "like_literal_desc "
       "const& lit) {\n"
    << "  #pragma unroll\n"
    << "  for (int c = 0; c < 4; ++c) {\n"
    << "    if (c == lit.nchunks) { break; }\n"
    << "    uint64_t const win = read_window(p + static_cast<uintptr_t>(8 * c), limit);\n"
    << "    if ((win & lit.chunk_mask[c]) != lit.chunk_val[c]) { return false; }\n"
    << "  }\n"
    << "  return true;\n"
    << "}\n\n"
    // `%lit1%lit2%...%` (and contains = one literal): scan the string one aligned 8-byte word at
    // a time, locating candidate starts of the current literal with SWAR digram masks and
    // verifying them in order. Each literal must start at or after the previous match's end.
    << "template <bool Invert>\n"
    << "__device__ __forceinline__ reg_val<bool> eval_like_multiliteral(reg_val<str_view> s, "
       "like_pattern_desc const& pat) {\n"
    << "  if (!s.valid || s.overflow) { return reg_val<bool>{false, s.valid, s.overflow}; }\n"
    << "  bool match = false;\n"
    << "  if (s.val.len >= pat.total_len) {\n"
    << "    uintptr_t const start = reinterpret_cast<uintptr_t>(s.val.ptr);\n"
    << "    uintptr_t const end = start + static_cast<uintptr_t>(s.val.len);\n"
    << "    uintptr_t const limit = reinterpret_cast<uintptr_t>(s.val.limit);\n"
    << "    int32_t li = 0;\n"
    << "    uintptr_t min_pos = start;\n"
    << "    uint64_t b0 = pat.literals[0].b0_bcast;\n"
    << "    uint64_t b1 = pat.literals[0].b1_bcast;\n"
    << "    int32_t len = pat.literals[0].len;\n"
    << "    uintptr_t w = start & ~static_cast<uintptr_t>(7u);\n"
    << "    uint64_t cur = load_bounded_word(w, limit);\n"
    << "    for (; w < end && !match; w += 8u) {\n"
    << "      uint64_t const next = load_bounded_word(w + 8u, limit);\n"
    << "      uint64_t cand = like_candidates(cur, next, b0, b1, len);\n"
    << "      while (cand != 0ULL) {\n"
    << "        int const k = __ffsll(static_cast<long long>(cand)) - 1;\n"
    << "        cand &= cand - 1ULL;\n"
    << "        uintptr_t const p = w + static_cast<uintptr_t>(k >> 3);\n"
    << "        if (p < min_pos || p + static_cast<uintptr_t>(len) > end) { continue; }\n"
    << "        if (!like_verify(p, limit, like_literal_at(pat, li))) { continue; }\n"
    << "        min_pos = p + static_cast<uintptr_t>(len);\n"
    << "        ++li;\n"
    << "        if (li == pat.n) {\n"
    << "          match = true;\n"
    << "          break;\n"
    << "        }\n"
    << "        like_literal_desc const& nl = like_literal_at(pat, li);\n"
    << "        b0 = nl.b0_bcast;\n"
    << "        b1 = nl.b1_bcast;\n"
    << "        len = nl.len;\n"
    << "        cand = like_candidates(cur, next, b0, b1, len);\n"
    << "      }\n"
    << "      cur = next;\n"
    << "    }\n"
    << "  }\n"
    << "  return reg_val<bool>{match != Invert, true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ int str_view_cmp(str_view a, str_view b) {\n"
    << "  int32_t const min_len = a.len < b.len ? a.len : b.len;\n"
    << "  for (int32_t i = 0; i < min_len; ++i) {\n"
    << "    uint8_t const ca = static_cast<uint8_t>(a.ptr[i]);\n"
    << "    uint8_t const cb = static_cast<uint8_t>(b.ptr[i]);\n"
    << "    if (ca != cb) { return (ca < cb) ? -1 : 1; }\n"
    << "  }\n"
    << "  return (a.len < b.len) ? -1 : ((a.len > b.len) ? 1 : 0);\n"
    << "}\n\n"
    << "__device__ __forceinline__ bool operator==(str_view a, str_view b) {\n"
    << "  if (a.len != b.len) { return false; }\n"
    << "  for (int32_t i = 0; i < a.len; ++i) {\n"
    << "    if (a.ptr[i] != b.ptr[i]) { return false; }\n"
    << "  }\n"
    << "  return true;\n"
    << "}\n"
    << "__device__ __forceinline__ bool operator!=(str_view a, str_view b) { return !(a == b); }\n"
    << "__device__ __forceinline__ bool operator<(str_view a, str_view b) { return "
       "str_view_cmp(a, b) < 0; }\n"
    << "__device__ __forceinline__ bool operator<=(str_view a, str_view b) { return "
       "str_view_cmp(a, b) <= 0; }\n"
    << "__device__ __forceinline__ bool operator>(str_view a, str_view b) { return "
       "str_view_cmp(a, b) > 0; }\n"
    << "__device__ __forceinline__ bool operator>=(str_view a, str_view b) { return "
       "str_view_cmp(a, b) >= 0; }\n\n"
    << "template <typename OutT>\n"
    << "__device__ __forceinline__ reg_val<OutT> eval_strlen(reg_val<str_view> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }\n"
    << "  return reg_val<OutT>{static_cast<OutT>(in.val.len), true, false};\n"
    << "}\n\n"
    << "template <typename OutT>\n"
    << "__device__ __forceinline__ reg_val<OutT> eval_utf8_length(reg_val<str_view> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }\n"
    << "  int32_t count = 0;\n"
    << "  for (int32_t i = 0; i < in.val.len; ++i) {\n"
    << "    if ((static_cast<uint8_t>(in.val.ptr[i]) & 0xC0u) != 0x80u) { ++count; }\n"
    << "  }\n"
    << "  return reg_val<OutT>{static_cast<OutT>(count), true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ reg_val<bool> eval_prefix(reg_val<str_view> s, "
       "reg_val<str_view> pat) {\n"
    << "  bool const ovf = s.overflow || pat.overflow;\n"
    << "  bool const valid = s.valid && pat.valid;\n"
    << "  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }\n"
    << "  if (pat.val.len > s.val.len) { return reg_val<bool>{false, true, false}; }\n"
    << "  for (int32_t i = 0; i < pat.val.len; ++i) {\n"
    << "    if (s.val.ptr[i] != pat.val.ptr[i]) { return reg_val<bool>{false, true, false}; }\n"
    << "  }\n"
    << "  return reg_val<bool>{true, true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ reg_val<bool> eval_suffix(reg_val<str_view> s, "
       "reg_val<str_view> pat) {\n"
    << "  bool const ovf = s.overflow || pat.overflow;\n"
    << "  bool const valid = s.valid && pat.valid;\n"
    << "  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }\n"
    << "  if (pat.val.len > s.val.len) { return reg_val<bool>{false, true, false}; }\n"
    << "  int32_t const start = s.val.len - pat.val.len;\n"
    << "  for (int32_t i = 0; i < pat.val.len; ++i) {\n"
    << "    if (s.val.ptr[start + i] != pat.val.ptr[i]) {\n"
    << "      return reg_val<bool>{false, true, false};\n"
    << "    }\n"
    << "  }\n"
    << "  return reg_val<bool>{true, true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ uint64_t load_aligned_str_word(\n"
    << "    char const* base_ptr, int32_t total_len, int32_t byte_pos) {\n"
    << "  if (byte_pos >= total_len) { return 0ULL; }\n"
    << "  uintptr_t const addr = reinterpret_cast<uintptr_t>(base_ptr + byte_pos);\n"
    << "  if ((addr & 7u) == 0u && (total_len - byte_pos) >= 8) {\n"
    << "    return *reinterpret_cast<uint64_t const*>(addr);\n"
    << "  }\n"
    << "  uintptr_t const aligned_addr = addr & ~static_cast<uintptr_t>(7u);\n"
    << "  int32_t const shift = static_cast<int32_t>(addr & 7u) << 3;\n"
    << "  uintptr_t const end_addr = reinterpret_cast<uintptr_t>(base_ptr + total_len);\n"
    << "  if (aligned_addr + 8u <= end_addr) {\n"
    << "    uint64_t const lo = *reinterpret_cast<uint64_t const*>(aligned_addr);\n"
    << "    if (shift == 0) { return lo; }\n"
    << "    if (aligned_addr + 16u <= end_addr) {\n"
    << "      uint64_t const hi = *reinterpret_cast<uint64_t const*>(aligned_addr + 8u);\n"
    << "      return (lo >> shift) | (hi << (64 - shift));\n"
    << "    }\n"
    << "  }\n"
    << "  uint64_t val = 0ULL;\n"
    << "  int32_t const rem = (total_len - byte_pos) < 8 ? (total_len - byte_pos) : 8;\n"
    << "  for (int32_t i = 0; i < rem; ++i) {\n"
    << "    val |= static_cast<uint64_t>(static_cast<uint8_t>(base_ptr[byte_pos + i])) << (i << "
       "3);\n"
    << "  }\n"
    << "  return val;\n"
    << "}\n\n"
    << "__device__ __forceinline__ int32_t find_byte_swar(\n"
    << "    char const* s, int32_t slen, int32_t start, char target) {\n"
    << "  if (start >= slen) { return slen; }\n"
    << "  uint64_t const bcast = 0x0101010101010101ULL * static_cast<uint8_t>(target);\n"
    << "  int32_t pos = start;\n"
    << "  while (pos + 8 <= slen) {\n"
    << "    uint64_t const w = load_aligned_str_word(s, slen, pos);\n"
    << "    uint64_t cand = swar_byte_eq_mask(w, bcast);\n"
    << "    while (cand != 0ULL) {\n"
    << "      int32_t const lane = (__ffsll(static_cast<long long>(cand)) - 1) >> 3;\n"
    << "      cand &= (cand - 1ULL);\n"
    << "      if (s[pos + lane] == target) { return pos + lane; }\n"
    << "    }\n"
    << "    pos += 8;\n"
    << "  }\n"
    << "  while (pos < slen) {\n"
    << "    if (s[pos] == target) { return pos; }\n"
    << "    ++pos;\n"
    << "  }\n"
    << "  return slen;\n"
    << "}\n\n"
    << "__device__ __forceinline__ reg_val<bool> eval_contains(reg_val<str_view> s, "
       "reg_val<str_view> pat) {\n"
    << "  bool const ovf = s.overflow || pat.overflow;\n"
    << "  bool const valid = s.valid && pat.valid;\n"
    << "  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }\n"
    << "  if (pat.val.len <= 0) { return reg_val<bool>{true, true, false}; }\n"
    << "  if (s.val.len < pat.val.len) { return reg_val<bool>{false, true, false}; }\n"
    << "  char const first = pat.val.ptr[0];\n"
    << "  char const last = pat.val.ptr[pat.val.len - 1];\n"
    << "  int32_t const limit = s.val.len - pat.val.len;\n"
    << "  int32_t i = find_byte_swar(s.val.ptr, limit + 1, 0, first);\n"
    << "  while (i <= limit) {\n"
    << "    if (s.val.ptr[i + pat.val.len - 1] == last) {\n"
    << "      bool ok = true;\n"
    << "      int32_t j = 1;\n"
    << "      for (; j + 8 <= pat.val.len - 1; j += 8) {\n"
    << "        if (load_aligned_str_word(s.val.ptr, s.val.len, i + j) !=\n"
    << "            load_aligned_str_word(pat.val.ptr, pat.val.len, j)) {\n"
    << "          ok = false;\n"
    << "          break;\n"
    << "        }\n"
    << "      }\n"
    << "      for (; ok && j < pat.val.len - 1; ++j) {\n"
    << "        if (s.val.ptr[i + j] != pat.val.ptr[j]) { ok = false; break; }\n"
    << "      }\n"
    << "      if (ok) { return reg_val<bool>{true, true, false}; }\n"
    << "    }\n"
    << "    i = find_byte_swar(s.val.ptr, limit + 1, i + 1, first);\n"
    << "  }\n"
    << "  return reg_val<bool>{false, true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ int32_t next_utf8_char(char const* s, int32_t slen, int32_t "
       "pos) {\n"
    << "  int32_t next = pos + 1;\n"
    << "  while (next < slen && (static_cast<uint8_t>(s[next]) & 0xC0u) == 0x80u) {\n"
    << "    ++next;\n"
    << "  }\n"
    << "  return next;\n"
    << "}\n\n"
    << "__device__ __forceinline__ int32_t find_byte_plain(\n"
    << "    char const* s, int32_t slen, int32_t start, char target) {\n"
    << "  int32_t pos = start;\n"
    << "  while (pos < slen && s[pos] != target) { ++pos; }\n"
    << "  return pos;\n"
    << "}\n\n"
    << "__device__ __forceinline__ bool raw_like(str_view s, str_view pat) {\n"
    << "  int32_t si = 0;\n"
    << "  int32_t pi = 0;\n"
    << "  int32_t star_pi = -1;\n"
    << "  int32_t star_si = -1;\n"
    << "  while (si < s.len) {\n"
    << "    char const pc = (pi < pat.len) ? pat.ptr[pi] : '\\0';\n"
    << "    if (pi < pat.len && pc == '%') {\n"
    << "      while (pi < pat.len && pat.ptr[pi] == '%') { ++pi; }\n"
    << "      if (pi == pat.len) { return true; }\n"
    << "      star_pi = pi;\n"
    << "      if (pat.ptr[star_pi] != '_') {\n"
    << "        si = find_byte_plain(s.ptr, s.len, si, pat.ptr[star_pi]);\n"
    << "      }\n"
    << "      star_si = si;\n"
    << "    } else if (pi < pat.len && pc == '_') {\n"
    << "      si = next_utf8_char(s.ptr, s.len, si);\n"
    << "      ++pi;\n"
    << "    } else if (pi < pat.len && s.ptr[si] == pc) {\n"
    << "      ++si;\n"
    << "      ++pi;\n"
    << "    } else if (star_pi != -1) {\n"
    << "      pi = star_pi;\n"
    << "      if (pat.ptr[star_pi] != '_') {\n"
    << "        star_si = find_byte_plain(s.ptr, s.len, star_si + 1, pat.ptr[star_pi]);\n"
    << "      } else {\n"
    << "        star_si = next_utf8_char(s.ptr, s.len, star_si);\n"
    << "      }\n"
    << "      si = star_si;\n"
    << "    } else {\n"
    << "      return false;\n"
    << "    }\n"
    << "  }\n"
    << "  while (pi < pat.len && pat.ptr[pi] == '%') { ++pi; }\n"
    << "  return pi == pat.len;\n"
    << "}\n\n"
    << "template <bool Invert>\n"
    << "__device__ __forceinline__ reg_val<bool> eval_like(reg_val<str_view> s, reg_val<str_view> "
       "pat) {\n"
    << "  bool const ovf = s.overflow || pat.overflow;\n"
    << "  bool const valid = s.valid && pat.valid;\n"
    << "  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }\n"
    << "  bool const m = raw_like(s.val, pat.val);\n"
    << "  return reg_val<bool>{Invert ? !m : m, true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ str_view slice_utf8_forward(str_view s, int32_t start_char, "
       "int32_t stop_char) {\n"
    << "  if (stop_char <= start_char || s.len <= 0) { return str_view{s.ptr, 0, s.limit}; }\n"
    << "  int32_t char_idx = 0;\n"
    << "  int32_t byte_start = s.len;\n"
    << "  int32_t byte_stop = s.len;\n"
    << "  for (int32_t b = 0; b < s.len; ++b) {\n"
    << "    if ((static_cast<uint8_t>(s.ptr[b]) & 0xC0u) != 0x80u) {\n"
    << "      if (char_idx == start_char) { byte_start = b; }\n"
    << "      if (char_idx == stop_char) {\n"
    << "        byte_stop = b;\n"
    << "        break;\n"
    << "      }\n"
    << "      ++char_idx;\n"
    << "    }\n"
    << "  }\n"
    << "  if (byte_start >= byte_stop) { return str_view{s.ptr, 0, s.limit}; }\n"
    << "  return str_view{s.ptr + byte_start, byte_stop - byte_start, s.limit};\n"
    << "}\n\n"
    << "__device__ __forceinline__ reg_val<str_view> eval_substring(\n"
    << "    reg_val<str_view> in, int32_t start_char, int32_t stop_char) {\n"
    << "  if (!in.valid || in.overflow) {\n"
    << "    return reg_val<str_view>{str_view{nullptr, 0, nullptr}, in.valid, in.overflow};\n"
    << "  }\n"
    << "  if (start_char >= 0 && stop_char >= 0) {\n"
    << "    return reg_val<str_view>{slice_utf8_forward(in.val, start_char, stop_char), true, "
       "false};\n"
    << "  }\n"
    << "  int32_t char_len = 0;\n"
    << "  for (int32_t b = 0; b < in.val.len; ++b) {\n"
    << "    if ((static_cast<uint8_t>(in.val.ptr[b]) & 0xC0u) != 0x80u) { ++char_len; }\n"
    << "  }\n"
    << "  int32_t r_start = (start_char < 0) ? (char_len + start_char) : start_char;\n"
    << "  if (r_start < 0) { r_start = 0; }\n"
    << "  if (r_start > char_len) { r_start = char_len; }\n"
    << "  int32_t r_stop = (stop_char < 0) ? (char_len + stop_char) : stop_char;\n"
    << "  if (r_stop < 0) { r_stop = 0; }\n"
    << "  if (r_stop > char_len) { r_stop = char_len; }\n"
    << "  return reg_val<str_view>{slice_utf8_forward(in.val, r_start, r_stop), true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ bool is_bit_set(uint32_t const* mask, int32_t bit_idx) {\n"
    << "  return ((mask[bit_idx >> 5] >> (bit_idx & 31)) & 1u) != 0u;\n"
    << "}\n\n"
    << "__device__ __forceinline__ reg_val<bool> eval_kleene_and(reg_val<bool> acc, reg_val<bool> "
       "rhs) {\n"
    << "  bool const acc_false = acc.valid && !acc.overflow && !acc.val;\n"
    << "  bool const rhs_false = rhs.valid && !rhs.overflow && !rhs.val;\n"
    << "  if (acc_false || rhs_false) { return reg_val<bool>{false, true, false}; }\n"
    << "  if (acc.overflow || rhs.overflow) { return reg_val<bool>{false, true, true}; }\n"
    << "  if (acc.valid && rhs.valid) { return reg_val<bool>{true, true, false}; }\n"
    << "  return reg_val<bool>{false, false, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ reg_val<bool> eval_kleene_or(reg_val<bool> acc, reg_val<bool> "
       "rhs) {\n"
    << "  bool const acc_true = acc.valid && !acc.overflow && acc.val;\n"
    << "  bool const rhs_true = rhs.valid && !rhs.overflow && rhs.val;\n"
    << "  if (acc_true || rhs_true) { return reg_val<bool>{true, true, false}; }\n"
    << "  if (acc.overflow || rhs.overflow) { return reg_val<bool>{false, true, true}; }\n"
    << "  if (acc.valid && rhs.valid) { return reg_val<bool>{false, true, false}; }\n"
    << "  return reg_val<bool>{false, false, false};\n"
    << "}\n\n"
    << "template <typename T, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<T> eval_add(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }\n"
    << "  if constexpr (is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{static_cast<T>(l.val) + static_cast<T>(r.val), true, false};\n"
    // 64-bit sums: wrap in the unsigned domain and read the overflow off the sign bits (signed)
    // or the carry (unsigned) instead of widening to 128 bits.
    << "  } else if constexpr (sizeof(T) == 8) {\n"
    << "    T const lv = static_cast<T>(l.val);\n"
    << "    T const rv = static_cast<T>(r.val);\n"
    << "    T const res = static_cast<T>(static_cast<uint64_t>(lv) + static_cast<uint64_t>(rv));\n"
    << "    bool ovf;\n"
    << "    if constexpr (is_signed_v<T>) {\n"
    << "      ovf = static_cast<int64_t>((lv ^ res) & (rv ^ res)) < 0;\n"
    << "    } else {\n"
    << "      ovf = res < lv;\n"
    << "    }\n"
    << "    return reg_val<T>{ovf ? T{} : res, true, ovf};\n"
    << "  } else if constexpr (is_signed_v<T>) {\n"
    << "    using Wide = conditional_t<(sizeof(T) < 8), int64_t, __int128_t>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) + "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = (res < static_cast<Wide>(limits<T>::lowest())) ||\n"
    << "                     (res > static_cast<Wide>(limits<T>::max()));\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  } else {\n"
    << "    using Wide = conditional_t<(sizeof(T) < 8), uint64_t, unsigned __int128>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) + "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = res > static_cast<Wide>(limits<T>::max());\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<T> eval_sub(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }\n"
    << "  if constexpr (is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{static_cast<T>(l.val) - static_cast<T>(r.val), true, false};\n"
    << "  } else if constexpr (sizeof(T) == 8 && is_signed_v<T>) {\n"
    << "    T const lv = static_cast<T>(l.val);\n"
    << "    T const rv = static_cast<T>(r.val);\n"
    << "    T const res = static_cast<T>(static_cast<uint64_t>(lv) - static_cast<uint64_t>(rv));\n"
    << "    bool const ovf = static_cast<int64_t>((lv ^ rv) & (lv ^ res)) < 0;\n"
    << "    return reg_val<T>{ovf ? T{} : res, true, ovf};\n"
    << "  } else if constexpr (is_signed_v<T>) {\n"
    << "    using Wide = conditional_t<(sizeof(T) < 8), int64_t, __int128_t>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) - "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = (res < static_cast<Wide>(limits<T>::lowest())) ||\n"
    << "                     (res > static_cast<Wide>(limits<T>::max()));\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  } else {\n"
    << "    T const lv = static_cast<T>(l.val);\n"
    << "    T const rv = static_cast<T>(r.val);\n"
    << "    bool const ovf = lv < rv;\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(lv - rv), true, ovf};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<T> eval_mul(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }\n"
    << "  if constexpr (is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{static_cast<T>(l.val) * static_cast<T>(r.val), true, false};\n"
    << "  } else if constexpr (sizeof(T) == 8) {\n"
    // 64-bit products: the high half of the 128-bit product is one instruction, and the product
    // fits iff that half is the sign extension of the low half (signed) or zero (unsigned).
    << "    T const lv = static_cast<T>(l.val);\n"
    << "    T const rv = static_cast<T>(r.val);\n"
    << "    T const lo = static_cast<T>(static_cast<uint64_t>(lv) * static_cast<uint64_t>(rv));\n"
    << "    bool ovf;\n"
    << "    if constexpr (is_signed_v<T>) {\n"
    << "      ovf = __mul64hi(lv, rv) != (lo >> 63);\n"
    << "    } else {\n"
    << "      ovf = __umul64hi(lv, rv) != 0ULL;\n"
    << "    }\n"
    << "    return reg_val<T>{ovf ? T{} : lo, true, ovf};\n"
    << "  } else if constexpr (is_signed_v<T>) {\n"
    << "    int64_t const res = static_cast<int64_t>(static_cast<T>(l.val)) * "
       "static_cast<int64_t>(static_cast<T>(r.val));\n"
    << "    bool const ovf = (res < static_cast<int64_t>(limits<T>::lowest())) "
       "||\n"
    << "                     (res > static_cast<int64_t>(limits<T>::max()));\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  } else {\n"
    << "    uint64_t const res = static_cast<uint64_t>(static_cast<T>(l.val)) * "
       "static_cast<uint64_t>(static_cast<T>(r.val));\n"
    << "    bool const ovf = res > static_cast<uint64_t>(limits<T>::max());\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<T> eval_div(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }\n"
    << "  T const lv = static_cast<T>(l.val);\n"
    << "  T const rv = static_cast<T>(r.val);\n"
    << "  if constexpr (is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{lv / rv, true, false};\n"
    << "  } else {\n"
    << "    if (rv == T{0}) { return reg_val<T>{T{}, false, false}; }\n"
    << "    if constexpr (is_signed_v<T>) {\n"
    << "      if (lv == limits<T>::lowest() && rv == T{-1}) {\n"
    << "        return reg_val<T>{T{}, true, true};\n"
    << "      }\n"
    << "    }\n"
    << "    return reg_val<T>{static_cast<T>(lv / rv), true, false};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<T> eval_mod(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }\n"
    << "  T const lv = static_cast<T>(l.val);\n"
    << "  T const rv = static_cast<T>(r.val);\n"
    << "  if constexpr (is_floating_point_v<T>) {\n"
    << "    if constexpr (sizeof(T) == 4) {\n"
    << "      return reg_val<T>{::fmodf(lv, rv), true, false};\n"
    << "    } else {\n"
    << "      return reg_val<T>{::fmod(lv, rv), true, false};\n"
    << "    }\n"
    << "  } else {\n"
    << "    if (rv == T{0}) { return reg_val<T>{T{}, false, false}; }\n"
    << "    if constexpr (is_signed_v<T>) {\n"
    << "      if (lv == limits<T>::lowest() && rv == T{-1}) {\n"
    << "        return reg_val<T>{T{0}, true, false};\n"
    << "      }\n"
    << "    }\n"
    << "    return reg_val<T>{static_cast<T>(lv % rv), true, false};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T, typename InT>\n"
    << "__device__ __forceinline__ reg_val<T> eval_abs(reg_val<InT> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<T>{T{}, in.valid, in.overflow}; }\n"
    << "  T const v = static_cast<T>(in.val);\n"
    << "  if constexpr (is_floating_point_v<T>) {\n"
    << "    if constexpr (sizeof(T) == 4) {\n"
    << "      return reg_val<T>{::fabsf(v), true, false};\n"
    << "    } else {\n"
    << "      return reg_val<T>{::fabs(v), true, false};\n"
    << "    }\n"
    << "  } else if constexpr (is_signed_v<T>) {\n"
    << "    if (v == limits<T>::lowest()) {\n"
    << "      return reg_val<T>{T{}, true, true};\n"
    << "    }\n"
    << "    return reg_val<T>{static_cast<T>(v < T{0} ? -v : v), true, false};\n"
    << "  } else {\n"
    << "    return reg_val<T>{v, true, false};\n"
    << "  }\n"
    << "}\n\n"
    << "__device__ __forceinline__ bool is_finite_double(double x) {\n"
    << "  return (x - x) == 0.0;\n"
    << "}\n\n"
    << "template <typename T, bool NegativePrecision>\n"
    << "__device__ __forceinline__ reg_val<T> eval_round(reg_val<T> in, double modifier) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<T>{T{}, in.valid, in.overflow}; }\n"
    << "  double const x = static_cast<double>(in.val);\n"
    << "  double rounded;\n"
    << "  if constexpr (NegativePrecision) {\n"
    << "    rounded = ::round(x / modifier) * modifier;\n"
    << "    if (!is_finite_double(rounded)) { rounded = 0.0; }\n"
    << "  } else {\n"
    << "    rounded = ::round(x * modifier) / modifier;\n"
    << "    if (!is_finite_double(rounded)) { rounded = x; }\n"
    << "  }\n"
    << "  return reg_val<T>{static_cast<T>(rounded), true, false};\n"
    << "}\n\n"
    << "template <typename OutRep, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<OutRep> eval_dec_add(\n"
    << "    reg_val<L> l, reg_val<R> r, __int128_t f_l, __int128_t f_r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }\n"
    // Operands of at most 64 bits rescaled by at most 10^18 (the widest DECIMAL64 scale) stay
    // far inside 128 bits, so only the final narrowing can overflow.
    << "  if constexpr (sizeof(L) <= 8 && sizeof(R) <= 8) {\n"
    << "    __int128_t const sum = static_cast<__int128_t>(l.val) * f_l + "
       "static_cast<__int128_t>(r.val) * f_r;\n"
    << "    auto const out = overflow_cast<OutRep>(sum);\n"
    << "    return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};\n"
    << "  }\n"
    << "  auto const sl = mul_overflow128(static_cast<__int128_t>(l.val), f_l);\n"
    << "  auto const sr = mul_overflow128(static_cast<__int128_t>(r.val), f_r);\n"
    << "  if (sl.overflow || sr.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }\n"
    << "  auto const sum = add_overflow128(sl.value, sr.value);\n"
    << "  if (sum.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }\n"
    << "  auto const out = overflow_cast<OutRep>(sum.value);\n"
    << "  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <typename OutRep, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<OutRep> eval_dec_sub(\n"
    << "    reg_val<L> l, reg_val<R> r, __int128_t f_l, __int128_t f_r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }\n"
    // Operands of at most 64 bits rescaled by at most 10^18 (the widest DECIMAL64 scale) stay
    // far inside 128 bits, so only the final narrowing can overflow.
    << "  if constexpr (sizeof(L) <= 8 && sizeof(R) <= 8) {\n"
    << "    __int128_t const diff = static_cast<__int128_t>(l.val) * f_l - "
       "static_cast<__int128_t>(r.val) * f_r;\n"
    << "    auto const out = overflow_cast<OutRep>(diff);\n"
    << "    return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};\n"
    << "  }\n"
    << "  auto const sl = mul_overflow128(static_cast<__int128_t>(l.val), f_l);\n"
    << "  auto const sr = mul_overflow128(static_cast<__int128_t>(r.val), f_r);\n"
    << "  if (sl.overflow || sr.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }\n"
    << "  auto const diff = sub_overflow128(sl.value, sr.value);\n"
    << "  if (diff.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }\n"
    << "  auto const out = overflow_cast<OutRep>(diff.value);\n"
    << "  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <typename OutRep, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<OutRep> eval_dec_mul(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }\n"
    // Two operands narrower than the result can never overflow it (the product of two 64-bit
    // values always fits in 128 bits); only a same-width product needs the checked multiply.
    << "  if constexpr (sizeof(L) < sizeof(OutRep) && sizeof(R) < sizeof(OutRep)) {\n"
    << "    return reg_val<OutRep>{static_cast<OutRep>(l.val) * static_cast<OutRep>(r.val), true, "
       "false};\n"
    << "  } else {\n"
    << "    auto const prod = mul_overflow<OutRep>(static_cast<OutRep>(l.val), "
       "static_cast<OutRep>(r.val));\n"
    << "    return reg_val<OutRep>{prod.overflow ? OutRep{} : prod.value, true, prod.overflow};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename OutRep, bool ScaleUp = true, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<OutRep> eval_dec_div(\n"
    << "    reg_val<L> l, reg_val<R> r, __int128_t f_out = 1) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }\n"
    << "  __int128_t const lv = static_cast<__int128_t>(l.val);\n"
    << "  __int128_t const rv = static_cast<__int128_t>(r.val);\n"
    << "  if (rv == 0) { return reg_val<OutRep>{OutRep{}, false, false}; }\n"
    << "  if (lv == limits<__int128_t>::lowest() && rv == -1) {\n"
    << "    return reg_val<OutRep>{OutRep{}, true, true};\n"
    << "  }\n"
    << "  __int128_t q = lv / rv;\n"
    << "  if constexpr (ScaleUp) {\n"
    << "    if (f_out != 1) {\n"
    << "      auto const scaled = mul_overflow128(q, f_out);\n"
    << "      if (scaled.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }\n"
    << "      q = scaled.value;\n"
    << "    }\n"
    << "  } else {\n"
    << "    q /= f_out;\n"
    << "  }\n"
    << "  auto const out = overflow_cast<OutRep>(q);\n"
    << "  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <typename OutRep, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<OutRep> eval_dec_mod(\n"
    << "    reg_val<L> l, reg_val<R> r, __int128_t f_l, __int128_t f_r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }\n"
    << "  auto const sl = mul_overflow128(static_cast<__int128_t>(l.val), f_l);\n"
    << "  auto const sr = mul_overflow128(static_cast<__int128_t>(r.val), f_r);\n"
    << "  if (sl.overflow || sr.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }\n"
    << "  if (sr.value == 0) { return reg_val<OutRep>{OutRep{}, false, false}; }\n"
    << "  if (sl.value == limits<__int128_t>::lowest() && sr.value == -1) {\n"
    << "    return reg_val<OutRep>{OutRep{0}, true, false};\n"
    << "  }\n"
    << "  __int128_t const rem = sl.value % sr.value;\n"
    << "  auto const out = overflow_cast<OutRep>(rem);\n"
    << "  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <typename DstT, typename SrcT>\n"
    << "__device__ __forceinline__ reg_val<DstT> eval_cast_int_to_int(reg_val<SrcT> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstT>{DstT{}, in.valid, in.overflow}; }\n"
    << "  bool const fits = in_range<DstT>(in.val);\n"
    << "  return reg_val<DstT>{fits ? static_cast<DstT>(in.val) : DstT{}, true, !fits};\n"
    << "}\n\n"
    << "template <typename SrcT>\n"
    << "__device__ __forceinline__ reg_val<bool> eval_cast_to_bool(reg_val<SrcT> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<bool>{false, in.valid, in.overflow}; }\n"
    << "  return reg_val<bool>{in.val != SrcT{0}, true, false};\n"
    << "}\n\n"
    << "template <typename DstRep, bool CanFitOne>\n"
    << "__device__ __forceinline__ reg_val<DstRep> eval_cast_bool_to_dec(reg_val<bool> in, DstRep "
       "one_scaled) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; "
       "}\n"
    << "  if (!in.val) { return reg_val<DstRep>{DstRep{0}, true, false}; }\n"
    << "  if constexpr (CanFitOne) {\n"
    << "    return reg_val<DstRep>{one_scaled, true, false};\n"
    << "  } else {\n"
    << "    return reg_val<DstRep>{DstRep{}, true, true};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename DstFloat, typename SrcFloat>\n"
    << "__device__ __forceinline__ reg_val<DstFloat> eval_cast_float_to_float(reg_val<SrcFloat> "
       "in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstFloat>{DstFloat{}, in.valid, "
       "in.overflow}; }\n"
    << "  if constexpr (sizeof(DstFloat) < sizeof(SrcFloat)) {\n"
    << "    double const d = static_cast<double>(in.val);\n"
    << "    if (!is_finite_double(d)) { return reg_val<DstFloat>{static_cast<DstFloat>(d), true, "
       "false}; }\n"
    << "    float const f = static_cast<float>(d);\n"
    << "    bool const f_finite = (f - f) == 0.0f;\n"
    << "    return reg_val<DstFloat>{f_finite ? f : DstFloat{}, true, !f_finite};\n"
    << "  } else {\n"
    << "    return reg_val<DstFloat>{static_cast<DstFloat>(in.val), true, false};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename DstInt, typename SrcFloat>\n"
    << "__device__ __forceinline__ reg_val<DstInt> eval_cast_float_to_int(reg_val<SrcFloat> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstInt>{DstInt{}, in.valid, in.overflow}; "
       "}\n"
    << "  double const d = static_cast<double>(in.val);\n"
    << "  if (!is_finite_double(d)) { return reg_val<DstInt>{DstInt{}, true, true}; }\n"
    << "  if constexpr (sizeof(DstInt) < 8) {\n"
    << "    double const lo = static_cast<double>(limits<DstInt>::lowest());\n"
    << "    double const hi = static_cast<double>(limits<DstInt>::max());\n"
    << "    if (d < lo || d > hi) { return reg_val<DstInt>{DstInt{}, true, true}; }\n"
    << "    return reg_val<DstInt>{static_cast<DstInt>(in.val), true, false};\n"
    << "  } else if constexpr (is_signed_v<DstInt>) {\n"
    << "    if (d < -9223372036854775808.0 || d >= 9223372036854775808.0) {\n"
    << "      return reg_val<DstInt>{DstInt{}, true, true};\n"
    << "    }\n"
    << "    return reg_val<DstInt>{static_cast<DstInt>(in.val), true, false};\n"
    << "  } else {\n"
    << "    if (d < 0.0 || d >= 18446744073709551616.0) {\n"
    << "      return reg_val<DstInt>{DstInt{}, true, true};\n"
    << "    }\n"
    << "    return reg_val<DstInt>{static_cast<DstInt>(in.val), true, false};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename DstRep, typename SrcInt>\n"
    << "__device__ __forceinline__ reg_val<DstRep> eval_cast_int_to_dec(\n"
    << "    reg_val<SrcInt> in, __int128_t int_limit, __int128_t factor) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; "
       "}\n"
    << "  __int128_t const v = static_cast<__int128_t>(in.val);\n"
    << "  if (v <= -int_limit || v >= int_limit) { return reg_val<DstRep>{DstRep{}, true, true}; "
       "}\n"
    << "  auto const scaled = mul_overflow128(v, factor);\n"
    << "  if (scaled.overflow) { return reg_val<DstRep>{DstRep{}, true, true}; }\n"
    << "  auto const out = overflow_cast<DstRep>(scaled.value);\n"
    << "  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <typename DstInt, typename SrcRep>\n"
    << "__device__ __forceinline__ reg_val<DstInt> eval_cast_dec_to_int(\n"
    << "    reg_val<SrcRep> in, __int128_t factor) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstInt>{DstInt{}, in.valid, in.overflow}; "
       "}\n"
    << "  __int128_t const q = static_cast<__int128_t>(in.val) / factor;\n"
    << "  bool const fits = in_range<DstInt>(q);\n"
    << "  return reg_val<DstInt>{fits ? static_cast<DstInt>(q) : DstInt{}, true, !fits};\n"
    << "}\n\n"
    << "template <typename DstRep, typename SrcRep>\n"
    << "__device__ __forceinline__ reg_val<DstRep> eval_cast_dec_scale_up(\n"
    << "    reg_val<SrcRep> in, __int128_t factor, __int128_t limit) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; "
       "}\n"
    << "  auto const scaled = mul_overflow128(static_cast<__int128_t>(in.val), "
       "factor);\n"
    << "  if (scaled.overflow || scaled.value <= -limit || scaled.value >= limit) {\n"
    << "    return reg_val<DstRep>{DstRep{}, true, true};\n"
    << "  }\n"
    << "  auto const out = overflow_cast<DstRep>(scaled.value);\n"
    << "  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <typename DstRep, typename SrcRep>\n"
    << "__device__ __forceinline__ reg_val<DstRep> eval_cast_dec_scale_down(\n"
    << "    reg_val<SrcRep> in, __int128_t factor, __int128_t limit) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; "
       "}\n"
    << "  __int128_t const q = static_cast<__int128_t>(in.val) / factor;\n"
    << "  if (q <= -limit || q >= limit) { return reg_val<DstRep>{DstRep{}, true, true}; }\n"
    << "  auto const out = overflow_cast<DstRep>(q);\n"
    << "  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <typename DstFloat, typename SrcRep>\n"
    << "__device__ __forceinline__ reg_val<DstFloat> eval_cast_dec_to_float(\n"
    << "    reg_val<SrcRep> in, double scale_pow10) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstFloat>{DstFloat{}, in.valid, "
       "in.overflow}; }\n"
    << "  if constexpr (sizeof(DstFloat) == 4) {\n"
    << "    return reg_val<float>{static_cast<float>(in.val) / static_cast<float>(scale_pow10), "
       "true, false};\n"
    << "  } else {\n"
    << "    return reg_val<double>{static_cast<double>(in.val) / scale_pow10, true, false};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename DstRep, typename SrcFloat>\n"
    << "__device__ __forceinline__ reg_val<DstRep> eval_cast_float_to_dec(\n"
    << "    reg_val<SrcFloat> in, double scale_pow10, double limit_pow10, __int128_t limit_128) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; "
       "}\n"
    << "  double const d = static_cast<double>(in.val);\n"
    << "  if (!is_finite_double(d)) { return reg_val<DstRep>{DstRep{}, true, true}; }\n"
    << "  double const scaled = d * scale_pow10;\n"
    << "  if (!is_finite_double(scaled) || scaled <= -limit_pow10 || scaled >= limit_pow10) {\n"
    << "    return reg_val<DstRep>{DstRep{}, true, true};\n"
    << "  }\n"
    << "  __int128_t const iv = static_cast<__int128_t>(scaled);\n"
    << "  if (iv <= -limit_128 || iv >= limit_128) { return reg_val<DstRep>{DstRep{}, true, true}; "
       "}\n"
    << "  auto const out = overflow_cast<DstRep>(iv);\n"
    << "  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};\n"
    << "}\n\n"
    << "template <int64_t Scale>\n"
    << "__device__ __forceinline__ reg_val<int64_t> eval_cast_date_to_timestamp(reg_val<int32_t> "
       "in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }\n"
    << "  constexpr int32_t date_inf = limits<int32_t>::max();\n"
    << "  constexpr int64_t ts_inf = limits<int64_t>::max();\n"
    << "  constexpr int64_t micros_per_day = 86400000000LL;\n"
    << "  constexpr int64_t max_days = ts_inf / (Scale > micros_per_day ? Scale : "
       "micros_per_day);\n"
    << "  int32_t const day = in.val;\n"
    << "  if (day == date_inf) { return reg_val<int64_t>{ts_inf, true, false}; }\n"
    << "  if (day == -date_inf) { return reg_val<int64_t>{-ts_inf, true, false}; }\n"
    << "  if (day < -max_days || day > max_days) {\n"
    << "    return reg_val<int64_t>{0, true, true};\n"
    << "  }\n"
    << "  return reg_val<int64_t>{static_cast<int64_t>(day) * Scale, true, false};\n"
    << "}\n\n"
    << "template <int64_t Scale>\n"
    << "__device__ __forceinline__ reg_val<int64_t> eval_cast_ts_scale_up(reg_val<int64_t> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }\n"
    << "  constexpr int64_t ts_inf = limits<int64_t>::max();\n"
    << "  constexpr int64_t ts_min = limits<int64_t>::lowest();\n"
    << "  int64_t const t = in.val;\n"
    << "  if (t == ts_inf || t == -ts_inf) { return reg_val<int64_t>{t, true, false}; }\n"
    << "  if (t < (ts_min / Scale) || t > (ts_inf / Scale)) {\n"
    << "    return reg_val<int64_t>{0, true, true};\n"
    << "  }\n"
    << "  return reg_val<int64_t>{t * Scale, true, false};\n"
    << "}\n\n"
    << "__device__ __forceinline__ reg_val<int64_t> eval_cast_ts_ns_to_us(reg_val<int64_t> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }\n"
    << "  constexpr int64_t ts_inf = limits<int64_t>::max();\n"
    << "  int64_t const t = in.val;\n"
    << "  if (t == ts_inf || t == -ts_inf) { return reg_val<int64_t>{t, true, false}; }\n"
    << "  int64_t const half = t / 500LL;\n"
    << "  int64_t const rounded = half + (half < 0 ? -1LL : 1LL);\n"
    << "  return reg_val<int64_t>{rounded / 2LL, true, false};\n"
    << "}\n\n"
    << "template <int64_t Scale>\n"
    << "__device__ __forceinline__ reg_val<int64_t> eval_cast_ts_scale_down(reg_val<int64_t> in) "
       "{\n"
    << "  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }\n"
    << "  constexpr int64_t ts_inf = limits<int64_t>::max();\n"
    << "  int64_t const t = in.val;\n"
    << "  if (t == ts_inf || t == -ts_inf) { return reg_val<int64_t>{t, true, false}; }\n"
    << "  int64_t const q = (t >= 0) ? (t / Scale) : ((t - Scale + 1LL) / Scale);\n"
    << "  return reg_val<int64_t>{q, true, false};\n"
    << "}\n\n"
    << "template <int64_t Scale>\n"
    << "__device__ __forceinline__ reg_val<int32_t> eval_cast_timestamp_to_date(reg_val<int64_t> "
       "in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<int32_t>{0, in.valid, in.overflow}; }\n"
    << "  constexpr int32_t date_inf = limits<int32_t>::max();\n"
    << "  constexpr int64_t ts_inf = limits<int64_t>::max();\n"
    << "  int64_t const t = in.val;\n"
    << "  if (t == ts_inf) { return reg_val<int32_t>{date_inf, true, false}; }\n"
    << "  if (t == -ts_inf) { return reg_val<int32_t>{-date_inf, true, false}; }\n"
    << "  int64_t const days = (t >= 0) ? (t / Scale) : ((t - Scale + 1LL) / Scale);\n"
    << "  if (days < static_cast<int64_t>(limits<int32_t>::lowest()) ||\n"
    << "      days >= static_cast<int64_t>(date_inf)) {\n"
    << "    return reg_val<int32_t>{0, true, true};\n"
    << "  }\n"
    << "  return reg_val<int32_t>{static_cast<int32_t>(days), true, false};\n"
    << "}\n\n"
    // Proleptic Gregorian date from a 32-bit count of days since 1970-01-01. This is
    // cuda::std::chrono::year_month_day's algorithm (H. Hinnant's civil_from_days) with the same
    // 32-bit `int` arithmetic, so the +/-infinity sentinels wrap exactly as they do in the
    // cuDF-based evaluators; the wrapping steps use unsigned arithmetic to stay well defined.
    << "__device__ __forceinline__ void civil_from_days(int32_t days, int16_t& y, int16_t& m, "
       "int16_t& d) {\n"
    << "  int32_t const z = static_cast<int32_t>(static_cast<uint32_t>(days) + 719468u);\n"
    << "  int32_t const z_neg = static_cast<int32_t>(static_cast<uint32_t>(z) - 146096u);\n"
    // `z >= 0` is tested as `days >= -719468`: that is how the optimizer folds the (formally
    // overflowing) test in the compiled chrono code, and it is what decides the sentinel's value.
    << "  int32_t const era = (days >= -719468 ? z : z_neg) / 146097;\n"
    << "  uint32_t const doe = static_cast<uint32_t>(z) - static_cast<uint32_t>(era) * 146097u;\n"
    << "  uint32_t const yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;\n"
    << "  int32_t const yr = static_cast<int32_t>(yoe) + era * 400;\n"
    << "  uint32_t const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);\n"
    << "  uint32_t const mp = (5 * doy + 2) / 153;\n"
    << "  uint32_t const dd = doy - (153 * mp + 2) / 5 + 1;\n"
    << "  uint32_t const mth = mp + (mp < 10 ? 3u : static_cast<uint32_t>(-9));\n"
    // chrono::year holds a short and chrono::month / chrono::day an unsigned char.
    << "  y = static_cast<int16_t>(yr + (mth <= 2 ? 1 : 0));\n"
    << "  m = static_cast<int16_t>(static_cast<uint8_t>(mth));\n"
    << "  d = static_cast<int16_t>(static_cast<uint8_t>(dd));\n"
    << "}\n\n"
    << "template <typename OutT, int Comp>\n"
    << "__device__ __forceinline__ reg_val<OutT> eval_extract_date(reg_val<int32_t> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }\n"
    << "  int32_t const days = in.val;\n"
    << "  if constexpr (Comp == 6 || Comp == 7) {\n"
    << "    constexpr int32_t date_inf = limits<int32_t>::max();\n"
    << "    if (days == date_inf || days == -date_inf) {\n"
    << "      return reg_val<OutT>{OutT{}, false, false};\n"
    << "    }\n"
    << "    return reg_val<OutT>{OutT{0}, true, false};\n"
    << "  } else if constexpr (Comp == 3 || Comp == 4 || Comp == 5) {\n"
    << "    return reg_val<OutT>{OutT{0}, true, false};\n"
    << "  } else {\n"
    << "    int16_t y = 0, m = 0, d = 0;\n"
    << "    civil_from_days(days, y, m, d);\n"
    << "    int16_t const res = (Comp == 0) ? y : ((Comp == 1) ? m : d);\n"
    << "    return reg_val<OutT>{static_cast<OutT>(res), true, false};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename OutT, int64_t TicksPerSec, int Comp>\n"
    << "__device__ __forceinline__ reg_val<OutT> eval_extract_timestamp(reg_val<int64_t> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }\n"
    << "  int64_t const ticks = in.val;\n"
    << "  if constexpr (Comp == 6 || Comp == 7) {\n"
    << "    constexpr int64_t ts_inf = limits<int64_t>::max();\n"
    << "    if (ticks == ts_inf || ticks == -ts_inf) {\n"
    << "      return reg_val<OutT>{OutT{}, false, false};\n"
    << "    }\n"
    << "  }\n"
    << "  constexpr int64_t ticks_per_day = 86400LL * TicksPerSec;\n"
    << "  if constexpr (Comp <= 5) {\n"
    // cuda::std::chrono::floor<days>: the day count is a 32-bit `int` (truncating division, then
    // one step back when the result overshoots), so the sentinel timestamps wrap exactly as in
    // the cuDF-based evaluators; finite values get the ordinary floor.
    << "    int32_t day32 = static_cast<int32_t>(ticks / ticks_per_day);\n"
    << "    if (static_cast<int64_t>(day32) * ticks_per_day > ticks) {\n"
    << "      day32 = static_cast<int32_t>(static_cast<uint32_t>(day32) - 1u);\n"
    << "    }\n"
    << "    if constexpr (Comp <= 2) {\n"
    << "      int16_t y = 0, m = 0, d = 0;\n"
    << "      civil_from_days(day32, y, m, d);\n"
    << "      int16_t const res = (Comp == 0) ? y : ((Comp == 1) ? m : d);\n"
    << "      return reg_val<OutT>{static_cast<OutT>(res), true, false};\n"
    << "    } else {\n"
    // cuDF's extract_component_operator: the time since midnight is moved forward by one day
    // when the wrapped day count leaves it negative, then split with truncating duration_casts.
    << "      int64_t tsm = ticks - static_cast<int64_t>(day32) * ticks_per_day;\n"
    << "      if (tsm < 0) { tsm += ticks_per_day; }\n"
    << "      int64_t const hrs = tsm / (3600LL * TicksPerSec);\n"
    << "      if constexpr (Comp == 3) {\n"
    << "        return reg_val<OutT>{static_cast<OutT>(static_cast<int16_t>(hrs)), true, false};\n"
    << "      } else if constexpr (Comp == 4) {\n"
    << "        return reg_val<OutT>{static_cast<OutT>(static_cast<int16_t>(tsm / (60LL * "
       "TicksPerSec) - hrs * 60LL)), true, false};\n"
    << "      } else {\n"
    << "        int64_t const mins = tsm / (60LL * TicksPerSec) - hrs * 60LL;\n"
    << "        return reg_val<OutT>{static_cast<OutT>(static_cast<int16_t>(tsm / TicksPerSec - "
       "hrs * 3600LL - mins * 60LL)), true, false};\n"
    << "      }\n"
    << "    }\n"
    << "  } else {\n"
    << "    int64_t rem = ticks % ticks_per_day;\n"
    << "    if (rem < 0) { rem += ticks_per_day; }\n"
    << "    int64_t const sec_of_day = rem / TicksPerSec;\n"
    << "    int64_t const subsec     = rem % TicksPerSec;\n"
    << "    {\n"
    << "      int64_t const sec = static_cast<int64_t>(static_cast<int16_t>(sec_of_day % 60LL));\n"
    << "      int64_t ms_part = 0LL;\n"
    << "      if constexpr (TicksPerSec == 1000LL) {\n"
    << "        ms_part = static_cast<int64_t>(static_cast<int16_t>(subsec));\n"
    << "      } else if constexpr (TicksPerSec == 1000000LL) {\n"
    << "        ms_part = static_cast<int64_t>(static_cast<int16_t>(subsec / 1000LL));\n"
    << "      } else if constexpr (TicksPerSec == 1000000000LL) {\n"
    << "        ms_part = static_cast<int64_t>(static_cast<int16_t>(subsec / 1000000LL));\n"
    << "      }\n"
    << "      int64_t const ms_total = sec * 1000LL + ms_part;\n"
    << "      if constexpr (Comp == 6) {\n"
    << "        return reg_val<OutT>{static_cast<OutT>(ms_total), true, false};\n"
    << "      } else {\n"
    << "        int64_t us_part = 0LL;\n"
    << "        if constexpr (TicksPerSec == 1000000LL) {\n"
    << "          us_part = static_cast<int64_t>(static_cast<int16_t>(subsec % 1000LL));\n"
    << "        } else if constexpr (TicksPerSec == 1000000000LL) {\n"
    << "          us_part = static_cast<int64_t>(static_cast<int16_t>((subsec / 1000LL) % "
       "1000LL));\n"
    << "        }\n"
    << "        return reg_val<OutT>{static_cast<OutT>(ms_total * 1000LL + us_part), true, "
       "false};\n"
    << "      }\n"
    << "    }\n"
    << "  }\n"
    << "}\n\n"
    << "template <int64_t Divisor>\n"
    << "__device__ __forceinline__ reg_val<int64_t> eval_date_trunc(reg_val<int64_t> in) {\n"
    << "  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }\n"
    << "  if constexpr (Divisor <= 1LL) {\n"
    << "    return in;\n"
    << "  } else {\n"
    << "    int64_t const t = in.val;\n"
    << "    int64_t q = t / Divisor;\n"
    << "    int64_t const r = t % Divisor;\n"
    << "    if (r < 0) { q -= 1LL; }\n"
    << "    return reg_val<int64_t>{q * Divisor, true, false};\n"
    << "  }\n"
    << "}\n\n";
}

// The device runtime preamble is identical for every kernel; render it once per process instead of
// streaming its several hundred fragments on every evaluation.
std::string const& nvrtc_device_runtime_preamble()
{
  static std::string const preamble = [] {
    std::ostringstream src;
    emit_nvrtc_device_runtime_preamble_impl(src);
    return src.str();
  }();
  return preamble;
}

// `cudaGetDeviceProperties` is slow enough to dominate small-batch evaluations; the compute
// capability of a device never changes, so resolve it once per device.
int cached_arch_cc_for_current_device()
{
  int device = 0;
  simpatico::throw_if_cuda_error(cudaGetDevice(&device), "cudaGetDevice(arch_cc)");
  static std::mutex mutex;
  static std::unordered_map<int, int> arch_by_device;
  std::lock_guard<std::mutex> const lock(mutex);
  auto const it = arch_by_device.find(device);
  if (it != arch_by_device.end()) { return it->second; }
  int const arch = codegen::jit::arch_cc_for_current_device();
  arch_by_device.emplace(device, arch);
  return arch;
}

// Debug aid: when `SIRIUS_NVRTC_DUMP_DIR` names a directory, every generated kernel source is
// written there as `<seq>_<entry>.cu` so it can be inspected or compiled offline (e.g. with
// `nvcc --resource-usage`). Resolved once per process; unset disables the dump.
void maybe_dump_kernel_source(std::string const& entry, std::string const& source)
{
  static std::string const dir = []() -> std::string {
    char const* env = std::getenv("SIRIUS_NVRTC_DUMP_DIR");
    return (env != nullptr && *env != '\0') ? std::string(env) : std::string{};
  }();
  if (dir.empty()) { return; }
  static std::atomic<unsigned> seq{0};
  auto const path = dir + "/" + std::to_string(seq.fetch_add(1)) + "_" + entry + ".cu";
  std::ofstream out(path, std::ios::binary);
  out << source;
}

// Thread-local pinned host slab for an evaluation's small staging copies: status and null-count
// words read back from the device, and string constants copied to it. A pinned buffer keeps
// both copies asynchronous. A queued host-to-device copy is tracked with an event, so the slab
// is never rewritten by the host before the copy has read it.
class pinned_host_slab {
 public:
  pinned_host_slab() = default;
  ~pinned_host_slab()
  {
    if (_event != nullptr) {
      if (_pending) { (void)cudaEventSynchronize(_event); }
      (void)cudaEventDestroy(_event);
    }
    if (_data != nullptr) { (void)cudaFreeHost(_data); }
  }
  pinned_host_slab(pinned_host_slab const&)            = delete;
  pinned_host_slab& operator=(pinned_host_slab const&) = delete;

  // Returns at least `bytes` of pinned memory that no queued copy still reads.
  [[nodiscard]] void* reserve(std::size_t bytes)
  {
    if (_pending) {
      simpatico::throw_if_cuda_error(cudaEventSynchronize(_event),
                                     "cudaEventSynchronize(pinned_host_slab)");
      _pending = false;
    }
    if (_data != nullptr && bytes <= _capacity_bytes) { return _data; }
    std::size_t const grown = std::max({bytes, _capacity_bytes * 2U, std::size_t{256}});
    void* fresh             = nullptr;
    simpatico::throw_if_cuda_error(cudaHostAlloc(&fresh, grown, cudaHostAllocPortable),
                                   "cudaHostAlloc(pinned_host_slab)");
    if (_data != nullptr) { (void)cudaFreeHost(_data); }
    _data           = fresh;
    _capacity_bytes = grown;
    return _data;
  }

  // Records that a copy reading the slab was queued on `stream`; the next reserve() waits for it.
  void mark_pending(::cuda::stream_ref stream)
  {
    if (_event == nullptr) {
      simpatico::throw_if_cuda_error(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming),
                                     "cudaEventCreate(pinned_host_slab)");
    }
    simpatico::throw_if_cuda_error(cudaEventRecord(_event, stream.get()),
                                   "cudaEventRecord(pinned_host_slab)");
    _pending = true;
  }

 private:
  void* _data{nullptr};
  std::size_t _capacity_bytes{0};
  cudaEvent_t _event{nullptr};
  bool _pending{false};
};

[[nodiscard]] std::uint32_t const* copy_status_words_from_device(std::uint32_t const* d_status,
                                                                 std::size_t words,
                                                                 ::cuda::stream_ref stream)
{
  thread_local pinned_host_slab slab;
  auto* const h_pinned = static_cast<std::uint32_t*>(slab.reserve(words * sizeof(std::uint32_t)));
  simpatico::throw_if_cuda_error(
    cudaMemcpyAsync(
      h_pinned, d_status, words * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream.get()),
    "cudaMemcpyAsync(status_words)");
  simpatico::throw_if_cuda_error(cudaStreamSynchronize(stream.get()),
                                 "cudaStreamSynchronize(status_words)");
  if ((h_pinned[0] & 2U) != 0U) {
    throw internal_exception(
      "[expression_evaluator:case]: More than one row returned by a subquery used as an "
      "expression.");
  }
  if ((h_pinned[0] & 1U) != 0U) {
    throw out_of_range_exception("Overflow in NVRTC expression evaluation");
  }
  return h_pinned;
}

void emit_input_and_constant_kernel_params(std::ostringstream& src, translation_context const& ctx)
{
  for (auto const& in_col : ctx.input_columns()) {
    auto const id_str = std::to_string(in_col.binding_id);
    if (in_col.physical_type.id() == cudf::type_id::STRING) {
      auto const off_type_str = cuda_type_name(in_col.offset_type);
      src << "    " << off_type_str << " const* __restrict__ in_" << id_str << "_offsets,\n"
          << "    char const* __restrict__ in_" << id_str << "_chars,\n"
          << "    char const* in_" << id_str << "_chars_end,\n"
          << "    uint32_t const* __restrict__ in_" << id_str << "_mask,\n"
          << "    int32_t in_" << id_str << "_bit_offset,\n";
    } else {
      auto const type_str = cuda_type_name(in_col.physical_type);
      src << "    " << type_str << " const* __restrict__ in_" << id_str << "_data,\n"
          << "    uint32_t const* __restrict__ in_" << id_str << "_mask,\n"
          << "    int32_t in_" << id_str << "_bit_offset,\n";
    }
  }

  for (auto const& pat : ctx.like_patterns()) {
    src << "    like_pattern_desc lp_" << pat.binding_id << ",\n";
  }

  for (auto const& cst : ctx.constants()) {
    if (!cst.is_valid) { continue; }  // NULL constants are baked into the source
    auto const id_str = std::to_string(cst.binding_id);
    if (cst.device_type.id() == cudf::type_id::STRING) {
      src << "    char const* __restrict__ c_" << id_str << "_ptr,\n"
          << "    int32_t c_" << id_str << "_len,\n";
    } else {
      auto const type_str = cuda_type_name(cst.device_type);
      src << "    " << type_str << " c_" << id_str << "_val,\n";
    }
  }
}

struct input_arg_storage {
  void const* data_ptr{nullptr};
  void const* offsets_ptr{nullptr};
  char const* chars_ptr{nullptr};
  char const* chars_end{nullptr};
  std::uint32_t const* mask_ptr{nullptr};
  std::int32_t bit_offset{0};
};

struct string_constant_arg_storage {
  char const* ptr{nullptr};
  std::int32_t len{0};
  std::uint8_t valid{0};
};

struct bound_kernel_input_and_constant_args {
  std::vector<input_arg_storage> in_args;
  rmm::device_uvector<char> d_const_chars;
  std::vector<constant_param_storage> const_args;
  std::vector<string_constant_arg_storage> str_const_args;

  bound_kernel_input_and_constant_args(translation_context const& ctx,
                                       cudf::table_view const& input,
                                       ::cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
    : in_args(ctx.input_columns().size()),
      d_const_chars(0, stream, mr),
      const_args(ctx.constants().size()),
      str_const_args(ctx.constants().size())
  {
    for (std::size_t i = 0; i < ctx.input_columns().size(); ++i) {
      auto const& binding = ctx.input_columns()[i];
      auto const col_view = input.column(binding.column_index);
      if (binding.physical_type.id() == cudf::type_id::STRING) {
        if (col_view.size() > 0 && col_view.num_children() > 0) {
          cudf::strings_column_view const scv(col_view);
          auto const offsets_col = scv.offsets();
          if (offsets_col.size() > col_view.offset() + col_view.size()) {
            if (offsets_col.type().id() == cudf::type_id::INT64) {
              in_args[i].offsets_ptr = offsets_col.data<std::int64_t>() + col_view.offset();
            } else {
              in_args[i].offsets_ptr = offsets_col.data<std::int32_t>() + col_view.offset();
            }
            in_args[i].chars_ptr = scv.chars_begin(stream);
            in_args[i].chars_end = in_args[i].chars_ptr + scv.chars_size(stream);
          }
        }
      } else {
        in_args[i].data_ptr = get_input_data_ptr(col_view);
      }
      in_args[i].mask_ptr   = binding.is_nullable ? col_view.null_mask() : nullptr;
      in_args[i].bit_offset = static_cast<std::int32_t>(col_view.offset());
    }

    std::string h_const_chars;
    std::vector<std::size_t> str_const_offsets(ctx.constants().size(), 0);
    for (std::size_t i = 0; i < ctx.constants().size(); ++i) {
      auto const& cst = ctx.constants()[i];
      if (cst.device_type.id() == cudf::type_id::STRING && cst.is_valid) {
        auto const& s        = std::get<std::string>(cst.payload);
        str_const_offsets[i] = h_const_chars.size();
        h_const_chars.append(s);
      }
    }
    if (!h_const_chars.empty()) {
      d_const_chars.resize(h_const_chars.size(), stream);
      // A copy from pageable memory blocks the host while the driver stages it; the pinned slab
      // keeps it asynchronous and waits for its previous copy before being rewritten.
      thread_local pinned_host_slab staging;
      void* const h_staging = staging.reserve(h_const_chars.size());
      std::memcpy(h_staging, h_const_chars.data(), h_const_chars.size());
      simpatico::throw_if_cuda_error(cudaMemcpyAsync(d_const_chars.data(),
                                                     h_staging,
                                                     h_const_chars.size(),
                                                     cudaMemcpyHostToDevice,
                                                     stream.get()),
                                     "cudaMemcpyAsync(d_const_chars)");
      staging.mark_pending(stream);
    }

    for (std::size_t i = 0; i < ctx.constants().size(); ++i) {
      auto const& cst = ctx.constants()[i];
      if (cst.device_type.id() == cudf::type_id::STRING) {
        str_const_args[i].valid = cst.is_valid ? 1U : 0U;
        if (cst.is_valid) {
          auto const& s         = std::get<std::string>(cst.payload);
          str_const_args[i].ptr = d_const_chars.data() + str_const_offsets[i];
          str_const_args[i].len = static_cast<std::int32_t>(s.size());
        }
      } else {
        populate_constant_param(cst, const_args[i]);
      }
    }
  }

  void append_to_kernel_args(translation_context const& ctx, std::vector<void*>& kernel_args)
  {
    static_assert(sizeof(sirius::detail::like_multiliteral_literal_desc) == 88 &&
                    sizeof(sirius::detail::like_multiliteral_pattern_desc) == 360,
                  "like_pattern_desc in the device runtime mirrors this layout");
    for (std::size_t i = 0; i < in_args.size(); ++i) {
      if (ctx.input_columns()[i].physical_type.id() == cudf::type_id::STRING) {
        kernel_args.push_back(&in_args[i].offsets_ptr);
        kernel_args.push_back(&in_args[i].chars_ptr);
        kernel_args.push_back(&in_args[i].chars_end);
        kernel_args.push_back(&in_args[i].mask_ptr);
        kernel_args.push_back(&in_args[i].bit_offset);
      } else {
        kernel_args.push_back(&in_args[i].data_ptr);
        kernel_args.push_back(&in_args[i].mask_ptr);
        kernel_args.push_back(&in_args[i].bit_offset);
      }
    }
    for (auto const& pat : ctx.like_patterns()) {
      kernel_args.push_back(const_cast<void*>(static_cast<void const*>(&pat.descriptor)));
    }
    for (std::size_t i = 0; i < ctx.constants().size(); ++i) {
      if (!ctx.constants()[i].is_valid) { continue; }
      if (ctx.constants()[i].device_type.id() == cudf::type_id::STRING) {
        kernel_args.push_back(&str_const_args[i].ptr);
        kernel_args.push_back(&str_const_args[i].len);
      } else {
        kernel_args.push_back(&const_args[i].val_bytes[0]);
      }
    }
  }
};

}  // namespace

std::unique_ptr<cudf::table> evaluate_project_sink(
  std::span<sirius::ast::node const* const> expressions,
  cudf::table_view const& input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::size_t* narrow_domain_comparison_count,
  boundary_column_map const* boundary_map)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) {
    if (narrow_domain_comparison_count != nullptr) {
      translation_context count_ctx(input, boundary_map);
      for (auto const* expr : expressions) {
        (void)count_ctx.produce(*expr);
      }
      *narrow_domain_comparison_count += count_ctx.narrow_domain_comparison_count();
    }
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    empty_cols.reserve(expressions.size());
    for (auto const* expr : expressions) {
      auto const out_type = resolve_output_cudf_type(*expr, input);
      empty_cols.push_back(make_unallocated_fixed_width_output_column(out_type, 0, stream, mr));
    }
    return std::make_unique<cudf::table>(std::move(empty_cols));
  }

  translation_context ctx(input, boundary_map);
  std::vector<register_value> out_regs;
  out_regs.reserve(expressions.size());
  for (auto const* expr : expressions) {
    out_regs.push_back(ctx.produce(*expr));
  }
  if (narrow_domain_comparison_count != nullptr) {
    *narrow_domain_comparison_count += ctx.narrow_domain_comparison_count();
  }

  constexpr int block_threads    = 256;
  constexpr int num_warps        = block_threads / 32;
  constexpr int items_per_thread = 4;
  constexpr int tile_rows        = block_threads * items_per_thread;
  bool any_nullable_output       = false;
  bool any_can_overflow          = false;
  bool const any_subquery_err    = ctx.has_subquery_error();
  for (auto const& reg : out_regs) {
    if (!reg.is_statically_non_null) { any_nullable_output = true; }
    if (reg.can_overflow) { any_can_overflow = true; }
  }

  // Assemble the CUDA kernel source string.
  std::ostringstream src;
  src << nvrtc_device_runtime_preamble();
  src << "extern \"C\" __global__ void sirius_nvrtc_project_kernel(\n";
  emit_input_and_constant_kernel_params(src, ctx);

  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    auto const k_str    = std::to_string(k);
    auto const type_str = cuda_type_name(out_regs[k].device_type);
    src << "    " << type_str << "* __restrict__ out_" << k_str << "_data,\n"
        << "    uint32_t* __restrict__ out_" << k_str << "_mask,\n";
  }

  src << "    int32_t num_rows,\n"
      << "    uint32_t* __restrict__ status_and_null_counts)\n"
      << "{\n"
      << "  int32_t const tile_row0 = static_cast<int32_t>(blockIdx.x) * " << tile_rows << ";\n"
      << "  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;\n"
      << "  int32_t const warp_in_block = static_cast<int32_t>(threadIdx.x) >> 5;\n";

  // Per-warp null counters live in lane 0's registers across the tile's items and are reduced
  // once per block, so the block issues at most one atomic per nullable output.
  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    if (!out_regs[k].is_statically_non_null) { src << "  uint32_t warp_nulls_" << k << " = 0u;\n"; }
  }
  if (any_can_overflow) { src << "  bool row_overflow = false;\n"; }
  if (any_subquery_err) { src << "  bool row_subquery_error = false;\n"; }

  // Each thread evaluates `items_per_thread` rows, strided by the block width so that every
  // item's loads and stores stay coalesced across the warp and the unrolled items give the
  // memory system several independent requests per thread. A full tile (every block but
  // possibly the last) runs the items without the per-row bounds guard, which lets the compiler
  // batch the loads of all items ahead of the arithmetic.
  auto const emit_items = [&](bool guarded) {
    std::string const indent = guarded ? "      " : "    ";
    src << "  #pragma unroll\n"
        << "  for (int item = 0; item < " << items_per_thread << "; ++item) {\n"
        << "    int32_t const row = tile_row0 + item * " << block_threads
        << " + static_cast<int32_t>(threadIdx.x);\n";
    for (std::size_t k = 0; k < out_regs.size(); ++k) {
      if (!out_regs[k].is_statically_non_null) {
        src << "    bool out_" << k << "_valid = false;\n";
      }
    }
    if (guarded) { src << "    if (row < num_rows) {\n"; }
    src << ctx.body_code();
    for (std::size_t k = 0; k < out_regs.size(); ++k) {
      auto const& reg = out_regs[k];
      if (reg.is_statically_non_null) {
        src << indent << "out_" << k << "_data[row] = " << reg.var_name << ".val;\n";
      } else {
        src << indent << "if (" << reg.var_name << ".valid) {\n"
            << indent << "  out_" << k << "_data[row] = " << reg.var_name << ".val;\n"
            << indent << "}\n"
            << indent << "out_" << k << "_valid = " << reg.var_name << ".valid;\n";
      }
      if (reg.can_overflow) {
        src << indent << "row_overflow = row_overflow || (" << reg.var_name << ".valid && "
            << reg.var_name << ".overflow);\n";
      }
    }
    if (guarded) { src << "    }\n"; }

    if (any_nullable_output) {
      // The warp's 32 rows map to one validity bitmask word; a partial last warp masks off the
      // lanes past `num_rows` and counts only its live rows.
      src << "    int32_t const warp_base_row = row - lane;\n";
      if (guarded) {
        src << "    int32_t const rem = num_rows - warp_base_row;\n"
            << "    int32_t const active_in_warp = rem <= 0 ? 0 : (rem < 32 ? rem : 32);\n"
            << "    uint32_t const valid_row_mask = (active_in_warp == 32) ? 0xFFFFFFFFu : "
               "((active_in_warp <= 0) ? 0u : ((1u << active_in_warp) - 1u));\n";
      } else {
        src << "    constexpr int32_t active_in_warp = 32;\n"
            << "    constexpr uint32_t valid_row_mask = 0xFFFFFFFFu;\n";
      }
      for (std::size_t k = 0; k < out_regs.size(); ++k) {
        if (out_regs[k].is_statically_non_null) { continue; }
        src << "    uint32_t const ballot_" << k << " = __ballot_sync(0xFFFFFFFFu, out_" << k
            << "_valid) & valid_row_mask;\n"
            << "    if (lane == 0 && active_in_warp > 0) {\n"
            << "      out_" << k << "_mask[warp_base_row >> 5] = ballot_" << k << ";\n"
            << "      warp_nulls_" << k
            << " += static_cast<uint32_t>(active_in_warp - __popc(ballot_" << k << "));\n"
            << "    }\n";
      }
    }
    src << "  }\n";
  };
  // Written as a difference: `tile_row0 + tile_rows` overflows int32 for a batch within one
  // tile of INT32_MAX rows, `num_rows - tile_row0` never does (tile_row0 < num_rows).
  src << "  if (num_rows - tile_row0 >= " << tile_rows << ") {\n";
  emit_items(/*guarded=*/false);
  src << "  } else {\n";
  emit_items(/*guarded=*/true);
  src << "  }\n";

  if (any_can_overflow) {
    src << "  uint32_t const ovf_ballot = __ballot_sync(0xFFFFFFFFu, row_overflow);\n"
        << "  if (ovf_ballot != 0u && lane == 0) {\n"
        << "    atomicOr(&status_and_null_counts[0], 1u);\n"
        << "  }\n";
  }
  if (any_subquery_err) {
    src << "  uint32_t const subq_ballot = __ballot_sync(0xFFFFFFFFu, row_subquery_error);\n"
        << "  if (subq_ballot != 0u && lane == 0) {\n"
        << "    atomicOr(&status_and_null_counts[0], 2u);\n"
        << "  }\n";
  }

  if (any_nullable_output) {
    for (std::size_t k = 0; k < out_regs.size(); ++k) {
      if (out_regs[k].is_statically_non_null) { continue; }
      src << "  __shared__ uint32_t s_nulls_" << k << "[" << num_warps << "];\n"
          << "  if (lane == 0) { s_nulls_" << k << "[warp_in_block] = warp_nulls_" << k << "; }\n";
    }
    src << "  __syncthreads();\n"
        << "  if (threadIdx.x == 0) {\n";
    for (std::size_t k = 0; k < out_regs.size(); ++k) {
      if (out_regs[k].is_statically_non_null) { continue; }
      src << "    uint32_t block_nulls_" << k << " = 0u;\n"
          << "    #pragma unroll\n"
          << "    for (int w = 0; w < " << num_warps << "; ++w) {\n"
          << "      block_nulls_" << k << " += s_nulls_" << k << "[w];\n"
          << "    }\n"
          << "    if (block_nulls_" << k << " > 0u) {\n"
          << "      atomicAdd(&status_and_null_counts[" << (1 + k) << "], block_nulls_" << k
          << ");\n"
          << "    }\n";
    }
    src << "  }\n";
  }

  src << "}\n";

  codegen::jit::CompileOptions opts;
  opts.arch_cc       = cached_arch_cc_for_current_device();
  opts.device_int128 = true;
  std::shared_ptr<const codegen::jit::CompiledKernel> compiled;
  try {
    auto const source = src.str();
    maybe_dump_kernel_source("sirius_nvrtc_project_kernel", source);
    compiled = codegen::jit::KernelCache::instance().get_or_compile_plain(
      source, "sirius_nvrtc_project_kernel", opts);
  } catch (codegen::jit::CompileError const& err) {
    throw internal_exception("[nvrtc] {}\nLog:\n{}\nSource:\n{}", err.what(), err.log, err.source);
  }
  CUfunction const fn = compiled->func_for_current_device();

  // Allocate output columns and bitmasks.
  struct output_column_staging {
    cudf::data_type type;
    rmm::device_buffer null_mask;
    std::unique_ptr<cudf::column> col;
    void* data_ptr{nullptr};
    std::uint32_t* mask_ptr{nullptr};
  };
  std::vector<output_column_staging> outputs;
  outputs.reserve(out_regs.size());

  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    auto const& reg = out_regs[k];
    output_column_staging stg{reg.device_type, rmm::device_buffer{0, stream, mr}, nullptr};
    if (reg.is_statically_non_null) {
      stg.col = make_unallocated_fixed_width_output_column(reg.device_type, num_rows, stream, mr);
      auto mview   = stg.col->mutable_view();
      stg.data_ptr = get_output_data_ptr(mview);
      stg.mask_ptr = nullptr;
    } else {
      stg.null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
      stg.mask_ptr  = static_cast<std::uint32_t*>(stg.null_mask.data());
      stg.col = make_unallocated_fixed_width_output_column(reg.device_type, num_rows, stream, mr);
      auto mview   = stg.col->mutable_view();
      stg.data_ptr = get_output_data_ptr(mview);
    }
    outputs.push_back(std::move(stg));
  }

  // Allocate status and null counts buffer if needed: [0] = status bitmask, [1..K] = null counts.
  bool const need_status_buffer  = any_nullable_output || any_can_overflow || any_subquery_err;
  std::size_t const status_words = 1 + out_regs.size();
  rmm::device_uvector<std::uint32_t> d_status(need_status_buffer ? status_words : 0, stream, mr);
  std::uint32_t* status_ptr = need_status_buffer ? d_status.data() : nullptr;
  if (need_status_buffer) {
    simpatico::throw_if_cuda_error(
      cudaMemsetAsync(status_ptr, 0, status_words * sizeof(std::uint32_t), stream.get()),
      "cudaMemsetAsync(status_and_null_counts)");
  }

  bound_kernel_input_and_constant_args bound_args(ctx, input, stream, mr);
  std::int32_t num_rows_arg = static_cast<std::int32_t>(num_rows);

  std::vector<void*> kernel_args;
  kernel_args.reserve(bound_args.in_args.size() * 5 + bound_args.const_args.size() * 3 +
                      outputs.size() * 2 + 2);
  bound_args.append_to_kernel_args(ctx, kernel_args);
  for (auto& out_arg : outputs) {
    kernel_args.push_back(&out_arg.data_ptr);
    kernel_args.push_back(&out_arg.mask_ptr);
  }
  kernel_args.push_back(&num_rows_arg);
  kernel_args.push_back(&status_ptr);

  auto const grid_dim_x =
    static_cast<unsigned int>(num_rows / tile_rows + (num_rows % tile_rows != 0 ? 1 : 0));
  simpatico::throw_if_cu_error(cuLaunchKernel(fn,
                                              grid_dim_x,
                                              1,
                                              1,
                                              block_threads,
                                              1,
                                              1,
                                              0,
                                              reinterpret_cast<CUstream>(stream.get()),
                                              kernel_args.data(),
                                              nullptr),
                               "cuLaunchKernel(sirius_nvrtc_project_kernel)");

  std::uint32_t const* h_status =
    need_status_buffer ? copy_status_words_from_device(status_ptr, status_words, stream) : nullptr;

  std::vector<std::unique_ptr<cudf::column>> final_columns;
  final_columns.reserve(outputs.size());
  for (std::size_t k = 0; k < outputs.size(); ++k) {
    auto& stg = outputs[k];
    if (!out_regs[k].is_statically_non_null) {
      auto const null_count = static_cast<cudf::size_type>(h_status[1 + k]);
      if (null_count > 0) { stg.col->set_null_mask(std::move(stg.null_mask), null_count); }
    }
    final_columns.push_back(std::move(stg.col));
  }

  return std::make_unique<cudf::table>(std::move(final_columns));
}

bool can_compile_filter_sink(sirius::ast::node const& predicate,
                             cudf::table_view const& input,
                             boundary_column_map const* boundary_map) noexcept
{
  if (!is_node_supported(predicate, input, boundary_map)) { return false; }
  return effective_node_cudf_type(predicate, input, boundary_map).id() == cudf::type_id::BOOL8;
}

namespace {

// Fixed (expression-independent) kernels completing the `FilterSink` compaction for multi-tile
// batches: an exclusive scan over the per-tile survivor counts, and a warp-cooperative scatter of
// the surviving row indices from the per-warp ballot words. Compiled once per process.
constexpr int kFilterBlockThreads   = 256;
constexpr int kFilterItemsPerThread = 16;
constexpr int kFilterTileRows       = kFilterBlockThreads * kFilterItemsPerThread;
constexpr int kFilterScanThreads    = 1024;

constexpr char const* kFilterCompactionSource = R"(
#include "codegen/stdint_shim.hpp"

// One block: exclusive prefix sum of tile_counts in place; status_and_counts[1] = total.
extern "C" __global__ void sirius_nvrtc_filter_scan_kernel(
    uint32_t* __restrict__ tile_counts,
    int32_t num_tiles,
    uint32_t* __restrict__ status_and_counts)
{
  constexpr int kThreads = 1024;
  __shared__ uint32_t s_warp_sums[kThreads / 32];
  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
  int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
  uint32_t carry = 0u;
  for (int32_t base = 0; base < num_tiles; base += kThreads) {
    int32_t const idx = base + static_cast<int32_t>(threadIdx.x);
    uint32_t const v = (idx < num_tiles) ? tile_counts[idx] : 0u;
    uint32_t incl = v;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      uint32_t const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);
      if (lane >= off) { incl += n; }
    }
    if (lane == 31) { s_warp_sums[warp] = incl; }
    __syncthreads();
    uint32_t warp_prefix = 0u;
    #pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) {
      if (w < warp) { warp_prefix += s_warp_sums[w]; }
    }
    uint32_t chunk_total = 0u;
    #pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) {
      chunk_total += s_warp_sums[w];
    }
    if (idx < num_tiles) { tile_counts[idx] = carry + warp_prefix + incl - v; }
    carry += chunk_total;
    __syncthreads();
  }
  if (threadIdx.x == 0) { status_and_counts[1] = carry; }
}

// One block per tile: each warp walks its share of the tile's 32-row ballot words and writes the
// surviving row indices contiguously (lane rank within the word, word rank within the tile,
// tile prefix from the scan), so the output is ascending and every store instruction is coalesced.
extern "C" __global__ void sirius_nvrtc_filter_scatter_kernel(
    uint32_t const* __restrict__ ballots,
    uint32_t const* __restrict__ tile_prefix,
    int32_t num_rows,
    int32_t* __restrict__ survivors)
{
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  constexpr int kWordsPerTile = kThreads * 16 / 32;
  constexpr int kWordsPerWarp = kWordsPerTile / kWarps;
  __shared__ uint32_t s_word_prefix[kWordsPerTile];
  __shared__ uint32_t s_warp_sums[kWarps];
  int32_t const tile = static_cast<int32_t>(blockIdx.x);
  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
  int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
  int32_t const num_words = (num_rows >> 5) + ((num_rows & 31) != 0 ? 1 : 0);
  int32_t const word0 = tile * kWordsPerTile;

  // Popcount of every ballot word of the tile (kWordsPerTile <= kThreads: one word per thread).
  static_assert(kWordsPerTile <= kThreads, "one ballot word per thread");
  if (threadIdx.x < kWordsPerTile) {
    int32_t const gw = word0 + static_cast<int32_t>(threadIdx.x);
    s_word_prefix[threadIdx.x] = (gw < num_words) ? static_cast<uint32_t>(__popc(ballots[gw])) : 0u;
  }
  __syncthreads();
  // Each warp turns its contiguous run of kWordsPerWarp counts into exclusive offsets. Every lane
  // takes part in each shuffle round (idle lanes contribute zero) so the warp never diverges.
  uint32_t run_total = 0u;
  for (int k = 0; k < kWordsPerWarp; k += 32) {
    bool const in_run = (k + lane) < kWordsPerWarp;
    int32_t const wi = warp * kWordsPerWarp + k + lane;
    uint32_t const c = in_run ? s_word_prefix[wi] : 0u;
    uint32_t incl = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      uint32_t const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);
      if (lane >= off) { incl += n; }
    }
    if (in_run) { s_word_prefix[wi] = run_total + incl - c; }
    run_total += __shfl_sync(0xFFFFFFFFu, incl, 31);
  }
  if (lane == 0) { s_warp_sums[warp] = run_total; }
  __syncthreads();
  uint32_t warp_base = tile_prefix[tile];
  for (int w = 0; w < warp; ++w) {
    warp_base += s_warp_sums[w];
  }
  for (int32_t wi = warp * kWordsPerWarp; wi < (warp + 1) * kWordsPerWarp; ++wi) {
    int32_t const gw = word0 + wi;
    if (gw >= num_words) { break; }
    uint32_t const ballot = ballots[gw];
    if (ballot == 0u) { continue; }
    uint32_t const base = warp_base + s_word_prefix[wi];
    if ((ballot >> lane) & 1u) {
      uint32_t const rank = static_cast<uint32_t>(__popc(ballot & ((1u << lane) - 1u)));
      survivors[base + rank] = (gw << 5) + lane;
    }
  }
}
)";

[[nodiscard]] std::unique_ptr<cudf::table> make_empty_projected_table(
  cudf::table_view const& projected_input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (projected_input.num_columns() == 0) {
    return std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  }
  auto sliced = cudf::slice(projected_input, {0, 0}, stream);
  return std::make_unique<cudf::table>(sliced.front(), stream, mr);
}

}  // namespace

std::unique_ptr<cudf::table> evaluate_filter_sink(sirius::ast::node const& predicate,
                                                  cudf::table_view const& input,
                                                  cudf::table_view const& projected_input,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr,
                                                  std::unique_ptr<cudf::column>* out_survivors,
                                                  std::size_t* narrow_domain_comparison_count,
                                                  boundary_column_map const* boundary_map)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) {
    if (narrow_domain_comparison_count != nullptr) {
      translation_context count_ctx(input, boundary_map);
      (void)count_ctx.produce(predicate);
      *narrow_domain_comparison_count += count_ctx.narrow_domain_comparison_count();
    }
    if (out_survivors != nullptr) {
      *out_survivors = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
    }
    return make_empty_projected_table(projected_input, stream, mr);
  }

  translation_context ctx(input, boundary_map);
  auto const pred_reg = ctx.produce(predicate);
  if (narrow_domain_comparison_count != nullptr) {
    *narrow_domain_comparison_count += ctx.narrow_domain_comparison_count();
  }

  constexpr int block_threads    = kFilterBlockThreads;
  constexpr int num_warps        = block_threads / 32;
  constexpr int items_per_thread = kFilterItemsPerThread;
  constexpr int tile_rows        = kFilterTileRows;
  bool const any_can_overflow    = pred_reg.can_overflow;
  bool const any_subquery_err    = ctx.has_subquery_error();

  std::ostringstream src;
  src << nvrtc_device_runtime_preamble();
  src << "extern \"C\" __global__ void sirius_nvrtc_filter_kernel(\n";
  emit_input_and_constant_kernel_params(src, ctx);

  // status_and_counts: [0] = status flags, [1] = total survivors (written here only when the
  // batch is a single tile). `survivors` is written directly for a single tile; otherwise the
  // per-tile count and per-warp ballot words feed the scan and scatter kernels.
  src << "    int32_t num_rows,\n"
      << "    uint32_t* __restrict__ status_and_counts,\n"
      << "    uint32_t* __restrict__ tile_counts,\n"
      << "    uint32_t* __restrict__ ballots,\n"
      << "    int32_t* __restrict__ survivors)\n"
      << "{\n"
      << "  int32_t const tile = static_cast<int32_t>(blockIdx.x);\n"
      << "  bool const single_tile = (gridDim.x == 1u);\n"
      << "  int32_t const tile_row0 = tile * " << tile_rows << ";\n"
      << "  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;\n"
      << "  int32_t const warp_in_block = static_cast<int32_t>(threadIdx.x) >> 5;\n"
      << "  __shared__ uint32_t s_counts[" << items_per_thread * num_warps << "];\n"
      << "  __shared__ uint32_t s_block_count;\n"
      << "  uint32_t survive_bits = 0u;\n";

  if (any_can_overflow) { src << "  bool row_overflow = false;\n"; }
  if (any_subquery_err) { src << "  bool row_subquery_error = false;\n"; }

  // Phase 1: evaluate the predicate for every row of the tile; each thread keeps one survive
  // bit per item and lane 0 of each warp records the per-item warp count. Full tiles skip the
  // per-row bounds guard so the items' loads can be issued together.
  auto const emit_items = [&](bool guarded) {
    std::string const indent = guarded ? "      " : "    ";
    src << "  #pragma unroll\n"
        << "  for (int item = 0; item < " << items_per_thread << "; ++item) {\n"
        << "    int32_t const row = tile_row0 + item * " << block_threads
        << " + static_cast<int32_t>(threadIdx.x);\n"
        << "    bool survives = false;\n";
    if (guarded) { src << "    if (row < num_rows) {\n"; }
    src << ctx.body_code();
    if (pred_reg.can_overflow) {
      src << indent << "row_overflow = row_overflow || (" << pred_reg.var_name << ".valid && "
          << pred_reg.var_name << ".overflow);\n"
          << indent << "survives = (" << pred_reg.var_name << ".valid && !" << pred_reg.var_name
          << ".overflow && " << pred_reg.var_name << ".val);\n";
    } else if (pred_reg.is_statically_non_null) {
      src << indent << "survives = " << pred_reg.var_name << ".val;\n";
    } else {
      src << indent << "survives = (" << pred_reg.var_name << ".valid && " << pred_reg.var_name
          << ".val);\n";
    }
    if (guarded) { src << "    }\n"; }
    src << "    uint32_t const warp_ballot = __ballot_sync(0xFFFFFFFFu, survives);\n"
        << "    survive_bits |= (survives ? 1u : 0u) << item;\n"
        << "    if (lane == 0) {\n"
        << "      s_counts[item * " << num_warps
        << " + warp_in_block] = "
           "static_cast<uint32_t>(__popc(warp_ballot));\n"
        << "      if (!single_tile" << (guarded ? " && row - lane < num_rows" : "")
        << ") { ballots[(row - lane) >> 5] = warp_ballot; }\n"
        << "    }\n"
        << "  }\n";
  };
  // Written as a difference: `tile_row0 + tile_rows` overflows int32 for a batch within one
  // tile of INT32_MAX rows, `num_rows - tile_row0` never does (tile_row0 < num_rows).
  src << "  if (num_rows - tile_row0 >= " << tile_rows << ") {\n";
  emit_items(/*guarded=*/false);
  src << "  } else {\n";
  emit_items(/*guarded=*/true);
  src << "  }\n";

  if (any_can_overflow) {
    src << "  uint32_t const ovf_ballot = __ballot_sync(0xFFFFFFFFu, row_overflow);\n"
        << "  if (ovf_ballot != 0u && lane == 0) {\n"
        << "    atomicOr(&status_and_counts[0], 1u);\n"
        << "  }\n";
  }
  if (any_subquery_err) {
    src << "  uint32_t const subq_ballot = __ballot_sync(0xFFFFFFFFu, row_subquery_error);\n"
        << "  if (subq_ballot != 0u && lane == 0) {\n"
        << "    atomicOr(&status_and_counts[0], 2u);\n"
        << "  }\n";
  }

  // Sparse-to-Dense Compaction, pass 1: warp ballot and intra-warp rank in registers and a
  // block-wide scan of the (item, warp) counts in shared memory. A single-tile batch scatters its
  // survivors here; a multi-tile batch publishes the tile count and ballot words for the scan
  // and scatter kernels, which avoids serializing tiles on a prefix chain.
  constexpr int scan_entries = items_per_thread * num_warps;
  static_assert(scan_entries % 32 == 0, "the warp scan below consumes whole 32-entry chunks");
  src
    << "  __syncthreads();\n"
    << "  if (warp_in_block == 0) {\n"
    << "    uint32_t carry = 0u;\n"
    << "    #pragma unroll\n"
    << "    for (int chunk = 0; chunk < " << (scan_entries / 32) << "; ++chunk) {\n"
    << "      uint32_t const c = s_counts[chunk * 32 + lane];\n"
    << "      uint32_t incl = c;\n"
    << "      #pragma unroll\n"
    << "      for (int off = 1; off < 32; off <<= 1) {\n"
    << "        uint32_t const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);\n"
    << "        if (lane >= off) { incl += n; }\n"
    << "      }\n"
    << "      s_counts[chunk * 32 + lane] = carry + incl - c;\n"
    << "      carry += __shfl_sync(0xFFFFFFFFu, incl, 31);\n"
    << "    }\n"
    << "    if (lane == 0) {\n"
    << "      s_block_count = carry;\n"
    << "      if (single_tile) { status_and_counts[1] = carry; } else { tile_counts[tile] = carry; "
       "}\n"
    << "    }\n"
    << "  }\n"
    << "  if (!single_tile) { return; }\n"
    << "  __syncthreads();\n"
    << "  #pragma unroll\n"
    << "  for (int item = 0; item < " << items_per_thread << "; ++item) {\n"
    << "    bool const survives = ((survive_bits >> item) & 1u) != 0u;\n"
    << "    uint32_t const warp_ballot = __ballot_sync(0xFFFFFFFFu, survives);\n"
    << "    if (survives) {\n"
    << "      uint32_t const lane_rank = static_cast<uint32_t>(__popc(warp_ballot & ((1u << lane) "
       "- 1u)));\n"
    << "      survivors[s_counts[item * " << num_warps
    << " + warp_in_block] + lane_rank] = "
       "tile_row0 + item * "
    << block_threads << " + static_cast<int32_t>(threadIdx.x);\n"
    << "    }\n"
    << "  }\n"
    << "}\n";

  codegen::jit::CompileOptions opts;
  opts.arch_cc       = cached_arch_cc_for_current_device();
  opts.device_int128 = true;
  std::shared_ptr<const codegen::jit::CompiledKernel> compiled;
  try {
    auto const source = src.str();
    maybe_dump_kernel_source("sirius_nvrtc_filter_kernel", source);
    compiled = codegen::jit::KernelCache::instance().get_or_compile_plain(
      source, "sirius_nvrtc_filter_kernel", opts);
  } catch (codegen::jit::CompileError const& err) {
    throw internal_exception("[nvrtc] {}\nLog:\n{}\nSource:\n{}", err.what(), err.log, err.source);
  }
  CUfunction const fn = compiled->func_for_current_device();

  auto const grid_dim_x =
    static_cast<unsigned int>(num_rows / tile_rows + (num_rows % tile_rows != 0 ? 1 : 0));
  bool const single_tile = (grid_dim_x == 1U);

  // Workspace: [0] = status flags, [1] = total survivors, then (multi-tile only) one count per
  // tile and one ballot word per 32 rows, both fully written by the kernels. Survivor row indices
  // are scattered into a scratch buffer with room for every row and copied into an exact-size
  // column afterwards.
  constexpr std::size_t status_words = 2;
  std::size_t const num_words        = (static_cast<std::size_t>(num_rows) + 31U) / 32U;
  std::size_t const workspace_words =
    status_words + (single_tile ? 0U : (static_cast<std::size_t>(grid_dim_x) + num_words));
  rmm::device_uvector<std::uint32_t> d_workspace(workspace_words, stream, mr);
  rmm::device_uvector<std::int32_t> d_survivors(static_cast<std::size_t>(num_rows), stream, mr);
  std::uint32_t* status_ptr = d_workspace.data();
  simpatico::throw_if_cuda_error(
    cudaMemsetAsync(status_ptr, 0, status_words * sizeof(std::uint32_t), stream.get()),
    "cudaMemsetAsync(filter_status)");
  std::uint32_t* tile_counts_ptr  = single_tile ? nullptr : (d_workspace.data() + status_words);
  std::uint32_t* ballots_ptr      = single_tile ? nullptr : (tile_counts_ptr + grid_dim_x);
  std::int32_t* survivors_scratch = d_survivors.data();

  bound_kernel_input_and_constant_args bound_args(ctx, input, stream, mr);
  std::int32_t num_rows_arg = static_cast<std::int32_t>(num_rows);

  std::vector<void*> kernel_args;
  kernel_args.reserve(bound_args.in_args.size() * 5 + bound_args.const_args.size() * 3 + 5);
  bound_args.append_to_kernel_args(ctx, kernel_args);
  kernel_args.push_back(&num_rows_arg);
  kernel_args.push_back(&status_ptr);
  kernel_args.push_back(&tile_counts_ptr);
  kernel_args.push_back(&ballots_ptr);
  kernel_args.push_back(&survivors_scratch);

  simpatico::throw_if_cu_error(cuLaunchKernel(fn,
                                              grid_dim_x,
                                              1,
                                              1,
                                              block_threads,
                                              1,
                                              1,
                                              0,
                                              reinterpret_cast<CUstream>(stream.get()),
                                              kernel_args.data(),
                                              nullptr),
                               "cuLaunchKernel(sirius_nvrtc_filter_kernel)");

  if (!single_tile) {
    auto const compaction = codegen::jit::KernelCache::instance().get_or_compile_plain(
      kFilterCompactionSource, "sirius_nvrtc_filter_scan_kernel", opts);
    auto const scatter = codegen::jit::KernelCache::instance().get_or_compile_plain(
      kFilterCompactionSource, "sirius_nvrtc_filter_scatter_kernel", opts);
    std::int32_t num_tiles_arg = static_cast<std::int32_t>(grid_dim_x);
    void* scan_args[]          = {&tile_counts_ptr, &num_tiles_arg, &status_ptr};
    simpatico::throw_if_cu_error(cuLaunchKernel(compaction->func_for_current_device(),
                                                1,
                                                1,
                                                1,
                                                kFilterScanThreads,
                                                1,
                                                1,
                                                0,
                                                reinterpret_cast<CUstream>(stream.get()),
                                                scan_args,
                                                nullptr),
                                 "cuLaunchKernel(sirius_nvrtc_filter_scan_kernel)");
    void* scatter_args[] = {&ballots_ptr, &tile_counts_ptr, &num_rows_arg, &survivors_scratch};
    simpatico::throw_if_cu_error(cuLaunchKernel(scatter->func_for_current_device(),
                                                grid_dim_x,
                                                1,
                                                1,
                                                block_threads,
                                                1,
                                                1,
                                                0,
                                                reinterpret_cast<CUstream>(stream.get()),
                                                scatter_args,
                                                nullptr),
                                 "cuLaunchKernel(sirius_nvrtc_filter_scatter_kernel)");
  }

  std::uint32_t const* const h_status = copy_status_words_from_device(status_ptr, 2U, stream);
  auto const survivor_count           = static_cast<cudf::size_type>(h_status[1]);
  if (survivor_count == 0) {
    if (out_survivors != nullptr) {
      *out_survivors = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
    }
    return make_empty_projected_table(projected_input, stream, mr);
  }

  // 100% selectivity fastpath when `out_survivors` is not requested: no gather needed!
  if (survivor_count == num_rows && out_survivors == nullptr) {
    return std::make_unique<cudf::table>(projected_input, stream, mr);
  }

  // Materialize the exact M-row ascending `INT32` survivors column.
  auto survivors_col   = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                 survivor_count,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  auto* survivors_data = survivors_col->mutable_view().data<std::int32_t>();
  simpatico::throw_if_cuda_error(
    cudaMemcpyAsync(survivors_data,
                    survivors_scratch,
                    static_cast<std::size_t>(survivor_count) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice,
                    stream.get()),
    "cudaMemcpyAsync(survivors)");

  std::unique_ptr<cudf::table> result_table;
  if (projected_input.num_columns() == 0) {
    result_table = std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  } else if (survivor_count == num_rows) {
    result_table = std::make_unique<cudf::table>(projected_input, stream, mr);
  } else {
    result_table = cudf::gather(
      projected_input, survivors_col->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  }

  if (out_survivors != nullptr) { *out_survivors = std::move(survivors_col); }
  return result_table;
}

}  // namespace sirius::nvrtc
