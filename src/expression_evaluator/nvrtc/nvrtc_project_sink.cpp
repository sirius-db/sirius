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
#include <cudf/cudf_utils.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda.h>

#include <expression/ast/constant_range.hpp>
#include <expression/function_id.hpp>
#include <expression/join_condition.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
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

[[nodiscard]] cudf::data_type effective_node_cudf_type(sirius::ast::node const& expr,
                                                       cudf::table_view const& input) noexcept
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

[[nodiscard]] bool is_node_supported(sirius::ast::node const& expr,
                                     cudf::table_view const& input) noexcept
{
  if (expr.holds<sirius::ast::reference>()) {
    auto const& ref = expr.get<sirius::ast::reference>();
    if (ref.column_index >= static_cast<std::uint32_t>(input.num_columns())) { return false; }
    auto const col_view = input.column(ref.column_index);
    if (!is_supported_fixed_width(col_view.type())) { return false; }
    auto const& logical = ref.return_type();
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
    return opt_cudf.has_value() && is_supported_fixed_width(*opt_cudf);
  }

  if (expr.holds<sirius::ast::comparison>()) {
    auto const& cmp = expr.get<sirius::ast::comparison>();
    if (!cmp.left || !cmp.right) { return false; }
    if (!is_node_supported(*cmp.left, input) || !is_node_supported(*cmp.right, input)) {
      return false;
    }
    auto const lt = effective_node_cudf_type(*cmp.left, input);
    auto const rt = effective_node_cudf_type(*cmp.right, input);
    if (!is_supported_fixed_width(lt) || !is_supported_fixed_width(rt)) { return false; }
    if (cudf::is_fixed_point(lt) || cudf::is_fixed_point(rt)) {
      return cudf::is_fixed_point(lt) && cudf::is_fixed_point(rt) && lt.scale() == rt.scale();
    }
    return true;
  }

  if (expr.holds<sirius::ast::between>()) {
    auto const& btw = expr.get<sirius::ast::between>();
    if (!btw.input || !btw.lower || !btw.upper) { return false; }
    if (!is_node_supported(*btw.input, input) || !is_node_supported(*btw.lower, input) ||
        !is_node_supported(*btw.upper, input)) {
      return false;
    }
    auto const it = effective_node_cudf_type(*btw.input, input);
    auto const lt = effective_node_cudf_type(*btw.lower, input);
    auto const ut = effective_node_cudf_type(*btw.upper, input);
    if (!is_supported_fixed_width(it) || !is_supported_fixed_width(lt) ||
        !is_supported_fixed_width(ut)) {
      return false;
    }
    if (cudf::is_fixed_point(it) || cudf::is_fixed_point(lt) || cudf::is_fixed_point(ut)) {
      return cudf::is_fixed_point(it) && cudf::is_fixed_point(lt) && cudf::is_fixed_point(ut) &&
             it.scale() == lt.scale() && it.scale() == ut.scale();
    }
    return true;
  }

  if (expr.holds<sirius::ast::unary_op>()) {
    auto const& uo = expr.get<sirius::ast::unary_op>();
    if (!uo.child || !is_node_supported(*uo.child, input)) { return false; }
    switch (uo.op) {
      case sirius::ast::unary_op::kind::op_not: {
        auto const ct = effective_node_cudf_type(*uo.child, input);
        return ct.id() == cudf::type_id::BOOL8;
      }
      case sirius::ast::unary_op::kind::op_is_null:
      case sirius::ast::unary_op::kind::op_is_not_null: return true;
      case sirius::ast::unary_op::kind::op_try: {
        auto const rt = effective_node_cudf_type(*uo.child, input);
        return is_supported_fixed_width(rt);
      }
      default: return false;
    }
  }

  if (expr.holds<sirius::ast::function_call>()) {
    auto const& fc   = expr.get<sirius::ast::function_call>();
    auto const& args = fc.arguments();
    for (auto const& arg : args) {
      if (!arg || !is_node_supported(*arg, input)) { return false; }
    }
    auto const opt_ret = sirius::get_cudf_type(fc.return_type());
    if (!opt_ret || !is_supported_fixed_width(*opt_ret)) { return false; }
    auto const ret_type = *opt_ret;

    switch (fc.function()) {
      case sirius::function_id::add:
      case sirius::function_id::sub:
      case sirius::function_id::mul:
      case sirius::function_id::div:
      case sirius::function_id::int_div:
      case sirius::function_id::mod: {
        if (args.size() != 2) { return false; }
        if (!is_integer_or_float(ret_type)) { return false; }
        auto const lt = effective_node_cudf_type(*args[0], input);
        auto const rt = effective_node_cudf_type(*args[1], input);
        return is_integer_or_float(lt) && is_integer_or_float(rt);
      }
      case sirius::function_id::abs: {
        if (args.size() != 1) { return false; }
        auto const ct = effective_node_cudf_type(*args[0], input);
        return is_integer_or_float(ret_type) && ct == ret_type;
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
        return args.size() >= 2 && is_supported_fixed_width(ret_type);
      }
      default: return false;
    }
  }

  return false;
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

[[nodiscard]] void const* get_input_data_ptr(cudf::column_view const& col)
{
  switch (col.type().id()) {
    case cudf::type_id::INT8: return col.data<std::int8_t>();
    case cudf::type_id::INT16: return col.data<std::int16_t>();
    case cudf::type_id::INT32: return col.data<std::int32_t>();
    case cudf::type_id::INT64: return col.data<std::int64_t>();
    case cudf::type_id::UINT8: return col.data<std::uint8_t>();
    case cudf::type_id::UINT16: return col.data<std::uint16_t>();
    case cudf::type_id::UINT32: return col.data<std::uint32_t>();
    case cudf::type_id::UINT64: return col.data<std::uint64_t>();
    case cudf::type_id::FLOAT32: return col.data<float>();
    case cudf::type_id::FLOAT64: return col.data<double>();
    case cudf::type_id::BOOL8: return col.data<bool>();
    case cudf::type_id::TIMESTAMP_DAYS: return col.data<std::int32_t>();
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return col.data<std::int64_t>();
    case cudf::type_id::DECIMAL32: return col.data<std::int32_t>();
    case cudf::type_id::DECIMAL64: return col.data<std::int64_t>();
    case cudf::type_id::DECIMAL128: return col.data<__int128_t>();
    default:
      throw not_implemented_exception("[nvrtc] Unsupported input column type: {}",
                                      cudf::type_to_name(col.type()));
  }
}

[[nodiscard]] void* get_output_data_ptr(cudf::mutable_column_view& col)
{
  switch (col.type().id()) {
    case cudf::type_id::INT8: return col.data<std::int8_t>();
    case cudf::type_id::INT16: return col.data<std::int16_t>();
    case cudf::type_id::INT32: return col.data<std::int32_t>();
    case cudf::type_id::INT64: return col.data<std::int64_t>();
    case cudf::type_id::UINT8: return col.data<std::uint8_t>();
    case cudf::type_id::UINT16: return col.data<std::uint16_t>();
    case cudf::type_id::UINT32: return col.data<std::uint32_t>();
    case cudf::type_id::UINT64: return col.data<std::uint64_t>();
    case cudf::type_id::FLOAT32: return col.data<float>();
    case cudf::type_id::FLOAT64: return col.data<double>();
    case cudf::type_id::BOOL8: return col.data<bool>();
    case cudf::type_id::TIMESTAMP_DAYS: return col.data<std::int32_t>();
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return col.data<std::int64_t>();
    case cudf::type_id::DECIMAL32: return col.data<std::int32_t>();
    case cudf::type_id::DECIMAL64: return col.data<std::int64_t>();
    case cudf::type_id::DECIMAL128: return col.data<__int128_t>();
    default:
      throw not_implemented_exception("[nvrtc] Unsupported output column type: {}",
                                      cudf::type_to_name(col.type()));
  }
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
  _input_columns.push_back(
    input_column_binding{column_index, col_view.type(), is_nullable, binding_id});
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

  if (!col_binding.is_nullable) {
    emit_line("reg_val<" + dst_type_str + "> " + var_name + "{static_cast<" + dst_type_str +
              ">(in_" + std::to_string(binding_id) + "_data[row]), true, false};");
  } else {
    emit_line("bool const " + var_name + "_valid = (in_" + std::to_string(binding_id) +
              "_mask == nullptr) || is_bit_set(in_" + std::to_string(binding_id) + "_mask, in_" +
              std::to_string(binding_id) + "_bit_offset + row);");
    emit_line("reg_val<" + dst_type_str + "> " + var_name + "{" + var_name +
              "_valid ? static_cast<" + dst_type_str + ">(in_" + std::to_string(binding_id) +
              "_data[row]) : " + dst_type_str + "{}, " + var_name + "_valid, false};");
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

  emit_line("reg_val<" + type_str + "> " + var_name + "{c_" + std::to_string(binding_id) +
            "_val, c_" + std::to_string(binding_id) + "_valid != 0, false};");

  register_value reg{var_name,
                     expr.return_type(),
                     device_type,
                     /*is_statically_non_null=*/false,
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

  emit_line("reg_val<" + type_str + "> " + var_name + "{c_" + std::to_string(binding_id) +
            "_val, c_" + std::to_string(binding_id) + "_valid != 0, false};");

  register_value reg{var_name,
                     expr.return_type(),
                     carrier,
                     /*is_statically_non_null=*/false,
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

  auto const cmp_type_str = [&]() -> std::string {
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
    if (!cudf::is_unsigned(lhs.device_type) && !cudf::is_unsigned(rhs.device_type)) {
      return "int64_t";
    }
    return "__int128_t";
  }();

  auto const lhs_val = "static_cast<" + cmp_type_str + ">(" + lhs.var_name + ".val)";
  auto const rhs_val = "static_cast<" + cmp_type_str + ">(" + rhs.var_name + ".val)";
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

  auto const cmp_type_str = cuda_type_name(in_reg.device_type);
  auto const in_val       = "static_cast<" + cmp_type_str + ">(" + in_reg.var_name + ".val)";
  auto const lo_val       = "static_cast<" + cmp_type_str + ">(" + lo_reg.var_name + ".val)";
  auto const hi_val       = "static_cast<" + cmp_type_str + ">(" + hi_reg.var_name + ".val)";

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
    bool const can_ovf =
      child.can_overflow || (cudf::is_integral(ret_type) && !cudf::is_unsigned(ret_type));
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

  throw not_implemented_exception("[nvrtc] Unsupported function_id in translation_context: {}",
                                  static_cast<int>(fid));
}

register_value translation_context::produce(sirius::ast::node const& expr)
{
  if (expr.holds<sirius::ast::reference>()) {
    return produce_reference(expr.get<sirius::ast::reference>());
  }
  if (expr.holds<sirius::ast::constant>()) {
    return produce_constant(expr.get<sirius::ast::constant>());
  }
  if (expr.holds<sirius::ast::comparison>()) {
    return produce_comparison(expr.get<sirius::ast::comparison>());
  }
  if (expr.holds<sirius::ast::between>()) {
    return produce_between(expr.get<sirius::ast::between>());
  }
  if (expr.holds<sirius::ast::unary_op>()) {
    return produce_unary_op(expr.get<sirius::ast::unary_op>());
  }
  if (expr.holds<sirius::ast::function_call>()) {
    return produce_function_call(expr.get<sirius::ast::function_call>());
  }
  throw not_implemented_exception("[nvrtc] Unsupported AST node in translation_context::produce");
}

bool can_compile_project_sink(std::span<sirius::ast::node const* const> expressions,
                              cudf::table_view const& input) noexcept
{
  if (expressions.empty()) { return false; }
  for (auto const* expr : expressions) {
    if (expr == nullptr || !is_node_supported(*expr, input)) { return false; }
  }
  return true;
}

std::unique_ptr<cudf::table> evaluate_project_sink(
  std::span<sirius::ast::node const* const> expressions,
  cudf::table_view const& input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::size_t* narrow_domain_comparison_count)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) {
    if (narrow_domain_comparison_count != nullptr) {
      translation_context count_ctx(input);
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

  translation_context ctx(input);
  std::vector<register_value> out_regs;
  out_regs.reserve(expressions.size());
  for (auto const* expr : expressions) {
    out_regs.push_back(ctx.produce(*expr));
  }
  if (narrow_domain_comparison_count != nullptr) {
    *narrow_domain_comparison_count += ctx.narrow_domain_comparison_count();
  }

  constexpr int block_threads = 256;
  constexpr int num_warps     = block_threads / 32;
  bool any_nullable_output    = false;
  bool any_can_overflow       = false;
  for (auto const& reg : out_regs) {
    if (!reg.is_statically_non_null) { any_nullable_output = true; }
    if (reg.can_overflow) { any_can_overflow = true; }
  }

  // Assemble the CUDA kernel source string.
  std::ostringstream src;
  src
    << "#include \"codegen/stdint_shim.hpp\"\n"
    << "#include <cuda/std/limits>\n"
    << "#include <cuda/std/type_traits>\n\n"
    << "template <typename T>\n"
    << "struct reg_val {\n"
    << "  T val;\n"
    << "  bool valid;\n"
    << "  bool overflow;\n"
    << "};\n\n"
    << "__device__ __forceinline__ bool is_bit_set(uint32_t const* mask, int32_t bit_idx) {\n"
    << "  return ((mask[bit_idx >> 5] >> (bit_idx & 31)) & 1u) != 0u;\n"
    << "}\n\n"
    << "template <typename T, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<T> eval_add(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }\n"
    << "  if constexpr (cuda::std::is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{static_cast<T>(l.val) + static_cast<T>(r.val), true, false};\n"
    << "  } else if constexpr (cuda::std::is_signed_v<T>) {\n"
    << "    using Wide = cuda::std::conditional_t<(sizeof(T) < 8), int64_t, __int128_t>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) + "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = (res < static_cast<Wide>(cuda::std::numeric_limits<T>::lowest())) ||\n"
    << "                     (res > static_cast<Wide>(cuda::std::numeric_limits<T>::max()));\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  } else {\n"
    << "    using Wide = cuda::std::conditional_t<(sizeof(T) < 8), uint64_t, unsigned __int128>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) + "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = res > static_cast<Wide>(cuda::std::numeric_limits<T>::max());\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  }\n"
    << "}\n\n"
    << "template <typename T, typename L, typename R>\n"
    << "__device__ __forceinline__ reg_val<T> eval_sub(reg_val<L> l, reg_val<R> r) {\n"
    << "  bool const ovf_in = l.overflow || r.overflow;\n"
    << "  bool const valid = l.valid && r.valid;\n"
    << "  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }\n"
    << "  if constexpr (cuda::std::is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{static_cast<T>(l.val) - static_cast<T>(r.val), true, false};\n"
    << "  } else if constexpr (cuda::std::is_signed_v<T>) {\n"
    << "    using Wide = cuda::std::conditional_t<(sizeof(T) < 8), int64_t, __int128_t>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) - "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = (res < static_cast<Wide>(cuda::std::numeric_limits<T>::lowest())) ||\n"
    << "                     (res > static_cast<Wide>(cuda::std::numeric_limits<T>::max()));\n"
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
    << "  if constexpr (cuda::std::is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{static_cast<T>(l.val) * static_cast<T>(r.val), true, false};\n"
    << "  } else if constexpr (cuda::std::is_signed_v<T>) {\n"
    << "    using Wide = cuda::std::conditional_t<(sizeof(T) < 8), int64_t, __int128_t>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) * "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = (res < static_cast<Wide>(cuda::std::numeric_limits<T>::lowest())) ||\n"
    << "                     (res > static_cast<Wide>(cuda::std::numeric_limits<T>::max()));\n"
    << "    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};\n"
    << "  } else {\n"
    << "    using Wide = cuda::std::conditional_t<(sizeof(T) < 8), uint64_t, unsigned __int128>;\n"
    << "    Wide const res = static_cast<Wide>(static_cast<T>(l.val)) * "
       "static_cast<Wide>(static_cast<T>(r.val));\n"
    << "    bool const ovf = res > static_cast<Wide>(cuda::std::numeric_limits<T>::max());\n"
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
    << "  if constexpr (cuda::std::is_floating_point_v<T>) {\n"
    << "    return reg_val<T>{lv / rv, true, false};\n"
    << "  } else {\n"
    << "    if (rv == T{0}) { return reg_val<T>{T{}, false, false}; }\n"
    << "    if constexpr (cuda::std::is_signed_v<T>) {\n"
    << "      if (lv == cuda::std::numeric_limits<T>::lowest() && rv == T{-1}) {\n"
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
    << "  if constexpr (cuda::std::is_floating_point_v<T>) {\n"
    << "    if constexpr (sizeof(T) == 4) {\n"
    << "      return reg_val<T>{::fmodf(lv, rv), true, false};\n"
    << "    } else {\n"
    << "      return reg_val<T>{::fmod(lv, rv), true, false};\n"
    << "    }\n"
    << "  } else {\n"
    << "    if (rv == T{0}) { return reg_val<T>{T{}, false, false}; }\n"
    << "    if constexpr (cuda::std::is_signed_v<T>) {\n"
    << "      if (lv == cuda::std::numeric_limits<T>::lowest() && rv == T{-1}) {\n"
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
    << "  if constexpr (cuda::std::is_floating_point_v<T>) {\n"
    << "    if constexpr (sizeof(T) == 4) {\n"
    << "      return reg_val<T>{::fabsf(v), true, false};\n"
    << "    } else {\n"
    << "      return reg_val<T>{::fabs(v), true, false};\n"
    << "    }\n"
    << "  } else if constexpr (cuda::std::is_signed_v<T>) {\n"
    << "    if (v == cuda::std::numeric_limits<T>::lowest()) {\n"
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
    << "extern \"C\" __global__ void sirius_nvrtc_project_kernel(\n";

  for (auto const& in_col : ctx.input_columns()) {
    auto const id_str   = std::to_string(in_col.binding_id);
    auto const type_str = cuda_type_name(in_col.physical_type);
    src << "    " << type_str << " const* __restrict__ in_" << id_str << "_data,\n"
        << "    uint32_t const* __restrict__ in_" << id_str << "_mask,\n"
        << "    int32_t in_" << id_str << "_bit_offset,\n";
  }

  for (auto const& cst : ctx.constants()) {
    auto const id_str   = std::to_string(cst.binding_id);
    auto const type_str = cuda_type_name(cst.device_type);
    src << "    " << type_str << " c_" << id_str << "_val,\n"
        << "    uint8_t c_" << id_str << "_valid,\n";
  }

  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    auto const k_str    = std::to_string(k);
    auto const type_str = cuda_type_name(out_regs[k].device_type);
    src << "    " << type_str << "* __restrict__ out_" << k_str << "_data,\n"
        << "    uint32_t* __restrict__ out_" << k_str << "_mask,\n";
  }

  src << "    int32_t num_rows,\n"
      << "    uint32_t* __restrict__ status_and_null_counts)\n"
      << "{\n"
      << "  int32_t const row = static_cast<int32_t>(blockIdx.x) * " << block_threads
      << " + static_cast<int32_t>(threadIdx.x);\n"
      << "  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;\n"
      << "  int32_t const warp_in_block = static_cast<int32_t>(threadIdx.x) >> 5;\n"
      << "  int32_t const global_warp = (static_cast<int32_t>(blockIdx.x) * " << block_threads
      << " + static_cast<int32_t>(threadIdx.x)) >> 5;\n"
      << "  int32_t const warp_base_row = global_warp << 5;\n"
      << "  bool const active = (row < num_rows);\n";

  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    if (!out_regs[k].is_statically_non_null) { src << "  bool out_" << k << "_valid = false;\n"; }
  }
  if (any_can_overflow) { src << "  bool row_overflow = false;\n"; }

  src << "  if (active) {\n" << ctx.body_code();
  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    auto const& reg = out_regs[k];
    if (reg.is_statically_non_null) {
      src << "    out_" << k << "_data[row] = " << reg.var_name << ".val;\n";
    } else {
      src << "    if (" << reg.var_name << ".valid) {\n"
          << "      out_" << k << "_data[row] = " << reg.var_name << ".val;\n"
          << "    }\n"
          << "    out_" << k << "_valid = " << reg.var_name << ".valid;\n";
    }
    if (reg.can_overflow) {
      src << "    row_overflow = row_overflow || (" << reg.var_name << ".valid && " << reg.var_name
          << ".overflow);\n";
    }
  }
  src << "  }\n";

  if (any_can_overflow) {
    src << "  uint32_t const ovf_ballot = __ballot_sync(0xFFFFFFFFu, active && row_overflow);\n"
        << "  if (ovf_ballot != 0u && lane == 0) {\n"
        << "    atomicExch(&status_and_null_counts[0], 1u);\n"
        << "  }\n";
  }

  if (any_nullable_output) {
    src << "  int32_t const rem = num_rows - warp_base_row;\n"
        << "  int32_t const active_in_warp = rem <= 0 ? 0 : (rem < 32 ? rem : 32);\n"
        << "  uint32_t const valid_row_mask = (active_in_warp == 32) ? 0xFFFFFFFFu : "
           "((active_in_warp <= 0) ? 0u : ((1u << active_in_warp) - 1u));\n";

    for (std::size_t k = 0; k < out_regs.size(); ++k) {
      if (out_regs[k].is_statically_non_null) { continue; }
      src << "  __shared__ uint32_t s_nulls_" << k << "[" << num_warps << "];\n"
          << "  uint32_t const ballot_" << k << " = __ballot_sync(0xFFFFFFFFu, active && out_" << k
          << "_valid) & valid_row_mask;\n"
          << "  if (lane == 0) {\n"
          << "    if (active_in_warp > 0 && out_" << k << "_mask != nullptr) {\n"
          << "      out_" << k << "_mask[global_warp] = ballot_" << k << ";\n"
          << "    }\n"
          << "    s_nulls_" << k
          << "[warp_in_block] = static_cast<uint32_t>(active_in_warp - __popc(ballot_" << k
          << "));\n"
          << "  }\n";
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
  opts.arch_cc       = codegen::jit::arch_cc_for_current_device();
  opts.device_int128 = true;
  std::shared_ptr<const codegen::jit::CompiledKernel> compiled;
  try {
    compiled = codegen::jit::KernelCache::instance().get_or_compile_plain(
      src.str(), "sirius_nvrtc_project_kernel", opts);
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

  // Allocate status and null counts buffer if needed: [0] = overflow status, [1..K] = null counts.
  bool const need_status_buffer  = any_nullable_output || any_can_overflow;
  std::size_t const status_words = 1 + out_regs.size();
  rmm::device_uvector<std::uint32_t> d_status(need_status_buffer ? status_words : 0, stream, mr);
  std::uint32_t* status_ptr = need_status_buffer ? d_status.data() : nullptr;
  if (need_status_buffer) {
    simpatico::throw_if_cuda_error(
      cudaMemsetAsync(status_ptr, 0, status_words * sizeof(std::uint32_t), stream.get()),
      "cudaMemsetAsync(status_and_null_counts)");
  }

  // Prepare kernel argument storage.
  struct input_arg_storage {
    void const* data_ptr{nullptr};
    std::uint32_t const* mask_ptr{nullptr};
    std::int32_t bit_offset{0};
  };
  std::vector<input_arg_storage> in_args(ctx.input_columns().size());
  for (std::size_t i = 0; i < ctx.input_columns().size(); ++i) {
    auto const& binding   = ctx.input_columns()[i];
    auto const col_view   = input.column(binding.column_index);
    in_args[i].data_ptr   = get_input_data_ptr(col_view);
    in_args[i].mask_ptr   = binding.is_nullable ? col_view.null_mask() : nullptr;
    in_args[i].bit_offset = static_cast<std::int32_t>(col_view.offset());
  }

  std::vector<constant_param_storage> const_args(ctx.constants().size());
  for (std::size_t i = 0; i < ctx.constants().size(); ++i) {
    populate_constant_param(ctx.constants()[i], const_args[i]);
  }

  std::int32_t num_rows_arg = static_cast<std::int32_t>(num_rows);

  std::vector<void*> kernel_args;
  kernel_args.reserve(in_args.size() * 3 + const_args.size() * 2 + outputs.size() * 2 + 2);
  for (auto& in_arg : in_args) {
    kernel_args.push_back(&in_arg.data_ptr);
    kernel_args.push_back(&in_arg.mask_ptr);
    kernel_args.push_back(&in_arg.bit_offset);
  }
  for (auto& cst_arg : const_args) {
    kernel_args.push_back(&cst_arg.val_bytes[0]);
    kernel_args.push_back(&cst_arg.valid);
  }
  for (auto& out_arg : outputs) {
    kernel_args.push_back(&out_arg.data_ptr);
    kernel_args.push_back(&out_arg.mask_ptr);
  }
  kernel_args.push_back(&num_rows_arg);
  kernel_args.push_back(&status_ptr);

  auto const grid_dim_x = static_cast<unsigned int>((num_rows + block_threads - 1) / block_threads);
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

  std::vector<std::uint32_t> h_status;
  if (need_status_buffer) {
    h_status.resize(status_words, 0);
    simpatico::throw_if_cuda_error(cudaMemcpyAsync(h_status.data(),
                                                   status_ptr,
                                                   status_words * sizeof(std::uint32_t),
                                                   cudaMemcpyDeviceToHost,
                                                   stream.get()),
                                   "cudaMemcpyAsync(status_and_null_counts)");
    simpatico::throw_if_cuda_error(cudaStreamSynchronize(stream.get()),
                                   "cudaStreamSynchronize(sirius_nvrtc_project_kernel)");
    if (h_status[0] != 0U) {
      throw out_of_range_exception("Overflow in NVRTC expression evaluation");
    }
  }

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

}  // namespace sirius::nvrtc
