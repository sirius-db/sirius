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

#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <variant>
#include <vector>

namespace sirius::nvrtc {

namespace {

[[nodiscard]] bool is_node_supported(sirius::ast::node const& expr,
                                     cudf::table_view const& input) noexcept
{
  if (expr.holds<sirius::ast::reference>()) {
    auto const& ref = expr.get<sirius::ast::reference>();
    if (ref.column_index >= static_cast<std::uint32_t>(input.num_columns())) { return false; }
    auto const col_view = input.column(ref.column_index);
    if (!is_supported_fixed_width(col_view.type())) { return false; }
    auto const& logical = ref.return_type();
    if (logical.id() != sirius::type_id::INVALID) {
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

  return false;
}

[[nodiscard]] cudf::data_type resolve_output_cudf_type(sirius::ast::node const& expr,
                                                       cudf::table_view const& input)
{
  if (expr.holds<sirius::ast::reference>()) {
    auto const& ref     = expr.get<sirius::ast::reference>();
    auto const col_view = input.column(ref.column_index);
    auto const& logical = ref.return_type();
    if (logical.id() == sirius::type_id::INVALID) { return col_view.type(); }
    auto const native = sirius::require_cudf_type(logical);
    if (is_narrowable_numeric_type(logical) && can_restore_to(col_view.type(), native)) {
      return native;
    }
    return col_view.type();
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

  if (logical.id() != sirius::type_id::INVALID) {
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

register_value translation_context::produce(sirius::ast::node const& expr)
{
  if (expr.holds<sirius::ast::reference>()) {
    return produce_reference(expr.get<sirius::ast::reference>());
  }
  if (expr.holds<sirius::ast::constant>()) {
    return produce_constant(expr.get<sirius::ast::constant>());
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
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) {
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
  src << "#include \"codegen/stdint_shim.hpp\"\n\n";
  src << "template <typename T>\n"
      << "struct reg_val {\n"
      << "  T val;\n"
      << "  bool valid;\n"
      << "  bool overflow;\n"
      << "};\n\n"
      << "__device__ __forceinline__ bool is_bit_set(uint32_t const* mask, int32_t bit_idx) {\n"
      << "  return ((mask[bit_idx >> 5] >> (bit_idx & 31)) & 1u) != 0u;\n"
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
  opts.arch_cc        = codegen::jit::arch_cc_for_current_device();
  auto const compiled = codegen::jit::KernelCache::instance().get_or_compile_plain(
    src.str(), "sirius_nvrtc_project_kernel", opts);
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
