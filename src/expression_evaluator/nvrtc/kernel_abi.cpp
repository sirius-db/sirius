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

#include <cudf/strings/strings_column_view.hpp>

#include <expression_evaluator/nvrtc/kernel_abi.hpp>
#include <sirius/exception.hpp>

#include <cstring>
#include <format>
#include <variant>

namespace sirius::nvrtc {

namespace {

[[nodiscard]] void const* input_data_pointer(cudf::column_view const& col)
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
      throw not_implemented_exception("[nvrtc] Unsupported column type: {}",
                                      cudf::type_to_name(col.type()));
  }
}

template <typename T>
void store_bytes(std::uint8_t* dst, T const& v)
{
  std::memcpy(dst, &v, sizeof(v));
}

void store_constant_payload(scalar_constant_binding const& binding, std::uint8_t* bytes)
{
  switch (binding.device_type.id()) {
    case cudf::type_id::INT8: store_bytes(bytes, std::get<std::int8_t>(binding.payload)); break;
    case cudf::type_id::INT16: store_bytes(bytes, std::get<std::int16_t>(binding.payload)); break;
    case cudf::type_id::INT32: store_bytes(bytes, std::get<std::int32_t>(binding.payload)); break;
    case cudf::type_id::INT64: store_bytes(bytes, std::get<std::int64_t>(binding.payload)); break;
    case cudf::type_id::UINT8: store_bytes(bytes, std::get<std::uint8_t>(binding.payload)); break;
    case cudf::type_id::UINT16: store_bytes(bytes, std::get<std::uint16_t>(binding.payload)); break;
    case cudf::type_id::UINT32: store_bytes(bytes, std::get<std::uint32_t>(binding.payload)); break;
    case cudf::type_id::UINT64: store_bytes(bytes, std::get<std::uint64_t>(binding.payload)); break;
    case cudf::type_id::FLOAT32: store_bytes(bytes, std::get<float>(binding.payload)); break;
    case cudf::type_id::FLOAT64: store_bytes(bytes, std::get<double>(binding.payload)); break;
    case cudf::type_id::BOOL8: store_bytes(bytes, std::get<bool>(binding.payload)); break;
    case cudf::type_id::TIMESTAMP_DAYS:
      store_bytes(bytes, std::get<sirius::date_value>(binding.payload).days);
      break;
    case cudf::type_id::TIMESTAMP_SECONDS:
      store_bytes(bytes, std::get<sirius::timestamp_sec_value>(binding.payload).value);
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      store_bytes(bytes, std::get<sirius::timestamp_ms_value>(binding.payload).value);
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      store_bytes(bytes, std::get<sirius::timestamp_us_value>(binding.payload).value);
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      store_bytes(bytes, std::get<sirius::timestamp_ns_value>(binding.payload).value);
      break;
    case cudf::type_id::DECIMAL32:
      store_bytes(bytes, std::get<sirius::decimal32>(binding.payload).value);
      break;
    case cudf::type_id::DECIMAL64:
      store_bytes(bytes, std::get<sirius::decimal64>(binding.payload).value);
      break;
    case cudf::type_id::DECIMAL128:
      store_bytes(bytes, std::get<sirius::decimal128>(binding.payload).value);
      break;
    default:
      throw not_implemented_exception("[nvrtc] Unsupported constant parameter type: {}",
                                      cudf::type_to_name(binding.device_type));
  }
}

}  // namespace

std::string input_param_name(std::size_t binding_id, std::string_view field)
{
  return std::format("in_{}_{}", binding_id, field);
}

std::string constant_param_name(std::size_t binding_id, std::string_view field)
{
  return std::format("c_{}_{}", binding_id, field);
}

std::string like_pattern_param_name(std::size_t binding_id)
{
  return std::format("lp_{}", binding_id);
}

//===----------------------------------------------------------------------===//
// kernel_abi
//===----------------------------------------------------------------------===//
kernel_abi::kernel_abi(translation_context const& ctx)
{
  for (auto const& in_col : ctx.input_columns()) {
    auto const id = in_col.binding_id;
    if (in_col.physical_type.id() == cudf::type_id::STRING) {
      _string_input_bindings.push_back(id);
      add(source::INPUT_OFFSETS,
          id,
          cuda_type_name(in_col.offset_type) + " const* __restrict__",
          input_param_name(id, "offsets"));
      add(source::INPUT_CHARS, id, "char const* __restrict__", input_param_name(id, "chars"));
    } else {
      add(source::INPUT_DATA,
          id,
          cuda_type_name(in_col.physical_type) + " const* __restrict__",
          input_param_name(id, "data"));
    }
    add(source::INPUT_MASK, id, "uint32_t const* __restrict__", input_param_name(id, "mask"));
    add(source::INPUT_BIT_OFFSET, id, "int32_t", input_param_name(id, "bit_offset"));
  }
  for (auto const& pat : ctx.like_patterns()) {
    add(source::LIKE_PATTERN,
        pat.binding_id,
        "like_pattern_desc",
        like_pattern_param_name(pat.binding_id));
  }
  for (auto const& cst : ctx.constants()) {
    if (!cst.is_valid) { continue; }
    auto const id = cst.binding_id;
    if (cst.device_type.id() == cudf::type_id::STRING) {
      add(source::CONSTANT_STRING_PTR,
          id,
          "char const* __restrict__",
          constant_param_name(id, "ptr"));
      add(source::CONSTANT_STRING_LEN, id, "int32_t", constant_param_name(id, "len"));
    } else {
      add(source::CONSTANT_VALUE,
          id,
          cuda_type_name(cst.device_type),
          constant_param_name(id, "val"));
    }
  }
}

void kernel_abi::add(source from, std::size_t index, std::string cuda_type, std::string identifier)
{
  _parameters.push_back(kernel_parameter{std::move(cuda_type), std::move(identifier)});
  _slots.push_back(slot{from, index});
}

void kernel_abi::add_sink_parameter(std::string cuda_type, std::string identifier)
{
  add(source::SINK, 0, std::move(cuda_type), std::move(identifier));
  ++_sink_parameter_count;
}

void kernel_abi::emit_parameter_list(std::string& out) const
{
  out += "(\n";
  for (std::size_t i = 0; i < _parameters.size(); ++i) {
    out += std::format("    {} {}{}\n",
                       _parameters[i].cuda_type,
                       _parameters[i].identifier,
                       i + 1 < _parameters.size() ? "," : "");
  }
  out += ")\n";
}

void kernel_abi::emit_kernel_prelude(std::string& out) const
{
  for (auto const id : _string_input_bindings) {
    out +=
      std::format("  char const* const {0} = ({1} != nullptr) ? {2} + {1}[num_rows] : nullptr;\n",
                  input_param_name(id, "chars_end"),
                  input_param_name(id, "offsets"),
                  input_param_name(id, "chars"));
  }
}

//===----------------------------------------------------------------------===//
// kernel_arguments
//===----------------------------------------------------------------------===//
kernel_arguments::kernel_arguments(kernel_abi const& abi,
                                   translation_context const& ctx,
                                   cudf::table_view const& input,
                                   string_constant_cache const* string_constants,
                                   ::cuda::stream_ref stream)
  : _abi(abi),
    _inputs(ctx.input_columns().size()),
    _constants(ctx.constants().size()),
    _string_constants(ctx.constants().size())
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
            _inputs[i].offsets_ptr = offsets_col.data<std::int64_t>() + col_view.offset();
          } else {
            _inputs[i].offsets_ptr = offsets_col.data<std::int32_t>() + col_view.offset();
          }
          _inputs[i].chars_ptr = scv.chars_begin(stream);
        }
      }
    } else {
      _inputs[i].data_ptr = input_data_pointer(col_view);
    }
    _inputs[i].mask_ptr   = binding.is_nullable ? col_view.null_mask() : nullptr;
    _inputs[i].bit_offset = static_cast<std::int32_t>(col_view.offset());
  }

  std::string pool;
  std::vector<std::size_t> string_offsets(ctx.constants().size(), 0);
  for (std::size_t i = 0; i < ctx.constants().size(); ++i) {
    auto const& cst = ctx.constants()[i];
    if (cst.device_type.id() == cudf::type_id::STRING && cst.is_valid) {
      string_offsets[i] = pool.size();
      pool.append(std::get<std::string>(cst.payload));
    }
  }
  char const* pool_data = nullptr;
  if (!pool.empty()) {
    if (string_constants == nullptr) {
      throw internal_exception("[nvrtc] string constants need a string_constant_cache");
    }
    pool_data = _pool.emplace(string_constants->acquire(pool, stream)).data();
  }
  for (std::size_t i = 0; i < ctx.constants().size(); ++i) {
    auto const& cst = ctx.constants()[i];
    if (!cst.is_valid) { continue; }
    if (cst.device_type.id() == cudf::type_id::STRING) {
      _string_constants[i].ptr = pool_data + string_offsets[i];
      _string_constants[i].len =
        static_cast<std::int32_t>(std::get<std::string>(cst.payload).size());
    } else {
      store_constant_payload(cst, _constants[i].val_bytes.data());
    }
  }

  static_assert(sizeof(sirius::detail::like_multiliteral_literal_desc) == 88 &&
                  sizeof(sirius::detail::like_multiliteral_pattern_desc) == 360,
                "like_pattern_desc in the device runtime mirrors this layout");
  _pointers.reserve(abi._slots.size());
  for (auto const& s : abi._slots) {
    switch (s.from) {
      case kernel_abi::source::INPUT_DATA: _pointers.push_back(&_inputs[s.index].data_ptr); break;
      case kernel_abi::source::INPUT_OFFSETS:
        _pointers.push_back(&_inputs[s.index].offsets_ptr);
        break;
      case kernel_abi::source::INPUT_CHARS: _pointers.push_back(&_inputs[s.index].chars_ptr); break;
      case kernel_abi::source::INPUT_MASK: _pointers.push_back(&_inputs[s.index].mask_ptr); break;
      case kernel_abi::source::INPUT_BIT_OFFSET:
        _pointers.push_back(&_inputs[s.index].bit_offset);
        break;
      case kernel_abi::source::LIKE_PATTERN:
        // cuLaunchKernel takes `void**` to argument values it only reads; the descriptor lives in
        // the (const) translation context, which must outlive the launch.
        _pointers.push_back(
          const_cast<void*>(static_cast<void const*>(&ctx.like_patterns()[s.index].descriptor)));
        break;
      case kernel_abi::source::CONSTANT_VALUE:
        _pointers.push_back(&_constants[s.index].val_bytes[0]);
        break;
      case kernel_abi::source::CONSTANT_STRING_PTR:
        _pointers.push_back(&_string_constants[s.index].ptr);
        break;
      case kernel_abi::source::CONSTANT_STRING_LEN:
        _pointers.push_back(&_string_constants[s.index].len);
        break;
      case kernel_abi::source::SINK: break;  // supplied by add_sink_argument
    }
  }
}

void kernel_arguments::add_sink_argument(void* host_value)
{
  if (_sink_arguments_added >= _abi.sink_parameter_count()) {
    throw internal_exception("[nvrtc] more sink arguments than sink parameters");
  }
  _pointers.push_back(host_value);
  ++_sink_arguments_added;
}

void** kernel_arguments::data()
{
  if (_sink_arguments_added != _abi.sink_parameter_count()) {
    throw internal_exception("[nvrtc] {} of {} sink arguments supplied",
                             _sink_arguments_added,
                             _abi.sink_parameter_count());
  }
  return _pointers.data();
}

}  // namespace sirius::nvrtc
