// SPDX-License-Identifier: Apache-2.0
//
// bit_spec.cpp — bitextract / bitjoin spec-string parsing, split out of
// representation.hpp. Pure string parsing, orthogonal to the rep classes.

#include "codegen/plan/bit_spec.hpp"

#include <cctype>

namespace simpatico {

std::optional<std::string_view> strip_bitextract_prefix(std::string_view name)
{
  if (name.size() > kBitextractPrefix.size() &&
      name.compare(0, kBitextractPrefix.size(), kBitextractPrefix) == 0) {
    return name.substr(kBitextractPrefix.size());
  }
  return std::nullopt;
}

std::optional<bitextract_spec_result> parse_float_alias(std::string_view suffix)
{
  if (suffix == "f16")
    return bitextract_spec_result{{{1, "sign"}, {5, "exponent"}, {10, "mantissa"}},
                                  cudf::data_type(cudf::type_id::UINT16)};
  if (suffix == "f32")
    return bitextract_spec_result{{{1, "sign"}, {8, "exponent"}, {23, "mantissa"}},
                                  cudf::data_type(cudf::type_id::FLOAT32)};
  if (suffix == "f64")
    return bitextract_spec_result{{{1, "sign"}, {11, "exponent"}, {52, "mantissa"}},
                                  cudf::data_type(cudf::type_id::FLOAT64)};
  return std::nullopt;
}

cudf::data_type parse_packed_type_token(std::string_view tok)
{
  if (tok == "u8" || tok == "uint8") return cudf::data_type(cudf::type_id::UINT8);
  if (tok == "u16" || tok == "uint16") return cudf::data_type(cudf::type_id::UINT16);
  if (tok == "u32" || tok == "uint32") return cudf::data_type(cudf::type_id::UINT32);
  if (tok == "u64" || tok == "uint64") return cudf::data_type(cudf::type_id::UINT64);
  if (tok == "i8" || tok == "int8") return cudf::data_type(cudf::type_id::INT8);
  if (tok == "i16" || tok == "int16") return cudf::data_type(cudf::type_id::INT16);
  if (tok == "i32" || tok == "int32") return cudf::data_type(cudf::type_id::INT32);
  if (tok == "i64" || tok == "int64") return cudf::data_type(cudf::type_id::INT64);
  if (tok == "f32") return cudf::data_type(cudf::type_id::FLOAT32);
  if (tok == "f64") return cudf::data_type(cudf::type_id::FLOAT64);
  return cudf::data_type(cudf::type_id::EMPTY);
}

std::string bitextract_canonical_name(std::string const& compressor)
{
  auto suffix = strip_bitextract_prefix(compressor);
  if (!suffix) return compressor;
  if (*suffix == "f32" || *suffix == "f64") return compressor;
  if (suffix->empty() || std::isdigit(static_cast<unsigned char>((*suffix)[0]))) {
    return compressor;  // generic form, no type prefix to strip
  }
  size_t us = suffix->find('_');
  if (us == std::string_view::npos) return compressor;
  if (parse_packed_type_token(suffix->substr(0, us)).id() == cudf::type_id::EMPTY) {
    return compressor;  // first segment isn't a type token, leave alone
  }
  return std::string("bitextract_") + std::string(suffix->substr(us + 1));
}

std::string canonicalize_path(std::string const& path)
{
  size_t dot            = path.find('.');
  std::string head      = (dot == std::string::npos) ? path : path.substr(0, dot);
  std::string canonical = bitextract_canonical_name(head);
  if (canonical == head) return path;
  return (dot == std::string::npos) ? canonical : canonical + path.substr(dot);
}

std::vector<bitfield_spec> parse_bitfield_list(std::string_view fields_suffix)
{
  std::vector<bitfield_spec> fields;
  size_t pos = 0;
  while (pos <= fields_suffix.size()) {
    size_t next          = fields_suffix.find('_', pos);
    std::string_view tok = (next == std::string_view::npos) ? fields_suffix.substr(pos)
                                                            : fields_suffix.substr(pos, next - pos);
    size_t d             = 0;
    while (d < tok.size() && std::isdigit(static_cast<unsigned char>(tok[d])))
      ++d;
    if (d == 0 || d == tok.size()) return {};
    uint32_t bits = 0;
    for (size_t k = 0; k < d; ++k)
      bits = bits * 10 + static_cast<uint32_t>(tok[k] - '0');
    if (bits == 0) return {};
    fields.push_back({bits, std::string(tok.substr(d))});
    if (next == std::string_view::npos) break;
    pos = next + 1;
  }
  return fields;
}

bitextract_spec_result parse_bitextract_spec(std::string_view suffix)
{
  if (suffix != "f16") {
    if (auto alias = parse_float_alias(suffix)) return *alias;
  }

  // Optional leading packed-type token: unambiguous because field tokens
  // always start with a digit (`<bits><name>`).
  cudf::data_type explicit_type(cudf::type_id::EMPTY);
  std::string_view fields_suffix = suffix;
  if (!suffix.empty() && !std::isdigit(static_cast<unsigned char>(suffix[0]))) {
    size_t us = suffix.find('_');
    if (us == std::string_view::npos) return {};
    explicit_type = parse_packed_type_token(suffix.substr(0, us));
    if (explicit_type.id() == cudf::type_id::EMPTY) return {};
    fields_suffix = suffix.substr(us + 1);
  }

  auto fields = parse_bitfield_list(fields_suffix);
  if (fields.empty()) return {};

  cudf::data_type out_type = explicit_type;
  if (out_type.id() == cudf::type_id::EMPTY) {
    uint32_t total_bits = 0;
    for (auto const& f : fields)
      total_bits += f.bits;
    out_type = (total_bits <= 8)    ? cudf::data_type(cudf::type_id::UINT8)
               : (total_bits <= 16) ? cudf::data_type(cudf::type_id::UINT16)
               : (total_bits <= 32) ? cudf::data_type(cudf::type_id::UINT32)
                                    : cudf::data_type(cudf::type_id::UINT64);
  }
  return {std::move(fields), out_type};
}

bitextract_spec_result parse_bitjoin_spec(std::string_view suffix)
{
  if (auto alias = parse_float_alias(suffix)) return *alias;

  size_t last               = suffix.rfind('_');
  std::string_view type_tok = (last == std::string_view::npos) ? suffix : suffix.substr(last + 1);
  std::string_view fields_tok =
    (last == std::string_view::npos) ? std::string_view{} : suffix.substr(0, last);

  cudf::data_type out_type = parse_packed_type_token(type_tok);
  if (out_type.id() == cudf::type_id::EMPTY) return {};

  // Bare output type (e.g. "bitjoin_u32"): caller provides bit ranges in the DSL input list.
  if (fields_tok.empty()) return {{}, out_type};

  auto fields = parse_bitfield_list(fields_tok);
  if (fields.empty()) return {};
  return {std::move(fields), out_type};
}

}  // namespace simpatico
