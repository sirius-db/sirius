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

#include "helper/arrow_host_import.hpp"

#include "cudf/cudf_utils.hpp"  // sirius::get_cudf_type
#include "sirius/exception.hpp"

// DuckDB's Arrow C structs: a hard dependency of every build flavour and layout-identical to
// Apache Arrow's abi.h, which this tree does not depend on.
#include "duckdb/common/arrow/arrow.hpp"

#include <cudf/interop.hpp>
#include <cudf/unary.hpp>                      // cudf::cast
#include <cudf/utilities/traits.hpp>           // cudf::is_fixed_point
#include <cudf/utilities/type_dispatcher.hpp>  // cudf::type_to_name

#include <algorithm>
#include <array>
#include <bit>       // std::popcount
#include <charconv>  // std::from_chars
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius {

namespace {

// Arrow C Data Interface format strings.
constexpr std::string_view struct_format    = "+s";
constexpr std::string_view decimal_prefix   = "d:";  // "d:<precision>,<scale>[,<bitwidth>]"
constexpr std::string_view timestamp_prefix = "ts";  // "ts<unit>:<timezone>"
constexpr std::array<std::string_view, 3> large_offset_formats{"+L", "U", "Z"};
constexpr int decimal128_bits        = 128;
constexpr int decimal256_bits        = 256;
constexpr std::int64_t bits_per_byte = std::numeric_limits<std::uint8_t>::digits;

// What `cudf::from_arrow_column` yields for each scalar format.
constexpr std::array<std::pair<std::string_view, cudf::type_id>, 17> scalar_formats{{
  {"b", cudf::type_id::BOOL8},
  {"c", cudf::type_id::INT8},
  {"C", cudf::type_id::UINT8},
  {"s", cudf::type_id::INT16},
  {"S", cudf::type_id::UINT16},
  {"i", cudf::type_id::INT32},
  {"I", cudf::type_id::UINT32},
  {"l", cudf::type_id::INT64},
  {"L", cudf::type_id::UINT64},
  {"f", cudf::type_id::FLOAT32},
  {"g", cudf::type_id::FLOAT64},
  {"u", cudf::type_id::STRING},
  {"tdD", cudf::type_id::TIMESTAMP_DAYS},
  {"tss:", cudf::type_id::TIMESTAMP_SECONDS},
  {"tsm:", cudf::type_id::TIMESTAMP_MILLISECONDS},
  {"tsu:", cudf::type_id::TIMESTAMP_MICROSECONDS},
  {"tsn:", cudf::type_id::TIMESTAMP_NANOSECONDS},
}};

std::string_view format_of(const ArrowSchema& schema)
{
  return schema.format == nullptr ? std::string_view{} : std::string_view{schema.format};
}

std::optional<int> parse_int(std::string_view digits)
{
  int value             = 0;
  const auto* const end = digits.data() + digits.size();
  const auto [ptr, ec]  = std::from_chars(digits.data(), end, value);
  if (digits.empty() || ec != std::errc{} || ptr != end) { return std::nullopt; }
  return value;
}

struct decimal_format {
  int precision;
  int scale;
  int bitwidth;
};

// nullopt for a format that is not a decimal.
std::optional<decimal_format> parse_decimal(std::string_view format)
{
  if (!format.starts_with(decimal_prefix)) { return std::nullopt; }
  const auto fields = format.substr(decimal_prefix.size());
  const auto first  = fields.find(',');
  if (first == std::string_view::npos) { return std::nullopt; }
  const auto second    = fields.find(',', first + 1);
  const auto precision = parse_int(fields.substr(0, first));
  const auto scale     = parse_int(fields.substr(first + 1, second - first - 1));
  const auto bitwidth  = second == std::string_view::npos ? std::optional<int>{decimal128_bits}
                                                          : parse_int(fields.substr(second + 1));
  if (!precision || !scale || !bitwidth) { return std::nullopt; }
  return decimal_format{*precision, *scale, *bitwidth};
}

// Refused before any buffer is read, so a bad batch costs no device memory.
void refuse_unsupported_shape(std::string_view error_prefix,
                              std::size_t index,
                              const std::string& name,
                              const ArrowSchema& child,
                              const std::optional<decimal_format>& decimal)
{
  const auto refuse = [&](std::string_view reason) {
    throw invalid_input_exception("{}: column {} ({}) {}", error_prefix, index, name, reason);
  };
  const auto format = format_of(child);

  if (child.dictionary != nullptr) { refuse("is dictionary-encoded; decode it before pushing"); }
  if (std::ranges::find(large_offset_formats, format) != large_offset_formats.end()) {
    refuse("has 64-bit offsets (large_list/large_utf8/large_binary); send 32-bit offsets");
  }
  // An empty timezone is a naive timestamp.
  constexpr auto timezone_colon = timestamp_prefix.size() + 1;
  if (format.starts_with(timestamp_prefix) && format.size() > timezone_colon + 1 &&
      format[timezone_colon] == ':') {
    refuse("is a timezone-aware timestamp; convert it to a naive timestamp first");
  }
  if (decimal && decimal->bitwidth == decimal256_bits) {
    refuse("is a decimal256; cudf has no 256-bit decimal");
  }
}

// nullopt for a format outside the scalar set; the check on the imported column covers those.
std::optional<cudf::data_type> cudf_type_of_format(std::string_view format,
                                                   const std::optional<decimal_format>& decimal)
{
  if (decimal && decimal->bitwidth == decimal128_bits) {
    return cudf::data_type{cudf::type_id::DECIMAL128, -decimal->scale};
  }
  const auto it =
    std::ranges::find(scalar_formats, format, &decltype(scalar_formats)::value_type::first);
  if (it == scalar_formats.end()) { return std::nullopt; }
  return cudf::data_type{it->second};
}

// The one disagreement tolerated: fixed-point types with the same scale and different widths.
bool differs_in_width_only(cudf::data_type actual, cudf::data_type expected)
{
  return cudf::is_fixed_point(actual) && cudf::is_fixed_point(expected) &&
         actual.scale() == expected.scale();
}

[[noreturn]] void throw_type_mismatch(std::string_view error_prefix,
                                      std::size_t index,
                                      const std::string& name,
                                      const logical_type& declared,
                                      cudf::data_type expected,
                                      cudf::data_type actual)
{
  throw invalid_input_exception("{}: column {} ({}) is declared {} ({}) but carries {} (scale {})",
                                error_prefix,
                                index,
                                name,
                                declared.to_string(),
                                cudf::type_to_name(expected),
                                cudf::type_to_name(actual),
                                -actual.scale());
}

// Nulls in bits [begin, end) of an Arrow validity bitmap (LSB first).
std::int64_t count_nulls(const std::uint8_t* validity, std::int64_t begin, std::int64_t end)
{
  std::int64_t valid = 0;
  auto bit           = begin;
  const auto bit_at  = [&](std::int64_t i) {
    return (validity[i / bits_per_byte] >> (i % bits_per_byte)) & 1;
  };
  for (; bit < end && bit % bits_per_byte != 0; ++bit) {
    valid += bit_at(bit);
  }
  for (; bit + bits_per_byte <= end; bit += bits_per_byte) {
    valid += static_cast<std::int64_t>(std::popcount(validity[bit / bits_per_byte]));
  }
  for (; bit < end; ++bit) {
    valid += bit_at(bit);
  }
  return (end - begin) - valid;
}

const std::uint8_t* validity_of(const ArrowArray& array)
{
  return array.n_buffers > 0 && array.buffers != nullptr
           ? static_cast<const std::uint8_t*>(array.buffers[0])
           : nullptr;
}

void validate_batch(const ArrowArray* array,
                    const ArrowSchema* schema,
                    std::string_view error_prefix,
                    const std::vector<std::string>& names,
                    const std::vector<logical_type>& types)
{
  if (schema == nullptr || array == nullptr) {
    throw invalid_input_exception("{}: requires non-null ArrowSchema and ArrowArray pointers",
                                  error_prefix);
  }
  // A released struct has dangling buffer pointers.
  if (schema->release == nullptr || array->release == nullptr) {
    throw invalid_input_exception("{}: the ArrowSchema/ArrowArray were already released",
                                  error_prefix);
  }
  if (names.size() != types.size()) {
    throw internal_exception(
      "{}: {} declared names but {} declared types", error_prefix, names.size(), types.size());
  }
  if (format_of(*schema) != struct_format) {
    throw invalid_input_exception(
      "{}: the top-level Arrow array must be a struct, not '{}'", error_prefix, format_of(*schema));
  }
  if (static_cast<std::size_t>(schema->n_children) != types.size() ||
      static_cast<std::size_t>(array->n_children) != types.size()) {
    throw invalid_input_exception(
      "{}: carries {} columns (schema) / {} columns (array) but the stream declares {}",
      error_prefix,
      schema->n_children,
      array->n_children,
      types.size());
  }
  constexpr std::int64_t max_rows = std::numeric_limits<cudf::size_type>::max();
  if (array->offset < 0 || array->length < 0 || array->length > max_rows ||
      array->offset > std::numeric_limits<std::int64_t>::max() - array->length) {
    throw invalid_input_exception(
      "{}: the struct window (offset {}, length {}) is invalid or longer than cudf's {} rows",
      error_prefix,
      array->offset,
      array->length,
      max_rows);
  }
  // A null struct row has no place in a table; cudf would import its children as present.
  if (const auto* validity = validity_of(*array);
      validity != nullptr && array->null_count != 0 &&
      count_nulls(validity, array->offset, array->offset + array->length) > 0) {
    throw invalid_input_exception("{}: the struct array has null rows; a record batch has none",
                                  error_prefix);
  }
  for (std::size_t i = 0; i < types.size(); ++i) {
    const auto* child_schema = schema->children == nullptr ? nullptr : schema->children[i];
    const auto* child_array  = array->children == nullptr ? nullptr : array->children[i];
    if (child_schema == nullptr || child_array == nullptr || child_schema->release == nullptr ||
        child_array->release == nullptr) {
      throw invalid_input_exception(
        "{}: column {} ({}) is missing or released", error_prefix, i, names[i]);
    }
    if (child_array->length < array->offset + array->length) {
      throw invalid_input_exception(
        "{}: column {} ({}) has {} rows but the batch spans rows [{}, {})",
        error_prefix,
        i,
        names[i],
        child_array->length,
        array->offset,
        array->offset + array->length);
    }
  }
}

// Per-column checks before any copy. Returns the declared cudf type of every column.
std::vector<cudf::data_type> check_columns(const ArrowSchema& schema,
                                           std::string_view error_prefix,
                                           const std::vector<std::string>& names,
                                           const std::vector<logical_type>& types)
{
  std::vector<cudf::data_type> expected;
  expected.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    const auto& child  = *schema.children[i];
    const auto decimal = parse_decimal(format_of(child));
    refuse_unsupported_shape(error_prefix, i, names[i], child, decimal);
    // No 128-bit integer on the GPU, and nested children are not type-checked.
    const auto declared = types[i].id();
    if (declared == type_id::HUGEINT || declared == type_id::UHUGEINT ||
        cudf::is_nested(expected.emplace_back(get_cudf_type(types[i])))) {
      throw invalid_input_exception("{}: column {} ({}) is declared {}, which cannot be imported",
                                    error_prefix,
                                    i,
                                    names[i],
                                    types[i].to_string());
    }
    const auto carried = cudf_type_of_format(format_of(child), decimal);
    if (carried && *carried != expected[i] && !differs_in_width_only(*carried, expected[i])) {
      throw_type_mismatch(error_prefix, i, names[i], types[i], expected[i], *carried);
    }
    // Narrowing to the declared width truncates digits beyond the declared precision.
    if (decimal && types[i].id() == type_id::DECIMAL &&
        decimal->precision > types[i].decimal_precision()) {
      throw invalid_input_exception("{}: column {} ({}) is declared {} but carries precision {}",
                                    error_prefix,
                                    i,
                                    names[i],
                                    types[i].to_string(),
                                    decimal->precision);
    }
  }
  return expected;
}

// Shallow copy of child `i` restricted to the struct's window; it borrows the caller's buffers.
ArrowArray windowed_child(const ArrowArray& array, std::size_t i)
{
  ArrowArray child = *array.children[i];
  if (array.offset == 0 && child.length == array.length) { return child; }
  child.offset += array.offset;
  child.length = array.length;
  // The child's count covers all its rows; -1 makes cudf recount the window from the bitmap.
  if (child.null_count != 0 && validity_of(child) != nullptr) { child.null_count = -1; }
  return child;
}

}  // namespace

std::unique_ptr<cudf::table> import_arrow_host_table(const ArrowArray* array,
                                                     const ArrowSchema* schema,
                                                     std::string_view error_prefix,
                                                     const std::vector<std::string>& names,
                                                     const std::vector<logical_type>& types,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  validate_batch(array, schema, error_prefix, names, types);
  const auto expected = check_columns(*schema, error_prefix, names, types);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(types.size());
  try {
    for (std::size_t i = 0; i < types.size(); ++i) {
      const ArrowArray window = windowed_child(*array, i);
      std::unique_ptr<cudf::column> column;
      try {
        column = cudf::from_arrow_column(schema->children[i], &window, stream, mr);
      } catch (const std::logic_error& e) {  // cudf::logic_error, cudf::data_type_error
        throw invalid_input_exception(
          "{}: column {} ({}) was refused by cudf: {}", error_prefix, i, names[i], e.what());
      }
      const auto actual = column->type();
      if (actual != expected[i]) {
        if (!differs_in_width_only(actual, expected[i])) {
          throw_type_mismatch(error_prefix, i, names[i], types[i], expected[i], actual);
        }
        column = cudf::cast(column->view(), expected[i], stream, mr);
      }
      columns.push_back(std::move(column));
    }
  } catch (...) {
    // The async copies may still read the producer's buffers, which it may free once we return.
    stream.synchronize_no_throw();
    throw;
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius
