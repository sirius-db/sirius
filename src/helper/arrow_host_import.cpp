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

#include <bit>       // std::popcount
#include <charconv>  // std::from_chars
#include <cstdint>
#include <optional>
#include <utility>

namespace sirius {

namespace {

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

// "d:<precision>,<scale>[,<bitwidth>]" split into its fields; nullopt for any other format.
struct decimal_format {
  int precision;
  int scale;
  int bitwidth;
};

std::optional<decimal_format> parse_decimal(std::string_view format)
{
  if (format.substr(0, 2) != "d:") { return std::nullopt; }
  const auto fields = format.substr(2);
  const auto first  = fields.find(',');
  if (first == std::string_view::npos) { return std::nullopt; }
  const auto second    = fields.find(',', first + 1);
  const auto precision = parse_int(fields.substr(0, first));
  const auto scale     = parse_int(fields.substr(first + 1, second - first - 1));
  const auto bitwidth  = second == std::string_view::npos ? std::optional<int>{128}
                                                          : parse_int(fields.substr(second + 1));
  if (!precision || !scale || !bitwidth) { return std::nullopt; }
  return decimal_format{*precision, *scale, *bitwidth};
}

// Refused before any buffer is read, so a bad batch costs no device memory.
void refuse_unsupported_shape(std::string_view what,
                              std::size_t index,
                              const std::string& name,
                              const ArrowSchema& child,
                              const logical_type& declared)
{
  const auto refuse = [&](std::string_view reason) {
    throw invalid_input_exception("{}: column {} ({}) {}", what, index, name, reason);
  };
  const auto format = format_of(child);

  if (child.dictionary != nullptr) { refuse("is dictionary-encoded; decode it before pushing"); }
  if (format == "+L" || format == "U" || format == "Z") {
    refuse("has 64-bit offsets (large_list/large_utf8/large_binary); send 32-bit offsets");
  }
  // Timestamps are "ts<unit>:<timezone>"; an empty timezone is a naive timestamp.
  if (format.size() > 4 && format.substr(0, 2) == "ts" && format[3] == ':') {
    refuse("is a timezone-aware timestamp; convert it to a naive timestamp first");
  }
  if (const auto decimal = parse_decimal(format); decimal && decimal->bitwidth == 256) {
    refuse("is a decimal256; cudf has no 256-bit decimal");
  }
  if (declared.id() == type_id::HUGEINT || declared.id() == type_id::UHUGEINT) {
    refuse("is declared " + declared.to_string() +
           "; the GPU has no 128-bit integer, so declare a DECIMAL or a 64-bit integer");
  }
}

// The cudf type `cudf::from_arrow_column` yields for the scalar formats, or nullopt for a format
// not listed here; the check on the imported column covers those.
std::optional<cudf::data_type> cudf_type_of_format(std::string_view format)
{
  using cudf::type_id;
  if (format == "b") { return cudf::data_type{type_id::BOOL8}; }
  if (format == "c") { return cudf::data_type{type_id::INT8}; }
  if (format == "C") { return cudf::data_type{type_id::UINT8}; }
  if (format == "s") { return cudf::data_type{type_id::INT16}; }
  if (format == "S") { return cudf::data_type{type_id::UINT16}; }
  if (format == "i") { return cudf::data_type{type_id::INT32}; }
  if (format == "I") { return cudf::data_type{type_id::UINT32}; }
  if (format == "l") { return cudf::data_type{type_id::INT64}; }
  if (format == "L") { return cudf::data_type{type_id::UINT64}; }
  if (format == "f") { return cudf::data_type{type_id::FLOAT32}; }
  if (format == "g") { return cudf::data_type{type_id::FLOAT64}; }
  if (format == "u") { return cudf::data_type{type_id::STRING}; }
  if (format == "tdD") { return cudf::data_type{type_id::TIMESTAMP_DAYS}; }
  if (format == "tss:") { return cudf::data_type{type_id::TIMESTAMP_SECONDS}; }
  if (format == "tsm:") { return cudf::data_type{type_id::TIMESTAMP_MILLISECONDS}; }
  if (format == "tsu:") { return cudf::data_type{type_id::TIMESTAMP_MICROSECONDS}; }
  if (format == "tsn:") { return cudf::data_type{type_id::TIMESTAMP_NANOSECONDS}; }
  if (const auto decimal = parse_decimal(format); decimal && decimal->bitwidth == 128) {
    return cudf::data_type{type_id::DECIMAL128, -decimal->scale};
  }
  return std::nullopt;
}

// The one disagreement tolerated: fixed-point types with the same scale and different widths.
bool differs_in_width_only(cudf::data_type actual, cudf::data_type expected)
{
  return cudf::is_fixed_point(actual) && cudf::is_fixed_point(expected) &&
         actual.scale() == expected.scale();
}

[[noreturn]] void throw_type_mismatch(std::string_view what,
                                      std::size_t index,
                                      const std::string& name,
                                      const logical_type& declared,
                                      cudf::data_type expected,
                                      cudf::data_type actual)
{
  throw invalid_input_exception("{}: column {} ({}) is declared {} ({}) but carries {} (scale {})",
                                what,
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
  for (; bit < end && (bit % 8) != 0; ++bit) {
    valid += (validity[bit / 8] >> (bit % 8)) & 1;
  }
  for (; bit + 8 <= end; bit += 8) {
    valid += std::popcount(validity[bit / 8]);
  }
  for (; bit < end; ++bit) {
    valid += (validity[bit / 8] >> (bit % 8)) & 1;
  }
  return (end - begin) - valid;
}

const std::uint8_t* validity_of(const ArrowArray& array)
{
  return array.n_buffers > 0 && array.buffers != nullptr
           ? static_cast<const std::uint8_t*>(array.buffers[0])
           : nullptr;
}

void validate_batch(const ArrowSchema* schema,
                    const ArrowArray* array,
                    std::string_view what,
                    const std::vector<std::string>& names,
                    const std::vector<logical_type>& types)
{
  if (schema == nullptr || array == nullptr) {
    throw invalid_input_exception("{}: requires non-null ArrowSchema and ArrowArray pointers",
                                  what);
  }
  // A released struct has dangling buffer pointers.
  if (schema->release == nullptr || array->release == nullptr) {
    throw invalid_input_exception("{}: the ArrowSchema/ArrowArray were already released", what);
  }
  if (names.size() != types.size()) {
    throw internal_exception(
      "{}: {} declared names but {} declared types", what, names.size(), types.size());
  }
  if (format_of(*schema) != "+s") {
    throw invalid_input_exception(
      "{}: the top-level Arrow array must be a struct, not '{}'", what, format_of(*schema));
  }
  if (static_cast<std::size_t>(schema->n_children) != types.size() ||
      static_cast<std::size_t>(array->n_children) != types.size()) {
    throw invalid_input_exception(
      "{}: carries {} columns (schema) / {} columns (array) but the stream declares {}",
      what,
      schema->n_children,
      array->n_children,
      types.size());
  }
  if (array->offset < 0 || array->length < 0) {
    throw invalid_input_exception("{}: the struct has a negative offset ({}) or length ({})",
                                  what,
                                  array->offset,
                                  array->length);
  }
  // A null struct row has no place in a table; cudf would import its children as present.
  if (const auto* validity = validity_of(*array);
      validity != nullptr && array->null_count != 0 &&
      count_nulls(validity, array->offset, array->offset + array->length) > 0) {
    throw invalid_input_exception("{}: the struct array has null rows; a record batch has none",
                                  what);
  }
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (array->children[i]->length < array->offset + array->length) {
      throw invalid_input_exception(
        "{}: column {} ({}) has {} rows but the batch spans rows [{}, {})",
        what,
        i,
        names[i],
        array->children[i]->length,
        array->offset,
        array->offset + array->length);
    }
  }
}

// Per-column checks before any copy. Returns the declared cudf type of every column.
std::vector<cudf::data_type> check_columns(const ArrowSchema& schema,
                                           std::string_view what,
                                           const std::vector<std::string>& names,
                                           const std::vector<logical_type>& types)
{
  std::vector<cudf::data_type> expected;
  expected.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    const auto& child = *schema.children[i];
    refuse_unsupported_shape(what, i, names[i], child, types[i]);
    expected.push_back(get_cudf_type(types[i]));
    const auto carried = cudf_type_of_format(format_of(child));
    if (carried && *carried != expected[i] && !differs_in_width_only(*carried, expected[i])) {
      throw_type_mismatch(what, i, names[i], types[i], expected[i], *carried);
    }
    // Narrowing to the declared width truncates digits beyond the declared precision.
    const auto decimal = parse_decimal(format_of(child));
    if (decimal && types[i].id() == type_id::DECIMAL &&
        decimal->precision > types[i].decimal_precision()) {
      throw invalid_input_exception("{}: column {} ({}) is declared {} but carries precision {}",
                                    what,
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

std::unique_ptr<cudf::table> import_arrow_host_table(const ArrowSchema* schema,
                                                     const ArrowArray* array,
                                                     std::string_view what,
                                                     const std::vector<std::string>& names,
                                                     const std::vector<logical_type>& types,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  validate_batch(schema, array, what, names, types);
  const auto expected = check_columns(*schema, what, names, types);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(types.size());
  try {
    for (std::size_t i = 0; i < types.size(); ++i) {
      const ArrowArray window = windowed_child(*array, i);
      auto column             = cudf::from_arrow_column(schema->children[i], &window, stream, mr);
      const auto actual       = column->type();
      if (actual != expected[i]) {
        if (!differs_in_width_only(actual, expected[i])) {
          throw_type_mismatch(what, i, names[i], types[i], expected[i], actual);
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
