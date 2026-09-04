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

// The Arrow C Data Interface structs (ArrowSchema, ArrowArray). DuckDB's copy is the one every
// other TU of this library sees (sirius_ffi.cpp reads result streams through it), it is a hard
// dependency in every build flavour, and it is layout-identical to Apache Arrow's abi.h. Apache
// Arrow's own header is NOT a dependency of this tree (cudf's vcpkg port brings nanoarrow, the
// pixi default env only has it through pyarrow), so including it here would break the vcpkg
// build and give libsirius two definitions of the same struct.
#include "duckdb/common/arrow/arrow.hpp"

#include <cudf/interop.hpp>
#include <cudf/unary.hpp>                      // cudf::cast, cudf::is_supported_cast
#include <cudf/utilities/traits.hpp>           // cudf::is_fixed_point
#include <cudf/utilities/type_dispatcher.hpp>  // cudf::type_to_name

#include <bit>       // std::popcount
#include <charconv>  // std::from_chars
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace sirius {

namespace {

std::string_view format_of(const ArrowSchema& schema)
{
  return schema.format == nullptr ? std::string_view{} : std::string_view{schema.format};
}

// Shapes cudf would import into something the engine cannot consume, or would import with a
// silently changed meaning. Refused by name so the producer learns which column and why; checked
// before any buffer is read, so a bad batch costs no device memory.
void refuse_unsupported_shape(std::string_view what,
                              std::size_t index,
                              const std::string& name,
                              const ArrowSchema& child,
                              const logical_type& declared)
{
  const auto refuse = [&](const std::string& reason) {
    throw invalid_input_exception("{}: column {} ({}) {}", what, index, name, reason);
  };
  const auto format = format_of(child);

  if (child.dictionary != nullptr) {
    refuse("is dictionary-encoded; the engine consumes plain columns — decode it before pushing");
  }
  if (format == "+L") {
    refuse("is a large_list (64-bit offsets); the engine's list columns use 32-bit offsets");
  }
  if (format == "U") {
    refuse(
      "is large_utf8 (64-bit offsets); the engine's string kernels take 32-bit offsets — send "
      "utf8");
  }
  if (format == "Z") {
    refuse(
      "is large_binary (64-bit offsets); the engine's string kernels take 32-bit offsets — send "
      "binary");
  }
  // Timestamps are "ts<unit>:<timezone>"; an empty timezone is a naive timestamp.
  if (format.size() > 4 && format.substr(0, 2) == "ts" && format[3] == ':') {
    refuse("is a timezone-aware timestamp (" + std::string(format.substr(4)) +
           "); the engine has no timezone carrier — convert to a naive timestamp first");
  }
  // Decimals are "d:<precision>,<scale>[,<bitwidth>]"; a bitwidth of 256 has no cudf carrier.
  if (format.substr(0, 2) == "d:" && format.size() > 4 &&
      format.substr(format.size() - 4) == ",256") {
    refuse("is a decimal256; cudf has no 256-bit decimal carrier");
  }
  if (declared.id() == type_id::HUGEINT || declared.id() == type_id::UHUGEINT) {
    refuse("is declared " + declared.to_string() +
           "; the GPU has no 128-bit integer carrier and the value would be narrowed to 64 bits — "
           "declare a DECIMAL or a 64-bit integer instead");
  }
}

// The cudf type `cudf::from_arrow` yields for the scalar formats the engine's columns arrive in,
// or nullopt for a format this table does not know (nested types, binary, float16, date64,
// durations, the explicit-bitwidth decimals): the check on the imported table is the backstop for
// those. Knowing the type from the format string alone is what lets a plain type mismatch be
// refused before any buffer is copied, as `push_packed` refuses one before its deep copy.
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
  // Naive timestamps only ("ts<unit>:"); the timezone-aware form was refused by name above.
  if (format == "tss:") { return cudf::data_type{type_id::TIMESTAMP_SECONDS}; }
  if (format == "tsm:") { return cudf::data_type{type_id::TIMESTAMP_MILLISECONDS}; }
  if (format == "tsu:") { return cudf::data_type{type_id::TIMESTAMP_MICROSECONDS}; }
  if (format == "tsn:") { return cudf::data_type{type_id::TIMESTAMP_NANOSECONDS}; }
  // "d:<precision>,<scale>" is decimal128; a third field is an explicit bitwidth, not mapped here.
  if (format.substr(0, 2) == "d:") {
    const auto comma = format.find(',');
    if (comma == std::string_view::npos || format.find(',', comma + 1) != std::string_view::npos) {
      return std::nullopt;
    }
    const auto digits = format.substr(comma + 1);
    if (digits.empty()) { return std::nullopt; }
    int scale             = 0;
    const auto* const end = digits.data() + digits.size();
    const auto [ptr, ec]  = std::from_chars(digits.data(), end, scale);
    if (ec != std::errc{} || ptr != end) { return std::nullopt; }
    return cudf::data_type{type_id::DECIMAL128, -scale};
  }
  return std::nullopt;
}

// The one disagreement the import tolerates: fixed-point types that differ only in storage width.
// Arrow producers emit decimal128 whatever the precision; the declared precision picks the
// engine's width and the column is narrowed to it after the copy.
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
  // Two fixed-point types that differ in scale would pass the width cast with a silently shifted
  // value, so name the scales: the width alone would send the producer the wrong way.
  std::string scale_note;
  if (cudf::is_fixed_point(actual) && cudf::is_fixed_point(expected) &&
      actual.scale() != expected.scale()) {
    scale_note = "; declared scale " + std::to_string(-expected.scale()) + " but carries scale " +
                 std::to_string(-actual.scale());
  }
  throw invalid_input_exception("{}: column {} ({}) is declared {} ({}) but carries {}{}",
                                what,
                                index,
                                name,
                                declared.to_string(),
                                cudf::type_to_name(expected),
                                cudf::type_to_name(actual),
                                scale_note);
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

// Storage for a struct whose window had to be pushed into its children: shallow copies of the
// caller's structs that point at the caller's buffers and are released by nobody.
struct windowed_struct {
  ArrowArray top{};
  std::vector<ArrowArray> children;
  std::vector<ArrowArray*> child_pointers;
};

// A producer may slice the struct itself (Arrow C++ `StructArray::Slice`) rather than its columns
// (arrow-rs `RecordBatch::slice`), or hand over children longer than the struct; both are legal
// C Data Interface shapes whose rows are the struct's [offset, offset + length). `cudf::from_arrow`
// imports each child by the child's own offset and length and ignores the struct's, so the window
// is pushed into shallow copies of the children first, the normalization arrow-rs applies on
// export. Only the window's rows are copied. Returns `array` itself when nothing needs pushing.
const ArrowArray* window_struct_children(const ArrowArray& array,
                                         std::string_view what,
                                         const std::vector<std::string>& names,
                                         windowed_struct& storage)
{
  if (array.offset < 0 || array.length < 0) {
    throw invalid_input_exception("{}: the struct array has a negative offset ({}) or length ({})",
                                  what,
                                  array.offset,
                                  array.length);
  }
  const auto count  = static_cast<std::size_t>(array.n_children);
  bool needs_window = array.offset != 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& child = *array.children[i];
    if (child.length < array.offset + array.length) {
      throw invalid_input_exception(
        "{}: column {} ({}) has {} rows but the batch spans rows [{}, {}) of its columns",
        what,
        i,
        names[i],
        child.length,
        array.offset,
        array.offset + array.length);
    }
    needs_window = needs_window || child.length != array.length;
  }
  if (!needs_window) { return &array; }

  storage.children.reserve(count);
  storage.child_pointers.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    ArrowArray child = *array.children[i];
    child.offset += array.offset;
    child.length = array.length;
    // null_count describes the whole child; the window's is recounted unless it is known to be 0.
    const auto* validity =
      child.n_buffers > 0 ? static_cast<const std::uint8_t*>(child.buffers[0]) : nullptr;
    child.null_count = (validity == nullptr || child.null_count == 0)
                         ? 0
                         : count_nulls(validity, child.offset, child.offset + child.length);
    storage.children.push_back(child);
  }
  for (auto& child : storage.children) {
    storage.child_pointers.push_back(&child);
  }
  storage.top          = array;
  storage.top.offset   = 0;
  storage.top.children = storage.child_pointers.data();
  return &storage.top;
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
  if (schema == nullptr || array == nullptr) {
    throw invalid_input_exception("{}: requires non-null ArrowSchema and ArrowArray pointers",
                                  what);
  }
  // The C Data Interface marks a released struct by nulling its `release`; its buffer pointers
  // are dangling from then on, so this has to be a loud error, not a read of freed memory.
  if (schema->release == nullptr || array->release == nullptr) {
    throw invalid_input_exception(
      "{}: the ArrowSchema/ArrowArray were already released (release == NULL)", what);
  }
  if (names.size() != types.size()) {
    throw internal_exception(
      "{}: {} declared names but {} declared types", what, names.size(), types.size());
  }
  if (format_of(*schema) != "+s") {
    throw invalid_input_exception(
      "{}: the top-level Arrow array must be a struct (one record batch), but its format is '{}'",
      what,
      format_of(*schema));
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

  std::vector<cudf::data_type> expected;
  expected.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    const auto& child = *schema->children[i];
    refuse_unsupported_shape(what, i, names[i], child, types[i]);
    expected.push_back(get_cudf_type(types[i]));
    // A plain type mismatch, refused from the format string before any buffer is copied. Formats
    // the table does not know fall through to the check on the imported table.
    const auto carried = cudf_type_of_format(format_of(child));
    if (carried && *carried != expected[i] && !differs_in_width_only(*carried, expected[i])) {
      throw_type_mismatch(what, i, names[i], types[i], expected[i], *carried);
    }
  }

  windowed_struct window;
  const ArrowArray* rows = window_struct_children(*array, what, names, window);

  // Host -> device copy of every buffer; cudf owns the result, the input is not released.
  std::vector<std::unique_ptr<cudf::column>> columns;
  try {
    columns = cudf::from_arrow(schema, rows, stream, mr)->release();
    for (std::size_t i = 0; i < columns.size(); ++i) {
      const auto actual = columns[i]->type();
      if (actual == expected[i]) { continue; }
      if (differs_in_width_only(actual, expected[i]) &&
          cudf::is_supported_cast(actual, expected[i])) {
        columns[i] = cudf::cast(columns[i]->view(), expected[i], stream, mr);
        continue;
      }
      throw_type_mismatch(what, i, names[i], types[i], expected[i], actual);
    }
  } catch (...) {
    // The copies may still be reading the producer's buffers (from pinned host memory a
    // cudaMemcpyAsync is truly asynchronous), and the contract lets the producer free them the
    // moment control returns, on the error path too. The device memory is released
    // stream-ordered by the columns themselves.
    stream.synchronize_no_throw();
    throw;
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius
