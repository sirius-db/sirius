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

#include <cstddef>
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
    refuse_unsupported_shape(what, i, names[i], *schema->children[i], types[i]);
    expected.push_back(get_cudf_type(types[i]));
  }

  // Host -> device copy of every buffer; cudf owns the result, the input is not released.
  auto imported = cudf::from_arrow(schema, array, stream, mr);
  auto columns  = imported->release();

  for (std::size_t i = 0; i < columns.size(); ++i) {
    const auto actual = columns[i]->type();
    if (actual == expected[i]) { continue; }
    // Arrow producers emit decimal128 whatever the precision; the declared precision picks the
    // engine's storage width, so narrow to it when only the width differs.
    if (cudf::is_fixed_point(actual) && cudf::is_fixed_point(expected[i]) &&
        actual.scale() == expected[i].scale() && cudf::is_supported_cast(actual, expected[i])) {
      columns[i] = cudf::cast(columns[i]->view(), expected[i], stream, mr);
      continue;
    }
    // Two fixed-point types that differ in scale would pass the width cast above with a silently
    // shifted value, so name the scales: the width alone would send the producer the wrong way.
    std::string scale_note;
    if (cudf::is_fixed_point(actual) && cudf::is_fixed_point(expected[i]) &&
        actual.scale() != expected[i].scale()) {
      scale_note = "; declared scale " + std::to_string(-expected[i].scale()) +
                   " but carries scale " + std::to_string(-actual.scale());
    }
    throw invalid_input_exception("{}: column {} ({}) is declared {} ({}) but carries {}{}",
                                  what,
                                  i,
                                  names[i],
                                  types[i].to_string(),
                                  cudf::type_to_name(expected[i]),
                                  cudf::type_to_name(actual),
                                  scale_note);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius
