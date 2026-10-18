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

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>           // cudf::is_fixed_point
#include <cudf/utilities/type_dispatcher.hpp>  // cudf::type_to_name

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>       // std::popcount
#include <charconv>  // std::from_chars
#include <cstdint>
#include <cstring>
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
  if (format.starts_with(decimal_prefix) &&
      (!decimal || decimal->bitwidth != decimal128_bits || decimal->precision < 1 ||
       decimal->precision > 38 || decimal->scale < 0 || decimal->scale > decimal->precision)) {
    refuse("has an invalid or unsupported decimal128 format");
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

std::int32_t string_offset(const void* buffer, std::int64_t index)
{
  std::int32_t value;
  std::memcpy(
    &value, static_cast<const std::uint8_t*>(buffer) + index * sizeof(value), sizeof(value));
  return value;
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
  if (schema->dictionary != nullptr || array->dictionary != nullptr || array->n_buffers != 1 ||
      array->buffers == nullptr || array->null_count < -1 || array->null_count > array->length ||
      (array->null_count > 0 && array->buffers[0] == nullptr)) {
    throw invalid_input_exception("{}: invalid struct layout or null metadata", error_prefix);
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
      array->offset > max_rows - array->length) {
    throw invalid_input_exception(
      "{}: the struct window (offset {}, length {}) is invalid or longer than cudf's {} rows",
      error_prefix,
      array->offset,
      array->length,
      max_rows);
  }
  if (types.empty() && array->length != 0) {
    throw invalid_input_exception(
      "{}: a nonempty zero-column batch requires a physical row-count carrier", error_prefix);
  }
  // A null struct row has no place in a table; cudf would import its children as present.
  if (const auto* validity = validity_of(*array);
      array->null_count > 0 ||
      (validity != nullptr &&
       count_nulls(validity, array->offset, array->offset + array->length) > 0)) {
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

struct checked_type {
  cudf::data_type target;
  std::optional<decimal_format> decimal;
};

// Parse types once, before inspecting values or starting any device work.
std::vector<checked_type> check_columns(const ArrowSchema& schema,
                                        std::string_view error_prefix,
                                        const std::vector<std::string>& names,
                                        const std::vector<logical_type>& types)
{
  std::vector<checked_type> expected;
  expected.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    const auto& child  = *schema.children[i];
    const auto decimal = parse_decimal(format_of(child));
    refuse_unsupported_shape(error_prefix, i, names[i], child, decimal);
    // No 128-bit integer on the GPU, and nested children are not type-checked.
    const auto declared        = types[i].id();
    const auto refuse_declared = [&]() {
      throw invalid_input_exception("{}: column {} ({}) is declared {}, which cannot be imported",
                                    error_prefix,
                                    i,
                                    names[i],
                                    types[i].to_string());
    };
    if (declared == type_id::HUGEINT || declared == type_id::UHUGEINT ||
        (declared == type_id::DECIMAL &&
         (types[i].decimal_precision() == 0 || types[i].decimal_precision() > 38 ||
          types[i].decimal_scale() > types[i].decimal_precision()))) {
      refuse_declared();
    }
    const auto mapped = try_get_cudf_type(types[i]);
    if (!mapped || cudf::is_nested(*mapped)) { refuse_declared(); }
    const auto target  = *mapped;
    const auto carried = cudf_type_of_format(format_of(child), decimal);
    if (!carried) {
      throw invalid_input_exception("{}: column {} ({}) has unsupported Arrow format '{}'",
                                    error_prefix,
                                    i,
                                    names[i],
                                    format_of(child));
    }
    if (*carried != target && !differs_in_width_only(*carried, target)) {
      throw_type_mismatch(error_prefix, i, names[i], types[i], target, *carried);
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
    expected.push_back({target, decimal});
  }
  return expected;
}

std::uint64_t allocation_estimate(std::uint64_t bytes)
{
  constexpr std::uint64_t alignment = 256;
  return bytes == 0 ? 0 : ((bytes - 1) / alignment + 1) * alignment;
}

}  // namespace

arrow_host_import_plan inspect_arrow_host_batch(const ArrowArray* array,
                                                const ArrowSchema* schema,
                                                std::string_view error_prefix,
                                                const std::vector<std::string>& names,
                                                const std::vector<logical_type>& types,
                                                std::pmr::memory_resource& host_resource)
{
  validate_batch(array, schema, error_prefix, names, types);
  const auto expected = check_columns(*schema, error_prefix, names, types);
  arrow_host_import_plan plan(host_resource);
  plan.rows_ = static_cast<cudf::size_type>(array->length);
  plan.columns_.reserve(types.size());
  const auto charge = [&](std::uint64_t bytes) {
    bytes = allocation_estimate(bytes);
    if (bytes > std::numeric_limits<std::uint64_t>::max() - plan.bytes_) {
      throw invalid_input_exception("{}: device byte estimate overflows", error_prefix);
    }
    plan.bytes_ += bytes;
  };
  for (std::size_t i = 0; i < types.size(); ++i) {
    const auto& child        = *array->children[i];
    const auto& child_schema = *schema->children[i];
    const auto format        = format_of(child_schema);
    const bool strings       = format == "u";
    const auto refuse        = [&](std::string_view reason) {
      throw invalid_input_exception("{}: column {} ({}) {}", error_prefix, i, names[i], reason);
    };
    if (child.dictionary != nullptr || child.n_children != 0 || child_schema.n_children != 0 ||
        child.n_buffers != (strings ? 3 : 2) || child.buffers == nullptr || child.offset < 0 ||
        child.null_count < -1 || child.null_count > child.length ||
        (child.null_count > 0 && child.buffers[0] == nullptr)) {
      refuse("has invalid scalar layout or null metadata");
    }
    const auto end = array->offset + array->length;
    // Bound the combined window before pointer arithmetic or cuDF's size_type conversions.
    if (child.offset > std::numeric_limits<cudf::size_type>::max() - end) {
      refuse("has a combined child/struct offset outside the supported range");
    }
    const auto begin = child.offset + array->offset;
    const auto stop  = begin + array->length;
    if (array->length > 0 && child.buffers[1] == nullptr) {
      refuse("has no value or offset buffer");
    }
    const auto* validity = validity_of(child);
    const auto nulls     = validity ? count_nulls(validity, begin, stop) : 0;
    if ((child.null_count >= 0 &&
         (nulls > child.null_count || child.null_count > nulls + (child.length - array->length))) ||
        ((child_schema.flags & ARROW_FLAG_NULLABLE) == 0 && nulls != 0)) {
      refuse("has validity bits inconsistent with its null metadata/schema");
    }
    if (strings && array->length == std::numeric_limits<cudf::size_type>::max()) {
      refuse("has too many rows for string offsets");
    }
    if (strings && array->length > 0) {
      if (string_offset(child.buffers[1], begin) < 0) { refuse("has negative string offsets"); }
      for (auto row = begin; row < stop; ++row) {
        if (string_offset(child.buffers[1], row + 1) < string_offset(child.buffers[1], row)) {
          refuse("has decreasing string offsets");
        }
      }
      const auto chars = static_cast<std::uint64_t>(string_offset(child.buffers[1], stop) -
                                                    string_offset(child.buffers[1], begin));
      if (chars != 0 && child.buffers[2] == nullptr) { refuse("has no string data buffer"); }
      charge(chars);
    } else if (!strings) {
      charge(static_cast<std::uint64_t>(array->length) * cudf::size_of(expected[i].target));
      if (expected[i].decimal) {
        // The carried precision is already checked against the target. Validate values too,
        // so narrowing cannot silently truncate an invalid decimal value.
        const auto& decimal = *expected[i].decimal;
        __uint128_t limit   = 1;
        for (int digit = 0; digit < decimal.precision; ++digit) {
          limit *= 10;
        }
        for (auto row = begin; row < stop; ++row) {
          if (validity && ((validity[row / 8] >> (row % 8)) & 1) == 0) { continue; }
          __int128_t value;
          std::memcpy(&value, static_cast<const std::uint8_t*>(child.buffers[1]) + row * 16, 16);
          auto magnitude = static_cast<__uint128_t>(value);
          if (value < 0) { magnitude = __uint128_t{0} - magnitude; }
          if (magnitude >= limit) { refuse("has a decimal value outside its declared precision"); }
        }
      }
    }
    arrow_host_import_plan::column_input input{nullptr,
                                               nulls,
                                               expected[i].target,
                                               std::pmr::vector<std::uint8_t>(&host_resource),
                                               std::pmr::vector<std::int32_t>(&host_resource),
                                               std::pmr::vector<std::uint8_t>(&host_resource)};
    if (strings) {
      input.offsets.resize(static_cast<std::size_t>(array->length) + 1);
      if (array->length != 0) {
        const auto first = string_offset(child.buffers[1], begin);
        for (std::int64_t row = 0; row <= array->length; ++row) {
          input.offsets[row] = string_offset(child.buffers[1], begin + row) - first;
        }
        if (input.offsets.back() != 0) {
          input.source = static_cast<const std::uint8_t*>(child.buffers[2]) + first;
        }
      }
      charge(input.offsets.size() * sizeof(std::int32_t));
    } else if (input.target.id() == cudf::type_id::BOOL8) {
      input.values.resize(array->length);
      const auto* bits = static_cast<const std::uint8_t*>(child.buffers[1]);
      for (std::int64_t row = 0; row < array->length; ++row) {
        input.values[row] = (bits[(begin + row) / 8] >> ((begin + row) % 8)) & 1;
      }
    } else if (expected[i].decimal && input.target.id() != cudf::type_id::DECIMAL128) {
      const auto width = cudf::size_of(input.target);
      input.values.resize(static_cast<std::size_t>(array->length) * width);
      for (std::int64_t row = 0; row < array->length; ++row) {
        const auto source_row = begin + row;
        if (validity && ((validity[source_row / 8] >> (source_row % 8)) & 1) == 0) { continue; }
        __int128_t value;
        std::memcpy(
          &value, static_cast<const std::uint8_t*>(child.buffers[1]) + source_row * 16, 16);
        if (width == sizeof(std::int32_t)) {
          const auto narrowed = static_cast<std::int32_t>(value);
          std::memcpy(input.values.data() + row * width, &narrowed, width);
        } else {
          const auto narrowed = static_cast<std::int64_t>(value);
          std::memcpy(input.values.data() + row * width, &narrowed, width);
        }
      }
    } else if (array->length != 0) {
      input.source =
        static_cast<const std::uint8_t*>(child.buffers[1]) + begin * cudf::size_of(input.target);
    }
    if (nulls != 0) {
      input.validity.resize(cudf::bitmask_allocation_size_bytes(plan.rows_), 0);
      for (std::int64_t row = 0; row < array->length; ++row) {
        const auto bit = (validity[(begin + row) / 8] >> ((begin + row) % 8)) & 1;
        input.validity[row / 8] |= static_cast<std::uint8_t>(bit << (row % 8));
      }
      charge(input.validity.size());
    }
    const auto staging =
      input.values.size() + input.offsets.size() * sizeof(std::int32_t) + input.validity.size();
    if (staging > std::numeric_limits<std::uint64_t>::max() - plan.host_bytes_) {
      refuse("has overflowing host staging size");
    }
    plan.host_bytes_ += staging;
    plan.columns_.push_back(std::move(input));
  }
  return plan;
}

std::unique_ptr<cudf::table> import_arrow_host_table(const arrow_host_import_plan& plan,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  if (plan.rows_ < 0) { throw internal_exception("Cannot import a moved-from Arrow plan"); }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(plan.columns_.size());
  const auto copy = [&](void* destination, const void* source, std::size_t bytes) {
    if (bytes != 0) {
      CUDF_CUDA_TRY(
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream.value()));
    }
  };
  try {
    for (std::size_t i = 0; i < plan.columns_.size(); ++i) {
      const auto& input = plan.columns_[i];
      rmm::device_buffer mask(input.validity.size(), stream, mr);
      copy(mask.data(), input.validity.data(), input.validity.size());
      std::unique_ptr<cudf::column> column;
      if (input.target.id() == cudf::type_id::STRING) {
        auto offsets =
          cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::INT32),
                                        static_cast<cudf::size_type>(input.offsets.size()),
                                        cudf::mask_state::UNALLOCATED,
                                        stream,
                                        mr);
        copy(offsets->mutable_view().head<std::uint8_t>(),
             input.offsets.data(),
             input.offsets.size() * sizeof(std::int32_t));
        rmm::device_buffer chars(static_cast<std::size_t>(input.offsets.back()), stream, mr);
        if (chars.size() != 0) { copy(chars.data(), input.source, chars.size()); }
        column = cudf::make_strings_column(plan.rows_,
                                           std::move(offsets),
                                           std::move(chars),
                                           static_cast<cudf::size_type>(input.null_count),
                                           std::move(mask));
      } else {
        column = cudf::make_fixed_width_column(
          input.target, plan.rows_, cudf::mask_state::UNALLOCATED, stream, mr);
        if (plan.rows_ != 0) {
          const auto width   = cudf::size_of(input.target);
          const auto* source = input.values.empty() ? input.source : input.values.data();
          copy(column->mutable_view().head<std::uint8_t>(),
               source,
               static_cast<std::size_t>(plan.rows_) * width);
        }
        column->set_null_mask(std::move(mask), static_cast<cudf::size_type>(input.null_count));
      }
      columns.push_back(std::move(column));
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    return std::make_unique<cudf::table>(std::move(columns));
  } catch (...) {
    const auto original = std::current_exception();
    columns.clear();
    const auto status = cudaStreamSynchronize(stream.value());
    if (status != cudaSuccess) {
      throw arrow_host_import_drain_error(static_cast<int>(status), original);
    }
    std::rethrow_exception(original);
  }
}

}  // namespace sirius
