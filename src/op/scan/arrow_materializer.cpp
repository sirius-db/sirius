/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow_materializer.hpp"

#include "op/scan/arrow/arrow_device_import.hpp"
#include "op/scan/arrow/arrow_string_offsets.hpp"
#include "transparent/replay_admission.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream_ref>  // IWYU pragma: keep

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace sirius::op::scan {
owned_arrow_batch::owned_arrow_batch(ArrowSchema& s,
                                     ArrowArray& a,
                                     std::shared_ptr<void> permit,
                                     std::pmr::memory_resource& staging_resource,
                                     arrow_metadata_lengths metadata_lengths,
                                     std::shared_ptr<arrow_source_profile const> source)
  : capacity(std::move(permit)),
    staging_resource(&staging_resource),
    metadata_lengths(std::move(metadata_lengths)),
    source(std::move(source))
{
  if (!s.release || !a.release || !capacity)
    throw std::invalid_argument("Arrow input requires ownership and capacity");
  schema    = s;
  array     = a;
  s.release = nullptr;
  a.release = nullptr;
}
void owned_arrow_batch::bind_staging(std::shared_ptr<void> owner,
                                     std::pmr::memory_resource& resource)
{
  if (!owner || staging_owner) throw std::logic_error("Arrow staging already bound or absent");
  staging_owner    = std::move(owner);
  staging_resource = &resource;
}
void owned_arrow_batch::bind_credit(std::shared_ptr<void> value)
{
  if (!value || credit) throw std::logic_error("Arrow credit already bound or absent");
  credit = std::move(value);
}
owned_arrow_batch::~owned_arrow_batch()
{
  if (array.release) array.release(&array);
  if (schema.release) schema.release(&schema);
}

namespace {
[[noreturn]] void invalid()
{
  throw std::invalid_argument(
    "arrow_layout_unsupported: unsupported or invalid arrow_host_flat_v1 input");
}
cudf::data_type type_for(std::string_view f)
{
  using T = cudf::type_id;
  if (f == "b") return cudf::data_type{T::BOOL8};
  if (f == "c") return cudf::data_type{T::INT8};
  if (f == "s") return cudf::data_type{T::INT16};
  if (f == "i") return cudf::data_type{T::INT32};
  if (f == "l") return cudf::data_type{T::INT64};
  if (f == "C") return cudf::data_type{T::UINT8};
  if (f == "S") return cudf::data_type{T::UINT16};
  if (f == "I") return cudf::data_type{T::UINT32};
  if (f == "L") return cudf::data_type{T::UINT64};
  if (f == "f") return cudf::data_type{T::FLOAT32};
  if (f == "g") return cudf::data_type{T::FLOAT64};
  if (f == "u" || f == "U") return cudf::data_type{T::STRING};
  if (f == "tdD") return cudf::data_type{T::TIMESTAMP_DAYS};
  if (f == "tss:") return cudf::data_type{T::TIMESTAMP_SECONDS};
  if (f == "tsm:") return cudf::data_type{T::TIMESTAMP_MILLISECONDS};
  if (f == "tsu:") return cudf::data_type{T::TIMESTAMP_MICROSECONDS};
  if (f == "tsn:") return cudf::data_type{T::TIMESTAMP_NANOSECONDS};
  if (f.starts_with("d:")) {
    int32_t precision = 0, scale = 0;
    auto p = std::from_chars(f.data() + 2, f.data() + f.size(), precision);
    if (p.ec != std::errc{} || p.ptr == f.data() + f.size() || *p.ptr != ',') invalid();
    auto s = std::from_chars(p.ptr + 1, f.data() + f.size(), scale);
    if (s.ec != std::errc{} ||
        (s.ptr != f.data() + f.size() &&
         std::string_view(s.ptr, f.data() + f.size() - s.ptr) != ",128") ||
        precision < 1 || precision > 38 || scale == std::numeric_limits<int32_t>::min())
      invalid();
    // Arrow's scale may be negative or greater than precision. Import preserves
    // the unscaled integer; only negation into cuDF's scale must be representable.
    return cudf::data_type{T::DECIMAL128, -scale};
  }
  invalid();
}
uint8_t const* validity_for(ArrowArray const& array)
{
  return array.null_count == 0 ? nullptr : static_cast<uint8_t const*>(array.buffers[0]);
}
int64_t count_nulls(uint8_t const* validity, int64_t begin, int64_t end)
{
  int64_t count = 0;
  if (validity) {
    for (auto row = begin; row < end; ++row)
      count += !(validity[row / 8] & (1U << (row % 8)));
  }
  return count;
}
struct normalized_column {
  explicit normalized_column(cudf::data_type type, std::pmr::memory_resource& resource)
    : type(type), values(&resource), offsets(&resource), validity(&resource)
  {
  }
  cudf::data_type type;
  void const* source    = nullptr;
  cudf::size_type nulls = 0;
  std::pmr::vector<uint8_t> values;
  std::pmr::vector<int32_t> offsets;
  std::pmr::vector<uint8_t> validity;
};

struct normalized_batch {
  normalized_batch(std::shared_ptr<owned_arrow_batch> owner, std::pmr::memory_resource& resource)
    : owner(std::move(owner)), columns(&resource)
  {
  }
  // Release PMR buffers before the producer/capacity owner that keeps the resource alive.
  std::shared_ptr<owned_arrow_batch> owner;
  std::pmr::vector<normalized_column> columns;
};

void normalize_columns(normalized_batch& batch,
                       ArrowSchema const& schema,
                       ArrowArray const& array,
                       std::vector<cudf::data_type> const& types,
                       std::pmr::memory_resource& resource,
                       bool stage_payload = false)
{
  batch.columns.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    auto const& child    = *array.children[i];
    auto const begin     = child.offset + array.offset;
    auto const end       = begin + array.length;
    auto const* validity = validity_for(child);
    auto& column         = batch.columns.emplace_back(types[i], resource);
    column.nulls         = static_cast<cudf::size_type>(count_nulls(validity, begin, end));
    if (types[i].id() == cudf::type_id::STRING) {
      column.offsets.resize(static_cast<std::size_t>(array.length) + 1);
      if (array.length) {
        detail::arrow_string_offsets offsets(schema.children[i]->format, child.buffers[1]);
        auto first = offsets[begin];
        for (int64_t row = 0; row <= array.length; ++row)
          column.offsets[row] = detail::arrow_string_window_size(first, offsets[begin + row]);
        if (column.offsets.back())
          column.source = static_cast<uint8_t const*>(child.buffers[2]) + first;
      }
    } else if (types[i].id() == cudf::type_id::BOOL8) {
      column.values.resize(array.length);
      auto const* bits = static_cast<uint8_t const*>(child.buffers[1]);
      for (int64_t row = 0; row < array.length; ++row)
        column.values[row] = (bits[(begin + row) / 8] >> ((begin + row) % 8)) & 1U;
    } else if (array.length) {
      column.source =
        static_cast<uint8_t const*>(child.buffers[1]) + begin * cudf::size_of(types[i]);
    }
    if (stage_payload && column.source) {
      auto bytes = types[i].id() == cudf::type_id::STRING
                     ? static_cast<size_t>(column.offsets.back())
                     : static_cast<size_t>(array.length) * cudf::size_of(types[i]);
      column.values.resize(bytes);
      std::memcpy(column.values.data(), column.source, bytes);
      column.source = column.values.data();
    }
    if (column.nulls) {
      column.validity.resize(cudf::bitmask_allocation_size_bytes(array.length), 0);
      for (int64_t row = 0; row < array.length; ++row) {
        auto bit = (validity[(begin + row) / 8] >> ((begin + row) % 8)) & 1U;
        column.validity[row / 8] |= static_cast<uint8_t>(bit << (row % 8));
      }
    }
  }
}

std::unique_ptr<cudf::table> copy_columns(normalized_batch const& batch,
                                          cudf::size_type rows,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(batch.columns.size());
  auto copy = [&](void* destination, void const* source, std::size_t bytes) {
    if (bytes)
      CUDF_CUDA_TRY(
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream.get()));
  };
  for (auto const& input : batch.columns) {
    // create_null_mask returns the mask buffer type of the linked cuDF, so the same
    // code builds against the pinned release and the nightly API.
    auto const mask_state =
      input.nulls ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED;
    auto mask = cudf::create_null_mask(rows, mask_state, stream, mr);
    copy(mask.data(), input.validity.data(), input.validity.size());
    std::unique_ptr<cudf::column> column;
    if (input.type.id() == cudf::type_id::STRING) {
      auto offsets =
        cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                      static_cast<cudf::size_type>(input.offsets.size()),
                                      cudf::mask_state::UNALLOCATED,
                                      stream,
                                      mr);
      copy(offsets->mutable_view().head<uint8_t>(),
           input.offsets.data(),
           input.offsets.size() * sizeof(int32_t));
      rmm::device_buffer chars(static_cast<std::size_t>(input.offsets.back()), stream, mr);
      copy(chars.data(), input.source, chars.size());
      column = cudf::make_strings_column(
        rows, std::move(offsets), std::move(chars), input.nulls, std::move(mask));
    } else {
      column =
        cudf::make_fixed_width_column(input.type, rows, cudf::mask_state::UNALLOCATED, stream, mr);
      copy(column->mutable_view().head<uint8_t>(),
           input.values.empty() ? input.source : input.values.data(),
           static_cast<std::size_t>(rows) * cudf::size_of(input.type));
      column->set_null_mask(std::move(mask), input.nulls);
    }
    columns.push_back(std::move(column));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}
void validate_metadata(char const* metadata, arrow_metadata_lengths const& metadata_lengths)
{
  if (!metadata) return;
  auto const bound = metadata_lengths.find(metadata);
  if (bound == metadata_lengths.end()) invalid();
  auto const* cursor = reinterpret_cast<std::uint8_t const*>(metadata);
  auto remaining     = bound->second;
  auto require       = [&](std::size_t bytes) {
    if (bytes > remaining) invalid();
  };
  auto advance = [&](std::size_t bytes) {
    cursor += bytes;
    remaining -= bytes;
  };
  auto integer = [&]() {
    require(sizeof(int32_t));
    int32_t n;
    std::memcpy(&n, cursor, sizeof(n));
    advance(sizeof(n));
    if (n < 0 || n > (1 << 20)) invalid();
    return n;
  };
  auto entries = integer();
  if (entries > 4096) invalid();
  for (int32_t i = 0; i < entries; ++i) {
    auto key_length = integer();
    require(static_cast<std::size_t>(key_length));
    std::string_view key(reinterpret_cast<char const*>(cursor), key_length);
    advance(key_length);
    auto value_length = integer();
    require(static_cast<std::size_t>(value_length));
    advance(value_length);
    if (key == "ARROW:extension:name" || key == "ARROW:extension:metadata") invalid();
  }
}
void common(ArrowSchema const& s,
            ArrowArray const& a,
            arrow_metadata_lengths const& metadata_lengths)
{
  if (!s.format || !s.release || !a.release || s.dictionary || a.dictionary || a.length < 0 ||
      a.offset < 0 || a.offset > std::numeric_limits<int64_t>::max() - a.length ||
      a.n_buffers < 1 || a.null_count < -1 || a.null_count > a.length || !a.buffers)
    invalid();
  validate_metadata(s.metadata, metadata_lengths);
  if (a.null_count > 0 && !a.buffers[0]) invalid();
}
}  // namespace

namespace {
std::vector<cudf::data_type> validate_flat(ArrowSchema const& schema,
                                           ArrowArray const& array,
                                           arrow_metadata_lengths const& metadata_lengths,
                                           bool payload)
{
  if constexpr (std::endian::native != std::endian::little) invalid();
  common(schema, array, metadata_lengths);
  if (std::string_view(schema.format) != "+s" || schema.n_children < 0 ||
      schema.n_children != array.n_children || array.n_buffers != 1 ||
      (array.n_children && (!array.children || !schema.children)))
    invalid();
  if (payload && array.length > std::numeric_limits<cudf::size_type>::max()) invalid();
  if (array.null_count > 0) invalid();
  if (payload && array.null_count != 0 && array.buffers[0]) {
    auto bits = static_cast<uint8_t const*>(array.buffers[0]);
    for (int64_t i = array.offset; i < array.offset + array.length; ++i)
      if (!(bits[i / 8] & (1U << (i % 8)))) invalid();
  }
  std::vector<cudf::data_type> types;
  for (int64_t i = 0; i < array.n_children; ++i) {
    if (!array.children[i] || !schema.children[i]) invalid();
    auto const& a = *array.children[i];
    auto const& s = *schema.children[i];
    common(s, a, metadata_lengths);
    if (a.n_children || s.n_children || a.length < array.offset + array.length) invalid();
    auto type    = type_for(s.format);
    bool strings = type.id() == cudf::type_id::STRING;
    if (strings && array.length >= std::numeric_limits<cudf::size_type>::max()) invalid();
    if (a.n_buffers != (strings ? 3 : 2)) invalid();
    if (array.length && !a.buffers[1]) invalid();
    auto const begin = a.offset + array.offset;
    auto const end   = begin + array.length;
    types.push_back(type);
    if (!payload) continue;
    auto const* validity = validity_for(a);
    auto const nulls     = count_nulls(validity, begin, end);
    if ((a.null_count >= 0 &&
         (nulls > a.null_count || a.null_count > nulls + (a.length - array.length))) ||
        (!(s.flags & ARROW_FLAG_NULLABLE) && nulls))
      invalid();
    if (strings && array.length) {
      detail::arrow_string_offsets offsets(s.format, a.buffers[1]);
      if (array.length == std::numeric_limits<cudf::size_type>::max() || offsets[begin] < 0)
        invalid();
      for (auto row = begin; row < end; ++row)
        if (offsets[row + 1] < offsets[row]) invalid();
      auto bytes = detail::arrow_string_window_size(offsets[begin], offsets[end]);
      if (bytes && !a.buffers[2]) invalid();
    } else if (type.id() == cudf::type_id::DECIMAL128) {
      int precision = 0;
      std::string_view format(s.format);
      (void)std::from_chars(format.data() + 2, format.data() + format.size(), precision);
      __uint128_t limit = 1;
      for (int digit = 0; digit < precision; ++digit)
        limit *= 10;
      for (auto row = begin; row < end; ++row) {
        if (validity && !(validity[row / 8] & (1U << (row % 8)))) continue;
        __int128_t value;
        std::memcpy(&value, static_cast<uint8_t const*>(a.buffers[1]) + row * 16, 16);
        auto magnitude = static_cast<__uint128_t>(value);
        if (value < 0) magnitude = __uint128_t{0} - magnitude;
        if (magnitude >= limit) invalid();
      }
    }
  }
  return types;
}
}  // namespace

std::vector<cudf::data_type> validate_arrow_host_flat(ArrowSchema const& schema,
                                                      ArrowArray const& array,
                                                      arrow_metadata_lengths const& lengths)
{
  return validate_flat(schema, array, lengths, true);
}

namespace {
uint64_t checked_add(uint64_t a, uint64_t b)
{
  if (b > UINT64_MAX - a) invalid();
  return a + b;
}
uint64_t checked_mul(uint64_t a, uint64_t b)
{
  if (a && b > UINT64_MAX / a) invalid();
  return a * b;
}
uint64_t mask_bytes(uint64_t rows)
{
  // cuDF pads validity to 64 bytes. Metadata preflight must not narrow rows
  // before the separate R_max admission check has classified an oversized root.
  return checked_mul(checked_add(rows, 511) / 512, 64);
}
std::string canonical_format(std::string_view format)
{
  (void)type_for(format);
  if (format.starts_with("d:") && format.ends_with(",128")) format.remove_suffix(4);
  return std::string(format);
}
}  // namespace

uint64_t arrow_input_estimate::total() const
{
  return checked_add(checked_add(retained, staging), checked_add(destination, control));
}

void validate_arrow_binding(arrow_bound_schema const& schema,
                            materialization_program const& program)
{
  if (!program.input_types || schema.columns.size() != program.input_types->size())
    throw std::invalid_argument("arrow_schema_drift: invalid bound Arrow width");
  for (size_t i = 0; i < schema.columns.size(); ++i) {
    auto const& field = schema.columns[i];
    auto type         = type_for(field.format);
    if (type != field.type || type != program.input_types->at(i))
      throw std::invalid_argument("arrow_schema_drift: invalid bound Arrow type");
    if (type.id() == cudf::type_id::DECIMAL128) {
      int32_t precision = 0;
      std::from_chars(
        field.format.data() + 2, field.format.data() + field.format.size(), precision);
      if (program.decimal_precisions.size() != schema.columns.size() ||
          precision != program.decimal_precisions[i])
        throw std::invalid_argument("arrow_schema_drift: invalid bound Arrow precision");
    }
  }
}

arrow_input_estimate validate_arrow_source(ArrowSchema const& schema,
                                           ArrowArray const& array,
                                           arrow_source_profile const& profile,
                                           arrow_bound_schema const& bound,
                                           arrow_metadata_lengths const& lengths)
{
  auto types = validate_flat(schema, array, lengths, false);
  if (!profile.leases || !profile.control_bytes || profile.completed == bool(profile.ready_event) ||
      (profile.memory == arrow_source_memory::device ? profile.device < 0 : profile.device != -1) ||
      (profile.memory != arrow_source_memory::device &&
       profile.memory != arrow_source_memory::pinned_host &&
       profile.memory != arrow_source_memory::pageable_host) ||
      profile.buffers.size() != types.size() || profile.strings.size() != types.size())
    throw std::invalid_argument("arrow_layout_unsupported: incomplete source profile");
  if (bound.columns.size() != types.size())
    throw transparent::arrow_schema_drift_error("Arrow width changed");
  arrow_input_estimate estimate;
  for (size_t i = 0; i < profile.allocations.size(); ++i) {
    auto const& allocation = profile.allocations[i];
    auto base              = reinterpret_cast<uintptr_t>(allocation.base);
    if (!allocation.id || !allocation.base || !allocation.bytes ||
        allocation.bytes > UINTPTR_MAX - base)
      invalid();
    for (size_t j = 0; j < i; ++j) {
      auto const& other = profile.allocations[j];
      auto start        = reinterpret_cast<uintptr_t>(other.base);
      if (other.id == allocation.id ||
          (base < start + other.bytes && start < base + allocation.bytes))
        throw std::invalid_argument("arrow_layout_unsupported: ambiguous allocation identity");
    }
    estimate.retained = checked_add(estimate.retained, allocation.bytes);
  }
  auto check_buffer =
    [&](void const* pointer, std::optional<arrow_buffer_use> const& use, uint64_t required) {
      if (!pointer) {
        if (use || required) invalid();
        return;
      }
      if (!use) invalid();
      auto found = std::find_if(profile.allocations.begin(),
                                profile.allocations.end(),
                                [&](auto const& a) { return a.id == use->allocation; });
      if (found == profile.allocations.end() || use->offset > found->bytes ||
          use->bytes > found->bytes - use->offset || required > use->bytes ||
          reinterpret_cast<uintptr_t>(pointer) !=
            reinterpret_cast<uintptr_t>(found->base) + use->offset)
        throw std::invalid_argument("arrow_layout_unsupported: buffer exceeds retained allocation");
    };
  auto rows = static_cast<uint64_t>(array.length);
  check_buffer(
    array.buffers[0], profile.root_validity, array.buffers[0] ? (array.offset + rows + 7) / 8 : 0);
  estimate.control = checked_add(profile.control_bytes,
                                 sizeof(owned_arrow_batch) + sizeof(arrow_source_profile) +
                                   sizeof(ArrowArray) + sizeof(ArrowSchema) + 256);
  estimate.control = checked_add(
    estimate.control, checked_mul(profile.allocations.capacity(), sizeof(arrow_allocation)));
  // Queue bookkeeping retains another allocation description. Count capacities
  // of the supplied vectors as well as their populated entries.
  estimate.control = checked_add(
    estimate.control, checked_mul(profile.allocations.size(), 2 * sizeof(arrow_allocation)));
  estimate.control = checked_add(
    estimate.control,
    checked_mul(profile.buffers.capacity(), sizeof(decltype(profile.buffers)::value_type)));
  estimate.control = checked_add(
    estimate.control,
    checked_mul(profile.strings.capacity(), sizeof(decltype(profile.strings)::value_type)));
  estimate.control = checked_add(
    estimate.control,
    checked_add(checked_mul(lengths.bucket_count(), sizeof(void*)),
                checked_mul(lengths.size(), sizeof(arrow_metadata_lengths::value_type) + 32)));
  for (size_t i = 0; i < types.size(); ++i) {
    auto const& field    = *schema.children[i];
    auto const& child    = *array.children[i];
    auto const& expected = bound.columns[i];
    if (canonical_format(field.format) != canonical_format(expected.format) ||
        types[i] != expected.type || expected.name != (field.name ? field.name : "") ||
        expected.nullable != bool(field.flags & ARROW_FLAG_NULLABLE) ||
        (field.flags & ~ARROW_FLAG_NULLABLE))
      throw transparent::arrow_schema_drift_error("Arrow storage, name or nullability changed");
    auto const& buffers = profile.buffers[i];
    if (buffers.size() != static_cast<size_t>(child.n_buffers)) invalid();
    auto end = static_cast<uint64_t>(child.offset + array.offset) + rows;
    check_buffer(child.buffers[0], buffers[0], child.buffers[0] ? (end + 7) / 8 : 0);
    uint64_t payload = 0;
    if (types[i].id() == cudf::type_id::STRING) {
      auto width = detail::arrow_string_offsets(field.format, child.buffers[1]).width();
      check_buffer(child.buffers[1], buffers[1], rows ? checked_mul(end + 1, width) : 0);
      if (rows && !profile.strings[i]) invalid();
      auto window = profile.strings[i].value_or(arrow_string_window{});
      payload     = detail::arrow_string_window_size(window.first, window.last);
      if (!rows && payload) invalid();
      check_buffer(child.buffers[2], buffers[2], rows && payload ? window.last : 0);
      payload = checked_add(payload, checked_mul(rows + 1, sizeof(int32_t)));
    } else {
      if (profile.strings[i]) invalid();
      auto boolean = types[i].id() == cudf::type_id::BOOL8;
      check_buffer(
        child.buffers[1],
        buffers[1],
        rows ? (boolean ? (end + 7) / 8 : checked_mul(end, cudf::size_of(types[i]))) : 0);
      payload = checked_mul(rows, cudf::size_of(types[i]));
    }
    estimate.destination =
      checked_add(estimate.destination, checked_add(payload, mask_bytes(rows)));
    estimate.control =
      checked_add(estimate.control,
                  sizeof(ArrowSchema) + sizeof(ArrowArray) + 128 +
                    checked_mul(buffers.capacity(), sizeof(arrow_buffer_use) + 32) +
                    std::strlen(field.format) + (field.name ? std::strlen(field.name) : 0));
  }
  for (auto const& [pointer, bytes] : lengths)
    estimate.control = checked_add(estimate.control, bytes);
  if (types.empty() && rows) estimate.destination = rows;
  if (profile.memory != arrow_source_memory::device) {
    estimate.staging = arrow_import_staging_bytes(array, types);
    if (profile.memory == arrow_source_memory::pageable_host)
      estimate.staging = checked_add(estimate.staging, estimate.destination);
  } else {
    // Per-column validation status and normalized masks, independent of payload size.
    estimate.destination = checked_add(estimate.destination, 64);
  }
  (void)estimate.total();
  return estimate;
}

std::size_t arrow_import_staging_bytes(ArrowArray const& array,
                                       std::span<cudf::data_type const> types)
{
  std::size_t total = 0;
  auto add          = [&](uint64_t bytes) {
    constexpr auto alignment = alignof(std::max_align_t);
    auto max                 = std::numeric_limits<std::size_t>::max();
    if (bytes > max - (alignment - 1)) throw std::overflow_error("Arrow staging size overflow");
    auto rounded = (bytes + alignment - 1) / alignment * alignment;
    if (rounded > max - total) throw std::overflow_error("Arrow staging size overflow");
    total += rounded;
  };
  add(types.size() * sizeof(normalized_column));
  auto rows = static_cast<uint64_t>(array.length);
  for (auto type : types) {
    if (type.id() == cudf::type_id::STRING) add((rows + 1) * sizeof(int32_t));
    if (type.id() == cudf::type_id::BOOL8) add(rows);
    add(mask_bytes(rows));
  }
  return total;
}

std::vector<cudf::data_type> owned_arrow_batch::input_types() const
{
  return validate_flat(schema, array, metadata_lengths, !source);
}
arrow_input_estimate owned_arrow_batch::estimate() const
{
  if (!source) throw std::logic_error("Arrow estimate requires an allocation profile");
  // This descriptor is used for sizing only. validate_for separately compares
  // every batch, including the first, against the planning-owned binding.
  arrow_bound_schema actual;
  auto types = input_types();
  for (size_t i = 0; i < types.size(); ++i) {
    auto const& field = *schema.children[i];
    actual.columns.push_back({field.name ? field.name : "",
                              field.format,
                              types[i],
                              bool(field.flags & ARROW_FLAG_NULLABLE)});
  }
  return validate_arrow_source(schema, array, *source, actual, metadata_lengths);
}
std::size_t owned_arrow_batch::estimated_device_bytes() const
{
  if (source) return estimate().destination;
  auto types        = input_types();
  std::size_t total = 0;
  auto add          = [&](uint64_t count, uint64_t width) {
    if (count > (std::numeric_limits<std::size_t>::max() - total) / width)
      throw std::overflow_error("Arrow device size overflow");
    total += count * width;
  };
  for (std::size_t i = 0; i < types.size(); ++i) {
    auto const& child = *array.children[i];
    auto rows         = static_cast<uint64_t>(array.length);
    // Account for allocated validity alignment even for unknown null counts.
    add(rows / 512 + 1, 64);
    if (types[i].id() == cudf::type_id::STRING) {
      add(rows + 1, sizeof(int32_t));
      if (rows) {
        auto begin = child.offset + array.offset;
        detail::arrow_string_offsets offsets(schema.children[i]->format, child.buffers[1]);
        auto bytes =
          detail::arrow_string_window_size(offsets[begin], offsets[begin + array.length]);
        add(static_cast<uint64_t>(bytes), 1);
      }
    } else {
      add(rows, cudf::size_of(types[i]));
    }
  }
  // A positive-cardinality zero-column batch needs an internal row-count carrier.
  if (types.empty()) add(static_cast<uint64_t>(array.length), sizeof(int8_t));
  return total;
}

void owned_arrow_batch::validate_for(materialization_program const& program) const
{
  if (!capacity || !schema.release || !array.release || !program.contract || !program.generation)
    throw std::invalid_argument("Arrow import requires admitted input and program");
  if (program.arrow_schema) {
    if (!source) throw std::invalid_argument("arrow_layout_unsupported: source profile missing");
    validate_arrow_binding(*program.arrow_schema, program);
    (void)validate_arrow_source(schema, array, *source, *program.arrow_schema, metadata_lengths);
  } else if (source) {
    throw transparent::arrow_schema_drift_error("qualified input has no bound storage schema");
  }
  auto types = input_types();
  if (!program.input_types || types != *program.input_types)
    throw transparent::arrow_schema_drift_error("Arrow schema does not match its bound program");
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (types[i].id() != cudf::type_id::DECIMAL128) continue;
    int precision = 0;
    std::string_view format(schema.children[i]->format);
    (void)std::from_chars(format.data() + 2, format.data() + format.size(), precision);
    if (program.decimal_precisions.size() != types.size() ||
        program.decimal_precisions[i] != precision)
      throw transparent::arrow_schema_drift_error(
        "Arrow decimal precision does not match its bound program");
  }
}

materialized_batch import_arrow_host(std::shared_ptr<owned_arrow_batch> input,
                                     materialization_program const& program,
                                     exec::retire_lane& lane,
                                     rmm::device_async_resource_ref mr)
{
  if (!input || !input->capacity || !program.contract || !program.generation)
    throw std::invalid_argument("Arrow import requires admitted input and program");
  input->validate_for(program);
  if (input->array.length > std::numeric_limits<cudf::size_type>::max())
    throw transparent::classified_execution_error(
      transparent::late_failure_cause::resource,
      "arrow_batch_too_large: Arrow rows exceed the destination row limit");
  auto const& types = *program.input_types;
  auto normalized   = std::make_shared<normalized_batch>(input, *input->staging_resource);
  auto submission   = lane.begin();
  // Register roots, capacity and PMR staging before any partially failing copy.
  submission.on_retire([owner = normalized](cudaError_t) noexcept {});
  ::cuda::stream_ref stream{submission.stream()};
  auto const* source = input->source.get();
  if (source) {
    int device = -1;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    if (source->memory == arrow_source_memory::device && source->device != device)
      throw std::invalid_argument(
        "arrow_layout_unsupported: source device differs from task device");
    for (auto const& allocation : source->allocations) {
      cudaPointerAttributes attributes{};
      auto pointer_status = cudaPointerGetAttributes(&attributes, allocation.base);
      if (pointer_status == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        if (source->memory == arrow_source_memory::pageable_host) continue;
        throw std::invalid_argument("arrow_layout_unsupported: declared memory placement mismatch");
      }
      CUDF_CUDA_TRY(pointer_status);
      bool valid = source->memory == arrow_source_memory::device
                     ? attributes.type == cudaMemoryTypeDevice && attributes.device == device
                   : source->memory == arrow_source_memory::pinned_host
                     ? attributes.type == cudaMemoryTypeHost
                     : attributes.type == cudaMemoryTypeUnregistered;
      if (!valid)
        throw std::invalid_argument("arrow_layout_unsupported: declared memory placement mismatch");
    }
    if (source->ready_event) {
      CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), source->ready_event, 0));
      // CPU normalization must not inspect host data still written by the source.
      if (source->memory != arrow_source_memory::device)
        CUDF_CUDA_TRY(cudaEventSynchronize(source->ready_event));
    }
  }
  std::unique_ptr<cudf::table> table;
  if (source && source->memory == arrow_source_memory::device) {
    table =
      detail::import_arrow_device_columns(input->array, *program.arrow_schema, *source, stream, mr);
  } else {
    (void)validate_arrow_host_flat(input->schema, input->array, input->metadata_lengths);
    if (source) {
      for (size_t i = 0; i < types.size(); ++i) {
        if (types[i].id() != cudf::type_id::STRING || !input->array.length) continue;
        auto const& child = *input->array.children[i];
        detail::arrow_string_offsets offsets(input->schema.children[i]->format, child.buffers[1]);
        auto begin  = child.offset + input->array.offset;
        auto window = *source->strings[i];
        if (offsets[begin] != window.first || offsets[begin + input->array.length] != window.last)
          throw std::invalid_argument("arrow_layout_unsupported: declared string window mismatch");
      }
    }
    normalize_columns(*normalized,
                      input->schema,
                      input->array,
                      types,
                      *input->staging_resource,
                      source && source->memory == arrow_source_memory::pageable_host);
    if (source) {
      // All normalized data (and pageable payload) must use the charged pinned arena.
      for (auto const& column : normalized->columns) {
        auto pinned = [](void const* pointer, size_t bytes) {
          if (!bytes) return;
          cudaPointerAttributes attributes{};
          CUDF_CUDA_TRY(cudaPointerGetAttributes(&attributes, pointer));
          if (attributes.type != cudaMemoryTypeHost)
            throw std::invalid_argument(
              "arrow_envelope_unavailable: normalization arena is not pinned");
        };
        pinned(column.values.data(), column.values.size());
        pinned(column.offsets.data(), column.offsets.size() * sizeof(int32_t));
        pinned(column.validity.data(), column.validity.size());
      }
    }
    table =
      copy_columns(*normalized, static_cast<cudf::size_type>(input->array.length), stream, mr);
  }
  materialized_batch batch;
  batch.table                   = owning_table_view(std::move(table));
  batch.rows                    = input->array.length;
  batch.contract                = program.contract;
  batch.generation              = program.generation;
  batch.logical_schema_complete = true;
  batch.layout                  = program.input_layout;
  ensure_row_count_carrier(batch, stream, mr);
  validate_materialized_batch(batch, program);
  auto error = submission.commit();
  // On commit failure, unwinding can enqueue output frees after its recovery
  // boundary. The caller must quiesce after unwinding, retaining the output MR.
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  return batch;
}

arrow_stream_result materialize_arrow_stream(arrow_batch_source& source,
                                             materialization_program const& program,
                                             exec::retire_lane& lane,
                                             rmm::device_async_resource_ref mr,
                                             arrow_batch_sink sink,
                                             arrow_stream_limits limits)
{
  arrow_stream_result result;
  try {
    if (!sink || limits.max_batch_rows < 0)
      throw std::invalid_argument("invalid Arrow stream limits or sink");
    while (true) {
      auto item = source.next();
      if (!item) return result;
      if (!item->input || !item->input->has_capacity())
        throw std::invalid_argument("Arrow stream batch has no ownership capacity");
      if (item->sequence != result.batches)
        throw std::invalid_argument("Arrow stream batch sequence is not contiguous");
      auto const actual_rows = item->input->row_count();
      if (item->rows < 0 || actual_rows < 0 || actual_rows > limits.max_batch_rows ||
          actual_rows != item->rows || item->bytes > limits.max_batch_bytes)
        throw std::invalid_argument("Arrow stream batch exceeds its declared limits");

      auto batch = import_arrow_host(item->input, program, lane, mr);
      if (batch.rows != item->rows)
        throw std::invalid_argument("Arrow stream row envelope does not match the batch");
      if (result.rows > std::numeric_limits<std::int64_t>::max() - batch.rows)
        throw std::overflow_error("Arrow stream row count overflow");
      sink(std::move(batch));
      ++result.batches;
      result.rows += item->rows;
    }
  } catch (...) {
    source.cancel();
    throw;
  }
}
}  // namespace sirius::op::scan
