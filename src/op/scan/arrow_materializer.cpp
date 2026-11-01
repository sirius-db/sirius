/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow_materializer.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream_ref>

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
                                     arrow_metadata_lengths metadata_lengths)
  : capacity(std::move(permit)),
    staging_resource(staging_resource),
    metadata_lengths(std::move(metadata_lengths))
{
  if (!s.release || !a.release || !capacity)
    throw std::invalid_argument("Arrow input requires ownership and capacity");
  schema    = s;
  array     = a;
  s.release = nullptr;
  a.release = nullptr;
}
owned_arrow_batch::~owned_arrow_batch()
{
  if (array.release) array.release(&array);
  if (schema.release) schema.release(&schema);
}

namespace {
[[noreturn]] void invalid()
{
  throw std::invalid_argument("unsupported or invalid arrow_host_flat_v1 input");
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
  if (f == "u") return cudf::data_type{T::STRING};
  if (f == "tdD") return cudf::data_type{T::TIMESTAMP_DAYS};
  if (f == "tsu:") return cudf::data_type{T::TIMESTAMP_MICROSECONDS};
  if (f.starts_with("d:")) {
    int precision = 0, scale = 0;
    auto p = std::from_chars(f.data() + 2, f.data() + f.size(), precision);
    if (p.ec != std::errc{} || p.ptr == f.data() + f.size() || *p.ptr != ',') invalid();
    auto s = std::from_chars(p.ptr + 1, f.data() + f.size(), scale);
    if (s.ec != std::errc{} ||
        (s.ptr != f.data() + f.size() &&
         std::string_view(s.ptr, f.data() + f.size() - s.ptr) != ",128") ||
        precision < 1 || precision > 38 || scale < 0 || scale > precision)
      invalid();
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
int32_t string_offset(void const* buffer, int64_t index)
{
  int32_t value;
  std::memcpy(&value, static_cast<uint8_t const*>(buffer) + index * sizeof(value), sizeof(value));
  return value;
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
                       ArrowArray const& array,
                       std::vector<cudf::data_type> const& types,
                       std::pmr::memory_resource& resource)
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
        auto first = string_offset(child.buffers[1], begin);
        for (int64_t row = 0; row <= array.length; ++row)
          column.offsets[row] = string_offset(child.buffers[1], begin + row) - first;
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
      a.length > std::numeric_limits<cudf::size_type>::max() || a.offset < 0 ||
      a.offset > std::numeric_limits<int64_t>::max() - a.length || a.n_buffers < 1 ||
      a.null_count < -1 || a.null_count > a.length || !a.buffers)
    invalid();
  validate_metadata(s.metadata, metadata_lengths);
  if (a.null_count > 0 && !a.buffers[0]) invalid();
}
}  // namespace

std::vector<cudf::data_type> validate_arrow_host_flat(
  ArrowSchema const& schema,
  ArrowArray const& array,
  arrow_metadata_lengths const& metadata_lengths)
{
  if constexpr (std::endian::native != std::endian::little) invalid();
  common(schema, array, metadata_lengths);
  if (std::string_view(schema.format) != "+s" || schema.n_children < 0 ||
      schema.n_children != array.n_children || array.n_buffers != 1 ||
      array.offset > std::numeric_limits<cudf::size_type>::max() - array.length ||
      (array.n_children && (!array.children || !schema.children)))
    invalid();
  if (array.null_count > 0) invalid();
  if (array.null_count != 0 && array.buffers[0]) {
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
    if (a.n_children || s.n_children || a.length < array.offset + array.length ||
        a.offset > std::numeric_limits<cudf::size_type>::max() - array.offset - array.length)
      invalid();
    auto type    = type_for(s.format);
    bool strings = type.id() == cudf::type_id::STRING;
    if (a.n_buffers != (strings ? 3 : 2)) invalid();
    if (array.length && !a.buffers[1]) invalid();
    auto const begin     = a.offset + array.offset;
    auto const end       = begin + array.length;
    auto const* validity = validity_for(a);
    auto const nulls     = count_nulls(validity, begin, end);
    if ((a.null_count >= 0 &&
         (nulls > a.null_count || a.null_count > nulls + (a.length - array.length))) ||
        (!(s.flags & ARROW_FLAG_NULLABLE) && nulls))
      invalid();
    if (strings && array.length) {
      if (array.length == std::numeric_limits<cudf::size_type>::max() ||
          string_offset(a.buffers[1], begin) < 0)
        invalid();
      for (auto row = begin; row < end; ++row)
        if (string_offset(a.buffers[1], row + 1) < string_offset(a.buffers[1], row)) invalid();
      if (string_offset(a.buffers[1], end) != string_offset(a.buffers[1], begin) && !a.buffers[2])
        invalid();
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
    types.push_back(type);
  }
  return types;
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
    add(cudf::bitmask_allocation_size_bytes(array.length));
  }
  return total;
}

std::vector<cudf::data_type> owned_arrow_batch::input_types() const
{
  return validate_arrow_host_flat(schema, array, metadata_lengths);
}
std::size_t owned_arrow_batch::estimated_device_bytes() const
{
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
        auto bytes = string_offset(child.buffers[1], begin + array.length) -
                     string_offset(child.buffers[1], begin);
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
  auto types = validate_arrow_host_flat(schema, array, metadata_lengths);
  if (!program.input_types || types != *program.input_types)
    throw std::invalid_argument("Arrow schema does not match its bound program");
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (types[i].id() != cudf::type_id::DECIMAL128) continue;
    int precision = 0;
    std::string_view format(schema.children[i]->format);
    (void)std::from_chars(format.data() + 2, format.data() + format.size(), precision);
    if (program.decimal_precisions.size() != types.size() ||
        program.decimal_precisions[i] != precision)
      throw std::invalid_argument("Arrow decimal precision does not match its bound program");
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
  auto const& types = *program.input_types;
  auto normalized   = std::make_shared<normalized_batch>(input, input->staging_resource);
  normalize_columns(*normalized, input->array, types, input->staging_resource);
  auto submission = lane.begin();
  // Register roots, capacity and PMR staging before any partially failing copy.
  submission.on_retire([owner = normalized](cudaError_t) noexcept {});
  ::cuda::stream_ref stream{submission.stream()};
  auto table =
    copy_columns(*normalized, static_cast<cudf::size_type>(input->array.length), stream, mr);
  materialized_batch batch;
  batch.table                   = owning_table_view(std::move(table));
  batch.rows                    = input->array.length;
  batch.contract                = program.contract;
  batch.generation              = program.generation;
  batch.logical_schema_complete = true;
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
