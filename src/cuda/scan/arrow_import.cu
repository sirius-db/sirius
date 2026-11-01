/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow/arrow_device_import.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>

namespace sirius::op::scan::detail {
namespace {
struct validation_status {
  unsigned invalid, nulls;
};
__device__ bool bit(void const* data, int64_t row)
{
  return !data || ((static_cast<uint8_t const*>(data)[row / 8] >> (row % 8)) & 1);
}
template <class T>
__device__ T load(void const* source, int64_t index)
{
  T result;
  auto* out      = reinterpret_cast<uint8_t*>(&result);
  auto const* in = static_cast<uint8_t const*>(source) + index * sizeof(T);
  for (unsigned i = 0; i < sizeof(T); ++i)
    out[i] = in[i];
  return result;
}
__device__ int64_t offset(void const* source, int64_t index, bool large)
{
  return large ? load<int64_t>(source, index) : load<int32_t>(source, index);
}
__global__ void validate_payload(void const* root_mask,
                                 int64_t root_begin,
                                 void const* validity,
                                 void const* values,
                                 int64_t begin,
                                 int64_t rows,
                                 bool nullable,
                                 bool strings,
                                 bool large,
                                 int64_t first,
                                 int64_t last,
                                 bool decimal,
                                 uint64_t limit_low,
                                 uint64_t limit_high,
                                 validation_status* status)
{
  for (int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += int64_t(blockDim.x) * gridDim.x) {
    if (!bit(root_mask, root_begin + row)) atomicExch(&status->invalid, 1U);
    bool valid = bit(validity, begin + row);
    if (!valid) {
      atomicAdd(&status->nulls, 1U);
      if (!nullable) atomicExch(&status->invalid, 1U);
    }
    if (strings) {
      auto a = offset(values, begin + row, large);
      auto b = offset(values, begin + row + 1, large);
      if (a < first || b < a || b > last || (row == 0 && a != first) ||
          (row == rows - 1 && b != last))
        atomicExch(&status->invalid, 1U);
    } else if (decimal && valid) {
      auto low  = load<uint64_t>(values, (begin + row) * 2);
      auto high = load<uint64_t>(values, (begin + row) * 2 + 1);
      if (high >> 63) {
        low  = ~low + 1;
        high = ~high + (low == 0);
      }
      if (high > limit_high || (high == limit_high && low >= limit_low))
        atomicExch(&status->invalid, 1U);
    }
  }
}
__global__ void normalize_bits(
  uint8_t* target, void const* source, int64_t begin, int64_t rows, bool boolean)
{
  auto count = boolean ? rows : (rows + 7) / 8;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    uint8_t value = 0;
    if (boolean)
      value = bit(source, begin + i);
    else
      for (unsigned j = 0; j < 8 && i * 8 + j < rows; ++j)
        value |= uint8_t(bit(source, begin + i * 8 + j)) << j;
    target[i] = value;
  }
}
__global__ void normalize_offsets(
  int32_t* target, void const* source, int64_t begin, int64_t rows, int64_t first, bool large)
{
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i <= rows;
       i += int64_t(blockDim.x) * gridDim.x)
    target[i] = rows ? static_cast<int32_t>(offset(source, begin + i, large) - first) : 0;
}
int blocks(int64_t rows)
{
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(4096, (rows + 255) / 256)));
}
}  // namespace

std::unique_ptr<cudf::table> import_arrow_device_columns(ArrowArray const& root,
                                                         arrow_bound_schema const& schema,
                                                         arrow_source_profile const& profile,
                                                         ::cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto rows = static_cast<cudf::size_type>(root.length);
  rmm::device_buffer status_buffer(sizeof(validation_status), stream, mr);
  auto* device_status = static_cast<validation_status*>(status_buffer.data());
  std::vector<cudf::size_type> null_counts(schema.columns.size());
  // Complete validation before allocating or copying any destination payload.
  for (size_t i = 0; i < schema.columns.size(); ++i) {
    auto const& child = *root.children[i];
    auto const& field = schema.columns[i];
    bool strings      = field.type.id() == cudf::type_id::STRING;
    bool decimal      = field.type.id() == cudf::type_id::DECIMAL128;
    auto window       = profile.strings[i].value_or(arrow_string_window{});
    __uint128_t limit = 1;
    if (decimal) {
      int precision = 0;
      std::from_chars(
        field.format.data() + 2, field.format.data() + field.format.size(), precision);
      for (int j = 0; j < precision; ++j)
        limit *= 10;
    }
    CUDF_CUDA_TRY(cudaMemsetAsync(device_status, 0, sizeof(validation_status), stream.get()));
    validate_payload<<<blocks(rows), 256, 0, stream.get()>>>(
      root.null_count ? root.buffers[0] : nullptr,
      root.offset,
      child.null_count ? child.buffers[0] : nullptr,
      child.buffers[1],
      root.offset + child.offset,
      rows,
      field.nullable,
      strings,
      field.format == "U",
      window.first,
      window.last,
      decimal,
      static_cast<uint64_t>(limit),
      static_cast<uint64_t>(limit >> 64),
      device_status);
    CUDF_CUDA_TRY(cudaGetLastError());
    validation_status observed{};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &observed, device_status, sizeof(observed), cudaMemcpyDeviceToHost, stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    if (observed.invalid ||
        (child.null_count >= 0 && (observed.nulls > child.null_count ||
                                   child.null_count > observed.nulls + child.length - rows)))
      throw std::invalid_argument(
        "arrow_layout_unsupported: invalid device payload or declared string window");
    null_counts[i] = observed.nulls;
  }
  if (schema.columns.empty() && root.null_count && root.buffers[0] && rows) {
    CUDF_CUDA_TRY(cudaMemsetAsync(device_status, 0, sizeof(validation_status), stream.get()));
    validate_payload<<<blocks(rows), 256, 0, stream.get()>>>(root.buffers[0],
                                                             root.offset,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             rows,
                                                             true,
                                                             false,
                                                             false,
                                                             0,
                                                             0,
                                                             false,
                                                             0,
                                                             0,
                                                             device_status);
    CUDF_CUDA_TRY(cudaGetLastError());
    validation_status observed{};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &observed, device_status, sizeof(observed), cudaMemcpyDeviceToHost, stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    if (observed.invalid)
      throw std::invalid_argument("arrow_layout_unsupported: nullable record batch");
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto copy = [&](void* dst, void const* src, size_t bytes) {
    if (bytes)
      CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream.get()));
  };
  for (size_t i = 0; i < schema.columns.size(); ++i) {
    auto const& child = *root.children[i];
    auto const& field = schema.columns[i];
    auto begin        = root.offset + child.offset;
    auto nulls        = null_counts[i];
    auto mask         = cudf::create_null_mask(
      rows, nulls ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED, stream, mr);
    if (nulls) {
      normalize_bits<<<blocks(rows), 256, 0, stream.get()>>>(
        static_cast<uint8_t*>(mask.data()), child.buffers[0], begin, rows, false);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    std::unique_ptr<cudf::column> column;
    if (field.type.id() == cudf::type_id::STRING) {
      auto window  = profile.strings[i].value_or(arrow_string_window{});
      auto offsets = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT32}, rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
      normalize_offsets<<<blocks(int64_t(rows) + 1), 256, 0, stream.get()>>>(
        offsets->mutable_view().data<int32_t>(),
        child.buffers[1],
        begin,
        rows,
        window.first,
        field.format == "U");
      CUDF_CUDA_TRY(cudaGetLastError());
      rmm::device_buffer chars(window.last - window.first, stream, mr);
      if (chars.size())
        copy(
          chars.data(), static_cast<uint8_t const*>(child.buffers[2]) + window.first, chars.size());
      column = cudf::make_strings_column(
        rows, std::move(offsets), std::move(chars), nulls, std::move(mask));
    } else {
      column =
        cudf::make_fixed_width_column(field.type, rows, cudf::mask_state::UNALLOCATED, stream, mr);
      if (field.type.id() == cudf::type_id::BOOL8) {
        normalize_bits<<<blocks(rows), 256, 0, stream.get()>>>(
          column->mutable_view().data<uint8_t>(), child.buffers[1], begin, rows, true);
        CUDF_CUDA_TRY(cudaGetLastError());
      } else if (rows) {
        auto width = cudf::size_of(field.type);
        copy(column->mutable_view().head<uint8_t>(),
             static_cast<uint8_t const*>(child.buffers[1]) + begin * width,
             size_t(rows) * width);
      }
      column->set_null_mask(std::move(mask), nulls);
    }
    columns.push_back(std::move(column));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}
}  // namespace sirius::op::scan::detail
