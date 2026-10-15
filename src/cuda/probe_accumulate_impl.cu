/*
 * Copyright 2026, Sirius Contributors.
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

// clang-format off
#include <helper/utils.hpp>
#include <op/aggregate/probe_accumulate_impl.hpp>
#include <sirius/exception.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda/atomic>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
// clang-format on

namespace sirius::op {

namespace {

constexpr int k_block_size = 256;

/// Device element type a column of a given cudf type is accumulated as.
enum class rep : std::uint8_t { none, i32, i64, u32, u64 };

[[nodiscard]] rep rep_of(cudf::data_type type) noexcept
{
  switch (type.id()) {
    case cudf::type_id::INT32:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS: return rep::i32;
    case cudf::type_id::INT64:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS: return rep::i64;
    case cudf::type_id::UINT32: return rep::u32;
    case cudf::type_id::UINT64: return rep::u64;
    default: return rep::none;
  }
}

[[nodiscard]] std::uint32_t rep_bytes(rep r) noexcept
{
  switch (r) {
    case rep::i32:
    case rep::u32: return 4;
    case rep::i64:
    case rep::u64: return 8;
    case rep::none: break;
  }
  return 0;
}

[[nodiscard]] __host__ __device__ constexpr bool is_extremum(accumulate_kind kind) noexcept
{
  return kind == accumulate_kind::MIN || kind == accumulate_kind::MAX;
}

[[nodiscard]] unsigned grid_for(std::int64_t n)
{
  return static_cast<unsigned>(
    std::max<std::int64_t>(sirius::utils::ceil_div<std::int64_t>(n, k_block_size), 1));
}

/// Byte offsets of one aggregate's fields inside a record.
struct field_offsets {
  std::uint32_t value = 0;  ///< MIN/MAX accumulator
  std::uint32_t count = 0;  ///< COUNT accumulator (uint64)
  std::uint32_t seen  = 0;  ///< MIN/MAX: whether any non-NULL input was folded in (one byte)
};

/// Per-aggregate launch descriptor. All of them travel in one by-value kernel argument.
struct kernel_spec {
  accumulate_kind kind;
  rep value_rep;
  field_offsets offsets;
  void const* values;
  cudf::bitmask_type const* validity;
  cudf::size_type validity_offset;
};

struct kernel_args {
  std::array<kernel_spec, probe_accumulate_max_specs> specs;
  int num_specs;
  std::uint32_t record_bytes;
};

template <class T>
__device__ __forceinline__ void atomic_fold(std::byte* slot, T value, bool is_min)
{
  cuda::atomic_ref<T, cuda::thread_scope_device> ref(*reinterpret_cast<T*>(slot));
  if (is_min) {
    ref.fetch_min(value, cuda::memory_order_relaxed);
  } else {
    ref.fetch_max(value, cuda::memory_order_relaxed);
  }
}

__device__ __forceinline__ void atomic_bump(std::byte* slot)
{
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>(
    *reinterpret_cast<std::uint64_t*>(slot))
    .fetch_add(1, cuda::memory_order_relaxed);
}

__device__ __forceinline__ bool row_is_valid(kernel_spec const& spec, cudf::size_type row)
{
  return spec.validity == nullptr || cudf::bit_is_set(spec.validity, spec.validity_offset + row);
}

template <class T>
__device__ __forceinline__ void fold_value(kernel_spec const& spec,
                                           std::byte* record,
                                           cudf::size_type row)
{
  auto const value = static_cast<T const*>(spec.values)[row];
  atomic_fold<T>(record + spec.offsets.value, value, spec.kind == accumulate_kind::MIN);
  // Every writer stores the same byte, so the unsynchronised store is benign.
  *reinterpret_cast<volatile std::uint8_t*>(record + spec.offsets.seen) = 1;
}

// One thread per match pair, grid-stride. The pairs are unordered, so concurrent threads regularly
// target the same build row; the folds are commutative and associative, which makes that safe.
__global__ void accumulate_kernel(kernel_args args,
                                  cudf::size_type const* probe_rows,
                                  cudf::size_type const* build_rows,
                                  std::int64_t num_pairs,
                                  std::byte* records)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (auto p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < num_pairs;
       p += stride) {
    auto const row = probe_rows[p];
    auto* record   = records + static_cast<std::size_t>(build_rows[p]) * args.record_bytes;
    atomic_bump(record);  // hits: always the first field
    for (int s = 0; s < args.num_specs; ++s) {
      auto const& spec = args.specs[s];
      if (spec.kind == accumulate_kind::COUNT_STAR || !row_is_valid(spec, row)) { continue; }
      if (spec.kind == accumulate_kind::COUNT) {
        atomic_bump(record + spec.offsets.count);
        continue;
      }
      switch (spec.value_rep) {
        case rep::i32: fold_value<std::int32_t>(spec, record, row); break;
        case rep::i64: fold_value<std::int64_t>(spec, record, row); break;
        case rep::u32: fold_value<std::uint32_t>(spec, record, row); break;
        case rep::u64: fold_value<std::uint64_t>(spec, record, row); break;
        case rep::none: break;
      }
    }
  }
}

template <class T>
__device__ __forceinline__ void write_identity(kernel_spec const& spec, std::byte* record)
{
  *reinterpret_cast<T*>(record + spec.offsets.value) = spec.kind == accumulate_kind::MIN
                                                         ? std::numeric_limits<T>::max()
                                                         : std::numeric_limits<T>::lowest();
}

__global__ void initialize_kernel(kernel_args args, std::int64_t num_records, std::byte* records)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_records;
       i += stride) {
    auto* record = records + static_cast<std::size_t>(i) * args.record_bytes;
    for (std::uint32_t b = 0; b < args.record_bytes; b += 8) {
      *reinterpret_cast<std::uint64_t*>(record + b) = 0;
    }
    for (int s = 0; s < args.num_specs; ++s) {
      auto const& spec = args.specs[s];
      if (!is_extremum(spec.kind)) { continue; }
      switch (spec.value_rep) {
        case rep::i32: write_identity<std::int32_t>(spec, record); break;
        case rep::i64: write_identity<std::int64_t>(spec, record); break;
        case rep::u32: write_identity<std::uint32_t>(spec, record); break;
        case rep::u64: write_identity<std::uint64_t>(spec, record); break;
        case rep::none: break;
      }
    }
  }
}

struct hit_predicate {
  std::byte const* records;
  std::uint32_t record_bytes;
  __device__ bool operator()(cudf::size_type i) const
  {
    return *reinterpret_cast<std::uint64_t const*>(records +
                                                   static_cast<std::size_t>(i) * record_bytes) > 0;
  }
};

// out[j] = <field at offset> of the j-th selected record; one thread per output row.
template <class T>
__global__ void gather_field_kernel(std::byte const* records,
                                    std::uint32_t record_bytes,
                                    std::uint32_t offset,
                                    cudf::size_type const* selected,
                                    std::int64_t num_out,
                                    T* out)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (auto j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; j < num_out;
       j += stride) {
    out[j] = *reinterpret_cast<T const*>(
      records + static_cast<std::size_t>(selected[j]) * record_bytes + offset);
  }
}

__global__ void gather_seen_kernel(std::byte const* records,
                                   std::uint32_t record_bytes,
                                   std::uint32_t offset,
                                   cudf::size_type const* selected,
                                   std::int64_t num_out,
                                   bool* out)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (auto j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; j < num_out;
       j += stride) {
    out[j] = *reinterpret_cast<std::uint8_t const*>(
               records + static_cast<std::size_t>(selected[j]) * record_bytes + offset) != 0;
  }
}

template <class T>
void launch_gather_field(std::byte const* records,
                         std::uint32_t record_bytes,
                         std::uint32_t offset,
                         cudf::size_type const* selected,
                         std::int64_t num_out,
                         void* out,
                         ::cuda::stream_ref stream)
{
  if (num_out == 0) { return; }
  gather_field_kernel<T><<<grid_for(num_out), k_block_size, 0, stream.get()>>>(
    records, record_bytes, offset, selected, num_out, static_cast<T*>(out));
  CUDF_CHECK_CUDA(stream.get());
}

}  // namespace

struct probe_accumulate_state::layout {
  std::uint32_t record_bytes = 0;
  std::vector<field_offsets> fields;  ///< One per spec
};

bool probe_accumulate_supports_type(cudf::data_type type) noexcept
{
  return rep_of(type) != rep::none;
}

probe_accumulate_state::probe_accumulate_state(std::vector<accumulate_spec> specs)
  : _specs(std::move(specs)), _layout(std::make_unique<layout>())
{
  if (_specs.size() > probe_accumulate_max_specs) {
    throw sirius::internal_exception("probe_accumulate: {} aggregates exceed the limit of {}",
                                     _specs.size(),
                                     probe_accumulate_max_specs);
  }
  for (auto const& spec : _specs) {
    if (is_extremum(spec.kind) && rep_of(spec.value_type) == rep::none) {
      throw sirius::internal_exception(
        "probe_accumulate: MIN/MAX over type id {} has no device atomic",
        static_cast<int>(spec.value_type.id()));
    }
  }

  // Lay the record out widest field first so every field is naturally aligned without padding:
  // hits (8), then the 8-byte accumulators, then the 4-byte ones, then the one-byte flags.
  _layout->fields.assign(_specs.size(), field_offsets{});
  std::uint32_t cursor = 8;  // hits occupy [0, 8)
  for (std::size_t i = 0; i < _specs.size(); ++i) {
    if (_specs[i].kind == accumulate_kind::COUNT) {
      _layout->fields[i].count = cursor;
      cursor += 8;
    }
  }
  for (auto const width : {8u, 4u}) {
    for (std::size_t i = 0; i < _specs.size(); ++i) {
      if (is_extremum(_specs[i].kind) && rep_bytes(rep_of(_specs[i].value_type)) == width) {
        _layout->fields[i].value = cursor;
        cursor += width;
      }
    }
  }
  for (std::size_t i = 0; i < _specs.size(); ++i) {
    if (is_extremum(_specs[i].kind)) { _layout->fields[i].seen = cursor++; }
  }
  // Pad to a power of two while it fits a 64-byte line, so a record never straddles a sector
  // boundary; otherwise to the 8-byte word that initialisation writes in.
  std::uint32_t padded = 8;
  while (padded < cursor && padded < 64) {
    padded *= 2;
  }
  _layout->record_bytes = cursor <= 64 ? padded : (cursor + 7) / 8 * 8;
}

probe_accumulate_state::~probe_accumulate_state() = default;

std::size_t probe_accumulate_state::record_bytes_for(cudf::size_type num_build_rows) const noexcept
{
  return static_cast<std::size_t>(num_build_rows) * _layout->record_bytes;
}

std::size_t probe_accumulate_state::emit_peak_bytes(cudf::size_type num_build_rows) const noexcept
{
  // selected indices (4) + key copy (up to 8) + per aggregate an INT64-wide column and a flag.
  auto const per_row = std::size_t{16} + _specs.size() * 10;
  return static_cast<std::size_t>(num_build_rows) * per_row;
}

void probe_accumulate_state::initialize(cudf::size_type num_build_rows,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  if (_initialized) {
    throw sirius::internal_exception("probe_accumulate_state::initialize called twice");
  }
  _num_build_rows = num_build_rows;
  _records        = rmm::device_buffer(record_bytes_for(num_build_rows), stream, mr);
  if (num_build_rows > 0) {
    kernel_args args{};
    args.num_specs    = static_cast<int>(_specs.size());
    args.record_bytes = _layout->record_bytes;
    for (std::size_t i = 0; i < _specs.size(); ++i) {
      args.specs[i].kind      = _specs[i].kind;
      args.specs[i].value_rep = rep_of(_specs[i].value_type);
      args.specs[i].offsets   = _layout->fields[i];
    }
    initialize_kernel<<<grid_for(num_build_rows), k_block_size, 0, stream.get()>>>(
      args, num_build_rows, static_cast<std::byte*>(_records.data()));
    CUDF_CHECK_CUDA(stream.get());
  }
  _initialized = true;
}

void probe_accumulate_state::accumulate(cudf::table_view const& probe,
                                        cudf::device_span<cudf::size_type const> probe_rows,
                                        cudf::device_span<cudf::size_type const> build_rows,
                                        ::cuda::stream_ref stream)
{
  if (!_initialized) {
    throw sirius::internal_exception("probe_accumulate_state::accumulate before initialize");
  }
  if (probe_rows.size() != build_rows.size()) {
    throw sirius::internal_exception(
      "probe_accumulate: {} probe rows are paired with {} build rows",
      probe_rows.size(),
      build_rows.size());
  }
  auto const num_pairs = static_cast<std::int64_t>(probe_rows.size());
  if (num_pairs == 0) { return; }

  kernel_args args{};
  args.num_specs    = static_cast<int>(_specs.size());
  args.record_bytes = _layout->record_bytes;
  for (std::size_t i = 0; i < _specs.size(); ++i) {
    auto const& spec = _specs[i];
    auto& out        = args.specs[i];
    out.kind         = spec.kind;
    out.value_rep    = rep_of(spec.value_type);
    out.offsets      = _layout->fields[i];
    if (spec.kind == accumulate_kind::COUNT_STAR) { continue; }
    if (spec.probe_column < 0 || spec.probe_column >= probe.num_columns()) {
      throw sirius::internal_exception(
        "probe_accumulate: aggregate {} reads column {} of a {}-column probe batch",
        i,
        spec.probe_column,
        probe.num_columns());
    }
    auto const column = probe.column(spec.probe_column);
    // The kernel reads the column as the planned type; a different carrier (e.g. a narrowed one)
    // would be misread, so refuse it rather than touch memory with the wrong stride.
    if (is_extremum(spec.kind) && column.type() != spec.value_type) {
      throw sirius::internal_exception(
        "probe_accumulate: aggregate {} expects type id {} but the probe column {} carries {}",
        i,
        static_cast<int>(spec.value_type.id()),
        spec.probe_column,
        static_cast<int>(column.type().id()));
    }
    out.validity        = column.null_mask();
    out.validity_offset = column.offset();
    if (is_extremum(spec.kind)) {
      out.values = reinterpret_cast<std::byte const*>(column.head()) +
                   static_cast<std::size_t>(column.offset()) * rep_bytes(out.value_rep);
    }
  }
  accumulate_kernel<<<grid_for(num_pairs), k_block_size, 0, stream.get()>>>(
    args,
    probe_rows.data(),
    build_rows.data(),
    num_pairs,
    static_cast<std::byte*>(_records.data()));
  CUDF_CHECK_CUDA(stream.get());
}

std::unique_ptr<cudf::table> probe_accumulate_state::emit(cudf::column_view const& build_key,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  if (!_initialized) {
    throw sirius::internal_exception("probe_accumulate_state::emit before initialize");
  }
  if (_records.data() == nullptr && _num_build_rows > 0) {
    throw sirius::internal_exception("probe_accumulate_state::emit consumes the state; call once");
  }
  if (build_key.size() != _num_build_rows) {
    throw sirius::internal_exception(
      "probe_accumulate: build key has {} rows but the state holds {}",
      build_key.size(),
      _num_build_rows);
  }

  auto const n        = _num_build_rows;
  auto const* records = static_cast<std::byte const*>(_records.data());
  auto const rb       = _layout->record_bytes;

  // Rows of the build table that matched at least once, in build order.
  auto const first = thrust::make_counting_iterator<cudf::size_type>(0);
  auto const last  = first + n;
  hit_predicate const hit{records, rb};
  auto const num_out = static_cast<cudf::size_type>(
    thrust::count_if(rmm::exec_policy_nosync(stream), first, last, hit));
  rmm::device_uvector<cudf::size_type> selected(num_out, stream, mr);
  thrust::copy_if(rmm::exec_policy_nosync(stream), first, last, selected.begin(), hit);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(_specs.size() + 1);
  {
    cudf::column_view const map(
      cudf::data_type{cudf::type_id::INT32}, num_out, selected.data(), nullptr, 0);
    auto key = cudf::gather(
      cudf::table_view({build_key}), map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    columns.push_back(std::move(key->release()[0]));
  }

  for (std::size_t i = 0; i < _specs.size(); ++i) {
    auto const& spec    = _specs[i];
    auto const& offsets = _layout->fields[i];
    switch (spec.kind) {
      case accumulate_kind::MIN:
      case accumulate_kind::MAX: {
        auto column = cudf::make_fixed_width_column(
          spec.value_type, num_out, cudf::mask_state::UNALLOCATED, stream, mr);
        auto* out = column->mutable_view().head();
        switch (rep_of(spec.value_type)) {
          case rep::i32:
            launch_gather_field<std::int32_t>(
              records, rb, offsets.value, selected.data(), num_out, out, stream);
            break;
          case rep::i64:
            launch_gather_field<std::int64_t>(
              records, rb, offsets.value, selected.data(), num_out, out, stream);
            break;
          case rep::u32:
            launch_gather_field<std::uint32_t>(
              records, rb, offsets.value, selected.data(), num_out, out, stream);
            break;
          case rep::u64:
            launch_gather_field<std::uint64_t>(
              records, rb, offsets.value, selected.data(), num_out, out, stream);
            break;
          case rep::none: break;
        }
        auto seen = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                              num_out,
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
        if (num_out > 0) {
          gather_seen_kernel<<<grid_for(num_out), k_block_size, 0, stream.get()>>>(
            records, rb, offsets.seen, selected.data(), num_out, seen->mutable_view().data<bool>());
          CUDF_CHECK_CUDA(stream.get());
        }
        auto [null_mask, nulls] = cudf::bools_to_mask(seen->view(), stream, mr);
        if (nulls > 0) { column->set_null_mask(std::move(*null_mask), nulls); }
        columns.push_back(std::move(column));
        break;
      }
      case accumulate_kind::COUNT:
      case accumulate_kind::COUNT_STAR: {
        auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                num_out,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);
        launch_gather_field<std::int64_t>(records,
                                          rb,
                                          spec.kind == accumulate_kind::COUNT ? offsets.count : 0,
                                          selected.data(),
                                          num_out,
                                          column->mutable_view().head(),
                                          stream);
        columns.push_back(std::move(column));
        break;
      }
    }
  }

  // The kernels above read the records asynchronously; keep them alive until they are done.
  stream.sync();
  _records = rmm::device_buffer{};
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op
