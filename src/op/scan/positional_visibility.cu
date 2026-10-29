/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/positional_visibility.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/error.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sirius::op::scan {
namespace {
struct interval {
  uint64_t first, end;
};
struct source_binding {
  uint64_t file_rows;
  std::size_t range_first, range_end, delete_first, delete_end;
};

__global__ void position_mask(uint64_t const* rows,
                              int32_t const* sources,
                              cudf::size_type count,
                              source_binding const* bindings,
                              std::size_t source_count,
                              interval const* ranges,
                              int64_t const* deletes,
                              bool* keep,
                              int* invalid)
{
  auto const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  auto const source = sources ? sources[index] : 0;
  if (source < 0 || static_cast<std::size_t>(source) >= source_count) {
    atomicExch(invalid, 1);
    keep[index] = false;
    return;
  }
  auto const binding = bindings[source];
  auto const row     = rows[index];
  auto lo            = binding.range_first;
  auto hi            = binding.range_end;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (ranges[mid].end <= row)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (row >= binding.file_rows || lo == binding.range_end || row < ranges[lo].first) {
    atomicExch(invalid, 1);
    keep[index] = false;
    return;
  }
  lo = binding.delete_first;
  hi = binding.delete_end;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (static_cast<uint64_t>(deletes[mid]) < row)
      lo = mid + 1;
    else
      hi = mid;
  }
  keep[index] = lo == binding.delete_end || static_cast<uint64_t>(deletes[lo]) != row;
}

struct position_work {
  std::unique_ptr<cudf::table> table;
  std::unique_ptr<gpu_row_positions> positions;
  std::vector<source_binding> bindings;
  std::vector<interval> ranges;
  std::vector<int64_t> deletes;
  std::unique_ptr<rmm::device_buffer> device_bindings, device_ranges, device_deletes;
  int initial_invalid = 0;
  std::unique_ptr<rmm::device_scalar<int>> invalid;
  std::unique_ptr<cudf::column> mask;
};
}  // namespace

std::unique_ptr<cudf::table> apply_positional_visibility(
  std::unique_ptr<cudf::table> table,
  std::unique_ptr<gpu_row_positions> positions,
  position_lookup const& lookup,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (!table || !positions || !positions->rows || positions->sources.empty())
    throw std::invalid_argument("visibility requires reader positions and source bindings");
  auto const rows   = table->num_rows();
  auto valid_column = [rows](auto const& column, cudf::type_id type) {
    return column && column->size() == rows && column->type().id() == type &&
           column->null_count() == 0;
  };
  if (!valid_column(positions->rows, cudf::type_id::UINT64) ||
      (positions->source_indices &&
       !valid_column(positions->source_indices, cudf::type_id::INT32)) ||
      (!positions->source_indices && positions->sources.size() != 1))
    throw std::invalid_argument("invalid reader position columns");

  auto work       = std::make_unique<position_work>();
  work->table     = std::move(table);
  work->positions = std::move(positions);
  for (auto const& source : work->positions->sources) {
    if (source.file.empty() || source.file_rows > static_cast<uint64_t>(INT64_MAX))
      throw std::invalid_argument("invalid reader position source");
    source_binding binding{source.file_rows, work->ranges.size(), 0, work->deletes.size(), 0};
    uint64_t end = 0;
    for (auto const& [first, next] : source.ranges) {
      if (first < end || first >= next || next > source.file_rows)
        throw std::invalid_argument("invalid reader input ranges");
      work->ranges.push_back({first, next});
      end = next;
    }
    binding.range_end  = work->ranges.size();
    auto const deletes = lookup(source.file);
    // Transfer only deletes in selected row groups, never a long file's full list.
    for (auto const& [first, end] : source.ranges) {
      auto lo = std::lower_bound(deletes.begin(), deletes.end(), first);
      auto hi = std::lower_bound(lo, deletes.end(), end);
      work->deletes.insert(work->deletes.end(), lo, hi);
    }
    binding.delete_end = work->deletes.size();
    work->bindings.push_back(binding);
  }
  if (!rows) return std::move(work->table);
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    auto upload = [&](auto const& values) {
      return std::make_unique<rmm::device_buffer>(
        values.data(), values.size() * sizeof(values[0]), stream, mr);
    };
    work->device_bindings = upload(work->bindings);
    work->device_ranges   = upload(work->ranges);
    work->device_deletes  = upload(work->deletes);
    work->invalid = std::make_unique<rmm::device_scalar<int>>(work->initial_invalid, stream, mr);
    work->mask    = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    position_mask<<<(static_cast<int64_t>(rows) + 255) / 256, 256, 0, stream.get()>>>(
      work->positions->rows->view().data<uint64_t>(),
      work->positions->source_indices ? work->positions->source_indices->view().data<int32_t>()
                                      : nullptr,
      rows,
      static_cast<source_binding const*>(work->device_bindings->data()),
      work->bindings.size(),
      static_cast<interval const*>(work->device_ranges->data()),
      static_cast<int64_t const*>(work->device_deletes->data()),
      work->mask->mutable_view().data<bool>(),
      work->invalid->data());
    RMM_CUDA_TRY(cudaPeekAtLastError());
    // Copy one validation word, never the per-row positions, to the host.
    if (work->invalid->value(stream))
      throw std::invalid_argument("reader position is outside its bound input source");
    // The validation word joins the mask kernel. Coordinates have no consumer
    // after this point and must not coexist with the gathered output payload.
    work->positions.reset();
    auto result = sirius::apply_retention_mask(work->table->view(), work->mask->view(), stream, mr);
    detail::check_read_checkpoint(detail::read_checkpoint::PAYLOAD_GATHER);
    return result;
  } catch (...) {
    detail::retire_reads(work, stream, unjoined);
    throw;
  }
}
}  // namespace sirius::op::scan
