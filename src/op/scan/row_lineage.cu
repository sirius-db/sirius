/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "memory/size_arithmetic.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/row_lineage.hpp"
#include "op/scan/scan_plan.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace sirius::op::scan {
namespace {
struct selection_run {
  int64_t output_first, input_first, count;
};
__global__ void check_ordinals(int64_t const* sorted, int64_t count, int64_t bound, int* invalid)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && (sorted[i] < 0 || sorted[i] >= bound || (i && sorted[i] == sorted[i - 1])))
    atomicExch(invalid, 1);
}
template <typename T>
__global__ void select_runs(int64_t* output,
                            T const* indices,
                            int64_t rows,
                            selection_run const* runs,
                            int64_t count,
                            int64_t shift = 0)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  int64_t index = indices ? indices[i] : i;
  int64_t lo = 0, hi = count;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (runs[mid].output_first + runs[mid].count <= index)
      lo = mid + 1;
    else
      hi = mid;
  }
  output[i] = runs[lo].input_first + index - runs[lo].output_first + shift;
}
__global__ void copy_shift(int64_t* output, int64_t const* input, int64_t rows, int64_t shift)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < rows) output[i] = input[i] + shift;
}
struct visibility_run {
  int64_t ordinal, first, count, delete_first, delete_count;
};
struct physical_run {
  int64_t ordinal, file, first, count;
};
__device__ physical_run resolve_run(physical_run const* runs, int64_t count, int64_t ordinal)
{
  int64_t lo = 0, hi = count;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (runs[mid].ordinal + runs[mid].count <= ordinal)
      lo = mid + 1;
    else
      hi = mid;
  }
  return runs[lo];
}
__global__ void physical_positions(int64_t const* ordinals,
                                   int64_t count,
                                   physical_run const* runs,
                                   int64_t run_count,
                                   int64_t* rows,
                                   int32_t* files)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  auto run = resolve_run(runs, run_count, ordinals[i]);
  if (rows) rows[i] = run.first + ordinals[i] - run.ordinal;
  if (files) files[i] = run.file;
}
__global__ void file_rows_to_ordinals(uint64_t* values,
                                      int32_t const* sources,
                                      int64_t count,
                                      physical_run const* runs,
                                      int64_t run_count,
                                      int64_t source_count,
                                      int* invalid)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  auto file = sources ? sources[i] : 0;
  if (file < 0 || file >= source_count || values[i] > INT64_MAX) {
    atomicExch(invalid, 1);
    return;
  }
  auto row   = static_cast<int64_t>(values[i]);
  int64_t lo = 0, hi = run_count;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (runs[mid].file < file ||
        (runs[mid].file == file && runs[mid].first + runs[mid].count <= row))
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == run_count || runs[lo].file != file || row < runs[lo].first) {
    atomicExch(invalid, 1);
    return;
  }
  values[i] = runs[lo].ordinal + row - runs[lo].first;
}
__global__ void check_composition(int64_t count,
                                  int64_t const* selected,
                                  selection_run const* selections,
                                  int64_t selection_count,
                                  physical_run const* before,
                                  int64_t before_count,
                                  physical_run const* next,
                                  int64_t next_count,
                                  int* invalid)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int64_t ordinal = i;
  if (selected)
    ordinal = selected[i];
  else if (selection_count) {
    int64_t lo = 0, hi = selection_count;
    while (lo < hi) {
      auto mid = lo + (hi - lo) / 2;
      if (selections[mid].output_first + selections[mid].count <= i)
        lo = mid + 1;
      else
        hi = mid;
    }
    ordinal = selections[lo].input_first + i - selections[lo].output_first;
  }
  auto a = resolve_run(before, before_count, ordinal);
  auto b = resolve_run(next, next_count, i);
  if (a.file != b.file || a.first + ordinal - a.ordinal != b.first + i - b.ordinal)
    atomicExch(invalid, 1);
}
__global__ void visibility_mask(uint8_t* mask,
                                int64_t const* ordinals,
                                int64_t rows,
                                visibility_run const* runs,
                                int64_t count,
                                int64_t const* deletes)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  auto ordinal = ordinals[i];
  int64_t lo = 0, hi = count;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (runs[mid].ordinal + runs[mid].count <= ordinal)
      lo = mid + 1;
    else
      hi = mid;
  }
  auto const run = runs[lo];
  auto row       = run.first + ordinal - run.ordinal;
  lo             = run.delete_first;
  hi             = lo + run.delete_count;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (deletes[mid] < row)
      lo = mid + 1;
    else
      hi = mid;
  }
  mask[i] = !(lo < run.delete_first + run.delete_count && deletes[lo] == row);
}

__global__ void fill_ordinals(int64_t* output,
                              int64_t rows,
                              selection_run const* runs,
                              int64_t count)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  int64_t lo = 0, hi = count;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (runs[mid].output_first + runs[mid].count <= i)
      lo = mid + 1;
    else
      hi = mid;
  }
  output[i] = runs[lo].input_first + i - runs[lo].output_first;
}
}  // namespace

void row_lineage::validate_on(int64_t rows,
                              ::cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr) const
{
  validate(rows);
  if (state != row_position_state::per_row || !rows) return;
  rmm::device_uvector<int64_t> sorted(rows, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(sorted.data(),
                                per_row->view().data<int64_t>(),
                                rows * sizeof(int64_t),
                                cudaMemcpyDeviceToDevice,
                                stream.get()));
  thrust::sort(rmm::exec_policy(stream, mr), sorted.begin(), sorted.end());
  int const zero = 0;
  rmm::device_scalar<int> invalid(zero, stream, mr);
  check_ordinals<<<(rows + 255) / 256, 256, 0, stream.get()>>>(
    sorted.data(), rows, input_domain_length(), invalid.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  if (invalid.value(stream)) throw std::invalid_argument("invalid or duplicate lineage ordinal");
}

std::size_t lineage_conversion_workspace_bytes(std::size_t rows, std::size_t intervals)
{
  auto const max = std::numeric_limits<std::size_t>::max();
  // Validation owns a sorted copy and sort scratch; conversion reuses UINT64 storage.
  if (rows > (max - sizeof(int)) / 32 ||
      intervals > (max - rows * 32 - sizeof(int)) / sizeof(physical_run))
    return max;
  return rows * 32 + intervals * sizeof(physical_run) + sizeof(int);
}

row_lineage consume_reader_positions(std::unique_ptr<gpu_row_positions> positions,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  if (!positions || !positions->rows || positions->rows->type().id() != cudf::type_id::UINT64 ||
      positions->rows->has_nulls() || positions->sources.empty())
    throw std::invalid_argument("reader positions require bound non-null UINT64 rows");
  auto const rows = positions->rows->size();
  if ((positions->sources.size() != 1 && !positions->source_indices) ||
      (positions->source_indices &&
       (positions->source_indices->size() != rows ||
        positions->source_indices->type().id() != cudf::type_id::INT32 ||
        positions->source_indices->has_nulls())))
    throw std::invalid_argument("reader positions have invalid source indices");
  struct conversion_state {
    std::unique_ptr<gpu_row_positions> input;
    std::vector<physical_run> host;
    std::unique_ptr<rmm::device_uvector<physical_run>> device;
    std::unique_ptr<rmm::device_scalar<int>> invalid;
    row_lineage result;
  };
  auto state     = std::make_unique<conversion_state>();
  state->input   = std::move(positions);
  auto domain    = std::make_shared<lineage_domain>();
  int64_t offset = 0;
  for (std::size_t i = 0; i < state->input->sources.size(); ++i) {
    auto const& source = state->input->sources[i];
    if (source.file_rows > INT64_MAX) throw std::invalid_argument("reader file length overflow");
    domain->files.push_back({source.file, static_cast<int64_t>(source.file_rows)});
    uint64_t prior = 0;
    for (auto [first, end] : source.ranges) {
      if (first < prior || end < first || end > source.file_rows ||
          end - first > INT64_MAX - offset)
        throw std::invalid_argument("reader position input intervals overlap or overflow");
      prior = end;
      if (first == end) continue;
      domain->file_runs.push_back(
        {i, static_cast<int64_t>(first), static_cast<int64_t>(end - first)});
      state->host.push_back({offset,
                             static_cast<int64_t>(i),
                             static_cast<int64_t>(first),
                             static_cast<int64_t>(end - first)});
      offset += end - first;
    }
  }
  row_lineage{row_position_state::identity, domain}.validate(offset);
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    state->device =
      std::make_unique<rmm::device_uvector<physical_run>>(state->host.size(), stream, mr);
    int const zero = 0;
    state->invalid = std::make_unique<rmm::device_scalar<int>>(zero, stream, mr);
    if (!state->host.empty())
      CUDF_CUDA_TRY(cudaMemcpyAsync(state->device->data(),
                                    state->host.data(),
                                    state->host.size() * sizeof(physical_run),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
    if (rows) {
      file_rows_to_ordinals<<<(rows + 255) / 256, 256, 0, stream.get()>>>(
        state->input->rows->mutable_view().data<uint64_t>(),
        state->input->source_indices ? state->input->source_indices->view().data<int32_t>()
                                     : nullptr,
        rows,
        state->device->data(),
        state->device->size(),
        domain->files.size(),
        state->invalid->data());
      CUDF_CUDA_TRY(cudaGetLastError());
      if (state->invalid->value(stream))
        throw std::invalid_argument("reader position outside input domain");
    }
    auto contents         = state->input->rows->release();
    state->result         = {row_position_state::per_row, std::move(domain)};
    state->result.per_row = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                                           rows,
                                                           std::move(*contents.data),
                                                           std::move(*contents.null_mask),
                                                           0);
    state->result.validate_on(rows, stream, mr);
    detail::retire_reads(state, stream, unjoined);
    return std::move(state->result);
  } catch (...) {
    if (state) detail::retire_reads(state, stream, unjoined);
    throw;
  }
}

row_lineage select_host_lineage(row_lineage const& before,
                                std::span<int64_t const> selected,
                                ::cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  if (selected.size() > std::numeric_limits<cudf::size_type>::max())
    throw std::invalid_argument("delivered selection exceeds column limit");
  auto indices = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           selected.size(),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  if (!selected.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(indices->mutable_view().data<int64_t>(),
                                  selected.data(),
                                  selected.size_bytes(),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  return before.select(indices->view(), stream, mr);
}

void validate_lineage_composition(row_lineage const& before,
                                  row_lineage const& step,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  std::unordered_map<std::string, int64_t> files;
  for (std::size_t i = 0; i < before.domain->files.size(); ++i)
    files.emplace(before.domain->files[i].identity, i);
  std::vector<int64_t> remap;
  for (auto const& file : step.domain->files) {
    auto it = files.find(file.identity);
    if (it == files.end() || before.domain->files[it->second].rows != file.rows)
      throw std::invalid_argument("selection composition file binding mismatch");
    remap.push_back(it->second);
  }
  std::vector<physical_run> a, b;
  int64_t offset = 0;
  for (auto const& run : before.domain->file_runs) {
    if (run.count) a.push_back({offset, static_cast<int64_t>(run.file), run.first, run.count});
    offset += run.count;
  }
  offset = 0;
  for (auto const& run : step.domain->file_runs) {
    if (run.count) b.push_back({offset, remap[run.file], run.first, run.count});
    offset += run.count;
  }
  if (!before.output_length()) return;
  std::vector<selection_run> selection;
  offset = 0;
  for (auto const& [first, count] : before.runs) {
    selection.push_back({offset, first, count});
    offset += count;
  }
  rmm::device_uvector<physical_run> da(a.size(), stream, mr), db(b.size(), stream, mr);
  rmm::device_uvector<selection_run> ds(selection.size(), stream, mr);
  // Finish each small metadata upload before starting an operation that can throw.
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    da.data(), a.data(), a.size() * sizeof(physical_run), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    db.data(), b.data(), b.size() * sizeof(physical_run), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  if (!selection.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(ds.data(),
                                  selection.data(),
                                  selection.size() * sizeof(selection_run),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  int const zero = 0;
  rmm::device_scalar<int> invalid(zero, stream, mr);
  auto rows = before.output_length();
  check_composition<<<(rows + 255) / 256, 256, 0, stream.get()>>>(
    rows,
    before.per_row ? before.per_row->view().data<int64_t>() : nullptr,
    ds.data(),
    ds.size(),
    da.data(),
    da.size(),
    db.data(),
    db.size(),
    invalid.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  if (invalid.value(stream))
    throw std::invalid_argument("selection composition input sequence mismatch");
}

std::unique_ptr<cudf::column> concatenate_lineage_ordinals(std::span<row_lineage const> parts,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  int64_t rows = 0;
  for (auto const& part : parts) {
    auto count = part.output_length();
    if (count > std::numeric_limits<cudf::size_type>::max() - rows)
      throw std::invalid_argument("concatenated lineage exceeds column limit");
    rows += count;
  }
  auto result = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
  int64_t output_offset = 0, domain_offset = 0;
  for (auto const& part : parts) {
    auto count = part.output_length();
    if (count && part.state == row_position_state::per_row) {
      copy_shift<<<(count + 255) / 256, 256, 0, stream.get()>>>(
        result->mutable_view().data<int64_t>() + output_offset,
        part.per_row->view().data<int64_t>(),
        count,
        domain_offset);
    } else if (count) {
      std::vector<selection_run> host;
      if (part.state == row_position_state::identity)
        host.push_back({0, 0, count});
      else {
        int64_t offset = 0;
        for (auto const& [first, size] : part.runs) {
          host.push_back({offset, first, size});
          offset += size;
        }
      }
      rmm::device_uvector<selection_run> device(host.size(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(device.data(),
                                    host.data(),
                                    host.size() * sizeof(selection_run),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      select_runs<int64_t><<<(count + 255) / 256, 256, 0, stream.get()>>>(
        result->mutable_view().data<int64_t>() + output_offset,
        nullptr,
        count,
        device.data(),
        host.size(),
        domain_offset);
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    }
    CUDF_CUDA_TRY(cudaGetLastError());
    output_offset += count;
    domain_offset += part.input_domain_length();
  }
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return result;
}

std::unique_ptr<cudf::column> row_lineage::ordinal_column(::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr) const
{
  auto rows = output_length();
  validate(rows);
  if (rows > std::numeric_limits<cudf::size_type>::max())
    throw std::invalid_argument("lineage selection exceeds column limit");
  if (state == row_position_state::per_row)
    return std::make_unique<cudf::column>(per_row->view(), stream, mr);
  std::vector<selection_run> host;
  if (state == row_position_state::identity)
    host.push_back({0, 0, rows});
  else {
    int64_t offset = 0;
    for (auto const& [first, count] : runs) {
      host.push_back({offset, first, count});
      offset += count;
    }
  }
  auto result = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (rows) {
    rmm::device_uvector<selection_run> device(host.size(), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(device.data(),
                                  host.data(),
                                  device.size() * sizeof(selection_run),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    fill_ordinals<<<(rows + 255) / 256, 256, 0, stream.get()>>>(
      result->mutable_view().data<int64_t>(), rows, device.data(), host.size());
    CUDF_CUDA_TRY(cudaGetLastError());
    // Retire the pageable interval upload before host storage is destroyed.
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  return result;
}

row_lineage row_lineage::select(cudf::column_view selected,
                                ::cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr) const
{
  if (state == row_position_state::lost || state == row_position_state::none) return *this;
  if ((selected.type().id() != cudf::type_id::INT32 &&
       selected.type().id() != cudf::type_id::INT64) ||
      selected.has_nulls())
    throw std::invalid_argument("delivered selection requires non-null integer ordinals");
  validate_on(output_length(), stream, mr);
  // Validate indices before DONT_CHECK gather; selection is independent of the
  // producer's coverage report and may be supplied by a replacement decoder.
  auto indices = cudf::cast(selected, cudf::data_type{cudf::type_id::INT64}, stream, mr);
  row_lineage index_lineage;
  auto index_domain       = std::make_shared<lineage_domain>();
  index_domain->files     = {{"delivered selection", output_length()}};
  index_domain->file_runs = {{0, 0, output_length()}};
  index_lineage.domain    = std::move(index_domain);
  index_lineage.state     = row_position_state::per_row;
  index_lineage.per_row   = std::move(indices);
  index_lineage.validate_on(selected.size(), stream, mr);
  row_lineage result;
  result.domain = domain;
  result.state  = row_position_state::per_row;
  if (state == row_position_state::identity) {
    result.per_row = std::move(index_lineage.per_row);
  } else if (state == row_position_state::per_row) {
    auto table     = cudf::gather(cudf::table_view{{per_row->view()}},
                              selected,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              stream,
                              mr);
    result.per_row = std::move(table->release().front());
  } else {
    std::vector<selection_run> host;
    int64_t offset = 0;
    for (auto const& [first, count] : runs) {
      host.push_back({offset, first, count});
      offset += count;
    }
    auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                            selected.size(),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            mr);
    if (selected.size()) {
      rmm::device_uvector<selection_run> device(host.size(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(device.data(),
                                    host.data(),
                                    host.size() * sizeof(selection_run),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      select_runs<int64_t><<<(selected.size() + 255) / 256, 256, 0, stream.get()>>>(
        output->mutable_view().data<int64_t>(),
        index_lineage.per_row->view().data<int64_t>(),
        selected.size(),
        device.data(),
        host.size());
      CUDF_CUDA_TRY(cudaGetLastError());
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    }
    result.per_row = std::move(output);
  }
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return result;
}

void append_lineage_virtual_columns(owning_table_view& table,
                                    row_lineage const& lineage,
                                    scan_plan const& plan,
                                    std::span<uint64_t const> file_indices,
                                    ::cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  if (plan.virtual_columns.empty()) return;
  if (table.n_columns() != plan.data_columns.size())
    throw std::invalid_argument("virtual assembly requires normalized physical columns");
  lineage.validate_on(table.num_rows(), stream, mr);
  if (!lineage.domain) throw std::invalid_argument("virtual assembly requires bound file context");
  using kind     = scan_plan::parquet_virtual_column_kind;
  bool need_rows = false, need_files = false;
  for (auto const& col : plan.virtual_columns) {
    need_rows |= col.kind == kind::FILE_ROW_NUMBER;
    need_files |= col.kind != kind::FILE_ROW_NUMBER;
    if (col.kind == kind::FILE_INDEX && file_indices.size() != lineage.domain->files.size())
      throw std::invalid_argument("file-index virtual column needs bound file indices");
  }
  bool const resolve_positions = need_rows || (need_files && lineage.domain->files.size() != 1);
  if (resolve_positions &&
      (lineage.state == row_position_state::none || lineage.state == row_position_state::lost))
    throw std::invalid_argument("virtual assembly requires original positions");
  struct virtual_state {
    row_lineage lineage;
    std::unique_ptr<cudf::column> ordinals, rows, files;
    std::vector<physical_run> runs;
    std::unique_ptr<rmm::device_uvector<physical_run>> device;
    std::vector<std::unique_ptr<cudf::column>> columns;
    owning_table_view input;
  };
  auto state       = std::make_unique<virtual_state>();
  state->lineage   = lineage;
  auto const count = table.num_rows();
  int64_t offset   = 0;
  for (auto const& run : lineage.domain->file_runs) {
    if (run.count)
      state->runs.push_back({offset, static_cast<int64_t>(run.file), run.first, run.count});
    offset += run.count;
  }
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    if (resolve_positions) state->ordinals = lineage.ordinal_column(stream, mr);
    if (need_rows)
      state->rows = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64}, count, cudf::mask_state::UNALLOCATED, stream, mr);
    if (need_files && lineage.domain->files.size() > 1)
      state->files = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, count, cudf::mask_state::UNALLOCATED, stream, mr);
    state->device =
      std::make_unique<rmm::device_uvector<physical_run>>(state->runs.size(), stream, mr);
    if (!state->runs.empty())
      CUDF_CUDA_TRY(cudaMemcpyAsync(state->device->data(),
                                    state->runs.data(),
                                    state->runs.size() * sizeof(physical_run),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
    if (count && resolve_positions) {
      physical_positions<<<(count + 255) / 256, 256, 0, stream.get()>>>(
        state->ordinals->view().data<int64_t>(),
        count,
        state->device->data(),
        state->device->size(),
        state->rows ? state->rows->mutable_view().data<int64_t>() : nullptr,
        state->files ? state->files->mutable_view().data<int32_t>() : nullptr);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    auto upload = [&](auto const& values) {
      using T = typename std::decay_t<decltype(values)>::value_type;
      rmm::device_buffer result(values.data(), values.size() * sizeof(T), stream, mr);
      stream.sync();
      return result;
    };
    for (auto const& col : plan.virtual_columns) {
      if (col.kind == kind::FILE_ROW_NUMBER) {
        state->columns.push_back(std::make_unique<cudf::column>(state->rows->view(), stream, mr));
        continue;
      }
      if (lineage.domain->files.size() == 1) {
        if (col.kind == kind::FILENAME) {
          cudf::string_scalar value(lineage.domain->files[0].identity, true, stream, mr);
          state->columns.push_back(cudf::make_column_from_scalar(value, count, stream, mr));
        } else {
          cudf::numeric_scalar<uint64_t> value(file_indices[0], true, stream, mr);
          state->columns.push_back(cudf::make_column_from_scalar(value, count, stream, mr));
        }
        continue;
      }
      std::unique_ptr<cudf::column> dictionary;
      if (col.kind == kind::FILE_INDEX) {
        std::vector<uint64_t> indices(file_indices.begin(), file_indices.end());
        dictionary = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                                    indices.size(),
                                                    upload(indices),
                                                    rmm::device_buffer{},
                                                    0);
      } else {
        std::string chars;
        std::vector<int64_t> offsets{0};
        for (auto const& file : lineage.domain->files) {
          chars += file.identity;
          offsets.push_back(chars.size());
        }
        auto offset_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                                         offsets.size(),
                                                         upload(offsets),
                                                         rmm::device_buffer{},
                                                         0);
        dictionary      = cudf::make_strings_column(lineage.domain->files.size(),
                                               std::move(offset_col),
                                               upload(chars),
                                               0,
                                               rmm::device_buffer{});
      }
      auto output = cudf::gather(cudf::table_view{{dictionary->view()}},
                                 state->files->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
      stream.sync();
      state->columns.push_back(std::move(output->release().front()));
    }
    state->input     = std::move(table);
    auto input_table = state->input.release(stream, mr);
    auto columns =
      input_table ? input_table->release() : std::vector<std::unique_ptr<cudf::column>>{};
    for (auto& col : state->columns)
      columns.push_back(std::move(col));
    table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))};
    detail::retire_reads(state, stream, unjoined);
  } catch (...) {
    if (state) detail::retire_reads(state, stream, unjoined);
    throw;
  }
}

void apply_lineage_visibility(
  owning_table_view& table,
  row_lineage& lineage,
  std::function<std::span<int64_t const>(std::string const&)> const& lookup,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  lineage.validate_on(table.num_rows(), stream, mr);
  if (!table.num_rows()) return;
  if (!lineage.domain || lineage.state == row_position_state::none ||
      lineage.state == row_position_state::lost)
    throw std::invalid_argument("visibility requires recoverable row lineage");
  struct visibility_state {
    owning_table_view input;
    row_lineage lineage;
    std::vector<int64_t> deletes;
    std::vector<visibility_run> runs;
    std::unique_ptr<rmm::device_uvector<int64_t>> device_deletes;
    std::unique_ptr<rmm::device_uvector<visibility_run>> device_runs;
    std::unique_ptr<cudf::column> owned_ordinals, mask;
    std::unique_ptr<cudf::table> filtered;
  };
  auto state     = std::make_unique<visibility_state>();
  state->lineage = lineage;
  std::vector<std::pair<int64_t, int64_t>> windows;
  for (auto const& file : lineage.domain->files) {
    auto values = lookup(file.identity);
    if (!std::is_sorted(values.begin(), values.end()) ||
        std::adjacent_find(values.begin(), values.end()) != values.end() ||
        (!values.empty() && (values.front() < 0 || values.back() >= file.rows)))
      throw std::invalid_argument(
        "lineage visibility requires valid sorted unique delete positions");
    windows.emplace_back(state->deletes.size(), values.size());
    state->deletes.insert(state->deletes.end(), values.begin(), values.end());
  }
  if (state->deletes.empty()) return;
  int64_t offset = 0;
  for (auto const& run : lineage.domain->file_runs) {
    state->runs.push_back(
      {offset, run.first, run.count, windows[run.file].first, windows[run.file].second});
    offset += run.count;
  }
  auto const unjoined = detail::prepare_unjoined_failure();
  state->input        = std::move(table);
  try {
    state->device_deletes =
      std::make_unique<rmm::device_uvector<int64_t>>(state->deletes.size(), stream, mr);
    state->device_runs =
      std::make_unique<rmm::device_uvector<visibility_run>>(state->runs.size(), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(state->device_deletes->data(),
                                  state->deletes.data(),
                                  state->deletes.size() * sizeof(int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(state->device_runs->data(),
                                  state->runs.data(),
                                  state->runs.size() * sizeof(visibility_run),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    cudf::column_view ordinals;
    if (lineage.state == row_position_state::per_row)
      ordinals = lineage.per_row->view();
    else {
      state->owned_ordinals = lineage.ordinal_column(stream, mr);
      ordinals              = state->owned_ordinals->view();
    }
    auto rows   = state->input.num_rows();
    state->mask = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    visibility_mask<<<(rows + 255) / 256, 256, 0, stream.get()>>>(
      state->mask->mutable_view().data<uint8_t>(),
      ordinals.data<int64_t>(),
      rows,
      state->device_runs->data(),
      state->runs.size(),
      state->device_deletes->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    auto columns = state->input.view();
    std::vector<cudf::column_view> augmented(columns.begin(), columns.end());
    augmented.push_back(ordinals);
    state->filtered =
      sirius::apply_retention_mask(cudf::table_view{augmented}, state->mask->view(), stream, mr);
    detail::retire_reads(state, stream, unjoined);
    auto output     = state->filtered->release();
    lineage.per_row = std::move(output.back());
    output.pop_back();
    lineage.state = row_position_state::per_row;
    lineage.runs.clear();
    table = owning_table_view{std::make_unique<cudf::table>(std::move(output))};
  } catch (...) {
    if (state) detail::retire_reads(state, stream, unjoined);
    throw;
  }
}
}  // namespace sirius::op::scan
