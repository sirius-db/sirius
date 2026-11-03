/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "memory/size_arithmetic.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/row_lineage.hpp"
#include "op/scan/scan_plan.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/null_mask.hpp>
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
unsigned int detail::lineage_grid_size(int64_t rows)
{
  if (rows < 0) throw std::invalid_argument("negative lineage launch size");
  if (!rows) return 0;
  // Division before addition also covers INT64_MAX without overflow.
  auto const blocks = rows / 256 + (rows % 256 != 0);
  int device, max_grid_x;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device));
  if (blocks > max_grid_x || blocks > std::numeric_limits<unsigned int>::max())
    throw std::invalid_argument("lineage launch exceeds device grid limit");
  return static_cast<unsigned int>(blocks);
}

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
  output[i] = runs[lo].input_first + (index - runs[lo].output_first) + shift;
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
  if (rows) rows[i] = run.first + (ordinals[i] - run.ordinal);
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
  values[i] = runs[lo].ordinal + (row - runs[lo].first);
}
__global__ void check_composition(int64_t count,
                                  int64_t const* selected,
                                  physical_run const* before,
                                  int64_t before_count,
                                  physical_run const* next,
                                  int64_t next_count,
                                  int* invalid)
{
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  auto const ordinal = selected[i];
  auto a             = resolve_run(before, before_count, ordinal);
  auto b             = resolve_run(next, next_count, i);
  if (a.file != b.file || a.first + (ordinal - a.ordinal) != b.first + (i - b.ordinal))
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
  auto row       = run.first + (ordinal - run.ordinal);
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
  output[i] = runs[lo].input_first + (i - runs[lo].output_first);
}
}  // namespace

void row_lineage::validate_on(int64_t rows,
                              ::cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr) const
{
  validate_on_impl(rows, stream, mr, false);
}

void row_lineage::validate_on_impl(int64_t rows,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr,
                                   bool strict) const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  validate_impl(rows, strict);
  if (!strict && is_validated()) return;
  if (state != row_position_state::per_row || !rows) return;
  struct validation_state {
    row_lineage input;
    std::unique_ptr<rmm::device_uvector<int64_t>> sorted;
    std::unique_ptr<rmm::device_scalar<int>> invalid;
  };
  auto retained       = std::make_unique<validation_state>();
  retained->input     = *this;
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    retained->sorted = std::make_unique<rmm::device_uvector<int64_t>>(rows, stream, mr);
    auto& sorted     = *retained->sorted;
    CUDF_CUDA_TRY(cudaMemcpyAsync(sorted.data(),
                                  per_row->view().data<int64_t>(),
                                  rows * sizeof(int64_t),
                                  cudaMemcpyDeviceToDevice,
                                  stream.get()));
    thrust::sort(rmm::exec_policy(stream, mr), sorted.begin(), sorted.end());
    retained->invalid = std::make_unique<rmm::device_scalar<int>>(stream, mr);
    retained->invalid->set_value_to_zero_async(stream);
    check_ordinals<<<detail::lineage_grid_size(rows), 256, 0, stream.get()>>>(
      sorted.data(), rows, input_domain_length(), retained->invalid->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    if (retained->invalid->value(stream))
      throw std::invalid_argument("invalid or duplicate lineage ordinal");
  } catch (...) {
    detail::retire_reads(retained, stream, unjoined);
    throw;
  }
}

row_lineage row_lineage::validated_copy(int64_t rows,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr) const
{
  validate(rows);
  // A copy of an already sealed object still goes through strict device
  // validation when it crosses the external decoder boundary again.
  if (is_validated()) {
    validate_on_impl(rows, stream, mr, true);
    return *this;
  }
  struct import_state {
    row_lineage source, output;
    std::unique_ptr<cudf::column> copied_column;
    std::shared_ptr<cudf::column const> column;
  };
  auto state          = std::make_unique<import_state>();
  state->source       = *this;
  auto frozen_domain  = domain() ? std::make_shared<lineage_domain const>(*domain()) : nullptr;
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    if (per_row()) {
      state->copied_column = std::make_unique<cudf::column>(per_row()->view(), stream, mr);
      state->column        = std::move(state->copied_column);
    }
    state->output = {this->state(), std::move(frozen_domain), runs(), state->column};
    state->output.validate_on(rows, stream, mr);
    // validate_on's device result read joins all preceding copy/validation work.
    state->output.seal_validation(source_binding());
    return std::move(state->output);
  } catch (...) {
    detail::retire_reads(state, stream, unjoined);
    throw;
  }
}

std::size_t lineage_conversion_workspace_bytes(std::size_t rows, std::size_t intervals)
{
  auto const max = std::numeric_limits<std::size_t>::max();
  // Validation owns a sorted copy and sort scratch, including the pinned
  // Thrust implementation's fixed small-sort bookkeeping. Row storage is reused.
  constexpr std::size_t fixed = 256 + 2 * sizeof(int);
  if (rows > (max - fixed) / 32 || intervals > (max - rows * 32 - fixed) / sizeof(physical_run))
    return max;
  return rows * 32 + intervals * sizeof(physical_run) + fixed;
}

row_lineage consume_reader_positions(std::unique_ptr<gpu_row_positions> positions,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr,
                                     std::shared_ptr<lineage_domain const> expected_domain)
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
  auto state   = std::make_unique<conversion_state>();
  state->input = std::move(positions);
  auto domain  = std::make_shared<lineage_domain>();
  std::unordered_map<std::string, std::size_t> files;
  int64_t offset = 0;
  for (std::size_t i = 0; i < state->input->sources.size(); ++i) {
    auto const& source = state->input->sources[i];
    if (source.file_rows > INT64_MAX) throw std::invalid_argument("reader file length overflow");
    auto [file, inserted] = files.emplace(source.file, domain->files.size());
    if (inserted)
      domain->files.push_back({source.file, static_cast<int64_t>(source.file_rows)});
    else if (domain->files[file->second].rows != static_cast<int64_t>(source.file_rows))
      throw std::invalid_argument("reader sources disagree on file length");
    uint64_t prior = 0;
    for (auto [first, end] : source.ranges) {
      if (first < prior || end < first || end > source.file_rows ||
          end - first > INT64_MAX - offset)
        throw std::invalid_argument("reader position input intervals overlap or overflow");
      prior = end;
      if (first == end) continue;
      domain->file_runs.push_back(
        {file->second, static_cast<int64_t>(first), static_cast<int64_t>(end - first)});
      state->host.push_back({offset,
                             static_cast<int64_t>(i),
                             static_cast<int64_t>(first),
                             static_cast<int64_t>(end - first)});
      offset += end - first;
    }
  }
  row_lineage{row_position_state::identity, domain}.validate(offset);
  if (expected_domain && *expected_domain != *domain)
    throw std::invalid_argument("decoder changed its immutable input domain");
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    state->device =
      std::make_unique<rmm::device_uvector<physical_run>>(state->host.size(), stream, mr);
    state->invalid = std::make_unique<rmm::device_scalar<int>>(stream, mr);
    state->invalid->set_value_to_zero_async(stream);
    if (!state->host.empty())
      CUDF_CUDA_TRY(cudaMemcpyAsync(state->device->data(),
                                    state->host.data(),
                                    state->host.size() * sizeof(physical_run),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
    if (rows) {
      file_rows_to_ordinals<<<detail::lineage_grid_size(rows), 256, 0, stream.get()>>>(
        state->input->rows->mutable_view().data<uint64_t>(),
        state->input->source_indices ? state->input->source_indices->view().data<int32_t>()
                                     : nullptr,
        rows,
        state->device->data(),
        state->device->size(),
        state->input->sources.size(),
        state->invalid->data());
      CUDF_CUDA_TRY(cudaGetLastError());
      if (state->invalid->value(stream))
        throw std::invalid_argument("reader position outside input domain");
    }
    auto contents = state->input->rows->release();
    state->result = {row_position_state::per_row,
                     std::move(domain),
                     {},
                     std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                                    rows,
                                                    std::move(*contents.data),
                                                    std::move(*contents.null_mask),
                                                    0)};
    state->result.validate_on(rows, stream, mr);
    state->result.seal_validation(std::move(expected_domain));
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
  struct selection_state {
    std::vector<int64_t> host;
    std::unique_ptr<cudf::column> indices;
  };
  auto retained = std::make_unique<selection_state>();
  retained->host.assign(selected.begin(), selected.end());
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    retained->indices = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                  selected.size(),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
    if (!selected.empty()) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(retained->indices->mutable_view().data<int64_t>(),
                                    retained->host.data(),
                                    selected.size_bytes(),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      detail::check_read_checkpoint(detail::read_checkpoint::COPY);
    }
    detail::retire_reads(retained, stream, unjoined);
  } catch (...) {
    if (retained) detail::retire_reads(retained, stream, unjoined);
    throw;
  }
  std::shared_ptr<cudf::column const> indices = std::move(retained->indices);
  auto const view                             = indices->view();
  return before.select_impl(view, stream, mr, std::move(indices));
}

void validate_lineage_composition(row_lineage const& before,
                                  row_lineage const& step,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  std::unordered_map<std::string, int64_t> files;
  for (std::size_t i = 0; i < before.domain()->files.size(); ++i)
    files.emplace(before.domain()->files[i].identity, i);
  std::vector<int64_t> remap;
  for (auto const& file : step.domain()->files) {
    auto it = files.find(file.identity);
    if (it == files.end() || before.domain()->files[it->second].rows != file.rows)
      throw std::invalid_argument("selection composition file binding mismatch");
    remap.push_back(it->second);
  }
  std::vector<physical_run> a, b;
  int64_t offset = 0;
  for (auto const& run : before.domain()->file_runs) {
    if (run.count) a.push_back({offset, static_cast<int64_t>(run.file), run.first, run.count});
    offset += run.count;
  }
  offset = 0;
  for (auto const& run : step.domain()->file_runs) {
    if (run.count) b.push_back({offset, remap[run.file], run.first, run.count});
    offset += run.count;
  }
  if (!before.output_length()) return;
  if (before.state() != row_position_state::per_row) {
    // Compare the complete delivered sequence by interval, regardless of how either
    // domain partitions it. Compact domains can be far larger than a cuDF column.
    std::size_t next = 0;
    int64_t consumed = 0;
    auto compare     = [&](int64_t first, int64_t count) {
      auto input = std::lower_bound(a.begin(), a.end(), first, [](auto const& run, auto ordinal) {
        return run.ordinal + run.count <= ordinal;
      });
      while (count) {
        auto const& output = b[next];
        auto const take =
          std::min({count, input->ordinal + input->count - first, output.count - consumed});
        if (input->file != output.file ||
            input->first + (first - input->ordinal) != output.first + consumed)
          throw std::invalid_argument("selection composition input sequence mismatch");
        count -= take;
        first += take;
        consumed += take;
        if (first == input->ordinal + input->count) ++input;
        if (consumed == output.count) {
          ++next;
          consumed = 0;
        }
      }
    };
    if (before.state() == row_position_state::identity)
      compare(0, before.output_length());
    else
      for (auto const& [first, count] : before.runs())
        compare(first, count);
    return;
  }
  struct composition_state {
    row_lineage input;
    std::vector<physical_run> a, b;
    std::unique_ptr<rmm::device_uvector<physical_run>> da, db;
    std::unique_ptr<rmm::device_scalar<int>> invalid;
  };
  auto retained       = std::make_unique<composition_state>();
  retained->input     = before;
  retained->a         = std::move(a);
  retained->b         = std::move(b);
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    auto upload = [&](auto const& host, auto& output) {
      using T = typename std::decay_t<decltype(host)>::value_type;
      output  = std::make_unique<rmm::device_uvector<T>>(host.size(), stream, mr);
      if (!host.empty())
        CUDF_CUDA_TRY(cudaMemcpyAsync(output->data(),
                                      host.data(),
                                      host.size() * sizeof(T),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
    };
    upload(retained->a, retained->da);
    upload(retained->b, retained->db);
    retained->invalid = std::make_unique<rmm::device_scalar<int>>(stream, mr);
    retained->invalid->set_value_to_zero_async(stream);
    auto rows = before.output_length();
    check_composition<<<detail::lineage_grid_size(rows), 256, 0, stream.get()>>>(
      rows,
      before.per_row()->view().data<int64_t>(),
      retained->da->data(),
      retained->da->size(),
      retained->db->data(),
      retained->db->size(),
      retained->invalid->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    if (retained->invalid->value(stream))
      throw std::invalid_argument("selection composition input sequence mismatch");
  } catch (...) {
    detail::retire_reads(retained, stream, unjoined);
    throw;
  }
}

std::unique_ptr<cudf::column> concatenate_lineage_ordinals(std::span<row_lineage const> parts,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  struct concat_state {
    std::vector<row_lineage> parts;
    std::vector<std::vector<selection_run>> runs;
    std::vector<std::unique_ptr<rmm::device_uvector<selection_run>>> devices;
    std::unique_ptr<cudf::column> output;
  };
  auto retained = std::make_unique<concat_state>();
  retained->parts.assign(parts.begin(), parts.end());
  retained->runs.resize(parts.size());
  retained->devices.resize(parts.size());
  int64_t rows = 0;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    auto const& part = parts[i];
    auto count       = part.output_length();
    if (count > std::numeric_limits<cudf::size_type>::max() - rows)
      throw std::invalid_argument("concatenated lineage exceeds column limit");
    rows += count;
    if (part.state() == row_position_state::identity)
      retained->runs[i].push_back({0, 0, count});
    else if (part.state() == row_position_state::runs) {
      int64_t offset = 0;
      for (auto const& [first, size] : part.runs()) {
        retained->runs[i].push_back({offset, first, size});
        offset += size;
      }
    }
  }
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    retained->output = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    int64_t output_offset = 0, domain_offset = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      auto const& part = parts[i];
      auto count       = part.output_length();
      if (count && part.state() == row_position_state::per_row) {
        copy_shift<<<detail::lineage_grid_size(count), 256, 0, stream.get()>>>(
          retained->output->mutable_view().data<int64_t>() + output_offset,
          part.per_row()->view().data<int64_t>(),
          count,
          domain_offset);
      } else if (count) {
        auto const& host = retained->runs[i];
        auto& device     = retained->devices[i];
        device = std::make_unique<rmm::device_uvector<selection_run>>(host.size(), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(device->data(),
                                      host.data(),
                                      host.size() * sizeof(selection_run),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
        select_runs<int64_t><<<detail::lineage_grid_size(count), 256, 0, stream.get()>>>(
          retained->output->mutable_view().data<int64_t>() + output_offset,
          nullptr,
          count,
          device->data(),
          host.size(),
          domain_offset);
      }
      CUDF_CUDA_TRY(cudaGetLastError());
      output_offset += count;
      domain_offset += part.input_domain_length();
    }
    detail::retire_reads(retained, stream, unjoined);
    return std::move(retained->output);
  } catch (...) {
    if (retained) detail::retire_reads(retained, stream, unjoined);
    throw;
  }
}

std::unique_ptr<cudf::column> row_lineage::ordinal_column(::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr) const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  auto rows                                                      = output_length();
  validate(rows);
  if (rows > std::numeric_limits<cudf::size_type>::max())
    throw std::invalid_argument("lineage selection exceeds column limit");
  struct ordinal_state {
    row_lineage input;
    std::vector<selection_run> runs;
    std::unique_ptr<rmm::device_uvector<selection_run>> device;
    std::unique_ptr<cudf::column> output;
  };
  auto retained   = std::make_unique<ordinal_state>();
  retained->input = *this;
  if (state == row_position_state::identity)
    retained->runs.push_back({0, 0, rows});
  else {
    int64_t offset = 0;
    for (auto const& [first, count] : runs) {
      retained->runs.push_back({offset, first, count});
      offset += count;
    }
  }
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    if (state == row_position_state::per_row)
      retained->output = std::make_unique<cudf::column>(per_row->view(), stream, mr);
    else {
      retained->output = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
      if (rows) {
        retained->device =
          std::make_unique<rmm::device_uvector<selection_run>>(retained->runs.size(), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(retained->device->data(),
                                      retained->runs.data(),
                                      retained->runs.size() * sizeof(selection_run),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
        fill_ordinals<<<detail::lineage_grid_size(rows), 256, 0, stream.get()>>>(
          retained->output->mutable_view().data<int64_t>(),
          rows,
          retained->device->data(),
          retained->runs.size());
        CUDF_CUDA_TRY(cudaGetLastError());
      }
    }
    if (rows) detail::retire_reads(retained, stream, unjoined);
    return std::move(retained->output);
  } catch (...) {
    if (retained) detail::retire_reads(retained, stream, unjoined);
    throw;
  }
}

row_lineage row_lineage::select(cudf::column_view selected,
                                ::cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr) const
{
  return select_impl(selected, stream, mr, {});
}

row_lineage row_lineage::select_impl(cudf::column_view selected,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr,
                                     std::shared_ptr<cudf::column const> selected_owner) const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  if (state == row_position_state::lost || state == row_position_state::none) return *this;
  if ((selected.type().id() != cudf::type_id::INT32 &&
       selected.type().id() != cudf::type_id::INT64) ||
      selected.has_nulls())
    throw std::invalid_argument("delivered selection requires non-null integer ordinals");
  validate_on(output_length(), stream, mr);
  struct selection_state {
    row_lineage source, indices, output;
    std::shared_ptr<cudf::column const> selected_owner;
    std::vector<selection_run> runs;
    std::unique_ptr<rmm::device_uvector<selection_run>> device;
    std::unique_ptr<cudf::column> column;
    std::shared_ptr<cudf::column const> output_column;
  };
  auto retained            = std::make_unique<selection_state>();
  retained->source         = *this;
  retained->selected_owner = std::move(selected_owner);
  auto index_domain        = std::make_shared<lineage_domain>();
  index_domain->files      = {{"delivered selection", output_length()}};
  index_domain->file_runs  = {{0, 0, output_length()}};
  auto const unjoined      = detail::prepare_unjoined_failure();
  try {
    // A selection supplied through the public API remains untrusted. Only the
    // already sealed source avoids another full copy and sort.
    retained->column = cudf::cast(selected, cudf::data_type{cudf::type_id::INT64}, stream, mr);
    // Keep an owner in the retirement state even if allocating the immutable
    // host record fails after the cast/gather has queued device work.
    retained->output_column = std::move(retained->column);
    retained->indices       = {
      row_position_state::per_row, std::move(index_domain), {}, retained->output_column};
    retained->indices.validate_on(selected.size(), stream, mr);
    auto& result = retained->output;
    if (state == row_position_state::identity) {
      result = {row_position_state::per_row, domain, {}, retained->indices.per_row()};
    } else if (state == row_position_state::per_row) {
      auto table              = cudf::gather(cudf::table_view{{per_row->view()}},
                                retained->indices.per_row()->view(),
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                stream,
                                mr);
      retained->column        = std::move(table->release().front());
      retained->output_column = std::move(retained->column);
      result                  = {row_position_state::per_row, domain, {}, retained->output_column};
    } else {
      int64_t offset = 0;
      for (auto const& [first, count] : runs) {
        retained->runs.push_back({offset, first, count});
        offset += count;
      }
      retained->column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                   selected.size(),
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);
      if (selected.size()) {
        retained->device =
          std::make_unique<rmm::device_uvector<selection_run>>(retained->runs.size(), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(retained->device->data(),
                                      retained->runs.data(),
                                      retained->runs.size() * sizeof(selection_run),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
        select_runs<int64_t><<<detail::lineage_grid_size(selected.size()), 256, 0, stream.get()>>>(
          retained->column->mutable_view().data<int64_t>(),
          retained->indices.per_row()->view().data<int64_t>(),
          selected.size(),
          retained->device->data(),
          retained->runs.size());
        CUDF_CUDA_TRY(cudaGetLastError());
      }
      retained->output_column = std::move(retained->column);
      result                  = {row_position_state::per_row, domain, {}, retained->output_column};
    }
    detail::check_read_checkpoint(detail::read_checkpoint::PAYLOAD_GATHER);
    // One checked lifetime boundary covers the input, uploaded intervals and gather.
    detail::retire_reads(retained, stream, unjoined);
    if (is_validated()) result.seal_validation(source_binding());
    return std::move(result);
  } catch (...) {
    if (retained) detail::retire_reads(retained, stream, unjoined);
    throw;
  }
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
  if (!lineage.domain())
    throw std::invalid_argument("virtual assembly requires bound file context");
  using kind     = scan_plan::parquet_virtual_column_kind;
  bool need_rows = false, need_files = false;
  for (auto const& col : plan.virtual_columns) {
    need_rows |= col.kind == kind::FILE_ROW_NUMBER;
    need_files |= col.kind != kind::FILE_ROW_NUMBER;
    if (col.kind == kind::FILE_INDEX && file_indices.size() != lineage.domain()->files.size())
      throw std::invalid_argument("file-index virtual column needs bound file indices");
  }
  bool const resolve_positions = need_rows || (need_files && lineage.domain()->files.size() != 1);
  if (resolve_positions &&
      (lineage.state() == row_position_state::none || lineage.state() == row_position_state::lost))
    throw std::invalid_argument("virtual assembly requires original positions");
  struct virtual_state {
    row_lineage lineage;
    std::unique_ptr<cudf::column> ordinals, rows, files;
    std::vector<physical_run> runs;
    std::unique_ptr<rmm::device_uvector<physical_run>> device;
    std::vector<std::unique_ptr<cudf::column>> columns;
    std::vector<std::unique_ptr<cudf::column>> dictionaries;
    std::vector<std::shared_ptr<void const>> uploads;
    owning_table_view input;
  };
  auto state       = std::make_unique<virtual_state>();
  state->lineage   = lineage;
  auto const count = table.num_rows();
  int64_t offset   = 0;
  for (auto const& run : lineage.domain()->file_runs) {
    if (run.count)
      state->runs.push_back({offset, static_cast<int64_t>(run.file), run.first, run.count});
    offset += run.count;
  }
  auto const unjoined = detail::prepare_unjoined_failure();
  try {
    cudf::column_view ordinal_view;
    if (count && resolve_positions) {
      if (lineage.state() == row_position_state::per_row)
        ordinal_view = lineage.per_row()->view();
      else {
        state->ordinals = lineage.ordinal_column(stream, mr);
        ordinal_view    = state->ordinals->view();
      }
    }
    if (need_rows)
      state->rows = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64}, count, cudf::mask_state::UNALLOCATED, stream, mr);
    if (need_files && lineage.domain()->files.size() > 1)
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
      physical_positions<<<detail::lineage_grid_size(count), 256, 0, stream.get()>>>(
        ordinal_view.data<int64_t>(),
        count,
        state->device->data(),
        state->device->size(),
        state->rows ? state->rows->mutable_view().data<int64_t>() : nullptr,
        state->files ? state->files->mutable_view().data<int32_t>() : nullptr);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    auto upload = [&](auto values) {
      using Values = std::decay_t<decltype(values)>;
      using T      = typename Values::value_type;
      auto owner   = std::make_shared<Values const>(std::move(values));
      state->uploads.push_back(owner);
      return rmm::device_buffer(owner->data(), owner->size() * sizeof(T), stream, mr);
    };
    for (auto const& col : plan.virtual_columns) {
      if (!count) {
        auto type = col.kind == kind::FILENAME     ? cudf::type_id::STRING
                    : col.kind == kind::FILE_INDEX ? cudf::type_id::UINT64
                                                   : cudf::type_id::INT64;
        state->columns.push_back(cudf::make_empty_column(type));
        continue;
      }
      if (col.kind == kind::FILE_ROW_NUMBER) {
        state->columns.push_back(std::make_unique<cudf::column>(state->rows->view(), stream, mr));
        continue;
      }
      if (lineage.domain()->files.size() == 1) {
        if (col.kind == kind::FILENAME) {
          cudf::string_scalar value(lineage.domain()->files[0].identity, true, stream, mr);
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
        auto const index_count = indices.size();
        dictionary             = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::UINT64},
          index_count,
          upload(std::move(indices)),
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
          0);
      } else {
        std::string chars;
        std::vector<int64_t> offsets{0};
        for (auto const& file : lineage.domain()->files) {
          chars += file.identity;
          offsets.push_back(chars.size());
        }
        auto const offset_count = offsets.size();
        auto offset_col         = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT64},
          offset_count,
          upload(std::move(offsets)),
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
          0);
        dictionary = cudf::make_strings_column(
          lineage.domain()->files.size(),
          std::move(offset_col),
          upload(std::move(chars)),
          0,
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
      }
      state->dictionaries.push_back(std::move(dictionary));
      auto output = cudf::gather(cudf::table_view{{state->dictionaries.back()->view()}},
                                 state->files->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
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
  if (!lineage.domain() || lineage.state() == row_position_state::none ||
      lineage.state() == row_position_state::lost)
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
  std::vector<std::vector<std::pair<int64_t, int64_t>>> ranges(lineage.domain()->files.size());
  for (auto const& run : lineage.domain()->file_runs)
    if (run.count) ranges[run.file].emplace_back(run.first, run.first + run.count);
  std::vector<std::pair<int64_t, int64_t>> windows;
  for (std::size_t file = 0; file < lineage.domain()->files.size(); ++file) {
    auto values = lookup(lineage.domain()->files[file].identity);
    if (!std::is_sorted(values.begin(), values.end()))
      throw std::invalid_argument("lineage visibility requires sorted delete positions");
    auto& selected = ranges[file];
    std::sort(selected.begin(), selected.end());
    auto start = state->deletes.size();
    // Merge repeated/overlapping input intervals. Each file's selected deletes
    // are uploaded once even when concatenation repeats the same physical rows.
    for (std::size_t i = 0; i < selected.size();) {
      auto [first, end] = selected[i++];
      while (i < selected.size() && selected[i].first <= end)
        end = std::max(end, selected[i++].second);
      // Signed searches ignore eager V2 positions outside these intervals.
      auto lo = std::lower_bound(values.begin(), values.end(), first);
      auto hi = std::lower_bound(lo, values.end(), end);
      state->deletes.insert(state->deletes.end(), lo, hi);
    }
    windows.emplace_back(start, state->deletes.size() - start);
  }
  if (state->deletes.empty()) return;
  int64_t offset = 0;
  for (auto const& run : lineage.domain()->file_runs) {
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
    if (lineage.state() == row_position_state::per_row)
      ordinals = lineage.per_row()->view();
    else {
      state->owned_ordinals = lineage.ordinal_column(stream, mr);
      ordinals              = state->owned_ordinals->view();
    }
    auto rows   = state->input.num_rows();
    state->mask = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    visibility_mask<<<detail::lineage_grid_size(rows), 256, 0, stream.get()>>>(
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
    auto output          = state->filtered->release();
    bool const validated = lineage.is_validated();
    auto binding         = lineage.source_binding();
    lineage = {row_position_state::per_row, lineage.domain(), {}, std::move(output.back())};
    output.pop_back();
    if (validated) lineage.seal_validation(std::move(binding));
    table = owning_table_view{std::make_unique<cudf::table>(std::move(output))};
  } catch (...) {
    if (state) detail::retire_reads(state, stream, unjoined);
    throw;
  }
}
}  // namespace sirius::op::scan
