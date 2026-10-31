/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/row_lineage.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace sirius::op::scan {
row_lineage::row_lineage(row_position_state state,
                         std::shared_ptr<lineage_domain const> domain,
                         std::vector<std::pair<int64_t, int64_t>> runs,
                         std::shared_ptr<cudf::column const> per_row)
  : _selection(std::make_shared<selection const>(
      state, std::move(domain), std::move(runs), std::move(per_row), false, nullptr))
{
}

std::shared_ptr<lineage_domain const> row_lineage::source_binding() const
{
  return is_validated() ? data().binding : domain();
}

bool row_lineage::matches_domain(std::shared_ptr<lineage_domain const> const& expected) const
{
  return domain() == expected || (expected && domain() && is_validated() &&
                                  data().binding == expected && *domain() == *expected);
}

void row_lineage::seal_validation(std::shared_ptr<lineage_domain const> binding)
{
  _selection = std::make_shared<selection const>(
    state(), domain(), runs(), per_row(), true, binding ? std::move(binding) : domain());
}

void validate_lineage_composition(row_lineage const&,
                                  row_lineage const&,
                                  ::cuda::stream_ref,
                                  rmm::device_async_resource_ref);
row_lineage select_host_lineage(row_lineage const&,
                                std::span<int64_t const>,
                                ::cuda::stream_ref,
                                rmm::device_async_resource_ref);
std::unique_ptr<cudf::column> concatenate_lineage_ordinals(std::span<row_lineage const>,
                                                           ::cuda::stream_ref,
                                                           rmm::device_async_resource_ref);
namespace {
int64_t add(int64_t a, int64_t b)
{
  if (a < 0 || b < 0 || b > std::numeric_limits<int64_t>::max() - a)
    throw std::invalid_argument("lineage domain length overflow");
  return a + b;
}
bool recoverable(row_position_state state)
{
  return state == row_position_state::identity || state == row_position_state::runs ||
         state == row_position_state::per_row;
}
row_lineage compact(std::shared_ptr<lineage_domain const> domain,
                    std::span<int64_t const> ordinals,
                    ::cuda::stream_ref stream,
                    rmm::device_async_resource_ref mr)
{
  auto state = row_position_state::runs;
  std::vector<std::pair<int64_t, int64_t>> runs;
  std::shared_ptr<cudf::column const> per_row;
  for (auto ordinal : ordinals) {
    if (!runs.empty() && runs.back().first + runs.back().second == ordinal)
      ++runs.back().second;
    else
      runs.emplace_back(ordinal, 1);
  }
  int64_t length = 0;
  for (auto const& run : domain->file_runs)
    length = add(length, run.count);
  if (ordinals.size() == static_cast<std::size_t>(length) &&
      (ordinals.empty() || (runs.size() == 1 && runs.front().first == 0))) {
    state = row_position_state::identity;
    runs.clear();
  } else if (runs.size() * 2 > ordinals.size() && !ordinals.empty()) {
    if (ordinals.size() > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()))
      throw std::invalid_argument("lineage selection exceeds column limit");
    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                            ordinals.size(),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<int64_t>(),
                                  ordinals.data(),
                                  ordinals.size_bytes(),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    // Host selection storage may be released by the caller immediately.
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    per_row = std::move(column);
    runs.clear();
    state = row_position_state::per_row;
  }
  return {state, std::move(domain), std::move(runs), std::move(per_row)};
}
}  // namespace

int64_t row_lineage::input_domain_length() const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  int64_t count                                                  = 0;
  if (domain)
    for (auto const& run : domain->file_runs)
      count = add(count, run.count);
  return count;
}

int64_t row_lineage::output_length() const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  switch (state) {
    case row_position_state::identity: return input_domain_length();
    case row_position_state::per_row: return per_row ? per_row->size() : 0;
    case row_position_state::runs: {
      int64_t count = 0;
      for (auto const& run : runs)
        count = add(count, run.second);
      return count;
    }
    default: throw std::invalid_argument("lineage has no recoverable selection");
  }
}

void row_lineage::validate(int64_t rows) const { validate_impl(rows, false); }

void row_lineage::validate_impl(int64_t rows, bool strict) const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  if (rows < 0) throw std::invalid_argument("negative lineage cardinality");
  if (!strict && is_validated()) {
    if (recoverable(state) && output_length() != rows)
      throw std::invalid_argument("lineage cardinality mismatch");
    return;
  }
  if (domain) {
    for (auto const& file : domain->files)
      if (file.identity.empty() || file.rows < 0)
        throw std::invalid_argument("invalid lineage file identity");
    std::unordered_set<std::string> names;
    for (auto const& file : domain->files)
      if (!names.insert(file.identity).second)
        throw std::invalid_argument("duplicate lineage file identity");
    for (auto const& run : domain->file_runs)
      if (run.file >= domain->files.size() || run.first < 0 || run.count < 0 ||
          run.first > domain->files[run.file].rows ||
          run.count > domain->files[run.file].rows - run.first)
        throw std::invalid_argument("lineage input interval outside file");
  }
  auto length = input_domain_length();
  if (state == row_position_state::none || state == row_position_state::lost) {
    if (per_row || !runs.empty() || (state == row_position_state::none && domain))
      throw std::invalid_argument("unavailable lineage carries a selection");
    return;
  }
  if (!recoverable(state) || !domain)
    throw std::invalid_argument("lineage selection has no input domain");
  if ((state != row_position_state::per_row && per_row) ||
      (state != row_position_state::runs && !runs.empty()))
    throw std::invalid_argument("lineage carries conflicting selections");
  if (state == row_position_state::per_row &&
      (!per_row || per_row->type().id() != cudf::type_id::INT64 || per_row->has_nulls()))
    throw std::invalid_argument("lineage ordinals require non-null INT64");
  if (output_length() != rows) throw std::invalid_argument("lineage cardinality mismatch");
  for (std::size_t i = 0; i < runs.size(); ++i) {
    auto const [first, count] = runs[i];
    if (first < 0 || count <= 0 || first > length || count > length - first)
      throw std::invalid_argument("lineage selection outside input domain");
    for (std::size_t j = 0; j < i; ++j)
      if (first < runs[j].first + runs[j].second && runs[j].first < first + count)
        throw std::invalid_argument("duplicate lineage selection");
  }
}

std::vector<int64_t> row_lineage::ordinals_on(::cuda::stream_ref stream) const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  auto const count                                               = output_length();
  validate(count);
  std::vector<int64_t> values(count);
  if (state == row_position_state::identity) {
    std::iota(values.begin(), values.end(), int64_t{0});
  } else if (state == row_position_state::runs) {
    auto out = values.begin();
    for (auto const& [first, length] : runs)
      for (int64_t i = 0; i < length; ++i)
        *out++ = first + i;
  } else if (!values.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(values.data(),
                                  per_row->view().data<int64_t>(),
                                  values.size() * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  return values;
}

std::pair<std::size_t, int64_t> row_lineage::resolve(int64_t ordinal) const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  if (!domain || ordinal < 0 || ordinal >= input_domain_length())
    throw std::invalid_argument("lineage ordinal outside input domain");
  for (auto const& run : domain->file_runs) {
    if (ordinal < run.count) return {run.file, run.first + ordinal};
    ordinal -= run.count;
  }
  throw std::invalid_argument("lineage domain is incomplete");
}

row_lineage row_lineage::select(std::span<int64_t const> selected,
                                ::cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr) const
{
  auto const& [state, domain, runs, per_row, validated, binding] = data();
  if (!recoverable(state)) return *this;
  validate_on(output_length(), stream, mr);
  if (state == row_position_state::per_row) return select_host_lineage(*this, selected, stream, mr);
  std::vector<int64_t> ordinals;
  std::unordered_set<int64_t> seen;
  ordinals.reserve(selected.size());
  for (auto index : selected) {
    if (index < 0 || index >= output_length() || !seen.insert(index).second)
      throw std::invalid_argument("invalid or duplicate delivered selection");
    if (state == row_position_state::identity) {
      ordinals.push_back(index);
    } else {
      for (auto const& [first, count] : runs) {
        if (index < count) {
          ordinals.push_back(first + index);
          break;
        }
        index -= count;
      }
    }
  }
  auto result = compact(domain, ordinals, stream, mr);
  if (is_validated()) result.seal_validation(source_binding());
  return result;
}

row_lineage row_lineage::compose(row_lineage const& before,
                                 row_lineage const& step,
                                 ::cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  if (!recoverable(before.state())) return before;
  if (!recoverable(step.state())) { return {row_position_state::lost, before.domain()}; }
  before.validate_on(before.output_length(), stream, mr);
  step.validate_on(step.output_length(), stream, mr);
  if (step.input_domain_length() != before.output_length())
    throw std::invalid_argument("selection composition input length mismatch");
  // The overwhelmingly common identity composition stays interval-only.
  if (before.state() == row_position_state::identity &&
      before.domain()->files.size() == step.domain()->files.size() &&
      before.domain()->file_runs.size() == step.domain()->file_runs.size()) {
    bool same = true;
    for (std::size_t i = 0; i < before.domain()->files.size(); ++i)
      same &= before.domain()->files[i].identity == step.domain()->files[i].identity &&
              before.domain()->files[i].rows == step.domain()->files[i].rows;
    for (std::size_t i = 0; i < before.domain()->file_runs.size(); ++i) {
      auto const& a = before.domain()->file_runs[i];
      auto const& b = step.domain()->file_runs[i];
      same &= a.file == b.file && a.first == b.first && a.count == b.count;
    }
    if (same) {
      row_lineage result{step.state(), before.domain(), step.runs(), step.per_row()};
      if (before.is_validated() && step.is_validated())
        result.seal_validation(before.source_binding());
      return result;
    }
  }
  validate_lineage_composition(before, step, stream, mr);
  if (step.state() == row_position_state::identity) return before;
  auto selected = step.ordinal_column(stream, mr);
  return before.select(selected->view(), stream, mr);
}

row_lineage row_lineage::concatenate(std::span<row_lineage const> parts,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto domain = std::make_shared<lineage_domain>();
  std::vector<std::pair<int64_t, int64_t>> compact_runs;
  bool has_device  = std::any_of(parts.begin(), parts.end(), [](auto const& part) {
    return part.state() == row_position_state::per_row;
  });
  int64_t offset   = 0;
  bool unavailable = false, all_none = true;
  for (auto const& part : parts) {
    part.validate(recoverable(part.state()) ? part.output_length() : 0);
    all_none &= part.state() == row_position_state::none;
    unavailable |= !recoverable(part.state());
    if (recoverable(part.state())) {
      part.validate_on(part.output_length(), stream, mr);
      if (has_device) {
        // Device selections are concatenated without staging per-row values on the host.
      } else if (part.state() == row_position_state::identity) {
        if (part.output_length()) compact_runs.emplace_back(offset, part.output_length());
      } else {
        for (auto const& [first, count] : part.runs())
          compact_runs.emplace_back(add(offset, first), count);
      }
    }
    if (part.domain()) {
      std::vector<std::size_t> remap;
      for (auto const& file : part.domain()->files) {
        auto found = std::find_if(domain->files.begin(), domain->files.end(), [&](auto const& f) {
          return f.identity == file.identity;
        });
        if (found != domain->files.end() && found->rows != file.rows)
          throw std::invalid_argument("concatenated lineage file binding mismatch");
        remap.push_back(found - domain->files.begin());
        if (found == domain->files.end()) domain->files.push_back(file);
      }
      for (auto run : part.domain()->file_runs) {
        run.file = remap[run.file];
        domain->file_runs.push_back(run);
      }
    }
    offset = add(offset, part.input_domain_length());
  }
  if (unavailable)
    return {all_none ? row_position_state::none : row_position_state::lost,
            all_none ? nullptr : std::move(domain)};
  row_lineage result;
  if (has_device) {
    result = {row_position_state::per_row,
              std::move(domain),
              {},
              concatenate_lineage_ordinals(parts, stream, mr)};
  } else {
    std::vector<std::pair<int64_t, int64_t>> runs;
    for (auto const& run : compact_runs) {
      if (!runs.empty() && runs.back().first + runs.back().second == run.first)
        runs.back().second += run.second;
      else
        runs.push_back(run);
    }
    auto const identity =
      (runs.empty() && offset == 0) ||
      (runs.size() == 1 && runs.front().first == 0 && runs.front().second == offset);
    if (identity) runs.clear();
    result = {identity ? row_position_state::identity : row_position_state::runs,
              std::move(domain),
              std::move(runs)};
  }
  if (std::all_of(parts.begin(), parts.end(), [](auto const& part) { return part.is_validated(); }))
    result.seal_validation();
  return result;
}
}  // namespace sirius::op::scan
