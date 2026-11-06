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

#include "data/data_batch_utils.hpp"
#include "op/order/bounded_sort.hpp"
#include "op/order/gpu_sampled_merge.hpp"
#include "op/sirius_physical_merge_sort.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/merge.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sirius::op {
namespace sm = sampled_merge;
namespace {
using batches                  = sirius_physical_sort_stage::batches;
constexpr uint64_t native_rows = std::numeric_limits<cudf::size_type>::max();

struct contribution_size {
  uint64_t rows;
  uint64_t bytes;
};

std::vector<contribution_size> sizes(batches const& values)
{
  std::vector<contribution_size> result;
  for (auto const& value : values) {
    auto info = get_batch_rows_and_bytes(*value);
    if (!info.rows) { throw internal_exception("Sort batch lacks row metadata"); }
    result.push_back({*info.rows, info.bytes});
  }
  return result;
}

}  // namespace

sirius_physical_sort_stage::task_data::task_data(batches values,
                                                 std::size_t partition,
                                                 std::optional<std::size_t> position)
  : pipelineable_operator_data(std::move(values), position), partition(partition)
{
}
std::unique_ptr<pipelineable_operator_data>
sirius_physical_sort_stage::task_data::with_batches_preserving_rows(batches values) const
{
  auto result = std::make_unique<task_data>(std::move(values), partition, get_position());
  result->adopt_row_identity(*this);
  return result;
}
sirius_physical_sort_extract::input_data::input_data(batches values,
                                                     std::size_t run,
                                                     std::size_t first_tile)
  : pipelineable_operator_data(std::move(values)), run(run), first_tile(first_tile)
{
}
std::unique_ptr<pipelineable_operator_data>
sirius_physical_sort_extract::input_data::with_batches_preserving_rows(batches values) const
{
  auto result = std::make_unique<input_data>(std::move(values), run, first_tile);
  result->adopt_row_identity(*this);
  return result;
}
sirius_physical_sort_extract::output_data::output_data(batches values,
                                                       std::size_t run,
                                                       std::vector<std::size_t> tiles)
  : pipelineable_operator_data(std::move(values)), run(run), tiles(std::move(tiles))
{
}
std::unique_ptr<pipelineable_operator_data>
sirius_physical_sort_extract::output_data::with_batches_preserving_rows(batches values) const
{
  auto result = std::make_unique<output_data>(std::move(values), run, tiles);
  result->adopt_row_identity(*this);
  return result;
}
sirius_physical_sort_stage::sirius_physical_sort_stage(
  SiriusPhysicalOperatorType type,
  duckdb::vector<sirius::logical_type> types,
  std::size_t cardinality,
  std::shared_ptr<const sort_comparison_config> comparison)
  : sirius_physical_operator(type, std::move(types), cardinality), comparison(std::move(comparison))
{
}
std::size_t sirius_physical_sort_stage::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  return merge_cold_peak_estimate(stats.bytes);
}
bool sirius_physical_sort_stage::all_ports_empty()
{
  std::lock_guard guard(lock);
  return !_outstanding && sirius_physical_operator::all_ports_empty();
}
std::optional<task_creation_hint> sirius_physical_sort_stage::get_next_task_hint()
{
  if (!is_source_pipeline_finished()) { return sirius_physical_operator::get_next_task_hint(); }
  std::lock_guard guard(lock);
  if (!sirius_physical_operator::all_ports_empty()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return std::nullopt;
}
std::unique_ptr<operator_data> sirius_physical_sort_normalize::get_next_task_input_data()
{
  if (!is_source_pipeline_finished()) { return nullptr; }
  std::lock_guard guard(lock);
  auto& repo = *get_port("default")->repo;
  if (repo.all_empty()) { return nullptr; }
  if (repo.get_ordering() != cucascade::data_repository_ordering::UNORDERED) {
    throw internal_exception("Sort normalization requires unordered range contributions");
  }
  while (_partition < repo.num_partitions()) {
    if (repo.empty(_partition)) {
      ++_partition;
      continue;
    }
    batches values;
    values.reserve(repo.size(_partition));
    while (auto batch = repo.pop_next_data_batch(_partition)) {
      values.push_back(std::move(batch));
    }
    auto result = std::make_unique<task_data>(std::move(values), _partition);
    ++_partition;
    ++_outstanding;
    return result;
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_sort_cuts::get_next_task_input_data()
{
  if (!is_source_pipeline_finished()) { return nullptr; }
  std::lock_guard guard(lock);
  auto& repo = *get_port("default")->repo;
  if (repo.all_empty()) { return nullptr; }
  if (repo.get_ordering() != cucascade::data_repository_ordering::ORDERED) {
    throw internal_exception("Sort stage requires positioned input");
  }
  while (_partition < repo.num_partitions()) {
    if (repo.empty(_partition)) {
      ++_partition;
      continue;
    }
    std::vector<std::pair<std::size_t, std::shared_ptr<cucascade::data_batch>>> positioned;
    positioned.reserve(repo.size(_partition));
    batches values;
    values.reserve(repo.size(_partition));
    while (true) {
      auto [batch, position] = repo.pop_next_data_batch_with_position(_partition);
      if (!batch) { break; }
      positioned.emplace_back(position.value(), std::move(batch));
    }
    std::sort(positioned.begin(), positioned.end(), [](auto const& a, auto const& b) {
      return a.first < b.first;
    });
    for (auto& [position, batch] : positioned) {
      values.push_back(std::move(batch));
    }
    auto result = std::make_unique<task_data>(std::move(values), _partition);
    ++_partition;
    ++_outstanding;
    return result;
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::get_next_task_input_data()
{
  if (!is_source_pipeline_finished()) { return nullptr; }
  std::lock_guard guard(lock);
  auto& repo = *get_port("default")->repo;
  if (repo.all_empty()) { return nullptr; }
  if (repo.get_ordering() != cucascade::data_repository_ordering::ORDERED) {
    throw internal_exception("Sort stage requires positioned input");
  }
  while (_partition < repo.num_partitions()) {
    if (repo.empty(_partition)) {
      ++_partition;
      continue;
    }
    std::vector<std::pair<std::size_t, std::shared_ptr<cucascade::data_batch>>> positioned;
    positioned.reserve(repo.size(_partition));
    batches values;
    values.reserve(repo.size(_partition));
    while (true) {
      auto [batch, position] = repo.pop_next_data_batch_with_position(_partition);
      if (!batch) { break; }
      positioned.emplace_back(position.value(), std::move(batch));
    }
    std::sort(positioned.begin(), positioned.end(), [](auto const& a, auto const& b) {
      return a.first < b.first;
    });
    for (auto& [position, batch] : positioned) {
      values.push_back(std::move(batch));
    }
    auto result = std::make_unique<task_data>(std::move(values), _partition);
    ++_partition;
    ++_outstanding;
    return result;
  }
  return nullptr;
}

void sirius_physical_sort_extract::sink(operator_data const& output, ::cuda::stream_ref)
{
  auto const& data   = dynamic_cast<output_data const&>(output);
  auto const& values = data.get_data_batches();
  if (values.size() != data.tiles.size()) {
    throw internal_exception("Extraction tile count mismatch");
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    for (auto const& next : get_next_ports_after_sink()) {
      next.next_operator->push_data_batch_at_position(
        next.next_operator_port_name, values[i], data.run, data.tiles[i]);
    }
  }
}
void sirius_physical_sort_normalize::sink(operator_data const& output, ::cuda::stream_ref)
{
  auto const& data   = dynamic_cast<task_data const&>(output);
  auto const& values = data.get_data_batches();
  for (std::size_t position = 0; position < values.size(); ++position) {
    for (auto const& next : get_next_ports_after_sink()) {
      next.next_operator->push_data_batch_at_position(
        next.next_operator_port_name, values[position], position, data.partition);
    }
  }
}

void sirius_physical_sort_cuts::sink(operator_data const& output, ::cuda::stream_ref)
{
  auto const& data   = dynamic_cast<task_data const&>(output);
  auto const& values = data.get_data_batches();
  for (std::size_t position = 0; position < values.size(); ++position) {
    for (auto const& next : get_next_ports_after_sink()) {
      next.next_operator->push_data_batch_at_position(
        next.next_operator_port_name, values[position], position, data.partition);
    }
  }
}

void sirius_physical_sort_stage::on_task_output_published(operator_data const&)
{
  std::lock_guard guard(lock);
  if (!_outstanding) { throw internal_exception("Sort stage completed without outstanding work"); }
  --_outstanding;
}

sirius_physical_sort_normalize::sirius_physical_sort_normalize(sirius_physical_order* order)
  : sirius_physical_sort_stage(SiriusPhysicalOperatorType::SORT_NORMALIZE,
                               order->types,
                               order->estimated_cardinality,
                               order->comparison)
{
  chunk_bytes = order->chunk_bytes;
}
sirius_physical_sort_cuts::sirius_physical_sort_cuts(sirius_physical_order* order)
  : sirius_physical_sort_stage(SiriusPhysicalOperatorType::SORT_CUTS,
                               order->types,
                               order->estimated_cardinality,
                               order->comparison)
{
  chunk_bytes = order->chunk_bytes;
}
sirius_physical_sort_extract::sirius_physical_sort_extract(sirius_physical_order* order)
  : sirius_physical_sort_stage(SiriusPhysicalOperatorType::SORT_EXTRACT,
                               order->types,
                               order->estimated_cardinality,
                               order->comparison)
{
  chunk_bytes = order->chunk_bytes;
}

std::unique_ptr<operator_data> sirius_physical_sort_normalize::execute(
  operator_data const& input_data, ::cuda::stream_ref stream)
{
  if (!chunk_bytes) { throw invalid_input_exception("Sort chunk target must be positive"); }
  auto const& input = dynamic_cast<task_data const&>(input_data);
  auto reads        = input.get_read_only_batches();
  auto* space       = reads.at(0).get_memory_space();
  auto mr           = space->get_default_allocator();

  // Estimate uniform row sizes and copy each resulting slice once. Variable-width
  // skew can leave a slice above the byte target; we do not recursively split it.
  batches normalized;
  auto const& values = input.get_data_batches();
  auto input_sizes   = sizes(values);
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto size = input_sizes[i];
    if (!size.rows) { continue; }
    if (size.bytes <= chunk_bytes || size.rows == 1) {
      normalized.push_back(values[i]);
      continue;
    }
    auto average        = 1 + (size.bytes - 1) / size.rows;
    auto rows_per_chunk = std::max(uint64_t{1}, chunk_bytes / average);
    auto view           = get_cudf_table_view(reads[i]);
    for (uint64_t begin = 0; begin < size.rows;) {
      auto end = begin + std::min(rows_per_chunk, size.rows - begin);
      auto slice =
        cudf::slice(
          view, {static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(end)}, stream)
          .front();
      normalized.push_back(make_data_batch(
        std::make_unique<cudf::table>(slice, stream, mr), *space, stream, batch_telemetry()));
      begin = end;
    }
  }

  // One fixed coalescing pass. Freeze run ordinals only after this pass.
  auto run_sizes = sizes(normalized);
  batches runs;
  for (std::size_t begin = 0; begin < normalized.size();) {
    auto end      = begin;
    uint64_t rows = 0, bytes = 0;
    while (end < normalized.size()) {
      auto size = run_sizes[end];
      if (!size.rows || size.rows > native_rows) {
        throw std::invalid_argument("Coalescing requires normalized nonempty contributions");
      }
      if (end != begin && (size.bytes >= chunk_bytes || size.bytes > chunk_bytes - bytes ||
                           size.rows > native_rows - rows)) {
        break;
      }
      ++end;
      rows += size.rows;
      bytes += size.bytes;
      // A batch at or above the target is always a singleton.
      if (size.bytes >= chunk_bytes) { break; }
    }
    if (end == begin + 1) {
      runs.push_back(normalized[begin]);
    } else {
      batches sources(normalized.begin() + begin, normalized.begin() + end);
      std::vector<cucascade::read_only_data_batch> source_reads;
      std::vector<cudf::table_view> views;
      source_reads.reserve(sources.size());
      views.reserve(sources.size());
      for (auto const& source : sources) {
        source_reads.push_back(source->to_read_only());
        views.push_back(get_cudf_table_view(source_reads.back()));
      }
      runs.push_back(make_data_batch(cudf::merge(views,
                                                 comparison->key_columns,
                                                 comparison->directions,
                                                 comparison->null_precedence,
                                                 stream,
                                                 mr),
                                     *space,
                                     stream,
                                     batch_telemetry()));
    }
    for (auto i = begin; i < end; ++i) {
      normalized[i].reset();
    }
    begin = end;
  }
  return std::make_unique<task_data>(std::move(runs), input.partition);
}

std::unique_ptr<operator_data> sirius_physical_sort_cuts::execute(operator_data const& input_data,
                                                                  ::cuda::stream_ref stream)
{
  if (!chunk_bytes) { throw invalid_input_exception("Sort chunk target must be positive"); }
  auto const& input = dynamic_cast<task_data const&>(input_data);
  auto reads        = input.get_read_only_batches();
  auto* space       = reads.at(0).get_memory_space();
  auto mr           = space->get_default_allocator();
  auto const& runs  = input.get_data_batches();
  auto run_sizes    = sizes(runs);
  uint64_t rows = 0, bytes = 0;
  for (auto size : run_sizes) {
    if (!size.rows) { throw internal_exception("Cut planning requires nonempty runs"); }
    rows += size.rows;
    bytes += size.bytes;
  }
  auto average   = bytes ? 1 + (bytes - 1) / rows : uint64_t{1};
  auto row_limit = std::min(native_rows, std::max(uint64_t{1}, chunk_bytes / average));

  std::shared_ptr<cucascade::data_batch> splitters;
  if (rows > row_limit) {
    // Bound each tile by stride * sample_step plus the unsampled rows
    // from each frozen run, reserving at most a quarter of the row limit
    // for that uncertainty.
    auto stride          = row_limit / runs.size() / 4 + 1;
    auto uncertainty     = runs.size() * (stride - 1);
    auto sample_step     = (row_limit - uncertainty) / stride;
    uint64_t sample_rows = 0;
    for (auto size : run_sizes) {
      sample_rows += size.rows / stride;
    }
    if (sample_rows > native_rows) {
      throw invalid_input_exception("Sort samples exceed native row limit");
    }
    batches samples;
    std::vector<cucascade::read_only_data_batch> sample_reads;
    for (std::size_t i = 0; i < runs.size(); ++i) {
      if (run_sizes[i].rows < stride) { continue; }
      auto view = get_cudf_table_view(reads[i]);
      samples.push_back(
        make_data_batch(sm::gather_samples(view, i, stride, *comparison, stream, mr),
                        *space,
                        stream,
                        batch_telemetry()));
    }
    std::vector<cudf::table_view> sample_views;
    sample_reads.reserve(samples.size());
    sample_views.reserve(samples.size());
    for (auto const& sample : samples) {
      sample_reads.push_back(sample->to_read_only());
      sample_views.push_back(get_cudf_table_view(sample_reads.back()));
    }
    splitters = make_data_batch(
      sm::order_sample_splitters(sample_views, sample_step, *comparison, stream, mr),
      *space,
      stream,
      batch_telemetry());
  }

  // Persist the actual cuts, paired with their exact runs. The source runs are
  // forwarded, so planning doesn't introduce another payload copy.
  std::optional<cucascade::read_only_data_batch> splitter_read;
  if (splitters) { splitter_read.emplace(splitters->to_read_only()); }
  batches output;
  for (std::size_t i = 0; i < runs.size(); ++i) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(splitter_read
                        ? sm::discover_cuts(get_cudf_table_view(reads[i]),
                                            i,
                                            get_cudf_table_view(*splitter_read),
                                            *comparison,
                                            stream,
                                            mr)
                        : cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
    auto cuts = make_data_batch(
      std::make_unique<cudf::table>(std::move(columns)), *space, stream, batch_telemetry());
    output.push_back(runs[i]);
    output.push_back(std::move(cuts));
  }
  return std::make_unique<task_data>(std::move(output), input.partition);
}

std::unique_ptr<operator_data> sirius_physical_sort_extract::get_next_task_input_data()
{
  if (!is_source_pipeline_finished()) { return nullptr; }
  std::lock_guard guard(lock);
  auto& repo = *get_port("default")->repo;
  if (repo.all_empty()) { return nullptr; }
  if (repo.get_ordering() != cucascade::data_repository_ordering::ORDERED) {
    throw internal_exception("Cut handoff requires positioned run/cut pairs");
  }
  while (_partition < repo.num_partitions()) {
    if (!_run_count) {
      auto positions = repo.get_positions(_partition);
      if (positions.empty()) {
        ++_partition;
        continue;
      }
      if (positions.size() % 2) { throw internal_exception("Unpaired sort run/cuts"); }
      std::size_t tile_count = 0;
      // Validate the whole partition before popping any of its pairs.
      for (std::size_t i = 0; i < positions.size(); i += 2) {
        if (positions[i] != i || positions[i + 1] != i + 1) {
          throw internal_exception("Noncontiguous sort run ordinals");
        }
        auto cuts  = repo.get_data_batch_by_position(i + 1, _partition);
        auto count = sizes(batches{cuts}).front().rows + 1;
        if (i && count != tile_count) { throw internal_exception("Inconsistent sort tile counts"); }
        tile_count = count;
      }
      _tile_count = tile_count;
      _run_count  = positions.size() / 2;
    }
    batches values;
    values.push_back(repo.pop_data_batch_by_position(2 * _run, _partition));
    values.push_back(repo.pop_data_batch_by_position(2 * _run + 1, _partition));
    auto result = std::make_unique<input_data>(std::move(values), _run, _first_tile);
    if (++_run == _run_count) {
      _first_tile += _tile_count;
      ++_partition;
      _run = _run_count = 0;
    }
    ++_outstanding;
    return result;
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_sort_extract::execute(
  operator_data const& task_input, ::cuda::stream_ref stream)
{
  auto const& input = dynamic_cast<input_data const&>(task_input);
  auto reads        = input.get_read_only_batches();
  if (reads.size() != 2) { throw internal_exception("Extraction requires one run/cuts pair"); }
  auto* space    = reads[0].get_memory_space();
  auto mr        = space->get_default_allocator();
  auto run       = get_cudf_table_view(reads[0]);
  auto cut_table = get_cudf_table_view(reads[1]);
  if (cut_table.num_columns() != 1 || cut_table.column(0).type().id() != cudf::type_id::INT32 ||
      cut_table.column(0).has_nulls()) {
    throw internal_exception("Invalid sort cut table");
  }
  std::vector<cudf::size_type> cuts(static_cast<std::size_t>(cut_table.num_rows()) + 2);
  batches output;
  std::vector<std::size_t> tiles;
  try {
    if (cut_table.num_rows()) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(cuts.data() + 1,
                                    cut_table.column(0).data<cudf::size_type>(),
                                    cut_table.num_rows() * sizeof(cudf::size_type),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
    }
    stream.sync();
    cuts.front() = 0;
    cuts.back()  = run.num_rows();
    if (!std::is_sorted(cuts.begin(), cuts.end())) {
      throw internal_exception("Sort cuts do not cover the run monotonically");
    }
    for (std::size_t t = 0; t + 1 < cuts.size(); ++t) {
      if (cuts[t] == cuts[t + 1]) { continue; }
      auto view = cudf::slice(run, {cuts[t], cuts[t + 1]}, stream).front();
      output.push_back(make_data_batch(
        std::make_unique<cudf::table>(view, stream, mr), *space, stream, batch_telemetry()));
      tiles.push_back(input.first_tile + t);
    }
    stream.sync();
  } catch (...) {
    cudaStreamSynchronize(stream.get());
    throw;
  }
  return std::make_unique<output_data>(std::move(output), input.run, std::move(tiles));
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute(operator_data const& input_data,
                                                                   ::cuda::stream_ref stream)
{
  auto const& input = dynamic_cast<task_data const&>(input_data);
  auto reads        = input.get_read_only_batches();
  auto* space       = reads.at(0).get_memory_space();
  auto mr           = space->get_default_allocator();
  uint64_t rows     = 0;
  for (auto size : sizes(input.get_data_batches())) {
    rows += size.rows;
  }
  if (rows > native_rows) { throw invalid_input_exception("Sort tile exceeds native row limit"); }
  std::unique_ptr<cudf::table> table;
  if (reads.size() == 1) {
    auto view = get_cudf_table_view(reads.front());
    if (!_final_projections.empty()) {
      view = view.select(_final_projections.begin(), _final_projections.end());
    }
    table = std::make_unique<cudf::table>(view, stream, mr);
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(reads.size());
    for (auto const& read : reads) {
      views.push_back(get_cudf_table_view(read));
    }
    auto columns = cudf::merge(views,
                               comparison->key_columns,
                               comparison->directions,
                               comparison->null_precedence,
                               stream,
                               mr)
                     ->release();
    if (!_final_projections.empty()) {
      std::vector<std::unique_ptr<cudf::column>> projected;
      projected.reserve(_final_projections.size());
      for (auto index : _final_projections) {
        projected.push_back(std::move(columns[index]));
      }
      columns = std::move(projected);
    }
    table = std::make_unique<cudf::table>(std::move(columns));
  }
  auto result = make_data_batch(std::move(table), *space, stream, batch_telemetry());
  return std::make_unique<task_data>(batches{std::move(result)}, input.partition, input.partition);
}
void sirius_physical_merge_sort::sink(operator_data const& output, ::cuda::stream_ref stream)
{
  sirius_physical_operator::sink(output, stream);
}
}  // namespace sirius::op
