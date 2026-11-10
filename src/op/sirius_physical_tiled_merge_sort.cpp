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

#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "op/order/bounded_sort.hpp"
#include "op/order/gpu_sampled_merge.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "pipeline/sirius_pipeline.hpp"

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
// Each range owns an equal share of this logical position space; refinement divides a
// node's share among its children. Consumers accept sparse positions.
constexpr std::size_t merge_position_space = std::size_t{1} << 62;

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

sirius_physical_merge_sort::work::work(batches values,
                                       step kind,
                                       std::size_t node,
                                       std::vector<uint64_t> ordinals,
                                       uint64_t parameter,
                                       std::optional<std::size_t> position)
  : pipelineable_operator_data(std::move(values), position),
    kind(kind),
    node(node),
    ordinals(std::move(ordinals)),
    parameter(parameter)
{
}
std::unique_ptr<pipelineable_operator_data>
sirius_physical_merge_sort::work::with_batches_preserving_rows(batches values) const
{
  auto copy =
    std::make_unique<work>(std::move(values), kind, node, ordinals, parameter, get_position());
  copy->adopt_row_identity(*this);
  return copy;
}
sirius_physical_merge_sort::result::result(batches values,
                                           step kind,
                                           std::size_t node,
                                           std::vector<uint64_t> ordinals,
                                           std::vector<piece> pieces,
                                           std::optional<std::size_t> position)
  : pipelineable_operator_data(std::move(values), position),
    kind(kind),
    node(node),
    ordinals(std::move(ordinals)),
    pieces(std::move(pieces))
{
}
std::unique_ptr<pipelineable_operator_data>
sirius_physical_merge_sort::result::with_batches_preserving_rows(batches values) const
{
  auto copy =
    std::make_unique<result>(std::move(values), kind, node, ordinals, pieces, get_position());
  copy->adopt_row_identity(*this);
  return copy;
}

void sirius_physical_merge_sort::register_refinement(
  cucascade::shared_data_repository_manager& manager)
{
  auto id = get_operator_id();
  for (auto [name, slot] : {std::pair{"sort_parts", &_parts},
                            std::pair{"sort_samples", &_samples},
                            std::pair{"sort_splitters", &_splitters}}) {
    manager.add_new_repository(id, name, std::make_unique<cucascade::shared_data_repository>());
    *slot = manager.get_repository(id, name).get();
  }
}

// Called once, under the lock, after every range contribution has arrived. Reads row and
// byte metadata only: nothing is restored to a device.
void sirius_physical_merge_sort::ingest()
{
  if (!_parts || !_samples || !_splitters) {
    throw internal_exception("Merge sort refinement repositories are not registered");
  }
  _ingested  = true;
  auto& repo = *get_port("default")->repo;
  if (repo.all_empty()) { return; }
  if (repo.get_ordering() == cucascade::data_repository_ordering::ORDERED) {
    throw internal_exception("Merge sort requires unordered range contributions");
  }
  auto ranges = repo.num_partitions();
  auto width  = merge_position_space / ranges;
  for (std::size_t range = 0; range < ranges; ++range) {
    node n{.lo = range * width, .hi = (range + 1) * width};
    uint64_t ordinal = 0;
    while (auto batch = repo.pop_next_data_batch(range)) {
      auto info = get_batch_rows_and_bytes(*batch);
      if (!info.rows) { throw internal_exception("Sort batch lacks row metadata"); }
      if (!*info.rows) { continue; }
      _parts->add_data_batch_at_position(batch, ordinal, _nodes.size());
      n.parts.push_back({ordinal++, *info.rows, info.bytes});
      n.rows += *info.rows;
      n.bytes += info.bytes;
    }
    if (n.parts.empty()) { continue; }
    n.state = phase::ready;
    _active.emplace(n.lo, _nodes.size());
    _nodes.push_back(std::move(n));
  }
}

bool sirius_physical_merge_sort::fits(node const& n) const
{
  return n.rows <= native_rows && (n.bytes <= chunk_bytes || n.rows == 1);
}

std::optional<std::size_t> sirius_physical_merge_sort::actionable() const
{
  for (auto [lo, id] : _active) {
    auto const& n = _nodes[id];
    switch (n.state) {
      case phase::ready:
      case phase::plan_ready: return id;
      case phase::sampling:
      case phase::cutting:
        if (n.cursor < n.parts.size()) { return id; }
        break;
      default: break;
    }
  }
  return std::nullopt;
}

void sirius_physical_merge_sort::retire(std::size_t id)
{
  auto& n = _nodes[id];
  _active.erase({n.lo, id});
  n.state = phase::done;
  n.parts.clear();
  n.parts.shrink_to_fit();
}

// Consecutive contributions up to chunk_bytes (at least one). Sampling borrows payload,
// which stays queued for cutting; cutting takes ownership and borrows the splitters.
std::unique_ptr<sirius_physical_merge_sort::work> sirius_physical_merge_sort::claim_group(
  std::size_t id, step kind, bool pop)
{
  auto& n = _nodes[id];
  batches values;
  std::vector<uint64_t> ordinals;
  uint64_t bytes = 0;
  while (n.cursor < n.parts.size()) {
    auto const& part = n.parts[n.cursor];
    if (!values.empty() && bytes + part.bytes > chunk_bytes) { break; }
    auto batch = pop ? _parts->pop_data_batch_by_position(part.ordinal, id)
                     : _parts->get_data_batch_by_position(part.ordinal, id);
    if (!batch) { throw internal_exception("Sort refinement contribution disappeared"); }
    values.push_back(std::move(batch));
    ordinals.push_back(part.ordinal);
    bytes += part.bytes;
    ++n.cursor;
  }
  uint64_t parameter = n.stride;
  if (kind == step::cut) {
    auto splitters = _splitters->get_data_batch_by_position(0, id);
    if (!splitters) { throw internal_exception("Sort refinement splitters disappeared"); }
    values.push_back(std::move(splitters));
    parameter = n.first_child;
  }
  ++n.outstanding;
  return std::make_unique<work>(std::move(values), kind, id, std::move(ordinals), parameter);
}

std::unique_ptr<sirius_physical_merge_sort::work> sirius_physical_merge_sort::claim(std::size_t id)
{
  auto& n = _nodes[id];
  switch (n.state) {
    case phase::ready: {
      if (fits(n)) {
        batches values;
        std::vector<uint64_t> ordinals;
        for (auto const& part : n.parts) {
          auto batch = _parts->pop_data_batch_by_position(part.ordinal, id);
          if (!batch) { throw internal_exception("Sort merge contribution disappeared"); }
          values.push_back(std::move(batch));
          ordinals.push_back(part.ordinal);
        }
        n.state = phase::merging;
        ++n.outstanding;
        return std::make_unique<work>(
          std::move(values), step::merge, id, std::move(ordinals), 0, n.lo);
      }
      // Oversized. With R rows per tile and k contributions, the sampled-merge bound
      // gives tiles of at most R rows. Since bytes > chunk_bytes, rows > R, so every
      // child has strictly fewer rows than this node.
      auto average  = 1 + (n.bytes - 1) / n.rows;
      auto limit    = std::min(native_rows, std::max(uint64_t{1}, chunk_bytes / average));
      uint64_t k    = n.parts.size();
      n.stride      = limit / k / 4 + 1;
      n.sample_step = (limit - k * (n.stride - 1)) / n.stride;
      if (!n.sample_step) { throw internal_exception("Sort refinement sample step is zero"); }
      uint64_t sample_rows = 0;
      for (auto const& part : n.parts) {
        sample_rows += part.rows / n.stride;
      }
      if (sample_rows > native_rows) {
        throw invalid_input_exception("Sort refinement samples exceed native row limit");
      }
      n.state  = phase::sampling;
      n.cursor = 0;
      return claim_group(id, step::sample, false);
    }
    case phase::sampling: return claim_group(id, step::sample, false);
    case phase::plan_ready: {
      batches values;
      if (id < _samples->num_partitions()) {
        for (auto position : _samples->get_positions(id)) {
          values.push_back(_samples->pop_data_batch_by_position(position, id));
        }
      }
      if (values.empty()) { throw internal_exception("Oversized sort node produced no samples"); }
      n.state = phase::planning;
      ++n.outstanding;
      return std::make_unique<work>(
        std::move(values), step::plan, id, std::vector<uint64_t>{}, n.sample_step);
    }
    case phase::cutting: return claim_group(id, step::cut, true);
    default: throw internal_exception("Sort refinement node is not claimable");
  }
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::get_next_task_input_data()
{
  if (!is_source_pipeline_finished()) { return nullptr; }
  std::lock_guard guard(lock);
  if (!_ingested) { ingest(); }
  auto id = actionable();
  if (!id) { return nullptr; }
  return claim(*id);
}

std::optional<task_creation_hint> sirius_physical_merge_sort::get_next_task_hint()
{
  if (!is_source_pipeline_finished()) { return sirius_physical_operator::get_next_task_hint(); }
  std::lock_guard guard(lock);
  bool ready = _ingested ? actionable().has_value() : !sirius_physical_operator::all_ports_empty();
  if (ready) { return task_creation_hint{TaskCreationHint::READY, this}; }
  return std::nullopt;
}

bool sirius_physical_merge_sort::all_ports_empty()
{
  std::lock_guard guard(lock);
  return _active.empty() && sirius_physical_operator::all_ports_empty();
}

// Publication is the only place refinement state advances, so retries never duplicate
// or skip a transition. New work is not discovered by downstream scheduling; request it.
void sirius_physical_merge_sort::on_task_output_published(operator_data const& output)
{
  auto const& data = dynamic_cast<result const&>(output);
  bool more        = false;
  {
    std::lock_guard guard(lock);
    auto id = data.node;
    if (!_nodes.at(id).outstanding) {
      throw internal_exception("Sort refinement completed without outstanding work");
    }
    --_nodes[id].outstanding;
    switch (data.kind) {
      case step::merge: retire(id); break;
      case step::sample: {
        auto& n = _nodes[id];
        if (n.cursor == n.parts.size() && !n.outstanding) { n.state = phase::plan_ready; }
        break;
      }
      case step::plan: {
        auto tiles = data.ordinals.at(0) + 1;
        auto lo = _nodes[id].lo, hi = _nodes[id].hi;
        auto width = (hi - lo) / tiles;
        if (!width) { throw invalid_input_exception("Sort refinement exhausted output positions"); }
        auto first = _nodes.size();
        for (std::size_t t = 0; t < tiles; ++t) {
          _nodes.push_back(
            node{.lo = lo + t * width, .hi = t + 1 == tiles ? hi : lo + (t + 1) * width});
        }
        auto& n       = _nodes[id];
        n.first_child = first;
        n.child_count = tiles;
        n.state       = phase::cutting;
        n.cursor      = 0;
        break;
      }
      case step::cut: {
        for (auto const& p : data.pieces) {
          auto& child = _nodes.at(p.child);
          child.parts.push_back({p.ordinal, p.rows, p.bytes});
          child.rows += p.rows;
          child.bytes += p.bytes;
        }
        auto& n = _nodes[id];
        if (n.cursor < n.parts.size() || n.outstanding) { break; }
        // Every producer of the children has published: they are complete.
        _splitters->pop_data_batch_by_position(0, id);
        auto first = n.first_child, count = n.child_count;
        auto rows = n.rows;
        retire(id);
        for (auto c = first; c < first + count; ++c) {
          auto& child = _nodes[c];
          if (child.parts.empty()) {
            child.state = phase::done;
            continue;
          }
          if (child.rows >= rows) { throw internal_exception("Sort refinement made no progress"); }
          std::sort(child.parts.begin(), child.parts.end(), [](auto const& a, auto const& b) {
            return a.ordinal < b.ordinal;
          });
          child.state = phase::ready;
          _active.emplace(child.lo, c);
        }
        break;
      }
    }
    more = actionable().has_value();
  }
  if (more) {
    if (auto pipeline = get_pipeline()) {
      if (auto* creator = pipeline->get_task_creator()) { creator->schedule(this); }
    }
  }
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
  auto const& input = dynamic_cast<work const&>(input_data);
  switch (input.kind) {
    case step::merge: return execute_merge(input, stream);
    case step::sample: return execute_sample(input, stream);
    case step::plan: return execute_plan(input, stream);
    case step::cut: return execute_cut(input, stream);
  }
  throw internal_exception("Unknown sort refinement step");
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_sample(work const& input,
                                                                          ::cuda::stream_ref stream)
{
  auto reads  = input.get_read_only_batches();
  auto* space = reads.at(0).get_memory_space();
  auto mr     = space->get_default_allocator();
  batches samples;
  std::vector<uint64_t> ordinals;
  for (std::size_t i = 0; i < reads.size(); ++i) {
    auto view = get_cudf_table_view(reads[i]);
    if (static_cast<uint64_t>(view.num_rows()) < input.parameter) { continue; }
    samples.push_back(make_data_batch(
      sm::gather_samples(view, input.ordinals.at(i), input.parameter, *comparison, stream, mr),
      *space,
      stream,
      batch_telemetry()));
    ordinals.push_back(input.ordinals[i]);
  }
  return std::make_unique<result>(
    std::move(samples), step::sample, input.node, std::move(ordinals), std::vector<piece>{});
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_plan(work const& input,
                                                                        ::cuda::stream_ref stream)
{
  auto reads  = input.get_read_only_batches();
  auto* space = reads.at(0).get_memory_space();
  auto mr     = space->get_default_allocator();
  std::vector<cudf::table_view> views;
  views.reserve(reads.size());
  for (auto const& read : reads) {
    views.push_back(get_cudf_table_view(read));
  }
  auto splitters = sm::order_sample_splitters(views, input.parameter, *comparison, stream, mr);
  uint64_t rows  = splitters->num_rows();
  if (!rows) { throw internal_exception("Oversized sort node produced no splitters"); }
  return std::make_unique<result>(
    batches{make_data_batch(std::move(splitters), *space, stream, batch_telemetry())},
    step::plan,
    input.node,
    std::vector<uint64_t>{rows},
    std::vector<piece>{});
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_cut(work const& input,
                                                                       ::cuda::stream_ref stream)
{
  auto reads = input.get_read_only_batches();
  if (reads.size() != input.ordinals.size() + 1) {
    throw internal_exception("Sort cut task requires contributions and splitters");
  }
  auto* space    = reads[0].get_memory_space();
  auto mr        = space->get_default_allocator();
  auto splitters = get_cudf_table_view(reads.back());
  batches output;
  std::vector<piece> pieces;
  try {
    for (std::size_t i = 0; i < input.ordinals.size(); ++i) {
      auto run        = get_cudf_table_view(reads[i]);
      auto ordinal    = input.ordinals[i];
      auto cut_column = sm::discover_cuts(run, ordinal, splitters, *comparison, stream, mr);
      std::vector<cudf::size_type> cuts(static_cast<std::size_t>(splitters.num_rows()) + 2);
      CUDF_CUDA_TRY(cudaMemcpyAsync(cuts.data() + 1,
                                    cut_column->view().data<cudf::size_type>(),
                                    splitters.num_rows() * sizeof(cudf::size_type),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      stream.sync();
      cuts.front() = 0;
      cuts.back()  = run.num_rows();
      if (!std::is_sorted(cuts.begin(), cuts.end())) {
        throw internal_exception("Sort cuts do not cover the contribution monotonically");
      }
      for (std::size_t t = 0; t + 1 < cuts.size(); ++t) {
        if (cuts[t] == cuts[t + 1]) { continue; }
        auto view  = cudf::slice(run, {cuts[t], cuts[t + 1]}, stream).front();
        auto batch = make_data_batch(
          std::make_unique<cudf::table>(view, stream, mr), *space, stream, batch_telemetry());
        auto info = get_batch_rows_and_bytes(*batch);
        pieces.push_back({input.parameter + t, ordinal, info.rows.value(), info.bytes});
        output.push_back(std::move(batch));
      }
    }
    stream.sync();
  } catch (...) {
    cudaStreamSynchronize(stream.get());
    throw;
  }
  return std::make_unique<result>(
    std::move(output), step::cut, input.node, std::vector<uint64_t>{}, std::move(pieces));
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_merge(work const& input,
                                                                         ::cuda::stream_ref stream)
{
  auto reads    = input.get_read_only_batches();
  auto* space   = reads.at(0).get_memory_space();
  auto mr       = space->get_default_allocator();
  uint64_t rows = 0;
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
  auto merged = make_data_batch(std::move(table), *space, stream, batch_telemetry());
  return std::make_unique<result>(batches{std::move(merged)},
                                  step::merge,
                                  input.node,
                                  std::vector<uint64_t>{},
                                  std::vector<piece>{},
                                  input.get_position());
}

// Refinement outputs go only to this operator's own repositories; publication into them
// precedes on_task_output_published, which makes them claimable.
void sirius_physical_merge_sort::sink(operator_data const& output, ::cuda::stream_ref stream)
{
  auto const& data   = dynamic_cast<result const&>(output);
  auto const& values = data.get_data_batches();
  switch (data.kind) {
    case step::merge: sirius_physical_operator::sink(output, stream); return;
    case step::sample:
      for (std::size_t i = 0; i < values.size(); ++i) {
        _samples->add_data_batch_at_position(values[i], data.ordinals.at(i), data.node);
      }
      return;
    case step::plan: _splitters->add_data_batch_at_position(values.at(0), 0, data.node); return;
    case step::cut:
      for (std::size_t i = 0; i < values.size(); ++i) {
        auto const& p = data.pieces.at(i);
        _parts->add_data_batch_at_position(values[i], p.ordinal, p.child);
      }
      return;
  }
}
}  // namespace sirius::op
